use std::cmp::PartialEq;
use std::collections::BTreeMap;
use std::mem::MaybeUninit;
use ndarray::{Array3,s};

use metatensor::TensorBlock;
use metatensor::{Labels, LabelsBuilder};

use crate::systems::Pair;
use crate::{Error, System, SystemBase};
use crate::types::Vector3D;


/// Sort a pair and return true if the pair was inverted
#[inline]
fn sort_pair<T: PartialOrd>((i, j): (T, T)) -> ((T, T), bool) {
    if i <= j {
        ((i, j), false)
    } else {
        ((j, i), true)
    }
}


/// This object is a simple representation of a bond-atom triplet (to represent a single neighbor atom to a bond environment)
/// it only makes sense for a given system
#[derive(Debug,Clone,Copy,PartialEq)]
pub struct BATripletInfo{
    /// number of the first atom (the bond's first atom) within the system
    pub atom_i: usize,
    /// number of the second atom (the bond's second atom) within the system
    pub atom_j: usize,
    /// number of the third atom (the neighbor atom) within the system
    pub atom_k: usize,
    /// number that uniquely identifies the bond within the system, in case of periodic boundary condition shenanigans
    /// it is independent of the order of the atoms within the bond
    pub bond_i: usize,
    /// number that uniquely identifies the bond-atom triplet within the system, in case of periodic boundary condition shenanigans
    /// it is independent of the order of the atoms within the triplet's bond
    pub triplet_i: usize,
    /// wether or not the third atom is the same as one of the first two (and NOT a periodic image thereof)
    pub is_self_contrib: bool,
    /// optional: the vector between first and second atom
    pub bond_vector: Option<Vector3D>,
    /// optional: the bector between the bond center and the third atom
    pub third_vector: Option<Vector3D>,
}


/// Manages a list of 'neighbors', where one neighbor is the center of a pair of atoms
/// (first and second atom), and the other neighbor is a simple atom (third atom).
/// Both the length of the bond and the distance between neighbors are subjected to a spherical cutoff.
/// This pre-calculator can compute and cache this list within a given system
/// (with two distance vectors per entry: one within the bond and one between neighbors).
/// Then, it can re-enumerate those neighbors, either for a full system, or with restrictions on the atoms or their species.
///
/// This saves memory/computational power by only working with "half" neighbor list
/// This is done by only including one entry for each `i - j` bond, not both `i - j` and `j - i`.
/// The order of i and j is that the atom with the smallest Z (or species ID in general) comes first.
///
/// The two first atoms must not be the same atom, but the third atom may be one of them.
/// (When periodic boundaries arise, the two first atoms may be images of each other.)
#[derive(Debug,Clone)]
#[derive(serde::Deserialize, serde::Serialize, schemars::JsonSchema)]
pub struct BATripletNeighborList {
    // /// Should we compute a full neighbor list (each pair appears twice, once as
    // /// `i-j` and once as `j-i`), or a half neighbor list (each pair only
    // /// appears once, (such that `species_i <= species_j`))
    // pub use_half_enumeration: bool,
    /// Spherical cutoffs to use to determine if two atoms are neighbors
    pub cutoffs: [f64;2],  // bond_, third_cutoff
}

/// the internal function doing the triplet computing itself
fn list_raw_triplets(system: &mut dyn SystemBase, bond_cutoff: f64, third_cutoff: f64) -> Result<Vec<BATripletInfo>,Error> {
    system.compute_neighbors(bond_cutoff)?;
    let bonds = system.pairs()?.to_owned();
    
    // atoms_cutoff needs to be a bit bigger than the one in the current 
    // implementation to be sure we get the same set of neighbors.
    system.compute_neighbors(third_cutoff + bond_cutoff/2.)?;
    let species = system.species()?;

    let reorient_pair =  move |b: Pair| {
        if species[b.first] <= species[b.second] {
            b
        } else {
            Pair{
                first: b.second,
                second: b.first,
                distance: b.distance,
                vector: -b.vector,
                cell_shift_indices: b.cell_shift_indices,  // not corrected because irrelevant here
            }
        }
    };
    
    let mut ba_triplets = vec![];
    let mut triplet_i = 0;
    for (bond_i,bond) in bonds.into_iter().map(reorient_pair).enumerate() {
        let halfbond = 0.5 * bond.vector;

        // first, record the self contribution
        {
            let ((pairatom_i,pairatom_j),inverted) = sort_pair((bond.first,bond.second));
            let halfbond = if inverted {
                -halfbond
            } else {
                halfbond
            };

            let mut tri = BATripletInfo{
                atom_i: bond.first, atom_j: bond.second, atom_k: pairatom_i,
                bond_i, triplet_i,
                bond_vector: Some(bond.vector),
                third_vector: Some(-halfbond),
                is_self_contrib: true,
            };
            // if half_enumeration {
            ba_triplets.push(tri.clone());
            tri.atom_k = pairatom_j;
            tri.third_vector = Some(halfbond);
            tri.triplet_i += 1;
            ba_triplets.push(tri);
            // } else {
            //     ba_triplets.push(tri.clone());
            //     tri.bond_vector = Some(-bond.vector);
            //     (tri.atom_i,tri.atom_j) = (tri.atom_j,tri.atom_i);
            //     ba_triplets.push(tri.clone());
            //     tri.atom_k = tri.atom_i;
            //     tri.third_vector = Some(halfbond);
            //     tri.triplet_i += 1;
            //     ba_triplets.push(tri.clone());
            //     tri.bond_vector = Some(bond.vector);
            //     (tri.atom_i,tri.atom_j) = (tri.atom_j,tri.atom_i);
            //     ba_triplets.push(tri);
            // }
            triplet_i += 2;
        }
        
        
        // note: pairs_containing does "full enumeration", but the underlying pair objects only form half enumeration.
        for one_three in system.pairs_containing(bond.first)?.iter().map(|p|reorient_pair(p.clone())) {
            let (third,third_vector) = if one_three.first == bond.first {
                (one_three.second, one_three.vector - halfbond)
            } else {
                debug_assert_eq!(one_three.second, bond.first);
                (one_three.first, -one_three.vector - halfbond)
            };
            
            if third_vector.norm2() < third_cutoff*third_cutoff {
                
                let is_self_contrib = {
                    //ASSUMPTION: is_self_contrib means that bond and one_three are the exact same object
                    (bond.vector-one_three.vector).norm2() <1E-5 &&
                    bond.second == third
                };
                if is_self_contrib{
                    //debug_assert_eq!(&bond as *const Pair, one_three as *const Pair);  // they come from different allocations lol
                    debug_assert_eq!(
                        (bond.first, bond.second, bond.cell_shift_indices),
                        (one_three.first, one_three.second, one_three.cell_shift_indices),
                    );
                    continue;
                }
                
                
                let tri = BATripletInfo{
                    atom_i: bond.first, atom_j: bond.second, atom_k: third,
                    bond_i, triplet_i,
                    bond_vector: Some(bond.vector),
                    third_vector: Some(third_vector),
                    is_self_contrib: false,
                };
                triplet_i += 1;
                // if half_enumeration {
                ba_triplets.push(tri);
                // } else {
                //     ba_triplets.push(tri.clone());
                //     tri.bond_vector = Some(-bond.vector);
                //     (tri.atom_i,tri.atom_j) = (tri.atom_j,tri.atom_i);
                //     ba_triplets.push(tri);
                // }
            }
        }
    }
    Ok(ba_triplets)
}

impl BATripletNeighborList {
    const CACHE_NAME_ATTR: &'static str = "bond_atom_triplets_cutoffs";
    const CACHE_NAME1: &'static str = "bond_atom_triplets_raw_list";
    const CACHE_NAME2: &'static str = "bond_atom_triplets_species_LUT";
    const CACHE_NAME3: &'static str = "bond_atom_triplets_center_LUT";
    //type CACHE_TYPE1 = TensorBlock;
    //type CACHE_TYPE2 = BTreeMap<(i32,i32,i32),Vec<usize>>;
    //type CACHE_TYPE3 = Vec<Vec<Vec<usize>>>;

    /// get the cutoff distance for the selection of bonds
    pub fn bond_cutoff(&self)-> f64 {
        self.cutoffs[0]
    }
    /// get the cutoff distance for neighbours to the center of a bond
    pub fn third_cutoff(&self)-> f64 {
        self.cutoffs[1]
    }

    /// validate that the cutoffs make sense
    pub fn validate_cutoffs(&self) {
        let (bond_cutoff, third_cutoff) = (self.bond_cutoff(), self.third_cutoff());
        assert!(bond_cutoff > 0.0 && bond_cutoff.is_finite());
        assert!(third_cutoff >= bond_cutoff && third_cutoff.is_finite());
    }
    
    /// internal function that deletages computing the triplets, but deals with storing them for a given system.
    fn do_compute_for_system(&self, system: &mut System) -> Result<(), Error> {
        // let triplets_raw = TripletNeighborsList::for_system(&**system, self.bond_cutoff(), self.third_cutoff())?;
        // let triplets = triplets_raw.triplets();
        let triplets = list_raw_triplets(&mut **system, self.cutoffs[0], self.cutoffs[1])?;

        let components = [Labels::new(
            ["vector_pair_component"],
            &[[0x00_i32],[0x01],[0x02], [0x10],[0x11],[0x12]],
        )];
        let properties = Labels::new(["dummy"],&[[0]]);
        
        let mut data = Array3::uninit([triplets.len(),6,1]);
        let mut samples = LabelsBuilder::new(
            vec!["bond_atom_1","bond_atom_2","atom_3","bond_i","triplet_i","is_self_contribution"]
        );
        samples.reserve(triplets.len());
        
        for (triplet_i,triplet) in triplets.iter().enumerate() {
            samples.add(&[
                triplet.atom_i, triplet.atom_j, triplet.atom_k,
                triplet.bond_i, triplet_i, triplet.is_self_contrib as usize,
            ]);
            let mut dataslice = data.slice_mut(s![triplet_i,..,0]);
            // safety: the function called to get the triplets does indeed populate the vectors
            let bv = triplet.bond_vector.as_ref().unwrap();
            let tv = triplet.third_vector.as_ref().unwrap();
            dataslice[0_usize] = MaybeUninit::new(bv[0]);
            dataslice[1_usize] = MaybeUninit::new(bv[1]);
            dataslice[2_usize] = MaybeUninit::new(bv[2]);
            dataslice[3_usize] = MaybeUninit::new(tv[0]);
            dataslice[4_usize] = MaybeUninit::new(tv[1]);
            dataslice[5_usize] = MaybeUninit::new( tv[2]);
        }
        
        // SAFETY: we just spent an entire loop filling this array
        let data: Array3<f64> = unsafe{data.assume_init()};
        //first,second,third,bond_i,is_self_contribution, bond_vec, third_vec
        let block /*:Self::CACHE_TYPE1*/ = TensorBlock::new(
            data.into_dyn(),
            &samples.finish(),
            &components,
            &properties,
        )?;
        system.store_data(Self::CACHE_NAME1.into(),block);
        //let block: &TensorBlock = system.data(&Self::CACHE_NAME1).expect("unreachable: store_data failed".into())
        //    .downcast_ref().expect("unreachable: store_data didn't store the right type".into());
            
        let species = system.species()?;  // calling this again so the previous borrow expires
        let mut triplets_by_species = BTreeMap::new();
        let mut triplets_by_center = {
            let sz = system.size()?;
            (0..sz).map(|i|vec![vec![];i+1]).collect::<Vec<_>>()//vec![vec![vec![];sz];sz]
        };
        for (triplet_i, triplet) in triplets.iter().enumerate() {
            let ((s1,s2),_) = sort_pair((species[triplet.atom_i],species[triplet.atom_j]));
            triplets_by_species.entry((s1,s2,species[triplet.atom_k]))
                .or_insert_with(Vec::new)
                .push(triplet_i);
            if triplet.atom_i >= triplet.atom_j{
                triplets_by_center[triplet.atom_i][triplet.atom_j].push(triplet_i);
            } else {
                triplets_by_center[triplet.atom_j][triplet.atom_i].push(triplet_i);
            }
            // triplets_by_species.entry((species[triplet.bond.first],species[triplet.bond.second],species[triplet.third]))
            //     .or_insert_with(Vec::new)
            //     .push(triplet_i);
            // if self.use_half_enumeration  {
            //     if sort_pair((species[triplet.bond.first], species[triplet.bond.second])).1 {
            //         triplets_by_center[triplet.bond.second][triplet.bond.first].push(triplet_i);
            //     } else {
            //         triplets_by_center[triplet.bond.first][triplet.bond.second].push(triplet_i);
            //     }
            // } else {
            //     triplets_by_center[triplet.bond.first][triplet.bond.second].push(triplet_i);
            //     triplets_by_center[triplet.bond.second][triplet.bond.first].push(triplet_i);
            // }
            
        }
        system.store_data(Self::CACHE_NAME2.into(),triplets_by_species);
        system.store_data(Self::CACHE_NAME3.into(),triplets_by_center);
        system.store_data(Self::CACHE_NAME_ATTR.into(),self.cutoffs);
        Ok(())
    }
    
    /// check that the precalculator has computed its values for a given system,
    /// and if not, compute them.
    pub fn ensure_computed_for_system(&self, system: &mut System) -> Result<(),Error> {
        self.validate_cutoffs();
        'cached_path: {
            let cutoffs2: &[f64;2] = match system.data(Self::CACHE_NAME_ATTR.into()) {
                Some(cutoff) => cutoff.downcast_ref()
                    .ok_or_else(||Error::Internal("Failed to downcast cache".into()))?,
                None => break 'cached_path,
            };
            if cutoffs2 == &self.cutoffs {
                return Ok(());
            } else {
                break 'cached_path
            }
        }
        // got out of the 'cached' path: need to compute this ourselves
        return self.do_compute_for_system(system);
    }
    
    /// for a given system, get a copy of all the bond-atom triplets.
    /// optionally include the vectors tied to these triplets
    pub fn get_for_system(&self, system: &System, with_vectors: bool) -> Result<Vec<BATripletInfo>, Error>{  
        let block: &TensorBlock = system.data(&Self::CACHE_NAME1)
            .ok_or_else(||Error::Internal("triplets not yet computed".into()))?
            .downcast_ref().ok_or_else(||{Error::Internal("Failed to downcast cache".into())})?;
        
        let res:Vec<_> = block.samples().iter_fixed_size().map(
            |[atom_i,atom_j,atom_k,bond_i,triplet_i,is_self_contrib]| {
                let (bond_vector,third_vector) = if with_vectors {
                    let values = block.values();
                    let vecslice = values.as_array().slice(s![triplet_i.usize(),..,0_usize]);
                    let v = vecslice.to_slice().ok_or_else(||Error::Internal("triplet cache does not have vectors be contiguous".into())).unwrap();
                    (Some(Vector3D::new(v[0],v[1],v[2])), Some(Vector3D::new(v[3],v[4],v[5])))
                } else {
                    (None,None)
                };
            BATripletInfo{
                atom_i: atom_i.usize(), atom_j: atom_j.usize(), atom_k: atom_k.usize(),
                bond_i: bond_i.usize(), triplet_i: triplet_i.usize(),
                is_self_contrib: (is_self_contrib.i32()!=0),
                bond_vector, third_vector,
            }
        }).collect();
        
        Ok(res)
    }

    /// for a given system, get a copy of the bond-atom triplets of given set of atomic species.
    /// optionally include the vectors tied to these triplets
    /// note: inverting s1 and s2 does not change the result, and the returned triplets may have these species swapped
    pub fn get_per_system_per_species(&self, system: &System, s1:i32,s2:i32,s3:i32, with_vectors: bool) -> Result<Vec<BATripletInfo>, Error>{  
        let block: &TensorBlock = system.data(&Self::CACHE_NAME1)
            .ok_or_else(||Error::Internal("triplets not yet computed".into()))?
            .downcast_ref().ok_or_else(||{Error::Internal("Failed to downcast cache".into())})?;
        let species_lut: &BTreeMap<(i32,i32,i32),Vec<usize>> = system.data(&Self::CACHE_NAME2)
            .ok_or_else(||Error::Internal("triplets not yet computed".into()))?
            .downcast_ref().ok_or_else(||{Error::Internal("Failed to downcast cache".into())})?;

        let ((s1,s2),_) = sort_pair((s1,s2));
        let species_lut = match species_lut.get(&(s1,s2,s3)) {
            None => {return Ok(vec![])},
            Some(lut) => lut,
        };
        
        let res:Vec<_> = species_lut.iter().map(|triplet_i|{
            //|[atom_i,atom_j,atom_k,bond_i,triplet_i,is_self_contrib]| {
            let samps = block.samples();
            let (atom_i,atom_j,atom_k,bond_i,triplet_i,is_self_contrib) = if let &[a,b,c,d,e,f] = &samps[*triplet_i] {
                (a,b,c,d,e,f)
            } else {
                unreachable!();  // error: wrong length for sample description
            };
            let (bond_vector,third_vector) = if with_vectors {
                let values = block.values();
                let vecslice = values.as_array().slice(s![triplet_i.usize(),..,0_usize]);                    let v = vecslice.to_slice().ok_or_else(||Error::Internal("triplet cache does not have vectors be contiguous".into())).unwrap();                    (Some(Vector3D::new(v[0],v[1],v[2])), Some(Vector3D::new(v[3],v[4],v[5])))
            } else {
                (None,None)
            };
            BATripletInfo{
                atom_i: atom_i.usize(), atom_j: atom_j.usize(), atom_k: atom_k.usize(),
                bond_i: bond_i.usize(), triplet_i: triplet_i.usize(),
                is_self_contrib: (is_self_contrib.i32()!=0),
                bond_vector, third_vector,
            }
        }).collect();
            
        Ok(res)
    }
    
    /// for a given system, get a copy of the bond-atom triplets of given set of atomic species.
    /// optionally include the vectors tied to these triplets
    /// note: the triplets may be for (c2,c1) rather than (c1,c2)
    pub fn get_per_system_per_center(&self, system: &System, c1:usize,c2:usize, with_vectors: bool) -> Result<Vec<BATripletInfo>, Error>{  
        {
            let sz = system.size()?;
            if c1 >= sz || c2 >= sz {
                return Err(Error::InvalidParameter("center ID too high for system".into()));
            }
        }
        
        let block: &TensorBlock = system.data(&Self::CACHE_NAME1)
            .ok_or_else(||Error::Internal("triplets not yet computed".into()))?
            .downcast_ref().ok_or_else(||{Error::Internal("Failed to downcast cache".into())})?;
        let centers_lut: &Vec<Vec<Vec<usize>>> = system.data(&Self::CACHE_NAME3)
            .ok_or_else(||Error::Internal("triplets not yet computed".into()))?
            .downcast_ref().ok_or_else(||{Error::Internal("Failed to downcast cache".into())})?;
        let centers_lut = if c1 >= c2 {
            &centers_lut[c1][c2]
        } else {
            &centers_lut[c2][c1]
        };
        
        let res:Vec<_> = centers_lut.iter().map(|triplet_i|{
            //|[atom_i,atom_j,atom_k,bond_i,triplet_i,is_self_contrib]| {
            let samps = block.samples();
            let (atom_i,atom_j,atom_k,bond_i,triplet_i,is_self_contrib) = if let &[a,b,c,d,e,f] = &samps[*triplet_i] {
                (a,b,c,d,e,f)
            } else {
                unreachable!();  // error: wrong length for sample description
            };
            let (bond_vector,third_vector) = if with_vectors {
                let values = block.values();
                let vecslice = values.as_array().slice(s![triplet_i.usize(),..,0_usize]);                    let v = vecslice.to_slice().ok_or_else(||Error::Internal("triplet cache does not have vectors be contiguous".into())).unwrap();                    (Some(Vector3D::new(v[0],v[1],v[2])), Some(Vector3D::new(v[3],v[4],v[5])))
            } else {
                (None,None)
            };
            BATripletInfo{
                atom_i: atom_i.usize(), atom_j: atom_j.usize(), atom_k: atom_k.usize(),
                bond_i: bond_i.usize(), triplet_i: triplet_i.usize(),
                is_self_contrib: (is_self_contrib.i32()!=0),
                bond_vector, third_vector,
            }
        }).collect();
        
        Ok(res)
    }
    
}



#[cfg(test)]
mod tests {
    use approx::assert_ulps_eq;
    use crate::systems::test_utils::{test_systems};
    //use crate::Matrix3;
    use super::*;

    #[test]
    fn simple_enum() {
        let mut tsysv = test_systems(&["water"]);
        let precalc = BATripletNeighborList{
            cutoffs: [6.,6.],
        };
        precalc.ensure_computed_for_system(&mut tsysv[0]).unwrap();
        
        // /// ensure the enumeration is correct
        let triplets = precalc.get_for_system(&mut tsysv[0], false).unwrap();
        assert_eq!(triplets, vec![
            BATripletInfo{atom_i:0,atom_j:1,atom_k:0,bond_i:0,triplet_i:0,is_self_contrib:true, bond_vector:None,third_vector:None},
            BATripletInfo{atom_i:0,atom_j:1,atom_k:1,bond_i:0,triplet_i:1,is_self_contrib:true, bond_vector:None,third_vector:None},
            BATripletInfo{atom_i:0,atom_j:1,atom_k:2,bond_i:0,triplet_i:2,is_self_contrib:false,bond_vector:None,third_vector:None},
            BATripletInfo{atom_i:0,atom_j:2,atom_k:0,bond_i:1,triplet_i:3,is_self_contrib:true, bond_vector:None,third_vector:None},
            BATripletInfo{atom_i:0,atom_j:2,atom_k:2,bond_i:1,triplet_i:4,is_self_contrib:true, bond_vector:None,third_vector:None},
            BATripletInfo{atom_i:0,atom_j:2,atom_k:1,bond_i:1,triplet_i:5,is_self_contrib:false,bond_vector:None,third_vector:None},
            BATripletInfo{atom_i:1,atom_j:2,atom_k:1,bond_i:2,triplet_i:6,is_self_contrib:true, bond_vector:None,third_vector:None},
            BATripletInfo{atom_i:1,atom_j:2,atom_k:2,bond_i:2,triplet_i:7,is_self_contrib:true, bond_vector:None,third_vector:None},
            BATripletInfo{atom_i:1,atom_j:2,atom_k:0,bond_i:2,triplet_i:8,is_self_contrib:false,bond_vector:None,third_vector:None},
        ]);
        
        // /// ensure the per-center enumeration is correct
        let triplets = precalc.get_per_system_per_center(&mut tsysv[0], 0,1,false).unwrap();
        assert_eq!(triplets, vec![
            BATripletInfo{atom_i:0,atom_j:1,atom_k:0,bond_i:0,triplet_i:0,is_self_contrib:true, bond_vector:None,third_vector:None},
            BATripletInfo{atom_i:0,atom_j:1,atom_k:1,bond_i:0,triplet_i:1,is_self_contrib:true, bond_vector:None,third_vector:None},
            BATripletInfo{atom_i:0,atom_j:1,atom_k:2,bond_i:0,triplet_i:2,is_self_contrib:false,bond_vector:None,third_vector:None},
        ]);
        let triplets = precalc.get_per_system_per_center(&mut tsysv[0], 1,0,false).unwrap();
        assert_eq!(triplets, vec![
            BATripletInfo{atom_i:0,atom_j:1,atom_k:0,bond_i:0,triplet_i:0,is_self_contrib:true, bond_vector:None,third_vector:None},
            BATripletInfo{atom_i:0,atom_j:1,atom_k:1,bond_i:0,triplet_i:1,is_self_contrib:true, bond_vector:None,third_vector:None},
            BATripletInfo{atom_i:0,atom_j:1,atom_k:2,bond_i:0,triplet_i:2,is_self_contrib:false,bond_vector:None,third_vector:None},
        ]);
        let triplets = precalc.get_per_system_per_center(&mut tsysv[0], 0,2,false).unwrap();
        assert_eq!(triplets, vec![
            BATripletInfo{atom_i:0,atom_j:2,atom_k:0,bond_i:1,triplet_i:3,is_self_contrib:true, bond_vector:None,third_vector:None},
            BATripletInfo{atom_i:0,atom_j:2,atom_k:2,bond_i:1,triplet_i:4,is_self_contrib:true, bond_vector:None,third_vector:None},
            BATripletInfo{atom_i:0,atom_j:2,atom_k:1,bond_i:1,triplet_i:5,is_self_contrib:false,bond_vector:None,third_vector:None},
        ]);
        let triplets = precalc.get_per_system_per_center(&mut tsysv[0], 2,0,false).unwrap();
        assert_eq!(triplets, vec![
            BATripletInfo{atom_i:0,atom_j:2,atom_k:0,bond_i:1,triplet_i:3,is_self_contrib:true, bond_vector:None,third_vector:None},
            BATripletInfo{atom_i:0,atom_j:2,atom_k:2,bond_i:1,triplet_i:4,is_self_contrib:true, bond_vector:None,third_vector:None},
            BATripletInfo{atom_i:0,atom_j:2,atom_k:1,bond_i:1,triplet_i:5,is_self_contrib:false,bond_vector:None,third_vector:None},
        ]);
        let triplets = precalc.get_per_system_per_center(&mut tsysv[0], 1,2,false).unwrap();
        assert_eq!(triplets, vec![
            BATripletInfo{atom_i:1,atom_j:2,atom_k:1,bond_i:2,triplet_i:6,is_self_contrib:true, bond_vector:None,third_vector:None},
            BATripletInfo{atom_i:1,atom_j:2,atom_k:2,bond_i:2,triplet_i:7,is_self_contrib:true, bond_vector:None,third_vector:None},
            BATripletInfo{atom_i:1,atom_j:2,atom_k:0,bond_i:2,triplet_i:8,is_self_contrib:false,bond_vector:None,third_vector:None},
        ]);
        let triplets = precalc.get_per_system_per_center(&mut tsysv[0], 2,1,false).unwrap();
        assert_eq!(triplets, vec![
            BATripletInfo{atom_i:1,atom_j:2,atom_k:1,bond_i:2,triplet_i:6,is_self_contrib:true, bond_vector:None,third_vector:None},
            BATripletInfo{atom_i:1,atom_j:2,atom_k:2,bond_i:2,triplet_i:7,is_self_contrib:true, bond_vector:None,third_vector:None},
            BATripletInfo{atom_i:1,atom_j:2,atom_k:0,bond_i:2,triplet_i:8,is_self_contrib:false,bond_vector:None,third_vector:None},
        ]);
        
        // /// ensure the per-species enumeration is correct
        let triplets = precalc.get_per_system_per_species(&mut tsysv[0], 1,1, -42,false).unwrap();
        assert_eq!(triplets, vec![
            BATripletInfo{atom_i:1,atom_j:2,atom_k:0,bond_i:2,triplet_i:8,is_self_contrib:false,bond_vector:None,third_vector:None},
        ]);
        let triplets = precalc.get_per_system_per_species(&mut tsysv[0], 1,1, 1,false).unwrap();
        assert_eq!(triplets, vec![
            BATripletInfo{atom_i:1,atom_j:2,atom_k:1,bond_i:2,triplet_i:6,is_self_contrib:true, bond_vector:None,third_vector:None},
            BATripletInfo{atom_i:1,atom_j:2,atom_k:2,bond_i:2,triplet_i:7,is_self_contrib:true, bond_vector:None,third_vector:None},
        ]);
        let triplets = precalc.get_per_system_per_species(&mut tsysv[0], 1,-42, 1,false).unwrap();
        assert_eq!(triplets, vec![
            BATripletInfo{atom_i:0,atom_j:1,atom_k:1,bond_i:0,triplet_i:1,is_self_contrib:true, bond_vector:None,third_vector:None},
            BATripletInfo{atom_i:0,atom_j:1,atom_k:2,bond_i:0,triplet_i:2,is_self_contrib:false,bond_vector:None,third_vector:None},
            BATripletInfo{atom_i:0,atom_j:2,atom_k:2,bond_i:1,triplet_i:4,is_self_contrib:true, bond_vector:None,third_vector:None},
            BATripletInfo{atom_i:0,atom_j:2,atom_k:1,bond_i:1,triplet_i:5,is_self_contrib:false,bond_vector:None,third_vector:None},
        ]);
        let triplets = precalc.get_per_system_per_species(&mut tsysv[0], -42,1, 1,false).unwrap();
        assert_eq!(triplets, vec![
            BATripletInfo{atom_i:0,atom_j:1,atom_k:1,bond_i:0,triplet_i:1,is_self_contrib:true, bond_vector:None,third_vector:None},
            BATripletInfo{atom_i:0,atom_j:1,atom_k:2,bond_i:0,triplet_i:2,is_self_contrib:false,bond_vector:None,third_vector:None},
            BATripletInfo{atom_i:0,atom_j:2,atom_k:2,bond_i:1,triplet_i:4,is_self_contrib:true, bond_vector:None,third_vector:None},
            BATripletInfo{atom_i:0,atom_j:2,atom_k:1,bond_i:1,triplet_i:5,is_self_contrib:false,bond_vector:None,third_vector:None},
        ]);
        
        // ///// deal with the vectors

        let triplets = precalc.get_for_system(&mut tsysv[0], true).unwrap();
        let (bondvecs, thirdvecs): (Vec<_>,Vec<_>) = triplets.into_iter().map(|t|(t.bond_vector.unwrap(),t.third_vector.unwrap()))
            .unzip();
        
        bondvecs.into_iter().map(|v|(v[0],v[1],v[2]))
            .zip(vec![
                (0.0, 0.75545, -0.58895),
                (0.0, 0.75545, -0.58895),
                (0.0, 0.75545, -0.58895),
                (0.0, -0.75545, -0.58895),
                (0.0, -0.75545, -0.58895),
                (0.0, -0.75545, -0.58895),
                (0.0, -1.5109, 0.0),
                (0.0, -1.5109, 0.0),
                (0.0, -1.5109, 0.0),
            ].into_iter())
            .map(|(v1,v2)|{
                assert_ulps_eq!(v1.0,v2.0);
                assert_ulps_eq!(v1.1,v2.1);
                assert_ulps_eq!(v1.2,v2.2);
            }).last();
        
        thirdvecs.into_iter().map(|v|(v[0],v[1],v[2]))
            .zip(vec![
                (0.0, -0.377725, 0.294475),
                (0.0, 0.377725, -0.294475),
                (0.0, -1.133175, -0.294475),
                (0.0, 0.377725, 0.294475),
                (0.0, -0.377725, -0.294475),
                (0.0, 1.133175, -0.294475),
                (0.0, 0.75545, 0.0),
                (0.0, -0.75545, 0.0),
                (0.0, 0.0, 0.58895),
            ].into_iter())
            .map(|(v1,v2)|{
                assert_ulps_eq!(v1.0,v2.0);
                assert_ulps_eq!(v1.1,v2.1);
                assert_ulps_eq!(v1.2,v2.2);
            }).last();
    }

    // #[test]
    // fn full_enum() {
    //     let mut tsysv = test_systems(&["water"]);
    //     let precalc = BATripletNeighborList{
    //         cutoffs: [6.,6.],
    //         use_half_enumeration: false,
    //     };
        
    //     precalc.ensure_computed_for_system(&mut tsysv[0]).unwrap();
    //     let triplets = precalc.get_for_system(&mut tsysv[0], false).unwrap();
    //     assert_eq!(triplets, vec![
    //         BATripletInfo{atom_i:0,atom_j:1,atom_k:0,bond_i:0,triplet_i:0,is_self_contrib:true, bond_vector:None,third_vector:None},
    //         BATripletInfo{atom_i:0,atom_j:1,atom_k:0,bond_i:0,triplet_i:0,is_self_contrib:true, bond_vector:None,third_vector:None},
    //         BATripletInfo{atom_i:0,atom_j:1,atom_k:1,bond_i:0,triplet_i:1,is_self_contrib:true, bond_vector:None,third_vector:None},
    //         BATripletInfo{atom_i:0,atom_j:1,atom_k:1,bond_i:0,triplet_i:1,is_self_contrib:true, bond_vector:None,third_vector:None},
    //         BATripletInfo{atom_i:0,atom_j:1,atom_k:2,bond_i:0,triplet_i:2,is_self_contrib:false,bond_vector:None,third_vector:None},
    //         BATripletInfo{atom_i:0,atom_j:1,atom_k:2,bond_i:0,triplet_i:2,is_self_contrib:false,bond_vector:None,third_vector:None},
    //         BATripletInfo{atom_i:0,atom_j:2,atom_k:0,bond_i:1,triplet_i:3,is_self_contrib:true, bond_vector:None,third_vector:None},
    //         BATripletInfo{atom_i:0,atom_j:2,atom_k:0,bond_i:1,triplet_i:3,is_self_contrib:true, bond_vector:None,third_vector:None},
    //         BATripletInfo{atom_i:0,atom_j:2,atom_k:1,bond_i:1,triplet_i:4,is_self_contrib:false,bond_vector:None,third_vector:None},
    //         BATripletInfo{atom_i:0,atom_j:2,atom_k:1,bond_i:1,triplet_i:4,is_self_contrib:false,bond_vector:None,third_vector:None},
    //         BATripletInfo{atom_i:0,atom_j:2,atom_k:2,bond_i:1,triplet_i:5,is_self_contrib:true, bond_vector:None,third_vector:None},
    //         BATripletInfo{atom_i:0,atom_j:2,atom_k:2,bond_i:1,triplet_i:5,is_self_contrib:true, bond_vector:None,third_vector:None},
    //         BATripletInfo{atom_i:1,atom_j:2,atom_k:0,bond_i:2,triplet_i:6,is_self_contrib:false,bond_vector:None,third_vector:None},
    //         BATripletInfo{atom_i:1,atom_j:2,atom_k:0,bond_i:2,triplet_i:6,is_self_contrib:false,bond_vector:None,third_vector:None},
    //         BATripletInfo{atom_i:1,atom_j:2,atom_k:1,bond_i:2,triplet_i:7,is_self_contrib:true, bond_vector:None,third_vector:None},
    //         BATripletInfo{atom_i:1,atom_j:2,atom_k:1,bond_i:2,triplet_i:7,is_self_contrib:true, bond_vector:None,third_vector:None},
    //         BATripletInfo{atom_i:1,atom_j:2,atom_k:2,bond_i:2,triplet_i:8,is_self_contrib:true, bond_vector:None,third_vector:None},
    //         BATripletInfo{atom_i:1,atom_j:2,atom_k:2,bond_i:2,triplet_i:8,is_self_contrib:true, bond_vector:None,third_vector:None},
    //     ]);
        
    //     let triplets = precalc.get_per_system_per_center(&mut tsysv[0], 0,1,false).unwrap();
    //     assert_eq!(triplets, vec![
    //         BATripletInfo{atom_i:0,atom_j:1,atom_k:0,bond_i:0,triplet_i:0,is_self_contrib:true, bond_vector:None,third_vector:None},
    //         BATripletInfo{atom_i:0,atom_j:1,atom_k:1,bond_i:0,triplet_i:1,is_self_contrib:true, bond_vector:None,third_vector:None},
    //         BATripletInfo{atom_i:0,atom_j:1,atom_k:2,bond_i:0,triplet_i:2,is_self_contrib:false,bond_vector:None,third_vector:None},
    //     ]);
    //     let triplets = precalc.get_per_system_per_center(&mut tsysv[0], 1,0,false).unwrap();
    //     assert_eq!(triplets, vec![
    //         BATripletInfo{atom_i:0,atom_j:1,atom_k:0,bond_i:0,triplet_i:0,is_self_contrib:true, bond_vector:None,third_vector:None},
    //         BATripletInfo{atom_i:0,atom_j:1,atom_k:1,bond_i:0,triplet_i:1,is_self_contrib:true, bond_vector:None,third_vector:None},
    //         BATripletInfo{atom_i:0,atom_j:1,atom_k:2,bond_i:0,triplet_i:2,is_self_contrib:false,bond_vector:None,third_vector:None},
    //     ]);
    //     let triplets = precalc.get_per_system_per_center(&mut tsysv[0], 0,2,false).unwrap();
    //     assert_eq!(triplets, vec![
    //         BATripletInfo{atom_i:0,atom_j:2,atom_k:0,bond_i:1,triplet_i:3,is_self_contrib:true, bond_vector:None,third_vector:None},
    //         BATripletInfo{atom_i:0,atom_j:2,atom_k:1,bond_i:1,triplet_i:4,is_self_contrib:false,bond_vector:None,third_vector:None},
    //         BATripletInfo{atom_i:0,atom_j:2,atom_k:2,bond_i:1,triplet_i:5,is_self_contrib:true, bond_vector:None,third_vector:None},
    //     ]);
    //     let triplets = precalc.get_per_system_per_center(&mut tsysv[0], 2,0,false).unwrap();
    //     assert_eq!(triplets, vec![
    //         BATripletInfo{atom_i:0,atom_j:2,atom_k:0,bond_i:1,triplet_i:3,is_self_contrib:true, bond_vector:None,third_vector:None},
    //         BATripletInfo{atom_i:0,atom_j:2,atom_k:1,bond_i:1,triplet_i:4,is_self_contrib:false,bond_vector:None,third_vector:None},
    //         BATripletInfo{atom_i:0,atom_j:2,atom_k:2,bond_i:1,triplet_i:5,is_self_contrib:true, bond_vector:None,third_vector:None},
    //     ]);
    //     let triplets = precalc.get_per_system_per_center(&mut tsysv[0], 1,2,false).unwrap();
    //     assert_eq!(triplets, vec![
    //         BATripletInfo{atom_i:1,atom_j:2,atom_k:0,bond_i:2,triplet_i:6,is_self_contrib:false,bond_vector:None,third_vector:None},
    //         BATripletInfo{atom_i:1,atom_j:2,atom_k:1,bond_i:2,triplet_i:7,is_self_contrib:true, bond_vector:None,third_vector:None},
    //         BATripletInfo{atom_i:1,atom_j:2,atom_k:2,bond_i:2,triplet_i:8,is_self_contrib:true, bond_vector:None,third_vector:None},
    //     ]);
    //     let triplets = precalc.get_per_system_per_center(&mut tsysv[0], 2,1,false).unwrap();
    //     assert_eq!(triplets, vec![
    //         BATripletInfo{atom_i:1,atom_j:2,atom_k:0,bond_i:2,triplet_i:6,is_self_contrib:false,bond_vector:None,third_vector:None},
    //         BATripletInfo{atom_i:1,atom_j:2,atom_k:1,bond_i:2,triplet_i:7,is_self_contrib:true, bond_vector:None,third_vector:None},
    //         BATripletInfo{atom_i:1,atom_j:2,atom_k:2,bond_i:2,triplet_i:8,is_self_contrib:true, bond_vector:None,third_vector:None},
    //     ]);
        
    //     // /// ensure the per-species enumeration is correct
    //     let triplets = precalc.get_per_system_per_species(&mut tsysv[0], 1,1, -42,false).unwrap();
    //     assert_eq!(triplets, vec![
    //         BATripletInfo{atom_i:1,atom_j:2,atom_k:0,bond_i:2,triplet_i:6,is_self_contrib:false,bond_vector:None,third_vector:None},
    //     ]);
    //     let triplets = precalc.get_per_system_per_species(&mut tsysv[0], 1,1, 1,false).unwrap();
    //     assert_eq!(triplets, vec![
    //         BATripletInfo{atom_i:1,atom_j:2,atom_k:1,bond_i:2,triplet_i:7,is_self_contrib:true, bond_vector:None,third_vector:None},
    //         BATripletInfo{atom_i:1,atom_j:2,atom_k:2,bond_i:2,triplet_i:8,is_self_contrib:true, bond_vector:None,third_vector:None},
    //     ]);
    //     let triplets = precalc.get_per_system_per_species(&mut tsysv[0], 1,-42, 1,false).unwrap();
    //     assert_eq!(triplets, vec![
    //         BATripletInfo{atom_i:0,atom_j:1,atom_k:1,bond_i:0,triplet_i:1,is_self_contrib:true, bond_vector:None,third_vector:None},
    //         BATripletInfo{atom_i:0,atom_j:1,atom_k:2,bond_i:0,triplet_i:2,is_self_contrib:false,bond_vector:None,third_vector:None},
    //         BATripletInfo{atom_i:0,atom_j:2,atom_k:1,bond_i:1,triplet_i:4,is_self_contrib:false,bond_vector:None,third_vector:None},
    //         BATripletInfo{atom_i:0,atom_j:2,atom_k:2,bond_i:1,triplet_i:5,is_self_contrib:true, bond_vector:None,third_vector:None},
    //     ]);
    //     let triplets = precalc.get_per_system_per_species(&mut tsysv[0], -42,1, 1,false).unwrap();
    //     assert_eq!(triplets, vec![
    //         BATripletInfo{atom_i:0,atom_j:1,atom_k:1,bond_i:0,triplet_i:1,is_self_contrib:true, bond_vector:None,third_vector:None},
    //         BATripletInfo{atom_i:0,atom_j:1,atom_k:2,bond_i:0,triplet_i:2,is_self_contrib:false,bond_vector:None,third_vector:None},
    //         BATripletInfo{atom_i:0,atom_j:2,atom_k:1,bond_i:1,triplet_i:4,is_self_contrib:false,bond_vector:None,third_vector:None},
    //         BATripletInfo{atom_i:0,atom_j:2,atom_k:2,bond_i:1,triplet_i:5,is_self_contrib:true, bond_vector:None,third_vector:None},
    //     ]);
    // }
}

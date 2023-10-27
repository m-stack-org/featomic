use std::collections::{BTreeSet,BTreeMap};
use std::collections::btree_map::Entry;
use log::warn;

use metatensor::{TensorMap, TensorBlockRefMut};
use metatensor::{Labels, LabelsBuilder, LabelValue};

use crate::calculators::CalculatorBase;

use crate::{Error, System};
use crate::types::{Vector3D,Matrix3};
use crate::systems::bond_atom_neighbors::{BATripletInfo,BATripletNeighborList};



/// for a given vector (`vec`), compute a rotation matrix (`M`) so that `M×vec`
/// is expressed as `(0,0,+z)`
/// currently, this matrix corresponds to a rotatoin expressed as `-z;+y;+z` in euler angles,
/// or as `(x,y,0),theta` in axis-angle representation.
fn rotate_vector_to_z(vec: Vector3D) -> Matrix3 {
    // re-orientation is done through a rotation matrix, computed through the axis-angle and quaternion representations of the rotation
    // axis/angle representation of the rotation: axis is norm(-y,x,0), angle is arctan2( sqrt(x**2+y**2), z)
    // meaning sin(angle) = sqrt((x**2+y**2) /r2); cos(angle) = z/sqrt(r2)

    let (xylen,len) = {
        let xyl = vec[0]*vec[0] + vec[1]*vec[1];
        (xyl.sqrt(), (xyl+vec[2]*vec[2]).sqrt())
    };
    
    if xylen.abs()<1E-7 {
        if vec[2] < 0. {
            return Matrix3::new([[-1.,0.,0.], [0.,1.,0.], [0.,0.,-1.]])
        }
        else {
            return Matrix3::new([[1.,0.,0.], [0.,1.,0.], [0.,0.,1.]])
        }
    }
    
    let c = vec[2]/len;
    let s = xylen/len;
    let t = 1. - c;
    
    let x2 = -vec[1]/xylen;
    let y2 =  vec[0]/xylen;

    let tx = t*x2;
    let sx = s*x2;
    let sy = s*y2;

    return Matrix3::new([
        [tx*x2 +c,  tx*y2,       -sy],
        [tx*y2,     t*y2*y2 + c, sx],
        [sy,       -sx,          c],
    ]);
}


/// returns the derivatives of the reoriention matrix with the three components of the vector to reorient
fn rotate_vector_to_z_derivatives(vec: Vector3D) -> (Matrix3,Matrix3,Matrix3) {

    let (xylen,len) = {
        let xyl = vec[0]*vec[0] + vec[1]*vec[1];
        (xyl.sqrt(), (xyl+vec[2]*vec[2]).sqrt())
    };
    
    if xylen.abs()<1E-7 {
        let co = 1./len;
        if vec[2] < 0. {
            warn!("trying to get the derivative of a rotation near a breaking point: expect pure jank");
            return (
                //Matrix3::new([[-1.,0.,0.], [0.,1.,0.], [0.,0.,-1.]]) <- the value to derive off of: a +y rotation
                Matrix3::new([[0.,0.,-co], [0.,0.,0.], [co,0.,0.]]),  // +x change -> +y rotation
                Matrix3::new([[0.,0.,0.], [0.,0.,-co], [0.,-co,0.]]),  // +y change -> -x rotation
                Matrix3::new([[0.,0.,0.], [0.,0.,0.], [0.,0.,0.]]),  // +z change -> nuthin
            )
        }
        else {
            return (
                //Matrix3::new([[1.,0.,0.], [0.,1.,0.], [0.,0.,1.]])  <- the value to derive off of
                Matrix3::new([[0.,0.,-co], [0.,0.,0.], [co,0.,0.]]),  // +x change -> -y rotation
                Matrix3::new([[0.,0.,0.], [0.,0.,-co], [0.,co,0.]]),  // +y change -> +x rotation
                Matrix3::new([[0.,0.,0.], [0.,0.,0.], [0.,0.,0.]]),  // +z change -> nuthin
            )
        }
    }
    
    let inv_len = 1./len;
    let inv_len2 = inv_len*inv_len;
    let inv_len3 = inv_len2*inv_len;
    let inv_xy = 1./xylen;
    let inv_xy2 = inv_xy*inv_xy;
    let inv_xy3 = inv_xy2*inv_xy;
    
    let c = vec[2]/len;  // needed
    let dcdz = 1./len - vec[2]*vec[2]*inv_len3;
    let dcdx = -vec[2]*vec[0]*inv_len3;
    let dcdy = -vec[2]*vec[1]*inv_len3;
    let s = xylen/len;
    let dsdx = vec[0]*inv_len*(inv_xy - xylen*inv_len2);
    let dsdy = vec[1]*inv_len*(inv_xy - xylen*inv_len2);
    let dsdz = -xylen*vec[2]*inv_len3;
    
    let t = 1. - c;
    
    let x2 = -vec[1]*inv_xy;
    let dx2dx = vec[1]*vec[0]*inv_xy3;
    let dx2dy = inv_xy * (-1. + vec[1]*vec[1]*inv_xy2);
    
    let y2 =  vec[0]/xylen;
    let dy2dy = -vec[1]*vec[0]*inv_xy3;
    let dy2dx = inv_xy * (1. - vec[0]*vec[0]*inv_xy2);

    let tx = t*x2;
    let dtxdx = -dcdx*x2 + t*dx2dx;
    let dtxdy = -dcdy*x2 + t*dx2dy;
    let dtxdz = -dcdz*x2;
    
    //let sx = s*x2;  // needed
    let dsxdx = dsdx*x2 + s*dx2dx;
    let dsxdy = dsdy*x2 + s*dx2dy;
    let dsxdz = dsdz*x2;
    
    //let sy = s*y2;  //needed
    let dsydx = dsdx*y2 + s*dy2dx;
    let dsydy = dsdy*y2 + s*dy2dy;
    let dsydz = dsdz*y2;
    
    //let t1 = tx*x2 +c;  // needed
    let dt1dx = dcdx + dtxdx*x2 + tx*dx2dx;
    let dt1dy = dcdy + dtxdy*x2 + tx*dx2dy;
    let dt1dz = dcdz + dtxdz*x2;
    
    //let t2 = tx*y2;  // needed
    let dt2dx = dtxdx*y2 + tx*dy2dx;
    let dt2dy = dtxdy*y2 + tx*dy2dy;
    let dt2dz = dtxdz*y2;
    
    //let t3 = t*y2*y2 +c;  // needed
    let dt3dx = -dcdx*y2*y2 + 2.*t*y2*dy2dx +dcdx; 
    let dt3dy = -dcdy*y2*y2 + 2.*t*y2*dy2dy +dcdy;
    let dt3dz = -dcdz*y2*y2 +dcdz;

    return (
        // Matrix3::new([
        //     [tx*x2 +c,  tx*y2,       -sy],
        //     [tx*y2,     t*y2*y2 + c, sx],
        //     [sy,       -sx,          c],
        // ]),
        Matrix3::new([
            [dt1dx,  dt2dx, -dsydx],
            [dt2dx,  dt3dx,  dsxdx],
            [dsydx, -dsxdx,  dcdx],
        ]),
        Matrix3::new([
            [dt1dy,  dt2dy, -dsydy],
            [dt2dy,  dt3dy,  dsxdy],
            [dsydy, -dsxdy,  dcdy],
        ]),
        Matrix3::new([
            [dt1dz,  dt2dz, -dsydz],
            [dt2dz,  dt3dz,  dsxdz],
            [dsydz, -dsxdz,  dcdz],
        ]),
    );
}



/// Manages a list of 'neighbors', where one neighbor is the center of a pair of atoms
/// (first and second atom), and the other neighbor is a simple atom (third atom).
/// Both the length of the bond and the distance between neighbors are subjected to a spherical cutoff.
/// 
/// Unlike the corresponding pre_calculator, this calculator focuses on storing
/// the canonical-orientation vector between bond and atom, rather than the bond vector and 'third vector'.
///
/// Users can request either a "full" neighbor list (including an entry for both
/// `i-j +k` triplets and `j-i +k` triplets) or save memory/computational by only
/// working with "half" neighbor list (only including one entry for each `i-j +k`
/// bond)
/// When using a half neighbor list, i and j are ordered so the atom with the smallest species comes first.
///
/// The two first atoms must not be the same atom, but the third atom may be one of them,
/// if the `bond_conbtribution` option is active
/// (When periodic boundaries arise, atom which  must not be the same may be images of each other.)
///
/// This sample produces a single property (`"distance"`) with three components
/// (`"vector_direction"`) containing the x, y, and z component of the vector from
/// the center of the triplet's 'bond' to the triplet's 'third atom', in the bond's canonical orientation.
/// 
/// In addition to the atom indexes, the samples also contain a pair and triplet index,
/// to be able to distinguish between multiple triplets involving the same atoms
/// (which can occur in periodic boundary conditions when the cutoffs are larger than the unit cell).
#[derive(Debug, Clone)]
#[derive(serde::Deserialize, serde::Serialize, schemars::JsonSchema)]
pub struct BANeighborList {
    /// the pre-calculator responsible for making a raw enumeration of the system's bond-atom triplets
    pub raw_triplets: BATripletNeighborList,
    /// Should we include triplets where the third atom is one of the bond's atoms?
    pub bond_contribution: bool,
    /// Should we compute a full neighbor list (each triplet appears twice, once as
    /// `i-j +k` and once as `j-i +k`), or a half neighbor list (each triplet only
    /// appears once, (such that `species_i <= species_j`))
    pub use_half_enumeration: bool,
}

/// Sort a pair and return true if the pair was inverted
#[inline]
fn sort_pair((i, j): (i32, i32)) -> ((i32, i32), bool) {
    if i <= j {
        ((i, j), false)
    } else {
        ((j, i), true)
    }
}

#[derive(Default,Debug)]
pub(super) struct DistanceResult{
    pub(super) vect: Vector3D,
    pub(super) grads: [Option<(usize,Matrix3)>;3],  // matrix is [quantity_component,gradient_component]
}

impl BANeighborList {
    /// get the cutoff distance for the selection of bonds
    pub fn bond_cutoff(&self)-> f64 {
        self.raw_triplets.bond_cutoff()
    }
    /// get the cutoff distance for neighbours to the center of a bond
    pub fn third_cutoff(&self)-> f64 {
        self.raw_triplets.third_cutoff()
    }

    /// a "flatter" initialisation method than the structure-based one
    pub fn from_params(cutoffs: [f64;2], use_half_enumeration: bool, bond_contribution: bool) -> Self {
        Self{
            raw_triplets: BATripletNeighborList {
                cutoffs,
            },
            use_half_enumeration,
            bond_contribution,
        }
    }
    
    /// the core of the calculation being done here:
    /// computing the canonical-orientation vector and distance of a given bond-atom triplet.
    pub(super) fn compute_single_triplet(
        triplet: &BATripletInfo,
        invert: bool,
        compute_grad: bool,
        mtx_cache: &mut BTreeMap<(usize,bool),Matrix3>,
        dmtx_cache: &mut BTreeMap<(usize,bool),(Matrix3,Matrix3,Matrix3)>,
    ) -> Result<DistanceResult,Error> {

        let bond_vector = triplet.bond_vector
            .ok_or_else(||Error::InvalidParameter("triplet for compute_single_triplet should have vectors set".into()))?;
        let third_vector = triplet.third_vector
            .ok_or_else(||Error::InvalidParameter("triplet for compute_single_triplet should have vectors set".into()))?;
        let (atom_i,atom_j,bond_vector) = if invert {
            (triplet.atom_j, triplet.atom_i, -bond_vector)
        } else {
            (triplet.atom_i, triplet.atom_j, bond_vector)
        };
        
        let mut res = DistanceResult::default();
        
        if triplet.is_self_contrib {
            let vec_len = third_vector.norm();
            let vec_len = if third_vector * bond_vector > 0. {
                // third atom on second atom
                vec_len
            } else {
                // third atom on first atom
                -vec_len
            };
                
            res.vect[2] = vec_len;
                        
            if compute_grad {
                            
                let inv_len = 1./vec_len;
                
                res.grads[0] = Some((atom_i,Matrix3::new([
                    [ -0.25* inv_len * third_vector[0], 0., 0.],
                    [ 0., -0.25* inv_len * third_vector[0], 0.],
                    [ 0., 0., -0.25* inv_len * third_vector[0]],
                ])));
                res.grads[1] = Some((atom_j,Matrix3::new([
                    [ 0.25* inv_len * third_vector[0], 0., 0.],
                    [ 0., 0.25* inv_len * third_vector[0], 0.],
                    [ 0., 0., 0.25* inv_len * third_vector[0]],
                ])));

            }
        } else {
                    
            let tf_mtx = match mtx_cache.entry((triplet.bond_i,invert)) {
                Entry::Occupied(entry) => entry.get().clone(),
                Entry::Vacant(entry) => {
                    entry.insert(rotate_vector_to_z(bond_vector)).clone()
                },
            };
            res.vect = tf_mtx * third_vector;
            
            //println!("{} {:?} {:?}", invert, triplet, res);

            if compute_grad {

                // for a transformed vector v from an untransformed vector u,
                // dv = TF*du + dTF*u
                // also: the indexing of the gradient array is: i_gradsample, derivation_component, value_component, i_property

                let du_term = -0.5* tf_mtx;
                let (tf_mtx_dx, tf_mtx_dy, tf_mtx_dz) = match dmtx_cache.entry((triplet.bond_i,invert)) {
                    Entry::Occupied(entry) => entry.get().clone(),
                    Entry::Vacant(entry) => {
                        entry.insert(rotate_vector_to_z_derivatives(bond_vector)).clone()
                    },
                };
                            
                let dmat_term_dx = tf_mtx_dx * third_vector;
                let dmat_term_dy = tf_mtx_dy * third_vector;
                let dmat_term_dz = tf_mtx_dz * third_vector;
                
                res.grads[0] = Some((atom_i,Matrix3::new([
                    [-dmat_term_dx[0] + du_term[0][0], -dmat_term_dy[0] + du_term[0][1], -dmat_term_dz[0] + du_term[0][2]],
                    [-dmat_term_dx[1] + du_term[1][0], -dmat_term_dy[1] + du_term[1][1], -dmat_term_dz[1] + du_term[1][2]],
                    [-dmat_term_dx[2] + du_term[2][0], -dmat_term_dy[2] + du_term[2][1], -dmat_term_dz[2] + du_term[2][2]],
                ])));
                res.grads[1] = Some((atom_j,Matrix3::new([
                    [dmat_term_dx[0] + du_term[0][0], dmat_term_dy[0] + du_term[0][1], dmat_term_dz[0] + du_term[0][2]],
                    [dmat_term_dx[1] + du_term[1][0], dmat_term_dy[1] + du_term[1][1], dmat_term_dz[1] + du_term[1][2]],
                    [dmat_term_dx[2] + du_term[2][0], dmat_term_dy[2] + du_term[2][1], dmat_term_dz[2] + du_term[2][2]],
                ])));
                res.grads[2] = Some((triplet.atom_k,tf_mtx));
            }
        }
        return Ok(res);
    }
    
    /// get the canonical-orientation vector and distance of a triplet
    /// and store it in a TensorBlock
    fn compute_single_triplet_inplace(
        triplet: &BATripletInfo,
        out_block: &mut TensorBlockRefMut,
        sample_i: usize,
        system_i: usize,
        invert: bool,
        mtx_cache: &mut BTreeMap<(usize,bool),Matrix3>,
        dmtx_cache: &mut BTreeMap<(usize,bool),(Matrix3,Matrix3,Matrix3)>,
    ) -> Result<(),Error> {
        let compute_grad = out_block.gradient_mut("positions").is_some();
        let block_data = out_block.data_mut();
        let array = block_data.values.to_array_mut();
        
        let res = Self::compute_single_triplet(
            triplet,
            invert,
            compute_grad,
            mtx_cache,
            dmtx_cache
        )?;
        
        array[[sample_i, 0, 0]] = res.vect[0];
        array[[sample_i, 1, 0]] = res.vect[1];
        array[[sample_i, 2, 0]] = res.vect[2];
        
        if let Some(mut gradient) = out_block.gradient_mut("positions") {
            let gradient = gradient.data_mut();
            let array = gradient.values.to_array_mut();
        
            for grad in res.grads {
                if let Some((atom_i, grad_mtx)) = grad {
                    let grad_sample_i = gradient.samples.position(&[
                        sample_i.into(), system_i.into(), atom_i.into()
                    ]).expect("missing gradient sample");
                    
                    array[[grad_sample_i, 0, 0, 0]] = grad_mtx[0][0];
                    array[[grad_sample_i, 1, 0, 0]] = grad_mtx[0][1];
                    array[[grad_sample_i, 2, 0, 0]] = grad_mtx[0][2];
                    array[[grad_sample_i, 0, 1, 0]] = grad_mtx[1][0];
                    array[[grad_sample_i, 1, 1, 0]] = grad_mtx[1][1];
                    array[[grad_sample_i, 2, 1, 0]] = grad_mtx[1][2];
                    array[[grad_sample_i, 0, 2, 0]] = grad_mtx[2][0];
                    array[[grad_sample_i, 1, 2, 0]] = grad_mtx[2][1];
                    array[[grad_sample_i, 2, 2, 0]] = grad_mtx[2][2];
                }
            }
        }
        Ok(())
    }

}

impl CalculatorBase for BANeighborList {
    fn name(&self) -> String {
        "neighbors list".into()
    }

    fn parameters(&self) -> String {
        serde_json::to_string(self).expect("failed to serialize to JSON")
    }
    
    fn cutoffs(&self) -> &[f64] {
        &self.raw_triplets.cutoffs
    }

    fn keys(&self, systems: &mut [System]) -> Result<Labels, Error> {
        let mut all_species_triplets = BTreeSet::new();
        for system in systems {
            self.raw_triplets.ensure_computed_for_system(system)?;
            let triplets = self.raw_triplets.get_for_system(system, false)?;
            let species = system.species()?;

            for triplet in triplets {
                // filter the self-contribtions if necessary
                // contain any 'third atoms' sith the same type as atoms of the triplet's bond
                if (!self.bond_contribution) && triplet.is_self_contrib {
                    continue;
                }
                if self.use_half_enumeration {
                    let (bond_type, _) = sort_pair((species[triplet.atom_i], species[triplet.atom_j]));
                    all_species_triplets.insert((bond_type.0, bond_type.1, species[triplet.atom_k]));
                } else {
                    all_species_triplets.insert((species[triplet.atom_i], species[triplet.atom_j], species[triplet.atom_k]));
                    all_species_triplets.insert((species[triplet.atom_j], species[triplet.atom_i], species[triplet.atom_k]));
                }
            }
        }

        let mut keys = LabelsBuilder::new(vec!["species_first_bond_atom", "species_second_bond_atom", "species_third_atom"]);
        for (first, second, third) in all_species_triplets {
            keys.add(&[first, second, third]);
        }

        return Ok(keys.finish());
    }

    fn sample_names(&self) -> Vec<&str> {
        return vec!["structure", "triplet_i", "pair_i", "first_bond_atom", "second_bond_atom", "third_atom"];
    }

    fn samples(&self, keys: &Labels, systems: &mut [System]) -> Result<Vec<Labels>, Error> {
        let mut results = Vec::new();

        for &[species_first, species_second, species_third] in keys.iter_fixed_size() {
            let mut builder = LabelsBuilder::new(
                vec!["structure", "triplet_i", "pair_i", "first_bond_atom", "second_bond_atom", "third_atom"]
            );
            for (system_i, system) in systems.iter_mut().enumerate() {
                let triplets = self.raw_triplets.get_for_system(system, false)?;
                let species = system.species()?;

                
                for triplet in triplets {
                //for (triplet_i, triplet) in system.triplets()?.iter().enumerate() {
                    // filter self-contributions of required
                    debug_assert_ne!(triplet.atom_i, triplet.atom_j); // triplets are ill-defined when their bonds are of twice the same atom
                    if (!self.bond_contribution) && triplet.is_self_contrib {
                        continue
                    }
                    
                    if self.use_half_enumeration {
                        let ((species_i, species_j), invert) = sort_pair((species[triplet.atom_i], species[triplet.atom_j]));
                        let (atom_i, atom_j) = if invert {
                            (triplet.atom_j, triplet.atom_i)
                        } else {
                            (triplet.atom_i, triplet.atom_j)
                        };
                        if species_i == species_first.i32() && species_j == species_second.i32() && species_third.i32() == species[triplet.atom_k] {
                            builder.add(&[system_i, triplet.triplet_i, triplet.bond_i, atom_i, atom_j, triplet.atom_k]);
                        }
                    } else {
                        if species[triplet.atom_i] == species_first.i32() && species[triplet.atom_j] == species_second.i32() && species[triplet.atom_k] == species_third.i32() {
                            builder.add(&[system_i, triplet.triplet_i, triplet.bond_i, triplet.atom_i, triplet.atom_j, triplet.atom_k]);
                            if species_first.i32() == species_second.i32() {
                                builder.add(&[system_i, triplet.triplet_i, triplet.bond_i, triplet.atom_j, triplet.atom_i, triplet.atom_k]);
                            }
                        } else if species[triplet.atom_j] == species_first.i32() && species[triplet.atom_i] == species_second.i32() && species[triplet.atom_k] == species_third.i32() {
                            builder.add(&[system_i, triplet.triplet_i, triplet.bond_i, triplet.atom_j, triplet.atom_i, triplet.atom_k]);
                        }
                    }
                }
            }
            results.push(builder.finish());
        }
        return Ok(results);
    }

    fn supports_gradient(&self, parameter: &str) -> bool {
        // the values of this calculator ARE NOT CONTINUOUS around the values of bond_vector == (0,0,-z)
        // so no gradients for now
        return false;
        match parameter {
            "positions" => true,
            // TODO: add support for cell gradients
            _ => false,
        }
    }

    fn positions_gradient_samples(&self, _keys: &Labels, samples: &[Labels], systems: &mut [System]) -> Result<Vec<Labels>, Error> {
        let mut results = Vec::new();
        assert_ne!(systems.len(),0);  // TODO not sure what the intended behaviour is in this case
        let (mut prev_system_i, mut triplets) = (0_usize,self.raw_triplets.get_for_system(&mut systems[0], false)?);
        for block_samples in samples {
            let mut builder = LabelsBuilder::new(vec!["sample", "structure", "atom"]);
            for (sample_i, &[system_i, triplet_i, _pair_i, first, second, third]) in block_samples.iter_fixed_size().enumerate() {
                if system_i.usize() != prev_system_i {
                    triplets = self.raw_triplets.get_for_system(
                        systems.get_mut(system_i.usize())
                        .ok_or_else(||Error::Internal("system list does not fit sample list".into()))?,
                        false,
                    )?;
                    prev_system_i = system_i.usize();
                }
                let triplet = triplets.get(triplet_i.usize()).ok_or_else(||Error::Internal("inconsistant triplet count".into()))?;
                debug_assert_eq!(first.usize(),triplet.atom_i);
                debug_assert_eq!(second.usize(),triplet.atom_j);
                debug_assert_eq!(third.usize(),triplet.atom_k);

                // self pairs do not contribute to gradients
                if (!self.bond_contribution) && triplet.is_self_contrib {
                    continue
                }
                builder.add(&[sample_i.into(), system_i.usize(), triplet.atom_i]);
                builder.add(&[sample_i.into(), system_i.usize(), triplet.atom_j]);
                if !triplet.is_self_contrib {
                    builder.add(&[sample_i.into(), system_i.usize(), triplet.atom_k]);
                }
            }
            results.push(builder.finish());
        }

        return Ok(results);
    }

    fn components(&self, keys: &Labels) -> Vec<Vec<Labels>> {
        let components = vec![Labels::new(["vector_direction"], &[[0], [1], [2]])];
        return vec![components; keys.count()];
    }

    fn property_names(&self) -> Vec<&str> {
        vec!["distance"]
    }

    fn properties(&self, keys: &Labels) -> Vec<Labels> {
        let mut properties = LabelsBuilder::new(self.property_names());
        properties.add(&[LabelValue::new(1)]);
        let properties = properties.finish();

        return vec![properties; keys.count()];
    }

    #[time_graph::instrument(name = "BANeighborList::compute")]
    fn compute(&mut self, systems: &mut [System], descriptor: &mut TensorMap) -> Result<(), Error> {
        for (system_i, system) in systems.iter_mut().enumerate() {
            let triplets = self.raw_triplets.get_for_system(system, true)?;
            let species = system.species()?;

            let mut mtx_cache:BTreeMap<(usize,bool),Matrix3> = BTreeMap::new();
            let mut dmtx_cache:BTreeMap<(usize,bool),(Matrix3,Matrix3,Matrix3)> = BTreeMap::new();

            
            for triplet in triplets {
            //for (triplet_i, triplet) in system.triplets()?.iter().enumerate() {
                if (!self.bond_contribution) && triplet.is_self_contrib {
                    continue
                }
                
                fn single_triplet_contribution(
                    block: &mut TensorBlockRefMut, triplet: &BATripletInfo,
                    mtx_cache: &mut BTreeMap<(usize, bool), Matrix3>,
                    dmtx_cache: &mut BTreeMap<(usize, bool), (Matrix3, Matrix3, Matrix3)>,
                    system_i:usize, triplet_i:usize,
                    bond_i:usize, atom_i:usize, atom_j:usize, atom_k:usize,
                    invert: bool,
                ) -> Result<(),Error> {
                    let block_sample = {
                        if block.samples().count() * block.properties().count() == 0 {
                            // nothing of interest here
                            None
                        } else {
                            let block_data = block.data_mut();
                            let sample_i = block_data.samples.position(&[
                                system_i.into(), triplet_i.into(), bond_i.into(), atom_i.into(), atom_j.into(), atom_k.into()
                            ]);
                            sample_i
                        } 
                    };
                    if let Some(sample_i) = block_sample {
                        BANeighborList::compute_single_triplet_inplace(
                            triplet, block, sample_i, system_i,
                            invert, mtx_cache, dmtx_cache,
                        )?;
                    }
                    Ok(())
                }

                if self.use_half_enumeration {
                    // Sort the species in the pair to ensure a canonical order of
                    // the atoms in it. This guarantee that multiple call to this
                    // calculator always returns pairs in the same order, even if
                    // the underlying neighbor list implementation (which comes from
                    // the systems) changes.
                    //
                    // The `invert` variable tells us if we need to invert the pair
                    // vector or not.
                    let ((species_i, species_j), invert) = sort_pair((species[triplet.atom_i], species[triplet.atom_j]));

                    let (atom_i, atom_j) = if invert {
                        (triplet.atom_j, triplet.atom_i)
                    } else {
                        (triplet.atom_i, triplet.atom_j)
                    };
    
                    let block_i = descriptor.keys().position(&[
                        species_i.into(), species_j.into(), species[triplet.atom_k].into(),
                    ]).expect("missing block");
                    let mut block = descriptor.block_mut_by_id(block_i);
                    
                    single_triplet_contribution(
                        &mut block, &triplet,
                        &mut mtx_cache, &mut dmtx_cache,
                        system_i, triplet.triplet_i, triplet.bond_i, atom_i, atom_j, triplet.atom_k,
                        invert,
                    )?;
                    
                } else {
                    // first, the pair first -> second (or the ordered pair)
                    let first_block_i = descriptor.keys().position(&[
                        species[triplet.atom_i].into(), species[triplet.atom_j].into(), species[triplet.atom_k].into(),
                    ]).expect("missing block");
                    
                    let mut block = descriptor.block_mut_by_id(first_block_i);
                    single_triplet_contribution(
                        &mut block, &triplet,
                        &mut mtx_cache, &mut dmtx_cache,
                        system_i, triplet.triplet_i, triplet.bond_i, triplet.atom_i,triplet.atom_j,triplet.atom_k,
                        false,
                    )?;
                    
                    // then the pair second -> first
                    let mut block = if species[triplet.atom_i] == species[triplet.atom_j] {
                        block
                    } else {
                        let second_block_i = descriptor.keys().position(&[
                            species[triplet.atom_j].into(), species[triplet.atom_i].into(), species[triplet.atom_k].into(),
                        ]).expect(&std::format!("missing block: {},{},{}  {:?} ", species[triplet.atom_i], species[triplet.atom_j], species[triplet.atom_k], descriptor.keys())[..]);
                        descriptor.block_mut_by_id(second_block_i)
                    };
                    single_triplet_contribution(
                        &mut block, &triplet,
                        &mut mtx_cache, &mut dmtx_cache,
                        system_i, triplet.triplet_i, triplet.bond_i, triplet.atom_j,triplet.atom_i,triplet.atom_k,
                        true,
                    )?;   
                }
            }
        }

        return Ok(());
    }
}


#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use metatensor::Labels;

    use crate::pre_calculators::BATripletNeighborList;
    use crate::systems::test_utils::test_systems;
    use crate::Calculator;
    use super::BANeighborList;
    use super::super::CalculatorBase;

    #[test]
    fn half_neighbor_list() {
        let mut calculator = Calculator::from(Box::new(BANeighborList{
            raw_triplets: BATripletNeighborList{
                cutoffs: [2.0,2.0],
            },
            use_half_enumeration: true,
            bond_contribution: false,
        }) as Box<dyn CalculatorBase>);

        let mut systems = test_systems(&["water"]);

        let descriptor = calculator.compute(&mut systems, Default::default()).unwrap();

        assert_eq!(*descriptor.keys(), Labels::new(
            ["species_first_bond_atom", "species_second_bond_atom", "species_third_atom"],
            &[[-42, 1, 1], [1, 1, -42]]
        ));

        // O-H-H block
        let block = descriptor.block_by_id(0);
        println!("key {:?}", &descriptor.keys()[1]);
        assert_eq!(block.properties(), Labels::new(["distance"], &[[1]]));

        assert_eq!(block.components().len(), 1);
        assert_eq!(block.components()[0], Labels::new(["vector_direction"], &[[0], [1], [2]]));

        assert_eq!(block.samples(), Labels::new(
            ["structure", "triplet_i", "pair_i", "first_bond_atom", "second_bond_atom", "third_atom"],
            // we have two O-H pairs
            &[[0, 2, 0, 0, 1, 2], [0, 5, 1, 0, 2, 1]]
        ));

        let array = block.values().to_array();
        let expected = &ndarray::arr3(&[
            [[0.0], [0.9289563], [-0.7126298]],
            [[0.0], [-0.9289563], [-0.7126298]],
        ]).into_dyn();
        assert_relative_eq!(array, expected, max_relative=1e-6);

        // H-H-O block
        let block = descriptor.block_by_id(1);
        assert_eq!(block.samples(), Labels::new(
            ["structure", "triplet_i", "pair_i", "first_bond_atom", "second_bond_atom", "third_atom"],
            // we have one H-H pair
            &[[0, 8, 2, 1, 2, 0]]
        ));

        let array = block.values().to_array();
        let expected = &ndarray::arr3(&[
            [[0.0], [0.58895], [0.0]]
        ]).into_dyn();
        assert_relative_eq!(array, expected, max_relative=1e-6);
    }

    #[test]
    fn full_neighbor_list() {
        let mut calculator = Calculator::from(Box::new(BANeighborList{
            raw_triplets: BATripletNeighborList{
                cutoffs: [2.0,2.0],
            },
            use_half_enumeration: false,
            bond_contribution: false,
        }) as Box<dyn CalculatorBase>);

        let mut systems = test_systems(&["water"]);

        let descriptor = calculator.compute(&mut systems, Default::default()).unwrap();

        assert_eq!(*descriptor.keys(), Labels::new(
            ["species_first_bond_atom", "species_second_bond_atom", "species_third_atom"],
            &[[-42, 1, 1], [1, -42, 1], [1, 1, -42]]
        ));

        // O-H-H block
        let block = descriptor.block_by_id(0);
        assert_eq!(block.properties(), Labels::new(["distance"], &[[1]]));

        assert_eq!(block.components().len(), 1);
        assert_eq!(block.components()[0], Labels::new(["vector_direction"], &[[0], [1], [2]]));

        assert_eq!(block.samples(), Labels::new(
            ["structure", "triplet_i", "pair_i", "first_bond_atom", "second_bond_atom", "third_atom"],
            // we have two O-H-H triplets
            &[[0, 2, 0, 0, 1, 2], [0, 5, 1, 0, 2, 1]],
        ));

        let array = block.values().to_array();
        let expected = &ndarray::arr3(&[
            [[0.0], [0.9289563], [-0.7126298]],
            [[0.0], [-0.9289563], [-0.7126298]],
        ]).into_dyn();
        assert_relative_eq!(array, expected, max_relative=1e-6);

        // H-O-H block
        let block = descriptor.block_by_id(1);
        assert_eq!(block.properties(), Labels::new(["distance"], &[[1]]));

        assert_eq!(block.components().len(), 1);
        assert_eq!(block.components()[0], Labels::new(["vector_direction"], &[[0], [1], [2]]));

        assert_eq!(block.samples(), Labels::new(
            ["structure", "triplet_i", "pair_i", "first_bond_atom", "second_bond_atom", "third_atom"],
            // we have two H-O-H triplets
            &[[0, 2, 0, 1, 0, 2], [0, 5, 1, 2, 0, 1]],
        ));

        let array = block.values().to_array();
        let expected = &ndarray::arr3(&[
            [[0.0], [-0.9289563], [0.7126298]],
            [[0.0], [0.9289563], [0.7126298]],
        ]).into_dyn();
        assert_relative_eq!(array, expected, max_relative=1e-6);

        // H-H-O block
        let block = descriptor.block_by_id(2);
        assert_eq!(block.samples(), Labels::new(
            ["structure", "triplet_i", "pair_i", "first_bond_atom", "second_bond_atom", "third_atom"],
            // we have one H-H-O pair, twice
            &[[0, 8, 2, 1, 2, 0], [0, 8, 2, 2, 1, 0]],
        ));

        let array = block.values().to_array();
        let expected = &ndarray::arr3(&[
            [[0.0], [0.58895], [0.0]],
            [[0.0], [-0.58895], [0.0]]
        ]).into_dyn();
        assert_relative_eq!(array, expected, max_relative=1e-6);
    }

    // note: the following test does pass, but gradients are disabled because we discovered that
    // the values of this calculator ARE NOT CONTINUOUS around the values of bond_vector == (0,0,-z)
    // ////
    // #[test]
    // fn finite_differences_positions() {
    //     // half neighbor list
    //     let calculator = Calculator::from(Box::new(BANeighborList::Half(HalfBANeighborList{
    //         cutoffs: [2.0,3.0],
    //         bond_contribution: false,
    //     })) as Box<dyn CalculatorBase>);

    //     let system = test_system("water");
    //     let options = crate::calculators::tests_utils::FinalDifferenceOptions {
    //         displacement: 1e-6,
    //         max_relative: 1e-9,
    //         epsilon: 1e-16,
    //     };
    //     crate::calculators::tests_utils::finite_differences_positions(calculator, &system, options);

    //   // full neighbor list
    //     let calculator = Calculator::from(Box::new(BANeighborList::Full(FullBANeighborList{
    //         cutoffs: [2.0,3.0],
    //         bond_contribution: false,
    //     })) as Box<dyn CalculatorBase>);
    //     crate::calculators::tests_utils::finite_differences_positions(calculator, &system, options);
    // }

    #[test]
    fn compute_partial() {
        // half neighbor list
        let calculator = Calculator::from(Box::new(BANeighborList{
            raw_triplets: BATripletNeighborList{
                cutoffs: [1.0,2.0],
            },
            use_half_enumeration: true,
            bond_contribution: false,
        }) as Box<dyn CalculatorBase>);
        let mut systems = test_systems(&["water", "methane"]);

        let samples = Labels::new(
            ["structure", "first_bond_atom"],
            &[[0, 1]],
        );

        let properties = Labels::new(
            ["distance"],
            &[[1]],
        );
        
        let keys = Labels::new(
            ["species_first_bond_atom", "species_second_bond_atom", "species_third_atom"],
            &[[-42, 1, 1], [1, -42, 1], [1, 1, -42], [1, 1, 1], [1, 6, 1], [-42, -42, -42]]  // …only the first one will be valid, whoops
        );

        crate::calculators::tests_utils::compute_partial(
            calculator, &mut systems, &keys, &samples, &properties
        );

        // full neighbor list
        let calculator = Calculator::from(Box::new(BANeighborList{
            raw_triplets: BATripletNeighborList{
                cutoffs: [1.0,3.0],
            },
            use_half_enumeration: false,
            bond_contribution: false,
        }) as Box<dyn CalculatorBase>);
        crate::calculators::tests_utils::compute_partial(
            calculator, &mut systems, &keys, &samples, &properties
        );
    }

    #[test]
    fn check_self_pairs() {
        let mut calculator = Calculator::from(Box::new(BANeighborList{
            raw_triplets: BATripletNeighborList{
                cutoffs: [1.0,2.0],
            },
            use_half_enumeration: false,
            bond_contribution: true,
        }) as Box<dyn CalculatorBase>);
        let mut systems = test_systems(&["water"]);

        let descriptor = calculator.compute(&mut systems, Default::default()).unwrap();

        // we have a block for O-O pairs (-42, -42)
        assert_eq!(descriptor.keys(), &Labels::new(
            ["species_first_bond_atom", "species_second_bond_atom", "species_third_atom"],
            &[[-42, 1, -42], [-42, 1, 1], [1, -42, -42], [1, -42, 1]],
        ));

        // O-H-O block
        let block = descriptor.block_by_id(0);
        let block = block.data();
        assert_eq!(*block.samples, Labels::new(
            ["structure", "triplet_i", "pair_i", "first_bond_atom", "second_bond_atom", "third_atom"],
            // we have two O-H-O self-contributions
            &[[0,0,0,0,1,0], [0,3,1,0,2,0]]
        ));
        // O-H-H block
        let block = descriptor.block_by_id(1);
        let block = block.data();
        assert_eq!(*block.samples, Labels::new(
            ["structure", "triplet_i", "pair_i", "first_bond_atom", "second_bond_atom", "third_atom"],
            // we have two O-H-H self-contributions and two other contributions
            &[[0,1,0,0,1,1], [0,2,0,0,1,2], [0,4,1,0,2,2], [0,5,1,0,2,1]]
       ));
    }
}

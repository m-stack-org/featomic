use std::collections::{BTreeMap, BTreeSet};
use std::collections::btree_map::Entry;
use std::cell::RefCell;

use ndarray::s;
use thread_local::ThreadLocal;
use rayon::prelude::*;

use metatensor::{Labels, LabelsBuilder, LabelValue, TensorMap};

use crate::{Error, System, Vector3D};

use crate::math::SphericalHarmonicsCache;

use super::super::CalculatorBase;
use super::super::bondatom_neighbor_list::BANeighborList;
use super::super::{split_tensor_map_by_system, array_mut_for_system};

use super::{CutoffFunction, RadialScaling};

use crate::calculators::radial_basis::RadialBasis;
use super::SoapRadialIntegralCache;

use super::radial_integral::SoapRadialIntegralParameters;

use super::spherical_expansion_pair::{
    SphericalExpansionParameters,
    SphericalExpansionByPair,
    GradientsOptions,
    PairContribution
};

/// Parameters for spherical expansion calculator for bond-centered neighbor densities.
///
/// (The spherical expansion is at the core of representations in the SOAP
/// (Smooth Overlap of Atomic Positions) family. See [this review
/// article](https://doi.org/10.1063/1.5090481) for more information on the SOAP
/// representation, and [this paper](https://doi.org/10.1063/5.0044689) for
/// information on how it is implemented in rascaline.)
///
/// This calculator is only needed to characterize local environments that are centered
/// on a pair of atoms rather than a single one.
#[derive(Debug, Clone)]
#[derive(serde::Deserialize, serde::Serialize, schemars::JsonSchema)]
pub struct SphericalExpansionForBondsParameters {
    /// Spherical cutoffs to use for atomic environments
    pub(super) cutoffs: [f64;2],
    /// Number of radial basis function to use in the expansion
    pub max_radial: usize,
    /// Number of spherical harmonics to use in the expansion
    pub max_angular: usize,
    /// Width of the atom-centered gaussian used to create the atomic density
    pub atomic_gaussian_width: f64,
    /// Weight of the central atom contribution to the
    /// features. If `1` the center atom contribution is weighted the same
    /// as any other contribution. If `0` the central atom does not
    /// contribute to the features at all.
    pub center_atoms_weight: f64,
    /// Radial basis to use for the radial integral
    pub radial_basis: RadialBasis,
    /// Cutoff function used to smooth the behavior around the cutoff radius
    pub cutoff_function: CutoffFunction,
    /// radial scaling can be used to reduce the importance of neighbor atoms
    /// further away from the center, usually improving the performance of the
    /// model
    #[serde(default)]
    pub radial_scaling: RadialScaling,
}

impl SphericalExpansionForBondsParameters {
    /// Validate all the parameters
    pub fn validate(&self) -> Result<(), Error> {
        self.cutoff_function.validate()?;
        self.radial_scaling.validate()?;

        // try constructing a radial integral
        SoapRadialIntegralCache::new(self.radial_basis.clone(), SoapRadialIntegralParameters {
            max_radial: self.max_radial,
            max_angular: self.max_angular,
            atomic_gaussian_width: self.atomic_gaussian_width,
            cutoff: self.third_cutoff(),
        })?;

        return Ok(());
    }
    pub fn bond_cutoff(&self) -> f64 {
        self.cutoffs[0]
    }
    pub fn third_cutoff(&self) -> f64 {
        self.cutoffs[1]
    }
}

impl Into<SphericalExpansionParameters> for SphericalExpansionForBondsParameters {
    fn into(self) -> SphericalExpansionParameters{
        SphericalExpansionParameters{
            cutoff: self.third_cutoff(),
            max_radial: self.max_radial,
            max_angular: self.max_angular,
            atomic_gaussian_width: self.atomic_gaussian_width,
            center_atom_weight: self.center_atoms_weight,
            radial_basis: self.radial_basis,
            cutoff_function: self.cutoff_function,
            radial_scaling: self.radial_scaling,
        }
    }
}

/// The actual calculator used to compute spherical expansion pair-by-pair
pub struct SphericalExpansionForBondType {
    pub(crate) parameters: SphericalExpansionForBondsParameters,
    /// several functions require the SphericalExpansionForBonds to behave like a regular spherical expansion
    /// let's store most of the data in an actual SphericalExpansionByPair object!
    faker: SphericalExpansionByPair,
    pub(super) distance_calculator: BANeighborList,
    
}

impl std::fmt::Debug for SphericalExpansionForBondType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.parameters)
    }
}


impl SphericalExpansionForBondType {
    pub fn new(parameters: SphericalExpansionForBondsParameters) -> Result<Self, Error> {
        parameters.validate()?;

        let m_1_pow_l = (0..=parameters.max_angular)
            .map(|l| f64::powi(-1.0, l as i32))
            .collect::<Vec<f64>>();

        Ok(SphericalExpansionForBondType {
            faker: SphericalExpansionByPair{
                parameters: parameters.clone().into(),
                radial_integral: ThreadLocal::new(),
                spherical_harmonics: ThreadLocal::new(),
                m_1_pow_l,
            },
            distance_calculator: BANeighborList::from_params(
                parameters.cutoffs,
                false,
                true,
            ),
            parameters,
        })
    }

    /// Access the spherical expansion parameters used by this calculator
    pub fn parameters(&self) -> &SphericalExpansionForBondsParameters {
        &self.parameters
    }

    /// Compute the product of radial scaling & cutoff smoothing functions
    fn scaling_functions(&self, r: f64) -> f64 {
        self.faker.scaling_functions(r)
    }

    /// Compute the gradient of the product of radial scaling & cutoff smoothing functions
    fn scaling_functions_gradient(&self, r: f64) -> f64 {
        self.faker.scaling_functions_gradient(r)
    }

    /// Compute the self-contribution (contribution coming from an atom "seeing"
    /// it's own density). This is equivalent to a normal pair contribution,
    /// with a distance of 0.
    ///
    /// For now, the same density is used for all atoms, so this function can be
    /// called only once and re-used for all atoms (see `do_self_contributions`
    /// below).
    ///
    /// By symmetry, the self-contribution is only non-zero for `L=0`, and does
    /// not contributes to the gradients.
    pub(super) fn compute_coefficients(&self, contribution: &mut PairContribution, vector: Vector3D, is_self_contribution: bool, gradients: Option<(Vector3D,Vector3D,Vector3D)>){
        let mut radial_integral = self.faker.radial_integral.get_or(|| {
            let radial_integral = SoapRadialIntegralCache::new(
                self.parameters.radial_basis.clone(),
                SoapRadialIntegralParameters {
                    max_radial: self.parameters.max_radial,
                    max_angular: self.parameters.max_angular,
                    atomic_gaussian_width: self.parameters.atomic_gaussian_width,
                    cutoff: self.parameters.third_cutoff(),
                }
            ).expect("invalid radial integral parameters");
            return RefCell::new(radial_integral);
        }).borrow_mut();

        let mut spherical_harmonics = self.faker.spherical_harmonics.get_or(|| {
            RefCell::new(SphericalHarmonicsCache::new(self.parameters.max_angular))
        }).borrow_mut();

        let distance = vector.norm();
        let direction = vector/distance;
        // Compute the three factors that appear in the center contribution.
        // Note that this is simply the pair contribution for the special
        // case where the pair distance is zero.
        radial_integral.compute(distance, gradients.is_some());
        spherical_harmonics.compute(direction, gradients.is_some());

        let f_scaling = self.scaling_functions(distance);
        let f_scaling = if is_self_contribution{
            f_scaling * self.parameters.center_atoms_weight
        } else {
            f_scaling
        };

        let (values, gradient_values_o) = (&mut contribution.values, contribution.gradients.as_mut());
        
        debug_assert_eq!(
            values.shape(),
            [(self.parameters.max_angular+1)*(self.parameters.max_angular+1), self.parameters.max_radial]
        );
        for l in 0..=self.parameters.max_angular {
            let l_offset = l*l;
            let msize = 2*l+1;
            //values.slice_mut(s![l_offset..l_offset+msize, ..]) *= spherical_harmonics.values.slice(l);
            for m in 0..msize {
                let lm = l_offset+m;
                for n in 0..self.parameters.max_radial {
                    values[[lm, n]] = spherical_harmonics.values[lm]
                                     * radial_integral.values[[l,n]]
                                     * f_scaling;
                }
            }
        }
        
        if let Some((dvdx,dvdy,dvdz)) = gradients {
            let gradient_values = gradient_values_o.unwrap();
            
            let ilen = 1./distance;
            let dlendv = vector*ilen;
            let dlendx = dlendv*dvdx;
            let dlendy = dlendv*dvdy;
            let dlendz = dlendv*dvdz;
            let ddirdx = dvdx*ilen - vector*dlendx*ilen*ilen;
            let ddirdy = dvdy*ilen - vector*dlendy*ilen*ilen;
            let ddirdz = dvdy*ilen - vector*dlendz*ilen*ilen;
            
            let single_grad = |l,n,m,dlenda,ddirda: Vector3D| {
                f_scaling * (
                    radial_integral.gradients[[l,n]]*dlenda*spherical_harmonics.values[[l as isize,m as isize]]
                    + radial_integral.values[[l,n]]*(
                        spherical_harmonics.gradients[0][[l as isize,m as isize]]*ddirda[0]
                       +spherical_harmonics.gradients[1][[l as isize,m as isize]]*ddirda[1]
                       +spherical_harmonics.gradients[2][[l as isize,m as isize]]*ddirda[2]
                    )
                    // todo scaling_function_gradient
                )
            };
            
            for l in 0..=self.parameters.max_angular {
                let l_offset = l*l;
                let msize = 2*l+1;
                for m in 0..(msize) {
                    let lm = l_offset+m;
                    for n in 0..self.parameters.max_radial {
                        gradient_values[[0,lm,n]] = single_grad(l,n,m,dlendx,ddirdx);
                        gradient_values[[1,lm,n]] = single_grad(l,n,m,dlendy,ddirdy);
                        gradient_values[[2,lm,n]] = single_grad(l,n,m,dlendz,ddirdz);
                    }
                }
            }
        }
    }

    /// a smart-ish way to obtain the coefficients of all bond expansions:
    /// this function's API is designed to be resource-efficient for both SphericalExpansionForBondType and
    /// SphericalExpansionForBonds, while being computationally efficient for the underlying BANeighborList calculator.
    pub(super) fn get_coefficients_for<'a>(
        &'a self, system: &'a System,
        s1: i32, s2: i32, s3_list: &'a Vec<i32>,
        do_gradients: GradientsOptions,
    ) -> Result<impl Iterator<Item = (usize, bool, std::rc::Rc<RefCell<PairContribution>>)> + 'a, Error> {
        
        let max_angular = self.parameters.max_angular;
        let max_radial = self.parameters.max_radial;
        let species = system.species().unwrap();
        
        
        let pre_iter = s3_list.iter().flat_map(|s3|{
            self.distance_calculator.raw_triplets.get_per_system_per_species(system,s1,s2,*s3,true).unwrap().into_iter()
        }).flat_map(|triplet| {
            let invert: &'static [bool] = {
                if s1==s2 {&[false,true]}
                else if species[triplet.atom_i] == s1 {&[false]}
                else {&[true]}
            };
            invert.iter().map(move |invert|(triplet,*invert))
        }).collect::<Vec<_>>();
        
        let contribution = std::rc::Rc::new(RefCell::new(
            PairContribution::new(max_radial, max_angular, do_gradients.either())
        ));

        let mut mtx_cache = BTreeMap::new();
        let mut dmtx_cache = BTreeMap::new();
        
        return Ok(pre_iter.into_iter().map(move |(triplet,invert)| {
            let vector = BANeighborList::compute_single_triplet(&triplet, invert, false, &mut mtx_cache, &mut dmtx_cache).unwrap();
            self.compute_coefficients(&mut *contribution.borrow_mut(), vector.vect,triplet.is_self_contrib,None);
            (triplet.triplet_i, invert, contribution.clone())
        }));

    }
}


impl CalculatorBase for SphericalExpansionForBondType {
    fn name(&self) -> String {
        "spherical expansion by pair".into()
    }
    
    fn cutoffs(&self) -> &[f64] {
        &self.parameters.cutoffs
    }

    fn parameters(&self) -> String {
        serde_json::to_string(&self.parameters).expect("failed to serialize to JSON")
    }

    fn keys(&self, systems: &mut [System]) -> Result<Labels, Error> {
        // the species part of the keys is the same for all l
        let species_keys = self.distance_calculator.keys(systems)?;

        let all_species_triplets = species_keys.iter().map(|p| (p[0], p[1], p[2])).collect::<BTreeSet<_>>();

        let mut keys = LabelsBuilder::new(vec![
            "spherical_harmonics_l",
            "species_bond_atom_1",
            "species_bond_atom_2",
            "species_third_atom",
        ]);

        for (s1, s2, s3) in all_species_triplets {
            for l in 0..=self.parameters.max_angular {
                keys.add(&[l.into(), s1, s2, s3]);
            }
        }


        return Ok(keys.finish());
    }

    fn sample_names(&self) -> Vec<&str> {
        return vec!["structure", "triplet_i", "bond_i", "first_bond_atom", "second_bond_atom", "third_atom"];
    }

    fn samples(&self, keys: &Labels, systems: &mut [System]) -> Result<Vec<Labels>, Error> {
        let newkey_values = keys.iter_fixed_size().map(|&[_l, s1,s2,s3]|{ [s1.i32(),s2.i32(),s3.i32()] }).collect::<Vec<_>>();
        let mut newkeys_lut = vec![0_usize;newkey_values.len()];
        let new_unique_keys: Vec<[i32;3]> = BTreeSet::from_iter(newkey_values.iter()).into_iter().map(|t|t.clone()).collect();  // note: this step sorts the  keys
        for (key,f_index) in newkey_values.iter().zip(newkeys_lut.iter_mut()) {
            *f_index = new_unique_keys.binary_search(key).expect("unreachable: new_unique_keys was constructed with all keys");
        }
        
        let newkeys = Labels::new(
                ["species_first_bond_atom", "species_second_bond_atom", "species_third_atom"],
                &new_unique_keys,
        );
        let samples = self.distance_calculator.samples(&newkeys, systems)?;
        let ret = Ok(newkeys_lut.into_iter().map(|i|samples[i].clone()).collect::<Vec<_>>());
        ret
    }

    fn supports_gradient(&self, parameter: &str) -> bool {
        return false;
        self.distance_calculator.supports_gradient(parameter)
    }

    fn positions_gradient_samples(&self, _keys: &Labels, samples: &[Labels], systems: &mut [System]) -> Result<Vec<Labels>, Error> {
        self.distance_calculator.positions_gradient_samples(_keys, samples, systems)
    }

    fn components(&self, keys: &Labels) -> Vec<Vec<Labels>> {
        assert_eq!(keys.names().len(), 4);
        assert_eq!(keys.names()[0], "spherical_harmonics_l");

        let mut result = Vec::new();
        // only compute the components once for each `spherical_harmonics_l`,
        // and re-use the results across the other keys.
        let mut cache: BTreeMap<_, Vec<Labels>> = BTreeMap::new();
        for &[spherical_harmonics_l, _, _, _] in keys.iter_fixed_size() {
            let components = match cache.entry(spherical_harmonics_l) {
                Entry::Occupied(entry) => entry.get().clone(),
                Entry::Vacant(entry) => {
                    let mut component = LabelsBuilder::new(vec!["spherical_harmonics_m"]);
                    for m in -spherical_harmonics_l.i32()..=spherical_harmonics_l.i32() {
                        component.add(&[LabelValue::new(m)]);
                    }

                    let components = vec![component.finish()];
                    entry.insert(components).clone()
                }
            };

            result.push(components);
        }

        return result;
    }

    fn property_names(&self) -> Vec<&str> {
        vec!["n"]
    }

    fn properties(&self, keys: &Labels) -> Vec<Labels> {
        let mut properties = LabelsBuilder::new(self.property_names());
        for n in 0..self.parameters.max_radial {
            properties.add(&[n]);
        }

        return vec![properties.finish(); keys.count()];
    }

    #[time_graph::instrument(name = "SphericalExpansionByBondType::compute")]
    fn compute(&mut self, systems: &mut [System], descriptor: &mut TensorMap) -> Result<(), Error> {
        assert_eq!(descriptor.keys().names(), ["spherical_harmonics_l", "species_bond_atom_1", "species_bond_atom_2", "species_third_atom"]);

        let max_angular = self.parameters.max_angular;
        let l_slices: Vec<_> = (0..=max_angular).map(|l|{
            let lsize = l*l;
            let msize = 2*l+1;
            lsize..lsize+msize
        }).collect();
        
        let do_gradients = GradientsOptions {
            positions: descriptor.block_by_id(0).gradient("positions").is_some(),
            cell: descriptor.block_by_id(0).gradient("cell").is_some(),
        };
        if do_gradients.positions {
            assert!(self.distance_calculator.supports_gradient("positions"));
        }
        if do_gradients.cell {
            assert!(self.distance_calculator.supports_gradient("cell"));
        }

        // first, create some partial-key -> block lookup tables to avoid linear searches within blocks later
        
        // {(s1,s2,s3) -> i_s3}
        let mut s1s2s3_to_is3: BTreeMap<(i32,i32,i32),usize> = BTreeMap::new();
        // {(s1,s2) -> [i_s3->(s3,[l->i_block])]}
        let mut s1s2_to_block_ids: BTreeMap<(i32,i32),Vec<(i32,Vec<usize>)>> = BTreeMap::new();
        
        for (block_i, &[l, s1,s2,s3]) in descriptor.keys().iter_fixed_size().enumerate(){
            let s1=s1.i32();
            let s2=s2.i32();
            let s3=s3.i32();
            let l=l.usize();
            let s1s2_blocks = s1s2_to_block_ids.entry((s1,s2))
                .or_insert_with(Vec::new);
            let l_blocks = match s1s2s3_to_is3.entry((s1,s2,s3)) {
                Entry::Occupied(i_s3_e) => {
                    let (s3_b,l_blocks) = &mut s1s2_blocks[*i_s3_e.get()];
                    debug_assert_eq!(s3_b, &s3);
                    l_blocks
                },
                Entry::Vacant(i_s3_e) => {
                    let i_s3 = s1s2_blocks.len();
                    i_s3_e.insert(i_s3);
                    s1s2_blocks.push((s3,vec![usize::MAX;max_angular+1]));
                    &mut s1s2_blocks[i_s3].1
                },
            };
            l_blocks[l] = block_i;
        }
        
        
        #[cfg(debug_assertions)]{
            // half-assume that blocks that share s1,s2,s3 have the same sample list
            for (_,s1s2_blocks) in s1s2_to_block_ids.iter() {
                for s3blocks in s1s2_blocks.iter().map(|t|&t.1) {
                    debug_assert!(s3blocks.len()>0);
                    let len = descriptor.block_by_id(s3blocks[0]).samples().size();
                    for lblock in s3blocks {
                        if lblock != &usize::MAX{
                            debug_assert_eq!(descriptor.block_by_id(*lblock).samples().size(), len);
                        }
                    }
                }
            }
        }
        
        let mut descriptors_by_system = split_tensor_map_by_system(descriptor, systems.len());
        
        systems.par_iter_mut().zip_eq(&mut descriptors_by_system)
            .try_for_each(|(system,descriptor)|{
        
            self.distance_calculator.raw_triplets.ensure_computed_for_system(system)?;
            let triplets = self.distance_calculator.raw_triplets.get_for_system(system, false)?;
            // then, for every of those partial-keys construct a similar lookup table that helps select
            // the right blocks and samples for the given compound and species.
            for ((s1,s2),s1s2_blocks) in s1s2_to_block_ids.iter() {
                let s3_list: Vec<i32> = s1s2_blocks.iter().map(|t|t.0).collect();
            
                // {(triplet_i,inverted)->(i_s3,sample_i)}
                let mut sample_lut: BTreeMap<(usize,bool),Vec<(usize,usize)>> = BTreeMap::new();
                #[cfg(debug_assertions)]let mut s3_samples = vec![];
            
                // also assume that the systems are in order in the samples
                for (i_s3,s3blocks) in s1s2_blocks.iter().map(|t|&t.1).enumerate() {
                    let good_block_i = s3blocks.iter().filter(|b_i|**b_i!=usize::MAX).next().unwrap();
                
                    let samples = descriptor.block_by_id(*good_block_i).samples();
                    #[cfg(debug_assertions)]{s3_samples.push(samples.clone());}
                    for (sample_i, &[_system_i, triplet_i, _bond_i, atom_1,atom_2,_atom_3]) in samples.iter_fixed_size().enumerate(){
                        let (triplet_i, atom_1, atom_2) = (triplet_i.usize(),atom_1.usize(),atom_2.usize());
                        let triplet = triplets[triplet_i];
                        if atom_1 != atom_2 && atom_1 == triplet.atom_i{
                            // simple case 1: we know the triplet is uninverted in the sample
                            sample_lut.entry((triplet_i, false))
                            .or_insert_with(||vec![]).push((i_s3,sample_i));
                        } else if atom_1 != atom_2 && atom_2 == triplet.atom_i {
                            // simple case 2: we know the triplet is inverted in the sample
                            sample_lut.entry((triplet_i, true))
                            .or_insert_with(||vec![]).push((i_s3,sample_i));
                        } else if atom_1 == atom_2 && atom_1 == triplet.atom_i {
                            // complex case: bond's atoms are images of each other:
                            // we probably already crashed because of duped samples. oh well.
                            unimplemented!("I'm surpised we haven't crashed earlier");
                        } else {unreachable!();}
                    }
                }
            
                if sample_lut.len() == 0 {
                    continue  // sometimes someone would specify extra samples which have no underlying data… welp.
                }
                //let system = &mut **system;
                //system.compute_triplet_neighbors(self.parameters.bond_cutoff(), self.parameters.third_cutoff())?;
                for (i_s3,sample_i,_triplet_i,contribution) in self.get_coefficients_for(system, *s1, *s2, &s3_list, do_gradients)?
                    .filter_map(|(triplet_i, inverted,contribution)|
                        sample_lut.get(&(triplet_i,inverted))
                        .map(|lutvec|(lutvec,triplet_i,contribution))
                    ).flat_map(|(lutvec,triplet_i,contribution)|
                        lutvec.into_iter().map(move |(i_s3,sample_i)|((i_s3,sample_i,triplet_i,contribution.clone())))
                    ){

                    let ret_blocks = &s1s2_blocks[*i_s3].1;
                    let contribution = contribution.borrow();
                    for (l,lslice) in l_slices.iter().enumerate() {
                        if ret_blocks[l] == usize::MAX{
                            continue;  // sometimes the key for that l,s1,s2,s3 combination was not provided
                        }
                        let mut block = descriptor.block_mut_by_id(ret_blocks[l]);
                        #[cfg(debug_assertions)]{
                            let samples_j =s3_samples.get(*i_s3).unwrap();
                            debug_assert_eq!(&block.samples(), samples_j);
                        }
                        let n_subset = block.properties();
                        let mut values = array_mut_for_system(block.values_mut());
                        for (i_n,&[n]) in n_subset.iter_fixed_size().enumerate() {
                            let mut value_slice = values.slice_mut(s![*sample_i,..,i_n]);
                            value_slice.assign(&contribution.values.slice(s![lslice.clone(),n.usize()]));
                        }
                    }
                }
            }
            Result::<(), Error>::Ok(())
        })?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use metatensor::Labels;
    use ndarray::{s, Axis};
    use approx::assert_ulps_eq;

    use crate::systems::test_utils::test_systems;
    use crate::Calculator;
    use crate::calculators::{CalculatorBase, SphericalExpansionForBonds};

    use super::{SphericalExpansionForBondType, SphericalExpansionForBondsParameters};
    use super::super::{CutoffFunction, RadialScaling};
    use crate::calculators::radial_basis::RadialBasis;


    fn parameters() -> SphericalExpansionForBondsParameters{
        SphericalExpansionForBondsParameters{
            cutoffs: [3.5,3.5],
            max_radial: 6,
            max_angular: 6,
            atomic_gaussian_width: 0.3,
            center_atoms_weight: 10.0,
            radial_basis: RadialBasis::splined_gto(1e-8),
            radial_scaling: RadialScaling::Willatt2018 { scale: 1.5, rate: 0.8, exponent: 2.0},
            cutoff_function: CutoffFunction::ShiftedCosine { width: 0.5 },
        }
    }

    // #[test]
    // fn finite_differences_positions() {
    //     let calculator = Calculator::from(Box::new(SphericalExpansionForBondType::new(
    //         parameters()
    //     ).unwrap()) as Box<dyn CalculatorBase>);

    //     let system = test_system("water");
    //     let options = crate::calculators::tests_utils::FinalDifferenceOptions {
    //         displacement: 1e-6,
    //         max_relative: 1e-5,
    //         epsilon: 1e-16,
    //     };
    //     crate::calculators::tests_utils::finite_differences_positions(calculator, &system, options);
    // }

    // #[test]
    // fn finite_differences_cell() {
    //     let calculator = Calculator::from(Box::new(SphericalExpansionForBondType::new(
    //         parameters()
    //     ).unwrap()) as Box<dyn CalculatorBase>);

    //     let system = test_system("water");
    //     let options = crate::calculators::tests_utils::FinalDifferenceOptions {
    //         displacement: 1e-6,
    //         max_relative: 1e-5,
    //         epsilon: 1e-16,
    //     };
    //     crate::calculators::tests_utils::finite_differences_cell(calculator, &system, options);
    // }

    #[test]
    fn compute_partial() {
        let calculator = Calculator::from(Box::new(SphericalExpansionForBondType::new(
            SphericalExpansionForBondsParameters {
                max_angular: 2,
                ..parameters()
            }
        ).unwrap()) as Box<dyn CalculatorBase>);

        let mut systems = test_systems(&["water"]);

        let properties = Labels::new(["n"], &[
            [0],
            [3],
            [2],
        ]);

        let samples = Labels::new([
            "structure",
            "triplet_i",
            "bond_i",
            "first_bond_atom",
            "second_bond_atom",
            "third_atom"
        ], &[
            //[0, 1, 2],
            //[0, 2, 1],
            [0, 7, 2, 1, 2, 1],
            [0, 7, 2, 2, 1, 1],
            [0, 8, 2, 1, 2, 2],
            [0, 8, 2, 2, 1, 2],
            [0, 5, 1, 0, 2, 2],
            
        ]);

        let keys = Labels::new([
            "spherical_harmonics_l",
            "species_bond_atom_1",
            "species_bond_atom_2",
            "species_third_atom",
        ], &[
            [0, -42, 1, -42],
            [0, -42, 1, 1],
            [0, 1, -42, -42],
            [0, 1, -42, 1],
            [0, 1, 1, 1],
            [0, 1, 1, -42],
            [0, 6, 1, 1], // not part of the default keys
            [1, -42, 1, -42],
            [1, -42, 1, 1],
            [1, 1, -42, -42],
            [1, 1, -42, 1],
            [1, 1, 1, 1],
            [1, 1, 1, -42],
            [2, -42, 1, -42],
            [2, -42, 1, 1],
            [2, 1, -42, -42],
            [2, 1, -42, 1],
            [2, 1, 1, 1],
            [2, 1, 1, -42],
        ]);

        crate::calculators::tests_utils::compute_partial(
            calculator, &mut systems, &keys, &samples, &properties
        );
    }

    #[test]
    fn sums_to_spherical_expansion() {
        let mut calculator_by_pair = Calculator::from(Box::new(SphericalExpansionForBondType::new(
            parameters()
        ).unwrap()) as Box<dyn CalculatorBase>);
        let mut calculator = Calculator::from(Box::new(SphericalExpansionForBonds::new(
            parameters()
        ).unwrap()) as Box<dyn CalculatorBase>);

        let mut systems = test_systems(&["water", "methane"]);

        let expected = calculator.compute(&mut systems, Default::default()).unwrap();

        
        let by_pair = calculator_by_pair.compute(&mut systems, Default::default()).unwrap();

        // check that keys are the same appart for the names
        assert_eq!(expected.keys().count(), by_pair.keys().count());//, "wrong key count: {} vs {}", expected.keys().count(), by_pair.keys().count());
        assert_eq!(
            expected.keys().iter().collect::<Vec<_>>(),
            by_pair.keys().iter().collect::<Vec<_>>(),
        );

        for (_bl_i,(block, spx)) in by_pair.blocks().iter().zip(expected.blocks()).enumerate() {
            let spx = spx.data();
            let spx_values = spx.values.as_array();

            let block = block.data();
            let values = block.values.as_array();

            for (&[spx_structure, spx_center1,spx_center2, spx_bond_i], expected) in spx.samples.iter_fixed_size().zip(spx_values.axis_iter(Axis(0))) {
                let mut sum = ndarray::Array::zeros(expected.raw_dim());

                for (sample_i, &[structure, _triplet_i, bond_i, center1, center2, _atom3]) in block.samples.iter_fixed_size().enumerate() {
                    if spx_structure == structure && spx_bond_i == bond_i && spx_center1 == center1 && spx_center2 == center2 {
                        sum += &values.slice(s![sample_i, .., ..]);
                    }
                }

                assert_ulps_eq!(sum, expected);
            }
        }
    }
}

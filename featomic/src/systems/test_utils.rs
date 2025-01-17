use crate::{System, Vector3D};
use super::{UnitCell, SimpleSystem};

pub fn test_systems(names: &[&str]) -> Vec<System> {
    return names.iter()
        .map(|&name| System::new(test_system(name)))
        .collect();
}

pub fn test_system(name: &str) -> SimpleSystem {
    match name {
        "methane" => get_methane(),
        "ethanol" => get_ethanol(),
        "water" => get_water(),
        "CH" => get_ch(),
        _ => panic!("unknown test system {}", name)
    }
}

fn get_methane() -> SimpleSystem {
    let mut system = SimpleSystem::new(UnitCell::cubic(5.0));
    system.add_atom(6, Vector3D::new(5.0000, 5.0000, 5.0000));
    system.add_atom(1, Vector3D::new(5.5288, 5.1610, 5.9359));
    system.add_atom(1, Vector3D::new(5.2051, 5.8240, 4.3214));
    system.add_atom(1, Vector3D::new(5.3345, 4.0686, 4.5504));
    system.add_atom(1, Vector3D::new(3.9315, 4.9463, 5.1921));
    return system;
}

fn get_water() -> SimpleSystem {
    let mut system = SimpleSystem::new(UnitCell::cubic(3.0));
    // atomic types do not have to be atomic number
    system.add_atom(-42, Vector3D::new(0.0, 0.0, 0.0));
    system.add_atom(1, Vector3D::new(0.0, 0.75545, -0.58895));
    system.add_atom(1, Vector3D::new(0.0, -0.75545, -0.58895));
    return system;
}

fn get_ch() -> SimpleSystem {
    let mut system = SimpleSystem::new(UnitCell::cubic(10.0));
    system.add_atom(6, Vector3D::new(0.0, 0.0, 0.0));
    system.add_atom(1, Vector3D::new(0.0, 1.2, 0.0));
    return system;
}

fn get_ethanol() -> SimpleSystem {
    let mut system = SimpleSystem::new(UnitCell::cubic(5.0));
    system.add_atom(1, Vector3D::new(3.8853, 1.9599, 3.0854));
    system.add_atom(6, Vector3D::new(3.2699, 1.9523, 2.1772));
    system.add_atom(1, Vector3D::new(3.5840, 2.8007, 1.5551));
    system.add_atom(1, Vector3D::new(3.5089, 1.0364, 1.6209));
    system.add_atom(6, Vector3D::new(1.7967, 2.0282, 2.5345));
    system.add_atom(1, Vector3D::new(1.5007, 1.1713, 3.1714));
    system.add_atom(1, Vector3D::new(1.5765, 2.9513, 3.1064));
    system.add_atom(8, Vector3D::new(1.0606, 2.0157, 1.3326));
    system.add_atom(1, Vector3D::new(0.1460, 2.0626, 1.5748));
    return system;
}

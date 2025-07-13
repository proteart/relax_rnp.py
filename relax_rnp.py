import math
import random
import numpy as np
from pyrosetta import rosetta
from pyrosetta import init
from pyrosetta import pose_from_pdb
from pyrosetta.rosetta.core.pack.task import TaskFactory
from pyrosetta.rosetta.core.pack.task.operation import RestrictToRepacking
from pyrosetta.rosetta.core.pack.task.operation import IncludeCurrent
from pyrosetta.rosetta.core.scoring import ScoreFunctionFactory
from pyrosetta.rosetta.core.scoring import score_type_from_name
from pyrosetta.rosetta.core.scoring.func import HarmonicFunc
from pyrosetta.rosetta.core.scoring.func import CircularHarmonicFunc
from pyrosetta.rosetta.core.scoring.constraints import AtomPairConstraint
from pyrosetta.rosetta.core.scoring.constraints import DihedralConstraint
from pyrosetta.rosetta.core.scoring.constraints import AngleConstraint
from pyrosetta.rosetta.core.pose import addVirtualResAsRoot
from pyrosetta.rosetta.core.id import AtomID
from pyrosetta.rosetta.numeric import xyzVector_double_t
from pyrosetta.rosetta.core.kinematics import MoveMap
from pyrosetta.rosetta.protocols.minimization_packing import MinMover
from pyrosetta.rosetta.protocols.minimization_packing import PackRotamersMover

# ----------------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #

print(" ")
print("Initializing PyRosetta...")

print(" ")
init("-ignore_unrecognized_res -ex1 -ex2 -ex2aro -ex3 -ex4 -extrachi_cutoff 0 -use_input_sc")

print(" ")
print("PyRosetta initialized.")

def windows_to_wsl_path(path):
    path = path.strip('"\'')
    if len(path) >= 3 and path[1] == ':' and path[2] == '\\':
        drive = path[0].lower()
        path = f"/mnt/{drive}" + path[2:]
        path = path.replace('\\', '/')
    return path

input_pdb  = windows_to_wsl_path(r"path_to_your_input_.pdb")
output_pdb = windows_to_wsl_path(r"path_to_your_output_directory")

print(" ")
pose = pose_from_pdb(input_pdb)
print(" ")
print(f"Loaded pose from: {input_pdb}")

addVirtualResAsRoot(pose)
anchor_res = pose.total_residue()
anchor_atom_id = AtomID(1, anchor_res)

print(" ")
scorefxn_cart = ScoreFunctionFactory.create_score_function("ref2015_cart")
fa_atr_init                = 1.0  
fa_rep_init                = 0.55
fa_sol_init                = 1.0  
fa_intra_rep_init          = 0.005
fa_intra_sol_xover4_init   = 1.0
lk_ball_wtd_init           = 1.0
fa_elec_init               = 1.0 
hbond_sr_bb_init           = 1.0
hbond_lr_bb_init           = 1.0
hbond_bb_sc_init           = 1.0 
hbond_sc_init              = 1.0  
dslf_fa13_init             = 1.25
omega_init                 = 0.4
fa_dun_init                = 0.7
p_aa_pp_init               = 0.6
yhh_planarity_init         = 0.625
ref_init                   = 1.0
rama_prepro_init           = 0.45
cart_bonded_init           = 0.5
rna_torsion_init           = 1.0
rna_sugar_close_init       = 1.0
dna_bb_torsion_init        = 1.0
dna_sugar_close_init       = 1.0
fa_stack_init              = 1.0
dihedral_init              = 1.0
atom_pair_init             = 1.0
print(" ")
print("Score function configured")

tf = TaskFactory()
tf.push_back(RestrictToRepacking())
tf.push_back(IncludeCurrent())
packer_task = tf.create_task_and_apply_taskoperations(pose)
print(" ")
print("TaskFactory configured.")

n_chains = pose.num_chains()
chain_starts = [pose.chain_begin(i+1) for i in range(n_chains)]
chain_ends = [pose.chain_end(i+1) for i in range(n_chains)]
virtual_root = pose.total_residue() 

print(" ")
print(f"Chains detected: {n_chains}")
for i in range(n_chains):
    chain_id = pose.pdb_info().chain(chain_starts[i])
    print(f"Chain {chain_id}: {chain_starts[i]}-{chain_ends[i]}")

print(" ")
print(f"Virtual root residue: {virtual_root}")

ft = rosetta.core.kinematics.FoldTree()
for i in range(n_chains):
    ft.add_edge(virtual_root, chain_starts[i], i+1)
for i in range(n_chains):
    ft.add_edge(chain_starts[i], chain_ends[i], -1)

if ft.check_fold_tree():
    pose.fold_tree(ft)
    print(" ")
    print("Fold tree set and valid.")
else:
    print("Fold tree invalid!")

print(" ")

print("Jump setup:")
for i in range(n_chains):
    chain_id = pose.pdb_info().chain(chain_starts[i])
    print(f"Jump {i+1}: virtual root -> chain {chain_id} ({chain_starts[i]})")
print(" ")
print("FoldTree configured.")

# ----------------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #

def get_interface_residues(pose, chain_indices, cutoff=5.0):

    """Returns a set of residue indices in the given chains that are within cutoff Å of any residue in another chain."""

    interface_res = set()
    nres = pose.total_residue()
    chainA = pose.pdb_info().chain(chain_indices[0])
    chainB = pose.pdb_info().chain(chain_indices[1])
    groupA = [idx for idx in range(1, nres + 1) if pose.pdb_info().chain(idx) == chainA]
    groupB = [idx for idx in range(1, nres + 1) if pose.pdb_info().chain(idx) == chainB]
    for res1 in groupA:
        r1 = pose.residue(res1)
        for res2 in groupB:
            r2 = pose.residue(res2)
            for a1 in range(1, r1.natoms() + 1):
                if r1.atom_name(a1).strip().startswith('H'):
                    continue
                xyz1 = r1.xyz(a1)
                for a2 in range(1, r2.natoms() + 1):
                    if r2.atom_name(a2).strip().startswith('H'):
                        continue
                    xyz2 = r2.xyz(a2)
                    if (xyz1 - xyz2).norm() < cutoff:
                        interface_res.add(res1)
                        interface_res.add(res2)
                        break
                else:
                    continue
                break

    return interface_res

def add_interface_constraints(pose, interface_residues_dict, distance_stddev=0.25, backbone_distance_cutoff=5.0):
    
    """Universal interface constraint: CA for proteins, C4' for nucleic acids (else P). Works for protein-protein, nucleic-nucleic, and protein-nucleic interfaces."""
    
    def main_atom(res):
        if res.is_protein() and res.has("CA"):
            return "CA"
        elif (res.is_DNA() or res.is_RNA()):
            if res.has("C4'"):
                return "C4'"
            elif res.has("P"):
                return "P"
        return None
    n_constraints = 0
    for key, value in interface_residues_dict.items():
        residues = value["residues"]
        chainA, chainB = value["chains"]
        groupA = [idx for idx in residues if pose.pdb_info().chain(idx) == chainA]
        groupB = [idx for idx in residues if pose.pdb_info().chain(idx) == chainB]
        for res1_idx in groupA:
            res1 = pose.residue(res1_idx)
            atom1 = main_atom(res1)
            if not atom1:
                continue
            for res2_idx in groupB:
                res2 = pose.residue(res2_idx)
                atom2 = main_atom(res2)
                if not atom2:
                    continue
                xyz1 = res1.xyz(atom1)
                xyz2 = res2.xyz(atom2)
                dist = (xyz1 - xyz2).norm()
                if dist < backbone_distance_cutoff:
                    id1 = AtomID(res1.atom_index(atom1), res1_idx)
                    id2 = AtomID(res2.atom_index(atom2), res2_idx)
                    func = HarmonicFunc(dist, distance_stddev)
                    pose.add_constraint(AtomPairConstraint(id1, id2, func))
                    n_constraints += 1

    print(f"Added {n_constraints} universal interface constraints across all chain pairs.")

def jitter_backbone(pose, max_jitter_protein=0.02, max_jitter_dna=0.04, max_jitter_rna=0.04):

    """Jitter backbone atoms with different ranges for proteins, DNA, and RNA."""

    jittered_pose = pose.clone()
    nres = pose.total_residue()
    for res_idx in range(1, nres + 1):
        res = pose.residue(res_idx)
        if res.is_protein():
            bb_atoms = ["N", "CA", "C", "O"]
            max_jitter = max_jitter_protein
        elif res.is_DNA():
            bb_atoms = ["P", "O5'", "C5'", "C4'", "C3'", "O3'"]
            max_jitter = max_jitter_dna
        elif res.is_RNA():
            bb_atoms = ["P", "O5'", "C5'", "C4'", "C3'", "O3'"]
            max_jitter = max_jitter_rna
        else:
            continue
        for atom_name in bb_atoms:
            if res.has(atom_name):
                atom_id = AtomID(res.atom_index(atom_name), res_idx)
                xyz = pose.xyz(atom_id)
                jittered_xyz = xyzVector_double_t(
                    xyz.x + (random.random() - 0.5) * 2 * max_jitter,
                    xyz.y + (random.random() - 0.5) * 2 * max_jitter,
                    xyz.z + (random.random() - 0.5) * 2 * max_jitter
                )
                jittered_pose.set_xyz(atom_id, jittered_xyz)

    return jittered_pose

def identify_watson_crick_pairs_by_criteria(pose, distance_cutoff=3.5):
    
    """Identify Watson-Crick base pairs based on distance and chain criteria."""

    wc_patterns = {
        ('A', 'T'): [('N6', 'O4'), ('N1', 'N3')], ('T', 'A'): [('N3', 'N1'), ('O4', 'N6')],  
        ('G', 'C'): [('N1', 'N3'), ('N2', 'O2'), ('O6', 'N4')], ('C', 'G'): [('N3', 'N1'), ('O2', 'N2'), ('N4', 'O6')],  
        ('A', 'U'): [('N6', 'O4'), ('N1', 'N3')], ('U', 'A'): [('N3', 'N1'), ('O4', 'N6')]
    }
    def get_base_type(residue):
        name = residue.name3().strip()
        base_map = {
            'DA': 'A', 'DT': 'T', 'DG': 'G', 'DC': 'C', 'A': 'A', 'T': 'T', 'G': 'G', 'C': 'C', 'U': 'U',
            'rA': 'A', 'rU': 'U', 'rG': 'G', 'rC': 'C', 'ADE': 'A', 'THY': 'T', 'GUA': 'G', 'CYT': 'C', 'URA': 'U'
        }
        return base_map.get(name, name)
    def get_chain_id(pose, res_idx):
        return pose.pdb_info().chain(res_idx) if pose.pdb_info() else 'A'
    def check_base_pair_distance(pose, res1_idx, res2_idx, atom1, atom2):
        res1 = pose.residue(res1_idx)
        res2 = pose.residue(res2_idx)
        if not (res1.has(atom1) and res2.has(atom2)):
            return False, 0.0
        xyz1 = res1.xyz(atom1)
        xyz2 = res2.xyz(atom2)
        distance = (xyz1 - xyz2).norm()
        return distance <= distance_cutoff, distance
    def meets_chain_criteria(pose, res1_idx, res2_idx, res1, res2):
        chain1 = get_chain_id(pose, res1_idx)
        chain2 = get_chain_id(pose, res2_idx)
        if chain1 != chain2:
            return True, f"inter-chain ({chain1}-{chain2})"
        if res1.is_RNA() and res2.is_RNA() and chain1 == chain2:
            return True, f"intra-RNA ({chain1})"
        if res1.is_DNA() and res2.is_DNA() and chain1 == chain2:
            return False, f"intra-DNA ({chain1}) - excluded"
        if ((res1.is_DNA() and res2.is_RNA()) or (res1.is_RNA() and res2.is_DNA())) and chain1 == chain2:
            return True, f"mixed DNA-RNA ({chain1})"
        return False, "unknown"
    def is_watson_crick_pair(pose, res1_idx, res2_idx):
        res1 = pose.residue(res1_idx)
        res2 = pose.residue(res2_idx)
        if not ((res1.is_DNA() or res1.is_RNA()) and (res2.is_DNA() or res2.is_RNA())):
            return False, None, 0.0, None
        meets_criteria, criteria_type = meets_chain_criteria(pose, res1_idx, res2_idx, res1, res2)
        if not meets_criteria:
            return False, None, 0.0, criteria_type
        base1 = get_base_type(res1)
        base2 = get_base_type(res2)
        pair_key = (base1, base2)
        if pair_key not in wc_patterns:
            return False, None, 0.0, criteria_type
        required_bonds = wc_patterns[pair_key]
        valid_bonds = 0
        min_distance = float('inf')
        for atom1, atom2 in required_bonds:
            is_valid, distance = check_base_pair_distance(pose, res1_idx, res2_idx, atom1, atom2)
            if is_valid:
                valid_bonds += 1
                min_distance = min(min_distance, distance)
        min_bonds_required = min(2, len(required_bonds))
        if valid_bonds >= min_bonds_required:
            pair_type = f"{base1}-{base2}"
            return True, pair_type, min_distance, criteria_type
        return False, None, 0.0, criteria_type
    protected_pairs = []
    nucleic_residues = []
    for i in range(1, pose.total_residue() + 1):
        res = pose.residue(i)
        if res.is_DNA() or res.is_RNA():
            nucleic_residues.append(i)
    for i, res1_idx in enumerate(nucleic_residues):
        for res2_idx in nucleic_residues[i+1:]:
            is_wc, pair_type, distance, criteria_type = is_watson_crick_pair(pose, res1_idx, res2_idx)
            if is_wc:
                protected_pairs.append((res1_idx, res2_idx, pair_type, distance, criteria_type))

    return protected_pairs, []

def add_nucleic_acid_constraints(pose, bond_stddev=0.015, angle_stddev=2.0, pucker_stddev_deg=10.0, hbond_stddev=1.0, coplanarity_stddev_deg=2.0, backbone_stddev_deg=15.0):
   
    """Comprehensive nucleic acid constraints."""

    pose.update_residue_neighbors()
    pose.conformation().detect_bonds()
    pucker_stddev_rad = math.radians(pucker_stddev_deg)
    backbone_stddev_rad = math.radians(backbone_stddev_deg)
    protected_pairs, _ = identify_watson_crick_pairs_by_criteria(pose)
    wc_patterns = {
        ('A', 'T'): [('N6', 'O4'), ('N1', 'N3')], ('T', 'A'): [('N3', 'N1'), ('O4', 'N6')], 
        ('G', 'C'): [('N1', 'N3'), ('N2', 'O2'), ('O6', 'N4')], ('C', 'G'): [('N3', 'N1'), ('O2', 'N2'), ('N4', 'O6')],  
        ('A', 'U'): [('N6', 'O4'), ('N1', 'N3')], ('U', 'A'): [('N3', 'N1'), ('O4', 'N6')]
    }
    purine_dihedrals = [('N9', 'C8', 'N7', 'C5'), ('C4', 'C5', 'C6', 'N1'), ('C6', 'N1', 'C2', 'N3')]
    pyrimidine_dihedrals = [('N1', 'C2', 'N3', 'C4'), ('C5', 'C4', 'N3', 'C2')]
    nucleic_torsions = {
        'alpha': [(0, "O3'", "P", "O5'", "C5'")], 'beta': [(0, "P", "O5'", "C5'", "C4'")], 'gamma': [(0, "O5'", "C5'", "C4'", "C3'")],
        'delta': [(0, "C5'", "C4'", "C3'", "O3'")], 'epsilon': [(0, "C4'", "C3'", "O3'", "P")], 'zeta': [(0, "C3'", "O3'", "P", "O5'")],
        'chi_pur': [(0, "O4'", "C1'", "N9", "C4")], 'chi_pyr': [(0, "O4'", "C1'", "N1", "C2")],
        'nu0': [(0, "C4'", "O4'", "C1'", "C2'")], 'nu1': [(0, "O4'", "C1'", "C2'", "C3'")], 'nu2': [(0, "C1'", "C2'", "C3'", "C4'")],
        'nu3': [(0, "C2'", "C3'", "C4'", "O4'")], 'nu4': [(0, "C3'", "C4'", "O4'", "C1'")]
    }
    def get_base_type(residue):
        name = residue.name3().strip()
        base_map = {
            'DA': 'A', 'DT': 'T', 'DG': 'G', 'DC': 'C', 'A': 'A', 'T': 'T', 'G': 'G', 'C': 'C', 'U': 'U',
            'rA': 'A', 'rU': 'U', 'rG': 'G', 'rC': 'C', 'ADE': 'A', 'THY': 'T', 'GUA': 'G', 'CYT': 'C', 'URA': 'U'
        }
        return base_map.get(name, name)
    def base_atoms(res):
        return ("N9", "C8") if res.is_purine() else ("N1", "C6")
    hbond_count = 0
    coplanarity_count = 0
    coplanarity_stddev_rad = math.radians(coplanarity_stddev_deg)
    for res1_idx, res2_idx, pair_type, distance, criteria_type in protected_pairs:
        res1 = pose.residue(res1_idx)
        res2 = pose.residue(res2_idx)
        base1 = get_base_type(res1)
        base2 = get_base_type(res2)
        pair_key = (base1, base2)
        if pair_key in wc_patterns:
            for atom1_name, atom2_name in wc_patterns[pair_key]:
                if res1.has(atom1_name) and res2.has(atom2_name):
                    atom1_id = AtomID(res1.atom_index(atom1_name), res1_idx)
                    atom2_id = AtomID(res2.atom_index(atom2_name), res2_idx)
                    current_distance = (res1.xyz(atom1_name) - res2.xyz(atom2_name)).norm()
                    tight_stddev = min(hbond_stddev * 0.5, 0.1)
                    func = HarmonicFunc(current_distance, tight_stddev)
                    pose.add_constraint(AtomPairConstraint(atom1_id, atom2_id, func))
                    hbond_count += 1
        a1_i, a2_i = base_atoms(res1)
        a1_j, a2_j = base_atoms(res2)
        if res1.has(a1_i) and res1.has(a2_i) and res2.has(a1_j) and res2.has(a2_j):
            ids = [AtomID(res1.atom_index(a1_i), res1_idx), AtomID(res1.atom_index(a2_i), res1_idx),
                   AtomID(res2.atom_index(a1_j), res2_idx), AtomID(res2.atom_index(a2_j), res2_idx)]
            current_dihedral = rosetta.numeric.dihedral_degrees(res1.xyz(a1_i), res1.xyz(a2_i), res2.xyz(a1_j), res2.xyz(a2_j))
            target_angle = 0.0 if abs(current_dihedral) < 90.0 else 180.0
            target_rad = math.radians(target_angle)
            func = CircularHarmonicFunc(target_rad, coplanarity_stddev_rad)
            pose.add_constraint(DihedralConstraint(*ids, func))
            coplanarity_count += 1
    for i in range(1, pose.total_residue() + 1):
        res = pose.residue(i)
        if not (res.is_RNA() or res.is_DNA()):
            continue
        ring_bonds = [("C1'", "O4'"), ("O4'", "C4'"), ("C4'", "C3'"), ("C3'", "C2'"), ("C2'", "C1'")]
        for atom1, atom2 in ring_bonds:
            if res.has(atom1) and res.has(atom2):
                id1 = AtomID(res.atom_index(atom1), i)
                id2 = AtomID(res.atom_index(atom2), i)
                dist = (res.xyz(atom1) - res.xyz(atom2)).norm()
                func = HarmonicFunc(dist, bond_stddev)
                pose.add_constraint(AtomPairConstraint(id1, id2, func))
        ring_angles = [("C1'", "O4'", "C4'"), ("O4'", "C4'", "C3'"), ("C4'", "C3'", "C2'"), ("C3'", "C2'", "C1'"), ("C2'", "C1'", "O4'")]
        for atoms in ring_angles:
            if all(res.has(a) for a in atoms):
                ids = [AtomID(res.atom_index(a), i) for a in atoms]
                current_angle = rosetta.numeric.angle_degrees(res.xyz(atoms[0]), res.xyz(atoms[1]), res.xyz(atoms[2]))
                func = HarmonicFunc(current_angle, angle_stddev)
                pose.add_constraint(AngleConstraint(*ids, func))
        ring_dihedrals = [("C4'", "O4'", "C1'", "C2'"), ("O4'", "C1'", "C2'", "C3'"), ("C1'", "C2'", "C3'", "C4'"), ("C2'", "C3'", "C4'", "O4'"), ("C3'", "C4'", "O4'", "C1'")]
        for atoms in ring_dihedrals:
            if all(res.has(a) for a in atoms):
                ids = [AtomID(res.atom_index(a), i) for a in atoms]
                current = math.radians(rosetta.numeric.dihedral_degrees(res.xyz(atoms[0]), res.xyz(atoms[1]), res.xyz(atoms[2]), res.xyz(atoms[3])))
                func = CircularHarmonicFunc(current, pucker_stddev_rad)
                pose.add_constraint(DihedralConstraint(*ids, func))
        critical_angles = [("O3'", "P", "O5'", 104.0), ("P", "O5'", "C5'", 120.0), ("O5'", "C5'", "C4'", 109.0), ("C5'", "C4'", "O4'", 109.0), ("C4'", "O4'", "C1'", 109.0)]
        for atoms in critical_angles:
            if len(atoms) == 4 and all(res.has(a) for a in atoms[:3]):
                ids = [AtomID(res.atom_index(a), i) for a in atoms[:3]]
                func = HarmonicFunc(atoms[3], 1.5)
                pose.add_constraint(AngleConstraint(*ids, func))
        all_nucleic_angles = [("C1'", "O4'", "C4'"), ("O4'", "C4'", "C3'"), ("C4'", "C3'", "C2'"), ("C3'", "C2'", "C1'"), ("C2'", "C1'", "O4'"), ("O4'", "C1'", "C2'"), ("C1'", "C2'", "C3'"), ("C2'", "C3'", "C4'"), ("C3'", "C4'", "O4'"), ("C4'", "O4'", "C1'"), ("P", "O5'", "C5'"), ("O5'", "C5'", "C4'"), ("C5'", "C4'", "C3'"), ("C4'", "C3'", "O3'")]
        for atoms in all_nucleic_angles:
            if all(res.has(a) for a in atoms):
                ids = [AtomID(res.atom_index(a), i) for a in atoms]
                current_angle = rosetta.numeric.angle_degrees(res.xyz(atoms[0]), res.xyz(atoms[1]), res.xyz(atoms[2]))
                func = HarmonicFunc(current_angle, 1.0)
                pose.add_constraint(AngleConstraint(*ids, func))
        planarity_stddev_rad = math.radians(5.0)
        dihedrals = purine_dihedrals if res.is_purine() else pyrimidine_dihedrals
        for atoms in dihedrals:
            if all(res.has(atom) for atom in atoms):
                ids = [AtomID(res.atom_index(atom), i) for atom in atoms]
                xyzs = [res.xyz(atom) for atom in atoms]
                angle = rosetta.numeric.dihedral_degrees(*xyzs)
                func = CircularHarmonicFunc(math.radians(angle), planarity_stddev_rad)
                pose.add_constraint(DihedralConstraint(*ids, func))
        torsions = nucleic_torsions.copy()
        if res.is_purine():
            torsions['chi'] = torsions['chi_pur']
        else:
            torsions['chi'] = torsions['chi_pyr']
        torsions.pop('chi_pur')
        torsions.pop('chi_pyr')
        for torsion_name, torsion_list in torsions.items():
            for offset, a1, a2, a3, a4 in torsion_list:
                res_idx = i + offset
                if res_idx < 1 or res_idx > pose.total_residue():
                    continue
                target_res = pose.residue(res_idx)
                if not all(target_res.has(atom) for atom in [a1, a2, a3, a4]):
                    continue
                ids = [AtomID(target_res.atom_index(atom), res_idx) for atom in [a1, a2, a3, a4]]
                angle_deg = rosetta.numeric.dihedral_degrees(target_res.xyz(a1), target_res.xyz(a2), target_res.xyz(a3), target_res.xyz(a4))
                angle_rad = math.radians(angle_deg)
                func = CircularHarmonicFunc(angle_rad, backbone_stddev_rad)
                pose.add_constraint(DihedralConstraint(*ids, func))
    nres = pose.total_residue()
    for i in range(1, nres):
        res_i = pose.residue(i)
        res_j = pose.residue(i + 1)
        if (res_i.is_DNA() or res_i.is_RNA()) and (res_j.is_DNA() or res_j.is_RNA()):
            if res_i.has("O3'") and res_j.has("P"):
                id1 = AtomID(res_i.atom_index("O3'"), i)
                id2 = AtomID(res_j.atom_index("P"), i + 1)
                dist = (res_i.xyz("O3'") - res_j.xyz("P")).norm()
                func = HarmonicFunc(dist, 0.06)  
                pose.add_constraint(AtomPairConstraint(id1, id2, func))
    
    print(f"Nucleic acid constraints added: {hbond_count} H-bonds, {coplanarity_count} coplanarity for {len(protected_pairs)} base pairs.")
    criteria_counts = {}
    for _, _, _, _, criteria_type in protected_pairs:
        criteria_counts[criteria_type] = criteria_counts.get(criteria_type, 0) + 1
    print(f" ")
    print("Protected base pairs by type:")
    for criteria, count_pairs in criteria_counts.items():
        print(f"{criteria}: {count_pairs} pairs")

def add_protein_constraints(pose, angle_stddev=1.5, backbone_stddev_deg=15.0):

    """Comprehensive protein constraints."""

    pose.update_residue_neighbors()
    pose.conformation().detect_bonds()
    backbone_stddev_rad = math.radians(backbone_stddev_deg)
    aromatic_dihedrals = {
        'PHE': [('CG', 'CD1', 'CE1', 'CZ'), ('CG', 'CD2', 'CE2', 'CZ'), ('CD1', 'CE1', 'CZ', 'CE2'), ('CD2', 'CE2', 'CZ', 'CE1')],
        'TYR': [('CG', 'CD1', 'CE1', 'CZ'), ('CG', 'CD2', 'CE2', 'CZ'), ('CD1', 'CE1', 'CZ', 'CE2'), ('CD2', 'CE2', 'CZ', 'CE1')],
        'TRP': [('CD2', 'CE2', 'NE1', 'CD1'), ('CG', 'CD1', 'NE1', 'CE2'), ('CD2', 'CE2', 'CZ2', 'CH2'), ('CE2', 'CZ2', 'CH2', 'CZ3')],
        'HIS': [('CG', 'ND1', 'CE1', 'NE2'), ('CG', 'CD2', 'NE2', 'CE1')]
    }
    nres = pose.total_residue()
    for i in range(1, pose.total_residue() + 1):
        res = pose.residue(i)
        if not res.is_protein():
            continue
        if res.has("N") and res.has("CA") and res.has("C"):
            id1 = AtomID(res.atom_index("N"), i)
            id2 = AtomID(res.atom_index("CA"), i)
            id3 = AtomID(res.atom_index("C"), i)
            current_angle = rosetta.numeric.angle_degrees(res.xyz("N"), res.xyz("CA"), res.xyz("C"))
            func = HarmonicFunc(current_angle, angle_stddev)
            pose.add_constraint(AngleConstraint(id1, id2, id3, func))
        resname = res.name3()
        if resname in aromatic_dihedrals:
            planarity_stddev_rad = math.radians(5.0)
            for atoms in aromatic_dihedrals[resname]:
                if all(res.has(atom) for atom in atoms):
                    ids = [AtomID(res.atom_index(atom), i) for atom in atoms]
                    xyzs = [res.xyz(atom) for atom in atoms]
                    angle = rosetta.numeric.dihedral_degrees(*xyzs)
                    func = CircularHarmonicFunc(math.radians(angle), planarity_stddev_rad)
                    pose.add_constraint(DihedralConstraint(*ids, func))
        if i > 1:
            prev_res = pose.residue(i - 1)
            if all([prev_res.has("C"), res.has("N"), res.has("CA"), res.has("C")]):
                ids = [AtomID(prev_res.atom_index("C"), i - 1), AtomID(res.atom_index("N"), i), AtomID(res.atom_index("CA"), i), AtomID(res.atom_index("C"), i)]
                angle_deg = rosetta.numeric.dihedral_degrees(prev_res.xyz("C"), res.xyz("N"), res.xyz("CA"), res.xyz("C"))
                angle_rad = math.radians(angle_deg)
                func = CircularHarmonicFunc(angle_rad, backbone_stddev_rad)
                pose.add_constraint(DihedralConstraint(*ids, func))
        if i < nres:
            next_res = pose.residue(i + 1)
            if all([res.has("N"), res.has("CA"), res.has("C"), next_res.has("N")]):
                ids = [AtomID(res.atom_index("N"), i), AtomID(res.atom_index("CA"), i), AtomID(res.atom_index("C"), i), AtomID(next_res.atom_index("N"), i + 1)]
                angle_deg = rosetta.numeric.dihedral_degrees(res.xyz("N"), res.xyz("CA"), res.xyz("C"), next_res.xyz("N"))
                angle_rad = math.radians(angle_deg)
                func = CircularHarmonicFunc(angle_rad, backbone_stddev_rad)
                pose.add_constraint(DihedralConstraint(*ids, func))
            if all([res.has("CA"), res.has("C"), next_res.has("N"), next_res.has("CA")]):
                ids = [AtomID(res.atom_index("CA"), i), AtomID(res.atom_index("C"), i), AtomID(next_res.atom_index("N"), i + 1), AtomID(next_res.atom_index("CA"), i + 1)]
                angle_deg = rosetta.numeric.dihedral_degrees(res.xyz("CA"), res.xyz("C"), next_res.xyz("N"), next_res.xyz("CA"))
                angle_rad = math.radians(angle_deg)
                func = CircularHarmonicFunc(angle_rad, backbone_stddev_rad)
                pose.add_constraint(DihedralConstraint(*ids, func))
    for i in range(1, nres):
        res_i = pose.residue(i)
        res_j = pose.residue(i + 1)
        if res_i.is_protein() and res_j.is_protein():
            if res_i.has("C") and res_j.has("N"):
                id1 = AtomID(res_i.atom_index("C"), i)
                id2 = AtomID(res_j.atom_index("N"), i + 1)
                dist = (res_i.xyz("C") - res_j.xyz("N")).norm()
                func = HarmonicFunc(dist, 0.06)
                pose.add_constraint(AtomPairConstraint(id1, id2, func))
                
    print("Protein constraints added.")

def reset_score_weights(scorefxn_cart):

    """Reset score weights to initial values."""

    scorefxn_cart.set_weight(score_type_from_name("fa_atr"), fa_atr_init)
    scorefxn_cart.set_weight(score_type_from_name("fa_rep"), fa_rep_init) 
    scorefxn_cart.set_weight(score_type_from_name("fa_sol"), fa_sol_init)
    scorefxn_cart.set_weight(score_type_from_name("fa_intra_rep"), fa_intra_rep_init)
    scorefxn_cart.set_weight(score_type_from_name("fa_intra_sol_xover4"), fa_intra_sol_xover4_init)
    scorefxn_cart.set_weight(score_type_from_name("lk_ball_wtd"), lk_ball_wtd_init)
    scorefxn_cart.set_weight(score_type_from_name("fa_elec"), fa_elec_init)
    scorefxn_cart.set_weight(score_type_from_name("hbond_sr_bb"), hbond_sr_bb_init)
    scorefxn_cart.set_weight(score_type_from_name("hbond_lr_bb"), hbond_lr_bb_init)
    scorefxn_cart.set_weight(score_type_from_name("hbond_bb_sc"), hbond_bb_sc_init)
    scorefxn_cart.set_weight(score_type_from_name("hbond_sc"), hbond_sc_init)
    scorefxn_cart.set_weight(score_type_from_name("dslf_fa13"), dslf_fa13_init)
    scorefxn_cart.set_weight(score_type_from_name("omega"), omega_init)
    scorefxn_cart.set_weight(score_type_from_name("fa_dun"), fa_dun_init)
    scorefxn_cart.set_weight(score_type_from_name("p_aa_pp"), p_aa_pp_init)
    scorefxn_cart.set_weight(score_type_from_name("yhh_planarity"), yhh_planarity_init)
    scorefxn_cart.set_weight(score_type_from_name("ref"), ref_init)
    scorefxn_cart.set_weight(score_type_from_name("rama_prepro"), rama_prepro_init)
    scorefxn_cart.set_weight(rosetta.core.scoring.cart_bonded, cart_bonded_init)
    scorefxn_cart.set_weight(rosetta.core.scoring.dihedral_constraint, 1.0)
    scorefxn_cart.set_weight(rosetta.core.scoring.atom_pair_constraint, 1.0)
    scorefxn_cart.set_weight(score_type_from_name("rna_torsion"), rna_torsion_init)
    scorefxn_cart.set_weight(score_type_from_name("rna_sugar_close"), rna_sugar_close_init) 
    scorefxn_cart.set_weight(score_type_from_name("dna_bb_torsion"), dna_bb_torsion_init)           
    scorefxn_cart.set_weight(score_type_from_name("dna_sugar_close"), dna_sugar_close_init) 

    return scorefxn_cart

def sinusoidal_ramp(start, end, n_steps):

    """Generate a list of weights using a sinusoidal ramp from start to end."""

    return [start + (end - start) * 0.5 * (1 - np.cos(np.pi * i / (n_steps - 1))) for i in range(n_steps)]

def adaptive_constraint_scale(current_score, prev_score, max_change=0.05):

    """Returns a scale factor for constraint stddevs based on the absolute percent score change. Tightens (up to 1.05) if score improves, loosens (down to 0.95) if worsens, max ±5%."""

    if prev_score is None or prev_score == 0 or current_score is None:
        return 1.0
    percent_change = abs((current_score - prev_score) / prev_score)
    percent_change = min(percent_change, max_change)
    if current_score < prev_score:

        return 1.0 + percent_change
    else:
        return 1.0 - percent_change

def detect_residue_types(pose):

    """Detect what types of residues are present in the pose."""

    has_protein = False
    has_nucleic = False
    has_dna = False
    has_rna = False
    for i in range(1, pose.total_residue() + 1):
        res = pose.residue(i)
        if res.is_protein():
            has_protein = True
        elif res.is_DNA():
            has_nucleic = True
            has_dna = True
        elif res.is_RNA():
            has_nucleic = True
            has_rna = True
    
    return has_protein, has_nucleic, has_dna, has_rna

def add_conditional_constraints(pose, interface_residues_dict, scale, distance_stddev=0.25, backbone_distance_cutoff=5.0):

    """Add constraints based on detected residue types."""

    has_protein, has_nucleic, has_dna, has_rna = detect_residue_types(pose)
    print(f"Detected residue types: Protein={has_protein}, DNA={has_dna}, RNA={has_rna}")

    if len(interface_residues_dict) > 0:
        add_interface_constraints(pose, interface_residues_dict, distance_stddev * scale, backbone_distance_cutoff * scale)
    if has_protein:
        add_protein_constraints(pose, angle_stddev=1.5 * scale, backbone_stddev_deg=15.0 * scale)
    else:
        print("No protein residues detected - skipping protein constraints")

    if has_nucleic:
        add_nucleic_acid_constraints(pose, bond_stddev=0.015 * scale, angle_stddev=2.0 * scale, pucker_stddev_deg=10.0 * scale, hbond_stddev=1.0 * scale, coplanarity_stddev_deg=2.0 * scale, backbone_stddev_deg=15.0 * scale)
    else:
        print("No nucleic acid residues detected - skipping nucleic acid constraints")

# ----------------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #

print(" ")
print("=== INPUT POSE ===")
print(" ")

pose.energies().clear()   
score_cart= scorefxn_cart(pose) 
print(" ")
print(f"Input score = {score_cart:.3f}")

print(" ")
terms = scorefxn_cart.get_nonzero_weighted_scoretypes()
print("Score terms:")
for term in terms:
    val = pose.energies().total_energies()[term]
    print(f"{term}: {val:.3f}")

n_chains = pose.num_chains()
chain_ids = [pose.pdb_info().chain(pose.chain_begin(i)) for i in range(1, n_chains + 1)]
print(" ")  
print("Detected chain IDs:", chain_ids)

print(" ")

all_interface_residues = set()
interface_residues_dict = {}
for i in range(len(chain_ids)):
    for j in range(i + 1, len(chain_ids)):
        residues = get_interface_residues(pose, [
            pose.chain_begin(i + 1),  
            pose.chain_begin(j + 1)
        ])
        key = f"interface_{chain_ids[i]}_{chain_ids[j]}"
        interface_residues_dict[key] = {
            "residues": list(residues),
            "chains": (chain_ids[i], chain_ids[j])
        }
        all_interface_residues.update(residues)

interface_residues = sorted(all_interface_residues)

prev_score = None
current_score = scorefxn_cart(pose)
num_cycles = 10

for cycle in range(num_cycles):
    
    print(f"=== INITIAL REFINEMENT CYCLE {cycle+1} ===")
    print(" ")
    movemap = MoveMap()
    movemap.set_bb(True)
    movemap.set_chi(True)
    movemap.set_jump(True)

    pose.remove_constraints()
    pose.update_residue_neighbors()
    scorefxn_cart = reset_score_weights(scorefxn_cart)

    scale = adaptive_constraint_scale(current_score, prev_score)
    add_conditional_constraints(pose, interface_residues_dict, scale)
    print(" ")

    jitter_backbone(pose, max_jitter_protein=0.08, max_jitter_dna=0.05, max_jitter_rna=0.05)
    print("Backbone jittered.")

    print(" ")

    scorefxn_cart.set_weight(score_type_from_name("fa_atr"), 0.01 * fa_atr_init)
    scorefxn_cart.set_weight(score_type_from_name("fa_rep"), 0.01 * fa_rep_init)  
    scorefxn_cart.set_weight(score_type_from_name("fa_sol"), 0.01 * fa_sol_init)
    scorefxn_cart.set_weight(score_type_from_name("fa_intra_rep"), 0.01 * fa_intra_rep_init)
    scorefxn_cart.set_weight(score_type_from_name("fa_intra_sol_xover4"), 0.01 * fa_intra_sol_xover4_init)
    scorefxn_cart.set_weight(score_type_from_name("lk_ball_wtd"), 0.01 * lk_ball_wtd_init)
    scorefxn_cart.set_weight(score_type_from_name("fa_elec"), 0.01 * fa_elec_init)
    scorefxn_cart.set_weight(score_type_from_name("hbond_sr_bb"), 0.01 * hbond_sr_bb_init)
    scorefxn_cart.set_weight(score_type_from_name("hbond_lr_bb"), 0.01 * hbond_lr_bb_init)
    scorefxn_cart.set_weight(score_type_from_name("hbond_bb_sc"), 0.01 * hbond_bb_sc_init)
    scorefxn_cart.set_weight(score_type_from_name("hbond_sc"), 0.01 * hbond_sc_init)
    scorefxn_cart.set_weight(score_type_from_name("dslf_fa13"), 0.01 * dslf_fa13_init)
    scorefxn_cart.set_weight(score_type_from_name("omega"), 0.01 * omega_init)
    scorefxn_cart.set_weight(score_type_from_name("fa_dun"), 0.01 * fa_dun_init)
    scorefxn_cart.set_weight(score_type_from_name("p_aa_pp"), 0.01 * p_aa_pp_init)
    scorefxn_cart.set_weight(score_type_from_name("yhh_planarity"), 0.01 * yhh_planarity_init)
    scorefxn_cart.set_weight(score_type_from_name("ref"), 0.01 * ref_init)
    scorefxn_cart.set_weight(score_type_from_name("rama_prepro"), 0.01 * rama_prepro_init)
    scorefxn_cart.set_weight(rosetta.core.scoring.cart_bonded, 0.01 * cart_bonded_init)  
    scorefxn_cart.set_weight(rosetta.core.scoring.dihedral_constraint, 0.01)
    scorefxn_cart.set_weight(rosetta.core.scoring.atom_pair_constraint, 0.01)
    scorefxn_cart.set_weight(score_type_from_name("rna_torsion"), 0.01 * rna_torsion_init)
    scorefxn_cart.set_weight(score_type_from_name("rna_sugar_close"), 0.01 * rna_sugar_close_init) 
    scorefxn_cart.set_weight(score_type_from_name("dna_bb_torsion"), 0.01 * dna_bb_torsion_init)           
    scorefxn_cart.set_weight(score_type_from_name("dna_sugar_close"), 0.01 * dna_sugar_close_init)  
    scorefxn_cart.set_weight(rosetta.core.scoring.fa_stack, 0.01 * fa_stack_init)

    tf = TaskFactory()
    tf.push_back(RestrictToRepacking())
    tf.push_back(IncludeCurrent())
    packer_task = tf.create_task_and_apply_taskoperations(pose)
    pack_mover = PackRotamersMover(scorefxn_cart, packer_task)
    pack_mover.apply(pose)

    pose.update_residue_neighbors()

    min_mover = MinMover()
    min_mover.movemap(movemap)
    min_mover.score_function(scorefxn_cart)
    min_mover.min_type('lbfgs_armijo_nonmonotone') 
    min_mover.max_iter(250)
    min_mover.tolerance(0.05)
    min_mover.cartesian(True)
    min_mover.apply(pose)

    pose.update_residue_neighbors()

    tf = TaskFactory()
    tf.push_back(RestrictToRepacking())
    tf.push_back(IncludeCurrent())
    packer_task = tf.create_task_and_apply_taskoperations(pose)
    pack_mover = PackRotamersMover(scorefxn_cart, packer_task)
    pack_mover.apply(pose)

    current_score = scorefxn_cart(pose)

    pose.energies().clear()   
    score_cart= scorefxn_cart(pose) 
    print(" ")   
    print(f"Initial refinement cycle {cycle+1} score = {score_cart:.3f}")
    print(" ")
    terms = scorefxn_cart.get_nonzero_weighted_scoretypes()
    print("Score terms:")
    for term in terms:
        val = pose.energies().total_energies()[term]
        print(f"{term}: {val:.3f}")

    current_score = scorefxn_cart(pose)
    prev_score = current_score

    print(" ")

# ----------------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #

num_cycles = 20

fa_atr_weights              = sinusoidal_ramp(0.01 * fa_atr_init, fa_atr_init, num_cycles+1)[1:]
fa_rep_weights              = sinusoidal_ramp(0.01 * fa_rep_init, fa_rep_init * 1.5, num_cycles+1)[1:]  
fa_sol_weights              = sinusoidal_ramp(0.01 * fa_sol_init, fa_sol_init, num_cycles+1)[1:]
fa_intra_rep_weights        = sinusoidal_ramp(0.01 * fa_intra_rep_init, fa_intra_rep_init, num_cycles+1)[1:]
fa_intra_sol_xover4_weights = sinusoidal_ramp(0.01 * fa_intra_sol_xover4_init, fa_intra_sol_xover4_init, num_cycles+1)[1:]
lk_ball_wtd_weights         = sinusoidal_ramp(0.01 * lk_ball_wtd_init, lk_ball_wtd_init, num_cycles+1)[1:]
fa_elec_weights             = sinusoidal_ramp(0.01 * fa_elec_init, fa_elec_init, num_cycles+1)[1:]
hbond_sr_bb_weights         = sinusoidal_ramp(0.01 * hbond_sr_bb_init, hbond_sr_bb_init, num_cycles+1)[1:]
hbond_lr_bb_weights         = sinusoidal_ramp(0.01 * hbond_lr_bb_init, hbond_lr_bb_init, num_cycles+1)[1:]
hbond_bb_sc_weights         = sinusoidal_ramp(0.01 * hbond_bb_sc_init, hbond_bb_sc_init, num_cycles+1)[1:]
hbond_sc_weights            = sinusoidal_ramp(0.01 * hbond_sc_init, hbond_sc_init, num_cycles+1)[1:]
dslf_fa13_weights           = sinusoidal_ramp(0.01 * dslf_fa13_init, dslf_fa13_init, num_cycles+1)[1:]
omega_weights               = sinusoidal_ramp(0.01 * omega_init, omega_init, num_cycles+1)[1:]
fa_dun_weights              = sinusoidal_ramp(0.01 * fa_dun_init, fa_dun_init, num_cycles+1)[1:]
p_aa_pp_weights             = sinusoidal_ramp(0.01 * p_aa_pp_init, p_aa_pp_init, num_cycles+1)[1:]
yhh_planarity_weights       = sinusoidal_ramp(0.01 * yhh_planarity_init, yhh_planarity_init, num_cycles+1)[1:]
ref_weights                 = sinusoidal_ramp(0.01 * ref_init, ref_init, num_cycles+1)[1:]
rama_prepro_weights         = sinusoidal_ramp(0.01 * rama_prepro_init, rama_prepro_init, num_cycles+1)[1:]
cart_bonded_weights         = sinusoidal_ramp(0.01 * cart_bonded_init, cart_bonded_init * 1.5, num_cycles+1)[1:] 
dihedral_weights            = sinusoidal_ramp(0.01, 1.25, num_cycles+1)[1:]
atom_pair_weights           = sinusoidal_ramp(0.01, 1.0, num_cycles+1)[1:]
rna_torsion_weights         = sinusoidal_ramp(0.01 * rna_torsion_init, rna_torsion_init, num_cycles+1)[1:]
rna_sugar_weights           = sinusoidal_ramp(0.01 * rna_sugar_close_init, rna_sugar_close_init * 2.0, num_cycles+1)[1:] 
dna_bb_weights              = sinusoidal_ramp(0.01 * dna_bb_torsion_init, dna_bb_torsion_init, num_cycles+1)[1:]
dna_sugar_weights           = sinusoidal_ramp(0.01 * dna_sugar_close_init, dna_sugar_close_init * 2.0, num_cycles+1)[1:] 
fa_stack_weights            = sinusoidal_ramp(0.01 * fa_stack_init, fa_stack_init, num_cycles+1)[1:]

for cycle in range(num_cycles):

    print(f"=== ITERATIVE CONSTRAINED REFINEMENT CYCLE {cycle+1} ===")
    print(" ")
    movemap = MoveMap()
    movemap.set_bb(True)
    movemap.set_chi(True)
    movemap.set_jump(True)
    
    pose.remove_constraints()
    pose.update_residue_neighbors()
    scorefxn_cart = reset_score_weights(scorefxn_cart)

    scale = adaptive_constraint_scale(current_score, prev_score)
    add_conditional_constraints(pose, interface_residues_dict, scale)
    print(" ")

    jitter_backbone(pose, max_jitter_protein=0.06, max_jitter_dna=0.04, max_jitter_rna=0.04)
    print("Backbone jittered.")

    print(" ")

    scorefxn_cart.set_weight(score_type_from_name("fa_atr"), fa_atr_weights[cycle])
    scorefxn_cart.set_weight(score_type_from_name("fa_rep"), fa_rep_weights[cycle])
    scorefxn_cart.set_weight(score_type_from_name("fa_sol"), fa_sol_weights[cycle])
    scorefxn_cart.set_weight(score_type_from_name("fa_intra_rep"), fa_intra_rep_weights[cycle])
    scorefxn_cart.set_weight(score_type_from_name("fa_intra_sol_xover4"), fa_intra_sol_xover4_weights[cycle])
    scorefxn_cart.set_weight(score_type_from_name("lk_ball_wtd"), lk_ball_wtd_weights[cycle])
    scorefxn_cart.set_weight(score_type_from_name("fa_elec"), fa_elec_weights[cycle])
    scorefxn_cart.set_weight(score_type_from_name("hbond_sr_bb"), hbond_sr_bb_weights[cycle])
    scorefxn_cart.set_weight(score_type_from_name("hbond_lr_bb"), hbond_lr_bb_weights[cycle])
    scorefxn_cart.set_weight(score_type_from_name("hbond_bb_sc"), hbond_bb_sc_weights[cycle])
    scorefxn_cart.set_weight(score_type_from_name("hbond_sc"), hbond_sc_weights[cycle])
    scorefxn_cart.set_weight(score_type_from_name("dslf_fa13"), dslf_fa13_weights[cycle])
    scorefxn_cart.set_weight(score_type_from_name("omega"), omega_weights[cycle])
    scorefxn_cart.set_weight(score_type_from_name("fa_dun"), fa_dun_weights[cycle])
    scorefxn_cart.set_weight(score_type_from_name("p_aa_pp"), p_aa_pp_weights[cycle])
    scorefxn_cart.set_weight(score_type_from_name("yhh_planarity"), yhh_planarity_weights[cycle])
    scorefxn_cart.set_weight(score_type_from_name("ref"), ref_weights[cycle])
    scorefxn_cart.set_weight(score_type_from_name("rama_prepro"), rama_prepro_weights[cycle])
    scorefxn_cart.set_weight(rosetta.core.scoring.cart_bonded, cart_bonded_weights[cycle])
    scorefxn_cart.set_weight(rosetta.core.scoring.dihedral_constraint, dihedral_weights[cycle])
    scorefxn_cart.set_weight(rosetta.core.scoring.atom_pair_constraint, atom_pair_weights[cycle])
    scorefxn_cart.set_weight(score_type_from_name("rna_torsion"), rna_torsion_weights[cycle])
    scorefxn_cart.set_weight(score_type_from_name("rna_sugar_close"), rna_sugar_weights[cycle]) 
    scorefxn_cart.set_weight(score_type_from_name("dna_bb_torsion"), dna_bb_weights[cycle])           
    scorefxn_cart.set_weight(score_type_from_name("dna_sugar_close"), dna_sugar_weights[cycle])  
    scorefxn_cart.set_weight(rosetta.core.scoring.fa_stack, fa_stack_weights[cycle])

    tf = TaskFactory()
    tf.push_back(RestrictToRepacking())
    tf.push_back(IncludeCurrent())
    packer_task = tf.create_task_and_apply_taskoperations(pose)
    pack_mover = PackRotamersMover(scorefxn_cart, packer_task)
    pack_mover.apply(pose)

    pose.update_residue_neighbors()

    min_mover = MinMover()
    min_mover.movemap(movemap)
    min_mover.score_function(scorefxn_cart)
    min_mover.min_type('lbfgs_armijo_nonmonotone') 
    min_mover.max_iter(250)
    min_mover.tolerance(0.01)
    min_mover.cartesian(True)
    min_mover.apply(pose)

    pose.update_residue_neighbors()

    tf = TaskFactory()
    tf.push_back(RestrictToRepacking())
    tf.push_back(IncludeCurrent())
    packer_task = tf.create_task_and_apply_taskoperations(pose)
    pack_mover = PackRotamersMover(scorefxn_cart, packer_task)
    pack_mover.apply(pose)

    current_score = scorefxn_cart(pose)

    pose.energies().clear()   
    score_cart= scorefxn_cart(pose)    
    print(" ") 
    print(f"Iterative constrained refinement cycle {cycle+1} score = {score_cart:.3f}")
    print(" ")
    terms = scorefxn_cart.get_nonzero_weighted_scoretypes()
    print("Score terms:")
    for term in terms:
        val = pose.energies().total_energies()[term]
        print(f"{term}: {val:.3f}")

    current_score = scorefxn_cart(pose)
    prev_score = current_score

    print(" ")

# ----------------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #

num_cycles = 10

for cycle in range(num_cycles):

    print(f"=== LATE REFINEMENT CYCLE {cycle+1} ===")
    print(" ")
    movemap = MoveMap()
    movemap.set_bb(True)
    movemap.set_chi(True)
    movemap.set_jump(True)

    pose.remove_constraints()
    pose.update_residue_neighbors()
    scorefxn_cart = reset_score_weights(scorefxn_cart)

    scale = adaptive_constraint_scale(current_score, prev_score)
    add_conditional_constraints(pose, interface_residues_dict, scale)
    print(" ")

    jitter_backbone(pose, max_jitter_protein=0.04, max_jitter_dna=0.03, max_jitter_rna=0.03)
    print("Backbone jittered.")

    print(" ")

    scorefxn_cart.set_weight(score_type_from_name("fa_atr"), fa_atr_init)
    scorefxn_cart.set_weight(score_type_from_name("fa_rep"), fa_rep_init * 1.5) 
    scorefxn_cart.set_weight(score_type_from_name("fa_sol"), fa_sol_init)
    scorefxn_cart.set_weight(score_type_from_name("fa_intra_rep"), fa_intra_rep_init)
    scorefxn_cart.set_weight(score_type_from_name("fa_intra_sol_xover4"), fa_intra_sol_xover4_init)
    scorefxn_cart.set_weight(score_type_from_name("lk_ball_wtd"), lk_ball_wtd_init)
    scorefxn_cart.set_weight(score_type_from_name("fa_elec"), fa_elec_init)
    scorefxn_cart.set_weight(score_type_from_name("hbond_sr_bb"), hbond_sr_bb_init)
    scorefxn_cart.set_weight(score_type_from_name("hbond_lr_bb"), hbond_lr_bb_init)
    scorefxn_cart.set_weight(score_type_from_name("hbond_bb_sc"), hbond_bb_sc_init)
    scorefxn_cart.set_weight(score_type_from_name("hbond_sc"), hbond_sc_init)
    scorefxn_cart.set_weight(score_type_from_name("dslf_fa13"), dslf_fa13_init)
    scorefxn_cart.set_weight(score_type_from_name("omega"), omega_init)
    scorefxn_cart.set_weight(score_type_from_name("fa_dun"), fa_dun_init)
    scorefxn_cart.set_weight(score_type_from_name("p_aa_pp"), p_aa_pp_init)
    scorefxn_cart.set_weight(score_type_from_name("yhh_planarity"), yhh_planarity_init)
    scorefxn_cart.set_weight(score_type_from_name("ref"), ref_init)
    scorefxn_cart.set_weight(score_type_from_name("rama_prepro"), rama_prepro_init)
    scorefxn_cart.set_weight(rosetta.core.scoring.cart_bonded, cart_bonded_init * 1.5) 
    scorefxn_cart.set_weight(rosetta.core.scoring.dihedral_constraint, dihedral_init)
    scorefxn_cart.set_weight(rosetta.core.scoring.atom_pair_constraint, atom_pair_init)
    scorefxn_cart.set_weight(score_type_from_name("rna_torsion"), rna_torsion_init)
    scorefxn_cart.set_weight(score_type_from_name("rna_sugar_close"), rna_sugar_close_init * 2.0)
    scorefxn_cart.set_weight(score_type_from_name("dna_bb_torsion"), dna_bb_torsion_init)
    scorefxn_cart.set_weight(score_type_from_name("dna_sugar_close"), dna_sugar_close_init * 2.0)
    scorefxn_cart.set_weight(rosetta.core.scoring.fa_stack, fa_stack_init)

    tf = TaskFactory()
    tf.push_back(RestrictToRepacking())
    tf.push_back(IncludeCurrent())
    packer_task = tf.create_task_and_apply_taskoperations(pose)
    pack_mover = PackRotamersMover(scorefxn_cart, packer_task)
    pack_mover.apply(pose)

    pose.update_residue_neighbors()

    min_mover = MinMover()
    min_mover.movemap(movemap)
    min_mover.score_function(scorefxn_cart)
    min_mover.min_type('lbfgs_armijo_nonmonotone') 
    min_mover.max_iter(250)
    min_mover.tolerance(0.005)
    min_mover.cartesian(True)
    min_mover.apply(pose)

    pose.update_residue_neighbors()

    tf = TaskFactory()
    tf.push_back(RestrictToRepacking())
    tf.push_back(IncludeCurrent())
    packer_task = tf.create_task_and_apply_taskoperations(pose)
    pack_mover = PackRotamersMover(scorefxn_cart, packer_task)
    pack_mover.apply(pose)

    current_score = scorefxn_cart(pose)

    pose.energies().clear()   
    score_cart= scorefxn_cart(pose)  
    print(" ")   
    print(f"Late refinement cycle {cycle+1} score = {score_cart:.3f}")
    print(" ")  
    terms = scorefxn_cart.get_nonzero_weighted_scoretypes()
    print("Score terms:")
    for term in terms:
        val = pose.energies().total_energies()[term]
        print(f"{term}: {val:.3f}")

    current_score = scorefxn_cart(pose)
    prev_score = current_score

    print(" ")

# ----------------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #

print("=== STRUCTURE POLISHING ===") 
print(" ")
movemap = MoveMap()
movemap.set_bb(True)
movemap.set_chi(True)
movemap.set_jump(True)

pose.remove_constraints()
pose.update_residue_neighbors()
scorefxn_cart = reset_score_weights(scorefxn_cart)

scale = adaptive_constraint_scale(current_score, prev_score)
add_conditional_constraints(pose, interface_residues_dict, scale)
print(" ")

scorefxn_cart.set_weight(score_type_from_name("fa_atr"), fa_atr_init)
scorefxn_cart.set_weight(score_type_from_name("fa_rep"), fa_rep_init * 1.5) 
scorefxn_cart.set_weight(score_type_from_name("fa_sol"), fa_sol_init)
scorefxn_cart.set_weight(score_type_from_name("fa_intra_rep"), fa_intra_rep_init)
scorefxn_cart.set_weight(score_type_from_name("fa_intra_sol_xover4"), fa_intra_sol_xover4_init)
scorefxn_cart.set_weight(score_type_from_name("lk_ball_wtd"), lk_ball_wtd_init)
scorefxn_cart.set_weight(score_type_from_name("fa_elec"), fa_elec_init)
scorefxn_cart.set_weight(score_type_from_name("hbond_sr_bb"), hbond_sr_bb_init)
scorefxn_cart.set_weight(score_type_from_name("hbond_lr_bb"), hbond_lr_bb_init)
scorefxn_cart.set_weight(score_type_from_name("hbond_bb_sc"), hbond_bb_sc_init)
scorefxn_cart.set_weight(score_type_from_name("hbond_sc"), hbond_sc_init)
scorefxn_cart.set_weight(score_type_from_name("dslf_fa13"), dslf_fa13_init)
scorefxn_cart.set_weight(score_type_from_name("omega"), omega_init)
scorefxn_cart.set_weight(score_type_from_name("fa_dun"), fa_dun_init)
scorefxn_cart.set_weight(score_type_from_name("p_aa_pp"), p_aa_pp_init)
scorefxn_cart.set_weight(score_type_from_name("yhh_planarity"), yhh_planarity_init)
scorefxn_cart.set_weight(score_type_from_name("ref"), ref_init)
scorefxn_cart.set_weight(score_type_from_name("rama_prepro"), rama_prepro_init)
scorefxn_cart.set_weight(rosetta.core.scoring.cart_bonded, cart_bonded_init * 1.5) 
scorefxn_cart.set_weight(rosetta.core.scoring.dihedral_constraint, dihedral_init)
scorefxn_cart.set_weight(rosetta.core.scoring.atom_pair_constraint, atom_pair_init)
scorefxn_cart.set_weight(score_type_from_name("rna_torsion"), rna_torsion_init)
scorefxn_cart.set_weight(score_type_from_name("rna_sugar_close"), rna_sugar_close_init * 2.0)
scorefxn_cart.set_weight(score_type_from_name("dna_bb_torsion"), dna_bb_torsion_init)
scorefxn_cart.set_weight(score_type_from_name("dna_sugar_close"), dna_sugar_close_init * 2.0)
scorefxn_cart.set_weight(rosetta.core.scoring.fa_stack, fa_stack_init)
    
min_mover = MinMover()
min_mover.movemap(movemap)
min_mover.score_function(scorefxn_cart)
min_mover.min_type('lbfgs_armijo_nonmonotone') 
min_mover.max_iter(1000)
min_mover.tolerance(0.0001)
min_mover.cartesian(True)
min_mover.apply(pose)

# ----------------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #

pose.energies().clear()   
score_cart= scorefxn_cart(pose) 
print(" ")  
print("=== OUTPUT POSE ===")
print(" ")
print(f"Output score = {score_cart:.3f}")
print(" ")
terms = scorefxn_cart.get_nonzero_weighted_scoretypes()
print("Score terms:")
for term in terms:
    val = pose.energies().total_energies()[term]
    print(f"{term}: {val:.3f}")
print(" ")

pose.dump_pdb(output_pdb)
print(f"Saved relaxed structure to: {output_pdb}")
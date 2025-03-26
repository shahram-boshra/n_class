import torch
import torch.nn.functional as F
from rdkit import Chem
from typing import Dict

class OneHotEncoder:
    """Encodes values as one-hot vectors."""
    @staticmethod
    def encode(value: int, range_size: int) -> torch.Tensor:
        """Creates a one-hot encoded tensor."""
        return F.one_hot(torch.tensor(value), num_classes=range_size).float()

class MoleculeFeatureExtractor:
    """Extracts features from molecules."""
    @staticmethod
    def get_atomic_features_one_hot(mol: Chem.Mol, feature_ranges: Dict[str, int]) -> torch.Tensor:
        """Extracts atomic features as one-hot vectors."""
        atomic_features = []
        formal_charges = []
        for atom in mol.GetAtoms():
            atomic_num = atom.GetAtomicNum()
            hybridization = int(atom.GetHybridization())
            degree = atom.GetDegree()
            formal_charge = atom.GetFormalCharge()
            formal_charges.append(formal_charge)
            chiral_tag = int(atom.GetChiralTag())
            implicit_valence = atom.GetImplicitValence()
            num_h = atom.GetTotalNumHs()

            atom_features = [
                OneHotEncoder.encode(atomic_num, feature_ranges["atomic_nums"]),
                OneHotEncoder.encode(hybridization, feature_ranges["hybridizations"]),
                OneHotEncoder.encode(degree, feature_ranges["degrees"]),
                OneHotEncoder.encode(MoleculeFeatureExtractor.shift_formal_charge(formal_charge, formal_charges), feature_ranges["formal_charges"]),
                OneHotEncoder.encode(chiral_tag, feature_ranges["chiral_tags"]),
                OneHotEncoder.encode(implicit_valence, feature_ranges["implicit_valences"]),
                OneHotEncoder.encode(num_h, feature_ranges["num_h_list"]),
            ]
            atomic_features.append(torch.cat(atom_features))
        return torch.stack(atomic_features)

    @staticmethod
    def get_bond_features_one_hot(mol: Chem.Mol) -> torch.Tensor:
        """Extracts bond features as one-hot vectors."""
        bond_features = []
        for bond in mol.GetBonds():
            bond_type = bond.GetBondType()
            is_conjugated = bond.GetIsConjugated()
            is_in_ring = bond.IsInRing()
            bond_stereo = int(bond.GetStereo())

            bond_features.append(torch.tensor([
                1 if bond_type == Chem.BondType.SINGLE else 0,
                1 if bond_type == Chem.BondType.DOUBLE else 0,
                1 if bond_type == Chem.BondType.TRIPLE else 0,
                1 if bond_type == Chem.BondType.AROMATIC else 0,
                1 if is_conjugated else 0,
                1 if is_in_ring else 0,
                bond_stereo,
            ], dtype=torch.float))
        return torch.stack(bond_features) if bond_features else torch.empty(0, 7)

    @staticmethod
    def shift_formal_charge(formal_charge, formal_charges):
        """Shifts formal charges to ensure non-negative values for one-hot encoding."""
        min_formal_charge = min(formal_charges)
        offset = abs(min_formal_charge) if min_formal_charge < 0 else 0
        return formal_charge + offset

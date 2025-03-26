import torch
import torch_geometric.data
from rdkit import Chem
from typing import Dict, Optional
import logging

from feature_extractor import MoleculeFeatureExtractor
from data_loader import RawDataLoader, ProcessedDataHandler
from config import Config
from rdkit_processor import process_molecule

logger = logging.getLogger(__name__)

class MoleculeProcessor:
    """Processes molecules into graph data."""
    def __init__(self, root: str, directed: bool = False, rdkit_config: Config = None):
        """Initializes the processor with root directory, directionality, and RDKit configuration."""
        self.root = root
        self.directed = directed
        self.raw_loader = RawDataLoader(root)
        self.data_handler = ProcessedDataHandler(root)
        self.feature_ranges = self.data_handler.load_feature_ranges(self.raw_loader.load_raw_files())
        self.rdkit_config = rdkit_config
        logger.debug(f"MoleculeProcessor initialized with root: {root}, directed: {directed}")

    def load_molecule(self, mol_path: str) -> Optional[Chem.Mol]:
        """Loads a molecule from a file path."""
        logger.debug(f"Loading molecule from {mol_path}")
        return process_molecule(mol_path, config=self.rdkit_config)

    def process_molecule(self, mol: Chem.Mol) -> Optional[torch_geometric.data.Data]:
        """Processes a molecule into a graph data object."""
        logger.debug("Processing molecule to graph.")
        return self.mol_to_graph(mol, self.feature_ranges, self.directed)

    @staticmethod
    def mol_to_graph(mol: Chem.Mol, feature_ranges: Dict[str, int], directed: bool = False) -> Optional[torch_geometric.data.Data]:
        """Converts an RDKit molecule to a PyTorch Geometric Data object."""
        if mol is None:
            logger.warning("Attempted to convert a None molecule to graph.")
            return None
        x = MoleculeFeatureExtractor.get_atomic_features_one_hot(mol, feature_ranges)
        edge_attr_onehot = MoleculeFeatureExtractor.get_bond_features_one_hot(mol)
        edge_index_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_index_list.append([i, j])
            if not directed:
                edge_index_list.append([j, i])
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        pos = mol.GetConformer().GetPositions()
        pos = torch.tensor(pos, dtype=torch.float)
        graph = torch_geometric.data.Data(x=x, edge_index=edge_index, edge_attr=edge_attr_onehot, pos=pos)
        return graph

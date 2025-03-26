import pandas as pd
import torch
import torch_geometric.data
import json
import os
from pathlib import Path
from typing import List, Dict, Optional
import logging
from rdkit import Chem
import diskcache
import numpy as np

from config import Config
from rdkit_processor import process_molecule
from feature_extractor import MoleculeFeatureExtractor

logger = logging.getLogger(__name__)

class RawDataLoader:
    """Loads raw molecule files."""
    def __init__(self, root: str):
        """Initializes the loader with the root directory."""
        self.root = Path(root)
        self.mol_dir = self.root / 'Mols'
        logger.debug(f"RawDataLoader initialized with root: {root}")

    def load_raw_files(self) -> List[str]:
        """Loads raw MOL file paths."""
        files = [str(f) for f in self.mol_dir.glob('*.mol')]
        logger.debug(f"Loaded {len(files)} raw files.")
        return files

class FeatureRangeCalculator:
    """Calculates feature ranges from molecule data."""
    @staticmethod
    def calculate_feature_ranges(mol_paths: List[str]) -> Dict[str, int]:
        """Calculates feature ranges based on the provided molecule file paths."""
        data = []
        formal_charges = []

        for mol_path in mol_paths:
            mol = Chem.MolFromMolFile(mol_path)
            if mol is not None:
                for atom in mol.GetAtoms():
                    data.append({
                        "atomic_num": atom.GetAtomicNum(),
                        "hybridization": int(atom.GetHybridization()),
                        "degree": atom.GetDegree(),
                        "formal_charge": atom.GetFormalCharge(),
                        "chiral_tag": int(atom.GetChiralTag()),
                        "implicit_valence": atom.GetImplicitValence(),
                        "num_h": atom.GetTotalNumHs()
                    })
                    formal_charges.append(atom.GetFormalCharge())

        df = pd.DataFrame(data)

        feature_ranges = {
            "atomic_nums": int(df["atomic_num"].max()) + 1 if not df.empty else 1,
            "hybridizations": int(df["hybridization"].max()) + 1 if not df.empty else 1,
            "degrees": int(df["degree"].max()) + 1 if not df.empty else 1,
            "formal_charges": max(formal_charges) + abs(min(formal_charges)) + 1 if formal_charges else 1,
            "chiral_tags": int(df["chiral_tag"].max()) + 1 if not df.empty else 1,
            "implicit_valences": int(df["implicit_valence"].max()) + 1 if not df.empty else 1,
            "num_h_list": int(df["num_h"].max()) + 1 if not df.empty else 1,
        }
        logger.debug(f"Calculated feature ranges: {feature_ranges}")
        return feature_ranges

class ProcessedDataHandler:
    """Handles saving and loading of processed data."""
    def __init__(self, root: str, use_cache: bool = True, cache_expiry: int = 3600):
        """Initializes the handler with root directory, cache settings, and cache expiry."""
        self.root = Path(root)
        self.use_cache = use_cache
        self.cache_dir = self.root / 'processed_graphs_cache'
        self.processed_dir = self.root / 'processed'
        self.feature_ranges_path = self.root / 'feature_ranges.json'
        self.cache_expiry = cache_expiry
        if self.use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
            self.cache = diskcache.Cache(self.cache_dir)  # Initialize diskcache
        else:
            self.cache = None
        os.makedirs(self.processed_dir, exist_ok=True)
        logger.debug(f"ProcessedDataHandler initialized with root: {root}, use_cache: {use_cache}")

    def save_feature_ranges(self, feature_ranges: Dict[str, int]) -> None:
        """Saves feature ranges to a JSON file."""
        with open(self.feature_ranges_path, 'w') as f:
            json.dump(feature_ranges, f)
        logger.debug(f"Saved feature ranges to {self.feature_ranges_path}")

    def load_feature_ranges(self, raw_file_names: List[str]) -> Dict[str, int]:
        """Loads feature ranges from a JSON file or calculates them if not present."""
        if os.path.exists(self.feature_ranges_path):
            with open(self.feature_ranges_path, 'r') as f:
                ranges = json.load(f)
                logger.debug(f"Loaded feature ranges from {self.feature_ranges_path}")
                return ranges
        else:
            feature_ranges = FeatureRangeCalculator.calculate_feature_ranges(raw_file_names)
            self.save_feature_ranges(feature_ranges)
            return feature_ranges

    def save_graph(self, graph: torch_geometric.data.Data, file_name: str) -> None:
        """Saves a graph data object to file or cache."""
        processed_path = self.processed_dir / file_name
        if self.use_cache and self.cache:
            self.cache.set(file_name, graph, expire=self.cache_expiry)
            logger.debug(f"Saved graph to cache: {file_name}")
        else:
            torch.save(graph, processed_path)
            logger.debug(f"Saved graph to file: {processed_path}")

    def load_graph(self, file_name: str) -> Optional[torch_geometric.data.Data]:
        """Loads a graph data object from file or cache."""
        processed_path = self.processed_dir / file_name
        if self.use_cache and self.cache:
            graph = self.cache.get(file_name)
            if graph is not None:
                logger.debug(f"Loaded graph from cache: {file_name}")
                return graph
            else:
                logger.info(f"Cache miss for {file_name}, recalculating.")
                return None
        else:
            try:
                torch.serialization.add_safe_globals([
                    torch_geometric.data.data.DataEdgeAttr,
                    torch_geometric.data.data.DataTensorAttr,
                    torch_geometric.data.storage.GlobalStorage
                ])
                graph = torch.load(processed_path)
                logger.debug(f"Loaded graph from file: {processed_path}")
                return graph
            except FileNotFoundError:
                logger.warning(f"Graph file not found: {processed_path}")
                return None

class DatasetError(Exception):
    '''Base exception for dataset related errors.'''
    pass

class EmptyDatasetError(DatasetError):
    '''Raised when the loaded CSV file is empty.'''
    pass

class ColumnNotFoundError(DatasetError):
    '''Raised when a specified column is not found in the CSV.'''
    def __init__(self, column_name: str) -> None:
        """Initialize with the missing column name."""
        super().__init__(f"Column '{column_name}' not found in CSV.")

class DataProcessor:
    """Processes molecule data and converts it to graph data."""
    def __init__(self, root: str, use_cache: bool = True, rdkit_config: Config = None):
        """Initializes the processor with root directory, cache settings, and RDKit configuration."""
        self.root = root
        self.use_cache = use_cache
        self.raw_loader = RawDataLoader(root)
        self.data_handler = ProcessedDataHandler(root, use_cache)
        self.molecule_processor = MoleculeProcessor(root, rdkit_config=rdkit_config)
        self.rdkit_config = rdkit_config
        logger.debug(f"DataProcessor initialized with root: {root}, use_cache: {use_cache}")

    def _load_molecule_data(self, raw_path: str) -> Optional[Chem.Mol]:
        """Loads a molecule from a file path."""
        logger.debug(f"Loading molecule data from {raw_path}")
        return self.molecule_processor.load_molecule(raw_path)

    def _process_molecule_graph(self, mol: Chem.Mol, mol_name: str, processed_name: str, node_target_df: pd.DataFrame) -> None:
        """Processes a molecule into a graph data object and saves it."""
        if mol is not None:
            graph = self.molecule_processor.process_molecule(mol)
            if graph is not None:
                try:
                    node_targets = []
                    for atom in mol.GetAtoms():
                        atom_index = atom.GetIdx()
                        node_target = node_target_df.loc[(mol_name, atom_index,)].values
                        node_targets.append(node_target)
                    node_targets = np.array(node_targets)    
                    node_targets = torch.tensor(node_targets, dtype=torch.long).reshape(-1)
                    graph.y = node_targets
                    self.data_handler.save_graph(graph, processed_name)
                    logger.debug(f"Processed and saved graph for {mol_name}")
                except KeyError as e:
                    logger.error(f'Molecule name {mol_name} not found or index error in target file: {e}')
                    raise ColumnNotFoundError(f'Molecule name {mol_name} not found or index error in target file: {e}')
                except Exception as e:
                    logger.error(f'Error saving graph {processed_name}: {e}')
                    raise DatasetError(f'Error saving graph {processed_name}: {e}')
            else:
                logger.warning(f'Trouble processing graph {processed_name} or empty graph')
        else:
            logger.warning(f'MOL file {mol_name}.mol not found or corrupted, or rdkit processing failed')

    def process_data(self, node_target_df: pd.DataFrame) -> None:
        """Processes all molecule data and saves the corresponding graph data objects."""
        raw_file_names = self.raw_loader.load_raw_files()
        if not raw_file_names:
            logger.error("No MOL files found in the specified directory.")
            raise EmptyDatasetError("No MOL files found in the specified directory.")
        for raw_path, processed_name in zip(raw_file_names, [Path(f).name.replace('.mol', '.pt') for f in raw_file_names]):
            mol_name = Path(raw_path).name.replace('.mol', '')
            if self.use_cache and self.data_handler.cache and self.data_handler.cache.get(processed_name):
                logger.debug(f"Graph {processed_name} found in cache, skipping processing.")
                continue
            mol = self._load_molecule_data(raw_path)
            self._process_molecule_graph(mol, mol_name, processed_name, node_target_df)

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


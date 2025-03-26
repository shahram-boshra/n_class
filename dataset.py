import torch
import torch_geometric.data
import torch_geometric.transforms
import pandas as pd
from pathlib import Path
import logging
from data_loader import DataProcessor
from config import Config

logger = logging.getLogger(__name__)

class MGDataset(torch_geometric.data.Dataset):
    """Dataset for molecule graph data."""
    def __init__(self, root, directed=False, transform=None, node_target_csv='targets_n_class.csv', use_cache=True, rdkit_config: Config = None):
        """Initializes the dataset with root directory, directionality, transform, target CSV, cache settings, and RDKit configuration."""
        self.directed = directed
        try:
            self.node_target_df = pd.read_csv(Path(root) / node_target_csv, index_col=['MoleculeName', 'atom_index',])
            logger.info("DataFrame loaded successfully.")
            if self.node_target_df.empty:
                logger.error("Target CSV file is empty.")
                raise EmptyDatasetError("Target CSV file is empty.")
        except FileNotFoundError:
            logger.error(f"Target CSV file not found: {target_csv}")
            raise DatasetError(f"Target CSV file not found: {target_csv}")
        except Exception as e:
            logger.error(f"Error loading target CSV: {e}")
            raise DatasetError(f"Error loading target CSV: {e}")
        self.use_cache = use_cache
        self.root = root
        self.data_processor = DataProcessor(root, use_cache, rdkit_config)  # DataProcessor is now Created here.
        self.pre_transform = torch_geometric.transforms.Compose([torch_geometric.transforms.NormalizeFeatures(), torch_geometric.transforms.AddSelfLoops(), torch_geometric.transforms.Distance()])
        super().__init__(root, transform)
        self.transform_list = torch_geometric.transforms.Compose([torch_geometric.transforms.RandomRotate(degrees=180), torch_geometric.transforms.RandomScale((0.9, 1.1)), torch_geometric.transforms.RandomJitter(0.01), torch_geometric.transforms.RandomFlip(0), ])
        self.data_processor.process_data(self.node_target_df)  # Process data is now called here, and the node_target_df is passed.
        self.data_list = self._load_and_transform_data()
        logger.debug(f"MGDataset initialized with root: {root}, directed: {directed}, use_cache: {use_cache}")

    def _load_and_transform_data(self):
        """Loads and transforms graph data objects."""
        data_list = []
        for processed_file_name in self.processed_file_names:
            data = self.data_processor.data_handler.load_graph(processed_file_name)
            if data is not None:
                data = self.transform_list(data)
                data_list.append(data)
            else:
                logger.warning(f"Skipping molecule {processed_file_name} due to missing data.")
        logger.debug(f"Loaded and transformed {len(data_list)} data points.")
        return data_list

    @property
    def raw_file_names(self):
        """Returns a list of raw file names."""
        return self.data_processor.raw_loader.load_raw_files()

    @property
    def processed_file_names(self):
        """Returns a list of processed file names."""
        return [Path(f).name.replace('.mol', '.pt') for f in self.raw_file_names]

    def download(self):
        """Placeholder for download functionality."""
        pass

    def process(self):
        """Placeholder for processing functionality."""
        pass  # process data is now done in the init function, and in the DataProcessor class.

    def len(self):
        """Returns the length of the dataset."""
        return len(self.data_list)

    def get(self, idx):
        """Returns the data object at the specified index."""
        return self.data_list[idx]

    def get_molecule_name(self, idx: int) -> str:
        """Returns the molecule name at the specified index."""
        processed_file_name = self.processed_file_names[idx]
        return Path(processed_file_name).stem

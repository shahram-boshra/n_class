from rdkit import Chem
from rdkit.Chem import AllChem
from typing import Callable, Optional
import logging
import functools
import enum

from config import Config, RDKitStep

logger = logging.getLogger(__name__)



class Hydrogenator:
    """Adds hydrogens to molecules."""
    def __call__(self, mol: Chem.Mol) -> Chem.Mol:
        logger.debug("Hydrogenating molecule.")
        return Chem.AddHs(mol, addCoords=True)

class Sanitizer:
    """Sanitizes molecules."""
    def __call__(self, mol: Chem.Mol) -> Chem.Mol:
        logger.debug("Sanitizing molecule.")
        Chem.SanitizeMol(mol)
        return mol

class Kekulizer:
    """Kekulizes molecules."""
    def __call__(self, mol: Chem.Mol) -> Chem.Mol:
        logger.debug("Kekulizing molecule.")
        Chem.Kekulize(mol)
        return mol

class Embedder:
    """Embeds molecules using ETKDGv2."""
    def __call__(self, mol: Chem.Mol) -> Chem.Mol:
        logger.debug("Embedding molecule.")
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv2())
        return mol

class Optimizer:
    """Optimizes molecules using MMFF."""
    def __call__(self, mol: Chem.Mol) -> Chem.Mol:
        logger.debug("Optimizing molecule.")
        AllChem.MMFFOptimizeMolecule(mol)
        return mol

RDKIT_STEPS: dict[RDKitStep, Callable[[Chem.Mol], Chem.Mol]] = {
    RDKitStep.HYDROGENATE: Hydrogenator(),
    RDKitStep.SANITIZE: Sanitizer(),
    RDKitStep.KEKULIZE: Kekulizer(),
    RDKitStep.EMBED: Embedder(),
    RDKitStep.OPTIMIZE: Optimizer(),
}

def compose(*functions):
    """Composes multiple functions into a single pipeline."""
    return functools.reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)

class RDKitProcessingError(Exception):
    """Base class for RDKit processing exceptions."""
    pass

class RDKitKekulizeError(RDKitProcessingError):
    """Exception for Kekulization errors."""
    pass

class RDKitMoleculeProcessor:
    """Processes RDKit molecules with a sequence of steps defined by configuration using an explicit pipeline."""

    def __init__(self, config: 'Config'):
        """Initializes the processor with the given configuration."""
        self.config = config.rdkit_processing
        self.pipeline = self._create_pipeline()
        logger.debug("RDKitMoleculeProcessor initialized.")

    def _create_pipeline(self) -> Callable[[Chem.Mol], Optional[Chem.Mol]]:
        """Creates the processing pipeline based on the configuration using a functional approach."""
        steps = [RDKIT_STEPS[step] for step in self.config.steps]
        return compose(*steps)

    def process(self, mol: Optional[Chem.Mol]) -> Optional[Chem.Mol]:
        """Processes a molecule through the defined steps using the pipeline."""
        if mol is None:
            logger.warning("Attempted to process a None molecule.")
            return None
        try:
            return self.pipeline(mol)
        except Chem.rdchem.KekulizeException as e:
            logger.error(f"Kekulization error: {e}")
            raise RDKitKekulizeError(f"Kekulization error: {e}")
        except Exception as e:
            logger.error(f"Error processing molecule: {e}")
            raise RDKitProcessingError(f"Error processing molecule: {e}")

def create_configurable_rdkit_processor(config: 'Config') -> RDKitMoleculeProcessor:
    """Creates a configurable RDKit processor."""
    return RDKitMoleculeProcessor(config)

def process_molecule(mol_path:str, config: 'Config') -> Optional[Chem.Mol]:
    """Processes a molecule using a specified or configurable processor."""
    processor = create_configurable_rdkit_processor(config)
    mol = Chem.MolFromMolFile(mol_path)
    return processor.process(mol)

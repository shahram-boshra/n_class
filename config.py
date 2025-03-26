import yaml
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
import enum
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class RDKitStep(enum.Enum):
    """Enumeration of available RDKit processing steps."""
    HYDROGENATE = "hydrogenate"
    SANITIZE = "sanitize"
    KEKULIZE = "kekulize"
    EMBED = "embed"
    OPTIMIZE = "optimize"

class RDKitConfig(BaseModel):
    """Configuration for RDKit molecule processing."""
    steps: List[RDKitStep] = Field(
        default=[RDKitStep.HYDROGENATE, RDKitStep.SANITIZE, RDKitStep.KEKULIZE, RDKitStep.EMBED, RDKitStep.OPTIMIZE],
        description="List of RDKit processing steps.",
    )

    @field_validator("steps")
    def validate_steps(cls, v):
        """Validates the list of RDKit steps."""
        valid_steps = list(RDKitStep)
        for step in v:
            if step not in valid_steps:
                raise ValueError(f"Invalid RDKit step: {step}. Valid steps are: {[s.value for s in valid_steps]}")
        return v

class DataConfig(BaseModel):
    """Configuration for data loading."""
    root_dir: str = Field(..., description="Root directory of the dataset.")
    node_target_csv: str = Field(..., description="Path to the target CSV file.")
    use_cache: bool = Field(True, description="Whether to use cached processed data.")
    train_split: float = Field(..., description="Ratio of the dataset to use for training.")
    valid_split: float = Field(..., description="Ratio of the dataset to use for validation.")

    @field_validator("root_dir")
    def validate_root_dir(cls, v):
        """Validates the root directory."""
        if not Path(v).is_dir():
            raise ValueError(f"Root directory does not exist: {v}")
        return v

    @field_validator("node_target_csv")
    def validate_target_csv(cls, v, info: 'pydantic_core.ValidationInfo'):
        """Validates the target CSV file path."""
        root_dir = info.data['root_dir']
        target_path = Path(root_dir) / v
        if not target_path.is_file():
            raise ValueError(f"Target CSV file does not exist: {target_path}")
        return v

    @field_validator("train_split", "valid_split")
    def validate_splits(cls, v):
        """Validates the split ratios."""
        if not 0 < v < 1:
            raise ValueError("Split ratios must be between 0 and 1.")
        return v

    @field_validator("train_split")
    def validate_split_sum(cls, train_split, info: 'pydantic_core.ValidationInfo'):
        """Validates the sum of train and validation split ratios."""
        valid_split = info.data.get("valid_split")
        if valid_split is not None and train_split + valid_split >= 1:
            raise ValueError("train_split + valid_split must be less than 1.")
        return train_split

class ModelConfig(BaseModel):
    """Configuration for the model."""
    batch_size: int = Field(32, description="Batch size for training.")
    learning_rate: float = Field(0.001, description="Learning rate for the optimizer.")
    weight_decay: float = Field(0.0001, description="Weight decay for the optimizer.")
    step_size: int = Field(50, description="Step size for the learning rate scheduler.")
    gamma: float = Field(0.5, description="Gamma for the learning rate scheduler.")
    reduce_lr_factor: float = Field(0.5, description="Factor for reducing learning rate on plateau.")
    reduce_lr_patience: int = Field(10, description="Patience for reducing learning rate on plateau.")
    early_stopping_patience: int = Field(20, description="Patience for early stopping.")
    early_stopping_delta: float = Field(0.001, description="Minimum change in validation loss to qualify as an improvement.")
    l1_regularization_lambda: float = Field(0.001, description="Lambda for L1 regularization.")
    first_layer_type: Optional[str] = Field("custom_mp", description="Type of the first layer.")
    hidden_channels: int = Field(256, description="Number of hidden channels in the model.")
    second_layer_type: Optional[str] = Field("custom_mp", description="Type of the second layer.")
    dropout_rate: float = Field(0.5, description="Dropout rate in the model.")

class Config(BaseModel):
    """Main configuration class."""
    rdkit_processing: RDKitConfig = Field(default_factory=RDKitConfig, description="RDKit processing configuration.")
    data: DataConfig = Field(..., description="Data loading configuration.")
    model: ModelConfig = Field(..., description="Model training configuration.")

    @classmethod
    def from_yaml(cls, config_path: str):
        """Loads and validates configuration from a YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

def load_config(config_path: str) -> 'Config':
    """Loads and validates configuration from a YAML file."""
    try:
        config = Config.from_yaml(config_path)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except (ValueError, FileNotFoundError) as e:
        logger.error(f"Error loading configuration: {e}")
        raise (f"Error loading configuration: {e}")




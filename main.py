from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch_geometric.loader import DataLoader
import numpy as np
import logging

from config import Config, load_config
from dataset import MGDataset
from early_stopping import EarlyStopping
from models import MGModel
from training import Trainer, Plot, split_dataset


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[logging.FileHandler('app.log'), logging.StreamHandler()])
    
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    from pathlib import Path
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
    from torch_geometric.loader import DataLoader
    import numpy as np

    data_dir = Path('C:/Chem_Data')

    config_path = data_dir / 'config.yaml'
    config = load_config(config_path)

    dataset = MGDataset(root=config.data.root_dir, node_target_csv=config.data.node_target_csv, use_cache=config.data.use_cache, rdkit_config=config)

    in_channels = dataset[0].x.shape[1]
    out_channels = 2

    torch.manual_seed(11)

    train_dataset, valid_dataset, test_dataset = split_dataset(dataset, config.data.train_split, config.data.valid_split, dataset.node_target_df)

    train_loader = DataLoader(train_dataset, batch_size=config.model.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config.model.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.model.batch_size, shuffle=False)

    model = MGModel(
        in_channels=in_channels,
        out_channels=out_channels,
        first_layer_type=config.model.first_layer_type,
        second_layer_type=config.model.second_layer_type,
        hidden_channels=config.model.hidden_channels,
        dropout_rate=config.model.dropout_rate,
        gat_heads=1,
        transformer_heads=1,
    )

    logger.info(f'Model Architecture {model}')

    criterion = nn.NLLLoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=config.model.learning_rate, weight_decay=config.model.weight_decay)

    step_lr = StepLR(optimizer, step_size=config.model.step_size, gamma=config.model.gamma)
    red_lr = ReduceLROnPlateau(optimizer, mode='min', factor=config.model.reduce_lr_factor, patience=config.model.reduce_lr_patience)

    early_stopping = EarlyStopping(patience=config.model.early_stopping_patience, verbose=True, delta=config.model.early_stopping_delta)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    trainer = Trainer(model, criterion, optimizer, step_lr, red_lr, early_stopping, config, device)
    train_losses, valid_losses, maes, mses, r2s, explained_variances = trainer.train_and_validate(train_loader, valid_loader)
    test_loss, test_mae, test_mse, test_r2, test_explained_variance, test_targets, test_predictions = trainer.test_epoch(test_loader, return_predictions=True)

    # Save test targets and predictions
    np.save('test_targets.npy', np.array(test_targets))
    np.save('test_predictions.npy', np.array(test_predictions))

    logger.info(f'Test Loss: {test_loss:.4f}, MAE: {test_mae:.4f}, MSE: {test_mse:.4f}, R2: {test_r2:.4f}, Explained Variance: {test_explained_variance:.4f}')

    # Plotting
    Plot.plot_losses(train_losses, valid_losses)
    Plot.plot_metrics_vs_epoch(maes, mses, r2s, explained_variances)


    

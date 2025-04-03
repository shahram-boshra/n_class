import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import torch_geometric
from torch_geometric.loader import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import logging

from config import Config
from dataset import MGDataset
from early_stopping import EarlyStopping
from models import MGModel

logger = logging.getLogger(__name__)

def split_dataset(dataset: torch_geometric.data.Dataset, train_ratio: float, valid_ratio: float, node_target_df: pd.DataFrame):
    """
    Splits a PyTorch Geometric dataset into train, validation, and test sets,
    with stratification based on node_target values.
    """

    indices = list(node_target_df.index)
    target_values = [node_target_df.loc[idx].values[0] for idx in indices]
    target_series = pd.Series(target_values)

    bins = pd.qcut(target_series, q=10, labels=False, duplicates='drop')

    train_indices, temp_indices, _, temp_bins = train_test_split(
        indices, bins, train_size=train_ratio, stratify=bins, random_state=42
    )
    valid_indices, test_indices, _, _ = train_test_split(
        temp_indices, temp_bins, train_size=valid_ratio / (1 - train_ratio), stratify=temp_bins, random_state=42
    )

    train_dataset = torch.utils.data.Subset(dataset, [dataset.processed_file_names.index(mol_name + ".pt") for mol_name, atom_index in train_indices])
    valid_dataset = torch.utils.data.Subset(dataset, [dataset.processed_file_names.index(mol_name + ".pt") for mol_name, atom_index in valid_indices])
    test_dataset = torch.utils.data.Subset(dataset, [dataset.processed_file_names.index(mol_name + ".pt") for mol_name, atom_index in test_indices])

    logger.info(f"Dataset split into train ({len(train_dataset)}), validation ({len(valid_dataset)}), and test ({len(test_dataset)}) sets.")
    return train_dataset, valid_dataset, test_dataset

# Training Loop Class
class TrainingLoop:
    """Encapsulates the training loop logic."""
    def __init__(self, model, criterion, optimizer, step_lr, device, l1_lambda):
        """Initializes the training loop with model, criterion, optimizer, learning rate scheduler, device, and L1 regularization lambda."""
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.step_lr = step_lr
        self.device = device
        self.l1_lambda = l1_lambda
        logger.debug("TrainingLoop initialized.")

    def train_epoch(self, train_loader):
        """Trains the model for one epoch."""
        self.model.train()
        total_loss = 0
        num_nodes = 0

        for batch in train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            node_out, l1_reg = self.model(batch)
            node_target = batch.y
            node_loss = self.criterion(node_out, node_target)
            node_loss += l1_reg * self.l1_lambda
            node_loss.backward()
            self.optimizer.step()
            total_loss += node_loss.item() * batch.num_nodes
            num_nodes += batch.num_nodes
        self.step_lr.step()
        avg_loss = total_loss / num_nodes
        logger.debug(f"Training Epoch Loss: {avg_loss:.4f}")
        return avg_loss

    def validate_epoch(self, valid_loader):
        """Validates the model for one epoch."""
        self.model.eval()
        total_loss = 0
        num_nodes = 0
        all_targets = []
        all_predictions = []
        all_probabilities = []

        with torch.no_grad():
            for batch in valid_loader:
                batch = batch.to(self.device)
                node_out, _ = self.model(batch)
                node_target = batch.y
                node_loss = self.criterion(node_out, node_target)
                total_loss += node_loss.item() * batch.num_nodes
                num_nodes += batch.num_nodes
                all_targets.extend(node_target.cpu().numpy())
                probabilities = torch.softmax(node_out, dim=1).cpu().numpy()
                all_probabilities.extend(probabilities)
                predictions = torch.argmax(node_out, dim=1).cpu().numpy()
                all_predictions.extend(predictions)

        avg_loss = total_loss / num_nodes
        accuracy = accuracy_score(all_targets, all_predictions)
        precision = precision_score(all_targets, all_predictions, average='weighted', zero_division=0)
        recall = recall_score(all_targets, all_predictions, average='weighted', zero_division=0)
        f1 = f1_score(all_targets, all_predictions, average='weighted', zero_division=0)

        try:
            auc = roc_auc_score(all_targets, all_probabilities, multi_class='ovr')
        except ValueError:
            auc = 0.0
            logging.warning("AUC calculation failed. Setting AUC to 0.0")

        logger.debug(f"Validation Epoch Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
        return avg_loss, accuracy, precision, recall, f1, auc

    def test_epoch(self, test_loader, return_predictions=False):
        """Tests the model for one epoch."""
        self.model.eval()
        total_loss = 0
        num_nodes = 0
        all_targets = []
        all_predictions = []
        all_probabilities = []

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)
                node_out, _ = self.model(batch)
                node_target = batch.y
                node_loss = self.criterion(node_out, node_target)
                total_loss += node_loss.item() * batch.num_nodes
                num_nodes += batch.num_nodes
                all_targets.extend(node_target.cpu().numpy())
                probabilities = torch.softmax(node_out, dim=1).cpu().numpy()
                all_probabilities.extend(probabilities)
                predictions = torch.argmax(node_out, dim=1).cpu().numpy()
                all_predictions.extend(predictions)

        avg_loss = total_loss / num_nodes
        accuracy = accuracy_score(all_targets, all_predictions)
        precision = precision_score(all_targets, all_predictions, average='weighted', zero_division=0)
        recall = recall_score(all_targets, all_predictions, average='weighted', zero_division=0)
        f1 = f1_score(all_targets, all_predictions, average='weighted', zero_division=0)

        try:
            auc = roc_auc_score(all_targets, all_probabilities, multi_class='ovr')
        except ValueError:
            auc = 0.0
            logging.warning("AUC calculation failed. Setting AUC to 0.0")

        logger.info(f"Test Epoch Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
        if return_predictions:
            return avg_loss, accuracy, precision, recall, f1, auc, all_targets, all_predictions
        return avg_loss, accuracy, precision, recall, f1, auc

# Trainer Class
class Trainer:
    """Manages the training and validation process."""
    def __init__(self, model, criterion, optimizer, step_lr, red_lr, early_stopping, config, device):
        """Initializes the trainer with model, criterion, optimizer, learning rate schedulers, early stopping, configuration, and device."""
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.step_lr = step_lr
        self.red_lr = red_lr
        self.early_stopping = early_stopping
        self.config = config
        self.device = device
        self.training_loop = TrainingLoop(self.model, self.criterion, self.optimizer, self.step_lr, self.device, self.config.model.l1_regularization_lambda)
        logger.debug("Trainer initialized.")

    def train_and_validate(self, train_loader, valid_loader):
        """Trains and validates the model over multiple epochs."""
        train_losses, valid_losses, accuracies, precisions, recalls, f1s, aucs = [], [], [], [], [], [], []
        for epoch in range(self.config.model.early_stopping_patience * 2):
            train_loss = self.training_loop.train_epoch(train_loader)
            valid_loss, accuracy, precision, recall, f1, auc = self.training_loop.validate_epoch(valid_loader)

            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            accuracies.append(accuracy)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            aucs.append(auc)

            self.red_lr.step(valid_loss)
            self.early_stopping(valid_loss, self.model)
            if self.early_stopping.early_stop:
                logger.info("Early stopping triggered.")
                break
        return train_losses, valid_losses, accuracies, precisions, recalls, f1s, aucs

    def test_epoch(self, test_loader, return_predictions=False):
        """Tests the model for one epoch."""
        return self.training_loop.test_epoch(test_loader, return_predictions)

class Plot:
    """Handles plotting of training and validation metrics."""
    @staticmethod
    def plot_losses(train_losses, valid_losses):
        """Plots training and validation losses."""
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(valid_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        plt.legend()
        plt.show()

    @staticmethod
    def plot_classification_metrics_vs_epoch(accuracies, precisions, recalls, f1s, aucs):
        """Plots classification metrics vs. epoch."""
        epochs = range(1, len(accuracies) + 1)

        plt.figure(figsize=(12, 8))

        plt.subplot(2, 3, 1)
        plt.plot(epochs, accuracies, label='Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs. Epoch')
        plt.legend()

        plt.subplot(2, 3, 2)
        plt.plot(epochs, precisions, label='Precision')
        plt.xlabel('Epoch')
        plt.ylabel('Precision')
        plt.title('Precision vs. Epoch')
        plt.legend()

        plt.subplot(2, 3, 3)
        plt.plot(epochs, recalls, label='Recall')
        plt.xlabel('Epoch')
        plt.ylabel('Recall')
        plt.title('Recall vs. Epoch')
        plt.legend()

        plt.subplot(2, 3, 4)
        plt.plot(epochs, f1s, label='F1 Score')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.title('F1 Score vs. Epoch')
        plt.legend()

        plt.subplot(2, 3, 5)
        plt.plot(epochs, aucs, label='AUC')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.title('AUC vs. Epoch')
        plt.legend()

        plt.tight_layout()
        plt.show()

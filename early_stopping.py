import torch
import logging

logger = logging.getLogger(__name__)

class EarlyStopping:
    """Early stopping mechanism."""
    def __init__(self, patience=7, verbose=False, delta=0.0, path='chk_learn.pt'):
        """Initializes the early stopping mechanism with patience, verbosity, delta, and path."""
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        logger.debug(f"EarlyStopping initialized with patience: {patience}, delta: {delta}")

    def __call__(self, valid_loss, model):
        """Checks if early stopping should be triggered."""
        if self.best_score is None:
            self.best_score = valid_loss
            self.save_model_state(model)
        elif valid_loss < self.best_score - self.delta:
            if self.verbose:
                logger.info(f'Validation Loss improves {self.best_score:.4f}->{valid_loss:.4f}=>$ave model')
            self.best_score = valid_loss
            self.save_model_state(model)
            self.counter = 0
        else:
            self.counter += 1
            logger.info(f'Machine Learning patience ticks {self.counter} from {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

    def save_model_state(self, model):
        """Saves the model state to the specified path."""
        torch.save(model.state_dict(), self.path)
        logger.debug(f"Saved model state to {self.path}")

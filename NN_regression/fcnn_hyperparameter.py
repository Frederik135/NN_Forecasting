


# New FCNN model

# FCNN for hyperparameter tuning
class FCNN_model(nn.Module):
    def __init__(self, seq_length, num_features, hidden_layers, n_out, dropout_prob):
        super(FCNN_model, self).__init__()
        self.seq_length = seq_length
        self.num_features = num_features
        self.layers = nn.ModuleList()
        
        # Input to first hidden layer
        input_size = seq_length * num_features
        for layer_size in hidden_layers:
            self.layers.append(nn.Linear(input_size, layer_size))
            self.layers.append(nn.Dropout(dropout_prob))
            input_size = layer_size
        
        # Output layer
        self.layers.append(nn.Linear(input_size, n_out))

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                x = F.relu(layer(x))
            else:
                x = layer(x)  # Apply dropout
        return x




# Hyperparameter optimization

import optuna
import torch
import torch.nn as nn
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from models import FCNN_model
from config import device, seq_length, architecture, num_features
from preprocessing import train_loader, val_loader, label_scaler
import wandb
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

class StockPredictionModule(LightningModule):
    def __init__(self, model, label_scaler, train_loader, val_loader, test_loader):
        super().__init__()
        self.model = model
        self.label_scaler = label_scaler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = torch.nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        seqs, labels = batch
        y_pred = self(seqs)
        loss = self.criterion(y_pred, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        seqs, labels = batch
        y_pred = self(seqs)
        loss = self.criterion(y_pred, labels)
        labels = labels.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()

        labels_rescaled = self.label_scaler.inverse_transform(labels.reshape(-1, 1)).flatten()
        predictions_rescaled = self.label_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

        r2 = r2_score(labels_rescaled , predictions_rescaled)
        mse = mean_squared_error(labels_rescaled , predictions_rescaled)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(labels_rescaled , predictions_rescaled)
        mape = np.mean(np.abs((labels_rescaled  - predictions_rescaled) / (predictions_rescaled + 1e-8)))
        pct_change_labels = [label - 1 for label in labels_rescaled]
        pct_change_predictions = [prediction - 1 for prediction in predictions_rescaled]
        hit_rate = np.mean(np.sign(pct_change_labels) == np.sign(pct_change_predictions))

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_r2", r2, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_mse", mse, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_rmse", rmse, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_mae", mae, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_mape", mape, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("hit_rate", hit_rate, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {"val_loss": loss, "val_r2": r2, "val_mse": mse, "val_rmse": rmse, "val_mae": mae, "val_mape": mape, "hit_rate": hit_rate}


def objective(trial):
    # Dynamic definition of number and size of layers
    num_hidden_layers = trial.suggest_int('num_hidden_layers', 1, 3)
    hidden_layers = [trial.suggest_int(f'hidden_size_{i}', 32, 128) for i in range(num_hidden_layers)]
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    dropout_prob = trial.suggest_float('dropout_prob', 0.0, 0.5)

    # Create model configuration for the FCNN with dropout
    model_config = {
        "seq_length": seq_length,
        "num_features": num_features,
        "hidden_layers": hidden_layers,
        "n_out": 1,  # Assuming output size of 1 for regression tasks
        "dropout_prob": dropout_prob
    }

    # Configuring W&B
    wandb_config = {
        "architecture": "FCNN",
        "learning_rate": learning_rate,
        "hidden_layers": hidden_layers,
        "dropout_prob": dropout_prob,
        "seq_length": seq_length,
        "epochs": 50
    }

    # Initialize W&B run
    wandb.init(project="fcnn_hyperparameter_test", entity="frederik135", config=wandb_config, reinit=True)

    model = FCNN_model(**model_config).to(device)
    module = StockPredictionModule(model=model, label_scaler=label_scaler, 
                                   train_loader=train_loader, val_loader=val_loader, test_loader=None)
    module.hparams.learning_rate = learning_rate

    if torch.cuda.is_available():
        accelerator = "gpu"
        devices = 1
    elif hasattr(torch, 'has_mps') and torch.backends.mps.is_built():
        accelerator = "mps"
        devices = 1
    else:
        accelerator = None
        devices = None

    wandb_logger = WandbLogger(project="fcnn_hyperparameter_test", log_model="all", config=wandb_config)
    trainer = Trainer(
        logger=wandb_logger,
        max_epochs=70,
        callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=10)],
        accelerator=accelerator,
        devices=devices,
        enable_checkpointing=False,
        enable_progress_bar=False
    )

    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    val_result = trainer.validate(module, dataloaders=val_loader, verbose=False)
    val_loss = val_result[0].get('val_loss', float('inf'))

    wandb.finish()
    return val_loss

# Initialization remains the same
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

print("Best hyperparameters: ", study.best_trial.params)

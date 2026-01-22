
# srun --time=04:00:00 --gres gpu:1 --mem=128G --resv-ports=1 --pty /bin/bash -l

from pathlib import Path
from lightning import pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import pandas as pd
import wandb
import numpy as np
import torch
from chemprop import data, featurizers, models, nn, utils

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define paths
chemprop_dir = Path.cwd()
input_path = chemprop_dir / "data" / "train_smiles.csv" 
descriptors_path = chemprop_dir / "data" / "descriptors.csv"
num_workers = 0 
smiles_column = 'full_smiles' 
target_columns = ['rejection'] 

# Load data
df_input = pd.read_csv(input_path)
print(df_input.head())
smis = df_input.loc[:, smiles_column].values
ys = df_input.loc[:, target_columns].values

# Extract additional descriptors
df_descriptors = pd.read_csv(descriptors_path)
extra_mol_descriptors = np.array(df_descriptors.values)
mols = [utils.make_mol(smi, keep_h=False, add_h=False) for smi in smis]

# Define datapoints
datapoints = [
    data.MoleculeDatapoint(mol, y, x_d=X_d)
    for mol, y, X_d in zip(
        mols,
        ys,
        extra_mol_descriptors,
    )
]

# Split data
train_indices, val_indices, test_indices = data.make_split_indices(mols, "random", (0.8, 0.1, 0.1), num_replicates=3)  # unpack the tuple into three separate lists
train_data, val_data, test_data = data.split_data_by_indices(
    datapoints, train_indices, val_indices, test_indices
)

# Define data loaders
featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

train_dset = data.MoleculeDataset(train_data[0], featurizer)
scaler = train_dset.normalize_targets()
extra_mol_descriptors_scaler = train_dset.normalize_inputs("X_d")

val_dset = data.MoleculeDataset(val_data[0], featurizer)
val_dset.normalize_targets(scaler)
val_dset.normalize_inputs("X_d", extra_mol_descriptors_scaler)

test_dset = data.MoleculeDataset(test_data[0], featurizer)

train_loader = data.build_dataloader(train_dset, num_workers=num_workers)
val_loader = data.build_dataloader(val_dset, num_workers=num_workers, shuffle=False)
test_loader = data.build_dataloader(test_dset, num_workers=num_workers, shuffle=False)

# Define the model
mp = nn.BondMessagePassing()
agg = nn.MeanAggregation()
ffn_input_dim = mp.output_dim + extra_mol_descriptors.shape[1]
output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
ffn = nn.RegressionFFN(n_layers=2, input_dim=ffn_input_dim, output_transform=output_transform, dropout=0.5)
batch_norm = True
print(nn.metrics.MetricRegistry)

metric_list = [
    nn.metrics.RMSE(), 
    nn.metrics.MSE(), 
    nn.metrics.R2Score()
] 

X_d_transform = nn.ScaleTransform.from_standard_scaler(extra_mol_descriptors_scaler)

# Configure optimizer and scheduler
learning_rate = 1e-4
warmup_epochs = 2

ensemble = []
n_models = 1
for _ in range(n_models):
    # Ensure the model is initialized with the full metric list
    model = models.MPNN(
        mp, 
        agg, 
        ffn, 
        metric_list, 
        X_d_transform=X_d_transform
    )
    model.to(device)
    ensemble.append(model)

# 2. Initialize wandb logger
wandb_logger = WandbLogger(project="membranes-chemprop", log_model=True)

# Configure model checkpointing
checkpointing = ModelCheckpoint(
    dirpath="model/checkpoints", 
    filename="best-{epoch}-{val_loss:.2f}",
    monitor="val_loss",
    mode="min",
    save_last=True,
)

trainers = []
for model in ensemble:
    trainer = pl.Trainer(
        logger=wandb_logger,
        enable_checkpointing=True,
        enable_progress_bar=True,
        accelerator="auto",
        devices=1,
        max_epochs=1000, 
        callbacks=[checkpointing],
    )
    trainers.append(trainer)

# 3. Train and then Test
for trainer, model in zip(trainers, ensemble):
    # Fit the model (this logs Train/Val MSE and RMSE)
    trainer.fit(model, train_loader, val_loader)
    
    # Run testing (this logs Test MSE and RMSE to the same WandB run)
    # We use the best model found during training
    trainer.test(model, dataloaders=test_loader, ckpt_path="best", weights_only=False)
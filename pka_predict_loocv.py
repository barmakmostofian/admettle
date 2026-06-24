"""
This script trains a ChemProp v2 model to predict pKa values from SMILES.
It applies a message-passing NN along "directed" bonds, progressively building larger substructures.

Input is a csv file with pKa values and SMILES strings (see below).
Output is a serialized checkpoint for PyTorch Lightning, covering all learned weights,
hyperparameters, and architecture metadata.

A LOO-CV is applied for model training.
"""


from pathlib import Path
import time
import pandas as pd
import numpy as np
from lightning import pytorch as pl
from chemprop import data, featurizers, models, nn


def load_pka_csv(
    csv_path: str,
    id_col: str = "Molecule ID",
    pka_cols: tuple[str, ...] = ("pKa1", "pKa2", "pKa3"),
    smiles_col: str = "canonical isomeric SMILES",
) -> pd.DataFrame:

    """
    Load an experimental pKa CSV of the form:
      Molecule ID, pKa1, pKa2, pKa3, [...], SMILES

    Some molecules have multiple measured pKa values (polyprotic / amphoteric
    fragments), reported in ascending order across pKa1/pKa2/pKa3 (NaN where
    not applicable). This script takes the "pKa of the most basic center", i.e.,
    the MAX non-null value across those columns per row.

    Some molecules also have multiple REPLICATE experimental measurements
    (same Molecule ID appearing on several rows with near-identical values).
    These are averaged together per Molecule ID so that replicates don't get
    treated as independent training examples and don't leak across a
    train/val split.

    Returns a dataframe with columns: mol_id, pKa, smiles
    (one row per unique molecule).
    """

    df = pd.read_csv(csv_path)
    print(f"Loaded {csv_path} with shape {df.shape}")
    print(f"Columns: {list(df.columns)}")

    missing = [c for c in (id_col, smiles_col, *pka_cols) if c not in df.columns]
    if missing:
        raise ValueError(
            f"Expected column(s) not found in {csv_path}: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    # Most-basic pKa per row = max non-null across pKa1/pKa2/pKa3
    df["pKa_most_basic"] = df[list(pka_cols)].max(axis=1, skipna=True)

    n_no_pka = df["pKa_most_basic"].isna().sum()
    if n_no_pka:
        print(f"  -> {n_no_pka} row(s) have no pKa value at all in {pka_cols}; dropping")

    df = df.dropna(subset=["pKa_most_basic", smiles_col])

    # Collapse replicate experimental rows per molecule by averaging
    n_rows_before = len(df)
    grouped = (
        df.groupby(id_col)
        .agg(pKa=("pKa_most_basic", "mean"), smiles=(smiles_col, "first"))
        .reset_index()
        .rename(columns={id_col: "mol_id"})
    )

    n_molecules = len(grouped)
    if n_rows_before > n_molecules:
        print(
            f"  -> collapsed {n_rows_before} row(s) into {n_molecules} unique "
            f"molecule(s) by averaging replicate measurements"
        )

    return grouped[["mol_id", "pKa", "smiles"]]



def loo_splits(df: pd.DataFrame):

    """
    Generate leave-one-out train/val splits from a molecule-level dataframe.
    """

    df = df.reset_index(drop=True)
    n = len(df)

    for i in range(n):
        val_df = df.iloc[[i]].reset_index(drop=True)
        train_df = df.drop(index=i).reset_index(drop=True)

        yield train_df, val_df



def build_datasets_from_dataframe(
    df: pd.DataFrame,
    smiles_col: str = "smiles",
    target_col: str = "pKa",
) -> list[data.MoleculeDatapoint]:

    """
    Convert a dataframe with SMILES + target columns into MoleculeDatapoints --
    a ChemProp-related object.
    """

    smiles = df[smiles_col].tolist()
    targets = df[[target_col]].values
    
    datapoints = [data.MoleculeDatapoint.from_smi(smi, y=target) for smi, target in zip(smiles, targets)]

    return datapoints


def train_pka_model_loo(
    csv_path: str,
    save_dir: str = "./pka_model",
    max_epochs: int = 50,
):

    """
    Train ChemProp pKa regression models using LOO-CV.

    Loads a single experimental pKa csv, collapses replicate measurements
    per molecule, then trains one model per molecule (holding that molecule
    out as the sole validation point each time). Each fold's model checkpoint
    is saved separately. At the end, the held-out predictions across all
    folds are collected and an overall LOO RMSE is reported.

    Returns a dataframe with columns [mol_id, smiles, true_pKa, predicted_pKa]
    covering every molecule's held-out prediction, plus the list of
    checkpoint paths (one per fold).
    """

    full_df = load_pka_csv(csv_path)
    n_molecules = len(full_df)
    print(f"Running LOO-CV over {n_molecules} molecules "
          f"({n_molecules} folds, one model trained per fold)")

    results = []
    ckpt_paths = []
    fold_start_times = []

    for fold_idx, (train_df, val_df) in enumerate(loo_splits(full_df)):
        fold_start = time.time()
        held_out_id = val_df["mol_id"].iloc[0]
        print(f"\n--- Fold {fold_idx + 1}/{n_molecules}: holding out '{held_out_id}' ---")

        if fold_start_times:
            avg_fold_time = sum(fold_start_times) / len(fold_start_times)
            remaining = n_molecules - fold_idx
            eta_minutes = (avg_fold_time * remaining) / 60
            print(f"    (avg {avg_fold_time:.1f}s/fold so far, ~{eta_minutes:.1f} min remaining)")

        # Convert training SMILES + pKa values into ChemProp MoleculeDatapoints.
        train_data = build_datasets_from_dataframe(train_df)

        # Convert the single held-out molecule into a MoleculeDatapoint.
        val_data = build_datasets_from_dataframe(val_df)

        # Instantiate the graph featurizer that converts SMILES into atom/bond feature tensors.
        featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

        # Build the training dataset by applying the featurizer to each MoleculeDatapoint.
        train_dset = data.MoleculeDataset(train_data, featurizer)

        # Build the validation dataset for the single held-out molecule.
        val_dset = data.MoleculeDataset(val_data, featurizer)

        # Scale targets (ChemProp recommends normalizing regression targets).
        # To avoid leaking, the scaler is fit on this fold's training set only, of course. 
        scaler = train_dset.normalize_targets()

        # Apply the same scaler to the validation target (transform only, not refit).
        val_dset.normalize_targets(scaler)

        # Wrap the training dataset in a dataloader for batched iteration during training.
        train_loader = data.build_dataloader(train_dset, shuffle=True)

        # Wrap the validation dataset in a dataloader; no shuffling needed for inference.
        val_loader = data.build_dataloader(val_dset, shuffle=False)

        # Instantiate the directed bond message-passing layer (the graph convolution core).
        mp = nn.BondMessagePassing()

        # Instantiate the mean aggregation layer that pools bond representations into one molecular vector.
        agg = nn.MeanAggregation()

        # Wrap the scaler into an output transform so predictions are returned in original pKa units.
        output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)

        # Instantiate the feed-forward regression head that maps the molecular vector to a pKa scalar.
        ffn = nn.RegressionFFN(output_transform=output_transform)

        # Assemble the full MPNN model from the message-passing, aggregation, and FFN components.
        mpnn = models.MPNN(mp, agg, ffn, batch_norm=True, metrics=[nn.metrics.RMSE()])

        # Define a per-fold subdirectory for saving this fold's checkpoint.
        fold_save_dir = Path(save_dir) / f"fold_{fold_idx:03d}_{held_out_id}"

        # Instantiate the Lightning trainer that manages the training loop, logging, and checkpointing.
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            default_root_dir=str(fold_save_dir),
            enable_checkpointing=True,
            enable_progress_bar=False,
            logger=False,
        )

        # Run the training loop for this fold.
        trainer.fit(mpnn, train_loader, val_loader)

        # Define the path where this fold's final checkpoint will be saved.
        ckpt_path = fold_save_dir / "final_model.ckpt"

        # Save the trained model weights, architecture metadata, and scaler to disk.
        trainer.save_checkpoint(ckpt_path)

        # Record this fold's checkpoint path for return to the caller.
        ckpt_paths.append(ckpt_path)


        # Predict on the held-out molecule with this fold's trained model
        mpnn.eval()

        # Run inference on the held-out molecule.
        predictions = trainer.predict(mpnn, val_loader)

        # Extract the scalar predicted pKa from the batched prediction output.
        predicted_pka = float(np.concatenate([p.numpy() for p in predictions]).flatten()[0])

        # Extract the experimental pKa for the held-out molecule.
        true_pka = float(val_df["pKa"].iloc[0])

        results.append({
            "mol_id": held_out_id,
            "smiles": val_df["smiles"].iloc[0],
            "true_pKa": true_pka,
            "predicted_pKa": predicted_pka,
        })

        fold_elapsed = time.time() - fold_start
        fold_start_times.append(fold_elapsed)
        running_rmse = float(np.sqrt(np.mean(
            (pd.DataFrame(results)["predicted_pKa"] - pd.DataFrame(results)["true_pKa"]) ** 2
        )))

        print(
            f"  true pKa = {true_pka:.2f}, predicted pKa = {predicted_pka:.2f} "
            f"(fold took {fold_elapsed:.1f}s)"
        )
        print(f"  running LOO RMSE after {fold_idx + 1}/{n_molecules} folds: {running_rmse:.3f}")

    results_df = pd.DataFrame(results)

    errors = results_df["predicted_pKa"] - results_df["true_pKa"]

    loo_rmse = float(np.sqrt(np.mean(errors ** 2)))

    loo_mae = float(np.mean(np.abs(errors)))

    print(f"\n=== Leave-one-out CV complete over {n_molecules} folds ===")
    print(f"LOO RMSE: {loo_rmse:.3f}")
    print(f"LOO MAE:  {loo_mae:.3f}")

    return results_df, ckpt_paths



if __name__ == "__main__":

    # EDIT THIS PATH to point at your experimental pKa CSV
    results_df, ckpt_paths = train_pka_model_loo(csv_path="data/experimental_pka_data.csv", max_epochs=50,)

    results_df.to_csv("loo_cv_predictions.csv", index=False)

    print("\nPer-molecule held-out predictions saved to loo_cv_predictions.csv")




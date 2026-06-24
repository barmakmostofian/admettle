# admettle

A growing collection of small-molecule property prediction tools for computational drug discovery,
with a focus on ADMET-relevant endpoints and CNS drug design.

---

## Tools

### 1. CNS MPO Scoring (`cns_mpo.py`)

Computes the **CNS Multiparameter Optimization (MPO)** score for one or more molecules, following
the method of Wager et al. (*ACS Chemical Neuroscience*, 2010). The score is a composite of six
physicochemical properties that collectively predict CNS drug-likeness: cLogP, cLogD, molecular
weight, topological polar surface area (TPSA), hydrogen bond donor count (HBD), and the pKa of
the most basic center.

Each property is passed through a **piecewise linear desirability function** that maps the raw
value to a score between 0 and 1, where 1 reflects the optimal range for CNS penetration. The
six component scores are summed to give a composite MPO score between 0 and 6. A score of 4 or
above is generally considered CNS-favorable.

RDKit is used to compute MW, cLogP, TPSA, and HBD directly from SMILES. cLogD is estimated from
cLogP; pKa uses a placeholder value by default and should be replaced with a measured or
predicted value (e.g. from `pka_predict_loocv.py` below) for production use.

**Input:** a SMILES string, or a list of SMILES strings.

**Output:** per-molecule MPO score and six component scores.



### 2. pKa Prediction (`pka_predict_loocv.py`)

Trains a **directed message-passing neural network (D-MPNN)** using ChemProp v2 to predict the
pKa of the most basic ionizable center from a molecular graph (derived from SMILES). The
architecture passes learned messages along directed bonds over several rounds, progressively
encoding larger chemical neighborhoods around each atom, then pools these into a single molecular
vector that is mapped to a scalar pKa value by a feed-forward regression head. This script uses 
**leave-one-out cross-validation (LOO-CV)**.

Replicate experimental measurements for the same molecule are averaged before splitting.
For polyprotic or amphoteric molecules with multiple measured pKa values, the maximum
non-null value across pKa columns is used as the training target (i.e. the most basic center).

Each fold's trained model is saved as a PyTorch Lightning checkpoint (`.ckpt`), which can be
reloaded for inference on new molecules.

**Input:** a CSV file with the format as in 'data/experimental_pka_values.csv'

**Output:**
- `loo_cv_predictions.csv` — one row per molecule with columns `mol_id`, `smiles`, `true_pKa`,
  `predicted_pKa`
- Per-fold model checkpoints saved under `./pka_model/fold_NNN_<MoleculeID>/final_model.ckpt`
- LOO RMSE and MAE printed to STDOUT at the end of the run


## Dependencies

chemprop


## Roadmap

Further property prediction tools planned for this repository:

- cLogD prediction
- Aqueous solubility 
- Integrated ADMET scoring (extended MPO combining rule-based and ML-predicted endpoints)

---

## References

- Wager et al., *ACS Chemical Neuroscience* 2010 — CNS MPO scoring
- Yang et al., *J. Chem. Inf. Model.* 2019 — ChemProp D-MPNN architecture
- Heid et al., *J. Chem. Inf. Model.* 2024 — ChemProp v2
- Wieder et al., *Drug Discov. Today* 2020 — graph neural networks for molecular property prediction

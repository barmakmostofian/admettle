"""
check_adme.py
--------------------

Obtaining a CNS MPO score for a set of compounds (loaded as SMILES).
This is based on the individual desirability scores from Wager et al. (2010, 2016).  

Usage:
	python check_adme.py  --mol_file example_compounds.csv 
"""


import argparse
import numpy as np
import pandas as pd

from rdkit.Chem import MolFromSmiles, Descriptors, rdMolDescriptors



# ----------------------------------------------------------------------------
# Define the component scoring funtion and the overall (possibly weighted) sum
# ----------------------------------------------------------------------------


def score(val, low_ideal, high_ideal, low_zero=None, high_zero=None):
	"""Piecewise linear desirability function → score between 0 and 1."""
	if low_zero is not None and val <= low_zero:
		return 0.0
	if high_zero is not None and val >= high_zero:
		return 0.0
	if low_ideal <= val <= high_ideal:
		return 1.0
	if val < low_ideal and low_zero is not None:
		return (val - low_zero) / (low_ideal - low_zero)
	if val > high_ideal and high_zero is not None:
		return (high_zero - val) / (high_zero - high_ideal)
	return 1.0


def cns_mpo(smiles: str) -> dict:
	mol = MolFromSmiles(smiles)
	if mol is None:
		raise ValueError(f"Invalid SMILES: {smiles}")

	# Derive the 6 properties
	mw	  = Descriptors.MolWt(mol)
	clogp = Descriptors.MolLogP(mol)
	clogd = clogp - 0.75  # Simplified
	tpsa  = rdMolDescriptors.CalcTPSA(mol)
	hbd   = rdMolDescriptors.CalcNumHBD(mol)
	pka   = 8.0  # Placeholder; requires external pKa calculator (e.g. epik, molgpka, or qupkake)

	# Compute component desirability scores
	comp_scores = {
		"cLogP": score(clogp, low_ideal=-np.inf, high_ideal=3,	low_zero=None, high_zero=5),
		"cLogD": score(clogd, low_ideal=-np.inf, high_ideal=2,	low_zero=None, high_zero=4),
		"MW":	 score(mw,	  low_ideal=0,		  high_ideal=360, low_zero=None, high_zero=500),
		"TPSA":  score(tpsa,  low_ideal=40,		  high_ideal=90,  low_zero=20,	 high_zero=120),
		"HBD":	 score(hbd,   low_ideal=0,		  high_ideal=0.5, low_zero=None, high_zero=3.5),
		"pKa":	 score(pka,   low_ideal=-np.inf,  high_ideal=8,   low_zero=None, high_zero=10),
	}

	mpo_score = sum(comp_scores.values())

	return {
		"SMILES": smiles,
		"properties": {"MW": mw, "cLogP": clogp, "cLogD": clogd,
					   "TPSA": tpsa, "HBD": hbd, "pKa": pka},
		"component_scores": comp_scores,
		"MPO_score": round(mpo_score, 3),
		"CNS_favorable": mpo_score >= 4.0
	}




def main():

	parser = argparse.ArgumentParser(description='Read data file from the command line.')
	parser.add_argument('--mol_file', type=str,  help='Path to the feature compound file')
	args = parser.parse_args()

	df = pd.read_csv(args.mol_file, sep=',', header=0)


	# Long version:
	for mol in df['Compound Structure'] :
		result = cns_mpo(mol)
		for key, value in result.items():
			print(f"{key}: {value}")
		print("\n")

	# Short version:
	print("compound \t\t mpo score")
	for mol in df['Compound Structure'] :
		result = cns_mpo(mol)
		print(mol, result['MPO_score'])


if __name__ == '__main__':
	main()



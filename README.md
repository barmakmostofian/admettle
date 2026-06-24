# ADMETTLE

A growing collection of small-molecule property prediction tools for computational drug discovery,
with a focus on ADMET-relevant endpoints and CNS drug design.


Tools

1. CNS MPO Scoring (cns_mpo.py)

Computes the CNS Multiparameter Optimization (MPO) score for one or more molecules, following
the method of Wager et al. (ACS Chemical Neuroscience, 2010). The score is a composite of six
physicochemical properties that collectively predict CNS drug-likeness: cLogP, cLogD, molecular
weight, topological polar surface area (TPSA), hydrogen bond donor count (HBD), and the pKa of
the most basic center.

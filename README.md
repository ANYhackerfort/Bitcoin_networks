# Bitcoin Trust Network Topological Analysis

This repository contains multiple Jupyter notebook found in graph analysis focused on topological and spectral analysis of Bitcoin trust networks (Bitcoin OTC and Alpha) using:

- **Hodge Laplacians** (\( L_0 \), \( L_1 \))  
- **Persistent Homology (PH)** with barcode visualization  
- **Greedy Trust Walks** on harmonic cycle boundaries  
- **Random Graph Null Models** (Erdős–Rényi)

## Methods

The notebook performs the following steps:

- Constructs graphs from Bitcoin trust datasets (positive-only, negative-only, full).
- Applies spectral analysis using:
  - \( L_0 \) Laplacian: for connected components, fragmentation, Fiedler vectors.
  - \( L_1 \) Hodge Laplacian: for detecting harmonic cycles and 1D holes.
- Conducts temporal Persistent Homology to track:
  - \( H_0 \): component mergers over time
  - \( H_1 \): hole formation and resolution
- Implements a custom `Greedy_Trust_Walk` algorithm to simulate trust propagation.
- Compares all findings with equivalent Erdős–Rényi random graphs to assess statistical significance.

## Findings

- Real networks exhibit **early closure of small loops**, reflecting trust consolidation.
- **Persistent large holes** suggest structural bottlenecks and potentially exploratory or adversarial regions.
- **Negative-only subgraphs** reveal statistically significant high-order simplices, indicating conflict-driven market factions.
- **Greedy walks** avoid loop boundaries, reinforcing the interpretation of user-driven trust routing.

## Dependencies

- `networkx`
- `gudhi`
- `numpy`
- `matplotlib`
- `scipy`
- `pandas`

Install with:

```bash
pip install -r requirements.txt

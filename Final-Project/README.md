# Final Project — Symmetric Non-negative Matrix Factorization (SymNMF)

An implementation of **Symmetric Non-negative Matrix Factorization (SymNMF)** for clustering, using a **C numerical core**, a **Python extension built with the CPython C API**, and a comparison against **K-means**.

Completed as the final assignment in the Software Project course at Tel Aviv University.

## Overview

Given a dataset of points, the project computes a similarity matrix, derives a diagonal degree matrix, normalizes the similarities, and applies an iterative multiplicative update to find a non-negative factor matrix. Each point is assigned to the cluster corresponding to its largest factor value.

The accompanying `analysis.py` compares the SymNMF and K-means assignments using **silhouette scores** and the **Adjusted Rand Index (ARI)**.

## Components

| File | Purpose |
| --- | --- |
| `symnmf.c` / `symnmf.h` | C implementation of similarity, degree, normalization, SymNMF updates, and a command-line program |
| `symnmfmodule.c` | CPython extension exposing the C operations to Python |
| `symnmf.py` | Python CLI for the matrix operations and SymNMF clustering |
| `analysis.py` | Comparison of SymNMF and K-means using silhouette scores and ARI |
| `kmeans.py` | Python implementation of K-means used as the comparison baseline |
| `setup.py` | Builds the Python extension module |
| `Makefile` | Builds the standalone C executable |

## Requirements

- C compiler with `gcc` and `make`
- Python 3, with `setuptools` and `numpy`
- `scikit-learn` for `analysis.py`
- Python development headers to build the CPython extension

To install the Python packages in a suitable environment:

```bash
python3 -m pip install numpy setuptools scikit-learn
```

## Input format

Provide a text file containing one data point per line, with comma-separated numerical coordinates. Each row must have the same number of coordinates, for example:

```text
1.0,2.0
2.0,3.0
3.0,4.0
8.0,9.0
9.0,10.0
10.0,11.0
```

Save the example as `data.txt` to use the commands below.

## Build and run

### Standalone C implementation

```bash
make
./symnmf sym data.txt
./symnmf ddg data.txt
./symnmf norm data.txt
```

The C executable supports the `sym`, `ddg`, and `norm` goals.

### Python interface with the C extension

Build the extension first:

```bash
python3 setup.py build_ext --inplace
```

Run the Python CLI with a cluster count `k`, a goal, and the input file:

```bash
python3 symnmf.py 2 symnmf data.txt
python3 symnmf.py 2 sym data.txt
python3 symnmf.py 2 ddg data.txt
python3 symnmf.py 2 norm data.txt
```

The `symnmf` goal outputs the final `n × k` non-negative factor matrix. The other goals output `n × n` matrices. Values are printed to four decimal places.

### Compare clustering results

```bash
python3 analysis.py 2 data.txt
```

The analysis prints the silhouette score for SymNMF, the silhouette score for K-means, and the adjusted Rand index comparing their cluster assignments. The sample dataset above produces:

```text
nmf: 0.8065
kmeans: 0.8065
ari: 1.0000
```

These values are specific to this sample dataset, not a general performance benchmark.

## Implementation notes

- The similarity matrix uses Gaussian similarity between distinct points, with zeros along the diagonal.
- Normalization uses the diagonal degree matrix to build the normalized similarity matrix.
- The SymNMF factor is initialized in Python using a fixed NumPy random seed (`1234`) for reproducibility.
- The multiplicative updates are implemented in C; convergence uses a squared Frobenius-difference threshold of `1e-4` or a maximum of 300 updates.
- The Python extension explicitly converts between Python lists and C matrices, and releases allocated C memory after returning results.
- The K-means comparison uses the implementation included here from the earlier coursework.

## Notes

This is an academic implementation intended to demonstrate clustering algorithms and Python–C interoperability; it is not a production-grade library. Generated build artifacts are not included in the repository.

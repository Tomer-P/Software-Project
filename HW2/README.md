# HW2 – K-means++ with a C Extension

This assignment extends the K-means implementation from HW1 by adding K-means++ initialization in Python and running the K-means iterations through a C extension built with the Python C API.

## Files

- `kmeans_pp.py` – main Python program. Reads and merges the input files, validates the command-line arguments, selects the initial centroids with K-means++, calls the C extension, and prints the result.
- `kmeansmodule.c` – C extension containing the K-means implementation and exposing the `fit` function through the module `mykmeanssp`.
- `setup.py` – builds the `mykmeanssp` Python extension.

## K-means++ Initialization

The Python program:

1. Reads two input files with Pandas.
2. Performs an inner join using the first column as the key.
3. Sorts the joined observations by the key.
4. Uses `np.random.seed(1234)` for deterministic initialization.
5. Selects the first centroid randomly.
6. Repeatedly selects the next centroid with probability proportional to its distance from the nearest centroid already selected.
7. Passes the selected centroids and datapoints to the C extension.

The key column is used only to identify observations and is not included as a clustering coordinate.

## Arguments

The program is run with:

```bash
python3 kmeans_pp.py [K] iter eps file_name_1 file_name_2
```

`K` is optional. If it is omitted, the default value is `3`.

Validation rules:

- `1 < K < N` and `K` must be an integer.
- `1 < iter < 400` and `iter` must be an integer.
- `eps >= 0`.

## Build

Build the C extension in place with:

```bash
python3 setup.py build_ext --inplace
```

The extension is built as the Python module `mykmeanssp`.

## Run

Example with an explicit value of `K`:

```bash
python3 kmeans_pp.py 3 100 0.01 input_1.txt input_2.txt
```

Example using the default `K = 3`:

```bash
python3 kmeans_pp.py 100 0.01 input_1.txt input_2.txt
```

## Output

The first output line contains the keys of the observations selected as the initial K-means++ centroids, separated by commas.

Each following line contains one final centroid. Coordinates are printed to four decimal places.

## Error Messages

Depending on the invalid input, the program prints one of:

```text
Incorrect number of clusters!
Incorrect maximum iteration!
Incorrect epsilon!
An Error Has Occurred
```

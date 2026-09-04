# HW1 – K-Means Clustering

This assignment implements the K-means clustering algorithm in both **C** and **Python**.

## Files

- `kmeans.c` – C implementation
- `kmeans.py` – Python implementation

## Algorithm

The program:

1. Uses the first `K` data points as the initial centroids.
2. Assigns each point to its closest centroid using Euclidean distance.
3. Recomputes each centroid as the mean of the points assigned to its cluster.
4. Repeats until either:
   - every centroid moves by less than `0.001`, or
   - the maximum number of iterations is reached.

If `iter` is not supplied, the default value is `400`.

## Input

The input points are read from **standard input (STDIN)**.
Each line represents one point, with coordinates separated by commas.

Example:

```text
1.0,2.0
3.0,4.0
5.0,6.0
```

Command-line arguments:

```text
kmeans K [iter]
```

where:

- `K` must be an integer satisfying `1 < K < N`
- `iter` must be an integer satisfying `1 < iter < 800`

## Running the C version

Compile:

```bash
gcc -ansi -Wall -Wextra -Werror -pedantic-errors kmeans.c -o kmeans -lm
```

Run:

```bash
./kmeans 3 100 < input_data.txt
```

or, using the default iteration limit:

```bash
./kmeans 3 < input_data.txt
```

## Running the Python version

```bash
python3 kmeans.py 3 100 < input_data.txt
```

or, using the default iteration limit:

```bash
python3 kmeans.py 3 < input_data.txt
```

## Output

The final centroids are printed to **STDOUT**, one centroid per line, with every coordinate formatted to four decimal places.

Example:

```text
-5.6720,7.7200,-9.1478,-9.1870,2.1394
-5.3645,8.0872,5.4206,9.6482,-6.5119
-6.8116,-2.9751,3.7578,-0.8476,-10.5475
```

## Error Handling

The program validates the command-line arguments and prints the required error messages for invalid `K` or `iter` values. Other invalid input or runtime errors result in:

```text
An Error Has Occurred
```

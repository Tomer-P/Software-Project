"""Command-line interface for the SymNMF calculations."""

import sys
import numpy as np
import symnmfmodule


K_ERROR = "Incorrect number of clusters!"
OTHER_ERROR = "An Error Has Occurred"

VALID_GOALS = ("symnmf", "sym", "ddg", "norm")

def print_error(message):
    """Print the given error message and exit with a nonzero status."""
    print(message)
    sys.exit(1)

def parse_arguments():
    """Read k, the requested goal, and the input filename from argv.
    Reject an incorrect argument count or a nonnumeric k.
    """
    args = sys.argv

    if len(args) != 4:
        print_error(OTHER_ERROR)

    k_string = args[1]
    goal = args[2]
    file_name = args[3]

    try:
        k_value = float(k_string)
    except ValueError:
        print_error(K_ERROR)

    return k_value, goal, file_name

def read_points(file_name):
    """Load comma-separated data points as a two-dimensional NumPy array.
    Report a general error if the file cannot be read or parsed.
    """
    try:
        with open(file_name) as input_file:
            # Check for data before loading to avoid NumPy's empty-file warning.
            if not any(line.strip() and not line.lstrip().startswith("#")
                       for line in input_file):
                print_error(OTHER_ERROR)
            input_file.seek(0)
            return np.loadtxt(input_file, delimiter=",", ndmin=2)
    except Exception:
        print_error(OTHER_ERROR)

def validate_arguments(k_value, goal, number_of_points):
    """Require an integer-valued k between 1 and the number of points.
    Also check that the requested goal is supported.
    """
    if (
        not k_value.is_integer()
        or k_value <= 1
        or k_value >= number_of_points
    ):
        print_error(K_ERROR)

    if goal not in VALID_GOALS:
        print_error(OTHER_ERROR)

def print_matrix(matrix):
    """Print each matrix row as comma-separated values with four decimals."""
    for row in matrix:
        formatted_row = ",".join(
            "%.4f" % value
            for value in row
        )
        print(formatted_row)

def initialize_h(W, n, k):
    """Initialize H (n x k) from the n x n matrix W using a fixed seed."""
    np.random.seed(1234)
    m = np.average(W)
    upper = 2 * np.sqrt(m / k)
    H = np.random.uniform(low=0.0, high=upper, size=(n, k))
    return H


def main():
    """Read k, goal and file from argv, then print the selected matrix."""
    k_value, goal, file_name = parse_arguments()

    points = read_points(file_name)
    number_of_points = len(points)

    validate_arguments(
        k_value,
        goal,
        number_of_points
    )

    k = int(k_value)
    points_list = points.tolist()

    if goal == "sym":
        result = symnmfmodule.sym(points_list)
    elif goal == "ddg":
        result = symnmfmodule.ddg(points_list)
    elif goal == "norm":
        result = symnmfmodule.norm(points_list)
    else:
        W = symnmfmodule.norm(points_list)
        H = initialize_h(W, number_of_points, k)
        result = symnmfmodule.symnmf(
            H.tolist(),
            W,
            number_of_points,
            k
        )

    print_matrix(result)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print_error(OTHER_ERROR)

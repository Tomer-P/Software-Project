import sys
import numpy as np
import pandas as pd
import mykmeanssp


# Error messages
K_ERROR = "Incorrect number of clusters!"
ITER_ERROR = "Incorrect maximum iteration!"
EPS_ERROR = "Incorrect epsilon!"
OTHER_ERROR = "An Error Has Occurred"


def print_error(message):
    """
    Print an error message and terminate the program.
    """
    print(message)
    sys.exit(1)


def parse_arguments():
    """
    Read the command-line arguments.
    If K is not provided, its default value is 3.
    """
    args = sys.argv

    # The user provided K:
    if len(args) == 6:
        k_string = args[1]
        iter_string = args[2]
        eps_string = args[3]
        file_name_1 = args[4]
        file_name_2 = args[5]

    # The user did not provide K:
    elif len(args) == 5:
        k_string = "3"
        iter_string = args[1]
        eps_string = args[2]
        file_name_1 = args[3]
        file_name_2 = args[4]

    # other number of arguments is invalid.
    else:
        print_error(OTHER_ERROR)

    # Convert K to a numeric value.
    try:
        k_value = float(k_string)
    except ValueError:
        print_error(K_ERROR)

    # Convert iter to a numeric value.
    try:
        iter_value = float(iter_string)
    except ValueError:
        print_error(ITER_ERROR)

    # Convert eps to a floating-point value.
    try:
        eps = float(eps_string)
    except ValueError:
        print_error(EPS_ERROR)

    return (
        k_value,
        iter_value,
        eps,
        file_name_1,
        file_name_2
    )


def validate_arguments(k_value, iter_value, eps, number_of_points):
    """
    Validate K, iter, and eps
    """
    # 1 < K < N and K must be an integer.
    if (
        not k_value.is_integer()
        or k_value <= 1
        or k_value >= number_of_points
    ):
        print_error(K_ERROR)

    # 1 < iter < 400 and iter must be an integer.
    if (
        not iter_value.is_integer()
        or iter_value <= 1
        or iter_value >= 400
    ):
        print_error(ITER_ERROR)

    #eps >= 0.
    if eps < 0:
        print_error(EPS_ERROR)


def read_and_merge_files(file_name_1, file_name_2):
    """
    Read both input files, perform an inner join using
    the first column as the key, and sort by the key.
    """
    try:
        # Read both files without header rows.
        file_1 = pd.read_csv(file_name_1, header=None)
        file_2 = pd.read_csv(file_name_2, header=None)

        # Perform an inner join using the first column as the key.
        merged_data = pd.merge(
            file_1,
            file_2,
            on=0,
            how="inner"
        )

        # Sort the joined data by the key in ascending order.
        merged_data = merged_data.sort_values(by=0)

        return merged_data

    except Exception:
        print_error(OTHER_ERROR)


def initialize_centroids(points, keys, k):
    """
    Choose the initial centroids using the K-means++ algorithm.
    """

    # Use the seed required
    np.random.seed(1234)

    number_of_points = len(points)

    # Store the array positions of the selected observations.
    positions = []

    # Store the selected centroids.
    centroids = []

    # Choose the first centroid randomly.
    first_position = np.random.choice(number_of_points)

    positions.append(first_position)
    centroids.append(points[first_position])

    # Repeat until k centroids are selected.
    while len(centroids) < k:

        # the minimal Euclidean distance between point i and the already selected centroids.
        distances = np.zeros(number_of_points)

        # Compute D(x) for every data point.
        for i in range(number_of_points):
            minimum_distance = float("inf")

            # Find the nearest already selected centroid.
            for centroid in centroids:
                current_distance = np.linalg.norm(
                    points[i] - centroid
                )

                if current_distance < minimum_distance:
                    minimum_distance = current_distance

            distances[i] = minimum_distance

        # Convert the D(x) values into probabilities.
        probabilities = distances / np.sum(distances)

        # Choose the next centroid
        next_position = np.random.choice(
            number_of_points,
            p=probabilities
        )

        positions.append(next_position)
        centroids.append(points[next_position])

    # Convert the selected array positions to the original keys.
    selected_keys = [
        keys[position]
        for position in positions
    ]

    return selected_keys, centroids


def print_results(selected_keys, final_centroids):
    """
    Print the selected observation keys and the final centroids
    """

    # Print the selected observation keys 
    formatted_keys = ",".join(
        str(int(key))
        for key in selected_keys
    )
    print(formatted_keys)

    # Print each final centroid on a separate line 
    for centroid in final_centroids:
        formatted_centroid = ",".join(
            "%.4f" % coordinate
            for coordinate in centroid
        )
        print(formatted_centroid)


def main():
    
    # Read the command-line arguments.
    (
        k_value,
        iter_value,
        eps,
        file_name_1,
        file_name_2
    ) = parse_arguments()

    # Read the files, perform the inner join, and sort the data by the key.
    merged_data = read_and_merge_files(
        file_name_1,
        file_name_2
    )

    number_of_points = len(merged_data)

    # Validate K, iter, and eps.
    validate_arguments(
        k_value,
        iter_value,
        eps,
        number_of_points
    )

    k = int(k_value)
    max_iter = int(iter_value)

    # Store the observation keys from the first column.
    keys = merged_data.iloc[:, 0].to_numpy()

    # Store only the vector coordinates.
    points = merged_data.iloc[:, 1:].to_numpy(dtype=float)

    # Choose the initial centroids using K-means++.
    selected_keys, initial_centroids = initialize_centroids(
        points,
        keys,
        k
    )

    # Convert NumPy arrays to regular Python lists 
    points_list = points.tolist()

    initial_centroids_list = [
        centroid.tolist()
        for centroid in initial_centroids
    ]

    # Run the K-means algorithm 
    try:
        final_centroids = mykmeanssp.fit(
            initial_centroids_list,
            points_list,
            max_iter,
            eps
        )
    except Exception:
        print_error(OTHER_ERROR)

    # Print the result 
    print_results(
        selected_keys,
        final_centroids
    )


if __name__ == "__main__":
    main()

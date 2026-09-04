# -*- coding: utf-8 -*-
"""
K-means clustering implementation.

The program reads points from standard input, validates the command line
arguments, runs the K-means algorithm, and prints the final centroids.
"""

import sys
import math
EPSILON = 0.001
OTHER_ERROR = "An Error Has Occurred"
K_ERROR = "Incorrect number of clusters!"
ITER_ERROR = "Incorrect maximum iteration!"

def main():
  """Read the input, validate the arguments, run K-means, and print the result."""
  points = read_points()

  # Empty input cannot be clustered, so it is treated as a general input error.
  if len(points) == 0:
    print(OTHER_ERROR)
    sys.exit(1)

  args = sys.argv

  # The program accepts k, and optionally the maximum number of iterations.
  if (len(args) < 2) or (len(args) > 3):
    print(OTHER_ERROR)
    sys.exit(1)

  # Parse k separately so that an invalid k gets the correct error message.
  try:
    k_float = float(args[1])
  except ValueError:
    print(K_ERROR)
    sys.exit(1)

  # If iter is not given, use the default value required by the assignment.
  if len(args) == 3:
      try:
        iter_val_float = float(args[2])
      except ValueError:
        print(ITER_ERROR)
        sys.exit(1)
  else:
    iter_val_float = 400.0

  # Check the numeric restrictions before converting the values to integers.
  requirement(k_float, len(points), iter_val_float)

  k= int(k_float)
  iter_val = int(iter_val_float)

  centroids = k_means(points, k, iter_val)
  print_centroids(centroids)

def read_points():
  """Read comma-separated points from standard input and return them as floats."""
  points = []
  for line in sys.stdin:
    line = line.strip()

    # Ignore empty lines so that blank spacing in the input does not matter.
    if not line:
      continue

    try:
      points.append([float(x) for x in line.split(',')])
    except ValueError:
      print(OTHER_ERROR)
      sys.exit(1)

  return points

def k_means(points, k, iter):
  """Run the K-means algorithm and return the final centroids."""
  # Use the first k points as the initial centroids.
  centroids = points[:k]

  # Repeat the clustering process until convergence or until iter is reached.
  for i in range(iter):
    # Create one empty cluster for each centroid.
    clusters = [[] for j in range(k)]

    # Assign every point to the closest centroid.
    for point in points:
      best_centroid = closest_centroid(point, centroids)
      clusters[centroids.index(best_centroid)].append(point)

    # Recalculate each centroid from the points assigned to its cluster.
    new_centroids = [[] for j in range(k)]
    for j in range(k):
      if len(clusters[j]) == 0:
        new_centroids[j] = centroids[j]
      else:
        new_centroids[j] = calculate_centroid(clusters[j])

    # Stop once the centroids move by less than EPSILON.
    if converged(centroids, new_centroids):
      centroids = new_centroids
      break
    centroids = new_centroids
  return centroids

def Euclidean_Distance(p1,p2):
  """Return the Euclidean distance between two points."""
  if invalid_input(p1, p2):
    print(OTHER_ERROR)
    sys.exit(1)

  sum = 0
  for i in range(len(p1)):
    sum += pow((p1[i]-p2[i]),2)

  return math.sqrt(sum)

def closest_centroid(point, centroids):
  """Find the centroid that is closest to the given point."""
  min_point = centroids[0]
  min_dist = Euclidean_Distance(point, centroids[0])
  for i in range(1, len(centroids)):
    curr_dist = Euclidean_Distance(point, centroids[i])
    if(curr_dist < min_dist):
      min_dist = curr_dist
      min_point = centroids[i]
  return min_point

def calculate_centroid(cluster):
  """Calculate the average point of a cluster."""
  denominator = len(cluster)
  if(denominator == 0):
    print(OTHER_ERROR)
    sys.exit(1)
  my_centroid = [0 for x in range(len(cluster[0]))]
  for i in range(len(cluster)):
    for j in range(len(cluster[0])):
      my_centroid[j] += cluster[i][j]
  for i in range(len(my_centroid)):
    my_centroid[i] /= denominator
  return my_centroid

def print_centroids(centroids):
  """Print all centroids with four digits after the decimal point."""
  for centroid in centroids:
    formatted_centroid = ",".join(["%.4f" % x for x in centroid])
    print(formatted_centroid)

def converged(old_centroids, new_centroids):
  """Check whether all centroids moved less than EPSILON."""
  for i in range(len(old_centroids)):
    if(Euclidean_Distance(old_centroids[i],new_centroids[i])>=EPSILON):
      return False
  return True

def k_error(k, n):
  """Validate k according to the assignment restrictions."""
  if((k<=1) or (k>=n) or (not k.is_integer())):
    print(K_ERROR)
    sys.exit(1)

def iter_error(iter):
  """Validate the maximum number of iterations."""
  if((iter<=1) or (iter>=800) or (not iter.is_integer())):
    print(ITER_ERROR)
    sys.exit(1)

def requirement(k,n,iter):
  """Run all argument validation checks."""
  k_error(k,n)
  iter_error(iter)

def invalid_input(p1,p2):
  """Return True when two points do not have the same dimension."""
  if len(p1) != len(p2):
    return True
  return False

if __name__ == "__main__":
  main()

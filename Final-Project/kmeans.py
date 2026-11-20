"""K-means clustering with data points read from standard input.
The first k points serve as the initial centroids.
"""

import sys
import math
EPSILON = 0.0001
OTHER_ERROR = "An Error Has Occurred"
K_ERROR = "Incorrect number of clusters!"
ITER_ERROR = "Incorrect maximum iteration!"

def main():
  """Reads the inputs, runs K-means, and prints the resulting centroids."""
  points = read_points()

  if len(points) == 0:
    print(OTHER_ERROR)
    sys.exit(1)

  args = sys.argv

  if (len(args) < 2) or (len(args) > 3):
    print(OTHER_ERROR)
    sys.exit(1)

  # Keeps invalid k separate from general argument errors.
  try:
    k_float = float(args[1])
  except ValueError:
    print(K_ERROR)
    sys.exit(1)

  # The standalone K-means program defaults to 400 iterations.
  if len(args) == 3:
      try:
        iter_val_float = float(args[2])
      except ValueError:
        print(ITER_ERROR)
        sys.exit(1)
  else:
    iter_val_float = 400.0

  # Validates before casting so fractional values are not truncated.
  requirement(k_float, len(points), iter_val_float)

  k= int(k_float)
  iter_val = int(iter_val_float)

  centroids = k_means(points, k, iter_val)
  print_centroids(centroids)

def read_points():
  """Reads nonempty CSV lines as lists of floats; exits on invalid values."""
  points = []
  for line in sys.stdin:
    line = line.strip()

    # Ignores blank lines in the input.
    if not line:
      continue

    try:
      points.append([float(x) for x in line.split(',')])
    except ValueError:
      print(OTHER_ERROR)
      sys.exit(1)

  return points

def k_means(points, k, iter):
  """Iteratively assigns points and updates the k centroids.
  Leaves an empty cluster's centroid unchanged and stops on convergence
  or after the requested maximum number of iterations.
  """
  centroids = points[:k]

  for i in range(iter):
    clusters = [[] for j in range(k)]

    for point in points:
      best_centroid = closest_centroid(point, centroids)
      clusters[centroids.index(best_centroid)].append(point)

    new_centroids = [[] for j in range(k)]
    for j in range(k):
      if len(clusters[j]) == 0:
        new_centroids[j] = centroids[j]
      else:
        new_centroids[j] = calculate_centroid(clusters[j])

    if converged(centroids, new_centroids):
      centroids = new_centroids
      break
    centroids = new_centroids
  return centroids

def Euclidean_Distance(p1,p2):
  """Calculates Euclidean distance, rejecting points of different dimensions."""
  if invalid_input(p1, p2):
    print(OTHER_ERROR)
    sys.exit(1)

  sum = 0
  for i in range(len(p1)):
    sum += pow((p1[i]-p2[i]),2)

  return math.sqrt(sum)

def closest_centroid(point, centroids):
  """Returns the nearest centroid; ties keep the first matching centroid."""
  min_point = centroids[0]
  min_dist = Euclidean_Distance(point, centroids[0])
  for i in range(1, len(centroids)):
    curr_dist = Euclidean_Distance(point, centroids[i])
    if(curr_dist < min_dist):
      min_dist = curr_dist
      min_point = centroids[i]
  return min_point

def calculate_centroid(cluster):
  """Calculates the coordinate-wise mean of the points in a nonempty cluster."""
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
  """Prints one centroid per line, with four decimals per coordinate."""
  for centroid in centroids:
    formatted_centroid = ",".join(["%.4f" % x for x in centroid])
    print(formatted_centroid)

def converged(old_centroids, new_centroids):
  """Returns True only if every centroid moved less than EPSILON."""
  for i in range(len(old_centroids)):
    if(Euclidean_Distance(old_centroids[i],new_centroids[i])>=EPSILON):
      return False
  return True

def k_error(k, n):
  """Rejects k unless it is an integer-valued number strictly between 1 and n."""
  if((k<=1) or (k>=n) or (not k.is_integer())):
    print(K_ERROR)
    sys.exit(1)

def iter_error(iter):
  """Rejects an iteration limit unless it is an integer between 2 and 799."""
  if((iter<=1) or (iter>=800) or (not iter.is_integer())):
    print(ITER_ERROR)
    sys.exit(1)

def requirement(k,n,iter):
  """Checks the cluster count and iteration limit before running K-means."""
  k_error(k,n)
  iter_error(iter)

def invalid_input(p1,p2):
  """Checks whether two points have different coordinate counts."""
  if len(p1) != len(p2):
    return True
  return False

if __name__ == "__main__":
  main()

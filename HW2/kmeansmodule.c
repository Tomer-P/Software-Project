#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <stdlib.h>
#include <math.h>

#define OTHER_ERROR "An Error Has Occurred"


/* Function declarations for input handling, validation, clustering, and memory cleanup. */
double euclidean_distance(double *p1, double *p2, int d);
int closest_centroid(double *point, double **centroids, int k, int d);
double *calculate_centroid(double **cluster, int cluster_size, int d);
int converged(double **old_centroids, double **new_centroids, int k, int d, double eps);
double **k_means(double **points, double **initial_centroids, int n, int d, int k, int iter, double eps);

double **init_centroids(double **points, int k, int d);
double **init_empty_centroids(int k);
void free_matrix(double **matrix, int rows);
void free_clusters(double ***clusters, int k);
double **allocate_matrix(int rows, int cols);
double **pylist_to_matrix(PyObject *list, int rows, int cols);
PyObject *matrix_to_pylist(double **matrix, int rows, int cols);


/* Compute the Euclidean distance between two d-dimensional points. */
double euclidean_distance(double *p1, double *p2, int d)
{
    int i;
    double sum = 0.0;
    double diff;

    for (i = 0; i < d; i++)
    {
        diff = p1[i] - p2[i];
        sum += diff * diff;
    }

    return sqrt(sum);
}

/* Return the index of the centroid closest to the given point. */
int closest_centroid(double *point, double **centroids, int k, int d)
{
    int min_index;
    int i;
    double min_distance;
    double curr_dist;

    min_index = 0;
    min_distance = euclidean_distance(point, centroids[0], d);

    for (i = 1; i < k; i++)
    {
        curr_dist = euclidean_distance(point, centroids[i], d);
        if (curr_dist < min_distance)
        {
            min_distance = curr_dist;
            min_index = i;
        }
    }

    return min_index;
}


/* Compute a new centroid by averaging all points in a cluster. */
double *calculate_centroid(double **cluster, int cluster_size, int d)
{
    double *centroid;
    int i;
    int j;

    /* A centroid cannot be computed from an empty cluster. */
    if (cluster_size == 0)
    {
        return NULL;
    }

     /* Allocate memory for the new centroid. */
    centroid = (double *)calloc(d, sizeof(double));
    /* Return NULL if the allocation failed. */
    if (centroid == NULL)
    {
        return NULL;
    }

    /* Sum each coordinate over all points in the cluster. */
    for (i = 0; i < cluster_size; i++)
    {
        for (j = 0; j < d; j++)
        {
            centroid[j] += cluster[i][j];
        }
    }

     /* Divide by the cluster size to get the average point. */
    for (j = 0; j < d; j++)
    {
        centroid[j] /= cluster_size;
    }

    /* Return the computed centroid. */
    return centroid;
}

/* Check whether all centroids moved less than EPSILON. */
int converged(double **old_centroids, double **new_centroids, int k, int d, double eps)
{
    int i;

    for (i = 0; i < k; i++)
    {
        if (euclidean_distance(old_centroids[i], new_centroids[i], d) >= eps)
        {
            return 0; 
        }
    }

    return 1; 
}

/* Run the K-means iterations and return the final centroids. */
double **k_means(double **points, double **initial_centroids, int n, int d, int k, int iter, double eps)
{
    double **centroids;
    double **new_centroids;
    double ***clusters;
    double **temp;
    int *cluster_sizes;
    int i;
    int j;
    int c;
    int best_index;

    centroids = init_centroids(initial_centroids, k, d);

    if (centroids == NULL)
    {
        return NULL;
    }

    for (i = 0; i < iter; i++)
    {
        clusters = (double ***)malloc(k * sizeof(double **));

        if (clusters == NULL)
        {
            free_matrix(centroids, k);
            return NULL;
        }

        for (c = 0; c < k; c++)
        {
            clusters[c] = NULL;
        }

        cluster_sizes = (int *)calloc(k, sizeof(int));

        if (cluster_sizes == NULL)
        {
            free_clusters(clusters, k);
            free_matrix(centroids, k);
            return NULL;
        }

        for (j = 0; j < n; j++)
        {
            best_index = closest_centroid(
                points[j],
                centroids,
                k,
                d
            );

            temp = (double **)realloc(
                clusters[best_index],
                (cluster_sizes[best_index] + 1) * sizeof(double *)
            );

            if (temp == NULL)
            {
                free_clusters(clusters, k);
                free(cluster_sizes);
                free_matrix(centroids, k);
                return NULL;
            }

            clusters[best_index] = temp;
            clusters[best_index][cluster_sizes[best_index]] = points[j];
            cluster_sizes[best_index]++;
        }

        new_centroids = init_empty_centroids(k);

        if (new_centroids == NULL)
        {
            free_clusters(clusters, k);
            free(cluster_sizes);
            free_matrix(centroids, k);
            return NULL;
        }

        for (c = 0; c < k; c++)
        {
            if (cluster_sizes[c] == 0)
            {
                new_centroids[c] = (double *)malloc(
                    d * sizeof(double)
                );

                if (new_centroids[c] == NULL)
                {
                    free_clusters(clusters, k);
                    free(cluster_sizes);
                    free_matrix(centroids, k);
                    free_matrix(new_centroids, c);
                    return NULL;
                }

                for (j = 0; j < d; j++)
                {
                    new_centroids[c][j] = centroids[c][j];
                }
            }
            else
            {
                new_centroids[c] = calculate_centroid(
                    clusters[c],
                    cluster_sizes[c],
                    d
                );

                if (new_centroids[c] == NULL)
                {
                    free_clusters(clusters, k);
                    free(cluster_sizes);
                    free_matrix(centroids, k);
                    free_matrix(new_centroids, c);
                    return NULL;
                }
            }
        }

        free_clusters(clusters, k);
        free(cluster_sizes);

        if (converged(
                centroids,
                new_centroids,
                k,
                d,
                eps))
        {
            free_matrix(centroids, k);
            centroids = new_centroids;
            break;
        }

        free_matrix(centroids, k);
        centroids = new_centroids;
    }

    return centroids;
}

/* Allocate the outer array used to store centroid pointers. */
double **init_empty_centroids(int k)
{
    double **centroids;

    centroids = malloc(k * sizeof(double *));
    if (centroids == NULL)
    {
        return NULL;
    }

    return centroids;
}

/* Initialize the centroids with copies of the first k points. */
double **init_centroids(double **points, int k, int d)
{
    double **centroids;
    int i;
    int j;

    centroids = init_empty_centroids(k);

    if (centroids == NULL)
    {
        return NULL;
    }

    for (i = 0; i < k; i++)
    {
        centroids[i] = malloc(d * sizeof(double));
        if (centroids[i] == NULL)
        {
            free_matrix(centroids, i);
            return NULL;
        }

        for (j = 0; j < d; j++)
        {
            centroids[i][j] = points[i][j];
        }
    }

    return centroids;
}

/* Free a matrix whose rows were allocated separately. */
void free_matrix(double **matrix, int rows)
{
    int i;

    for (i = 0; i < rows; i++)
    {
        free(matrix[i]);
    }

    free(matrix);
}

/* Free the temporary cluster arrays used in one iteration. */
void free_clusters(double ***clusters, int k)
{
    int i;

    for (i = 0; i < k; i++)
    {
        free(clusters[i]);
    }

    free(clusters);
}

double **allocate_matrix(int rows, int cols)
{
    double **matrix;
    int i;

    matrix = malloc(rows * sizeof(double *));
    if (matrix == NULL)
    {
        PyErr_NoMemory();
        return NULL;
    }

    for (i = 0; i < rows; i++)
    {
        matrix[i] = malloc(cols * sizeof(double));
        if (matrix[i] == NULL)
        {
            free_matrix(matrix, i);
            PyErr_NoMemory();
            return NULL;
        }
    }

    return matrix;
}

double **pylist_to_matrix(PyObject *list, int rows, int cols)
{
    double **matrix;
    PyObject *row_obj;
    PyObject *value_obj;
    int i;
    int j;

    matrix = allocate_matrix(rows, cols);

    if (matrix == NULL)
    {
        PyErr_NoMemory();
        return NULL;
    }

    for (i = 0; i < rows; i++)
    {
        row_obj = PyList_GetItem(list, i);

        for (j = 0; j < cols; j++)
        {
            value_obj = PyList_GetItem(row_obj, j);
            matrix[i][j] = PyFloat_AsDouble(value_obj);

            if (PyErr_Occurred())
            {
                free_matrix(matrix, rows);
                return NULL;
            }
        }
    }

    return matrix;
}

PyObject *matrix_to_pylist(double **matrix, int rows, int cols)
{
    PyObject *list, *row_list;
    int i, j;

    list = PyList_New(rows);
    if (list == NULL)
    {
        return NULL;
    }

    for (i = 0; i < rows; i++)
    {
        row_list = PyList_New(cols);
        if (row_list == NULL)
        {
            Py_DECREF(list);
            return NULL;
        }

        for (j = 0; j < cols; j++)
        {
            PyObject *value = PyFloat_FromDouble(matrix[i][j]);
            if (value == NULL)
            {
                Py_DECREF(row_list);
                Py_DECREF(list);
                return NULL;
            }
            PyList_SET_ITEM(row_list, j, value);
        }

        PyList_SET_ITEM(list, i, row_list);
    }

    return list;
}

static PyObject *fit(PyObject *self, PyObject *args)
{
    PyObject *points_obj;
    PyObject *initial_centroids_obj;
    PyObject *result;

    Py_ssize_t n;
    Py_ssize_t d;
    Py_ssize_t k;

    double **points;
    double **initial_centroids;
    double **final_centroids;

    int iter;
    double eps;

    (void)self;

    if (!PyArg_ParseTuple(
            args,
            "OOid",
            &initial_centroids_obj,
            &points_obj,
            &iter,
            &eps))
    {
        return NULL;
    }

    if (!PyList_Check(points_obj) ||
        !PyList_Check(initial_centroids_obj))
    {
        PyErr_SetString(
            PyExc_TypeError,
            "Expected lists"
        );
        return NULL;
    }

    n = PyList_Size(points_obj);
    k = PyList_Size(initial_centroids_obj);

    if (n == 0 || k == 0)
    {
        PyErr_SetString(
            PyExc_ValueError,
            "Empty input"
        );
        return NULL;
    }

    if (!PyList_Check(PyList_GetItem(points_obj, 0)) ||
        !PyList_Check(
            PyList_GetItem(initial_centroids_obj, 0)))
    {
        PyErr_SetString(
            PyExc_TypeError,
            "Expected lists of vectors"
        );
        return NULL;
    }

    d = PyList_Size(PyList_GetItem(points_obj, 0));

    if (d == 0 ||
        PyList_Size(
            PyList_GetItem(initial_centroids_obj, 0)) != d)
    {
        PyErr_SetString(
            PyExc_ValueError,
            "Inconsistent dimensions"
        );
        return NULL;
    }

    points = pylist_to_matrix(
        points_obj,
        (int)n,
        (int)d
    );

    if (points == NULL)
    {
        return NULL;
    }

    initial_centroids = pylist_to_matrix(
        initial_centroids_obj,
        (int)k,
        (int)d
    );

    if (initial_centroids == NULL)
    {
        free_matrix(points, (int)n);
        return NULL;
    }

    final_centroids = k_means(
        points,
        initial_centroids,
        (int)n,
        (int)d,
        (int)k,
        iter,
        eps
    );

    free_matrix(points, (int)n);
    free_matrix(initial_centroids, (int)k);

    if (final_centroids == NULL)
    {
        PyErr_NoMemory();
        return NULL;
    }

    result = matrix_to_pylist(
        final_centroids,
        (int)k,
        (int)d
    );

    free_matrix(final_centroids, (int)k);

    if (result == NULL)
    {
        return NULL;
    }

    return result;
}

static PyMethodDef MyKMeansMethods[] = {
    {
        "fit",
        fit,
        METH_VARARGS,
        "fit(initial_centroids, datapoints, iter, eps) -> final centroids"
    },
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef mykmeansmodule = {
    PyModuleDef_HEAD_INIT,
    "mykmeanssp",
    NULL,
    -1,
    MyKMeansMethods
};

PyMODINIT_FUNC PyInit_mykmeanssp(void)
{
    return PyModule_Create(&mykmeansmodule);
}

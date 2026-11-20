#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdlib.h>
#include "symnmf.h"

static int copy_python_row(PyObject *row, double *destination, int cols);
static double **pylist_to_matrix(PyObject *list, int rows, int cols);
static PyObject *row_to_pylist(const double *values, int cols);
static PyObject *matrix_to_pylist(double **matrix, int rows, int cols);
static int get_dimensions(PyObject *list, int *rows, int *cols);
static PyObject *py_sym(PyObject *self, PyObject *args);
static PyObject *py_ddg(PyObject *self, PyObject *args);
static PyObject *py_norm(PyObject *self, PyObject *args);
static PyObject *py_symnmf(PyObject *self, PyObject *args);
static double **degree_from_points(double **points, int n, int d);
static double **norm_from_points(double **points, int n, int d);

/* Copies one Python list into a C matrix row, checking its size and values.
   Returns 0 if the row is invalid and leaves the Python error set. */
static int copy_python_row(PyObject *row, double *destination, int cols)
{
    PyObject *item;
    int j;
    if (!PyList_Check(row) || PyList_Size(row) != cols)
    {
        PyErr_SetString(PyExc_ValueError, "Invalid input");
        return 0;
    }
    for (j = 0; j < cols; j++)
    {
        item = PyList_GetItem(row, j);
        destination[j] = PyFloat_AsDouble(item);
        if (PyErr_Occurred())
            return 0;
    }
    return 1;
}

/* Converts a Python list of rows into an allocated C matrix.
   Checks every row and frees the matrix if a conversion fails. */
static double **pylist_to_matrix(PyObject *list, int rows, int cols)
{
    double **matrix;
    int i;
    if (!PyList_Check(list) || PyList_Size(list) != rows)
    {
        PyErr_SetString(PyExc_ValueError, "Invalid input");
        return NULL;
    }
    matrix = allocate_matrix(rows, cols);
    if (matrix == NULL)
    {
        PyErr_NoMemory();
        return NULL;
    }
    for (i = 0; i < rows; i++)
    {
        if (!copy_python_row(PyList_GetItem(list, i), matrix[i], cols))
        {
            free_matrix(matrix, rows);
            return NULL;
        }
    }
    return matrix;
}

/* Creates a Python list from a C row, converting each value to a float.
   Returns NULL if Python cannot allocate an item or the list. */
static PyObject *row_to_pylist(const double *values, int cols)
{
    PyObject *row, *value;
    int j;
    row = PyList_New(cols);
    if (row == NULL)
        return NULL;
    for (j = 0; j < cols; j++)
    {
        value = PyFloat_FromDouble(values[j]);
        if (value == NULL)
        {
            Py_DECREF(row);
            return NULL;
        }
        PyList_SET_ITEM(row, j, value);
    }
    return row;
}

/* Builds a Python list of lists from a C matrix.
   Releases any Python objects already created if conversion fails. */
static PyObject *matrix_to_pylist(double **matrix, int rows, int cols)
{
    PyObject *list, *row;
    int i;
    list = PyList_New(rows);
    if (list == NULL)
        return NULL;
    for (i = 0; i < rows; i++)
    {
        row = row_to_pylist(matrix[i], cols);
        if (row == NULL)
        {
            Py_DECREF(list);
            return NULL;
        }
        PyList_SET_ITEM(list, i, row);
    }
    return list;
}

/* Reads n and d from the input list and its first point.
   Returns 0 if either list is missing or empty; checks other rows later. */
static int get_dimensions(PyObject *list, int *rows, int *cols)
{
    PyObject *first_row;
    if (!PyList_Check(list) || PyList_Size(list) == 0)
        return 0;
    first_row = PyList_GetItem(list, 0);
    if (!PyList_Check(first_row) || PyList_Size(first_row) == 0)
        return 0;
    *rows = (int)PyList_Size(list);
    *cols = (int)PyList_Size(first_row);
    return 1;
}

/* Converts the input points to C, computes their similarity matrix,
   and returns it as a Python list while freeing the C matrices. */
static PyObject *py_sym(PyObject *self, PyObject *args)
{
    PyObject *points_obj, *result;
    double **points, **A;
    int n, d;

    (void)self;
    if (!PyArg_ParseTuple(args, "O", &points_obj))
        return NULL;
    if (!get_dimensions(points_obj, &n, &d))
    {
        PyErr_SetString(PyExc_ValueError, "Invalid input");
        return NULL;
    }
    points = pylist_to_matrix(points_obj, n, d);
    if (points == NULL)
        return NULL;
    A = sym(points, n, d);
    free_matrix(points, n);
    if (A == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
        return NULL;
    }
    result = matrix_to_pylist(A, n, n);
    free_matrix(A, n);
    return result;
}

/* Builds the degree matrix from points using a temporary similarity matrix.
   The returned n x n matrix belongs to the caller. */
static double **degree_from_points(double **points, int n, int d)
{
    double **A, **D;
    A = sym(points, n, d);
    if (A == NULL)
        return NULL;
    D = ddg(A, n);
    free_matrix(A, n);
    return D;
}

/* Converts points to C, computes their diagonal degree matrix,
   and returns it to Python. All temporary C matrices are freed. */
static PyObject *py_ddg(PyObject *self, PyObject *args)
{
    PyObject *points_obj, *result;
    double **points, **D;
    int n, d;
    (void)self;
    if (!PyArg_ParseTuple(args, "O", &points_obj))
        return NULL;
    if (!get_dimensions(points_obj, &n, &d))
    {
        PyErr_SetString(PyExc_ValueError, "Invalid input");
        return NULL;
    }
    points = pylist_to_matrix(points_obj, n, d);
    if (points == NULL)
        return NULL;
    D = degree_from_points(points, n, d);
    free_matrix(points, n);
    if (D == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
        return NULL;
    }
    result = matrix_to_pylist(D, n, n);
    free_matrix(D, n);
    return result;
}

/* Computes normalized similarity from the input points.
   Frees the similarity and degree matrices, returning only W. */
static double **norm_from_points(double **points, int n, int d)
{
    double **A, **D, **W;
    A = sym(points, n, d);
    if (A == NULL)
        return NULL;
    D = ddg(A, n);
    if (D == NULL)
    {
        free_matrix(A, n);
        return NULL;
    }
    W = norm(A, D, n);
    free_matrix(A, n);
    free_matrix(D, n);
    return W;
}

/* Converts points to C, computes normalized similarity,
   and returns the n x n result as a Python list of lists. */
static PyObject *py_norm(PyObject *self, PyObject *args)
{
    PyObject *points_obj, *result;
    double **points, **W;
    int n, d;
    (void)self;
    if (!PyArg_ParseTuple(args, "O", &points_obj))
        return NULL;
    if (!get_dimensions(points_obj, &n, &d))
    {
        PyErr_SetString(PyExc_ValueError, "Invalid input");
        return NULL;
    }
    points = pylist_to_matrix(points_obj, n, d);
    if (points == NULL)
        return NULL;
    W = norm_from_points(points, n, d);
    free_matrix(points, n);
    if (W == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
        return NULL;
    }
    result = matrix_to_pylist(W, n, n);
    free_matrix(W, n);
    return result;
}

/* Takes H (n x k), W (n x n), n and k from Python and runs SymNMF.
   Returns the optimized H as nested lists, freeing temporary C matrices. */
static PyObject *py_symnmf(PyObject *self, PyObject *args)
{
    PyObject *H_obj, *W_obj, *result;
    double **H, **W, **H_final;
    int n, k;

    (void)self;
    if (!PyArg_ParseTuple(args, "OOii", &H_obj, &W_obj, &n, &k))
        return NULL;
    H = pylist_to_matrix(H_obj, n, k);
    if (H == NULL)
        return NULL;
    W = pylist_to_matrix(W_obj, n, n);
    if (W == NULL)
    {
        free_matrix(H, n);
        return NULL;
    }
    H_final = symnmf(W, H, n, k);
    free_matrix(W, n);
    free_matrix(H, n);
    if (H_final == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
        return NULL;
    }
    result = matrix_to_pylist(H_final, n, k);
    free_matrix(H_final, n);
    return result;
}

static PyMethodDef SymNMFMethods[] = {
    {"sym", py_sym, METH_VARARGS, "sym(points): return the similarity matrix."},
    {"ddg", py_ddg, METH_VARARGS, "ddg(points): return the diagonal degree matrix."},
    {"norm", py_norm, METH_VARARGS, "norm(points): return normalized similarity."},
    {"symnmf", py_symnmf, METH_VARARGS, "symnmf(H, W, n, k): return optimized H."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef symnmfmodule = {
    PyModuleDef_HEAD_INIT,
    "symnmfmodule",
    NULL,
    -1,
    SymNMFMethods
};

/* Registers the extension methods when Python imports the module. */
PyMODINIT_FUNC PyInit_symnmfmodule(void)
{
    return PyModule_Create(&symnmfmodule);
}

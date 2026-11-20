#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include "symnmf.h"

#define BETA 0.5
#define MAX_ITER 300
#define EPSILON 0.0001
#define READ_ERROR -1
#define READ_EOF 0
#define READ_OK 1
#define OTHER_ERROR "An Error Has Occurred"

double squared_distance(double *p1, double *p2, int d);
double **allocate_matrix(int rows, int cols);
void free_matrix(double **matrix, int rows);
void print_matrix(double **matrix, int rows, int cols);
double **sym(double **points, int n, int d);
double **ddg(double **similarity_matrix, int n);
double **norm(double **similarity_matrix,
              double **degree_matrix,
              int n);
double **matrix_multiply(
    double **A,
    double **B,
    int rows_A,
    int cols_A,
    int cols_B
);
double **transpose(double **matrix, int rows, int cols);
double frobenius_diff(double **A, double **B, int rows, int cols);
void free_update_matrices(
    double **H_transpose, double **HtH,
    double **numerator, double **denominator,
    int k, int n);
int update_allocation_failed(
    double **matrix,
    double **H_transpose, double **HtH,
    double **numerator, double **denominator,
    int k, int n);
void fill_updated_h(
    double **H_new,
    double **H,
    double **numerator,
    double **denominator,
    int n,
    int k);
double **update_h(double **W, double **H, int n, int k);
double **copy_matrix(double **matrix, int rows, int cols);
double **symnmf(double **W, double **H, int n, int k);
int grow_line_buffer(char **line, int *capacity);
int read_line_dynamic(FILE *file, char **line_out);
int trim_line(char *line);
int count_coordinates(char *line);
double *parse_point(char *line, int d);
int grow_points(double ***points, int *capacity);
int add_point_from_line(
    char *line,
    double ***points,
    int *n,
    int *d,
    int *capacity);
double **read_points(FILE *file, int *n, int *d);
int parse_double(char *token, double *result);
double **calculate_goal(
    const char *goal,
    double **points,
    int n,
    int d);
int main(int argc, char *argv[]);

/* Returns the squared Euclidean distance between two d-dimensional points.
   No square root is needed for the similarity calculation. */
double squared_distance(double *p1, double *p2, int d)
{
    double sum = 0.0;
    int i;
    double diff;

    for (i = 0; i < d; i++)
    {
        diff = p1[i] - p2[i];
        sum += diff * diff;
    }

    return sum;
}

/* Frees a matrix allocated as separate rows, followed by the row array.
   The row count must match the original allocation. */
void free_matrix(double **matrix, int rows)
{
    int i;

    for (i = 0; i < rows; i++)
    {
        free(matrix[i]);
    }

    free(matrix);
}

/* Allocates a zero-initialized rows x cols matrix, one row at a time.
   Returns NULL on failure; release the result with free_matrix(). */
double **allocate_matrix(int rows, int cols)
{
    double **matrix;
    int i;

    matrix = malloc(rows * sizeof(double *));
    if (matrix == NULL)
    {
        return NULL;
    }

    for (i = 0; i < rows; i++)
    {
        matrix[i] = calloc(cols, sizeof(double));        
        if (matrix[i] == NULL)
        {
            free_matrix(matrix, i);
            return NULL;
        }
    }

    return matrix;
}

/* Prints a rows x cols matrix as CSV, using four decimal places. */
void print_matrix(double **matrix, int rows, int cols)
{
    int i, j;

    for (i = 0; i < rows; i++)
    {
        for (j = 0; j < cols; j++)
        {
            printf("%.4f", matrix[i][j]);

            if (j < cols - 1)
            {
                printf(",");
            }
        }
        printf("\n");
    }
}

/* Builds an n x n similarity matrix with a zero diagonal.
   For i != j, A[i][j] = exp(-||xi - xj||^2 / 2).
   Returns a new matrix, or NULL if allocation fails. */
double **sym(double **points, int n, int d)
{
    double **similarity_matrix;
    int i, j;
    double dist;

    similarity_matrix = allocate_matrix(n, n);
    if (similarity_matrix == NULL)
    {
        return NULL;
    }

    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {

            if (i == j)
            {
                dist = 0.0;
            }

            else
            {
                dist = squared_distance(points[i], points[j], d);
                dist = exp(-dist/2.0);
            }

            similarity_matrix[i][j] = dist;
        }
    }

    return similarity_matrix;
}

/* Sums the rows of the n x n similarity matrix A into the diagonal of D.
   Returns an allocated n x n matrix; all other entries are zero. */
double **ddg(double **similarity_matrix, int n)
{
    double **degree_matrix;
    int i, j;
    double sum;

    degree_matrix = allocate_matrix(n, n);
    if (degree_matrix == NULL)
    {
        return NULL;
    }

    for (i = 0; i < n; i++)
    {
        sum = 0.0;
        for (j = 0; j < n; j++)
        {
            sum += similarity_matrix[i][j];
        }
        degree_matrix[i][i] = sum;
    }

    return degree_matrix;
}

/* Computes W[i][j] = A[i][j] / sqrt(D[i][i] * D[j][j]).
   Entries involving a zero degree are left at zero.
   Returns a new matrix, or NULL on allocation failure. */
double **norm(double **similarity_matrix, double **degree_matrix, int n)
{
    double **normalized_matrix;
    int i, j;

    normalized_matrix = allocate_matrix(n, n);
    if (normalized_matrix == NULL)
    {
        return NULL;
    }

    for (i = 0; i < n; i++)
    {
        if (degree_matrix[i][i] == 0.0)
        {
            continue;
        }

        for (j = 0; j < n; j++)
        {
            if (degree_matrix[j][j] == 0.0)
            {
                continue;
            }

            normalized_matrix[i][j] =
                similarity_matrix[i][j] /
                sqrt(degree_matrix[i][i] * degree_matrix[j][j]);
        }
    }

    return normalized_matrix;
}

/* Multiplies A (rows_A x cols_A) by B (cols_A x cols_B).
   Returns a new rows_A x cols_B matrix, or NULL on allocation failure. */
double **matrix_multiply(double **A, double **B, int rows_A, int cols_A, int cols_B)
{
    double **result;
    int i, j, k;

    result = allocate_matrix(rows_A, cols_B);
    if (result == NULL)
    {
        return NULL;
    }

    for (i = 0; i < rows_A; i++)
    {
        for (j = 0; j < cols_B; j++)
        {
            for (k = 0; k < cols_A; k++)
            {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return result;
}

/* Transposes the given rows x cols matrix into a new cols x rows matrix.
   Returns NULL on allocation failure; the input is unchanged. */
double **transpose(double **matrix, int rows, int cols)
{
    double **transposed;
    int i, j;

    transposed = allocate_matrix(cols, rows);
    if (transposed == NULL)
    {
        return NULL;
    }

    for (i = 0; i < rows; i++)
    {
        for (j = 0; j < cols; j++)
        {
            transposed[j][i] = matrix[i][j];
        }
    }

    return transposed;
}   

/* Computes the sum of squared entry-wise differences between A and B.
   Returns -1 if either matrix pointer is NULL. */
double frobenius_diff(double **A, double **B, int rows, int cols)
{
    double sum = 0.0;
    int i;

    if (A == NULL || B == NULL)
    {
        return -1.0;
    }

    for (i = 0; i < rows; i++)
    {
        sum += squared_distance(A[i], B[i], cols);
    }

    return sum;
}

/* Frees the temporary products used during an H update.
   Any of the pointers may be NULL. */
void free_update_matrices(
    double **H_transpose, double **HtH,
    double **numerator, double **denominator,
    int k, int n)
{
    if (H_transpose != NULL)
        free_matrix(H_transpose, k);

    if (HtH != NULL)
        free_matrix(HtH, k);

    if (numerator != NULL)
        free_matrix(numerator, n);

    if (denominator != NULL)
        free_matrix(denominator, n);
}

/* Checks an intermediate allocation and cleans up if it failed.
   Returns 1 on failure and 0 otherwise. */
int update_allocation_failed(
    double **matrix,
    double **H_transpose, double **HtH,
    double **numerator, double **denominator,
    int k, int n)
{
    if (matrix == NULL)
    {
        free_update_matrices(
            H_transpose, HtH,
            numerator, denominator,
            k, n);
        return 1;
    }

    return 0;
}

/* Fills H_new (n x k) from H using the SymNMF update rule.
   Uses a small offset in every denominator to avoid division by zero. */
void fill_updated_h(
    double **H_new,
    double **H,
    double **numerator,
    double **denominator,
    int n,
    int k)
{
    int i, j;

    for (i = 0; i < n; i++)
    {
        for (j = 0; j < k; j++)
        {
            H_new[i][j] = H[i][j] *
                (1.0 - BETA + BETA * numerator[i][j] /
                 (denominator[i][j] + 1e-12));
        }
    }
}

/* Updates H (n x k) using W (n x n) and H*(H^T*H).
   Returns a newly allocated H, or NULL if an allocation fails. */
double **update_h(double **W, double **H, int n, int k)
{
    double **H_transpose = NULL, **HtH = NULL;
    double **numerator = NULL, **denominator = NULL, **H_new = NULL;
    H_transpose = transpose(H, n, k);
    if (update_allocation_failed(H_transpose, H_transpose, HtH,
                                 numerator, denominator, k, n))
        return NULL;
    HtH = matrix_multiply(H_transpose, H, k, n, k);
    if (update_allocation_failed(HtH, H_transpose, HtH,
                                 numerator, denominator, k, n))
        return NULL;
    numerator = matrix_multiply(W, H, n, n, k);
    if (update_allocation_failed(numerator, H_transpose, HtH,
                                 numerator, denominator, k, n))
        return NULL;
    denominator = matrix_multiply(H, HtH, n, k, k);
    if (update_allocation_failed(denominator, H_transpose, HtH,
                                 numerator, denominator, k, n))
        return NULL;
    H_new = allocate_matrix(n, k);
    if (H_new == NULL)
    {
        free_update_matrices(H_transpose, HtH, numerator, denominator, k, n);
        return NULL;
    }
    fill_updated_h(H_new, H, numerator, denominator, n, k);
    free_update_matrices(H_transpose, HtH, numerator, denominator, k, n);
    return H_new;
}

/* Returns an independent rows x cols copy of the input matrix.
   Returns NULL if allocation fails. */
double **copy_matrix(double **matrix, int rows, int cols)
{
    double **copy;
    int i, j;

    copy = allocate_matrix(rows, cols);
    if (copy == NULL)
    {
        return NULL;
    }

    for (i = 0; i < rows; i++)
    {
        for (j = 0; j < cols; j++)
        {
            copy[i][j] = matrix[i][j];
        }
    }

    return copy;
}

/* Repeatedly updates H until the squared Frobenius change is below
   EPSILON, or MAX_ITER updates have been made. Returns an allocated
   n x k result (NULL on failure) without modifying the input H. */
double **symnmf(double **W, double **H, int n, int k)
{
    double **H_current;
    double **H_new;
    double diff;
    int iter = 0;

    H_current = copy_matrix(H, n, k);
    if (H_current == NULL)
    {
        return NULL;
    }

    while (iter < MAX_ITER)
    {
        H_new = update_h(W, H_current, n, k);
        if (H_new == NULL)
        {
            free_matrix(H_current, n);
            return NULL;
        }

        diff = frobenius_diff(H_new, H_current, n, k);
        free_matrix(H_current, n);
        H_current = H_new;

        if (diff < EPSILON)
        {
            break;
        }
        iter++;
    }

    return H_current;
}

/* Doubles the line buffer capacity using realloc.
   Returns 0 on failure without freeing the original buffer. */
int grow_line_buffer(char **line, int *capacity)
{
    char *temp;

    *capacity *= 2;
    temp = realloc(*line, (*capacity) * sizeof(char));

    if (temp == NULL)
    {
        return 0;
    }

    *line = temp;
    return 1;
}

/* Reads characters into the buffer until a newline or EOF.
   Grows the buffer when needed; returns 0 on a read/allocation error. */
static int read_line_body(FILE *file, char **line, int *length, int *capacity)
{
    int ch;
    while ((ch = fgetc(file)) != EOF)
    {
        if (*length + 1 >= *capacity &&
            !grow_line_buffer(line, capacity))
            return 0;
        (*line)[(*length)++] = (char)ch;
        if (ch == '\n')
            break;
    }
    return !ferror(file);
}

/* Reads one complete line into an allocated buffer.
   Returns READ_OK (caller frees line_out), READ_EOF, or READ_ERROR. */
int read_line_dynamic(FILE *file, char **line_out)
{
    char *line;
    int capacity = 128, length = 0;
    line = malloc(capacity * sizeof(char));
    if (line == NULL)
        return READ_ERROR;

    if (!read_line_body(file, &line, &length, &capacity))
    {
        free(line);
        return READ_ERROR;
    }
    if (length == 0)
    {
        free(line);
        return READ_EOF;
    }
    line[length] = '\0';
    *line_out = line;
    return READ_OK;
}

/* Removes trailing whitespace in place and returns the new length. */
int trim_line(char *line)
{
    int len;

    len = (int)strlen(line);

    while (len > 0 && isspace((unsigned char)line[len - 1]))
    {
        line[len - 1] = '\0';
        len--;
    }

    return len;
}

/* Counts comma-separated fields in one non-empty input line. */
int count_coordinates(char *line)
{
    int i;
    int count = 1;

    for (i = 0; line[i] != '\0'; i++)
    {
        if (line[i] == ',')
        {
            count++;
        }
    }

    return count;
}

/* Parses d comma-separated coordinates, modifying line with strtok().
   Returns an allocated point or NULL if parsing/allocation fails. */
double *parse_point(char *line, int d)
{
    double *point;
    char *token;
    int i;

    point = malloc(d * sizeof(double));
    if (point == NULL)
    {
        return NULL;
    }

    token = strtok(line, ",");

    for (i = 0; i < d; i++)
    {
        if (!parse_double(token, &point[i]))
        {
            free(point);
            return NULL;
        }

        token = strtok(NULL, ",");
    }

    return point;
}

/* Doubles the capacity of the dynamically grown point array.
   Returns 0 on allocation failure and leaves the original array valid. */
int grow_points(double ***points, int *capacity)
{
    double **temp;

    *capacity *= 2;

    temp = realloc(
        *points,
        (*capacity) * sizeof(double *)
    );

    if (temp == NULL)
    {
        return 0;
    }

    *points = temp;
    return 1;
}

/* Parses one point and appends it to the points array.
   Uses the first point to set d; later points must have that dimension.
   Returns 1 on success and 0 on invalid input or allocation failure. */
int add_point_from_line(char *line, double ***points, int *n, int *d, int *capacity)
{
    double *point;
    int current_d;

    current_d = count_coordinates(line);

    if (*n == 0)
    {
        *d = current_d;
    }
    else if (current_d != *d)
    {
        return 0;
    }

    point = parse_point(line, *d);
    if (point == NULL)
    {
        return 0;
    }

    if (*n == *capacity &&
        !grow_points(points, capacity))
    {
        free(point);
        return 0;
    }

    (*points)[*n] = point;
    (*n)++;

    return 1;
}

/* Reads CSV points from an open file, skipping blank lines.
   Sets n and d and returns allocated points; the caller frees them.
   Returns NULL on a read, parsing, or allocation error. */
double **read_points(FILE *file, int *n, int *d)
{
    double **points;
    char *line;
    int capacity = 10;
    int status;
    *n = 0;
    *d = 0;
    points = malloc(capacity * sizeof(double *));
    if (points == NULL)
        return NULL;

    while ((status = read_line_dynamic(file, &line)) == READ_OK)
    {
        if (trim_line(line) == 0)
        {
            free(line);
            continue;
        }

        if (!add_point_from_line(line, &points, n, d, &capacity))
        {
            free(line);
            free_matrix(points, *n);
            return NULL;
        }

        free(line);
    }
    if (status == READ_ERROR)
    {
        free_matrix(points, *n);
        return NULL;
    }
    return points;
}

/* Parses a complete numeric token, allowing trailing whitespace.
   Returns 0 for a missing or invalid token and 1 on success. */
int parse_double(char *token, double *result)
{
    char *endptr;

    if (token == NULL)
        return 0;

    *result = strtod(token, &endptr);

    if (endptr == token)
        return 0;

    while (*endptr != '\0')
    {
        if (!isspace((unsigned char)*endptr))
            return 0;
        endptr++;
    }

    return 1;
}

/* Computes D or W from the n x n similarity matrix A.
   Frees A and returns a new matrix (NULL on failure). */
static double **goal_from_similarity(const char *goal, double **A, int n)
{
    double **D, **result;
    D = ddg(A, n);
    if (D == NULL)
    {
        free_matrix(A, n);
        return NULL;
    }
    if (strcmp(goal, "ddg") == 0)
    {
        free_matrix(A, n);
        return D;
    }
    result = norm(A, D, n);
    free_matrix(A, n);
    free_matrix(D, n);
    return result;
}

/* Computes the requested C goal: sym, ddg, or norm.
   Returns an allocated n x n matrix or NULL on failure/invalid goal. */
double **calculate_goal(const char *goal, double **points, int n, int d)
{
    double **A;
    if (strcmp(goal, "sym") != 0 &&
        strcmp(goal, "ddg") != 0 &&
        strcmp(goal, "norm") != 0)
        return NULL;

    A = sym(points, n, d);
    if (A == NULL)
        return NULL;
    if (strcmp(goal, "sym") == 0)
        return A;
    return goal_from_similarity(goal, A, n);
}

/* Takes goal and filename from argv, prints the requested CSV matrix.
   Returns 0 on success and 1 after printing a general error. */
int main(int argc, char *argv[])
{
    FILE *file;
    double **points, **result;
    int n, d;
    if (argc != 3)
    {
        printf("%s\n", OTHER_ERROR);
        return 1;
    }
    file = fopen(argv[2], "r");
    if (file == NULL)
    {
        printf("%s\n", OTHER_ERROR);
        return 1;
    }
    points = read_points(file, &n, &d);
    fclose(file);
    if (points == NULL || n == 0)
    {
        if (points != NULL)
            free_matrix(points, n);
        printf("%s\n", OTHER_ERROR);
        return 1;
    }
    result = calculate_goal(argv[1], points, n, d);
    if (result == NULL)
    {
        free_matrix(points, n);
        printf("%s\n", OTHER_ERROR);
        return 1;
    }
    print_matrix(result, n, n);
    free_matrix(result, n);
    free_matrix(points, n);
    return 0;
}

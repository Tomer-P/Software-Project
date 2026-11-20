#ifndef SYMNMF_H
#define SYMNMF_H

/* Matrix-returning functions allocate their results row by row.
   The caller frees a returned matrix with free_matrix(), using the
   number of rows in that result. NULL indicates failure. */

/* Builds the n x n similarity matrix for n points of dimension d.
   Diagonal entries are zero; A[i][j] = exp(-||xi - xj||^2 / 2). */
double **sym(double **points, int n, int d);

/* Returns the n x n diagonal degree matrix of an n x n similarity matrix.
   Each diagonal entry is the sum of the corresponding row. */
double **ddg(double **similarity_matrix, int n);

/* Normalizes the n x n similarity matrix using the diagonal degrees.
   An entry is zero if either corresponding degree is zero. */
double **norm(double **similarity_matrix,
              double **degree_matrix,
              int n);

/* Optimizes the n x k factor H for the n x n similarity matrix W.
   The original W and H are not modified. Returns NULL if an update fails. */
double **symnmf(double **W, double **H, int n, int k);

/* Allocates a zero-initialized rows x cols matrix, or returns NULL. */
double **allocate_matrix(int rows, int cols);

/* Releases a matrix and all of its separately allocated rows. */
void free_matrix(double **matrix, int rows);

#endif

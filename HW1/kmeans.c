#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>

#define EPSILON 0.001
#define DEFAULT_ITER 400

#define OTHER_ERROR "An Error Has Occurred"
#define K_ERROR "Incorrect number of clusters!"
#define ITER_ERROR "Incorrect maximum iteration!"

/* Function declarations for input handling, validation, clustering, and memory cleanup. */
double **read_points(int *n, int *d);
double euclidean_distance(double *p1, double *p2, int d);
int closest_centroid(double *point, double **centroids, int k, int d);
double *calculate_centroid(double **cluster, int cluster_size, int d);
int converged(double **old_centroids, double **new_centroids, int k, int d);
double **k_means(double **points, int n, int d, int k, int iter);
void print_centroids(double **centroids, int k, int d);

int requirement(double k, int n, double iter);
int invalid_input(int current_d, int expected_d);
int k_error(double k, int n);
int iter_error(double iter);
double **init_centroids(double **points, int k, int d);
double **init_empty_centroids(int k);
void free_matrix(double **matrix, int rows);
void free_clusters(double ***clusters, int k);
int parse_double(char *token, double *result);
void error_exit_kmeans(double ***clusters, int *cluster_sizes, double **centroids, int k);
void error_exit_read_points(double **points, int n, double *point);
char *read_line_dynamic(void);

/* Program entry point. */
int main(int argc, char *argv[])
{
    double **points;      /* input points */
    double **centroids;   /* final centroids */
    double k_double;      /* K before converting to int */
    double iter_double;   /* iter before converting to int */
    int k;                /* number of clusters */
    int iter;             /* maximum number of iterations */
    int n;                /* number of points */
    int d;                /* dimension of each point */
    int req_status;       /*request status */

    /* Validate the number of command line arguments. */
    if (argc < 2 || argc > 3)
    {
        printf("%s\n", OTHER_ERROR);
        exit(1);
    }
    
    /* Read all input points from standard input. */
    points = read_points(&n, &d);

    if (n == 0)
    {
        free_matrix(points, n);
        printf("%s\n", OTHER_ERROR);
        exit(1);
    }

    /* Parse the number of clusters. */
    if (!parse_double(argv[1], &k_double))
    {
        free_matrix(points, n);
        printf("%s\n", K_ERROR);
        exit(1);
    }

    /* Parse the maximum number of iterations if it was provided. */
    if (argc == 3)
    {
        if (!parse_double(argv[2], &iter_double))
        {
            free_matrix(points, n);
            printf("%s\n", ITER_ERROR);
            exit(1);
        }
    }
    else
    {
        iter_double = DEFAULT_ITER;
    }

    /* Validate k and iter after parsing them successfully. */
    req_status = requirement(k_double, n, iter_double);

    if (req_status == 1)
    {
        free_matrix(points, n);
        printf("%s\n", K_ERROR);
        exit(1);
    }

    if (req_status == 2)
    {
        free_matrix(points, n);
        printf("%s\n", ITER_ERROR);
        exit(1);
    }

    /* Convert the validated values to integers. */
    k = (int)k_double;
    iter = (int)iter_double;

    /* Run the K-means algorithm. */
    centroids = k_means(points, n, d, k, iter);

    if (centroids == NULL)
    {
        free_matrix(points, n);
        printf("%s\n", OTHER_ERROR);
        exit(1);
    }

    /* Print the final centroids. */
    print_centroids(centroids, k, d);

    /* Free the input points matrix. */
    free_matrix(points, n);

    /* Free the final centroids matrix. */
    free_matrix(centroids, k);

    return 0;
}

/* Return whether k violates the assignment restrictions. */
int k_error(double k, int n)
{
    return ((k <= 1) || (k >= n) || ((int)k != k));
}

/* Return whether the iteration limit violates the assignment restrictions. */
int iter_error(double iter)
{
    return ((iter <= 1) || (iter >= 800) || ((int)iter != iter));
}

/* Validate k and iter and return the matching error code. */
int requirement(double k, int n, double iter)
{
    if (k_error(k, n))
    {
        return 1;
    }

    if (iter_error(iter))
    {
        return 2;
    }

    return 0;
}

/* Check whether the current point has the expected dimension. */
int invalid_input(int current_d, int expected_d)
{
    return current_d != expected_d;
}

/* Read all input points and store their count and dimension. */
double **read_points(int *n, int *d)
{
    char *line;
    char *line_copy;
    double **points;
    double **temp_points;
    double *point;
    char *token;
    int capacity;
    int coord_count;
    int i;
    int len;

    *n = 0;
    *d = 0;
    capacity = 10;

    points = (double **)malloc(capacity * sizeof(double *));
    if (points == NULL)
    {
        error_exit_read_points(NULL, 0, NULL);
    }

    while ((line = read_line_dynamic()) != NULL)
    {
        len = (int)strlen(line);

        /* Remove trailing whitespace, including \n and \r. */
        while (len > 0 && isspace((unsigned char)line[len - 1]))
        {
            line[len - 1] = '\0';
            len--;
        }

        /* Ignore empty lines. */
        if (len == 0)
        {
            free(line);
            continue;
        }

        /* Reject a missing coordinate after a final comma. */
        if (line[len - 1] == ',')
        {
            free(line);
            error_exit_read_points(points, *n, NULL);
        }

        line_copy = (char *)malloc((len + 1) * sizeof(char));
        if (line_copy == NULL)
        {
            free(line);
            error_exit_read_points(points, *n, NULL);
        }

        strcpy(line_copy, line);

        coord_count = 0;
        token = strtok(line_copy, ",");

        while (token != NULL)
        {
            coord_count++;
            token = strtok(NULL, ",");
        }

        free(line_copy);

        if (coord_count == 0)
        {
            free(line);
            error_exit_read_points(points, *n, NULL);
        }

        if (*n == 0)
        {
            *d = coord_count;
        }
        else if (invalid_input(coord_count, *d))
        {
            free(line);
            error_exit_read_points(points, *n, NULL);
        }

        point = (double *)malloc((*d) * sizeof(double));
        if (point == NULL)
        {
            free(line);
            error_exit_read_points(points, *n, NULL);
        }

        token = strtok(line, ",");

        for (i = 0; i < *d; i++)
        {
            if (!parse_double(token, &point[i]))
            {
                free(line);
                error_exit_read_points(points, *n, point);
            }

            token = strtok(NULL, ",");
        }

        free(line);

        if (*n == capacity)
        {
            capacity *= 2;

            temp_points = (double **)realloc(
                points,
                capacity * sizeof(double *)
            );

            if (temp_points == NULL)
            {
                error_exit_read_points(points, *n, point);
            }

            points = temp_points;
        }

        points[*n] = point;
        (*n)++;
    }

    return points;
}

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
int converged(double **old_centroids, double **new_centroids, int k, int d)
{
    int i;

    for (i = 0; i < k; i++)
    {
        if (euclidean_distance(old_centroids[i], new_centroids[i], d) >= EPSILON)
        {
            return 0; /* not converged */
        }
    }

    return 1; /* converged */
}


/* Print the centroids with four digits after the decimal point. */
void print_centroids(double **centroids, int k, int d)
{
    int i;
    int j;

    for (i = 0; i < k; i++)
    {
        for (j = 0; j < d; j++)
        {
            printf("%.4f", centroids[i][j]);

            if (j < d - 1)
            {
                printf(",");
            }
        }

        printf("\n");
    }
}

/* Run the K-means iterations and return the final centroids. */
double **k_means(double **points, int n, int d, int k, int iter)
{
    double **centroids;
    double **new_centroids;
    double ***clusters;
    int *cluster_sizes;
    double **temp;
    int i;
    int j;
    int c;
    int best_index;

    centroids = init_centroids(points, k, d);

    if (centroids == NULL)
    {
        return NULL;
    }

    for (i = 0; i < iter; i++)
    {
        clusters = malloc(k * sizeof(double **));

        if (clusters == NULL)
        {
            error_exit_kmeans(NULL, NULL, centroids, k);
        }

        for (c = 0; c < k; c++)
        {
            clusters[c] = NULL;
        }

        cluster_sizes = calloc(k, sizeof(int));

        if (cluster_sizes == NULL)
        {
            error_exit_kmeans(clusters, NULL, centroids, k);
        }

        for (j = 0; j < n; j++)
        {
            best_index = closest_centroid(points[j], centroids, k, d);

            temp = realloc(
                clusters[best_index],
                (cluster_sizes[best_index] + 1) * sizeof(double *)
            );

            if (temp == NULL)
            {
                error_exit_kmeans(clusters, cluster_sizes, centroids, k);
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
            printf("%s\n", OTHER_ERROR);
            exit(1);
        }

        for (c = 0; c < k; c++)
        {
            if (cluster_sizes[c] == 0)
            {
                new_centroids[c] = malloc(d * sizeof(double));
                if (new_centroids[c] == NULL)
                {
                    free_clusters(clusters, k);
                    free(cluster_sizes);
                    free_matrix(centroids, k);
                    free_matrix(new_centroids, c);
                    printf("%s\n", OTHER_ERROR);
                    exit(1);
                }

                for (j = 0; j < d; j++)
                {
                    new_centroids[c][j] = centroids[c][j];
                }
            }
            else
            {
                new_centroids[c] = calculate_centroid(clusters[c], cluster_sizes[c], d);
                if (new_centroids[c] == NULL)
                {
                    free_clusters(clusters, k);
                    free(cluster_sizes);
                    free_matrix(centroids, k);
                    free_matrix(new_centroids, c);
                    printf("%s\n", OTHER_ERROR);
                    exit(1);
                }
            }
        }

        free_clusters(clusters, k);
        free(cluster_sizes);

        if (converged(centroids, new_centroids, k, d))
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

/* Parse a valid double and reject tokens with extra non-space characters. */
int parse_double(char *token, double *result)
{
    char *endptr;

    if (token == NULL)
    {
        return 0;
    }

    *result = strtod(token, &endptr);

    if (endptr == token)
    {
        return 0;
    }

    while (*endptr != '\0')
    {
        if (!isspace((unsigned char)*endptr))
        {
            return 0;
        }
        endptr++;
    }

    return 1;
}

/* Clean up memory allocated during K-means and exit with a general error. */
void error_exit_kmeans(double ***clusters, int *cluster_sizes, double **centroids, int k)
{
    if (clusters != NULL)
    {
        free_clusters(clusters, k);
    }

    if (cluster_sizes != NULL)
    {
        free(cluster_sizes);
    }

    if (centroids != NULL)
    {
        free_matrix(centroids, k);
    }

    printf("%s\n", OTHER_ERROR);
    exit(1);
}

/* Clean up memory allocated while reading input and exit with a general error. */
void error_exit_read_points(double **points, int n, double *point)
{
    if (point != NULL)
    {
        free(point);
    }

    if (points != NULL)
    {
        free_matrix(points, n);
    }

    printf("%s\n", OTHER_ERROR);
    exit(1);
}

/* Read one complete input line using a dynamically growing buffer.
 * Returns NULL only when EOF is reached before any character is read.
 */
char *read_line_dynamic(void)
{
    char *line;
    char *temp;
    int capacity;
    int length;
    int ch;

    capacity = 128;
    length = 0;

    line = (char *)malloc(capacity * sizeof(char));
    if (line == NULL)
    {
        printf("%s\n", OTHER_ERROR);
        exit(1);
    }

    while ((ch = getchar()) != EOF)
    {
        if (length + 1 >= capacity)
        {
            capacity *= 2;

            temp = (char *)realloc(line, capacity * sizeof(char));
            if (temp == NULL)
            {
                free(line);
                printf("%s\n", OTHER_ERROR);
                exit(1);
            }

            line = temp;
        }

        line[length++] = (char)ch;

        if (ch == '\n')
        {
            break;
        }
    }

    if (length == 0 && ch == EOF)
    {
        free(line);
        return NULL;
    }

    line[length] = '\0';
    return line;
}

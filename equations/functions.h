#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/**
 * Function to allocate memory for a square matrix of size n x n.
 * Will exit if memory allocation fails.
 * 
 * @param size The size of the matrix
 * 
 * @return A pointer to the allocated matrix
 */
double **allocate_matrix(int size);

/**
 * Function to free the allocated memory for a matrix
 * 
 * @param m The number of rows in the matrix
 * @param matrix The matrix to be freed
 */
void free_matrix(int m, double **matrix);

/**
 * Function to generate a random matrix of size n x n
 * 
 * @param size The size of the matrix
 * @param matrix The matrix to be filled with random values
 */
void generate_matrix(int size, double **matrix);

/**
 * Function to print the system of equations
 * 
 * @param size The size of the matrix
 * @param mat The matrix to be printed
 */
void print_equation_system(int size, double **mat);

/**
 * Function to check a solution of a system of equations
 * 
 * @param size The size of the matrix
 * @param mat The matrix to be checked
 * @param sol The solution to be checked
 * 
 * @return 1 if the solution is correct, 0 otherwise
 */
int check_equation_system(int size, double **mat, double *sol);

/**
 * Function to copy a matrix
 * 
 * @param size The size of the matrix
 * @param src The source matrix
 * @param dest The destination matrix
 */
void copy_matrix(int size, double **src, double **dest);

#endif // FUNCTIONS_H
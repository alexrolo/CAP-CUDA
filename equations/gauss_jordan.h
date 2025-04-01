#ifndef GAUSS_JORDAN_H
#define GAUSS_JORDAN_H

#include <stdio.h>
#include <math.h>

#include "functions.h"
#include "log.h"

/**
 * Function to perform Gauss-Jordan elimination
 * 
 * @param size The size of the matrix
 * @param mat The matrix to be transformed
 */
void gauss_jordan(unsigned int size, double **mat);

#endif // GAUSS_JORDAN_H
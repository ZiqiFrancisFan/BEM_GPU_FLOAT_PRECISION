/*
 ©Copyright 2020 University of Florida Research Foundation, Inc. All Commercial Rights Reserved.
 For commercial license interest please contact UF Innovate | Tech Licensing, 747 SW 2nd Avenue, P.O. Box 115575, Gainesville, FL 32601,
 Phone (352) 392-8929 and reference UF technology T18423.
 */

#ifndef DATASTRUCT_H
#define DATASTRUCT_H
    
#include <cuComplex.h>

#define IDXC0(row,column,stride) ((column)*(stride)+(row))

struct tri_elem 
{
    int nodes[3];
    cuFloatComplex bc[3];
};

struct cart_coord_float 
{
    // x, y, z in ascending order
    float coords[3];
};

struct sph_coord_float
{
    // r, theta and phi in ascending order
    float coords[3];
};

typedef struct tri_elem tri_elem;

typedef struct cart_coord_float cart_coord_float;

typedef struct sph_coord_float sph_coord_float;

struct cart_coord_double
{
    // x, y, z in ascending order
    double coords[3];
};

struct sph_coord_double
{
    // r, theta, phi in ascending order
    double coords[3];
};

typedef struct cart_coord_double cart_coord_double;

typedef struct sph_coord_double sph_coord_double;

#endif /* DATASTRUCT_H */


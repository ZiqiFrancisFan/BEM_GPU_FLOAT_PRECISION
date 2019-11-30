/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   dataStruct.h
 * Author: ziqi
 *
 * Created on February 23, 2019, 1:17 PM
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

struct cart_coord_flt 
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

typedef struct cart_coord_flt cart_coord_flt;

typedef struct sph_coord_float sph_coord_float;

struct cart_coord_dbl
{
    // x, y, z in ascending order
    double coords[3];
};

struct sph_coord_double
{
    // r, theta, phi in ascending order
    double coords[3];
};

typedef struct cart_coord_dbl cart_coord_dbl;

typedef struct sph_coord_double sph_coord_double;

struct cube_dbl
{
    cart_coord_dbl cnr;
    double len;
};

typedef struct cube_double cube_double;

#endif /* DATASTRUCT_H */


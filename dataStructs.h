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
    int nod[3];
    cuFloatComplex bc[3];
};

struct rect_coord_flt 
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

typedef struct rect_coord_flt rect_coord_flt;

typedef struct sph_coord_float sph_coord_float;

struct rect_coord_dbl
{
    // x, y, z in ascending order
    double coords[3];
};

struct sph_coord_double
{
    // r, theta, phi in ascending order
    double coords[3];
};

typedef struct rect_coord_dbl rect_coord_dbl;

typedef struct sph_coord_double sph_coord_double;

struct aa_cube_dbl
{
    rect_coord_dbl cnr;
    double len;
};

struct aa_rect_dbl
{
    rect_coord_dbl cnr;
    double len[3]; //x, y, z directions
};

struct plane_dbl
{
    rect_coord_dbl pt;
    rect_coord_dbl n;
};

struct line_dbl
{
    rect_coord_dbl pt;
    rect_coord_dbl dir;
};

struct line_segment_dbl
{
    rect_coord_dbl nod[2];
};

struct tri_dbl
{
    rect_coord_dbl nod[3];
};

struct quad_dbl
{
    rect_coord_dbl nod[4];
};

typedef struct aa_cube_dbl aa_cube_dbl;

typedef struct aa_rect_dbl aa_rect_dbl;

typedef struct plane_dbl plane_dbl;

typedef struct line_dbl line_dbl;

typedef struct line_segment_dbl ln_seg_dbl;

typedef struct quad_dbl quad_dbl;

typedef struct tri_dbl tri_dbl;

#endif /* DATASTRUCT_H */


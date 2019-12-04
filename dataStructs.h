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

/*rectangular coordinate of float precision*/
struct rect_coord_flt
{
    // x, y, z in ascending order
    float coords[3];
};

typedef struct rect_coord_flt vec3f;

/*rectangular coordinate of double precision*/
struct rect_coord_dbl
{
    // x, y, z in ascending order
    double coords[3];
};

typedef struct rect_coord_dbl vec3d;

/*spherical coordinate of double precision*/
struct sph_coord_dbl
{
    // r, theta, phi in ascending order
    double coords[3];
};

typedef struct sph_coord_dbl sph3d;

/*spherical coordinate of float precision*/
struct sph_coord_flt
{
    // r, theta and phi in ascending order
    float coords[3];
};

typedef struct sph_coord_flt sph3f;

/*triangular element*/
struct tri_elem 
{
    int nod[3]; // node index
    cuFloatComplex bc[3]; // frequency dependent boundary condition
};

typedef struct tri_elem tri_elem;

/*2D rectangular coordinate of double precision*/
struct rect_coord_2D_dbl
{
    double coords[2];
};

typedef struct rect_coord_2D_dbl vec2d;

/*2D rectangular coordinate of float precision*/
struct rect_coord_2D_flt
{
    float coords[2];
};

typedef struct rect_coord_2D_flt vec2f;

/*axis aligned cube of double precision*/
struct aa_cube_dbl
{
    vec3d cnr;
    double len;
};

typedef struct aa_cube_dbl aa_cube_dbl;

/*axis aligned rectangular volume of double precision*/
struct aa_rect_dbl
{
    vec3d cnr;
    double len[3]; //x, y, z directions
};

typedef struct aa_rect_dbl aa_rect_dbl;

/*plane of double precision*/
struct plane_dbl
{
    vec3d pt;
    vec3d n;
};

typedef struct plane_dbl plane_dbl;

/*line of double precision*/
struct line_dbl
{
    vec3d pt;
    vec3d dir;
};

typedef struct line_dbl line_dbl;

/*line segment of double precision*/
struct line_segment_dbl
{
    vec3d nod[2];
};

typedef struct line_segment_dbl ln_seg_dbl;

/*triangle of double precision*/
struct tri_dbl
{
    vec3d nod[3];
};

typedef struct tri_dbl tri_dbl;

/*quadrilateral of double precision*/
struct quad_dbl
{
    vec3d nod[4];
};

typedef struct quad_dbl quad_dbl;


struct intvl_dbl
{
    double end[2];
};

typedef struct intvl_dbl intvl_dbl;

#endif /* DATASTRUCT_H */


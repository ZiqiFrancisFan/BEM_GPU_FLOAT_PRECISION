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

#ifdef __cplusplus
extern "C" {
#endif
    
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


#ifdef __cplusplus
}
#endif

#endif /* DATASTRUCT_H */


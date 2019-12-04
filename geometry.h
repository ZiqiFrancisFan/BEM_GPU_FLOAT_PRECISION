/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   geometry.h
 * Author: ziqi
 *
 * Created on December 2, 2019, 7:04 AM
 */

#ifndef GEOMETRY_H
#define GEOMETRY_H

#include <cuda_runtime.h>
#include <curand.h>

#include <cublas_v2.h>
#include "dataStructs.h"
#include "numerical.h"

__host__ __device__ double triArea(const tri_dbl s);

__host__ __device__ int deterPtPlaneRel(const rect_coord_dbl pt, const plane_dbl plane);

__host__ __device__ int deterPtCubeRel(const rect_coord_dbl pt, const aa_cube_dbl cube);

__host__ __device__ int deterPtCubeEdgeVolRel(const rect_coord_dbl pt, const aa_cube_dbl cb);

__host__ __device__ int deterLinePlaneInt(const line_dbl ln, const plane_dbl pln, double* t);

__host__ __device__ int deterPtCubeVtxVolRel(const rect_coord_dbl pt, const aa_cube_dbl cb);

__host__ __device__ int deterLinePlaneRel(const line_dbl ln, const plane_dbl pln, double* t);

__host__ __device__ double rectCoordDet(const rect_coord_dbl vec[3]);

__host__ __device__ int deterLnLnRel(const line_dbl ln1, const line_dbl ln2, double* t1, double* t2);

__host__ __device__ int deterPtLnSegRel(const rect_coord_dbl pt, const ln_seg_dbl lnSeg);

__host__ __device__ int deterLnSegLnSegRel(const ln_seg_dbl seg1, const ln_seg_dbl seg2);

__host__ __device__ int deterLnSegQuadRel(const ln_seg_dbl lnSeg, const quad_dbl qd);

__host__ __device__ int deterLnSegQuadRel(const ln_seg_dbl lnSeg, const quad_dbl qd);

__host__ __device__ int deterLnSegTriRel(const ln_seg_dbl lnSeg, const tri_dbl tri);

__host__ __device__ int deterTriCubeInt(const tri_dbl tri, const aa_cube_dbl cb);

__host__ int voxelSpace(const aa_cube_dbl sp, const int numEachDim, const rect_coord_dbl* pt, 
        const tri_elem* elem, const int numElem, int* flag);

__host__ int write_voxels(const int* flag, const int num, const char* file_path);

#endif /* GEOMETRY_H */


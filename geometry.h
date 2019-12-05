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

__host__ void SetHostBases();

__host__ int CopyBasesToConstant();

__host__ __device__ vec2d GetMin(const aarect2d rect);

__host__ __device__ vec2d GetMax(const aarect2d rect);

__host__ __device__ void PrintVec(const vec2d* vec, const int num);

__host__ __device__ double triArea(const tri_dbl& s);

__host__ __device__ int deterPtPlaneRel(const vec3d& pt, const plane3d& plane);

__host__ __device__ int deterPtCubeRel(const vec3d& pt, const aacb3d& cube);

__host__ __device__ int deterPtCubeEdgeVolRel(const vec3d& pt, const aacb3d& cb);

__host__ __device__ int deterLinePlaneInt(const line3d& ln, const plane3d& pln, double* t);

__host__ __device__ int deterPtCubeVtxVolRel(const vec3d& pt, const aacb3d& cb);

__host__ __device__ int deterLinePlaneRel(const line3d& ln, const plane3d& pln, double* t);

__host__ __device__ double rectCoordDet(const vec3d vec[3]);

__host__ __device__ int deterLnLnRel(const line3d& ln1, const line3d& ln2, double* t1, double* t2);

__host__ __device__ int deterPtLnSegRel(const vec3d& pt, const lnseg3d& lnSeg);

__host__ __device__ int deterLnSegLnSegRel(const lnseg3d& seg1, const lnseg3d& seg2);

__host__ __device__ int deterLnSegQuadRel(const lnseg3d& lnSeg, const quad_dbl& qd);

__host__ __device__ int deterLnSegQuadRel(const lnseg3d& lnSeg, const quad_dbl& qd);

__host__ __device__ int deterLnSegTriRel(const lnseg3d& lnSeg, const tri_dbl& tri);

__host__ __device__ int deterTriCubeInt(const tri_dbl& tri, const aacb3d& cb);

__host__ int voxelSpace(const aacb3d sp, const int numEachDim, const vec3d* pt, 
        const tri_elem* elem, const int numElem, int* flag);

__host__ int write_voxels(const int* flag, const int num, const char* file_path);

__host__ __device__ bool IntvlIntvlOvlp(const intvl2d intvl1, const intvl2d intvl2);

__host__ __device__ bool AaRectAaRectOvlp(const aarect2d rect1, const aarect2d rect2);

__host__ __device__ vec3d GetMin(const aarect3d& rect);

__host__ __device__ vec3d GetMax(const aarect3d& rect);

__host__ __device__ intvl3d GetInterval(const aarect3d& rect, const vec3d& axis);

__host__ __device__ intvl3d GetInterval(const tri3d& tri, const vec3d& ax);

__host__ __device__ bool OverlapTriangleAARect(const tri3d& tri, const aarect3d& rect);

__host__ int SpaceVoxelOnCPU(const aarect3d sp, const double voxlen, const vec3d* pt, 
        const tri_elem* elem, const int numElem, bool* flag);

__host__ int SpaceVoxelOnGPU(const aarect3d sp, const double voxlen, const vec3d* pt, 
        const tri_elem* elem, const int numElem, bool* flag);

__host__ int write_voxels(const bool* flag, const int numvox[3], const char* file_path);

#endif /* GEOMETRY_H */


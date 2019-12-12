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

/*determines the relation between a point and a plane*/
__host__ __device__ int DeterPtPlaneRel(const vec3d& pt, const plane3d& plane);

/*determines the relation between a point and a cube*/
__host__ __device__ int DeterPtCubeRel(const vec3d& pt, const aacb3d& cube);

__host__ __device__ int DeterPtCubeEdgeVolRel(const vec3d& pt, const aacb3d& cb);

__host__ __device__ int DeterPtCubeVtxVolRel(const vec3d& pt, const aacb3d& cb);

/*determines relation between a line and a plane*/
__host__ __device__ int DeterLinePlaneRel(const line3d& ln, const plane3d& pln, double* t);

/*calculate the determinant of a matrix formed by three coordinates as columns*/
__host__ __device__ double rectCoordDet(const vec3d vec[3]);

/*determines the relation between two lines*/
__host__ __device__ int DeterLnLnRel(const line3d& ln1, const line3d& ln2, double* t1, double* t2);

/*determines the relation between a point and a line segment*/
__host__ __device__ int DeterPtLnSegRel(const vec3d& pt, const lnseg3d& lnSeg);

/*determines the relation between two line segments*/
__host__ __device__ int DeterLnSegLnSegRel(const lnseg3d& seg1, const lnseg3d& seg2);

/*determines the relation between a line segment and a quadrilateral polygon*/
__host__ __device__ int DeterLnSegQuadRel(const lnseg3d& lnSeg, const quad_dbl& qd);

/*determines the relation between a line segment and a triangle*/
__host__ __device__ int DeterLnSegTriRel(const lnseg3d& lnSeg, const tri_dbl& tri);

/*determines the intersection between a triangle and a cube*/
__host__ __device__ int DeterTriCubeInt(const tri_dbl& tri, const aacb3d& cb);

__host__ __device__ int DeterTriAaCbRel(const tri3d& tri, const aacb3d& cb);

__host__ __device__ bool IntvlIntvlOvlp(const intvl2d intvl1, const intvl2d intvl2);

__host__ __device__ bool AaRectAaRectOvlp(const aarect2d rect1, const aarect2d rect2);

__host__ __device__ int DeterPtEdgePlaneRel(const vec3d& pt, const aacb3d& cb, const int i, 
        const int a[2]);

__host__ __device__ int DeterTriEdgePlaneRel(const tri3d& tri, const aacb3d& cb, 
        const int i, const int a[2]);

__host__ __device__ int DeterTriCubeEdgePlaneRel(const tri3d& tri, const aacb3d& cb);

__host__ __device__ int DeterPtVtxPlaneRel(const vec3d& pt, const aacb3d& cb, const int a[3]);

__host__ __device__ int DeterTriVtxPlaneRel(const tri3d& tri, const aacb3d& cb, const int a[3]);

__host__ __device__ int DeterTriCubeVtxPlaneRel(const tri3d& tri, const aacb3d& cb);

__host__ __device__ int DeterPtAaCbRel(const vec3d& pt, const aacb3d& cb);

__host__ __device__ int DeterLnSetAaCbFaceRel(const lnseg3d& lsg, const aacb3d& cb, 
        const int i, const double c);

__host__ __device__ int DeterTriAaCbRel(const tri3d& tri, const aacb3d& cb);

/*get the minimum corner of a 3D rectangle*/
__host__ __device__ vec3d GetMin(const aarect3d& rect);

/*get the maximum corner of a 3D rectangle*/
__host__ __device__ vec3d GetMax(const aarect3d& rect);

/*get the projection interval of an axis-aligned rectangle on an axis*/
__host__ __device__ intvl3d GetInterval(const aarect3d& rect, const vec3d& axis);

/*get the projection interval of an triangle on an axis*/
__host__ __device__ intvl3d GetInterval(const tri3d& tri, const vec3d& ax);

/*determines if a 3D triangle overlaps an 3D axis-aligned rectangle*/
__host__ __device__ bool OverlapTriangleAARect(const tri3d& tri, const aarect3d& rect);

/*convert a cube space into a voxelized occupancy grid using the accurate but slow approach*/
__host__ int CubeSpaceVoxelOnCPU(const aarect3d sp, const double voxlen, const vec3d* pt, 
        const tri_elem* elem, const int numElem, bool* flag);

/*convert a cube space to an occupancy grid on GPU using the accurate but slow approach*/
__host__ int CubeSpaceVoxelOnGPU(const aacb3d sp, const int numEachDim, const vec3d* pt, 
        const tri_elem* elem, const int numElem, int* flag);

/*convert a rectangular sapce into a voxelized occupancy grid using the separating axis theorem*/
__host__ int RectSpaceVoxelSATOnGPU(const aarect3d sp, const double voxlen, const vec3d* pt, 
        const tri_elem* elem, const int numElem, const char* filename);

/*converts a rectangular sapce to an occupancy grid on GPU using the accurate but slow approach*/
__host__ int RectSpaceToOccGridOnGPU(const aarect3d sp, const double len, const vec3d* pt, 
        const tri_elem* elem, const int numElem, const char* filePath);

/*write an occupancy grid to file*/
__host__ int write_voxels(const bool* flag, const int numvox[3], const char* file_path);

__host__ int write_voxels(int* flag, const int numvox[3], const char* file_path);

__host__ int write_field(const cuFloatComplex* field, const int numvox[3], const char* file_path);

int write_float_grid(const float* field, const int numvox[3], const char* file_path);

#endif /* GEOMETRY_H */


/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   numerical.h
 * Author: ziqi
 *
 * Created on February 23, 2019, 2:35 PM
 */

#ifndef NUMERICAL_H
#define NUMERICAL_H

#include <gsl/gsl_sf.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_blas.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>

#include <cublas_v2.h>
#include <cusolverDn.h>

#include "dataStructs.h"

#include <gsl/gsl_sf.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_blas.h>

#ifndef max
#define max(a,b) \
({ __typeof__ (a) _a = (a); \
__typeof__ (b) _b = (b); \
_a > _b ? _a : _b; })
#endif

#ifndef min
#define min(a,b) \
({ __typeof__ (a) _a = (a); \
__typeof__ (b) _b = (b); \
_a < _b ? _a : _b; })
#endif

#ifndef abs
#define abs(x) \
({ __typeof__ (x) _x = (x); \
__typeof__ (x) _y = 0; \
_x < _y ? -_x : _x; })
#endif

#ifndef PI
#define PI 3.1415926535897932f
#endif

#ifndef STRENGTH
#define STRENGTH (0.1)
#endif

#ifndef INTORDER
#define INTORDER 3
#endif

#ifndef NUMCHIEF
#define NUMCHIEF 5
#endif

#ifndef RHO_AIR
#define RHO_AIR 1.2041f
#endif

#ifndef SPEED_SOUND
#define SPEED_SOUND 343.21f
#endif

extern __constant__ float density;

extern __constant__ float speed;

//Integral points and weights
extern __constant__ float INTPT[INTORDER]; 

extern __constant__ float INTWGT[INTORDER];

#ifndef IDXC0
#define IDXC0(row,col,ld) ((ld)*(col)+(row))
#endif

#ifndef HOST_CALL
#define HOST_CALL(x) do {\
if(x!=EXIT_SUCCESS){\
printf("Error at %s:%d\n",__FILE__,__LINE__);\
return EXIT_FAILURE;}} while(0)
#endif

#ifndef CUDA_CALL
#define CUDA_CALL(x) do {\
if((x)!=cudaSuccess) {\
printf("Error at %s:%d\n",__FILE__,__LINE__);\
return EXIT_FAILURE;}} while(0)
#endif

#ifndef CURAND_CALL
#define CURAND_CALL(x) do {\
if((x)!=CURAND_STATUS_SUCCESS) {\
printf("Error at %s:%d\n",__FILE__,__LINE__);\
return EXIT_FAILURE;}} while(0)
#endif

#ifndef CUBLAS_CALL
#define CUBLAS_CALL(x) \
do {\
if((x)!=CUBLAS_STATUS_SUCCESS)\
{\
printf("Error at %s:%d\n",__FILE__,__LINE__); \
if(x==CUBLAS_STATUS_NOT_INITIALIZED) { \
printf("The library was not initialized.\n"); \
}\
if(x==CUBLAS_STATUS_INVALID_VALUE) {\
printf("There were problems with the parameters.\n");\
}\
if(x==CUBLAS_STATUS_MAPPING_ERROR) {\
printf("There was an error accessing GPU memory.\n"); \
}\
return EXIT_FAILURE; } \
}\
while(0)
#endif

#ifndef CUSOLVER_CALL
#define CUSOLVER_CALL(x) \
do {\
if((x)!=CUSOLVER_STATUS_SUCCESS)\
{\
printf("Error at %s:%d\n",__FILE__,__LINE__); \
if((x)==CUSOLVER_STATUS_NOT_INITIALIZED) { \
printf("The library was not initialized.\n"); \
}\
if((x)==CUSOLVER_STATUS_INVALID_VALUE) {\
printf("Invalid parameters were passed.\n");\
}\
if((x)==CUSOLVER_STATUS_ARCH_MISMATCH) {\
printf("Achitecture not supported.\n"); \
}\
if((x)==CUSOLVER_STATUS_INTERNAL_ERROR) {\
printf("An internal operation failed.\n"); \
}\
return EXIT_FAILURE; } \
}\
while(0)
#endif

#ifndef EPS
#define EPS 0.0000001
#endif

int genGaussParams(const int n, float *pt, float *wgt);

int gaussPtsToDevice(const float *evalPt, const float *wgt);

cart_coord_float cartCoordDouble2cartCoordFloat(const cart_coord_double t);

cart_coord_double cartCoordFloat2cartCoordDouble(const cart_coord_float t);

__host__ __device__ cart_coord_double triCentroid(cart_coord_double nod[]);

void printFltMat(const float *A, const int numRow, const int numCol, const int lda);

void printCuFloatComplexMat(const cuFloatComplex *A, const int numRow, const int numCol, 
        const int lda);

__host__ __device__ sph_coord_float cart2sph(const cart_coord_float s);

__host__ __device__ cart_coord_float scalarProd(const float lambda, const cart_coord_float v);

__host__ __device__ cart_coord_float crossProd(const cart_coord_float u, const cart_coord_float v);

__host__ __device__ cart_coord_float cartCoordAdd(const cart_coord_float u, const cart_coord_float v);

__host__ __device__ cart_coord_float cartCoordSub(const cart_coord_float u, const cart_coord_float v);

__host__ __device__ bool ray_intersect_triangle(const cart_coord_float O, const cart_coord_float dir, 
        const cart_coord_float nod[3]);

__global__ void rayTrisInt(const cart_coord_float pt_s, const cart_coord_float dir, const cart_coord_float *nod, 
        const tri_elem *elem, const int numElem, bool *flag);

int genCHIEF(const cart_coord_float *pt, const int numPt, const tri_elem *elem, const int numElem, 
        cart_coord_float *pCHIEF, const int numCHIEF);

int atomicGenSystem(const float k, const tri_elem *elem, const int numElem, 
        const cart_coord_float *pt, const int numNod, const cart_coord_float *chief, const int numCHIEF, 
        const cart_coord_float *src, const int numSrcs, cuFloatComplex *A, const int lda, 
        cuFloatComplex *B, const int ldb);

int qrSolver(const cuFloatComplex *A, const int mA, const int nA, const int ldA, 
        cuFloatComplex *B, const int nB, const int ldB);

int bemSolver_pt(const float k, const tri_elem *elem, const int numElem, 
        const cart_coord_float *nod, const int numNod, const cart_coord_float *chief, const int numCHIEF, 
        const cart_coord_float *src, const int numSrc, cuFloatComplex *B, const int ldb);

int bemSolver_dir(const float k, const tri_elem *elem, const int numElem, 
        const cart_coord_float *nod, const int numNod, const cart_coord_float *chief, const int numCHIEF, 
        const cart_coord_float *dir, const int numSrc, cuFloatComplex *B, const int ldb);

int bemSolver_mp(const float k, const tri_elem *elem, const int numElem, 
        const cart_coord_float *nod, const int numNod, const cart_coord_float *chief, const int numCHIEF, 
        const cart_coord_float *src, const int numSrc, cuFloatComplex *B, const int ldb);

void rigidSpherePlaneMultipleEval(const cart_coord_float *pt, const int numPt, const double a, 
        const double r, const double wavNum, const double strength);

gsl_complex rigid_sphere_plane(const double wavNum, const double strength, const double a, 
        const double r, const double theta);

gsl_complex rigid_sphere_point(const double wavNum, const double strength, const double rs, 
        const double a, const cart_coord_double y);

gsl_complex rigid_sphere_monopole(const double wavNum, const double strength, const double rs, 
        const double a, const cart_coord_double y);

int extrapolation_dirs_single_source(const float wavNum, const cart_coord_float* expPt, const int numExpPt, 
        const tri_elem* elem, const int numElem, const cart_coord_float* pt, const int numPt, 
        const cuFloatComplex* p, const float strength, const cart_coord_float dir, cuFloatComplex *pExp);

int field_extrapolation_single_pt(const float wavNum, const cart_coord_float* expPt, const int numExpPt, 
        const tri_elem* elem, const int numElem, const cart_coord_float* pt, const int numPt, 
        const cuFloatComplex* p, const float strength, const cart_coord_float src, cuFloatComplex *pExp);

int field_extrapolation_single_mp(const float wavNum, const cart_coord_float* expPt, const int numExpPt, 
        const tri_elem* elem, const int numElem, const cart_coord_float* pt, const int numPt, 
        const cuFloatComplex* p, const float strength, const cart_coord_float src, cuFloatComplex *pExp);

#endif /* NUMERICAL_H */


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

#include "dataStruct.h"

#ifndef PI
#define PI 3.1415926535897932f
#endif

#ifndef STRENGTH
#define STRENGTH 0.0f
#endif

#ifndef INTORDER
#define INTORDER 3
#endif

#ifndef NUMCHIEF
#define NUMCHIEF 5
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

void printFltMat(const float *A, const int numRow, const int numCol, const int lda);

void printCuFloatComplexMat(const cuFloatComplex *A, const int numRow, const int numCol, 
        const int lda);

__host__ __device__ cartCoord scalarProd(const float lambda, const cartCoord v);

__host__ __device__ cartCoord crossProd(const cartCoord u, const cartCoord v);

__host__ __device__ cartCoord cartCoordAdd(const cartCoord u, const cartCoord v);

__host__ __device__ cartCoord cartCoordSub(const cartCoord u, const cartCoord v);

__host__ __device__ bool ray_intersect_triangle(const cartCoord O, const cartCoord dir, 
        const cartCoord nod[3]);

__global__ void rayTrisInt(const cartCoord pt_s, const cartCoord dir, const cartCoord *nod, 
        const triElem *elem, const int numElem, bool *flag);

int genCHIEF(const cartCoord *pt, const int numPt, const triElem *elem, const int numElem, 
        cartCoord *pCHIEF, const int numCHIEF);

int atomicGenSystem(const float k, const triElem *elem, const int numElem, 
        const cartCoord *pt, const int numNod, const cartCoord *chief, const int numCHIEF, 
        const cartCoord *src, const int numSrcs, cuFloatComplex *A, const int lda, 
        cuFloatComplex *B, const int ldb);

int qrSolver(const cuFloatComplex *A, const int mA, const int nA, const int ldA, 
        cuFloatComplex *B, const int nB, const int ldB);

int bemSolver(const float k, const triElem *elem, const int numElem, 
        const cartCoord *nod, const int numNod, const cartCoord *chief, const int numCHIEF, 
        const cartCoord *src, const int numSrc, cuFloatComplex *B, const int ldb);



#endif /* NUMERICAL_H */


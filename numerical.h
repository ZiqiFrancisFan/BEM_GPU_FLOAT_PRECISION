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

//#include <gsl/gsl_sf.h>
//#include <gsl/gsl_complex_math.h>
//#include <gsl/gsl_blas.h>

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
#define STRENGTH 1.0f
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
#define EPS 0.000000001
#endif

int genGaussParams(const int n, float *pt, float *wgt);

__host__ void vecd2f(const vec3d* vec, const int len, vec3f* vecf);

int cuGenGaussParams(const int n, float* pt, float* wgt);

int gaussPtsToDevice(const float *evalPt, const float *wgt);

__host__ __device__ vec3d triCentroid(vec3d nod[]);

void print_float_mat(const float *A, const int numRow, const int numCol, const int lda);

void print_cuFloatComplex_mat(const cuFloatComplex *A, const int numRow, const int numCol, 
        const int lda);

__host__ __device__ void printVec(const vec3f* pt, const int numPt);

__host__ __device__ void printVec(const vec3d* pt, const int numPt);

__host__ __device__ sph3f vec2sph(const vec3f s);

__host__ __device__ sph3d vec2sph(const vec3d s);

__host__ __device__ vec3f sph2vec(const sph3f s);

__host__ __device__ vec3d sph2vec(const sph3d s);

__host__ __device__ float vecDotMul(const vec3f& u, const vec3f& v);

__host__ __device__ double vecDotMul(const vec3d& u, const vec3d& v);

__host__ __device__ float vecDotMul(const vec2f& u, const vec2f& v);

__host__ __device__ double vecDotMul(const vec2d& u, const vec2d& v);

__host__ __device__ vec3f scaVecMul(const float& lambda, const vec3f& v);

__host__ __device__ vec3d scaVecMul(const double& lambda, const vec3d& v);

__host__ __device__ vec2d scaVecMul(const double& lambda, const vec2d& v);

__host__ __device__ vec2f scaVecMul(const double& lambda, const vec2f& v);

__host__ __device__ vec3f vecCrossMul(const vec3f& a, const vec3f& b);

__host__ __device__ vec3d vecCrossMul(const vec3d& a, const vec3d& b);

__host__ __device__ vec3f vecAdd(const vec3f& u, const vec3f& v);

__host__ __device__ vec3d vecAdd(const vec3d& u, const vec3d& v);

__host__ __device__ vec2d vecAdd(const vec2d& u, const vec2d& v);

__host__ __device__ vec2f vecAdd(const vec2f& u, const vec2f& v);

__host__ __device__ vec3f vecSub(const vec3f& u, const vec3f& v);

__host__ __device__ vec3d vecSub(const vec3d& u, const vec3d& v);

__host__ __device__ vec2d vecSub(const vec2d& u, const vec2d& v);

__host__ __device__ vec2f vecSub(const vec2f& u, const vec2f& v);

__host__ __device__ vec3d vecNrmlz(const vec3d& v);

__host__ __device__ vec3f vecNrmlz(const vec3f& v);

__host__ __device__ float vecNorm(const vec3f& v);

__host__ __device__ double vecNorm(const vec3d& v);

__host__ __device__ void printMat(const double* mat, const int numRow, const int numCol, 
        const int lda);

__host__ __device__ void matRowSwap(double* mat, const int numCol, const int lda, 
        const int i, const int j);

__host__ __device__ void scaRowMul(double* mat, const int numCol, const int lda, 
        const int ridx, const double c);

__host__ __device__ void subScaRowFromRow(double* mat, const int numCol, const int lda, 
        const int i, const int j, const double c);

__host__ __device__ int vecEqual(const vec3f& v1, const vec3f& v2);

__host__ __device__ int vecEqual(const vec3d& v1, const vec3d& v2);

__host__ __device__ void GaussElim(double* mat, const int numRow, const int numCol, const int lda);

__host__ __device__ bool ray_intersect_triangle(const vec3f O, const vec3f dir, 
        const vec3f nod[3]);

__global__ void rayTrisInt(const vec3f pt_s, const vec3f dir, const vec3f *nod, 
        const tri_elem *elem, const int numElem, bool *flag);

int genCHIEF(const vec3f *pt, const int numPt, const tri_elem *elem, const int numElem, 
        vec3f *pCHIEF, const int numCHIEF);

int atomicGenSystem(const float k, const tri_elem *elem, const int numElem, 
        const vec3f *pt, const int numNod, const vec3f *chief, const int numCHIEF, 
        const vec3f *src, const int numSrcs, cuFloatComplex *A, const int lda, 
        cuFloatComplex *B, const int ldb);

int qrSolver(const cuFloatComplex *A, const int mA, const int nA, const int ldA, 
        cuFloatComplex *B, const int nB, const int ldB);

int bemSolver_pt(const float k, const tri_elem *elem, const int numElem, 
        const vec3f *nod, const int numNod, const vec3f *chief, const int numCHIEF, 
        const vec3f *src, const float* strength, const int numSrc, cuFloatComplex *B, const int ldb);

int bemSolver_dir(const float k, const tri_elem *elem, const int numElem, 
        const vec3f *nod, const int numNod, const vec3f *chief, const int numCHIEF, 
        const vec3f *dir, const float* strength, const int numSrc, cuFloatComplex *B, const int ldb);

int bemSolver_mp(const float k, const tri_elem *elem, const int numElem, 
        const vec3f *nod, const int numNod, const vec3f *chief, const int numCHIEF, 
        const vec3f *src, const float* strength, const int numSrc, cuFloatComplex *B, const int ldb);

__device__ cuFloatComplex extrapolation_mp(const float wavNum, const vec3f x, 
        const tri_elem* elem, const int numElem, const vec3f* pt, 
        const cuFloatComplex* p, const float& strength, const vec3f& src);

__global__ void extrap_mp_sgl_src(const float wavNum, const vec3f* expPt, const int numExpPt,
        const tri_elem* elem, const int numElem, const vec3f* pt, const cuFloatComplex* p, 
        const float strength, const vec3f src, cuFloatComplex *p_exp);

__global__ void extrap_mp_multi_src(const float wavNum, const vec3f* pt_extrap, const int numExtrap, 
        const tri_elem* elem, const int numElem, const vec3f* pt, const cuFloatComplex* B, 
        const int ldb, const float* strength, const vec3f* src, const int numSrc, cuFloatComplex* prsr);

void computeRigidSphereScattering(const vec3f *pt, const int numPt, const double a, 
        const double r, const double wavNum, const double strength);

gsl_complex rigidSphereScattering(const double wavNum, const double strength, const double a, 
        const double r, const double theta);

int field_extrapolation_single_dir(const float wavNum, const vec3f* expPt, const int numExpPt, 
        const tri_elem* elem, const int numElem, const vec3f* pt, const int numPt, 
        const cuFloatComplex* p, const float strength, const vec3f dir, cuFloatComplex *pExp);

int field_extrapolation_single_pt(const float wavNum, const vec3f* expPt, const int numExpPt, 
        const tri_elem* elem, const int numElem, const vec3f* pt, const int numPt, 
        const cuFloatComplex* p, const float strength, const vec3f dir, cuFloatComplex *pExp);

int genFields_MultiPtSrcSglObj(const float strength, const float wavNum, 
        const vec3f* srcs, const int numSrcs, const vec3d* pts, const int numPts, 
        const tri_elem* elems, const int numElems, const vec3d cnr, const double d, 
        const int level, cuFloatComplex* fields);

gsl_complex rigid_sphere_point(const double wavNum, const double strength, const double rs, 
        const double a, const vec3d y);

gsl_complex rigid_sphere_monopole(const double wavNum, const double strength, const double rs, 
        const double a, const vec3d y);

int field_extrapolation_single_pt(const float wavNum, const vec3f* expPt, const int numExpPt, 
        const tri_elem* elem, const int numElem, const vec3f* pt, const int numPt, 
        const cuFloatComplex* p, const float strength, const vec3f src, cuFloatComplex *pExp);

int field_extrapolation_single_mp(const float wavNum, const vec3f* expPt, const int numExpPt, 
        const tri_elem* elem, const int numElem, const vec3f* pt, const int numPt, 
        const cuFloatComplex* p, const float strength, const vec3f src, cuFloatComplex *pExp);

int GenerateFieldUsingBEM(const vec3f* nod, const int numNod, const tri_elem* elem, const int numElem,
        const float wavNum, const char* src_type, const vec3f* src_loc, const float* mag, const int numSrc, 
        const vec3f* pt_extrap, const int numExtrap, cuFloatComplex* prsr);

int GenerateVoxelField(const char* file_path, const float wavNum, const char* src_type, 
        const vec3f* src_loc, const float* mag, const int numSrc, const aarect3d rect, 
        const double len, const char* vox_grid_path, const char* field_grid_path);

int WriteLoudnessGeometry(const char* file_path, const float band[2], const char* src_type, 
        const float* mag, const vec3f* src_loc, const int numSrc, const aarect3d rect, 
        const double len, const char* vox_grid_path, const char* field_grid_path);

#endif /* NUMERICAL_H */


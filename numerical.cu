/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
#include "numerical.h"
#include "octree.h"
#include "mesh.h"
#include <math_constants.h>
#include <cusolverDn.h>

//air density and speed of sound
__constant__ float density = 1.2041;

__constant__ float speed = 343.21;

//Integral points and weights
__constant__ float INTPT[INTORDER]; 

__constant__ float INTWGT[INTORDER];
/*
int genGaussParams(const int n, float* pt, float* wgt) 
{
    int i, j;
    double t;
    gsl_vector *v = gsl_vector_alloc(n);
    for(i=0;i<n-1;i++) {
        gsl_vector_set(v,i,sqrt(pow(2*(i+1),2)-1));
    }
    for(i=0;i<n-1;i++) {
        t = gsl_vector_get(v,i);
        gsl_vector_set(v,i,(i+1)/t);
    }
    gsl_matrix *A = gsl_matrix_alloc(n,n);
    gsl_matrix *B = gsl_matrix_alloc(n,n);
    for(i=0;i<n;i++) {
        for(j=0;j<n;j++) {
            gsl_matrix_set(A,i,j,0);
            if(i==j) {
                gsl_matrix_set(B,i,j,1);
            } else {
                gsl_matrix_set(B,i,j,0);
            }
        }
    }
    for(i=0;i<n-1;i++) {
        t = gsl_vector_get(v,i);
        gsl_matrix_set(A,i+1,i,t);
        gsl_matrix_set(A,i,i+1,t);
    }
    gsl_eigen_symmv_workspace * wsp = gsl_eigen_symmv_alloc(n);
    HOST_CALL(gsl_eigen_symmv(A,v,B,wsp));
    for(i=0;i<n;i++) {
        pt[i] = gsl_vector_get(v,i);
        t = gsl_matrix_get(B,0,i);
        wgt[i] = 2*pow(t,2);
    }
    gsl_vector_free(v);
    gsl_matrix_free(A);
    gsl_matrix_free(B);
    return EXIT_SUCCESS;
}
*/

int cuGenGaussParams(const int n, float* pt, float* wgt)
{
    cusolverDnHandle_t handle;
    CUSOLVER_CALL(cusolverDnCreate(&handle));
    
    // allocate memory for vector v of length n
    float *v = (float*)malloc(n*sizeof(float));
    
    // set the vector v
    for(int i=0;i<n-1;i++) {
        v[i] = sqrt(pow(2*(i+1),2)-1);
    }
    for(int i=0;i<n-1;i++) {
        float t = v[i];
        v[i] = (i+1)/t;
    }
    //printf("The vector v is set properly.\n");
    
    float *A = (float*)malloc(n*n*sizeof(float));
    memset(A,0,n*n*sizeof(float));
    
    // set up matrix A
    for(int i=0;i<n-1;i++) {
        float t = v[i];
        A[IDXC0(i+1,i,n)] = t;
        A[IDXC0(i,i+1,n)] = t;
    }
    
    //printf("The matrix A is set properly.\n");
    
    float *A_d, *Lambda_d;
    CUDA_CALL(cudaMalloc(&A_d,n*n*sizeof(float)));
    //printf("A_d allocated.\n");
    CUDA_CALL(cudaMemcpy(A_d,A,n*n*sizeof(float),cudaMemcpyHostToDevice));
    //printf("A copied to A_d.\n");
    CUDA_CALL(cudaMalloc(&Lambda_d,n*sizeof(float)));
    //printf("Lambda_d allocated successfully.\n");
    
    int lwork;
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvalues and eigenvectors.
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
    CUSOLVER_CALL(cusolverDnSsyevd_bufferSize(handle,jobz,
            uplo,n,A_d,n,Lambda_d,&lwork));
    //printf("Buffer is set up.\n");
    float *work_d;
    CUDA_CALL(cudaMalloc(&work_d,lwork*sizeof(float)));
    int *devInfo;
    CUDA_CALL(cudaMalloc(&devInfo,sizeof(int)));
    CUSOLVER_CALL(cusolverDnSsyevd(handle,jobz,uplo,n,A_d,n,Lambda_d,work_d,lwork,devInfo));
    //printf("Eigenvalues and eigenvectors found.\n");
    float *Lambda = (float*)malloc(n*sizeof(float));
    CUDA_CALL(cudaMemcpy(A,A_d,n*n*sizeof(float),cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(Lambda,Lambda_d,n*sizeof(float),cudaMemcpyDeviceToHost));
    
    memcpy(pt,Lambda,n*sizeof(float));
    for(int i=0;i<n;i++) {
        float t = A[IDXC0(0,i,n)];
        wgt[i] = 2*pow(t,2);
    }
    
    if(A_d) {
        CUDA_CALL(cudaFree(A_d));
    }
    if(Lambda_d) {
        CUDA_CALL(cudaFree(Lambda_d));
    }
    if(work_d) {
        CUDA_CALL(cudaFree(work_d));
    }
    if(devInfo) {
        CUDA_CALL(cudaFree(devInfo));
    }
    if(handle) {
        CUSOLVER_CALL(cusolverDnDestroy(handle));
    }
    
    free(v);
    free(Lambda);
    free(A);
    
    
    return EXIT_SUCCESS;
}

int gaussPtsToDevice(const float *evalPt, const float *wgt) 
{
    CUDA_CALL(cudaMemcpyToSymbol(INTPT,evalPt,INTORDER*sizeof(float),0,cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpyToSymbol(INTWGT,wgt,INTORDER*sizeof(float),0,cudaMemcpyHostToDevice));
    return EXIT_SUCCESS;
}

void print_float_mat(const float *A, const int numRow, const int numCol, const int lda) 
{
    for(int i=0;i<numRow;i++) {
        for(int j=0;j<numCol;j++) {
            printf("%f ",A[IDXC0(i,j,lda)]);
        }
        printf("\n");
    }
}

void print_cuFloatComplex_mat(const cuFloatComplex *A, const int numRow, const int numCol, 
        const int lda)
{
    for(int i=0;i<numRow;i++) {
        for(int j=0;j<numCol;j++) {
            printf("(%f,%f) ",cuCrealf(A[IDXC0(i,j,lda)]),cuCimagf(A[IDXC0(i,j,lda)]));
        }
        printf("\n");
    }
}

__host__ __device__ void printRectCoord(const rect_coord_flt* pt, const int numPt)
{
    for(int i=0;i<numPt;i++) {
        printf("(%f,%f,%f), ",pt[i].coords[0],pt[i].coords[1],pt[i].coords[2]);
    }
    printf("\n");
}

__host__ __device__ void printRectCoord(const rect_coord_dbl* pt, const int numPt)
{
    for(int i=0;i<numPt;i++) {
        printf("(%f,%f,%f), ",pt[i].coords[0],pt[i].coords[1],pt[i].coords[2]);
    }
    printf("\n");
}

__host__ __device__ float rectDotMul(const rect_coord_flt u, const rect_coord_flt v)
{
    return u.coords[0]*v.coords[0]+u.coords[1]*v.coords[1]+u.coords[2]*v.coords[2];
}

__host__ __device__ double rectDotMul(const rect_coord_dbl u, const rect_coord_dbl v)
{
    return u.coords[0]*v.coords[0]+u.coords[1]*v.coords[1]+u.coords[2]*v.coords[2];
}

__host__ __device__ float rectNorm(const rect_coord_flt v)
{
    return sqrtf(rectDotMul(v,v));
}

__host__ __device__ double rectNorm(const rect_coord_dbl v)
{
    return sqrt(rectDotMul(v,v));
}

__host__ __device__ rect_coord_flt rectCrossMul(const rect_coord_flt a, const rect_coord_flt b)
{
    rect_coord_flt temp;
    temp.coords[0] = a.coords[1]*b.coords[2]-a.coords[2]*b.coords[1];
    temp.coords[1] = -(a.coords[0]*b.coords[2]-a.coords[2]*b.coords[0]);
    temp.coords[2] = a.coords[0]*b.coords[1]-a.coords[1]*b.coords[0];
    return temp;
}

__host__ __device__ rect_coord_dbl rectCrossMul(const rect_coord_dbl a, const rect_coord_dbl b)
{
    rect_coord_dbl temp;
    temp.coords[0] = a.coords[1]*b.coords[2]-a.coords[2]*b.coords[1];
    temp.coords[1] = -(a.coords[0]*b.coords[2]-a.coords[2]*b.coords[0]);
    temp.coords[2] = a.coords[0]*b.coords[1]-a.coords[1]*b.coords[0];
    return temp;
}

__host__ __device__ rect_coord_dbl nrmlzRectCoord(const rect_coord_dbl v)
{
    double nrm = sqrt(rectDotMul(v,v));
    return scaRectMul(1.0/nrm,v);
}

__host__ __device__ rect_coord_flt nrmlzRectCoord(const rect_coord_flt v)
{
    float nrm = sqrt(rectDotMul(v,v));
    return scaRectMul(1.0/nrm,v);
}

__host__ __device__ int equalRectCoord(const rect_coord_flt v1, const rect_coord_flt v2)
{
    rect_coord_flt v = rectCoordSub(v1,v2);
    if(rectNorm(v) < EPS) {
        return 1;
    } else {
        return 0;
    }
}

__host__ __device__ int equalRectCoord(const rect_coord_dbl v1, const rect_coord_dbl v2)
{
    rect_coord_dbl v = rectCoordSub(v1,v2);
    if(rectNorm(v) < EPS) {
        return 1;
    } else {
        return 0;
    }
}

__host__ __device__ rect_coord_flt crossProd(const rect_coord_flt u, const rect_coord_flt v)
{
    rect_coord_flt r;
    r.coords[0] = (u.coords[1])*(v.coords[2])-(u.coords[2])*(v.coords[1]);
    r.coords[1] = (u.coords[2])*(v.coords[0])-(u.coords[0])*(v.coords[2]);
    r.coords[2] = (u.coords[0])*(v.coords[1])-(u.coords[1])*(v.coords[0]);
    return r;
}

__host__ __device__ rect_coord_flt rectCoordAdd(const rect_coord_flt u, const rect_coord_flt v)
{
    rect_coord_flt result;
    for(int i=0;i<3;i++) {
        result.coords[i] = u.coords[i]+v.coords[i];
    }
    return result;
}

__host__ __device__ rect_coord_flt rectCoordSub(const rect_coord_flt u, const rect_coord_flt v)
{
    rect_coord_flt result;
    for(int i=0;i<3;i++) {
        result.coords[i] = u.coords[i]-v.coords[i];
    }
    return result;
}

__host__ __device__ rect_coord_flt scaRectMul(const float lambda, const rect_coord_flt v)
{
    rect_coord_flt result;
    for(int i=0;i<3;i++) {
        result.coords[i] = lambda*v.coords[i];
    }
    return result;
}

__host__ __device__ rect_coord_dbl scaRectMul(const double lambda, const rect_coord_dbl v)
{
    rect_coord_dbl result;
    for(int i=0;i<3;i++) {
        result.coords[i] = lambda*v.coords[i];
    }
    return result;
}

__host__ __device__ rect_coord_dbl rectCoordAdd(const rect_coord_dbl u, const rect_coord_dbl v)
{
    rect_coord_dbl result;
    for(int i=0;i<3;i++) {
        result.coords[i] = u.coords[i]+v.coords[i];
    }
    return result;
}

__host__ __device__ rect_coord_dbl rectCoordSub(const rect_coord_dbl u, const rect_coord_dbl v)
{
    rect_coord_dbl result;
    for(int i=0;i<3;i++) {
        result.coords[i] = u.coords[i]-v.coords[i];
    }
    return result;
}

__host__ __device__ rect_coord_dbl triCentroid(rect_coord_dbl nod[3])
{
    rect_coord_dbl ctr_23 = scaRectMul(0.5,rectCoordAdd(nod[1],nod[2]));
    rect_coord_dbl centroid = rectCoordAdd(nod[0],scaRectMul(2.0/3.0,rectCoordSub(ctr_23,nod[0])));
    return centroid;
}

__host__ __device__ bool ray_intersect_triangle(const rect_coord_flt O, const rect_coord_flt dir, 
        const rect_coord_flt nod[3])
{
    /*vert0 is chosen as reference point*/
    rect_coord_flt E1, E2;
    E1 = rectCoordSub(nod[1],nod[0]);
    E2 = rectCoordSub(nod[2],nod[0]);
    /*cross product of dir and v0 to v1*/
    rect_coord_flt P = crossProd(dir,E2);
    float det = rectDotMul(P,E1);
    if(abs(det)<EPS) {
        return false;
    }
    /*Computation of parameter u*/
    rect_coord_flt T = rectCoordSub(O,nod[0]);
    float u = 1.0f/det*rectDotMul(P,T);
    if(u<0 || u>1) {
        return false;
    }
    /*Computation of parameter v*/
    rect_coord_flt Q = crossProd(T,E1);
    float v = 1.0f/det*rectDotMul(Q,dir);
    if(v<0 || u+v>1) {
        return false;
    }
    /*Computation of parameter t*/
    float t = 1.0f/det*rectDotMul(Q,E2);
    if(t<EPS) {
        return false;
    }
    return true;
}

__global__ void rayTrisInt(const rect_coord_flt pt_s, const rect_coord_flt dir, const rect_coord_flt *nod, 
        const tri_elem *elem, const int numElem, bool *flag)
{
    // decides if a point pnt is in a closed surface elem
    int idx = blockDim.x*blockIdx.x+threadIdx.x;
    if(idx<numElem) {
        rect_coord_flt pt[3];
        for(int i=0;i<3;i++) {
            pt[i].coords[0] = nod[elem[idx].nod[i]].coords[0];
            pt[i].coords[1] = nod[elem[idx].nod[i]].coords[1];
            pt[i].coords[2] = nod[elem[idx].nod[i]].coords[2];
        }
        flag[idx] = ray_intersect_triangle(pt_s,dir,pt);
    }
}

__global__ void distPntPnts(const rect_coord_flt pt, const rect_coord_flt *nod, const int numNod, float *dist) {
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    if(idx < numNod) {
        dist[idx] = __fsqrt_rn((pt.coords[0]-nod[idx].coords[0])*(pt.coords[0]-nod[idx].coords[0])
                +(pt.coords[1]-nod[idx].coords[1])*(pt.coords[1]-nod[idx].coords[1])
                +(pt.coords[2]-nod[idx].coords[2])*(pt.coords[2]-nod[idx].coords[2]));
    }
}

__host__ __device__ float convRand(const float lb, const float ub, const float randNumber) {
    float result = (ub-lb)*randNumber+lb;
    return result;
}

bool inBdry(const bool *flag, const int numFlag) {
    int sum = 0;;
    for(int i=0;i<numFlag;i++) {
        if(flag[i]) {
            sum++;
        }
    }
    if(sum%2==0) {
        return false;
    } else {
        return true;
    }
}

int genCHIEF(const rect_coord_flt *pt, const int numPt, const tri_elem *elem, const int numElem, 
        rect_coord_flt *pCHIEF, const int numCHIEF) {
    int i, cnt;
    float threshold_inner = 0.0000001;
    float *dist_h = (float*)malloc(numPt*sizeof(float));
    float minDist; //minimum distance between the chief point to all surface nod
    float *dist_d;
    CUDA_CALL(cudaMalloc((void**)&dist_d, numPt*sizeof(float)));
    rect_coord_flt dir; 
    
    //transfer the point cloud to GPU
    rect_coord_flt *pt_d;
    CUDA_CALL(cudaMalloc((void**)&pt_d,numPt*sizeof(rect_coord_flt))); //point cloud allocated on device
    CUDA_CALL(cudaMemcpy(pt_d,pt,numPt*sizeof(rect_coord_flt),cudaMemcpyHostToDevice)); //point cloud copied to device
    
    //transfer the element cloud to GPU
    tri_elem *elem_d;
    CUDA_CALL(cudaMalloc((void**)&elem_d,numElem*sizeof(tri_elem))); //elements allcoated on device
    CUDA_CALL(cudaMemcpy(elem_d,elem,numElem*sizeof(tri_elem),cudaMemcpyHostToDevice)); //elements copied to device
    
    //create a flag array on CPU and on GPU
    bool *flag_h = (bool*)malloc(numElem*sizeof(bool));
    bool *flag_d;
    CUDA_CALL(cudaMalloc((void**)&flag_d,numElem*sizeof(bool))); //memory for flags allocated on device

    unsigned long long seed = 0;
    int blockWidth = 32;
    int gridWidth;
    float xrand, yrand, zrand, unifRandNum[3];
    rect_coord_flt chief;
    
    //Find the bounding box
    float xb[2], yb[2], zb[2];
    findBB(pt,numPt,0,xb,yb,zb);
    
    //create a handle to curand
    curandGenerator_t gen;
    CURAND_CALL(curandCreateGeneratorHost(&gen,CURAND_RNG_PSEUDO_DEFAULT)); //construct generator
    cnt = 0; // initialize count for number of points generated
    while(cnt<numCHIEF) {
        do
        {
            //set seed
            CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen,seed++));
            CURAND_CALL(curandGenerateUniform(gen,unifRandNum,3)); //generate a uniformly distributed random number
            //generate the direction
            for(i=0;i<3;i++) {
                dir.coords[i] = unifRandNum[i];
            }
            //Convert the rand numbers into a point in the bounding box
            xrand = convRand(xb[0],xb[1],unifRandNum[0]);
            yrand = convRand(yb[0],yb[1],unifRandNum[1]);
            zrand = convRand(zb[0],zb[1],unifRandNum[2]);
            chief.coords[0] = xrand;
            chief.coords[1] = yrand;
            chief.coords[2] = zrand;
            //(&chief,1);
            gridWidth = (numElem+blockWidth-1)/blockWidth;
            rayTrisInt<<<gridWidth,blockWidth>>>(chief,dir,pt_d,elem_d,numElem,flag_d);
            gridWidth = (numPt+blockWidth-1)/blockWidth;
            distPntPnts<<<gridWidth,blockWidth>>>(chief,pt_d,numPt,dist_d);
            CUDA_CALL(cudaMemcpy(dist_h,dist_d,numPt*sizeof(float),cudaMemcpyDeviceToHost));
            //printFltMat(dist_h,1,numPt,1);
            CUDA_CALL(cudaMemcpy(flag_h,flag_d,numElem*sizeof(bool),cudaMemcpyDeviceToHost));
            minDist = dist_h[0];
            for(i=1;i<numPt;i++) {
                if(dist_h[i]<minDist) {
                    minDist = dist_h[i];
                }
            }
            //printf("The minimum distance is %f, threshold is %f\n",dist_min,threshold_inner);
            //printf("inSurf: %d\n", inSurf(flags_h, numElem));
        } while (!inBdry(flag_h,numElem) || minDist<threshold_inner);
        pCHIEF[cnt] = chief;
        cnt++;
    }
    CURAND_CALL(curandDestroyGenerator(gen));
    free(flag_h);
    free(dist_h);
    CUDA_CALL(cudaFree(pt_d));
    CUDA_CALL(cudaFree(elem_d));
    CUDA_CALL(cudaFree(flag_d));
    CUDA_CALL(cudaFree(dist_d));
    return EXIT_SUCCESS;
}

inline __device__ void crossNorm(const rect_coord_flt a, const rect_coord_flt b, rect_coord_flt *norm, float *length) 
{
    rect_coord_flt c;
    c.coords[0] = a.coords[1]*b.coords[2]-a.coords[2]*b.coords[1];
    c.coords[1] = a.coords[2]*b.coords[0]-a.coords[0]*b.coords[2];
    c.coords[2] = a.coords[0]*b.coords[1]-a.coords[1]*b.coords[0];

    *length = __fsqrt_rn((c.coords[0]*c.coords[0])+(c.coords[1]*c.coords[1])+(c.coords[2]*c.coords[2]));

    norm->coords[0] = c.coords[0] / *length;
    norm->coords[1] = c.coords[1] / *length;
    norm->coords[2] = c.coords[2] / *length;
}

__device__ void g_h_c_nsgl(const float k, const rect_coord_flt x, const rect_coord_flt p[3], 
        cuFloatComplex gCoeff[3], cuFloatComplex hCoeff[3], float *cCoeff) {
    //Initalization of g, h and c
    //printf("(%f,%f,%f)\n",p[0].coords[0],p[0].coords[1],p[0].coords[2]);
    for(int i=0;i<3;i++) {
        gCoeff[i] = make_cuFloatComplex(0,0);
        hCoeff[i] = make_cuFloatComplex(0,0);
    }
    *cCoeff = 0;
    
    //Local variables
    float eta1, eta2, wn, wm, xi1, xi2, xi3, rho, theta, vertCrossProd, temp, 
            temp_gh[3], omega = k*speed, pPsiLpn2, radius, prpn2;
    rect_coord_flt y, normal, rVec;
    cuFloatComplex Psi, pPsipn2;
    crossNorm(
    {
        p[0].coords[0]-p[2].coords[0],p[0].coords[1]-p[2].coords[1],p[0].coords[2]-p[2].coords[2]
    },
    {
        p[1].coords[0]-p[2].coords[0],p[1].coords[1]-p[2].coords[1],p[1].coords[2]-p[2].coords[2]
    },&normal,&vertCrossProd);
    vertCrossProd = vertCrossProd*0.25f;
    //printf("%f\n",normal.coords[0]);
    const float prodRhoOmega = density*omega;
    const float fourPI = 4.0f*PI;
    const float recipFourPI = 1.0f/fourPI;
    //printf("%f\n",k);
    for(int n=0;n<INTORDER;n++) {
        eta2 = INTPT[n];
        wn = INTWGT[n];
        theta = 0.5f+0.5f*eta2;
        for(int m=0;m<INTORDER;m++) {
            eta1 = INTPT[m];
            wm = INTWGT[m];
            rho = 0.5f+0.5f*eta1;
            temp = wn*wm*rho*vertCrossProd;
            
            xi1 = rho*(1-theta);
            xi2 = rho-xi1;
            xi3 = 1-xi1-xi2;
            //printf("xi1 = %f, xi2 = %f\n",xi1,xi2);
            y= {
                p[0].coords[0]*xi1+p[1].coords[0]*xi2+p[2].coords[0]*xi3, 
                p[0].coords[1]*xi1+p[1].coords[1]*xi2+p[2].coords[1]*xi3, 
                p[0].coords[2]*xi1+p[1].coords[2]*xi2+p[2].coords[2]*xi3
            };
            //printf("x: (%f,%f,%f), y: (%f,%f,%f)\n",x.coords[0],x.coords[1],x.coords[2],
            //        y.coords[0],y.coords[1],y.coords[2]);
            rVec = rectCoordSub(y,x);
            radius = __fsqrt_rn(rVec.coords[0]*rVec.coords[0]+rVec.coords[1]*rVec.coords[1]
                    +rVec.coords[2]*rVec.coords[2]);
            //printf("radius = %f\n",radius);
            prpn2 = ((y.coords[0]-x.coords[0])*normal.coords[0]+(y.coords[1]-x.coords[1])*normal.coords[1]
                    +(y.coords[2]-x.coords[2])*normal.coords[2])/radius;
            //printf("prpn2=%f\n",prpn2);
            pPsiLpn2 = -recipFourPI/(radius*radius)*prpn2;
            //printf("%f\n",pPsiLpn2);
            Psi = make_cuFloatComplex(__cosf(-k*radius)/(fourPI*radius),__sinf(-k*radius)/(fourPI*radius));
            pPsipn2 = cuCmulf(Psi,make_cuFloatComplex(-1.0f/radius,-k));
            pPsipn2 = make_cuFloatComplex(prpn2*cuCrealf(pPsipn2),prpn2*cuCimagf(pPsipn2));
            temp_gh[0] = temp*xi1;
            temp_gh[1] = temp*xi2;
            temp_gh[2] = temp*xi3;
            
            gCoeff[0] = cuCaddf(gCoeff[0],make_cuFloatComplex(temp_gh[0]*cuCrealf(Psi),temp_gh[0]*cuCimagf(Psi)));
            gCoeff[1] = cuCaddf(gCoeff[1],make_cuFloatComplex(temp_gh[1]*cuCrealf(Psi),temp_gh[1]*cuCimagf(Psi)));
            gCoeff[2] = cuCaddf(gCoeff[2],make_cuFloatComplex(temp_gh[2]*cuCrealf(Psi),temp_gh[2]*cuCimagf(Psi)));
            
            hCoeff[0] = cuCaddf(hCoeff[0],make_cuFloatComplex(temp_gh[0]*cuCrealf(pPsipn2),temp_gh[0]*cuCimagf(pPsipn2)));
            hCoeff[1] = cuCaddf(hCoeff[1],make_cuFloatComplex(temp_gh[1]*cuCrealf(pPsipn2),temp_gh[1]*cuCimagf(pPsipn2)));
            hCoeff[2] = cuCaddf(hCoeff[2],make_cuFloatComplex(temp_gh[2]*cuCrealf(pPsipn2),temp_gh[2]*cuCimagf(pPsipn2)));
            
            *cCoeff += temp*pPsiLpn2;
        }
    }
    gCoeff[0] = make_cuFloatComplex(-prodRhoOmega*cuCimagf(gCoeff[0]),prodRhoOmega*cuCrealf(gCoeff[0]));
    gCoeff[1] = make_cuFloatComplex(-prodRhoOmega*cuCimagf(gCoeff[1]),prodRhoOmega*cuCrealf(gCoeff[1]));
    gCoeff[2] = make_cuFloatComplex(-prodRhoOmega*cuCimagf(gCoeff[2]),prodRhoOmega*cuCrealf(gCoeff[2]));
}

__device__ void g_h_c_sgl(const float k, const rect_coord_flt x_sgl1, const rect_coord_flt x_sgl2, 
        const rect_coord_flt x_sgl3, const rect_coord_flt p[3], 
        cuFloatComplex gCoeff_sgl1[3], cuFloatComplex hCoeff_sgl1[3], float *cCoeff_sgl1,
        cuFloatComplex gCoeff_sgl2[3], cuFloatComplex hCoeff_sgl2[3], float *cCoeff_sgl2,
        cuFloatComplex gCoeff_sgl3[3], cuFloatComplex hCoeff_sgl3[3], float *cCoeff_sgl3) 
{
    //Initalization of g, h and c
    for(int i=0;i<3;i++) {
        gCoeff_sgl1[i] = make_cuFloatComplex(0,0);
        hCoeff_sgl1[i] = make_cuFloatComplex(0,0);
        gCoeff_sgl2[i] = make_cuFloatComplex(0,0);
        hCoeff_sgl2[i] = make_cuFloatComplex(0,0);
        gCoeff_sgl3[i] = make_cuFloatComplex(0,0);
        hCoeff_sgl3[i] = make_cuFloatComplex(0,0);
    }
    *cCoeff_sgl1 = 0;
    *cCoeff_sgl2 = 0;
    *cCoeff_sgl3 = 0;
    
    //Local variables
    float eta1, eta2, wn, wm, xi1_sgl1, xi2_sgl1, xi3_sgl1, xi1_sgl2, xi2_sgl2, xi3_sgl2,
            xi1_sgl3, xi2_sgl3, xi3_sgl3, rho, theta, vertCrossProd, temp, 
            temp_gh[3], omega = k*speed, pPsiLpn2, radius, prpn2;
    rect_coord_flt y_sgl1, y_sgl2, y_sgl3, normal, rVec;
    cuFloatComplex Psi, pPsipn2;
    crossNorm(
    {
        p[0].coords[0]-p[2].coords[0],p[0].coords[1]-p[2].coords[1],p[0].coords[2]-p[2].coords[2]
    },
    {
        p[1].coords[0]-p[2].coords[0],p[1].coords[1]-p[2].coords[1],p[1].coords[2]-p[2].coords[2]
    },&normal,&vertCrossProd);
    vertCrossProd = vertCrossProd*0.25f;
    //printf("vert: %f\n",vertCrossProd);
    
    //printf("normal=(%f,%f,%f)\n",normal.coords[0],normal.coords[1],normal.coords[2]);
    const float prodRhoOmega = density*omega;
    const float fourPI = 4.0f*PI;
    const float recipFourPI = 1.0/fourPI;
    //printf("density*omega = %f\n",prodRhoOmega);
    //Compute integrals for g, h and c
    for(int n=0;n<INTORDER;n++) {
        eta2 = INTPT[n];
        wn = INTWGT[n];
        theta = 0.5f+0.5f*eta2;
        for(int m=0;m<INTORDER;m++) {
            eta1 = INTPT[m];
            wm = INTWGT[m];
            rho = 0.5f+0.5f*eta1;
            temp = wn*wm*rho*vertCrossProd;
            
            xi1_sgl3 = rho*(1-theta);
            xi2_sgl3 = rho-xi1_sgl3; //rho*theta
            xi3_sgl3 = 1-xi1_sgl3-xi2_sgl3;
            
            xi1_sgl1 = 1-rho;
            xi2_sgl1 = rho-xi2_sgl3; //rho-rho*theta
            xi3_sgl1 = 1-xi1_sgl1-xi2_sgl1;
            
            xi1_sgl2 = xi2_sgl3; //rho*theta
            xi2_sgl2 = 1-rho;
            xi3_sgl2 = 1-xi1_sgl2-xi2_sgl2;
            
            
            
            //printf("xi1 = %f, xi2 = %f\n",xi1,xi2);
            y_sgl1= {
                p[0].coords[0]*xi1_sgl1+p[1].coords[0]*xi2_sgl1+p[2].coords[0]*xi3_sgl1, 
                p[0].coords[1]*xi1_sgl1+p[1].coords[1]*xi2_sgl1+p[2].coords[1]*xi3_sgl1, 
                p[0].coords[2]*xi1_sgl1+p[1].coords[2]*xi2_sgl1+p[2].coords[2]*xi3_sgl1
            };
            y_sgl2= {
                p[0].coords[0]*xi1_sgl2+p[1].coords[0]*xi2_sgl2+p[2].coords[0]*xi3_sgl2, 
                p[0].coords[1]*xi1_sgl2+p[1].coords[1]*xi2_sgl2+p[2].coords[1]*xi3_sgl2, 
                p[0].coords[2]*xi1_sgl2+p[1].coords[2]*xi2_sgl2+p[2].coords[2]*xi3_sgl2
            };
            y_sgl3= {
                p[0].coords[0]*xi1_sgl3+p[1].coords[0]*xi2_sgl3+p[2].coords[0]*xi3_sgl3, 
                p[0].coords[1]*xi1_sgl3+p[1].coords[1]*xi2_sgl3+p[2].coords[1]*xi3_sgl3, 
                p[0].coords[2]*xi1_sgl3+p[1].coords[2]*xi2_sgl3+p[2].coords[2]*xi3_sgl3
            };
            
            //update coefficients with singularity on node 1
            rVec = rectCoordSub(y_sgl1,x_sgl1);
            radius = sqrtf(rVec.coords[0]*rVec.coords[0]+rVec.coords[1]*rVec.coords[1]+rVec.coords[2]*rVec.coords[2]);
            //printf("radius = %f\n",radius);
            prpn2 = ((y_sgl1.coords[0]-x_sgl1.coords[0])*normal.coords[0]+(y_sgl1.coords[1]-x_sgl1.coords[1])*normal.coords[1]
                    +(y_sgl1.coords[2]-x_sgl1.coords[2])*normal.coords[2])/radius;
            //printf("prpn2=%f\n",prpn2);
            pPsiLpn2 = -recipFourPI/(radius*radius)*prpn2;
            //printf("%f\n",pPsiLpn2);
            Psi = make_cuFloatComplex(__cosf(-k*radius)/(fourPI*radius),__sinf(-k*radius)/(fourPI*radius));
            pPsipn2 = cuCmulf(Psi,make_cuFloatComplex(-1.0f/radius,-k));
            pPsipn2 = make_cuFloatComplex(prpn2*cuCrealf(pPsipn2),prpn2*cuCimagf(pPsipn2));
            temp_gh[0] = temp*xi1_sgl1;
            temp_gh[1] = temp*xi2_sgl1;
            temp_gh[2] = temp*xi3_sgl1;
            
            gCoeff_sgl1[0] = cuCaddf(gCoeff_sgl1[0],make_cuFloatComplex(temp_gh[0]*cuCrealf(Psi),temp_gh[0]*cuCimagf(Psi)));
            gCoeff_sgl1[1] = cuCaddf(gCoeff_sgl1[1],make_cuFloatComplex(temp_gh[1]*cuCrealf(Psi),temp_gh[1]*cuCimagf(Psi)));
            gCoeff_sgl1[2] = cuCaddf(gCoeff_sgl1[2],make_cuFloatComplex(temp_gh[2]*cuCrealf(Psi),temp_gh[2]*cuCimagf(Psi)));
            
            hCoeff_sgl1[0] = cuCaddf(hCoeff_sgl1[0],make_cuFloatComplex(temp_gh[0]*cuCrealf(pPsipn2),temp_gh[0]*cuCimagf(pPsipn2)));
            hCoeff_sgl1[1] = cuCaddf(hCoeff_sgl1[1],make_cuFloatComplex(temp_gh[1]*cuCrealf(pPsipn2),temp_gh[1]*cuCimagf(pPsipn2)));
            hCoeff_sgl1[2] = cuCaddf(hCoeff_sgl1[2],make_cuFloatComplex(temp_gh[2]*cuCrealf(pPsipn2),temp_gh[2]*cuCimagf(pPsipn2)));
            
            *cCoeff_sgl1 += temp*pPsiLpn2;
            
            //update coefficients with singularity on node 2
            rVec = rectCoordSub(y_sgl2,x_sgl2);
            radius = sqrtf(rVec.coords[0]*rVec.coords[0]+rVec.coords[1]*rVec.coords[1]+rVec.coords[2]*rVec.coords[2]);
            //printf("radius = %f\n",radius);
            prpn2 = ((y_sgl1.coords[0]-x_sgl1.coords[0])*normal.coords[0]+(y_sgl1.coords[1]-x_sgl1.coords[1])*normal.coords[1]
                    +(y_sgl1.coords[2]-x_sgl1.coords[2])*normal.coords[2])/radius;
            //printf("prpn2=%f\n",prpn2);
            pPsiLpn2 = -recipFourPI/(radius*radius)*prpn2;
            //printf("%f\n",pPsiLpn2);
            Psi = make_cuFloatComplex(__cosf(-k*radius)/(fourPI*radius),__sinf(-k*radius)/(fourPI*radius));
            pPsipn2 = cuCmulf(Psi,make_cuFloatComplex(-1.0f/radius,-k));
            pPsipn2 = make_cuFloatComplex(prpn2*cuCrealf(pPsipn2),prpn2*cuCimagf(pPsipn2));
            temp_gh[0] = temp*xi1_sgl2;
            temp_gh[1] = temp*xi2_sgl2;
            temp_gh[2] = temp*xi3_sgl2;
            
            gCoeff_sgl2[0] = cuCaddf(gCoeff_sgl2[0],make_cuFloatComplex(temp_gh[0]*cuCrealf(Psi),temp_gh[0]*cuCimagf(Psi)));
            gCoeff_sgl2[1] = cuCaddf(gCoeff_sgl2[1],make_cuFloatComplex(temp_gh[1]*cuCrealf(Psi),temp_gh[1]*cuCimagf(Psi)));
            gCoeff_sgl2[2] = cuCaddf(gCoeff_sgl2[2],make_cuFloatComplex(temp_gh[2]*cuCrealf(Psi),temp_gh[2]*cuCimagf(Psi)));
            
            hCoeff_sgl2[0] = cuCaddf(hCoeff_sgl2[0],make_cuFloatComplex(temp_gh[0]*cuCrealf(pPsipn2),temp_gh[0]*cuCimagf(pPsipn2)));
            hCoeff_sgl2[1] = cuCaddf(hCoeff_sgl2[1],make_cuFloatComplex(temp_gh[1]*cuCrealf(pPsipn2),temp_gh[1]*cuCimagf(pPsipn2)));
            hCoeff_sgl2[2] = cuCaddf(hCoeff_sgl2[2],make_cuFloatComplex(temp_gh[2]*cuCrealf(pPsipn2),temp_gh[2]*cuCimagf(pPsipn2)));
            
            *cCoeff_sgl2 += temp*pPsiLpn2;
            
            //update coefficients with singularity on node 3
            rVec = rectCoordSub(y_sgl3,x_sgl3);
            radius = sqrtf(rVec.coords[0]*rVec.coords[0]+rVec.coords[1]*rVec.coords[1]+rVec.coords[2]*rVec.coords[2]);
            //printf("radius = %f\n",radius);
            prpn2 = ((y_sgl1.coords[0]-x_sgl1.coords[0])*normal.coords[0]+(y_sgl1.coords[1]-x_sgl1.coords[1])*normal.coords[1]
                    +(y_sgl1.coords[2]-x_sgl1.coords[2])*normal.coords[2])/radius;
            //printf("prpn2=%f\n",prpn2);
            pPsiLpn2 = -recipFourPI/(radius*radius)*prpn2;
            //printf("%f\n",pPsiLpn2);
            Psi = make_cuFloatComplex(__cosf(-k*radius)/(fourPI*radius),__sinf(-k*radius)/(fourPI*radius));
            pPsipn2 = cuCmulf(Psi,make_cuFloatComplex(-1.0f/radius,-k));
            pPsipn2 = make_cuFloatComplex(prpn2*cuCrealf(pPsipn2),prpn2*cuCimagf(pPsipn2));
            temp_gh[0] = temp*xi1_sgl3;
            temp_gh[1] = temp*xi2_sgl3;
            temp_gh[2] = temp*xi3_sgl3;
            
            gCoeff_sgl3[0] = cuCaddf(gCoeff_sgl3[0],make_cuFloatComplex(temp_gh[0]*cuCrealf(Psi),temp_gh[0]*cuCimagf(Psi)));
            gCoeff_sgl3[1] = cuCaddf(gCoeff_sgl3[1],make_cuFloatComplex(temp_gh[1]*cuCrealf(Psi),temp_gh[1]*cuCimagf(Psi)));
            gCoeff_sgl3[2] = cuCaddf(gCoeff_sgl3[2],make_cuFloatComplex(temp_gh[2]*cuCrealf(Psi),temp_gh[2]*cuCimagf(Psi)));
            
            hCoeff_sgl3[0] = cuCaddf(hCoeff_sgl3[0],make_cuFloatComplex(temp_gh[0]*cuCrealf(pPsipn2),temp_gh[0]*cuCimagf(pPsipn2)));
            hCoeff_sgl3[1] = cuCaddf(hCoeff_sgl3[1],make_cuFloatComplex(temp_gh[1]*cuCrealf(pPsipn2),temp_gh[1]*cuCimagf(pPsipn2)));
            hCoeff_sgl3[2] = cuCaddf(hCoeff_sgl3[2],make_cuFloatComplex(temp_gh[2]*cuCrealf(pPsipn2),temp_gh[2]*cuCimagf(pPsipn2)));
            
            *cCoeff_sgl3 += temp*pPsiLpn2;
        }
    }
    gCoeff_sgl1[0] = make_cuFloatComplex(-prodRhoOmega*cuCimagf(gCoeff_sgl1[0]),prodRhoOmega*cuCrealf(gCoeff_sgl1[0]));
    gCoeff_sgl1[1] = make_cuFloatComplex(-prodRhoOmega*cuCimagf(gCoeff_sgl1[1]),prodRhoOmega*cuCrealf(gCoeff_sgl1[1]));
    gCoeff_sgl1[2] = make_cuFloatComplex(-prodRhoOmega*cuCimagf(gCoeff_sgl1[2]),prodRhoOmega*cuCrealf(gCoeff_sgl1[2]));
    
    gCoeff_sgl2[0] = make_cuFloatComplex(-prodRhoOmega*cuCimagf(gCoeff_sgl2[0]),prodRhoOmega*cuCrealf(gCoeff_sgl2[0]));
    gCoeff_sgl2[1] = make_cuFloatComplex(-prodRhoOmega*cuCimagf(gCoeff_sgl2[1]),prodRhoOmega*cuCrealf(gCoeff_sgl2[1]));
    gCoeff_sgl2[2] = make_cuFloatComplex(-prodRhoOmega*cuCimagf(gCoeff_sgl2[2]),prodRhoOmega*cuCrealf(gCoeff_sgl2[2]));
    
    gCoeff_sgl3[0] = make_cuFloatComplex(-prodRhoOmega*cuCimagf(gCoeff_sgl3[0]),prodRhoOmega*cuCrealf(gCoeff_sgl3[0]));
    gCoeff_sgl3[1] = make_cuFloatComplex(-prodRhoOmega*cuCimagf(gCoeff_sgl3[1]),prodRhoOmega*cuCrealf(gCoeff_sgl3[1]));
    gCoeff_sgl3[2] = make_cuFloatComplex(-prodRhoOmega*cuCimagf(gCoeff_sgl3[2]),prodRhoOmega*cuCrealf(gCoeff_sgl3[2]));
}

__host__ __device__ cuFloatComplex ptSrc(const float k, const float amp, const rect_coord_flt srcLoc, const rect_coord_flt evalLoc)
{
    float fourPI = 4.0f*PI;
    rect_coord_flt rVec = rectCoordSub(evalLoc,srcLoc);
    float radius = sqrtf(rVec.coords[0]*rVec.coords[0]+rVec.coords[1]*rVec.coords[1]+rVec.coords[2]*rVec.coords[2]);
    return make_cuFloatComplex(amp*cosf(-k*radius)/(fourPI*radius),amp*sinf(-k*radius)/(fourPI*radius));
}

__host__ __device__ cuFloatComplex mpSrc(const float k, const float qs, const rect_coord_flt src, const rect_coord_flt eval)
{
    rect_coord_flt vec = rectCoordSub(eval,src);
    float radius = sqrtf(vec.coords[0]*vec.coords[0]+vec.coords[1]*vec.coords[1]+vec.coords[2]*vec.coords[2]);
    cuFloatComplex result = make_cuFloatComplex(0,RHO_AIR*SPEED_SOUND*k*qs/(4*PI));
    result = cuCmulf(result,make_cuFloatComplex(cos(-k*radius)/radius,sin(-k*radius)/radius));
    return result;
}

__host__ __device__ cuFloatComplex dirSrc(const float k, const float strength, const rect_coord_flt dir, const rect_coord_flt evalLoc)
{
    float theta = -k*rectDotMul(dir,evalLoc);
    return make_cuFloatComplex(strength*cosf(theta),strength*sinf(theta));
}

// compute non-singular relationship between points and elements
__global__ void atomicPtsElems_nsgl(const float k, const rect_coord_flt *pt, const int numNod, 
        const int idxPntStart, const int idxPntEnd, const tri_elem *elem, const int numElem, 
        cuFloatComplex *A, const int lda, cuFloatComplex *B, const int numSrc, const int ldb) {
    int xIdx = blockIdx.x*blockDim.x+threadIdx.x; //Index for points
    int yIdx = blockIdx.y*blockDim.y+threadIdx.y; //Index for elements
    //The thread with indices xIdx and yIdx process the point xIdx and elem yIdx
    if(xIdx>=idxPntStart && xIdx<=idxPntEnd && yIdx<numElem && xIdx!=elem[yIdx].nod[0] 
            && xIdx!=elem[yIdx].nod[1] && xIdx!=elem[yIdx].nod[2]) {
        int i, j;
        cuFloatComplex hCoeff[3], gCoeff[3], bc, pCoeffs[3], temp;
        float cCoeff;
        rect_coord_flt triNod[3];
        triNod[0] = pt[elem[yIdx].nod[0]];
        triNod[1] = pt[elem[yIdx].nod[1]];
        triNod[2] = pt[elem[yIdx].nod[2]];
        g_h_c_nsgl(k,pt[xIdx],triNod,gCoeff,hCoeff,&cCoeff);
        
        //Update the A matrix
        bc = cuCdivf(elem[yIdx].bc[0],elem[yIdx].bc[1]);
        for(i=0;i<3;i++) {
            pCoeffs[i] = cuCsubf(hCoeff[i],cuCmulf(bc,gCoeff[i]));
        }
        
        for(i=0;i<3;i++) {
            //atomicFloatComplexAdd(&A[IDXC0(xIdx,elem[yIdx].nod[i],lda)],pCoeffs[i]);
            atomicAdd(&A[IDXC0(xIdx,elem[yIdx].nod[i],lda)].x,cuCrealf(pCoeffs[i]));
            atomicAdd(&A[IDXC0(xIdx,elem[yIdx].nod[i],lda)].y,cuCimagf(pCoeffs[i]));
        }
        
        //Update from C coefficients
        if(xIdx<numNod) {
            //atomicFloatComplexSub(&A[IDXC0(xIdx,xIdx,lda)],make_cuFloatComplex(cCoeff,0));
            atomicAdd(&A[IDXC0(xIdx,xIdx,lda)].x,-cCoeff);
        }
        
        //Update the B matrix
        bc = cuCdivf(elem[yIdx].bc[2],elem[yIdx].bc[1]);
        //printf("bc: \n");
        //printComplexMatrix(&bc,1,1,1);
        for(i=0;i<numSrc;i++) {
            for(j=0;j<3;j++) {
                //atomicFloatComplexSub(&B[IDXC0(xIdx,i,ldb)],cuCmulf(bc,gCoeff[j]));
                temp = cuCmulf(bc,gCoeff[j]);
                atomicAdd(&B[IDXC0(xIdx,i,ldb)].x,-cuCrealf(temp));
                atomicAdd(&B[IDXC0(xIdx,i,ldb)].y,-cuCimagf(temp));
            }
        }
    }
}

__global__ void atomicPtsElems_sgl(const float k, const rect_coord_flt *pt, const tri_elem *elem, 
        const int numElem, cuFloatComplex *A, const int lda, cuFloatComplex *B, 
        const int numSrc, const int ldb) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx < numElem) {
        int i, j;
        cuFloatComplex hCoeff_sgl1[3], hCoeff_sgl2[3], hCoeff_sgl3[3], 
                gCoeff_sgl1[3], gCoeff_sgl2[3], gCoeff_sgl3[3], pCoeffs_sgl1[3], 
                pCoeffs_sgl2[3], pCoeffs_sgl3[3], bc, temp;
        float cCoeff_sgl1, cCoeff_sgl2, cCoeff_sgl3;
        
        rect_coord_flt nod[3];
        for(i=0;i<3;i++) {
            nod[i] = pt[elem[idx].nod[i]];
        }
        // Compute h and g coefficients
        g_h_c_sgl(k,pt[elem[idx].nod[0]],pt[elem[idx].nod[1]],pt[elem[idx].nod[2]],
                nod,gCoeff_sgl1,hCoeff_sgl1,&cCoeff_sgl1,gCoeff_sgl2,hCoeff_sgl2,&cCoeff_sgl2,
                gCoeff_sgl3,hCoeff_sgl3,&cCoeff_sgl3);
        
        //Compute p coefficients
        bc = cuCdivf(elem[idx].bc[0],elem[idx].bc[1]);
        for(j=0;j<3;j++) {
            pCoeffs_sgl1[j] = cuCsubf(hCoeff_sgl1[j],cuCmulf(bc,gCoeff_sgl1[j]));
            pCoeffs_sgl2[j] = cuCsubf(hCoeff_sgl2[j],cuCmulf(bc,gCoeff_sgl2[j]));
            pCoeffs_sgl3[j] = cuCsubf(hCoeff_sgl3[j],cuCmulf(bc,gCoeff_sgl3[j]));
        }
        
        //Update matrix A using pCoeffs
        for(j=0;j<3;j++) {
            //atomicFloatComplexAdd(&A[IDXC0(elem[idx].nod[0],elem[idx].nod[j],lda)],
            //        pCoeffs_sgl1[j]);
            atomicAdd(&A[IDXC0(elem[idx].nod[0],elem[idx].nod[j],lda)].x,
                    cuCrealf(pCoeffs_sgl1[j]));
            atomicAdd(&A[IDXC0(elem[idx].nod[0],elem[idx].nod[j],lda)].y,
                    cuCimagf(pCoeffs_sgl1[j]));
            //atomicFloatComplexAdd(&A[IDXC0(elem[idx].nod[1],elem[idx].nod[j],lda)],
            //        pCoeffs_sgl2[j]);
            atomicAdd(&A[IDXC0(elem[idx].nod[1],elem[idx].nod[j],lda)].x,
                    cuCrealf(pCoeffs_sgl2[j]));
            atomicAdd(&A[IDXC0(elem[idx].nod[1],elem[idx].nod[j],lda)].y,
                    cuCimagf(pCoeffs_sgl2[j]));
            //atomicFloatComplexAdd(&A[IDXC0(elem[idx].nod[2],elem[idx].nod[j],lda)],
            //        pCoeffs_sgl3[j]);
            atomicAdd(&A[IDXC0(elem[idx].nod[2],elem[idx].nod[j],lda)].x,
                    cuCrealf(pCoeffs_sgl3[j]));
            atomicAdd(&A[IDXC0(elem[idx].nod[2],elem[idx].nod[j],lda)].y,
                    cuCimagf(pCoeffs_sgl3[j]));
        }
        
        //atomicFloatComplexSub(&A[IDXC0(elem[idx].nod[0],elem[idx].nod[0],lda)],
        //        make_cuFloatComplex(cCoeff_sgl1,0));
        atomicAdd(&A[IDXC0(elem[idx].nod[0],elem[idx].nod[0],lda)].x,
                -cCoeff_sgl1);
        //atomicFloatComplexSub(&A[IDXC0(elem[idx].nod[1],elem[idx].nod[1],lda)],
        //        make_cuFloatComplex(cCoeff_sgl2,0));
        atomicAdd(&A[IDXC0(elem[idx].nod[1],elem[idx].nod[1],lda)].x,
                -cCoeff_sgl2);
        //atomicFloatComplexSub(&A[IDXC0(elem[idx].nod[2],elem[idx].nod[2],lda)],
        //        make_cuFloatComplex(cCoeff_sgl3,0));
        atomicAdd(&A[IDXC0(elem[idx].nod[2],elem[idx].nod[2],lda)].x,
                -cCoeff_sgl3);
        
        //Update matrix B using g Coefficients
        bc = cuCdivf(elem[idx].bc[2],elem[idx].bc[1]);
        for(i=0;i<numSrc;i++) {
            for(j=0;j<3;j++) {
                //atomicFloatComplexSub(&B[IDXC0(elem[idx].nod[0],i,ldb)],
                //        cuCmulf(bc,gCoeff_sgl1[j]));
                temp = cuCmulf(bc,gCoeff_sgl1[j]);
                atomicAdd(&B[IDXC0(elem[idx].nod[0],i,ldb)].x,-cuCrealf(temp));
                atomicAdd(&B[IDXC0(elem[idx].nod[0],i,ldb)].y,-cuCimagf(temp));
                //atomicFloatComplexSub(&B[IDXC0(elem[idx].nod[1],i,ldb)],
                //        cuCmulf(bc,gCoeff_sgl2[j]));
                temp = cuCmulf(bc,gCoeff_sgl2[j]);
                atomicAdd(&B[IDXC0(elem[idx].nod[1],i,ldb)].x,-cuCrealf(temp));
                atomicAdd(&B[IDXC0(elem[idx].nod[1],i,ldb)].y,-cuCimagf(temp));
                //atomicFloatComplexSub(&B[IDXC0(elem[idx].nod[2],i,ldb)],
                //        cuCmulf(bc,gCoeff_sgl3[j]));
                temp = cuCmulf(bc,gCoeff_sgl3[j]);
                atomicAdd(&B[IDXC0(elem[idx].nod[2],i,ldb)].x,-cuCrealf(temp));
                atomicAdd(&B[IDXC0(elem[idx].nod[2],i,ldb)].y,-cuCimagf(temp));
            }
        }
    }
}

int atomicGenSystem(const float k, const tri_elem *elem, const int numElem, 
        const rect_coord_flt *nod, const int numNod, const rect_coord_flt *chief, const int numCHIEF, 
        const rect_coord_flt *src, const int numSrc, cuFloatComplex *A, const int lda, 
        cuFloatComplex *B, const int ldb) {
    int i, j;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    //Move elements to GPU
    tri_elem *elem_d;
    CUDA_CALL(cudaMalloc(&elem_d,numElem*sizeof(tri_elem)));
    CUDA_CALL(cudaMemcpy(elem_d,elem,numElem*sizeof(tri_elem),cudaMemcpyHostToDevice));
    
    //Move points to GPU
    rect_coord_flt *pt_h = (rect_coord_flt*)malloc((numNod+numCHIEF)*sizeof(rect_coord_flt));
    for(i=0;i<numNod;i++) {
        pt_h[i] = nod[i];
    }
    for(i=0;i<numCHIEF;i++) {
        pt_h[numNod+i] = chief[i];
    }
    
    rect_coord_flt *pt_d;
    CUDA_CALL(cudaMalloc(&pt_d,(numNod+numCHIEF)*sizeof(rect_coord_flt)));
    CUDA_CALL(cudaMemcpy(pt_d,pt_h,(numNod+numCHIEF)*sizeof(rect_coord_flt),cudaMemcpyHostToDevice));
    
    //Initialization of A
    for(i=0;i<numNod+numCHIEF;i++) {
        for(j=0;j<numNod;j++) {
            if(i==j) {
                A[IDXC0(i,j,lda)] = make_cuFloatComplex(1,0);
            } else {
                A[IDXC0(i,j,lda)] = make_cuFloatComplex(0,0);
            }
        }
    }
    
    //Initialization of B
    for(i=0;i<numNod+numCHIEF;i++) {
        for(j=0;j<numSrc;j++) {
            B[IDXC0(i,j,ldb)] = ptSrc(k,STRENGTH,src[j],pt_h[i]);
        }
    }
    
    cuFloatComplex *A_d, *B_d;
    CUDA_CALL(cudaMalloc(&A_d,(numNod+numCHIEF)*numNod*sizeof(cuFloatComplex)));
    CUDA_CALL(cudaMemcpy(A_d,A,(numNod+numCHIEF)*numNod*sizeof(cuFloatComplex),cudaMemcpyHostToDevice));
    
    CUDA_CALL(cudaMalloc(&B_d,(numNod+numCHIEF)*numSrc*sizeof(cuFloatComplex)));
    CUDA_CALL(cudaMemcpy(B_d,B,(numNod+numCHIEF)*numSrc*sizeof(cuFloatComplex),cudaMemcpyHostToDevice));
    
    int xNumBlocks, xWidth = 16, yNumBlocks, yWidth = 16;
    xNumBlocks = (numNod+numCHIEF+xWidth-1)/xWidth;
    yNumBlocks = (numElem+yWidth-1)/yWidth;
    dim3 gridLayout, blockLayout;
    gridLayout.x = xNumBlocks;
    gridLayout.y = yNumBlocks;
    
    blockLayout.x = xWidth;
    blockLayout.y = yWidth;
    
    cudaEventRecord(start);
    atomicPtsElems_nsgl<<<gridLayout,blockLayout>>>(k,pt_d,numNod,0,numNod+numCHIEF-1,
            elem_d,numElem,A_d,lda,B_d,numSrc,ldb);
    
    //CUDA_CALL(cudaMemcpy(A,A_d,(numNod+numCHIEF)*numNod*sizeof(cuFloatComplex),cudaMemcpyDeviceToHost));
    //printCuFloatComplexMat(A,numNod+numCHIEF,numNod,numNod+numCHIEF);
    atomicPtsElems_sgl<<<yNumBlocks,yWidth>>>(k,pt_d,elem_d,numElem,A_d,lda,B_d,numSrc,ldb);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds,start,stop);
    printf("Elapsed system generation time: %f milliseconds.\n",milliseconds);
    CUDA_CALL(cudaMemcpy(A,A_d,(numNod+numCHIEF)*numNod*sizeof(cuFloatComplex),cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(B,B_d,(numNod+numCHIEF)*numSrc*sizeof(cuFloatComplex),cudaMemcpyDeviceToHost));
    
    
    CUDA_CALL(cudaFree(A_d));
    CUDA_CALL(cudaFree(B_d));
    CUDA_CALL(cudaFree(elem_d));
    CUDA_CALL(cudaFree(pt_d));
    
    return EXIT_SUCCESS;
}

int qrSolver(const cuFloatComplex *A, const int mA, const int nA, const int ldA, 
        cuFloatComplex *B, const int nB, const int ldB) {
    cudaEvent_t start, stop;
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));
    cusolverDnHandle_t cusolverH = NULL;
    CUSOLVER_CALL(cusolverDnCreate(&cusolverH));
    
    
    cuFloatComplex *A_d;
    CUDA_CALL(cudaMalloc(&A_d,ldA*nA*sizeof(cuFloatComplex)));
    CUDA_CALL(cudaMemcpy(A_d,A,ldA*nA*sizeof(cuFloatComplex),cudaMemcpyHostToDevice));
    
    cuFloatComplex *B_d;
    CUDA_CALL(cudaMalloc(&B_d,ldB*nB*sizeof(cuFloatComplex)));
    CUDA_CALL(cudaMemcpy(B_d,B,ldB*nB*sizeof(cuFloatComplex),cudaMemcpyHostToDevice));
    
    //A = QR
    int lwork;
    CUSOLVER_CALL(cusolverDnCgeqrf_bufferSize(cusolverH,mA,nA,A_d,ldA,&lwork));
    
    cuFloatComplex *workspace_d;
    CUDA_CALL(cudaMalloc(&workspace_d,lwork*sizeof(cuFloatComplex)));
    cuFloatComplex *tau_d;
    CUDA_CALL(cudaMalloc(&tau_d,max(mA,nA)*sizeof(cuFloatComplex)));
    int *deviceInfo_d, deviceInfo;
    CUDA_CALL(cudaMalloc(&deviceInfo_d,sizeof(int)));
    
    CUDA_CALL(cudaEventRecord(start));
    CUSOLVER_CALL(cusolverDnCgeqrf(cusolverH,mA,nA,A_d,ldA,tau_d,workspace_d,lwork,
            deviceInfo_d));
    CUDA_CALL(cudaMemcpy(&deviceInfo,deviceInfo_d,sizeof(int),cudaMemcpyDeviceToHost));
    
    //B = (Q^H)*B
    CUSOLVER_CALL(cusolverDnCunmqr(cusolverH,CUBLAS_SIDE_LEFT,CUBLAS_OP_C,mA,nB,
            nA,A_d,ldA,tau_d,B_d,ldB,workspace_d,lwork,deviceInfo_d));
    CUDA_CALL(cudaMemcpy(&deviceInfo,deviceInfo_d,sizeof(int),cudaMemcpyDeviceToHost));
    
    //Solve Rx = B
    cuFloatComplex alpha = make_cuFloatComplex(1,0);
    cublasHandle_t cublasH;
    CUBLAS_CALL(cublasCreate_v2(&cublasH));
    CUBLAS_CALL(cublasCtrsm_v2(cublasH,CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_UPPER,
            CUBLAS_OP_N,CUBLAS_DIAG_NON_UNIT,nA,nB,&alpha,A_d,ldA,B_d,ldB));
    CUDA_CALL(cudaEventRecord(stop));
    
    CUDA_CALL(cudaMemcpy(B,B_d,ldB*nB*sizeof(cuFloatComplex),cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CUDA_CALL(cudaEventElapsedTime(&milliseconds,start,stop));
    printf("Elapsed system solving time: %f milliseconds.\n",milliseconds);
    CUDA_CALL(cudaEventDestroy(start));
    CUDA_CALL(cudaEventDestroy(stop));
    CUDA_CALL(cudaFree(A_d));
    CUDA_CALL(cudaFree(B_d));
    CUDA_CALL(cudaFree(tau_d));
    CUDA_CALL(cudaFree(workspace_d));
    CUDA_CALL(cudaFree(deviceInfo_d));
    CUBLAS_CALL(cublasDestroy_v2(cublasH));
    CUSOLVER_CALL(cusolverDnDestroy(cusolverH));
    
    return EXIT_SUCCESS;
}

int bemSolver_pt(const float k, const tri_elem *elem, const int numElem, 
        const rect_coord_flt *nod, const int numNod, const rect_coord_flt *chief, const int numCHIEF, 
        const rect_coord_flt *src, const int numSrc, cuFloatComplex *B, const int ldb)
{
    int i, j;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    //Move elements to GPU
    tri_elem *elem_d;
    CUDA_CALL(cudaMalloc(&elem_d,numElem*sizeof(tri_elem)));
    CUDA_CALL(cudaMemcpy(elem_d,elem,numElem*sizeof(tri_elem),cudaMemcpyHostToDevice));
    
    //Move points to GPU
    // rect_coord_flt *pt_h = (rect_coord_flt*)malloc((numNod+numCHIEF)*sizeof(rect_coord_flt));
    // for(i=0;i<numNod;i++) {
    //     pt_h[i] = nod[i];
    // }
    // for(i=0;i<numCHIEF;i++) {
    //     pt_h[numNod+i] = chief[i];
    // }
    
    rect_coord_flt *pt_d;
    CUDA_CALL(cudaMalloc(&pt_d, (numNod + numCHIEF) * sizeof(rect_coord_flt)));
    CUDA_CALL(cudaMemcpy(pt_d, nod, numNod * sizeof(rect_coord_flt),cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(pt_d + numNod, chief, numCHIEF * sizeof(rect_coord_flt),cudaMemcpyHostToDevice));
    
    CUDA_CALL(cudaEventRecord(start));
    //Generate the system
    cuFloatComplex *A = (cuFloatComplex*)malloc((numNod+numCHIEF)*numNod*sizeof(cuFloatComplex));
    
    memset(A,0,(numNod+numCHIEF)*numNod*sizeof(cuFloatComplex));

    for(i=0;i<numNod;i++) 
    {
        A[IDXC0(i,i,numNod+numCHIEF)] = make_cuFloatComplex(1,0);
    }
    
    //Initialization of B
    for(i=0;i<numNod+numCHIEF;i++) 
    {
        for(j=0;j<numSrc;j++) 
        {
            if(i < numNod)
                B[IDXC0(i,j,ldb)] = ptSrc(k,STRENGTH,src[j],nod[i]);
            else
                B[IDXC0(i,j,ldb)] = ptSrc(k,STRENGTH,src[j],chief[i - numNod]);
        }
    }
    
    cuFloatComplex *A_d, *B_d;
    CUDA_CALL(cudaMalloc(&A_d,(numNod+numCHIEF)*numNod*sizeof(cuFloatComplex)));
    CUDA_CALL(cudaMemcpy(A_d,A,(numNod+numCHIEF)*numNod*sizeof(cuFloatComplex),cudaMemcpyHostToDevice));
    
    CUDA_CALL(cudaMalloc(&B_d,(numNod+numCHIEF)*numSrc*sizeof(cuFloatComplex)));
    CUDA_CALL(cudaMemcpy(B_d,B,(numNod+numCHIEF)*numSrc*sizeof(cuFloatComplex),cudaMemcpyHostToDevice));
    
    int xNumBlocks, xWidth = 16, yNumBlocks, yWidth = 16;
    xNumBlocks = (numNod+numCHIEF+xWidth-1)/xWidth;
    yNumBlocks = (numElem+yWidth-1)/yWidth;
    dim3 gridLayout, blockLayout;
    gridLayout.x = xNumBlocks;
    gridLayout.y = yNumBlocks;
    
    blockLayout.x = xWidth;
    blockLayout.y = yWidth;
    
    atomicPtsElems_nsgl<<<gridLayout,blockLayout>>>(k,pt_d,numNod,0,numNod+numCHIEF-1,
            elem_d,numElem,A_d,numNod+numCHIEF,B_d,numSrc,ldb);
    atomicPtsElems_sgl<<<yNumBlocks,yWidth>>>(k,pt_d,elem_d,numElem,A_d,numNod+numCHIEF,
            B_d,numSrc,ldb);
    
    //Solving the system
    cusolverDnHandle_t cusolverH = NULL;
    CUSOLVER_CALL(cusolverDnCreate(&cusolverH));
    
    //A = QR
    int lwork;
    CUSOLVER_CALL(cusolverDnCgeqrf_bufferSize(cusolverH,numNod+numCHIEF,numNod,A_d
            ,numNod+numCHIEF,&lwork));
    
    cuFloatComplex *workspace_d;
    CUDA_CALL(cudaMalloc(&workspace_d,lwork*sizeof(cuFloatComplex)));
    cuFloatComplex *tau_d;
    CUDA_CALL(cudaMalloc(&tau_d,(numNod+numCHIEF)*sizeof(cuFloatComplex)));
    int *deviceInfo_d, deviceInfo;
    CUDA_CALL(cudaMalloc(&deviceInfo_d,sizeof(int)));
    
    
    CUSOLVER_CALL(cusolverDnCgeqrf(cusolverH,numNod+numCHIEF,numNod,A_d,numNod+numCHIEF,
            tau_d,workspace_d,lwork,deviceInfo_d));
    CUDA_CALL(cudaMemcpy(&deviceInfo,deviceInfo_d,sizeof(int),cudaMemcpyDeviceToHost));
    
    //B = (Q^H)*B
    CUSOLVER_CALL(cusolverDnCunmqr(cusolverH,CUBLAS_SIDE_LEFT,CUBLAS_OP_C,numNod+numCHIEF,numSrc,
            numNod,A_d,numNod+numCHIEF,tau_d,B_d,ldb,workspace_d,lwork,deviceInfo_d));
    CUDA_CALL(cudaMemcpy(&deviceInfo,deviceInfo_d,sizeof(int),cudaMemcpyDeviceToHost));
    
    //Solve Rx = B
    cuFloatComplex alpha = make_cuFloatComplex(1,0);
    cublasHandle_t cublasH;
    CUBLAS_CALL(cublasCreate_v2(&cublasH));
    CUBLAS_CALL(cublasCtrsm_v2(cublasH,CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_UPPER,
            CUBLAS_OP_N,CUBLAS_DIAG_NON_UNIT,numNod,numSrc,&alpha,A_d,numNod+numCHIEF,B_d,ldb));
    CUDA_CALL(cudaEventRecord(stop));
    CUDA_CALL(cudaEventSynchronize(stop));
    CUDA_CALL(cudaMemcpy(B,B_d,ldb*numSrc*sizeof(cuFloatComplex),cudaMemcpyDeviceToHost));
    
    float milliseconds = 0;
    CUDA_CALL(cudaEventElapsedTime(&milliseconds,start,stop));
    printf("Elapsed system solving time: %f milliseconds.\n",milliseconds);
    
    //release memory
    CUDA_CALL(cudaEventDestroy(start));
    CUDA_CALL(cudaEventDestroy(stop));
    CUDA_CALL(cudaFree(A_d));
    CUDA_CALL(cudaFree(B_d));
    CUDA_CALL(cudaFree(tau_d));
    CUDA_CALL(cudaFree(workspace_d));
    CUDA_CALL(cudaFree(deviceInfo_d));
    CUBLAS_CALL(cublasDestroy_v2(cublasH));
    CUSOLVER_CALL(cusolverDnDestroy(cusolverH));
    CUDA_CALL(cudaFree(elem_d));
    CUDA_CALL(cudaFree(pt_d));
    free(A);
    return EXIT_SUCCESS;
}

int bemSolver_mp(const float k, const tri_elem *elem, const int numElem, 
        const rect_coord_flt *nod, const int numNod, const rect_coord_flt *chief, const int numCHIEF, 
        const rect_coord_flt *src, const int numSrc, cuFloatComplex *B, const int ldb)
{
    int i, j;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    //Move elements to GPU
    tri_elem *elem_d;
    CUDA_CALL(cudaMalloc(&elem_d,numElem*sizeof(tri_elem)));
    CUDA_CALL(cudaMemcpy(elem_d,elem,numElem*sizeof(tri_elem),cudaMemcpyHostToDevice));
    
    //Move points to GPU
    // rect_coord_flt *pt_h = (rect_coord_flt*)malloc((numNod+numCHIEF)*sizeof(rect_coord_flt));
    // for(i=0;i<numNod;i++) {
    //     pt_h[i] = nod[i];
    // }
    // for(i=0;i<numCHIEF;i++) {
    //     pt_h[numNod+i] = chief[i];
    // }
    
    rect_coord_flt *pt_d;
    CUDA_CALL(cudaMalloc(&pt_d, (numNod + numCHIEF) * sizeof(rect_coord_flt)));
    CUDA_CALL(cudaMemcpy(pt_d, nod, numNod * sizeof(rect_coord_flt),cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(pt_d + numNod, chief, numCHIEF * sizeof(rect_coord_flt),cudaMemcpyHostToDevice));
    
    CUDA_CALL(cudaEventRecord(start));
    //Generate the system
    cuFloatComplex *A = (cuFloatComplex*)malloc((numNod+numCHIEF)*numNod*sizeof(cuFloatComplex));
    
    memset(A,0,(numNod+numCHIEF)*numNod*sizeof(cuFloatComplex));
    memset(B,0,(numNod+numCHIEF)*numSrc*sizeof(cuFloatComplex));

    for(i=0;i<numNod;i++) 
    {
        A[IDXC0(i,i,numNod+numCHIEF)] = make_cuFloatComplex(1,0);
    }
    
    //Initialization of B
    for(i=0;i<numNod+numCHIEF;i++) 
    {
        for(j=0;j<numSrc;j++) 
        {
            if(i < numNod)
                B[IDXC0(i,j,ldb)] = mpSrc(k,STRENGTH,src[j],nod[i]);
            else
                B[IDXC0(i,j,ldb)] = mpSrc(k,STRENGTH,src[j],chief[i-numNod]);
        }
    }
    
    cuFloatComplex *A_d, *B_d;
    CUDA_CALL(cudaMalloc(&A_d,(numNod+numCHIEF)*numNod*sizeof(cuFloatComplex)));
    CUDA_CALL(cudaMemcpy(A_d,A,(numNod+numCHIEF)*numNod*sizeof(cuFloatComplex),cudaMemcpyHostToDevice));
    
    CUDA_CALL(cudaMalloc(&B_d,(numNod+numCHIEF)*numSrc*sizeof(cuFloatComplex)));
    CUDA_CALL(cudaMemcpy(B_d,B,(numNod+numCHIEF)*numSrc*sizeof(cuFloatComplex),cudaMemcpyHostToDevice));
    
    int xNumBlocks, xWidth = 16, yNumBlocks, yWidth = 16;
    xNumBlocks = (numNod+numCHIEF+xWidth-1)/xWidth;
    yNumBlocks = (numElem+yWidth-1)/yWidth;
    dim3 gridLayout, blockLayout;
    gridLayout.x = xNumBlocks;
    gridLayout.y = yNumBlocks;
    
    blockLayout.x = xWidth;
    blockLayout.y = yWidth;
    
    atomicPtsElems_nsgl<<<gridLayout,blockLayout>>>(k,pt_d,numNod,0,numNod+numCHIEF-1,
            elem_d,numElem,A_d,numNod+numCHIEF,B_d,numSrc,ldb);
    atomicPtsElems_sgl<<<yNumBlocks,yWidth>>>(k,pt_d,elem_d,numElem,A_d,numNod+numCHIEF,
            B_d,numSrc,ldb);
    
    //Solving the system
    cusolverDnHandle_t cusolverH = NULL;
    CUSOLVER_CALL(cusolverDnCreate(&cusolverH));
    
    //A = QR
    int lwork;
    CUSOLVER_CALL(cusolverDnCgeqrf_bufferSize(cusolverH,numNod+numCHIEF,numNod,A_d
            ,numNod+numCHIEF,&lwork));
    
    cuFloatComplex *workspace_d;
    CUDA_CALL(cudaMalloc(&workspace_d,lwork*sizeof(cuFloatComplex)));
    cuFloatComplex *tau_d;
    CUDA_CALL(cudaMalloc(&tau_d,(numNod+numCHIEF)*sizeof(cuFloatComplex)));
    int *deviceInfo_d, deviceInfo;
    CUDA_CALL(cudaMalloc(&deviceInfo_d,sizeof(int)));
    
    
    CUSOLVER_CALL(cusolverDnCgeqrf(cusolverH,numNod+numCHIEF,numNod,A_d,numNod+numCHIEF,
            tau_d,workspace_d,lwork,deviceInfo_d));
    CUDA_CALL(cudaMemcpy(&deviceInfo,deviceInfo_d,sizeof(int),cudaMemcpyDeviceToHost));
    
    //B = (Q^H)*B
    CUSOLVER_CALL(cusolverDnCunmqr(cusolverH,CUBLAS_SIDE_LEFT,CUBLAS_OP_C,numNod+numCHIEF,numSrc,
            numNod,A_d,numNod+numCHIEF,tau_d,B_d,ldb,workspace_d,lwork,deviceInfo_d));
    CUDA_CALL(cudaMemcpy(&deviceInfo,deviceInfo_d,sizeof(int),cudaMemcpyDeviceToHost));
    
    //Solve Rx = B
    cuFloatComplex alpha = make_cuFloatComplex(1,0);
    cublasHandle_t cublasH;
    CUBLAS_CALL(cublasCreate_v2(&cublasH));
    CUBLAS_CALL(cublasCtrsm_v2(cublasH,CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_UPPER,
            CUBLAS_OP_N,CUBLAS_DIAG_NON_UNIT,numNod,numSrc,&alpha,A_d,numNod+numCHIEF,B_d,ldb));
    CUDA_CALL(cudaEventRecord(stop));
    CUDA_CALL(cudaEventSynchronize(stop));
    CUDA_CALL(cudaMemcpy(B,B_d,ldb*numSrc*sizeof(cuFloatComplex),cudaMemcpyDeviceToHost));
    
    float milliseconds = 0;
    CUDA_CALL(cudaEventElapsedTime(&milliseconds,start,stop));
    printf("Elapsed system solving time: %f milliseconds.\n",milliseconds);
    
    //release memory
    CUDA_CALL(cudaEventDestroy(start));
    CUDA_CALL(cudaEventDestroy(stop));
    CUDA_CALL(cudaFree(A_d));
    CUDA_CALL(cudaFree(B_d));
    CUDA_CALL(cudaFree(tau_d));
    CUDA_CALL(cudaFree(workspace_d));
    CUDA_CALL(cudaFree(deviceInfo_d));
    CUBLAS_CALL(cublasDestroy_v2(cublasH));
    CUSOLVER_CALL(cusolverDnDestroy(cusolverH));
    CUDA_CALL(cudaFree(elem_d));
    CUDA_CALL(cudaFree(pt_d));
    free(A);
    return EXIT_SUCCESS;
}

int bemSolver_dir(const float k, const tri_elem *elem, const int numElem, 
        const rect_coord_flt *nod, const int numNod, const rect_coord_flt *chief, const int numCHIEF, 
        const rect_coord_flt *dir, const int numSrc, cuFloatComplex *B, const int ldb)
{
    int i, j;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    //Move elements to GPU
    tri_elem *elem_d;
    CUDA_CALL(cudaMalloc(&elem_d,numElem*sizeof(tri_elem)));
    CUDA_CALL(cudaMemcpy(elem_d,elem,numElem*sizeof(tri_elem),cudaMemcpyHostToDevice));
    
    //Move points to GPU
    // rect_coord_flt *pt_h = (rect_coord_flt*)malloc((numNod+numCHIEF)*sizeof(rect_coord_flt));
    // for(i=0;i<numNod;i++) {
    //     pt_h[i] = nod[i];
    // }
    // for(i=0;i<numCHIEF;i++) {
    //     pt_h[numNod+i] = chief[i];
    // }
    
    rect_coord_flt *pt_d;
    CUDA_CALL(cudaMalloc(&pt_d,(numNod+numCHIEF)*sizeof(rect_coord_flt)));
    CUDA_CALL(cudaMemcpy(pt_d,nod,numNod*sizeof(rect_coord_flt),cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(pt_d+numNod,chief,numCHIEF*sizeof(rect_coord_flt),cudaMemcpyHostToDevice));
    
    CUDA_CALL(cudaEventRecord(start));
    //Generate the system
    cuFloatComplex *A = (cuFloatComplex*)malloc((numNod+numCHIEF)*numNod*sizeof(cuFloatComplex));
    memset(A,0,(numNod+numCHIEF)*numNod*sizeof(cuFloatComplex));

    for(i=0;i<numNod;i++) 
    {
        A[IDXC0(i,i,numNod+numCHIEF)] = make_cuFloatComplex(1,0);
    }
    
    //Initialization of B
    for(i=0;i<numNod+numCHIEF;i++) 
    {
        for(j=0;j<numSrc;j++) 
        {
            if(i < numNod)
                //B[IDXC0(i,j,ldb)] = ptSrc(k,STRENGTH,src[j],nod[i]);
                B[IDXC0(i,j,ldb)] = dirSrc(k,STRENGTH,dir[j],nod[i]);
            else
                //B[IDXC0(i,j,ldb)] = ptSrc(k,STRENGTH,src[j],chief[i - numNod]);
                B[IDXC0(i,j,ldb)] = dirSrc(k,STRENGTH,dir[j],chief[i-numNod]);
        }
    }
    
    cuFloatComplex *A_d, *B_d;
    CUDA_CALL(cudaMalloc(&A_d,(numNod+numCHIEF)*numNod*sizeof(cuFloatComplex)));
    CUDA_CALL(cudaMemcpy(A_d,A,(numNod+numCHIEF)*numNod*sizeof(cuFloatComplex),cudaMemcpyHostToDevice));
    
    CUDA_CALL(cudaMalloc(&B_d,(numNod+numCHIEF)*numSrc*sizeof(cuFloatComplex)));
    CUDA_CALL(cudaMemcpy(B_d,B,(numNod+numCHIEF)*numSrc*sizeof(cuFloatComplex),cudaMemcpyHostToDevice));
    
    int xNumBlocks, xWidth = 16, yNumBlocks, yWidth = 16;
    xNumBlocks = (numNod+numCHIEF+xWidth-1)/xWidth;
    yNumBlocks = (numElem+yWidth-1)/yWidth;
    dim3 gridLayout, blockLayout;
    gridLayout.x = xNumBlocks;
    gridLayout.y = yNumBlocks;
    
    blockLayout.x = xWidth;
    blockLayout.y = yWidth;
    
    atomicPtsElems_nsgl<<<gridLayout,blockLayout>>>(k,pt_d,numNod,0,numNod+numCHIEF-1,
            elem_d,numElem,A_d,numNod+numCHIEF,B_d,numSrc,ldb);
    atomicPtsElems_sgl<<<yNumBlocks,yWidth>>>(k,pt_d,elem_d,numElem,A_d,numNod+numCHIEF,
            B_d,numSrc,ldb);
    
    //Solving the system
    cusolverDnHandle_t cusolverH = NULL;
    CUSOLVER_CALL(cusolverDnCreate(&cusolverH));
    
    //A = QR
    int lwork;
    CUSOLVER_CALL(cusolverDnCgeqrf_bufferSize(cusolverH,numNod+numCHIEF,numNod,A_d
            ,numNod+numCHIEF,&lwork));
    
    cuFloatComplex *workspace_d;
    CUDA_CALL(cudaMalloc(&workspace_d,lwork*sizeof(cuFloatComplex)));
    cuFloatComplex *tau_d;
    CUDA_CALL(cudaMalloc(&tau_d,(numNod+numCHIEF)*sizeof(cuFloatComplex)));
    int *deviceInfo_d, deviceInfo;
    CUDA_CALL(cudaMalloc(&deviceInfo_d,sizeof(int)));
    
    
    CUSOLVER_CALL(cusolverDnCgeqrf(cusolverH,numNod+numCHIEF,numNod,A_d,numNod+numCHIEF,
            tau_d,workspace_d,lwork,deviceInfo_d));
    CUDA_CALL(cudaMemcpy(&deviceInfo,deviceInfo_d,sizeof(int),cudaMemcpyDeviceToHost));
    
    //B = (Q^H)*B
    CUSOLVER_CALL(cusolverDnCunmqr(cusolverH,CUBLAS_SIDE_LEFT,CUBLAS_OP_C,numNod+numCHIEF,numSrc,
            numNod,A_d,numNod+numCHIEF,tau_d,B_d,ldb,workspace_d,lwork,deviceInfo_d));
    CUDA_CALL(cudaMemcpy(&deviceInfo,deviceInfo_d,sizeof(int),cudaMemcpyDeviceToHost));
    
    //Solve Rx = B
    cuFloatComplex alpha = make_cuFloatComplex(1,0);
    cublasHandle_t cublasH;
    CUBLAS_CALL(cublasCreate_v2(&cublasH));
    CUBLAS_CALL(cublasCtrsm_v2(cublasH,CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_UPPER,
            CUBLAS_OP_N,CUBLAS_DIAG_NON_UNIT,numNod,numSrc,&alpha,A_d,numNod+numCHIEF,B_d,ldb));
    CUDA_CALL(cudaEventRecord(stop));
    CUDA_CALL(cudaEventSynchronize(stop));
    CUDA_CALL(cudaMemcpy(B,B_d,ldb*numSrc*sizeof(cuFloatComplex),cudaMemcpyDeviceToHost));
    
    float milliseconds = 0;
    CUDA_CALL(cudaEventElapsedTime(&milliseconds,start,stop));
    printf("Elapsed system solving time: %f milliseconds.\n",milliseconds);
    
    //release memory
    CUDA_CALL(cudaEventDestroy(start));
    CUDA_CALL(cudaEventDestroy(stop));
    CUDA_CALL(cudaFree(A_d));
    CUDA_CALL(cudaFree(B_d));
    CUDA_CALL(cudaFree(tau_d));
    CUDA_CALL(cudaFree(workspace_d));
    CUDA_CALL(cudaFree(deviceInfo_d));
    CUBLAS_CALL(cublasDestroy_v2(cublasH));
    CUSOLVER_CALL(cusolverDnDestroy(cusolverH));
    CUDA_CALL(cudaFree(elem_d));
    CUDA_CALL(cudaFree(pt_d));
    free(A);
    return EXIT_SUCCESS;
}

/*
__host__ gsl_complex gsl_sf_bessel_hl(const int l, const double s)
{
    double x = gsl_sf_bessel_jl(l,s);
    double y = gsl_sf_bessel_yl(l,s);
    gsl_complex z = gsl_complex_rect(x,y);
    return z;
}

double jprime(const int n, const double r)
{
    double result;
    if(n == 0) {
        result = -gsl_sf_bessel_jl(1,r);
    } else {
        result = gsl_sf_bessel_jl(n-1,r)-(n+1)*gsl_sf_bessel_jl(n,r)/r;
    }
    return result;
}

gsl_complex hprime(const int n, const double r)
{
    gsl_complex result;
    if(n == 0) {
        result = gsl_complex_negative(gsl_sf_bessel_hl(1,r));
    } else {
        result = gsl_complex_sub(gsl_sf_bessel_hl(n-1,r),gsl_complex_mul_real(gsl_sf_bessel_hl(n,r),(n+1)/r));
    }
    return result;
}
*/

__host__ __device__ sph_coord_float rect2sph(const rect_coord_flt s)
{
    sph_coord_float temp;
    temp.coords[0] = sqrtf(powf(s.coords[0],2)+powf(s.coords[1],2)+powf(s.coords[2],2));
    temp.coords[1] = acosf(s.coords[2]/(temp.coords[0]));
    temp.coords[2] = atan2f(s.coords[1],s.coords[0]);
    return temp;
}

__host__ __device__ rect_coord_flt sph2rect(const sph_coord_float s)
{
    float r = s.coords[0], theta = s.coords[1], phi = s.coords[2];
    float x = r*sinf(theta)*cosf(phi), y = r*sinf(theta)*sinf(phi), z = r*cosf(theta);
    rect_coord_flt result;
    result.coords[0] = x;
    result.coords[1] = y;
    result.coords[2] = z;
    return result;
}

__host__ __device__ sph_coord_double rect2sph(const rect_coord_dbl s)
{
    sph_coord_double temp;
    temp.coords[0] = sqrt(pow(s.coords[0],2)+pow(s.coords[1],2)+pow(s.coords[2],2));
    temp.coords[1] = acos(s.coords[2]/(temp.coords[0]));
    temp.coords[2] = atan2(s.coords[1],s.coords[0]);
    return temp;
}

__host__ __device__ rect_coord_dbl sph2rect(const sph_coord_double s)
{
    double r = s.coords[0], theta = s.coords[1], phi = s.coords[2];
    double x = r*sin(theta)*cos(phi), y = r*sin(theta)*sin(phi), z = r*cos(theta);
    rect_coord_dbl result;
    result.coords[0] = x;
    result.coords[1] = y;
    result.coords[2] = z;
    return result;
}

/*
void computeRigidSphereScattering(const rect_coord_flt *pt, const int numPt, const double a, 
        const double r, const double wavNum, const double strength)
{
    gsl_complex *p = (gsl_complex*)malloc(numPt*sizeof(gsl_complex));
    sph_coord_float tempCoord;
    gsl_complex result;
    //double temp;
    //const int truncNum = 30;
    for(int i=0;i<numPt;i++)
    {
        tempCoord = rect2sph(pt[i]);
        result = rigidSphereScattering(wavNum,strength,a,tempCoord.coords[0],tempCoord.coords[1]);
        p[i] = result;
        printf("(%.8f,%.8f)\n",GSL_REAL(p[i]),GSL_IMAG(p[i]));
    }
    free(p);
}

gsl_complex rigidSphereScattering(const double wavNum, const double strength, const double a, 
        const double r, const double theta)
{
    gsl_complex result = gsl_complex_rect(0,0), temp_c;
    const int numTrunc = 70;
    for(int n=0;n<numTrunc;n++)
    {
        temp_c = gsl_complex_div(gsl_complex_rect(jprime(n,wavNum*a),0),hprime(n,wavNum*a));
        temp_c = gsl_complex_mul(temp_c,gsl_sf_bessel_hl(n,wavNum*r));
        temp_c = gsl_complex_sub(gsl_complex_rect(gsl_sf_bessel_jl(n,wavNum*r),0),temp_c);
        temp_c = gsl_complex_mul(gsl_complex_pow_real(gsl_complex_rect(0,1),n),temp_c);
        temp_c = gsl_complex_mul_real(temp_c,2*n+1);
        temp_c = gsl_complex_mul_real(temp_c,gsl_sf_legendre_Pl(n,cos(theta)));
        result = gsl_complex_add(result,temp_c);
    }
    result = gsl_complex_mul_real(result,strength);
    return result;
}
*/

__device__ cuFloatComplex extrapolation_dir(const float wavNum, const rect_coord_flt x, 
        const tri_elem* elem, const int numElem, const rect_coord_flt* pt, 
        const cuFloatComplex* p, const float strength, const rect_coord_flt dir)
{
    /*field extrapolation from the surface to a single point in free space
     wavNum: wave number
     elem: pointer for all elements
     pt: pointer for all points
     x: the point in free space
     dir: the direction of the plane wave*/
    cuFloatComplex result = dirSrc(wavNum,strength,dir,x);
    cuFloatComplex temp;
    for(int i=0;i<numElem;i++) {
        rect_coord_flt nod[3];
        for(int j=0;j<3;j++) {
            nod[j] = pt[elem[i].nod[j]];
        }
        cuFloatComplex gCoeff[3], hCoeff[3]; 
        float cCoeff[3];
        g_h_c_nsgl(wavNum,x,nod,gCoeff,hCoeff,cCoeff);
        for(int j=0;j<3;j++) {
            temp = cuCdivf(elem[i].bc[2],elem[i].bc[1]);
            temp = cuCmulf(temp,gCoeff[j]);
            result = cuCsubf(result,temp);
            temp = cuCdivf(elem[i].bc[0],elem[i].bc[1]);
            temp = cuCmulf(temp,gCoeff[j]);
            temp = cuCsubf(hCoeff[j],temp);
            temp = cuCmulf(temp,p[elem[i].nod[j]]);
            result = cuCsubf(result,temp);
        }
    }
    return result;
}

__device__ cuFloatComplex extrapolation_pt(const float wavNum, const rect_coord_flt x, 
        const tri_elem* elem, const int numElem, const rect_coord_flt* pt, 
        const cuFloatComplex* p, const float strength, const rect_coord_flt src)
{
    /*field extrapolation from the surface to a single point in free space
     x: the single point in free space
     elem: pointer to mesh elements
     pt: pointer to mesh nod and chief points
     p: surface pressure
     strength: intensity of the source
     src: source location*/
    cuFloatComplex result = ptSrc(wavNum,strength,src,x);
    cuFloatComplex temp;
    for(int i=0;i<numElem;i++) {
        rect_coord_flt nod[3];
        for(int j=0;j<3;j++) {
            nod[j] = pt[elem[i].nod[j]];
        }
        cuFloatComplex gCoeff[3], hCoeff[3]; 
        float cCoeff[3];
        g_h_c_nsgl(wavNum,x,nod,gCoeff,hCoeff,cCoeff);
        for(int j=0;j<3;j++) {
            temp = cuCdivf(elem[i].bc[2],elem[i].bc[1]);
            temp = cuCmulf(temp,gCoeff[j]);
            result = cuCsubf(result,temp);
            temp = cuCdivf(elem[i].bc[0],elem[i].bc[1]);
            temp = cuCmulf(temp,gCoeff[j]);
            temp = cuCsubf(hCoeff[j],temp);
            temp = cuCmulf(temp,p[elem[i].nod[j]]);
            result = cuCsubf(result,temp);
        }
    }
    return result;
}

__device__ cuFloatComplex extrapolation_mp(const float wavNum, const rect_coord_flt x, 
        const tri_elem* elem, const int numElem, const rect_coord_flt* pt, 
        const cuFloatComplex* p, const float strength, const rect_coord_flt src)
{
    /*field extrapolation from the surface to a single monopole in free space
     x: the single point in free space
     elem: pointer to mesh elements
     pt: pointer to mesh nod and chief points
     p: surface pressure
     strength: intensity of the source
     src: source location
     return: sound pressure at the extrapolation point*/
    cuFloatComplex result = mpSrc(wavNum,strength,src,x);
    cuFloatComplex temp;
    for(int i=0;i<numElem;i++) {
        rect_coord_flt nod[3];
        for(int j=0;j<3;j++) {
            nod[j] = pt[elem[i].nod[j]];
        }
        cuFloatComplex gCoeff[3], hCoeff[3]; 
        float cCoeff[3];
        g_h_c_nsgl(wavNum,x,nod,gCoeff,hCoeff,cCoeff);
        for(int j=0;j<3;j++) {
            temp = cuCdivf(elem[i].bc[2],elem[i].bc[1]);
            temp = cuCmulf(temp,gCoeff[j]);
            result = cuCsubf(result,temp);
            temp = cuCdivf(elem[i].bc[0],elem[i].bc[1]);
            temp = cuCmulf(temp,gCoeff[j]);
            temp = cuCsubf(hCoeff[j],temp);
            temp = cuCmulf(temp,p[elem[i].nod[j]]);
            result = cuCsubf(result,temp);
        }
    }
    return result;
}

__global__ void extrapolations_dir(const float wavNum, const rect_coord_flt* expPt, const int numExpPt,
        const tri_elem* elem, const int numElem, const rect_coord_flt* pt, const cuFloatComplex* p, 
        const float strength, const rect_coord_flt dir, cuFloatComplex *p_exp)
{
    /*
     extrapolation from surface pressure to multiple points in free space
     wavNum: wave number
     expPt: extrapolation points in free space
     p: surface pressure
     dir: direction of the plane wave
     p_exp: pressure at the extrapolation points
     */
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx < numExpPt) {
        p_exp[idx] = extrapolation_dir(wavNum,expPt[idx],elem,numElem,pt,p,strength,dir);
    }
}

__global__ void extrapolations_pt(const float wavNum, const rect_coord_flt* expPt, const int numExpPt,
        const tri_elem* elem, const int numElem, const rect_coord_flt* pt, const cuFloatComplex* p, 
        const float strength, const rect_coord_flt src, cuFloatComplex *p_exp)
{
    /*extrapolation from surface pressure to multiple points in free space
     wavNum: wave number
     expPt:  extrapolation  points in free space
     p: surface pressure
     src: location of the source
     p_exp: pressure at the extrapolation points*/
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx < numExpPt) {
        p_exp[idx] = extrapolation_pt(wavNum,expPt[idx],elem,numElem,pt,p,strength,src);
    }
}

__global__ void extrapolations_mp(const float wavNum, const rect_coord_flt* expPt, const int numExpPt,
        const tri_elem* elem, const int numElem, const rect_coord_flt* pt, const cuFloatComplex* p, 
        const float strength, const rect_coord_flt src, cuFloatComplex *p_exp)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx < numExpPt) {
        p_exp[idx] = extrapolation_mp(wavNum,expPt[idx],elem,numElem,pt,p,strength,src);
    }
}

int field_extrapolation_single_dir(const float wavNum, const rect_coord_flt* expPt, const int numExpPt, 
        const tri_elem* elem, const int numElem, const rect_coord_flt* pt, const int numPt, 
        const cuFloatComplex* p, const float strength, const rect_coord_flt dir, cuFloatComplex *pExp)
{
    /*extrapolation of acoustic field from surface pressure
     wavNum: wave number
     expPt: extrapolation points in free space
     elem: pointer to mesh elements
     pt: pointer to mesh nod and chief points
     p: surface pressure
     strength: intensity of the sound source
     dir: direction of the plane wave
     pExp: pressure at extrapolation points*/
    int width = 16, numBlock = (numExpPt+width-1)/width;
    
    // allocate memory on GPU and copy data to GPU memory
    rect_coord_flt *expPt_d, *pt_d;
    tri_elem *elem_d;
    cuFloatComplex *p_d, *pExp_d;
    
    CUDA_CALL(cudaMalloc(&expPt_d,numExpPt*sizeof(rect_coord_flt)));
    CUDA_CALL(cudaMemcpy(expPt_d,expPt,numExpPt*sizeof(rect_coord_flt),cudaMemcpyHostToDevice));
    
    CUDA_CALL(cudaMalloc(&pt_d,numPt*sizeof(rect_coord_flt)));
    CUDA_CALL(cudaMemcpy(pt_d,pt,numPt*sizeof(rect_coord_flt),cudaMemcpyHostToDevice));
    
    CUDA_CALL(cudaMalloc(&elem_d,numElem*sizeof(tri_elem)));
    CUDA_CALL(cudaMemcpy(elem_d,elem,numElem*sizeof(tri_elem),cudaMemcpyHostToDevice));
    
    CUDA_CALL(cudaMalloc(&p_d,numPt*sizeof(cuFloatComplex)));
    CUDA_CALL(cudaMemcpy(p_d,p,numPt*sizeof(cuFloatComplex),cudaMemcpyHostToDevice));
    
    CUDA_CALL(cudaMalloc(&pExp_d,numExpPt*sizeof(cuFloatComplex)));
    
    extrapolations_dir<<<numBlock,width>>>(wavNum,expPt_d,numExpPt,elem_d,numElem,pt_d,p_d,
            strength,dir,pExp_d);
    
    CUDA_CALL(cudaMemcpy(pExp,pExp_d,numExpPt*sizeof(cuFloatComplex),cudaMemcpyDeviceToHost));
    
    CUDA_CALL(cudaFree(expPt_d));
    CUDA_CALL(cudaFree(pt_d));
    CUDA_CALL(cudaFree(elem_d));
    CUDA_CALL(cudaFree(p_d));
    CUDA_CALL(cudaFree(pExp_d));
    
    return EXIT_SUCCESS;
}

int field_extrapolation_single_pt(const float wavNum, const rect_coord_flt* expPt, const int numExpPt, 
        const tri_elem* elem, const int numElem, const rect_coord_flt* pt, const int numPt, 
        const cuFloatComplex* p, const float strength, const rect_coord_flt src, cuFloatComplex *pExp)
{
    /*Extrapolation of an acoustic field from surface pressure and a single point source
     wavNum: wave number
     expPt: pointer for extrapolation points
     elem: mesh elements
     pt: nod and chief points
     p: surface pressure
     strength: intensity of a source
     src: location of the point source
     pExp: pressure at extrapolation points*/
    int width = 16, numBlock = (numExpPt+width-1)/width;
    
    // allocate memory on GPU and copy data to GPU memory
    rect_coord_flt *expPt_d, *pt_d;
    tri_elem *elem_d;
    cuFloatComplex *p_d, *pExp_d;
    
    CUDA_CALL(cudaMalloc(&expPt_d,numExpPt*sizeof(rect_coord_flt)));
    CUDA_CALL(cudaMemcpy(expPt_d,expPt,numExpPt*sizeof(rect_coord_flt),cudaMemcpyHostToDevice));
    
    CUDA_CALL(cudaMalloc(&pt_d,numPt*sizeof(rect_coord_flt)));
    CUDA_CALL(cudaMemcpy(pt_d,pt,numPt*sizeof(rect_coord_flt),cudaMemcpyHostToDevice));
    
    CUDA_CALL(cudaMalloc(&elem_d,numElem*sizeof(tri_elem)));
    CUDA_CALL(cudaMemcpy(elem_d,elem,numElem*sizeof(tri_elem),cudaMemcpyHostToDevice));
    
    CUDA_CALL(cudaMalloc(&p_d,numPt*sizeof(cuFloatComplex)));
    CUDA_CALL(cudaMemcpy(p_d,p,numPt*sizeof(cuFloatComplex),cudaMemcpyHostToDevice));
    
    CUDA_CALL(cudaMalloc(&pExp_d,numExpPt*sizeof(cuFloatComplex)));
    
    extrapolations_pt<<<numBlock,width>>>(wavNum,expPt_d,numExpPt,elem_d,numElem,pt_d,p_d,
            strength,src,pExp_d);
    
    CUDA_CALL(cudaMemcpy(pExp,pExp_d,numExpPt*sizeof(cuFloatComplex),cudaMemcpyDeviceToHost));
    
    CUDA_CALL(cudaFree(expPt_d));
    CUDA_CALL(cudaFree(pt_d));
    CUDA_CALL(cudaFree(elem_d));
    CUDA_CALL(cudaFree(p_d));
    CUDA_CALL(cudaFree(pExp_d));
    
    return EXIT_SUCCESS;
}

int field_extrapolation_single_mp(const float wavNum, const rect_coord_flt* expPt, const int numExpPt, 
        const tri_elem* elem, const int numElem, const rect_coord_flt* pt, const int numPt, 
        const cuFloatComplex* p, const float strength, const rect_coord_flt src, cuFloatComplex *pExp)
{
    int width = 16, numBlock = (numExpPt+width-1)/width;
    
    // allocate memory on GPU and copy data to GPU memory
    rect_coord_flt *expPt_d, *pt_d;
    tri_elem *elem_d;
    cuFloatComplex *p_d, *pExp_d;
    
    CUDA_CALL(cudaMalloc(&expPt_d,numExpPt*sizeof(rect_coord_flt)));
    CUDA_CALL(cudaMemcpy(expPt_d,expPt,numExpPt*sizeof(rect_coord_flt),cudaMemcpyHostToDevice));
    
    CUDA_CALL(cudaMalloc(&pt_d,numPt*sizeof(rect_coord_flt)));
    CUDA_CALL(cudaMemcpy(pt_d,pt,numPt*sizeof(rect_coord_flt),cudaMemcpyHostToDevice));
    
    CUDA_CALL(cudaMalloc(&elem_d,numElem*sizeof(tri_elem)));
    CUDA_CALL(cudaMemcpy(elem_d,elem,numElem*sizeof(tri_elem),cudaMemcpyHostToDevice));
    
    CUDA_CALL(cudaMalloc(&p_d,numPt*sizeof(cuFloatComplex)));
    CUDA_CALL(cudaMemcpy(p_d,p,numPt*sizeof(cuFloatComplex),cudaMemcpyHostToDevice));
    
    CUDA_CALL(cudaMalloc(&pExp_d,numExpPt*sizeof(cuFloatComplex)));
    
    extrapolations_mp<<<numBlock,width>>>(wavNum,expPt_d,numExpPt,elem_d,numElem,pt_d,p_d,
            strength,src,pExp_d);
    
    CUDA_CALL(cudaMemcpy(pExp,pExp_d,numExpPt*sizeof(cuFloatComplex),cudaMemcpyDeviceToHost));
    
    CUDA_CALL(cudaFree(expPt_d));
    CUDA_CALL(cudaFree(pt_d));
    CUDA_CALL(cudaFree(elem_d));
    CUDA_CALL(cudaFree(p_d));
    CUDA_CALL(cudaFree(pExp_d));
    
    return EXIT_SUCCESS;
}

rect_coord_flt rectCoordDbl2rectCoordFlt(const rect_coord_dbl t)
{
    rect_coord_flt result;
    for(int i=0;i<3;i++) {
        result.coords[i] = t.coords[i];
    }
    return result;
}

void rectCoordDblArr2rectCoordFltArr(const rect_coord_dbl* dArr, 
        const int num, rect_coord_flt* fArr)
{
    for(int i=0;i<num;i++) {
        fArr[i] = rectCoordDbl2rectCoordFlt(dArr[i]);
    }
}

void reorgField(cuFloatComplex* field, const int l)
{
    /*re-organize the acoustic fields from the order of z, y, x to x, y, z*/
    int totalNum = pow(8,l), dimNum = pow(2,l);
    cuFloatComplex *temp = (cuFloatComplex*)malloc(totalNum*sizeof(cuFloatComplex));
    memcpy(temp,field,totalNum*sizeof(cuFloatComplex));
    
    // reorganize
    for(int x=0;x<dimNum;x++) {
        for(int y=0;y<dimNum;y++) {
            for(int z=0;z<dimNum;z++) {
                int idx_old = x*dimNum*dimNum+y*dimNum+z;
                int idx_new = z*dimNum*dimNum+y*dimNum+x;
                field[idx_new] = temp[idx_old];
            }
        }
    }
    free(temp);
}

int genFields_MultiPtSrcSglObj(const float strength, const float wavNum, 
        const rect_coord_flt* srcs, const int numSrcs, const rect_coord_dbl* pts, const int numPts, 
        const tri_elem* elems, const int numElems, const rect_coord_dbl cnr, const double d, 
        const int level, cuFloatComplex* fields)
{
    /*generate an acoustic field with a given boundary
     level: octree level
     cnr: lowest corner of the bounding box
     d: side length of the bounding box
     fields: pressure array equal to the number of boxes at level l times number of sources*/
    rect_coord_flt *pts_f = (rect_coord_flt*)malloc(numPts*sizeof(rect_coord_flt));
    rectCoordDblArr2rectCoordFltArr(pts,numPts,pts_f);
    
    // generate chief points
    rect_coord_flt chief[NUMCHIEF];
    genCHIEF(pts_f,numPts,elems,numElems,chief,NUMCHIEF);
    
    // allocate memory for the right-hand side of the linear system
    cuFloatComplex *B = (cuFloatComplex*)malloc((numPts+NUMCHIEF)*numSrcs*sizeof(cuFloatComplex));
    // solve the linear system to get the surface pressure
    HOST_CALL(bemSolver_mp(wavNum,elems,numElems,pts_f,numPts,chief,NUMCHIEF,srcs,numSrcs,B,numPts+NUMCHIEF));
    
    // compute the extrapolation points of the field
    // note that the indices first increase in z, then in y and at last in x
    int numExpPts = (int)pow(8,level);
    rect_coord_dbl *expPts = (rect_coord_dbl*)malloc(numExpPts*sizeof(rect_coord_dbl));
    for(int i=0;i<numExpPts;i++) {
        rect_coord_dbl pt_scaled = boxCenter(i,level);
        rect_coord_dbl pt_descaled = descale(pt_scaled,cnr,d);
        expPts[i] = pt_descaled;
    }
    rect_coord_flt *expPts_f = (rect_coord_flt*)malloc(numExpPts*sizeof(rect_coord_flt));
    rectCoordDblArr2rectCoordFltArr(expPts,numExpPts,expPts_f);
    free(expPts);
    
    // extrapolate the acoustic field from the surface to free space
    cuFloatComplex *field = (cuFloatComplex*)malloc(numExpPts*sizeof(cuFloatComplex));
    for(int i=0;i<numSrcs;i++) {
        HOST_CALL(field_extrapolation_single_pt(wavNum,expPts_f,numExpPts,elems,numElems,
                pts_f,numPts,&B[i*(numPts+NUMCHIEF)],strength,srcs[i],field));
        reorgField(field,level);
        memcpy(&fields[i*numExpPts],field,numExpPts*sizeof(cuFloatComplex));
    }
    
    return EXIT_SUCCESS;
}

__host__ __device__ int deterPtPlaneRel(const rect_coord_dbl pt, const plane_dbl plane)
{
    rect_coord_dbl vec = rectCoordSub(pt,plane.pt);
    double result = rectDotMul(plane.n,vec);
    if(result>=0) {
        // on the positive side of the plane normal
        return 1;
    } else {
        return 0;
    }
}

__host__ __device__ int deterPtCubeEdgeVolRel(const rect_coord_dbl pt, const cube_dbl cb)
{
    /*determine the relationship between a point and the volume bounded by edge faces
     of a cube*/
    
    //declare two rect_coord_dbl arrays for nod at the bottom and the top face 
    rect_coord_dbl btm[4], top[4], left[4], right[4], back[4], front[4];
    
    // declare the basis unit vectors
    rect_coord_dbl dir_x = {1.0,0.0,0.0}, dir_y = {0.0,1.0,0.0}, dir_z = {0.0,0.0,1.0};
    
    //set up btm and top nod
    btm[0] = cb.cnr;
    btm[1] = rectCoordAdd(btm[0],scaRectMul(cb.len,dir_x));
    btm[2] = rectCoordAdd(btm[1],scaRectMul(cb.len,dir_y));
    btm[3] = rectCoordAdd(btm[0],scaRectMul(cb.len,dir_y));
    //printRectCoord(btm,4);
    
    top[0] = rectCoordAdd(btm[0],scaRectMul(cb.len,dir_z));
    top[1] = rectCoordAdd(top[0],scaRectMul(cb.len,dir_x));
    top[2] = rectCoordAdd(top[1],scaRectMul(cb.len,dir_y));
    top[3] = rectCoordAdd(top[0],scaRectMul(cb.len,dir_y));
    //printRectCoord(top,4);
    
    //set up left and right nod
    left[0] = cb.cnr;
    left[1] = rectCoordAdd(left[0],scaRectMul(cb.len,dir_x));
    left[2] = rectCoordAdd(left[1],scaRectMul(cb.len,dir_z));
    left[3] = rectCoordAdd(left[0],scaRectMul(cb.len,dir_z));
    //printRectCoord(left,4);
    
    right[0] = rectCoordAdd(left[0],scaRectMul(cb.len,dir_y));
    right[1] = rectCoordAdd(right[0],scaRectMul(cb.len,dir_x));
    right[2] = rectCoordAdd(right[1],scaRectMul(cb.len,dir_z));
    right[3] = rectCoordAdd(right[0],scaRectMul(cb.len,dir_z));
    //printRectCoord(right,4);
    
    //set up back and front nod
    back[0] = cb.cnr;
    back[1] = rectCoordAdd(back[0],scaRectMul(cb.len,dir_y));
    back[2] = rectCoordAdd(back[1],scaRectMul(cb.len,dir_z));
    back[3] = rectCoordAdd(back[0],scaRectMul(cb.len,dir_z));
    //printRectCoord(back,4);
    
    front[0] = rectCoordAdd(back[0],scaRectMul(cb.len,dir_x));
    front[1] = rectCoordAdd(front[0],scaRectMul(cb.len,dir_y));
    front[2] = rectCoordAdd(front[1],scaRectMul(cb.len,dir_z));
    front[3] = rectCoordAdd(front[0],scaRectMul(cb.len,dir_z));
    //printRectCoord(front,4);
    
    //declare an array nrml for determining the normal of the new plane
    rect_coord_dbl nrml[3];
    plane_dbl plane;
    int result;
    
    //deal with the bottom face
    for(int i=0;i<3;i++) {
        nrml[1] = dir_z;
        switch(i) {
            case 0: // edge determined by btm[0] and btm[1]
                nrml[0] = dir_y;
                break;
            case 1: // edge determined by btm[1] and btm[2]
                nrml[0] = scaRectMul(-1,dir_x);
                break;
            case 2: // edge determined by btm[2] and btm[3]
                nrml[0] = scaRectMul(-1,dir_y);
                break;
            case 3: // edge determined by btm[3] and btm[0]
                nrml[0] = dir_x;
                break;
            default:
                printf("Entered the wrong branch.\n");
        }
        nrml[3] = nrmlzRectCoord(rectCoordAdd(nrml[0],nrml[1]));
        plane.n = nrml[3];
        plane.pt = btm[i];
        result = deterPtPlaneRel(pt,plane);
        if(result == 0) {
            printf("bottom face: %dth node\n",i);
            return 0;
        }
    }
    
    //deal with the top face
    for(int i=0;i<3;i++) {
        nrml[1] = scaRectMul(-1,dir_z);
        switch(i) {
            case 0: // edge determined by top[0] and top[1]
                nrml[0] = dir_y;
                break;
            case 1: // edge determined by top[1] and top[2]
                nrml[0] = scaRectMul(-1,dir_x);
                break;
            case 2: // edge determined by top[2] and top[3]
                nrml[0] = scaRectMul(-1,dir_y);
                break;
            case 3: // edge determined by top[3] and top[0]
                nrml[0] = dir_x;
                break;
            default:
                printf("Entered the wrong branch.\n");
        }
        nrml[3] = nrmlzRectCoord(rectCoordAdd(nrml[0],nrml[1]));
        plane.n = nrml[3];
        plane.pt = top[i];
        result = deterPtPlaneRel(pt,plane);
        if(result == 0) {
            printf("top face: %dth node\n",i);
            return 0;
        }
    }
    
    //deal with the left face
    for(int i=0;i<3;i++) {
        nrml[1] = dir_y;
        switch(i) {
            case 0: // edge determined by left[0] and left[1]
                nrml[0] = dir_z;
                break;
            case 1: // edge determined by left[1] and left[2]
                nrml[0] = scaRectMul(-1,dir_x);
                break;
            case 2: // edge determined by left[2] and left[3]
                nrml[0] = scaRectMul(-1,dir_z);
                break;
            case 3: // edge determined by left[3] and left[0]
                nrml[0] = dir_x;
                break;
            default:
                printf("Entered the wrong branch.\n");
        }
        nrml[3] = nrmlzRectCoord(rectCoordAdd(nrml[0],nrml[1]));
        plane.n = nrml[3];
        plane.pt = left[i];
        result = deterPtPlaneRel(pt,plane);
        if(result == 0) {
            printf("left face: %dth node\n",i);
            return 0;
        }
    }
    
    //deal with the right face
    for(int i=0;i<3;i++) {
        nrml[1] = scaRectMul(-1,dir_y);
        switch(i) {
            case 0: // edge determined by right[0] and right[1]
                nrml[0] = dir_z;
                break;
            case 1: // edge determined by right[1] and right[2]
                nrml[0] = scaRectMul(-1,dir_x);
                break;
            case 2: // edge determined by right[2] and right[3]
                nrml[0] = scaRectMul(-1,dir_z);
                break;
            case 3: // edge determined by btm[3] and btm[0]
                nrml[0] = dir_x;
                break;
            default:
                printf("Entered the wrong branch.\n");
        }
        nrml[3] = nrmlzRectCoord(rectCoordAdd(nrml[0],nrml[1]));
        plane.n = nrml[3];
        plane.pt = right[i];
        result = deterPtPlaneRel(pt,plane);
        if(result == 0) {
            printf("right face: %dth node\n",i);
            return 0;
        }
    }
    
    //deal with the back face
    for(int i=0;i<3;i++) {
        nrml[1] = dir_x;
        switch(i) {
            case 0: // edge determined by back[0] and back[1]
                nrml[0] = dir_z;
                break;
            case 1: // edge determined by btm[1] and btm[2]
                nrml[0] = scaRectMul(-1,dir_y);
                break;
            case 2: // edge determined by btm[2] and btm[3]
                nrml[0] = scaRectMul(-1,dir_z);
                break;
            case 3: // edge determined by btm[3] and btm[0]
                nrml[0] = dir_y;
                break;
            default:
                printf("Entered the wrong branch.\n");
        }
        nrml[3] = nrmlzRectCoord(rectCoordAdd(nrml[0],nrml[1]));
        plane.n = nrml[3];
        plane.pt = back[i];
        result = deterPtPlaneRel(pt,plane);
        if(result == 0) {
            printf("back face: %dth node\n",i);
            return 0;
        }
    }
    
    //deal with the front face
    for(int i=0;i<3;i++) {
        nrml[1] = scaRectMul(-1,dir_x);
        switch(i) {
            case 0: // edge determined by front[0] and front[1]
                nrml[0] = dir_z;
                break;
            case 1: // edge determined by front[1] and front[2]
                nrml[0] = scaRectMul(-1,dir_y);
                break;
            case 2: // edge determined by front[2] and front[3]
                nrml[0] = scaRectMul(-1,dir_z);
                break;
            case 3: // edge determined by front[3] and front[0]
                nrml[0] = dir_y;
                break;
            default:
                printf("Entered the wrong branch.\n");
        }
        nrml[3] = nrmlzRectCoord(rectCoordAdd(nrml[0],nrml[1]));
        plane.n = nrml[3];
        plane.pt = front[i];
        result = deterPtPlaneRel(pt,plane);
        if(result == 0) {
            printf("front face: %dth node\n",i);
            return 0;
        }
    }
    
    // if not returned 0, then return 1, the point is inside the volume
    return 1;
}

__host__ __device__ int deterPtCubeVtxVolRel(const rect_coord_dbl pt, const cube_dbl cb)
{
    // declare the basis unit vectors
    rect_coord_dbl dir_x = {1.0,0.0,0.0}, dir_y = {0.0,1.0,0.0}, dir_z = {0.0,0.0,1.0}, 
            nrml[4], tempPt;
    plane_dbl plane;
    int result;
    // deal with the eight nod in order
    for(int i=0;i<8;i++) {
        switch(i) {
            case 0: //the first vertex
                tempPt = cb.cnr;
                nrml[0] = dir_x;
                nrml[1] = dir_y;
                nrml[2] = dir_z;
                break;
            case 1: //the second vertex
                tempPt = rectCoordAdd(cb.cnr,scaRectMul(cb.len,dir_x));
                nrml[0] = scaRectMul(-1,dir_x);
                nrml[1] = dir_y;
                nrml[2] = dir_z;
                break;
            case 2: //the third vertex
                tempPt = rectCoordAdd(rectCoordAdd(cb.cnr,scaRectMul(cb.len,dir_x)),scaRectMul(cb.len,dir_y));
                nrml[0] = scaRectMul(-1,dir_x);
                nrml[1] = scaRectMul(-1,dir_y);
                nrml[2] = dir_z;
                break;
            case 3: //the fourth vertex
                tempPt = rectCoordAdd(cb.cnr,scaRectMul(cb.len,dir_y));
                nrml[0] = dir_x;
                nrml[1] = scaRectMul(-1,dir_y);
                nrml[2] = dir_z;
                break;
            case 4: //the fifth vertex
                tempPt = rectCoordAdd(cb.cnr,scaRectMul(cb.len,dir_z));
                nrml[0] = dir_x;
                nrml[1] = dir_y;
                nrml[2] = scaRectMul(-1,dir_z);
                break;
            case 5: //the sixth vertex
                tempPt = rectCoordAdd(rectCoordAdd(cb.cnr,scaRectMul(cb.len,dir_z)),scaRectMul(cb.len,dir_x));
                nrml[0] = scaRectMul(-1,dir_x);
                nrml[1] = dir_y;
                nrml[2] = scaRectMul(-1,dir_z);
                break;
            case 6: //the seventh vertex
                tempPt = rectCoordAdd(rectCoordAdd(rectCoordAdd(cb.cnr,scaRectMul(cb.len,dir_z)),
                        scaRectMul(cb.len,dir_x)),scaRectMul(cb.len,dir_y));
                nrml[0] = scaRectMul(-1,dir_x);
                nrml[1] = scaRectMul(-1,dir_y);
                nrml[2] = scaRectMul(-1,dir_z);
                break;
            case 7: //the eighth vertex
                tempPt = rectCoordAdd(rectCoordAdd(cb.cnr,scaRectMul(cb.len,dir_z)),scaRectMul(cb.len,dir_y));
                nrml[0] = dir_x;
                nrml[1] = scaRectMul(-1,dir_y);
                nrml[2] = scaRectMul(-1,dir_z);
                break;
            default:
                printf("safety purpose.\n");
        }
        nrml[3] = nrmlzRectCoord(rectCoordAdd(rectCoordAdd(nrml[0],nrml[1]),nrml[2]));
        plane.n = nrml[3];
        plane.pt = tempPt;
        result = deterPtPlaneRel(pt,plane);
        if(result == 0) {
            return 0;
        }
    }
    return 1;
}

__host__ __device__ int deterPtCubeRel(const rect_coord_dbl pt, const cube_dbl cube)
{
    rect_coord_dbl cnr_fru = cube.cnr;
    cnr_fru = rectCoordAdd(cnr_fru,scaRectMul(cube.len,{1,0,0}));
    cnr_fru = rectCoordAdd(cnr_fru,scaRectMul(cube.len,{0,1,0}));
    cnr_fru = rectCoordAdd(cnr_fru,scaRectMul(cube.len,{0,0,1}));
    double x_min = cube.cnr.coords[0], y_min = cube.cnr.coords[1], z_min = cube.cnr.coords[2], 
            x_max = cnr_fru.coords[0], y_max = cnr_fru.coords[1], z_max = cnr_fru.coords[2],
            x = pt.coords[0], y = pt.coords[1], z = pt.coords[2];
    if(x >= x_min && x<= x_max && y >= y_min && y<= y_max && z >= z_min && z<= z_max) {
        return 1;
    } else {
        return 0;
    }
}

__host__ __device__ int deterLinePlaneRel(const line_dbl ln, const plane_dbl pln, double* t)
{
    if(abs(rectDotMul(ln.dir,pln.n))<EPS) {
        //line parallel to plane
        if(abs(rectDotMul(pln.n,rectCoordSub(ln.pt,pln.pt)))<EPS) {
            return 2;
        } else {
            return 0;
        }
    } else {
        double temp = rectDotMul(pln.n,rectCoordSub(pln.pt,ln.pt))/rectDotMul(pln.n,ln.dir);
        *t = temp;
        return 1;
    }
}

__host__ __device__ double triArea(const tri_dbl s)
{
    rect_coord_dbl vec[2];
    vec[0] = rectCoordSub(s.nod[1],s.nod[0]);
    vec[1] = rectCoordSub(s.nod[2],s.nod[0]);
    return 0.5*rectNorm(rectCrossMul(vec[0],vec[1]));
}

__host__ __device__ double quadArea(const quad_dbl s)
{
    rect_coord_dbl vec[2];
    vec[0] = rectCoordSub(s.nod[1],s.nod[0]);
    vec[1] = rectCoordSub(s.nod[2],s.nod[0]);
    return rectNorm(rectCrossMul(vec[0],vec[1]));
}

__host__ __device__ plane_dbl quad2plane(const quad_dbl qd)
{
    /*get the plane containing a quad*/
    plane_dbl pln;
    pln.pt = qd.nod[0];
    rect_coord_dbl vec[2];
    vec[0] = rectCoordSub(qd.nod[1],qd.nod[0]);
    vec[1] = rectCoordSub(qd.nod[2],qd.nod[0]);
    pln.n = nrmlzRectCoord(rectCrossMul(vec[0],vec[1]));
    return pln;
}

__host__ __device__ line_dbl lnSeg2ln(const ln_seg_dbl ls)
{
    line_dbl l;
    l.pt = ls.nod[0];
    l.dir = rectCoordSub(ls.nod[1],ls.nod[0]);
    return l;
}

__host__ __device__ int deterPtQuadRel(const rect_coord_dbl pt, const quad_dbl qd)
{
    /*determine the relationship between a point and a quad on the same plane*/
    double area = 0.0;
    rect_coord_dbl vec[2];
    for(int i=0;i<4;i++) {
        vec[0] = rectCoordSub(qd.nod[i%4],pt);
        vec[1] = rectCoordSub(qd.nod[(i+1)%4],pt);
        area += 0.5*rectNorm(rectCrossMul(vec[0],vec[1]));
    }
    double area_quad = quadArea(qd);
    if(abs(area-area_quad)<EPS) {
        return 1; // in
    } else {
        return 0; // out
    }
}

__host__ __device__ double rectCoordDet(const rect_coord_dbl vec[3])
{
    double result, v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z;
    
    v1x = vec[0].coords[0];
    v1y = vec[0].coords[1];
    v1z = vec[0].coords[2];
    
    v2x = vec[1].coords[0];
    v2y = vec[1].coords[1];
    v2z = vec[1].coords[2];
    
    v3x = vec[2].coords[0];
    v3y = vec[2].coords[1];
    v3z = vec[2].coords[2];
    
    result = v1x*(v2y*v3z-v3y*v2z)-v2x*(v1y*v3z-v3y*v1z)+v3x*(v1y*v2z-v2y*v1z);
    
    return result;
}

__host__ __device__ int deterLnLnRel(const line_dbl ln1, const line_dbl ln2, double *t1, double *t2)
{   
    if(abs(rectNorm(rectCrossMul(ln1.dir,ln2.dir)))<EPS) {
        // the two lines are either parallel or the same line
        
        // check if a point on line 1 is on line 2
        rect_coord_dbl vec = rectCoordSub(ln1.pt,ln2.pt);
        if(rectNorm(vec)<EPS) {
            //the points are the same
            return 2; 
        } else {
            if(rectNorm(rectCrossMul(vec,ln2.dir))<EPS) {
                // vec is a multiple of ln2.dir
                return 2; 
            } else {
                // the two lines are parallel
                return 0;
            }
        }
    } else {
        //the two lines are not parallel or the same
        
        //take two different points on each line
        if(rectNorm(rectCoordSub(ln1.pt,ln2.pt))<EPS) {
            //the two points on the line is the same point
            *t1 = 0;
            *t2 = 0;
            return 1;
        } else {
            rect_coord_dbl pt[4], vec[3];
            pt[0] = ln1.pt;
            pt[1] = rectCoordAdd(ln1.pt,scaRectMul(1.0,ln1.dir));
            pt[2] = ln2.pt;
            pt[3] = rectCoordAdd(ln2.pt,scaRectMul(1.0,ln2.dir));
            vec[0] = rectCoordSub(pt[1],pt[0]);
            vec[1] = rectCoordSub(pt[2],pt[0]);
            vec[2] = rectCoordSub(pt[3],pt[0]);
            if(abs(rectCoordDet(vec))<EPS) {
                //skew lines
                return 0;
            } else {
                // the two lines intersects. compute it.
                // first find the valid sub-system
                double coeff1[2], coeff2[2];
                for(int i=0;i<3;i++) {
                    coeff1[0] = ln1.dir.coords[i%3];
                    coeff1[1] = ln1.dir.coords[(i+1)%3];
                    coeff2[0] = ln2.dir.coords[i%3];
                    coeff2[1] = ln2.dir.coords[(i+1)%3];
                    //check the determinant of the current system;
                    double det = coeff1[0]*coeff2[1]-coeff1[1]*coeff2[0];
                    if(abs(det)>EPS) {
                        // get the right-hand side
                        double rhs1[2], rhs2[2];
                        rhs1[0] = ln1.pt.coords[i%3];
                        rhs1[1] = ln1.pt.coords[(i+1)%3];
                        rhs2[0] = ln2.pt.coords[i%3];
                        rhs2[1] = ln2.pt.coords[(i+1)%3];
                        double rhs;
                        rhs = (rhs2[0]-rhs1[0])*coeff2[1]-(rhs2[1]-rhs1[1])*coeff2[0];
                        *t1 = rhs/det;
                        rhs = (rhs2[0]-rhs1[0])*coeff1[1]-(rhs2[1]-rhs1[1])*coeff1[0];
                        *t2 = rhs/det;
                        break;
                    }
                }
                return 1;
            }
        }
        
        
    }
}

__host__ __device__ int deterLnSegQuadRel(const ln_seg_dbl lnSeg, const quad_dbl qd)
{
    /*determine if a line segment intersects a quad*/
    int flag;
    
    //make a line containing the line segment
    rect_coord_dbl dir = rectCoordSub(lnSeg.nod[1],lnSeg.nod[0]);
    line_dbl ln;
    ln.pt = lnSeg.nod[0];
    ln.dir = dir;
    
    // define a plane containing the quad
    plane_dbl pln = quad2plane(qd);
    
    // determine the intersection between the line and the plane
    double t;
    flag = deterLinePlaneRel(ln,pln,&t);
    if(flag==0) {
        // no intersection
        return 0;
    } else {
        if(flag==2) {
            // infinitely many intersections
            if((deterPtQuadRel(lnSeg.nod[0],qd)==1) || (deterPtQuadRel(lnSeg.nod[1],qd)==1)) {
                //oen of the nodes is within the quad
                return 1;
            } else {
                // none of the nodes is within the quad, test if segments intersect
                
            }
        } else {
            //determines if a point is within a quad
            if(t<0 || t>1) {
                return 0;
            } else {
                rect_coord_dbl intersection = rectCoordAdd(ln.pt,scaRectMul(t,ln.dir));
                if(deterPtQuadRel(intersection,qd)==1) {
                    return 1;
                } else {
                    return  0;
                }
            }
            
        }
    }
}




__host__ __device__ int deterTriCubeInt(const rect_coord_dbl nod[3], const cube_dbl cb)
{
    /*this function determines if a triangle intersects with a cube*/
    return 1;
}
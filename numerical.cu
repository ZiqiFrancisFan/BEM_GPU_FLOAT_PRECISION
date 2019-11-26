/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
#include "numerical.h"
#include "mesh.h"

//air density and speed of sound
__constant__ float density = 1.2041;

__constant__ float speed = 343.21;

//Integral points and weights
__constant__ float INTPT[INTORDER]; 

__constant__ float INTWGT[INTORDER];

int genGaussParams(const int n, float *pt, float *wgt) 
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

int gaussPtsToDevice(const float *evalPt, const float *wgt) 
{
    CUDA_CALL(cudaMemcpyToSymbol(INTPT,evalPt,INTORDER*sizeof(float),0,cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpyToSymbol(INTWGT,wgt,INTORDER*sizeof(float),0,cudaMemcpyHostToDevice));
    return EXIT_SUCCESS;
}

void printFltMat(const float *A, const int numRow, const int numCol, const int lda) 
{
    for(int i=0;i<numRow;i++) {
        for(int j=0;j<numCol;j++) {
            printf("%f ",A[IDXC0(i,j,lda)]);
        }
        printf("\n");
    }
}

void printCuFloatComplexMat(const cuFloatComplex *A, const int numRow, const int numCol, 
        const int lda)
{
    for(int i=0;i<numRow;i++) {
        for(int j=0;j<numCol;j++) {
            printf("(%f,%f) ",cuCrealf(A[IDXC0(i,j,lda)]),cuCimagf(A[IDXC0(i,j,lda)]));
        }
        printf("\n");
    }
}

cart_coord_float cartCoordDouble2cartCoordFloat(const cart_coord_double t)
{
    cart_coord_float result;
    for(int i=0;i<3;i++) {
        result.coords[i] = t.coords[i];
    }
    return result;
}

cart_coord_double cartCoordFloat2cartCoordDouble(const cart_coord_float t)
{
    cart_coord_double result;
    for(int i=0;i<3;i++) {
        result.coords[i] = t.coords[i];
    }
    return result;
}

__host__ __device__ float dotProd(const cart_coord_float u, const cart_coord_float v) {
    return u.coords[0]*v.coords[0]+u.coords[1]*v.coords[1]+u.coords[2]*v.coords[2];
}

__host__ __device__ cart_coord_float crossProd(const cart_coord_float u, const cart_coord_float v) {
    cart_coord_float r;
    r.coords[0] = (u.coords[1])*(v.coords[2])-(u.coords[2])*(v.coords[1]);
    r.coords[1] = (u.coords[2])*(v.coords[0])-(u.coords[0])*(v.coords[2]);
    r.coords[2] = (u.coords[0])*(v.coords[1])-(u.coords[1])*(v.coords[0]);
    return r;
}

__host__ __device__ cart_coord_float cartCoordAdd(const cart_coord_float u, const cart_coord_float v)
{
    cart_coord_float result;
    for(int i=0;i<3;i++) {
        result.coords[i] = u.coords[i]+v.coords[i];
    }
    return result;
}

__host__ __device__ cart_coord_float cartCoordSub(const cart_coord_float u, const cart_coord_float v)
{
    cart_coord_float result;
    for(int i=0;i<3;i++) {
        result.coords[i] = u.coords[i]-v.coords[i];
    }
    return result;
}

__host__ __device__ cart_coord_float scalarProd(const float lambda, const cart_coord_float v)
{
    cart_coord_float result;
    for(int i=0;i<3;i++) {
        result.coords[i] = lambda*v.coords[i];
    }
    return result;
}

__host__ __device__ cart_coord_double cartCoordAdd(const cart_coord_double u, const cart_coord_double v)
{
    cart_coord_double result;
    for(int i=0;i<3;i++) {
        result.coords[i] = u.coords[i]+v.coords[i];
    }
    return result;
}

__host__ __device__ cart_coord_double cartCoordSub(const cart_coord_double u, const cart_coord_double v)
{
    cart_coord_double result;
    for(int i=0;i<3;i++) {
        result.coords[i] = u.coords[i]-v.coords[i];
    }
    return result;
}

__host__ __device__ cart_coord_double scalarProd(const double lambda, const cart_coord_double v)
{
    cart_coord_double result;
    for(int i=0;i<3;i++) {
        result.coords[i] = lambda*v.coords[i];
    }
    return result;
}

__host__ __device__ cart_coord_double triCentroid(cart_coord_double nod[3])
{
    cart_coord_double ctr_23 = scalarProd(0.5,cartCoordAdd(nod[1],nod[2]));
    cart_coord_double centroid = cartCoordAdd(nod[0],scalarProd(2.0/3.0,cartCoordSub(ctr_23,nod[0])));
    return centroid;
}

__host__ __device__ bool ray_intersect_triangle(const cart_coord_float O, const cart_coord_float dir, 
        const cart_coord_float nod[3])
{
    /*vert0 is chosen as reference point*/
    cart_coord_float E1, E2;
    E1 = cartCoordSub(nod[1],nod[0]);
    E2 = cartCoordSub(nod[2],nod[0]);
    /*cross product of dir and v0 to v1*/
    cart_coord_float P = crossProd(dir,E2);
    float det = dotProd(P,E1);
    if(abs(det)<EPS) {
        return false;
    }
    /*Computation of parameter u*/
    cart_coord_float T = cartCoordSub(O,nod[0]);
    float u = 1.0f/det*dotProd(P,T);
    if(u<0 || u>1) {
        return false;
    }
    /*Computation of parameter v*/
    cart_coord_float Q = crossProd(T,E1);
    float v = 1.0f/det*dotProd(Q,dir);
    if(v<0 || u+v>1) {
        return false;
    }
    /*Computation of parameter t*/
    float t = 1.0f/det*dotProd(Q,E2);
    if(t<EPS) {
        return false;
    }
    return true;
}

__global__ void rayTrisInt(const cart_coord_float pt_s, const cart_coord_float dir, const cart_coord_float *nod, 
        const tri_elem *elem, const int numElem, bool *flag)
{
    // decides if a point pnt is in a closed surface elem
    int idx = blockDim.x*blockIdx.x+threadIdx.x;
    if(idx<numElem) {
        cart_coord_float pt[3];
        for(int i=0;i<3;i++) {
            pt[i].coords[0] = nod[elem[idx].nodes[i]].coords[0];
            pt[i].coords[1] = nod[elem[idx].nodes[i]].coords[1];
            pt[i].coords[2] = nod[elem[idx].nodes[i]].coords[2];
        }
        flag[idx] = ray_intersect_triangle(pt_s,dir,pt);
    }
}

__global__ void distPntPnts(const cart_coord_float pt, const cart_coord_float *nod, const int numNod, float *dist) {
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

int genCHIEF(const cart_coord_float *pt, const int numPt, const tri_elem *elem, const int numElem, 
        cart_coord_float *pCHIEF, const int numCHIEF) {
    int i, cnt;
    float threshold_inner = 0.0000001;
    float *dist_h = (float*)malloc(numPt*sizeof(float));
    float minDist; //minimum distance between the chief point to all surface nodes
    float *dist_d;
    CUDA_CALL(cudaMalloc((void**)&dist_d, numPt*sizeof(float)));
    cart_coord_float dir; 
    
    //transfer the point cloud to GPU
    cart_coord_float *pt_d;
    CUDA_CALL(cudaMalloc((void**)&pt_d,numPt*sizeof(cart_coord_float))); //point cloud allocated on device
    CUDA_CALL(cudaMemcpy(pt_d,pt,numPt*sizeof(cart_coord_float),cudaMemcpyHostToDevice)); //point cloud copied to device
    
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
    cart_coord_float chief;
    
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

inline __device__ void crossNorm(const cart_coord_float a, const cart_coord_float b, cart_coord_float *norm, float *length) 
{
    cart_coord_float c;
    c.coords[0] = a.coords[1]*b.coords[2]-a.coords[2]*b.coords[1];
    c.coords[1] = a.coords[2]*b.coords[0]-a.coords[0]*b.coords[2];
    c.coords[2] = a.coords[0]*b.coords[1]-a.coords[1]*b.coords[0];

    *length = __fsqrt_rn((c.coords[0]*c.coords[0])+(c.coords[1]*c.coords[1])+(c.coords[2]*c.coords[2]));

    norm->coords[0] = c.coords[0] / *length;
    norm->coords[1] = c.coords[1] / *length;
    norm->coords[2] = c.coords[2] / *length;
}

__device__ void g_h_c_nsgl(const float k, const cart_coord_float x, const cart_coord_float p[3], 
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
    cart_coord_float y, normal, rVec;
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
            rVec = cartCoordSub(y,x);
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

__device__ void g_h_c_sgl(const float k, const cart_coord_float x_sgl1, const cart_coord_float x_sgl2, 
        const cart_coord_float x_sgl3, const cart_coord_float p[3], 
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
    cart_coord_float y_sgl1, y_sgl2, y_sgl3, normal, rVec;
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
            rVec = cartCoordSub(y_sgl1,x_sgl1);
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
            rVec = cartCoordSub(y_sgl2,x_sgl2);
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
            rVec = cartCoordSub(y_sgl3,x_sgl3);
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

__host__ __device__ cuFloatComplex ptSrc(const float k, const float amp, const cart_coord_float srcLoc, const cart_coord_float evalLoc)
{
    float fourPI = 4.0f*PI;
    cart_coord_float rVec = cartCoordSub(evalLoc,srcLoc);
    float radius = sqrtf(rVec.coords[0]*rVec.coords[0]+rVec.coords[1]*rVec.coords[1]+rVec.coords[2]*rVec.coords[2]);
    return make_cuFloatComplex(amp*cosf(-k*radius)/(fourPI*radius),amp*sinf(-k*radius)/(fourPI*radius));
}

__host__ __device__ cuFloatComplex dirSrc(const float k, const float strength, const cart_coord_float dir, const cart_coord_float evalLoc)
{
    float theta = -k*dotProd(dir,evalLoc);
    return make_cuFloatComplex(strength*cosf(theta),strength*sinf(theta));
}

// compute non-singular relationship between points and elements
__global__ void atomicPtsElems_nsgl(const float k, const cart_coord_float *pt, const int numNod, 
        const int idxPntStart, const int idxPntEnd, const tri_elem *elem, const int numElem, 
        cuFloatComplex *A, const int lda, cuFloatComplex *B, const int numSrc, const int ldb) {
    int xIdx = blockIdx.x*blockDim.x+threadIdx.x; //Index for points
    int yIdx = blockIdx.y*blockDim.y+threadIdx.y; //Index for elements
    //The thread with indices xIdx and yIdx process the point xIdx and elem yIdx
    if(xIdx>=idxPntStart && xIdx<=idxPntEnd && yIdx<numElem && xIdx!=elem[yIdx].nodes[0] 
            && xIdx!=elem[yIdx].nodes[1] && xIdx!=elem[yIdx].nodes[2]) {
        int i, j;
        cuFloatComplex hCoeff[3], gCoeff[3], bc, pCoeffs[3], temp;
        float cCoeff;
        cart_coord_float triNod[3];
        triNod[0] = pt[elem[yIdx].nodes[0]];
        triNod[1] = pt[elem[yIdx].nodes[1]];
        triNod[2] = pt[elem[yIdx].nodes[2]];
        g_h_c_nsgl(k,pt[xIdx],triNod,gCoeff,hCoeff,&cCoeff);
        
        //Update the A matrix
        bc = cuCdivf(elem[yIdx].bc[0],elem[yIdx].bc[1]);
        for(i=0;i<3;i++) {
            pCoeffs[i] = cuCsubf(hCoeff[i],cuCmulf(bc,gCoeff[i]));
        }
        
        for(i=0;i<3;i++) {
            //atomicFloatComplexAdd(&A[IDXC0(xIdx,elem[yIdx].nodes[i],lda)],pCoeffs[i]);
            atomicAdd(&A[IDXC0(xIdx,elem[yIdx].nodes[i],lda)].x,cuCrealf(pCoeffs[i]));
            atomicAdd(&A[IDXC0(xIdx,elem[yIdx].nodes[i],lda)].y,cuCimagf(pCoeffs[i]));
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

__global__ void atomicPtsElems_sgl(const float k, const cart_coord_float *pt, const tri_elem *elem, 
        const int numElem, cuFloatComplex *A, const int lda, cuFloatComplex *B, 
        const int numSrc, const int ldb) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx < numElem) {
        int i, j;
        cuFloatComplex hCoeff_sgl1[3], hCoeff_sgl2[3], hCoeff_sgl3[3], 
                gCoeff_sgl1[3], gCoeff_sgl2[3], gCoeff_sgl3[3], pCoeffs_sgl1[3], 
                pCoeffs_sgl2[3], pCoeffs_sgl3[3], bc, temp;
        float cCoeff_sgl1, cCoeff_sgl2, cCoeff_sgl3;
        
        cart_coord_float nod[3];
        for(i=0;i<3;i++) {
            nod[i] = pt[elem[idx].nodes[i]];
        }
        // Compute h and g coefficients
        g_h_c_sgl(k,pt[elem[idx].nodes[0]],pt[elem[idx].nodes[1]],pt[elem[idx].nodes[2]],
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
            //atomicFloatComplexAdd(&A[IDXC0(elem[idx].nodes[0],elem[idx].nodes[j],lda)],
            //        pCoeffs_sgl1[j]);
            atomicAdd(&A[IDXC0(elem[idx].nodes[0],elem[idx].nodes[j],lda)].x,
                    cuCrealf(pCoeffs_sgl1[j]));
            atomicAdd(&A[IDXC0(elem[idx].nodes[0],elem[idx].nodes[j],lda)].y,
                    cuCimagf(pCoeffs_sgl1[j]));
            //atomicFloatComplexAdd(&A[IDXC0(elem[idx].nodes[1],elem[idx].nodes[j],lda)],
            //        pCoeffs_sgl2[j]);
            atomicAdd(&A[IDXC0(elem[idx].nodes[1],elem[idx].nodes[j],lda)].x,
                    cuCrealf(pCoeffs_sgl2[j]));
            atomicAdd(&A[IDXC0(elem[idx].nodes[1],elem[idx].nodes[j],lda)].y,
                    cuCimagf(pCoeffs_sgl2[j]));
            //atomicFloatComplexAdd(&A[IDXC0(elem[idx].nodes[2],elem[idx].nodes[j],lda)],
            //        pCoeffs_sgl3[j]);
            atomicAdd(&A[IDXC0(elem[idx].nodes[2],elem[idx].nodes[j],lda)].x,
                    cuCrealf(pCoeffs_sgl3[j]));
            atomicAdd(&A[IDXC0(elem[idx].nodes[2],elem[idx].nodes[j],lda)].y,
                    cuCimagf(pCoeffs_sgl3[j]));
        }
        
        //atomicFloatComplexSub(&A[IDXC0(elem[idx].nodes[0],elem[idx].nodes[0],lda)],
        //        make_cuFloatComplex(cCoeff_sgl1,0));
        atomicAdd(&A[IDXC0(elem[idx].nodes[0],elem[idx].nodes[0],lda)].x,
                -cCoeff_sgl1);
        //atomicFloatComplexSub(&A[IDXC0(elem[idx].nodes[1],elem[idx].nodes[1],lda)],
        //        make_cuFloatComplex(cCoeff_sgl2,0));
        atomicAdd(&A[IDXC0(elem[idx].nodes[1],elem[idx].nodes[1],lda)].x,
                -cCoeff_sgl2);
        //atomicFloatComplexSub(&A[IDXC0(elem[idx].nodes[2],elem[idx].nodes[2],lda)],
        //        make_cuFloatComplex(cCoeff_sgl3,0));
        atomicAdd(&A[IDXC0(elem[idx].nodes[2],elem[idx].nodes[2],lda)].x,
                -cCoeff_sgl3);
        
        //Update matrix B using g Coefficients
        bc = cuCdivf(elem[idx].bc[2],elem[idx].bc[1]);
        for(i=0;i<numSrc;i++) {
            for(j=0;j<3;j++) {
                //atomicFloatComplexSub(&B[IDXC0(elem[idx].nodes[0],i,ldb)],
                //        cuCmulf(bc,gCoeff_sgl1[j]));
                temp = cuCmulf(bc,gCoeff_sgl1[j]);
                atomicAdd(&B[IDXC0(elem[idx].nodes[0],i,ldb)].x,-cuCrealf(temp));
                atomicAdd(&B[IDXC0(elem[idx].nodes[0],i,ldb)].y,-cuCimagf(temp));
                //atomicFloatComplexSub(&B[IDXC0(elem[idx].nodes[1],i,ldb)],
                //        cuCmulf(bc,gCoeff_sgl2[j]));
                temp = cuCmulf(bc,gCoeff_sgl2[j]);
                atomicAdd(&B[IDXC0(elem[idx].nodes[1],i,ldb)].x,-cuCrealf(temp));
                atomicAdd(&B[IDXC0(elem[idx].nodes[1],i,ldb)].y,-cuCimagf(temp));
                //atomicFloatComplexSub(&B[IDXC0(elem[idx].nodes[2],i,ldb)],
                //        cuCmulf(bc,gCoeff_sgl3[j]));
                temp = cuCmulf(bc,gCoeff_sgl3[j]);
                atomicAdd(&B[IDXC0(elem[idx].nodes[2],i,ldb)].x,-cuCrealf(temp));
                atomicAdd(&B[IDXC0(elem[idx].nodes[2],i,ldb)].y,-cuCimagf(temp));
            }
        }
    }
}

int atomicGenSystem(const float k, const tri_elem *elem, const int numElem, 
        const cart_coord_float *nod, const int numNod, const cart_coord_float *chief, const int numCHIEF, 
        const cart_coord_float *src, const int numSrc, cuFloatComplex *A, const int lda, 
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
    cart_coord_float *pt_h = (cart_coord_float*)malloc((numNod+numCHIEF)*sizeof(cart_coord_float));
    for(i=0;i<numNod;i++) {
        pt_h[i] = nod[i];
    }
    for(i=0;i<numCHIEF;i++) {
        pt_h[numNod+i] = chief[i];
    }
    
    cart_coord_float *pt_d;
    CUDA_CALL(cudaMalloc(&pt_d,(numNod+numCHIEF)*sizeof(cart_coord_float)));
    CUDA_CALL(cudaMemcpy(pt_d,pt_h,(numNod+numCHIEF)*sizeof(cart_coord_float),cudaMemcpyHostToDevice));
    
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
        const cart_coord_float *nod, const int numNod, const cart_coord_float *chief, const int numCHIEF, 
        const cart_coord_float *src, const int numSrc, cuFloatComplex *B, const int ldb)
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
    // cart_coord_float *pt_h = (cart_coord_float*)malloc((numNod+numCHIEF)*sizeof(cart_coord_float));
    // for(i=0;i<numNod;i++) {
    //     pt_h[i] = nod[i];
    // }
    // for(i=0;i<numCHIEF;i++) {
    //     pt_h[numNod+i] = chief[i];
    // }
    
    cart_coord_float *pt_d;
    CUDA_CALL(cudaMalloc(&pt_d, (numNod + numCHIEF) * sizeof(cart_coord_float)));
    CUDA_CALL(cudaMemcpy(pt_d, nod, numNod * sizeof(cart_coord_float),cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(pt_d + numNod, chief, numCHIEF * sizeof(cart_coord_float),cudaMemcpyHostToDevice));
    
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

int bemSolver_dir(const float k, const tri_elem *elem, const int numElem, 
        const cart_coord_float *nod, const int numNod, const cart_coord_float *chief, const int numCHIEF, 
        const cart_coord_float *dir, const int numSrc, cuFloatComplex *B, const int ldb)
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
    // cart_coord_float *pt_h = (cart_coord_float*)malloc((numNod+numCHIEF)*sizeof(cart_coord_float));
    // for(i=0;i<numNod;i++) {
    //     pt_h[i] = nod[i];
    // }
    // for(i=0;i<numCHIEF;i++) {
    //     pt_h[numNod+i] = chief[i];
    // }
    
    cart_coord_float *pt_d;
    CUDA_CALL(cudaMalloc(&pt_d,(numNod+numCHIEF)*sizeof(cart_coord_float)));
    CUDA_CALL(cudaMemcpy(pt_d,nod,numNod*sizeof(cart_coord_float),cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(pt_d+numNod,chief,numCHIEF*sizeof(cart_coord_float),cudaMemcpyHostToDevice));
    
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

__host__ __device__ sph_coord_float cart2sph(const cart_coord_float s)
{
    sph_coord_float temp;
    temp.coords[0] = sqrtf(powf(s.coords[0],2)+powf(s.coords[1],2)+powf(s.coords[2],2));
    temp.coords[1] = acosf(s.coords[2]/(temp.coords[0]));
    temp.coords[2] = atan2f(s.coords[1],s.coords[0]);
    return temp;
}

__host__ __device__ cart_coord_float sph2cart(const sph_coord_float s)
{
    float r = s.coords[0], theta = s.coords[1], phi = s.coords[2];
    float x = r*sinf(theta)*cosf(phi), y = r*sinf(theta)*sinf(phi), z = r*cosf(theta);
    cart_coord_float result;
    result.coords[0] = x;
    result.coords[1] = y;
    result.coords[2] = z;
    return result;
}

__host__ __device__ sph_coord_double cart2sph(const cart_coord_double s)
{
    sph_coord_double temp;
    temp.coords[0] = sqrt(pow(s.coords[0],2)+pow(s.coords[1],2)+pow(s.coords[2],2));
    temp.coords[1] = acos(s.coords[2]/(temp.coords[0]));
    temp.coords[2] = atan2(s.coords[1],s.coords[0]);
    return temp;
}

__host__ __device__ cart_coord_double sph2cart(const sph_coord_double s)
{
    double r = s.coords[0], theta = s.coords[1], phi = s.coords[2];
    double x = r*sin(theta)*cos(phi), y = r*sin(theta)*sin(phi), z = r*cos(theta);
    cart_coord_double result;
    result.coords[0] = x;
    result.coords[1] = y;
    result.coords[2] = z;
    return result;
}

void rigidSpherePlaneMultipleEval(const cart_coord_float *pt, const int numPt, 
        const double a, const double wavNum, const double strength)
{
    gsl_complex *p = (gsl_complex*)malloc(numPt*sizeof(gsl_complex));
    sph_coord_float tempCoord;
    gsl_complex result;
    //double temp;
    //const int truncNum = 30;
    for(int i=0;i<numPt;i++)
    {
        tempCoord = cart2sph(pt[i]);
        result = rigid_sphere_plane(wavNum,strength,a,tempCoord.coords[0],tempCoord.coords[1]);
        p[i] = result;
        printf("(%.8f,%.8f)\n",GSL_REAL(p[i]),GSL_IMAG(p[i]));
    }
    free(p);
}

gsl_complex rigid_sphere_plane(const double wavNum, const double strength, const double a, 
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

gsl_complex rigid_sphere_point(const double wavNum, const double strength, const double rs, 
        const double a, const cart_coord_double y)
{
    const int truncNum = 100;
    const cart_coord_double src = {0,0,rs};
    cart_coord_double temp_cart_coord = cartCoordSub(y,src);
    sph_coord_double temp_sph_coord = cart2sph(temp_cart_coord);
    double R = temp_sph_coord.coords[0];
    temp_sph_coord = cart2sph(y);
    double r = temp_sph_coord.coords[0];
    double theta = temp_sph_coord.coords[1];
    gsl_complex result = gsl_complex_rect(strength*cos(wavNum*R)/(4*PI*R),strength*sin(wavNum*R)/(4*PI*R));
    for(int n=0;n<truncNum;n++) {
        gsl_complex temp[2];
        double t = (n+0.5)*jprime(n,wavNum*a)*wavNum*strength/(2*PI)*gsl_sf_legendre_Pl(n,cos(theta));
        temp[0] = gsl_complex_rect(0,t);
        temp[1] = gsl_complex_mul(gsl_sf_bessel_hl(n,wavNum*rs),gsl_sf_bessel_hl(n,wavNum*r));
        temp[0] = gsl_complex_mul(temp[0],temp[1]);
        temp[0] = gsl_complex_div(temp[0],hprime(n,wavNum*a));
        result = gsl_complex_sub(result,temp[0]);
    }
    return result;
}

__device__ cuFloatComplex extrapolation_dir(const float wavNum, const cart_coord_float x, 
        const tri_elem* elem, const int numElem, const cart_coord_float* pt, 
        const cuFloatComplex* p, const float strength, const cart_coord_float dir)
{
    cuFloatComplex result = dirSrc(wavNum,strength,dir,x);
    cuFloatComplex temp;
    for(int i=0;i<numElem;i++) {
        cart_coord_float nod[3];
        for(int j=0;j<3;j++) {
            nod[j] = pt[elem[i].nodes[j]];
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
            temp = cuCmulf(temp,p[elem[i].nodes[j]]);
            result = cuCsubf(result,temp);
        }
    }
    return result;
}

__global__ void extrapolation_dirs(const float wavNum, const cart_coord_float* expPt, const int numExpPt,
        const tri_elem* elem, const int numElem, const cart_coord_float* pt, const cuFloatComplex* p, 
        const float strength, const cart_coord_float dir, cuFloatComplex *p_exp)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx < numExpPt) {
        p_exp[idx] = extrapolation_dir(wavNum,expPt[idx],elem,numElem,pt,p,strength,dir);
    }
}

int extrapolation_dirs_single_source(const float wavNum, const cart_coord_float* expPt, const int numExpPt, 
        const tri_elem* elem, const int numElem, const cart_coord_float* pt, const int numPt, 
        const cuFloatComplex* p, const float strength, const cart_coord_float dir, cuFloatComplex *pExp)
{
    int width = 16, numBlock = (numExpPt+width-1)/width;
    
    // allocate memory on GPU and copy data to GPU memory
    cart_coord_float *expPt_d, *pt_d;
    tri_elem *elem_d;
    cuFloatComplex *p_d, *pExp_d;
    
    CUDA_CALL(cudaMalloc(&expPt_d,numExpPt*sizeof(cart_coord_float)));
    CUDA_CALL(cudaMemcpy(expPt_d,expPt,numExpPt*sizeof(cart_coord_float),cudaMemcpyHostToDevice));
    
    CUDA_CALL(cudaMalloc(&pt_d,numPt*sizeof(cart_coord_float)));
    CUDA_CALL(cudaMemcpy(pt_d,pt,numPt*sizeof(cart_coord_float),cudaMemcpyHostToDevice));
    
    CUDA_CALL(cudaMalloc(&elem_d,numElem*sizeof(tri_elem)));
    CUDA_CALL(cudaMemcpy(elem_d,elem,numElem*sizeof(tri_elem),cudaMemcpyHostToDevice));
    
    CUDA_CALL(cudaMalloc(&p_d,numPt*sizeof(cuFloatComplex)));
    CUDA_CALL(cudaMemcpy(p_d,p,numPt*sizeof(cuFloatComplex),cudaMemcpyHostToDevice));
    
    CUDA_CALL(cudaMalloc(&pExp_d,numExpPt*sizeof(cuFloatComplex)));
    
    extrapolation_dirs<<<numBlock,width>>>(wavNum,expPt_d,numExpPt,elem_d,numElem,pt_d,p_d,
            strength,dir,pExp_d);
    
    CUDA_CALL(cudaMemcpy(pExp,pExp_d,numExpPt*sizeof(cuFloatComplex),cudaMemcpyDeviceToHost));
    
    CUDA_CALL(cudaFree(expPt_d));
    CUDA_CALL(cudaFree(pt_d));
    CUDA_CALL(cudaFree(elem_d));
    CUDA_CALL(cudaFree(p_d));
    CUDA_CALL(cudaFree(pExp_d));
    
    return EXIT_SUCCESS;
}
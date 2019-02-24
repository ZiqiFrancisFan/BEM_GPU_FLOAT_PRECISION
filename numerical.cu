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

void printFltMat(const float *A, const int numRow, const int numCol, const int lda) 
{
    for(int i=0;i<numRow;i++) {
        for(int j=0;j<numCol;j++) {
            printf("%f ",A[IDXC0(i,j,lda)]);
        }
        printf("\n");
    }
}

__host__ __device__ float dotProd(const cartCoord u, const cartCoord v) {
    return u.coords[0]*v.coords[0]+u.coords[1]*v.coords[1]+u.coords[2]*v.coords[2];
}

__host__ __device__ cartCoord crossProd(const cartCoord u, const cartCoord v) {
    cartCoord r;
    r.coords[0] = (u.coords[1])*(v.coords[2])-(u.coords[2])*(v.coords[1]);
    r.coords[1] = (u.coords[2])*(v.coords[0])-(u.coords[0])*(v.coords[2]);
    r.coords[2] = (u.coords[0])*(v.coords[1])-(u.coords[1])*(v.coords[0]);
    return r;
}

__host__ __device__ cartCoord cartCoordAdd(const cartCoord u, const cartCoord v)
{
    cartCoord result;
    for(int i=0;i<3;i++) {
        result.coords[i] = u.coords[i]+v.coords[i];
    }
    return result;
}

__host__ __device__ cartCoord cartCoordSub(const cartCoord u, const cartCoord v)
{
    cartCoord result;
    for(int i=0;i<3;i++) {
        result.coords[i] = u.coords[i]-v.coords[i];
    }
    return result;
}

__host__ __device__ cartCoord scalarProd(const float lambda, const cartCoord v)
{
    cartCoord result;
    for(int i=0;i<3;i++) {
        result.coords[i] = lambda*v.coords[i];
    }
    return result;
}

__host__ __device__ bool ray_intersect_triangle(const cartCoord O, const cartCoord dir, 
        const cartCoord nod[3])
{
	/*vert0 is chosen as reference point*/
	cartCoord E1, E2;
        E1 = cartCoordSub(nod[1],nod[0]);
        E2 = cartCoordSub(nod[2],nod[0]);
	/*cross product of dir and v0 to v1*/
	cartCoord P = crossProd(dir,E2);
	float det = dotProd(P,E1);
	if (abs(det)<EPS) {
            return false;
	}
	/*Computation of parameter u*/
	cartCoord T = cartCoordSub(O,nod[0]);
	float u = 1.0f/det*dotProd(P,T);
	if (u < 0 || u>1) {
            return false;
	}
	/*Computation of parameter v*/
	cartCoord Q = crossProd(T,E1);
	float v = 1.0f/det*dotProd(Q,dir);
	if (v<0 || u+v>1) {
            return false;
	}
	/*Computation of parameter t*/
	float t = 1.0f/det*dotProd(Q,E2);
	if (t < EPS) {
            return false;
	}
	return true;
}

__global__ void rayTrisInt(const cartCoord pt_s, const cartCoord dir, const cartCoord *nod, 
        const triElem *elem, const int numElem, bool *flag)
{
    // decides if a point pnt is in a closed surface elems
    int idx = blockDim.x*blockIdx.x+threadIdx.x;
    if(idx<numElem) {
        cartCoord pt[3];
        for(int i=0;i<3;i++) {
            pt[i] = nod[elem[idx].nodes[i]];
        }
        flag[idx] = ray_intersect_triangle(pt_s,dir,pt);
    }
}

__global__ void distPntPnts(const cartCoord pt, const cartCoord *nod, const int numNod, float *dist) {
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

int genCHIEF(const cartCoord *pt, const int numPt, const triElem *elem, const int numElem, 
        cartCoord *pCHIEF, const int numCHIEF) {
    int i, cnt;
    float threshold_inner = 0.001;
    float *dist_h = (float*)malloc(numPt*sizeof(float));
    float minDist; //minimum distance between the chief point to all surface nodes
    float *dist_d;
    CUDA_CALL(cudaMalloc((void**)&dist_d, numPt*sizeof(float)));
    cartCoord dir; 
    
    //transfer the point cloud to GPU
    cartCoord *pt_d;
    CUDA_CALL(cudaMalloc((void**)&pt_d,numPt*sizeof(cartCoord))); //point cloud allocated on device
    CUDA_CALL(cudaMemcpy(pt_d,pt,numPt*sizeof(cartCoord),cudaMemcpyHostToDevice)); //point cloud copied to device
    
    //transfer the element cloud to GPU
    triElem *elem_d;
    CUDA_CALL(cudaMalloc((void**)&elem_d,numElem*sizeof(triElem))); //elements allcoated on device
    CUDA_CALL(cudaMemcpy(elem_d,elem,numElem*sizeof(triElem),cudaMemcpyHostToDevice)); //elements copied to device
    
    //create a flag array on CPU and on GPU
    bool *flag_h = (bool*)malloc(numElem*sizeof(bool));
    bool *flag_d;
    CUDA_CALL(cudaMalloc((void**)&flag_d,numElem*sizeof(bool))); //memory for flags allocated on device

    unsigned long long seed = 0;
    int blockWidth = 32;
    int gridWidth;
    float xrand, yrand, zrand, unifRandNum[3];
    cartCoord chief;
    
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
            printCartCoord(&chief,1);
            gridWidth = (numElem+blockWidth-1)/blockWidth;
            rayTrisInt<<<gridWidth,blockWidth>>>(chief,dir,pt,elem,numElem,flag_d);
            gridWidth = (numPt+blockWidth-1)/blockWidth;
            distPntPnts<<<gridWidth,blockWidth>>>(chief,pt,numPt,dist_d);
            CUDA_CALL(cudaMemcpy(dist_h,dist_d,numPt*sizeof(float),cudaMemcpyDeviceToHost));
            CUDA_CALL(cudaMemcpy(flag_h,flag_d,numElem*sizeof(bool),cudaMemcpyDeviceToHost));
            minDist = dist_h[0];
            for(i=1;i<numPt;i++) {
                if(dist_h[i]<minDist) {
                    minDist = dist_h[i];
                }
            }
            //printf("The minimum distance is %f, threshold is %f\n",dist_min,threshold_inner);
            //printf("inSurf: %d\n", inSurf(flags_h, numElems));
        } while (!inBdry(flag_h,numElem) || minDist<threshold_inner);
        for(int j=0;j<3;j++) {
            pCHIEF[cnt].coords[i] = chief.coords[i];
        }
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

inline __device__ void crossNorm(const cartCoord a, const cartCoord b, cartCoord *norm, float *length) 
{
    cartCoord c;
    c.coords[0] = a.coords[1]*b.coords[2]-a.coords[2]*b.coords[1];
    c.coords[1] = a.coords[2]*b.coords[0]-a.coords[0]*b.coords[2];
    c.coords[2] = a.coords[0]*b.coords[1]-a.coords[1]*b.coords[0];

    *length = __fsqrt_rn((c.coords[0]*c.coords[0])+(c.coords[1]*c.coords[1])+(c.coords[2]*c.coords[2]));

    norm->coords[0] = c.coords[0] / *length;
    norm->coords[1] = c.coords[1] / *length;
    norm->coords[2] = c.coords[2] / *length;
}

__device__ void g_h_c_nsgl(const float k, const cartCoord x, const cartCoord p[3], 
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
    cartCoord y, normal, rVec;
    cuFloatComplex Psi, pPsipn2;
    crossNorm(
    {
        p[0].coords[0]-p[2].coords[0],p[0].coords[1]-p[2].coords[1],p[0].coords[2]-p[2].coords[2]
    },
    {
        p[1].coords[0]-p[2].coords[0],p[1].coords[1]-p[2].coords[1],p[1].coords[2] - p[2].coords[2]
    },
            &normal,&vertCrossProd);
    vertCrossProd = vertCrossProd*0.25f;
    //printf("vert: %f\n",vertCrossProd);
    
    //printf("normal=(%f,%f,%f)\n",normal.coords[0],normal.coords[1],normal.coords[2]);
    const float prodRhoOmega = density*omega;
    const float fourPI = 4.0f*PI;
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
            radius = sqrtf(rVec.coords[0]*rVec.coords[0]+rVec.coords[1]*rVec.coords[1]+rVec.coords[2]*rVec.coords[2]);
            //printf("radius = %f\n",radius);
            prpn2 = ((y.coords[0]-x.coords[0])*normal.coords[0]+(y.coords[1]-x.coords[1])*normal.coords[1]
                    +(y.coords[2]-x.coords[2])*normal.coords[2])/radius;
            //printf("prpn2=%f\n",prpn2);
            pPsiLpn2 = -1.0f/fourPI/(radius*radius)*prpn2;
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

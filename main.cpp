/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <getopt.h>
#include <math.h>
#include <time.h>
#include "dataStructs.h"
#include "mesh.h"
#include "octree.h"
#include "numerical.h"
#include "geometry.h"
#include <float.h>

extern vec3d bases[3];

int main(int argc, char *argv[])
{
    
    SetHostBases();
    HOST_CALL(CopyBasesToConstant());
    printf("Set bases.\n");
    /*
    int numPt, numElem;
    findNum("./mesh/sphere_100mm_5120.obj",&numPt,&numElem);
    rect_coord_dbl *pt = (rect_coord_dbl*)malloc(numPt*sizeof(rect_coord_dbl));
    tri_elem *elem = (tri_elem*)malloc(numElem*sizeof(tri_elem));
    readOBJ("./mesh/sphere_100mm_5120.obj",pt,elem);
    
    aarect3d sp;
    sp.cnr = {-0.5,-0.5,-0.5};
    sp.len[0] = 1;
    sp.len[1] = 1;
    sp.len[2] = 1;
    double len = 0.01;
    int voxNum[3];
    for(int i=0;i<3;i++) {
        voxNum[i] = floor(sp.len[i]/len);
    }
    int totNumVox = voxNum[0]*voxNum[1]*voxNum[2];
    bool *flag = (bool*)malloc(totNumVox*sizeof(bool));
    
    HOST_CALL(SpaceVoxelOnGPU(sp,len,pt,elem,numElem,flag));
    write_voxels(flag,voxNum,"./data/vox");
    
    free(flag);
    free(elem);
    free(pt);
    CUDA_CALL(cudaDeviceReset());
    return EXIT_SUCCESS;
     */
    printf("started main.\n");
    int numPt, numElem;
    findNum("./mesh/sphere_100mm_5120.obj",&numPt,&numElem);
    vec3d *pt = (vec3d*)malloc(numPt*sizeof(vec3d));
    tri_elem *elem = (tri_elem*)malloc(numElem*sizeof(tri_elem));
    readOBJ("./mesh/sphere_100mm_5120.obj",pt,elem);
    
    printf("read objects, %d elements, %d points\n",numElem,numPt);
    
    vec3f *pt_f = (vec3f*)malloc(numPt*sizeof(vec3f));
    vecd2f(pt,numPt,pt_f);
    
    
    vec3f *chief = (vec3f*)malloc(NUMCHIEF*sizeof(vec3f));
    HOST_CALL(genCHIEF(pt_f,numPt,elem,numElem,chief,NUMCHIEF));
    printf("chief generated.\n");
    
    float f = 200;
    float wavNum = 2*PI*f/SPEED_SOUND;
    
    double rs = 0.5, a = 0.1;
    vec3d ev_pt = {1.2,1.2,0.1};
    vec3f ev_pt_f = {1.2,1.2,0.1};
    gsl_complex prs = rigid_sphere_monopole(wavNum,STRENGTH,rs,a,ev_pt);
    
    vec3f src = {0,0,(float)rs};
    cuFloatComplex p;
    cuFloatComplex *B = (cuFloatComplex*)malloc((numPt+NUMCHIEF)*sizeof(cuFloatComplex));
    HOST_CALL(bemSolver_mp(wavNum,elem,numElem,pt_f,numPt,chief,NUMCHIEF,&src,1,B,numPt+NUMCHIEF));
    HOST_CALL(field_extrapolation_single_mp(wavNum,&ev_pt_f,1,elem,numElem,pt_f,numPt,B,STRENGTH,{0,0,(float)rs},&p));
    printf("Analytical: (%f,%f), BEM: (%f,%f)\n",GSL_REAL(prs),GSL_IMAG(prs),cuCrealf(p),cuCimagf(p));
    
    free(chief);
    free(pt_f);
    free(elem);
    free(pt);
    free(B);
    CUDA_CALL(cudaDeviceReset());
    return EXIT_SUCCESS;
}

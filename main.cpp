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
    int numPt, numElem;
    findNum("./mesh/sphere_100mm_5120.obj",&numPt,&numElem);
    rect_coord_dbl *pt = (rect_coord_dbl*)malloc(numPt*sizeof(rect_coord_dbl));
    tri_elem *elem = (tri_elem*)malloc(numElem*sizeof(tri_elem));
    readOBJ("./mesh/sphere_100mm_5120.obj",pt,elem);
    
    double len = 0.0025;
    aarect3d sp;
    sp.cnr = {-0.2,-0.2,-0.2};
    sp.len[0] = 0.4;
    sp.len[1] = 0.4;
    sp.len[2] = 0.4;
    HOST_CALL(RectSpaceVoxelOnGPU(sp,len,pt,elem,numElem,"./data/vox"));
    
    free(elem);
    free(pt);
    CUDA_CALL(cudaDeviceReset());
    return EXIT_SUCCESS;
}

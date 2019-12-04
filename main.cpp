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

int main(int argc, char *argv[])
{
    int numPt, numElem;
    findNum("./mesh/sphere_100mm_320.obj",&numPt,&numElem);
    rect_coord_dbl *pt = (rect_coord_dbl*)malloc(numPt*sizeof(rect_coord_dbl));
    tri_elem *elem = (tri_elem*)malloc(numElem*sizeof(tri_elem));
    readOBJ("./mesh/sphere_100mm_320.obj",pt,elem);
    
    int numEachDim = 64;
    int *flag = (int*)malloc(numEachDim*numEachDim*numEachDim*sizeof(int));
    aacb3d sp;
    sp.cnr = {-0.2,-0.2,-0.2};
    sp.len = 0.4;
    HOST_CALL(voxelSpace(sp,numEachDim,pt,elem,numElem,flag));
    write_voxels(flag,numEachDim,"./data/vox");
    
    free(flag);
    free(elem);
    free(pt);
    return EXIT_SUCCESS;
}

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
    tri3d tri;
    tri.nod[0] = {0.1,0,0};
    tri.nod[1] = {0,-0.1,0};
    tri.nod[2] = {0,0,-0.1};
    aacb3d cb;
    cb.cnr = {0,0,0};
    cb.len = 1.0;
    int rel = DeterTriCubeVtxPlaneRel(tri,cb);
    if(rel==0) {
        printf("separated\n");
    }
    else {
        printf("not separated\n");
    }
    CUDA_CALL(cudaDeviceReset());
    return EXIT_SUCCESS;
}

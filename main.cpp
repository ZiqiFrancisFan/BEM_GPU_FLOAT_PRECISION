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
    tri.nod[0] = {6.001,0,0};
    tri.nod[1] = {0,6,0};
    tri.nod[2] = {0,0,6};
    
    aarect3d rect;
    rect.cnr = {0,0,0};
    rect.len[0] = 2;
    rect.len[1] = 2;
    rect.len[2] = 2;
    
    bool rel = OverlapTriangleAARect(tri,rect);
    if(rel) {
        printf("They overlap.\n");
    }
    else {
        printf("They don't overlap.\n");
    }
    CUDA_CALL(cudaDeviceReset());
    return EXIT_SUCCESS;
}

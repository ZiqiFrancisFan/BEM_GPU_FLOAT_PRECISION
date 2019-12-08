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
    vec3d pt = {0.5,-1,-1};
    aacb3d cb;
    cb.cnr = {0,0,0};
    cb.len = 1.0;
    int a[2] = {0,0};
    int rel = DeterPtEdgePlaneRel(pt,cb,0,a);
    if(rel==1) {
        printf("inside\n");
    }
    else {
        printf("outside\n");
    }
    CUDA_CALL(cudaDeviceReset());
    return EXIT_SUCCESS;
}

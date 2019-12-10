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
    vec3f src_loc = {0,-2,0};
    float mag = 1.0;
    aarect3d rect;
    rect.cnr = {-2.5,-2.5,-2.5};
    rect.len[0] = 5;
    rect.len[1] = 5;
    rect.len[2] = 5;
    HOST_CALL(GenerateVoxelField("./mesh/sphere_100mm_5120.obj",2*PI*1000/SPEED_SOUND,
            &src_loc,&mag,1,rect,0.05,"./data/vox","./data/field"));
    CUDA_CALL(cudaDeviceReset());
    return EXIT_SUCCESS;
}

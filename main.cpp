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
    tri.nod[0] = {3,0,0};
    tri.nod[1] = {0,3,0};
    tri.nod[2] = {0,0,3};
    vec3d ax = nrmlzVec(vecAdd(vecAdd(vecAdd({0,0,0},bases[0]),bases[1]),bases[2]));
    intvl3d intvl = GetInterval(tri,ax);
    printf("interval: [%lf,%lf]\n",intvl.min,intvl.max);
    CUDA_CALL(cudaDeviceReset());
    return EXIT_SUCCESS;
}

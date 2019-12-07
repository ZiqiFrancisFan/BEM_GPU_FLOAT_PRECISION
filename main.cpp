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
    double mat[9] = {0,2,3,4,5,6,7,1,0};
    printMat(mat,3,3,3);
    printf("\n");
    
    GaussElim(mat,3,3,3);
    printMat(mat,3,3,3);
    CUDA_CALL(cudaDeviceReset());
    return EXIT_SUCCESS;
}

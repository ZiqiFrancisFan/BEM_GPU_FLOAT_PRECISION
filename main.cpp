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

int main(int argc, char *argv[])
{
    time_t start, end;
    int numNod, numElem;
    findNum("./mesh/sphere_100mm_5120.obj",&numNod,&numElem);
    cart_coord_double *nod_dp = (cart_coord_double*)malloc(numNod*sizeof(cart_coord_double));
    cart_coord_double *nod_sc = (cart_coord_double*)malloc(numNod*sizeof(cart_coord_double));
    cart_coord_float *nod_fp = (cart_coord_float*)malloc(numNod*sizeof(cart_coord_float));
    tri_elem *elem = (tri_elem*)malloc(numElem*sizeof(tri_elem));
    readOBJ("./mesh/sphere_100mm_5120.obj",nod_dp,elem);
    readOBJ("./mesh/sphere_100mm_5120.obj",nod_fp,elem);
    cart_coord_double cnr = {-0.5,-0.5,-0.5};
    double sideLength = 1.0;
    scalePnts(nod_dp,numNod,cnr,sideLength,nod_sc);
    int l = deterLmax(nod_sc,numNod,1);
    printf("The level to be used: %d\n",l);
    cart_coord_float src = {-1.0,0,0};
    int numSrc = 1, numBox = pow(8,l);
    float freq = 500, wavNum = 2*PI*freq/SPEED_SOUND;
    cuFloatComplex *fields = (cuFloatComplex*)malloc(numSrc*numBox*sizeof(cuFloatComplex));
    int *grid = (int*)malloc(numBox*sizeof(int));
    printf("grid allocated.\n");
    time(&start);
    HOST_CALL(genFields_MultiPtSrcSglObj(STRENGTH,wavNum,&src,numSrc,nod_dp,numNod,elem,numElem,cnr,sideLength,l,fields));
    createMeshOccupancyGrid(nod_dp,numNod,elem,numElem,cnr,sideLength,l,grid);
    time(&end);
    double duration = double(end-start);
    printf("Duration for field extrapolation and generation of occupancy grid: %lf seconds\n",duration);
    //print_cuFloatComplex_mat(fields,1,10,1);
    
    free(nod_dp);
    free(nod_sc);
    free(nod_fp);
    free(elem);
    free(fields);
    free(grid);
    return EXIT_SUCCESS;
}

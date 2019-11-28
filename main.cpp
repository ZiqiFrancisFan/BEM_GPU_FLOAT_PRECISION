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
#include "dataStructs.h"
#include "mesh.h"
#include "octree.h"
#include "numerical.h"

int main(int argc, char *argv[])
{
    // test octree
    int numPts, numElems;
    findNum("sphere_500mm.obj",&numPts,&numElems);
    cart_coord_double *pts = (cart_coord_double*)malloc(numPts*sizeof(cart_coord_double));
    tri_elem *elems = (tri_elem*)malloc(numElems*sizeof(tri_elem));
    readOBJ("sphere_500mm.obj",pts,elems);
    
    float tempIntPts[3], tempIntWgts[3];
    genGaussParams(3,tempIntPts,tempIntWgts);
    gaussPtsToDevice(tempIntPts,tempIntWgts);
    for(int i=0;i<3;i++) {
        tempIntPts[i] = 0;
        tempIntWgts[i] = 0;
    }
    
    CUDA_CALL(cudaMemcpyFromSymbol(tempIntPts,INTPT,3*sizeof(float),0,cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpyFromSymbol(tempIntWgts,INTWGT,3*sizeof(float),0,cudaMemcpyDeviceToHost));
    
    for(int i=0;i<3;i++) {
        printf("(%f,%f)\n",tempIntPts[i],tempIntWgts[i]);
    }
    free(pts);
    free(elems);
}

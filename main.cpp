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
    
    float tempIntPts[INTORDER], tempIntWgts[INTORDER];
    cuGenGaussParams(INTORDER,tempIntPts,tempIntWgts);
    gaussPtsToDevice(tempIntPts,tempIntWgts);
    for(int i=0;i<INTORDER;i++) {
        tempIntPts[i] = 0;
        tempIntWgts[i] = 0;
    }
    
    CUDA_CALL(cudaMemcpyFromSymbol(tempIntPts,INTPT,INTORDER*sizeof(float),0,cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpyFromSymbol(tempIntWgts,INTWGT,INTORDER*sizeof(float),0,cudaMemcpyDeviceToHost));
    
    for(int i=0;i<INTORDER;i++) {
        printf("(%f,%f)\n",tempIntPts[i],tempIntWgts[i]);
    }
}

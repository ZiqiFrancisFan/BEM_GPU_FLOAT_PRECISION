/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "dataStruct.h"
#include "mesh.h"
#include "numerical.h"
int main(int argc, char *argv[]) {
    printf("Number of command line arguments: %d\n",argc);
    if(argc <= 1)
    {
        printf("Mesh file not provided!\n");
        return EXIT_FAILURE;
    }
    float freq = 1000;
    float c = 343.21;
    float k = 2*PI*freq/c;
    
    float intPt[INTORDER];
    float intWgt[INTORDER];
    HOST_CALL(genGaussParams(INTORDER,intPt,intWgt));
    HOST_CALL(gaussPtsToDevice(intPt,intWgt));
    int numPt, numElem;
    //cartCoord src = {10,10,10};
    cartCoord dir = {0,0,1};
    findNum(argv[1],&numPt,&numElem);
    cartCoord *pt = (cartCoord*)malloc(numPt*sizeof(cartCoord));
    triElem *elem = (triElem*)malloc(numElem*sizeof(triElem));
    readOBJ(argv[1],pt,elem);
    cartCoord chief[NUMCHIEF];
    genCHIEF(pt,numPt,elem,numElem,chief,NUMCHIEF);
    //printCartCoord(chief,NUMCHIEF);
    //printf("Completed.\n");
    
    cuFloatComplex *B = (cuFloatComplex*)malloc((numPt+NUMCHIEF)*sizeof(cuFloatComplex));
    //HOST_CALL(bemSolver(k,elem,numElem,pt,numPt,chief,NUMCHIEF,&src,1,B,numPt+NUMCHIEF));
    HOST_CALL(bemSolver_dir(k,elem,numElem,pt,numPt,chief,NUMCHIEF,&dir,1,B,numPt+NUMCHIEF));
    printCuFloatComplexMat(B,numPt,1,numPt+NUMCHIEF);
    printf("Analytical solution: \n");
    computeRigidSphereScattering(pt,numPt,0.1,k,1.0);
    
    free(pt);
    free(elem);
    free(B);
}

int parse_command(int argc, char *argv[])
{   
    char filename[50], src_type[50];
    int left_index, right_index;
    float low_freq, high_freq;
    if(argc != 7)
    {
        printf("Input arguments not complete");
        return EXIT_FAILURE;
    } else
    {
        for(int i=0;i<=7;i++)
        {
            
        }
    }
}

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
#include <stdio.h>
#include <stdlib.h>
#include "dataStruct.h"
#include "mesh.h"
#include "numerical.h"
int main () {
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
    findNum("sphere_100mm.obj",&numPt,&numElem);
    cartCoord *pt = (cartCoord*)malloc(numPt*sizeof(cartCoord));
    triElem *elem = (triElem*)malloc(numElem*sizeof(triElem));
    readOBJ("sphere_100mm.obj",pt,elem);
    cartCoord chief[NUMCHIEF];
    genCHIEF(pt,numPt,elem,numElem,chief,NUMCHIEF);
    printCartCoord(chief,NUMCHIEF);
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

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
    float freq = 171.5;
    float c = 343.21;
    float k = 2*PI*freq/c;
    
    float intPt[INTORDER];
    float intWgt[INTORDER];
    HOST_CALL(genGaussParams(INTORDER,intPt,intWgt));
    HOST_CALL(gaussPtsToDevice(intPt,intWgt));
    int numPt, numElem;
    cartCoord src = {10,10,10};
    findNum("sphere1.obj",&numPt,&numElem);
    cartCoord *pt = (cartCoord*)malloc(numPt*sizeof(cartCoord));
    triElem *elem = (triElem*)malloc(numElem*sizeof(triElem));
    readOBJ("sphere1.obj",pt,elem);
    cartCoord chief[NUMCHIEF];
    genCHIEF(pt,numPt,elem,numElem,chief,NUMCHIEF);
    printCartCoord(chief,NUMCHIEF);
    //printf("Completed.\n");
    cuFloatComplex *A = (cuFloatComplex*)malloc((numPt+NUMCHIEF)*numPt*sizeof(cuFloatComplex));
    cuFloatComplex *B = (cuFloatComplex*)malloc((numPt+NUMCHIEF)*sizeof(cuFloatComplex));
    HOST_CALL(atomicGenSystem(k,elem,numElem,pt,numPt,chief,NUMCHIEF,&src,1,A,numPt+NUMCHIEF,B,numPt+NUMCHIEF));
    //printCuFloatComplexMat(A,numPt+NUMCHIEF,numPt,numPt+NUMCHIEF);
    //printCuFloatComplexMat(B,numPt+NUMCHIEF,1,numPt+NUMCHIEF);
    HOST_CALL(qrSolver(A,numPt+NUMCHIEF,numPt,numPt+NUMCHIEF,B,1,numPt+NUMCHIEF));
    //printCuFloatComplexMat(B,numPt,1,numPt+NUMCHIEF);
    
    free(pt);
    free(elem);
    free(A);
    free(B);
}

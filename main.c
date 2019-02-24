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
    int numPt, numElem;
    findNum("sphere.obj",&numPt,&numElem);
    cartCoord *pt = (cartCoord*)malloc(numPt*sizeof(cartCoord));
    triElem *elem = (triElem*)malloc(numElem*sizeof(triElem));
    readOBJ("sphere.obj",pt,elem);
    cartCoord chief[3];
    genCHIEF(pt,numPt,elem,numElem,chief,3);
    printCartCoord(chief,3);
    printf("Completed.\n");
    free(pt);
    free(elem);
}

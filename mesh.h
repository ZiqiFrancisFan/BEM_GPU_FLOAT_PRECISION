/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   mesh.h
 * Author: ziqi
 *
 * Created on February 23, 2019, 1:30 PM
 */

#ifndef MESH_H
#define MESH_H

#ifdef __cplusplus
extern "C" {
#endif


#include "dataStruct.h"
#include <stdio.h>
#include <stdlib.h>
void findNum(const char * filename,int *pV, int *pE);
void readOBJ(const char *filename, cartCoord* p, triElem *e);
void printPts(const cartCoord *p,const int num);
void printElems(const triElem *elem, const int num);
void printCartCoord(const cartCoord *pt, const int numPt);
int findBB(const cartCoord *pt, const int numPt, const float threshold, float x[2], 
    float y[2], float z[2]);
int write_hrtfs_to_file(const cuFloatComplex *HRTFs_le, const cuFloatComplex *HRTFs_re, 
        const int numSrcs, const int numFreqs, const char* file_le, const char* file_re);
    




#ifdef __cplusplus
}
#endif

#endif /* MESH_H */


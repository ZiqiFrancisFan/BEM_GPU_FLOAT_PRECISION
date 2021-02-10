/*
 ©Copyright 2020 University of Florida Research Foundation, Inc. All Commercial Rights Reserved.
 For commercial license interest please contact UF Innovate | Tech Licensing, 747 SW 2nd Avenue, P.O. Box 115575, Gainesville, FL 32601,
 Phone (352) 392-8929 and reference UF technology T18423.
 */

#ifndef MESH_H
#define MESH_H



#include "dataStructs.h"
#include <stdio.h>
#include <stdlib.h>
void findNum(const char * filename,int *pV, int *pE);

void readOBJ(const char *filename, rect_coord_flt* p, tri_elem* e);

void readOBJ(const char *filename, rect_coord_dbl* p, tri_elem* e);

int FindNumObj(const char *file, int& num);

void printPts(const vec3f* p,const int num);
void printElems(const tri_elem* elem, const int num);

int findBB(const vec3f* pt, const int numPt, const float threshold, float x[2], 
    float y[2], float z[2]);

int write_hrtfs_to_file(const cuFloatComplex* HRTFs_le, const cuFloatComplex* HRTFs_re, 
        const int numSrcs, const int numFreqs, const char* file_le, const char* file_re);


#endif /* MESH_H */


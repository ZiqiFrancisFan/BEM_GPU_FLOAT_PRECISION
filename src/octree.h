/*
 ©Copyright 2020 University of Florida Research Foundation, Inc. All Commercial Rights Reserved.
 For commercial license interest please contact UF Innovate | Tech Licensing, 747 SW 2nd Avenue, P.O. Box 115575, Gainesville, FL 32601,
 Phone (352) 392-8929 and reference UF technology T18423.
 */

#ifndef OCTREE_H
#define OCTREE_H


#include "dataStructs.h"
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef NUM_BITs_BIN
#define NUM_BITs_BIN (sizeof(unsigned))
#endif
    
#ifndef MAX
#define MAX 100000
#endif

void printSet(const int *set);

bool arrEqual(const int *a, const int *b, const int num);

int parent(int num);

int child(int num, int cld);

void children(const int num, int *cldrn);

rect_coord_dbl scale(const rect_coord_dbl x, const rect_coord_dbl x_min, const double d);

rect_coord_dbl descale(const rect_coord_dbl x_s, const rect_coord_dbl x_min, const double d);

double descale_1d(const double a, const double D, const double v_min);

void scalePnts(const rect_coord_dbl* pnt, const int numPnts, const rect_coord_dbl pnt_min, 
        const double d, rect_coord_dbl* pnt_scaled);

void dec2bin_frac(double s, int l, int *h);

void dec2bin_int(unsigned num, int *rep, int *numBits);

void bitIntleave(const int *x, const int *y, const int *z, const int l, int *result);

void bitDeintleave(const int *result, const int l, int *x, int *y, int *z);

int indArr2num(const int *ind, const int l, const int d);

int pnt2boxnum(const rect_coord_dbl pnt, const int l);

rect_coord_dbl boxCenter(const int num, const int l);

int neighbors(const int num, const int l, int *numNeighbors, int *nbr);

void createSet(const int *elems, const int numElems, int *set);

bool isMember(const int t, const int *set);

bool isEmpty(const int *set);

void intersection(const int *set1, const int *set2, int *set3);

void Union(const int *set1, const int *set2, int *set3);

void difference(const int *set1, const int *set2, int *set3);

void pnts2numSet(const rect_coord_dbl *pnts, const int numPnts, const int l, 
        int *set);

void sampleSpace(const int l, int *set);

void I1(const int num, int *set);

void I2(const int num, const int l, int *set);

void I3(const int num, const int l, int *set);

//applicable to levels larger than or equal to 2
void I4(const int num, const int l, int *set);

void orderArray(const int *a, const int num, int *ind);

void printPnts(const rect_coord_dbl *pt, const int numPt);

void printPnts_d(const rect_coord_dbl *p, const int numPnts);

void genOctPt(const int level, rect_coord_dbl *pt);

int deterLmax(const rect_coord_dbl *pnts, const int numPnts, const int s);

void findBoundingCube(const rect_coord_dbl *pnts, const int numPnts, const double eps, 
        rect_coord_dbl *pnts_b, double *d);

void srcBoxes(const rect_coord_dbl *pnts, const tri_elem *elems, const int numElems, 
        const int s, int *srcBoxSet, int *lmax, double *D, rect_coord_dbl *pnt_min);

int findSetInd(const int *X, const int num);

void sortSet(int *set);

void createMeshOccupancyGrid(const rect_coord_dbl* nod, const int numNod, 
        const tri_elem* elem, const int numElem, const rect_coord_dbl cnr, 
        const double sideLen, const int level, int* grid);

int write_occupancy_grid(const int* grid, const int level, const char* filename);

#endif /* OCTREE_H */

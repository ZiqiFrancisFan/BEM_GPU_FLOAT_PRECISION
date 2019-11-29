/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   octree.h
 * Author: ziqi
 *
 * Created on March 5, 2019, 9:15 AM
 */

#ifndef OCTREE_H
#define OCTREE_H


#include "dataStructs.h"
#include <stdbool.h>
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

cart_coord_double scale(const cart_coord_double x, const cart_coord_double x_min, const double d);

cart_coord_double descale(const cart_coord_double x_s, const cart_coord_double x_min, const double d);

double descale_1d(const double a, const double D, const double v_min);

void scalePnts(const cart_coord_double* pnt, const int numPnts, const cart_coord_double pnt_min, 
        const double d, cart_coord_double* pnt_scaled);

void dec2bin_frac(double s, int l, int *h);

void dec2bin_int(unsigned num, int *rep, int *numBits);

void bitIntleave(const int *x, const int *y, const int *z, const int l, int *result);

void bitDeintleave(const int *result, const int l, int *x, int *y, int *z);

int indArr2num(const int *ind, const int l, const int d);

int pnt2boxnum(const cart_coord_double pnt, const int l);

cart_coord_double boxCenter(const int num, const int l);

int neighbors(const int num, const int l, int *numNeighbors, int *nbr);

void createSet(const int *elems, const int numElems, int *set);

bool isMember(const int t, const int *set);

bool isEmpty(const int *set);

void intersection(const int *set1, const int *set2, int *set3);

void Union(const int *set1, const int *set2, int *set3);

void difference(const int *set1, const int *set2, int *set3);

void pnts2numSet(const cart_coord_double *pnts, const int numPnts, const int l, 
        int *set);

void sampleSpace(const int l, int *set);

void I1(const int num, int *set);

void I2(const int num, const int l, int *set);

void I3(const int num, const int l, int *set);

//applicable to levels larger than or equal to 2
void I4(const int num, const int l, int *set);

void orderArray(const int *a, const int num, int *ind);

void printPnts(const cart_coord_double *pt, const int numPt);

void printPnts_d(const cart_coord_double *p, const int numPnts);

void genOctPt(const int level, cart_coord_double *pt);

int deterLmax(const cart_coord_double *pnts, const int numPnts, const int s);

void findBoundingCube(const cart_coord_double *pnts, const int numPnts, const double eps, 
        cart_coord_double *pnts_b, double *d);

void srcBoxes(const cart_coord_double *pnts, const tri_elem *elems, const int numElems, 
        const int s, int *srcBoxSet, int *lmax, double *D, cart_coord_double *pnt_min);

int findSetInd(const int *X, const int num);

void sortSet(int *set);


#endif /* OCTREE_H */

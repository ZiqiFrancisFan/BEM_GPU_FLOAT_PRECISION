/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
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
    
    free(pts);
    free(elems);
}

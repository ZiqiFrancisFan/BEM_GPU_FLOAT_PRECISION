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
#include <time.h>
#include "dataStructs.h"
#include "mesh.h"
#include "octree.h"
#include "numerical.h"
#include "geometry.h"

int main(int argc, char *argv[])
{
    ln_seg_dbl seg;
    seg.nod[0] = {1,1,0};
    seg.nod[1] = {0.5,0.5,0};
    tri_dbl tri;
    tri.nod[0] = {0,0,0};
    tri.nod[1] = {1,0,0};
    tri.nod[2] = {0,1,0};
    int rel = deterLnSegTriRel(seg,tri);
    if(rel==0) {
        printf("No intersection.\n");
    }
    else {
        printf("Intersection.\n");
    }
    return EXIT_SUCCESS;
}

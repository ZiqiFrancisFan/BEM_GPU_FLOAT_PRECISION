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
    cube_dbl cb;
    cb.cnr = {0,0,0};
    cb.len = 1.0;
    
    tri_dbl tri;
    tri.nod[0] = {-3,0,0};
    tri.nod[1] = {0,3,0};
    tri.nod[2] = {0,0,3};
    
    int rel = deterTriCubeInt(tri,cb);
    if(rel==1) {
        printf("Intersection is not empty.\n");
    }
    else {
        printf("Intersection is empty.\n");
    }
    return EXIT_SUCCESS;
}

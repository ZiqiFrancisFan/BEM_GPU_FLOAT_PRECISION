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

int main(int argc, char *argv[])
{
    rect_coord_dbl pt = {1.0,1.0,3.01};
    cube_dbl cb;
    plane_dbl pln;
    pln.n = {1,0,0};
    pln.pt = {0,0,0};
    cb.cnr = {0,0,0};
    cb.len = 2.0;
    int flag = deterPtCubeRel(pt,cb);
    if(flag) {
        printf("In.\n");
    } else {
        printf("Not in.\n");
    }
    flag = deterPtPlaneRel(pt,pln);
    if(flag) {
        printf("On the non-negative side.\n");
    } else {
        printf("On the negative side.\n");
    }
    flag = deterPtCubeEdgeVolRel(pt,cb);
    if(flag) {
        printf("In.\n");
    } else {
        printf("Not in.\n");
    }
    return EXIT_SUCCESS;
}

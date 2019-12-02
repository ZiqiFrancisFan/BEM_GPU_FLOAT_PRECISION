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
    ln_seg_dbl lnSeg;
    lnSeg.nod[0] = {2,0,0};
    lnSeg.nod[1] = {0,2,0};
    quad_dbl quad;
    quad.nod[0] = {-1,-1,0};
    quad.nod[1] = {1,-1,0};
    quad.nod[2] = {1,1,0};
    quad.nod[3] = {-1,1,0};
    
    int result = deterLnSegQuadRel(lnSeg,quad);
    if(result==1) {
        printf("The line segment intersects the quad.\n");
    } else {
        printf("The line segment does not intersects the quad.\n");
    }
    
    rect_coord_dbl vec[3];
    vec[0].coords[0] = 1;
    vec[0].coords[1] = 4;
    vec[0].coords[2] = 7;
    vec[1].coords[0] = 2;
    vec[1].coords[1] = 5;
    vec[1].coords[2] = 8;
    vec[2].coords[0] = 3;
    vec[2].coords[1] = 6;
    vec[2].coords[2] = 9;
    double t = rectCoordDet(vec);
    printf("%lf\n",t);
    return EXIT_SUCCESS;
}

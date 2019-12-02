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
    line_dbl ln1, ln2;
    ln1.pt = {-1,0,0};
    ln1.dir = {2,0,0};
    ln2.pt = {0,1,0};
    ln2.dir = {0,-2,0};
    double t1, t2;
    int result = deterLnLnRel(ln1,ln2,&t1,&t2);
    if(result==0) {
        printf("The two lines do not intersect.\n");
    } else {
        if(result==1) {
            rect_coord_dbl intersection = rectCoordAdd(ln1.pt,scaRectMul(t1,ln1.dir));
            printf("The one intersection point is (%lf,%lf,%lf).\n",intersection.coords[0],
                    intersection.coords[1],intersection.coords[2]);
        } else {
            printf("The two lines are the same.\n");
        }
    }
    return EXIT_SUCCESS;
}

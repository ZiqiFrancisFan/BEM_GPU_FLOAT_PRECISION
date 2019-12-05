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
#include <float.h>

extern vec3d bases[3];

int main(int argc, char *argv[])
{
    setHostBases();
    aarect3d rect;
    rect.cnr = {1,1,1};
    rect.len[0] = 1;
    rect.len[1] = 2;
    rect.len[2] = 1;
    vec3d ax = nrmlzVec(vecAdd(vecAdd(vecAdd({0,0,0},bases[0]),bases[1]),bases[2]));
    intvl3d intvl = GetIntvl(rect,ax);
    printf("interval: [%lf,%lf]\n",intvl.min,intvl.max);
    return EXIT_SUCCESS;
}

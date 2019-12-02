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
    ln_seg_dbl seg1, seg2;
    seg1.nod[0] = {1,1,0};
    seg1.nod[1] = {2,2,0};
    seg2.nod[0] = {0,1,0};
    seg2.nod[1] = {1,0,0};
    int rel = deterLnSegLnSegRel(seg1,seg2);
    if(rel==0) {
        printf("No intersection\n");
    }
    else {
        printf("Intersects.\n");
    }
    return EXIT_SUCCESS;
}

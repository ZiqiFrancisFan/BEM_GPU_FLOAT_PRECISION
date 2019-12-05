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
    aarect2d rect1, rect2;
    bool rel;
    rect1.cnr = {0,0};
    rect1.len[0] = 1;
    rect1.len[1] = 1;
    rect2.cnr = {-0.5,-0.5};
    rect2.len[0] = 0.49;
    rect2.len[1] = 0.49;
    rel = AaRectAaRectOvlp(rect1,rect2);
    if(rel) {
        printf("The two rectangles overlap.\n");
    }
    else {
        printf("The two rectangles do not overlap.\n");
    }
    return EXIT_SUCCESS;
}

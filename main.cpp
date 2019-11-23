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
    int index = 2;
    int level = 1;
    cart_coord_double t = boxCenter(index,level);
    printf("The center of the box with index %d and level %d is (%f,%f,%f).\n",
            index,level,t.coords[0],t.coords[1],t.coords[2]);
}

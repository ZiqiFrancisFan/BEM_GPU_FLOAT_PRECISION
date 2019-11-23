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

int main(int argc, char *argv[]) {
    // test octree
    int t = parent(10);
    printf("The parent box of 3 is %d.\n",t);
    cart_coord_double temp = {1.0,2.0,3.0};
}

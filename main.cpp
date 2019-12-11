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
extern float intpt[INTORDER];
extern float intwgt[INTORDER];

int main(int argc, char *argv[])
{
    HOST_CALL(cuGenGaussParams(INTORDER,intpt,intwgt));
    HOST_CALL(gaussPtsToDevice(intpt,intwgt));
    SetHostBases();
    HOST_CALL(CopyBasesToConstant());
    vec3f src_loc = {0,-0.5,0};
    
    float mag = 10;
    aarect3d rect;
    rect.cnr = {-0.2,-0.2,-0.2};
    rect.len[0] = 0.4;
    rect.len[1] = 0.4;
    rect.len[2] = 0.4;
    float band[2];
    band[0] = 2*PI*4000;
    band[2] = 2*PI*8000;
    //HOST_CALL(GenerateVoxelField("./mesh/sphere_100mm_5120.obj",2*PI*4000/SPEED_SOUND,"point",&src_loc,&mag,1,rect,0.005,"./data/vox","./data/field"));
    HOST_CALL(WriteLoudnessGeometry("./mesh/sphere_100mm_5120.obj",band,"point",
            &mag,&src_loc,1,rect,0.0025,"./data/vox","./data/loudness"));
    
    return EXIT_SUCCESS;
}

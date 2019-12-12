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
    vec3f src_loc[4] = {{0,-6,0.5},{-6,0,0.5},{0,6,0.5},{6,0,0.5}};
    
    float mag[4] = {1,1,1,1};
    aarect3d rect;
    rect.cnr = {-5,-5,0};
    rect.len[0] = 10;
    rect.len[1] = 10;
    rect.len[2] = 1;
    aarect2d rect_2d;
    rect_2d.cnr.coords[0] = rect.cnr.coords[0];
    rect_2d.cnr.coords[1] = rect.cnr.coords[1];
    rect_2d.len[0] = rect.len[0];
    rect_2d.len[1] = rect.len[1];
    double len = 0.01;
    double zCoord = rect.len[2]/2+rect.cnr.coords[2];
    float band[2];
    band[0] = 2*PI*1000;
    band[2] = 2*PI*2000;
    //HOST_CALL(GenerateVoxelField("./mesh/sphere_100mm_5120.obj",2*PI*4000/SPEED_SOUND,"point",&src_loc,&mag,1,rect,0.005,"./data/vox","./data/field"));
    //HOST_CALL(WriteLoudnessGeometry("./mesh/test.obj",band,"point",
            //mag,src_loc,4,rect,0.01,"./data/vox","./data/loudness"));
    HOST_CALL(WriteZSliceVoxLoudness("./mesh/test1.obj",band,"point",mag,src_loc,4,
            zCoord,len,rect_2d,"./data/vox","./data/loudness"));
    
    CUDA_CALL(cudaDeviceReset());
    return EXIT_SUCCESS;
}

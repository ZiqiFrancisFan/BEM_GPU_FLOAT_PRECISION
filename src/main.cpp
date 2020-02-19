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
    /*
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
    */
    HOST_CALL(cuGenGaussParams(INTORDER,intpt,intwgt));
    HOST_CALL(gaussPtsToDevice(intpt,intwgt));
    SetHostBases();
    HOST_CALL(CopyBasesToConstant());
    int c;
    int option_index = 0;
    char obj_file[200] = "./mesh/test.obj", src_type[50] = "point", vox_file[200] = "./data/vox", 
            field_file[200] = "./data/loudness";
    double len, radius = 3.0, src_z_coord = 0.0, src_mag = 1.0, x_cnr = -5, y_cnr = -5, z_cnr = -5, 
            x_len = 10, y_len = 10, z_len = 10;
    float band[2];
    int src_num = 4;
    aarect3d rect;
    struct option long_options[] = {
        {"obj_file", required_argument, NULL, 0},
        {"src_type", required_argument, NULL, 1},
        {"vox_file", required_argument, NULL, 2},
        {"field_file", required_argument, NULL, 3},
        {"low_ng_freq", required_argument, NULL, 4},
        {"up_ng_freq", required_argument, NULL, 5},
        {"src_radi", required_argument, NULL, 6},
        {"src_z_coord", required_argument, NULL, 7},
        {"src_num", required_argument, NULL, 8},
        {"src_mag", required_argument, NULL, 9},
        {"x_cnr", required_argument, NULL, 10},
        {"y_cnr", required_argument, NULL, 11},
        {"z_cnr", required_argument, NULL, 12},
        {"x_len", required_argument, NULL, 13}, 
        {"y_len", required_argument, NULL, 14},
        {"z_len", required_argument, NULL, 15},
        {"side_len", required_argument, NULL, 16},
        {0,0,0,0}
    };
    // parse command line arguments
    while(1) {
        c = getopt_long(argc,argv,"",long_options,&option_index);
        if(c == -1) 
            break;
        switch(c) {
            case 0:
                strcpy(obj_file,optarg);
                //printf("file name: %s\n",obj_file);
                break;
            case 1:
                strcpy(src_type,optarg);
                //printf("source type: %s\n",src_type);
                break;
            case 2:
                strcpy(vox_file,optarg);
                //printf("vox file: %s\n",vox_file);
                break;
            case 3:
                strcpy(field_file,optarg);
                //printf("field file: %s\n",field_file);
                break;
            case 4:
                band[0] = atof(optarg);
                //printf("lower frequency bound: %lf\n",band[0]);
                break;
            case 5:
                band[1] = atof(optarg);
                //printf("frequency interpolation: %lf\n",band[1]);
                break;
            case 6:
                radius = atof(optarg);
                //printf("radius of sources: %lf\n",radius);
                break;
            case 7:
                src_z_coord = atof(optarg);
                break;
            case 8:
                src_num = atoi(optarg);
                //printf("number of sources: %d\n",src_num);
                break;
            case 9:
                src_mag = atof(optarg);
                //printf("magnitude of sources: %lf\n",src_mag);
                break;
            case 10:
                x_cnr = atof(optarg);
                //printf("x coordinate of the corner: %lf\n",x_cnr);
                break;
            case 11:
                y_cnr = atof(optarg);
                //printf("y coordinate of the corner: %lf\n",y_cnr);
                break;
            case 12:
                z_cnr = atof(optarg);
                //printf("length in x direction: %lf\n",x_len);
                break;
            case 13:
                x_len = atof(optarg);
                //printf("length in y direction: %lf\n",y_len);
                break;
            case 14:
                y_len = atof(optarg);
                //printf("coordinate of z slice: %lf\n",z_coord);
                break;
            case 15:
                z_len = atof(optarg);
                //printf("side length: %lf\n",len);
                break;
            case 16:
                len = atof(optarg);
                break;
            default:
                printf("The current option is not recognized.\n");
        }   
    }
    
    // set up sources
    vec3f *src = (vec3f*)malloc(src_num*sizeof(vec3f));
    for(int i=0;i<src_num;i++) {
        double theta = 2*PI/src_num*i;
        double x = radius*cos(theta);
        double y = radius*sin(theta);
        src[i].coords[0] = x;
        src[i].coords[1] = y;
        src[i].coords[2] = src_z_coord;
    }
    float *mag = (float*)malloc(src_num*sizeof(float));
    for(int i=0;i<src_num;i++) {
        mag[i] = src_mag;
    }
    rect.cnr.coords[0] = x_cnr;
    rect.cnr.coords[1] = y_cnr;
    rect.cnr.coords[2] = z_cnr;
    rect.len[0] = x_len;
    rect.len[1] = y_len;
    rect.len[2] = z_len;
    
    HOST_CALL(WriteLoudnessGeometry(obj_file,band,"point",mag,src,src_num,rect,len,vox_file,field_file));
    
    CUDA_CALL(cudaDeviceReset());
    free(mag);
    free(src);
    
    return EXIT_SUCCESS;
}

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
#include "dataStruct.h"
#include "mesh.h"
#include "numerical.h"

int main(int argc, char *argv[]) {
    // parse command line arguments
    int c;
    int option_index = 0;
    char file_name[50] = "", source_name[20] = "";
    int left_index = 0, right_index = 0;
    float low_freq = 0, high_freq = 0, freq_interp = 0, low_phi = 0, high_phi = 0, phi_interp = 0, 
            low_theta = 0, high_theta = 0, theta_interp = 0;
    struct option long_options[] = {
        {"file", required_argument, NULL, 0},
        {"left", required_argument, NULL, 1},
        {"right", required_argument, NULL, 2},
        {"low_freq", required_argument, NULL, 3},
        {"high_freq", required_argument, NULL, 4},
        {"freq_interp", required_argument, NULL, 5},
        {"source", required_argument, NULL, 6},
        {"low_phi", required_argument, NULL, 7},
        {"high_phi", required_argument, NULL, 8},
        {"phi_interp", required_argument, NULL, 9},
        {"low_theta", required_argument, NULL, 10},
        {"high_theta", required_argument, NULL, 11},
        {"theta_interp", required_argument, NULL, 12},
        {0,0,0,0}
    };
    // parse command line arguments
    while(1) {
        c = getopt_long(argc,argv,"",long_options,&option_index);
        if(c == -1) 
            break;
        switch(c) {
            case 0:
                strcpy(file_name,optarg);
                printf("file name: %s\n",file_name);
                break;
            case 1:
                left_index = atoi(optarg);
                printf("index of the left ear: %d\n",left_index);
                break;
            case 2:
                right_index = atoi(optarg);
                printf("index of the right ear: %d\n",right_index);
                break;
            case 3:
                low_freq = atof(optarg);
                printf("lower frequency bound: %f\n",low_freq);
                break;
            case 4:
                high_freq = atof(optarg);
                printf("higher frequency bound: %f\n",high_freq);
                break;
            case 5:
                freq_interp = atof(optarg);
                printf("frequency interpolation: %f\n",freq_interp);
                break;
            case 6:
                strcpy(source_name,optarg);
                printf("the name of the source: %s\n",source_name);
                break;
            case 7:
                low_phi = atof(optarg);
                printf("the lower bound of phi: %f\n",low_phi);
                break;
            case 8:
                high_phi = atof(optarg);
                printf("the higher bound of phi: %f\n",high_phi);
                break;
            case 9:
                phi_interp = atof(optarg);
                printf("the interpolation of phi: %f\n",phi_interp);
                break;
            case 10:
                low_theta = atof(optarg);
                printf("the lower bound of theta: %f\n",low_theta);
                break;
            case 11:
                high_theta = atof(optarg);
                printf("the higher bound of theta: %f\n",high_theta);
                break;
            case 12:
                theta_interp = atof(optarg);
                printf("the interpolation of theta: %f\n",theta_interp);
                break;
            default:
                printf("The current option is not recognized.\n");
        }   
    }
    
    // provide default arguments to the file name and the source name
    if(strcmp(file_name,"") == 0) {
        strcpy(file_name,"sphere_100mm.obj");
    }
    if(strcmp(source_name,"") == 0) {
        strcpy(source_name,"plane");
    }
    
    // check evaluation points
    if(left_index == 0 || right_index == 0) {
        printf("Evaluation point missing.\n");
        return EXIT_FAILURE;
    }
    
    // check frequency
    if(high_freq == 0) {
        high_freq = low_freq;
        freq_interp = 0;
    }
    if(high_freq < low_freq) {
        printf("Upper frequency bound smaller than lower frequency bound!\n");
        return EXIT_FAILURE;
    }
    if(low_freq < high_freq && freq_interp <= 0) {
        printf("Frequency interpolation not set properly!\n");
        return EXIT_FAILURE;
    }
    
    // check phi
    if(high_phi == 0) {
        high_phi = low_phi;
        phi_interp = 0; // compute a single phi
    }
    if(high_phi < low_phi) {
        printf("Upper bound of phi smaller than lower bound of phi!\n");
        return EXIT_FAILURE;
    }
    if(low_phi < high_phi && phi_interp <= 0) {
        printf("Interpolation of phi not set properly!\n");
        return EXIT_FAILURE;
    }
    
    // check theta
    if(high_theta == 0) {
        high_theta = low_theta;
        theta_interp = 0; // compute a single theta
    }
    if(high_theta < low_theta) {
        printf("Upper bound of theta smaller than lower bound of theta!\n");
        return EXIT_FAILURE;
    }
    if(low_theta < high_theta && theta_interp <= 0) {
        printf("Interpolation of theta not set properly!\n");
        return EXIT_FAILURE;
    }
    
    // generate Gaussian-quadrature evaluation points and weights
    float intPt[INTORDER];
    float intWgt[INTORDER];
    HOST_CALL(genGaussParams(INTORDER,intPt,intWgt));
    HOST_CALL(gaussPtsToDevice(intPt,intWgt));
    
    // read in the mesh
    int numPt, numElem;
    findNum(file_name,&numPt,&numElem);
    cartCoord *pt = (cartCoord*)malloc(numPt*sizeof(cartCoord));
    triElem *elem = (triElem*)malloc(numElem*sizeof(triElem));
    readOBJ(file_name,pt,elem);
    
    // generate CHIEF points
    cartCoord chief[NUMCHIEF];
    genCHIEF(pt,numPt,elem,numElem,chief,NUMCHIEF);
    
    // conduct computating;
    cartCoord origin = {0,0,0}; // set up origin
    
    
    // plane wave
    if(strcmp(source_name,"plane")==0) {
        // compute the number of directions
        int numHorDirs = floor((high_phi-low_phi)/phi_interp)+1;
        int numVertDirs = floor((high_theta-low_theta)/theta_interp)+1;
        
        cartCoord* dirs = (cartCoord*)malloc((numHorDirs+numVertDirs)*sizeof(cartCoord)); // memory for directions
        
        // set up horizontal directions
        for(int i=0;i<numHorDirs;i++) {
            float phi = low_phi+i*phi_interp;
            float theta = 0;
            float r = 1;
            float x = r*sin(theta)*cos(phi);
            float y = r*sin(theta)*sin(phi);
            float z = 0;
            cartCoord tempPt = {x,y,z};
            cartCoord dir = cartCoordSub(origin,tempPt);
            float dirNrm = sqrt(pow(dir.coords[0],2)+pow(dir.coords[1],2)+pow(dir.coords[2],2));
            for(int j=0;j<3;j++) {
                dir.coords[j]/=dirNrm;
            }
            dirs[i] = dir;
        }
        
        // set up vertical directions
        for(int i=0;i<numVertDirs;i++) {
            float theta = low_theta+i*theta_interp;
            float r = 1;
            float x = 0;
            float y = r*sin(theta);
            float z = r*cos(theta);
            cartCoord tempPt = {x,y,z};
            cartCoord dir = cartCoordSub(origin,tempPt);
            float dirNrm = sqrt(pow(dir.coords[0],2)+pow(dir.coords[1],2)+pow(dir.coords[2],2));
            for(int j=0;j<3;j++) {
                dir.coords[j]/=dirNrm;
            }
            dirs[numHorDirs+i] = dir;
        }
        
    }
    
    
    // float freq = 1000;
    const float speed = 343.21;
    float k;
    for(float freq = low_freq;freq <= high_freq; freq += freq_interp) {
        k = 2*PI*freq/speed;
        
    }
    
    //cartCoord src = {10,10,10};
    cartCoord dir = {0,0,1};
    
    //printCartCoord(chief,NUMCHIEF);
    //printf("Completed.\n");
    
    cuFloatComplex *B = (cuFloatComplex*)malloc((numPt+NUMCHIEF)*sizeof(cuFloatComplex));
    //HOST_CALL(bemSolver(k,elem,numElem,pt,numPt,chief,NUMCHIEF,&src,1,B,numPt+NUMCHIEF));
    HOST_CALL(bemSolver_dir(k,elem,numElem,pt,numPt,chief,NUMCHIEF,&dir,1,B,numPt+NUMCHIEF));
    printCuFloatComplexMat(B,numPt,1,numPt+NUMCHIEF);
    printf("Analytical solution: \n");
    computeRigidSphereScattering(pt,numPt,0.1,k,1.0);
    
    free(pt);
    free(elem);
    free(B);
}

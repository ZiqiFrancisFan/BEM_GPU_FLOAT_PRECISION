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
            low_theta = 0, high_theta = 0, theta_interp = 0, radius = 0;
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
        {"radius",required_argument,NULL,13},
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
            case 13:
                radius = atof(optarg);
                printf("the radius of the sources is: %f\n",radius);
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
    
    if(strcmp(source_name,"point")==0 && (abs(radius)<EPS || abs(STRENGTH)<EPS)) {
        printf("the radius and strength of point sources not set properly!\n");
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
    printf("Total number of points: %d, total number of elements: %d\n",numPt,numElem);
    cart_coord_float *pt = (cart_coord_float*)malloc(numPt*sizeof(cart_coord_float));
    tri_elem *elem = (tri_elem*)malloc(numElem*sizeof(tri_elem));
    readOBJ(file_name,pt,elem);
    
    // generate CHIEF points
    cart_coord_float chief[NUMCHIEF];
    genCHIEF(pt,numPt,elem,numElem,chief,NUMCHIEF);
    printf("CHIEF points generated.\n");
    
    // check evaluation points
    if(left_index >= numPt || right_index >= numPt) {
        printf("Evaluation point not set properly.\n");
        return EXIT_FAILURE;
    }
    
    // conduct computating;
    const cart_coord_float origin = {0,0,0}; // set up origin
    
    // plane wave
    if(strcmp(source_name,"plane")==0) {
        printf("Plane wave used.\n");
        // compute the number of directions
        int numHorDirs = floor((high_phi-low_phi)/phi_interp)+1;
        int numVertDirs = floor((high_theta-low_theta)/theta_interp)+1;
        
        cart_coord_float* dirs = (cart_coord_float*)malloc((numHorDirs+numVertDirs)*sizeof(cart_coord_float)); // memory for directions
        
        // set up horizontal directions
        for(int i=0;i<numHorDirs;i++) {
            float phi = low_phi+i*phi_interp;
            phi = PI/180*phi;
            float theta = 90;
            theta = PI/180*theta;
            float r = 1;
            float x = r*sin(theta)*cos(phi);
            float y = r*sin(theta)*sin(phi);
            float z = 0;
            //printf("(%f,%f,%f)\n",x,y,z);
            cart_coord_float tempPt = {x,y,z};
            cart_coord_float dir = cartCoordSub(origin,tempPt);
            float dirNrm = sqrt(pow(dir.coords[0],2)+pow(dir.coords[1],2)+pow(dir.coords[2],2));
            for(int j=0;j<3;j++) {
                dir.coords[j]/=dirNrm;
            }
            dirs[i] = dir;
        }
        
        // set up vertical directions
        for(int i=0;i<numVertDirs;i++) {
            float theta = low_theta+i*theta_interp;
            theta = PI/180*theta;
            float r = 1;
            float x = r*sin(theta);
            float y = 0;
            float z = r*cos(theta);
            cart_coord_float tempPt = {x,y,z};
            cart_coord_float dir = cartCoordSub(origin,tempPt);
            float dirNrm = sqrt(pow(dir.coords[0],2)+pow(dir.coords[1],2)+pow(dir.coords[2],2));
            for(int j=0;j<3;j++) {
                dir.coords[j]/=dirNrm;
            }
            dirs[numHorDirs+i] = dir;
        }
        
        printf("Directions set up.\n");
        
        //for(int i=0;i<numHorDirs+numVertDirs;i++) {
        //    printf("%dth direction: (%f,%f,%f)\n",i,dirs[i].coords[0],dirs[i].coords[1],dirs[i].coords[2]);
        //}
        int numFreqs = floor((high_freq-low_freq)/freq_interp)+1;
        cuFloatComplex *left_hrtfs = (cuFloatComplex*)malloc((numHorDirs+numVertDirs)*numFreqs*sizeof(cuFloatComplex));
        cuFloatComplex *right_hrtfs = (cuFloatComplex*)malloc((numHorDirs+numVertDirs)*numFreqs*sizeof(cuFloatComplex));
        cuFloatComplex* B = (cuFloatComplex*)malloc((numPt+NUMCHIEF)*(numHorDirs+numVertDirs)*sizeof(cuFloatComplex));
        const float speed = 343.21;
        float wavNum;
        int freqIdx = 0;
        for(float freq=low_freq;freq<=high_freq;freq+=freq_interp) {
            printf("Current frequency: %f\n",freq);
            wavNum = 2*PI*freq/speed; //omega/c
            HOST_CALL(bemSolver_dir(wavNum,elem,numElem,pt,numPt,chief,NUMCHIEF,dirs,numHorDirs+numVertDirs,B,numPt+NUMCHIEF));
            //HOST_CALL(bemSolver_dir(k,elem,numElem,pt,numPt,chief,NUMCHIEF,&dir,1,B,numPt+NUMCHIEF));
            for(int i=0;i<numHorDirs+numVertDirs;i++) {
                left_hrtfs[i*numFreqs+freqIdx] = B[IDXC0(left_index,i,numPt+NUMCHIEF)];
                right_hrtfs[i*numFreqs+freqIdx] = B[IDXC0(right_index,i,numPt+NUMCHIEF)];
            }
            freqIdx++;
        }
        
        char left_file_name[50] = "left_hrtfs_dir", right_file_name[50] = "right_hrtfs_dir";
        HOST_CALL(write_hrtfs_to_file(left_hrtfs,right_hrtfs,numHorDirs+numVertDirs,numFreqs,left_file_name,right_file_name));
        
        free(dirs);
        free(B);
        free(left_hrtfs);
        free(right_hrtfs);
    }
    
    if(strcmp(source_name,"point")==0) {
        printf("Point sources used.\n");
        // compute the number of directions
        int numHorDirs = floor((high_phi-low_phi)/phi_interp)+1;
        int numVertDirs = floor((high_theta-low_theta)/theta_interp)+1;
        
        cart_coord_float* srcLocs = (cart_coord_float*)malloc((numHorDirs+numVertDirs)*sizeof(cart_coord_float)); // memory for directions
        for(int i=0;i<numHorDirs;i++) {
            float phi = low_phi+i*phi_interp;
            phi = PI/180*phi;
            float theta = 90;
            theta = PI/180*theta;
            float r = radius;
            float x = r*sin(theta)*cos(phi);
            float y = r*sin(theta)*sin(phi);
            float z = 0;
            //printf("(%f,%f,%f)\n",x,y,z);
            cart_coord_float srcLoc = {x,y,z};
            srcLocs[i] = srcLoc;
        }
        
        // set up vertical directions
        for(int i=0;i<numVertDirs;i++) {
            float theta = low_theta+i*theta_interp;
            theta = PI/180*theta;
            float r = radius;
            float x = r*sin(theta);
            float y = 0;
            float z = r*cos(theta);
            cart_coord_float srcLoc = {x,y,z};
            srcLocs[numHorDirs+i] = srcLoc;
        }
        
        printf("Locations of sources set up.\n");
        
        // find the number of frequencies
        int numFreqs = floor((high_freq-low_freq)/freq_interp)+1;
        cuFloatComplex *left_hrtfs = (cuFloatComplex*)malloc((numHorDirs+numVertDirs)*numFreqs*sizeof(cuFloatComplex));
        cuFloatComplex *right_hrtfs = (cuFloatComplex*)malloc((numHorDirs+numVertDirs)*numFreqs*sizeof(cuFloatComplex));
        cuFloatComplex* B = (cuFloatComplex*)malloc((numPt+NUMCHIEF)*(numHorDirs+numVertDirs)*sizeof(cuFloatComplex));
        const float speed = 343.21;
        float wavNum;
        int freqIdx = 0;
        for(float freq=low_freq;freq<=high_freq;freq+=freq_interp) {
            printf("Current frequency: %f\n",freq);
            wavNum = 2*PI*freq/speed; //omega/c
            HOST_CALL(bemSolver_pt(wavNum,elem,numElem,pt,numPt,chief,NUMCHIEF,srcLocs,numHorDirs+numVertDirs,B,numPt+NUMCHIEF));
            //HOST_CALL(bemSolver_dir(k,elem,numElem,pt,numPt,chief,NUMCHIEF,&dir,1,B,numPt+NUMCHIEF));
            for(int i=0;i<numHorDirs+numVertDirs;i++) {
                left_hrtfs[i*numFreqs+freqIdx] = B[IDXC0(left_index,i,numPt+NUMCHIEF)];
                right_hrtfs[i*numFreqs+freqIdx] = B[IDXC0(right_index,i,numPt+NUMCHIEF)];
            }
            freqIdx++;
        }
        char left_file_name[50] = "left_hrtfs_pt", right_file_name[50] = "right_hrtfs_pt";
        HOST_CALL(write_hrtfs_to_file(left_hrtfs,right_hrtfs,numHorDirs+numVertDirs,numFreqs,left_file_name,right_file_name));
        
        free(srcLocs);
        free(B);
        free(left_hrtfs);
        free(right_hrtfs);
    }
    
    
    //cart_coord_float src = {10,10,10};
    cart_coord_float dir = {0,0,1};
    
    //printCartCoord(chief,NUMCHIEF);
    //printf("Completed.\n");
    
    cuFloatComplex *B = (cuFloatComplex*)malloc((numPt+NUMCHIEF)*sizeof(cuFloatComplex));
    //HOST_CALL(bemSolver(k,elem,numElem,pt,numPt,chief,NUMCHIEF,&src,1,B,numPt+NUMCHIEF));
    HOST_CALL(bemSolver_dir(20,elem,numElem,pt,numPt,chief,NUMCHIEF,&dir,1,B,numPt+NUMCHIEF));
    //printCuFloatComplexMat(B,numPt,1,numPt+NUMCHIEF);
    printf("\n");
    printf("Analytical solution: \n");
    //computeRigidSphereScattering(pt,numPt,0.1,0.1,20,1.0);
    
    cart_coord_float expPt = {1.5,1.5,1.5};
    sph_coord_float s = cart2sph(expPt);
    gsl_complex temp = rigidSphereScattering(20,1,0.1,s.coords[0],s.coords[1]);
    printf("Analytical solution: (%f,%f)\n",GSL_REAL(temp),GSL_IMAG(temp));
    cuFloatComplex temp_cu;
    HOST_CALL(extrapolation_dirs_single_source(20,&expPt,1,elem,numElem,pt,numPt,B,1.0,dir,&temp_cu));
    printf("Numercial solution: (%f,%f)\n",cuCrealf(temp_cu),cuCimagf(temp_cu));
    
    free(pt);
    free(elem);
    free(B);
}

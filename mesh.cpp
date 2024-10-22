/*
 ©Copyright 2020 University of Florida Research Foundation, Inc. All Commercial Rights Reserved.
 For commercial license interest please contact UF Innovate | Tech Licensing, 747 SW 2nd Avenue, P.O. Box 115575, Gainesville, FL 32601, Phone (352) 392-8929
 and reference UF technology T18423.
 */
#include "mesh.h"

void findNum(const char * filename, int *pV, int *pE) 
{
    /*Find the number of vertices and elements in the current geometry*/
    int i = 0, j = 0; // For saving number of vertices and elements
    char line[50]; // For reading each line of file
    FILE *fp = fopen(filename,"r");
    if(fp==NULL) {
        printf("Failed to open file.\n");
        exit(EXIT_FAILURE);
    }
    while(fgets(line,49,fp)!=NULL) {
        if (line[0]=='v') {
            i++;
        }
        if (line[0]=='f') {
            j++;
        }
    }
    *pV = i;
    *pE = j;
    fclose(fp);
}

void readOBJ(const char *filename, cart_coord_float* p, tri_elem* e) 
{
    int temp[3];
    FILE *fp = fopen(filename, "r");
    if (fp == NULL) {
        printf("Failed to open file.\n");
        exit(EXIT_FAILURE);
    }
    int i = 0, j = 0;
    char line[50];
    char type[5];
    while(fgets(line,49,fp)!=NULL) {
        if(line[0] == 'v') {
            sscanf(line,"%s %f %f %f",type,&(p[i].coords[0]),&(p[i].coords[1]),&(p[i].coords[2]));
            i++;
        }

        if(line[0]=='f') {
            sscanf(line, "%s %d %d %d", type, &temp[0], &temp[1], &temp[2]);
            e[j].nodes[0] = temp[0]-1;
            e[j].nodes[1] = temp[1]-1;
            e[j].nodes[2] = temp[2]-1;
            e[j].bc[0] = make_cuFloatComplex(0,0); // ca=0
            e[j].bc[1] = make_cuFloatComplex(1,0); // cb=1
            e[j].bc[2] = make_cuFloatComplex(0,0); // cc=0
            j++;
        }
    }
    fclose(fp);
}

void printPts(const cart_coord_float* p, const int num) 
{
    for(int i=0;i<num;i++) {
        printf("(%f,%f,%f)\n",p[i].coords[0], p[i].coords[1], p[i].coords[2]);
    }
}

void printElems(const tri_elem* elem, const int num) 
{
    for(int i=0;i<num;i++) {
        printf("(%d,%d,%d)\n",elem[i].nodes[0],elem[i].nodes[1],elem[i].nodes[2]);
    }
}

void printCartCoord(const cart_coord_float* pt, const int numPt)
{
    for(int i=0;i<numPt;i++) {
        printf("(%f,%f,%f), ",pt[i].coords[0],pt[i].coords[1],pt[i].coords[2]);
    }
    printf("\n");
}

int findBB(const cart_coord_float* pt, const int numPt, const float threshold, float x[2], 
        float y[2], float z[2])
{
    if(numPt!=0) {
        x[0] = pt[0].coords[0]; 
        x[1] = pt[0].coords[0]; 
        y[0] = pt[0].coords[1]; 
        y[1] = pt[0].coords[1];
        z[0] = pt[0].coords[2]; 
        z[1] = pt[0].coords[2];
        for(int i=1;i<numPt;i++) {
            if(pt[i].coords[0] < x[0]) {
                x[0] = pt[i].coords[0];
            }
            if(pt[i].coords[0] > x[1]) {
                x[1] = pt[i].coords[0];
            }
            if(pt[i].coords[1] < y[0]) {
                y[0] = pt[i].coords[1];
            }
            if(pt[i].coords[1] > y[1]) {
                y[1] = pt[i].coords[1];
            }
            if(pt[i].coords[2] < z[0]) {
                z[0] = pt[i].coords[2];
            }
            if(pt[i].coords[2] > z[1]) {
                z[1] = pt[i].coords[2];
            }
        }
        x[0]-=threshold;
        x[1]+=threshold;
        y[0]-=threshold;
        y[1]+=threshold;
        z[0]-=threshold;
        z[1]+=threshold;
        
        return EXIT_SUCCESS;
    } else {
        printf("No point in the current cloud.\n");
        return EXIT_FAILURE;
    }
}

int write_hrtfs_to_file(const cuFloatComplex *HRTFs_le, const cuFloatComplex *HRTFs_re, 
        const int numSrcs, const int numFreqs, const char *file_le, const char* file_re)
{
    // create two files for left and right ears
    FILE *file_l = fopen(file_le, "w"), *file_r = fopen(file_re,"w");
    if(file_l==NULL || file_r==NULL) {
        printf("Failed to open file!\n");
        return EXIT_FAILURE;
    }
    
    // write left-ear hrtfs to file
    int status, i, j;
    for(i=0;i<numSrcs;i++) {
        for(j=0;j<numFreqs;j++) {
            status = fprintf(file_l,"(%f,%f) ",cuCrealf(HRTFs_le[IDXC0(i,j,numSrcs)]),
                    cuCimagf(HRTFs_le[IDXC0(i,j,numSrcs)]));
            if (status < 0) {
                printf("Failed to write the current line to file.\n");
                return EXIT_FAILURE;
            }
        }
        fprintf(file_l,"\n");
    }
    
    // write right-ear hrtfs to file
    for(i=0;i<numSrcs;i++) {
        for(j=0;j<numFreqs;j++) {
            status = fprintf(file_r,"(%f,%f) ",cuCrealf(HRTFs_re[IDXC0(i,j,numSrcs)]),
                    cuCimagf(HRTFs_re[IDXC0(i,j,numSrcs)]));
            if (status < 0) {
                printf("Failed to write the current line to file.\n");
                return EXIT_FAILURE;
            }
        }
        fprintf(file_r,"\n");
    }
    fclose(file_l);
    fclose(file_r);
    return EXIT_SUCCESS;
}

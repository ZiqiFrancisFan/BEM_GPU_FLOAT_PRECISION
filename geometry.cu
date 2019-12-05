/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
#include "geometry.h"
#include "octree.h"
#include <stdio.h>
#include <stdlib.h>
#include <float.h>

__constant__ vec3d BASES[3];

vec3d bases[3];

__host__ int CopyBasesToConstant()
{
    vec3d bases[3];
    bases[0].coords[0] = 1;
    bases[0].coords[1] = 0;
    bases[0].coords[2] = 0;
    
    bases[1].coords[0] = 0;
    bases[1].coords[1] = 1;
    bases[1].coords[2] = 0;
    
    bases[2].coords[0] = 0;
    bases[2].coords[1] = 0;
    bases[2].coords[2] = 1;
    
    CUDA_CALL(cudaMemcpyToSymbol(BASES,bases,3*sizeof(vec3d),0,cudaMemcpyHostToDevice));
    return EXIT_SUCCESS;
}

__host__ void SetHostBases()
{
    bases[0].coords[0] = 1;
    bases[0].coords[1] = 0;
    bases[0].coords[2] = 0;
    
    bases[1].coords[0] = 0;
    bases[1].coords[1] = 1;
    bases[1].coords[2] = 0;
    
    bases[2].coords[0] = 0;
    bases[2].coords[1] = 0;
    bases[2].coords[2] = 1;
}

__host__ __device__ int deterPtPlaneRel(const vec3d pt, const plane3d plane)
{
    vec3d vec = vecSub(pt,plane.pt);
    double result = vecDotMul(plane.n,vec);
    if(result>=0) {
        // on the positive side of the plane normal
        return 1;
    } else {
        return 0;
    }
}

__host__ __device__ int deterPtCubeEdgeVolRel(const vec3d pt, const aacb3d cb)
{
    /*determine the relationship between a point and the volume bounded by edge faces
     of a cube*/
    
    //declare two vec3d arrays for nod at the bottom and the top face 
    vec3d btm[4], top[4], left[4], right[4], back[4], front[4];
    
    // declare the basis unit vectors
    vec3d dir_x = {1.0,0.0,0.0}, dir_y = {0.0,1.0,0.0}, dir_z = {0.0,0.0,1.0};
    
    //set up btm and top nod
    btm[0] = cb.cnr;
    btm[1] = vecAdd(btm[0],scaVecMul(cb.len,dir_x));
    btm[2] = vecAdd(btm[1],scaVecMul(cb.len,dir_y));
    btm[3] = vecAdd(btm[0],scaVecMul(cb.len,dir_y));
    //printVec(btm,4);
    
    top[0] = vecAdd(btm[0],scaVecMul(cb.len,dir_z));
    top[1] = vecAdd(top[0],scaVecMul(cb.len,dir_x));
    top[2] = vecAdd(top[1],scaVecMul(cb.len,dir_y));
    top[3] = vecAdd(top[0],scaVecMul(cb.len,dir_y));
    //printVec(top,4);
    
    //set up left and right nod
    left[0] = cb.cnr;
    left[1] = vecAdd(left[0],scaVecMul(cb.len,dir_x));
    left[2] = vecAdd(left[1],scaVecMul(cb.len,dir_z));
    left[3] = vecAdd(left[0],scaVecMul(cb.len,dir_z));
    //printVec(left,4);
    
    right[0] = vecAdd(left[0],scaVecMul(cb.len,dir_y));
    right[1] = vecAdd(right[0],scaVecMul(cb.len,dir_x));
    right[2] = vecAdd(right[1],scaVecMul(cb.len,dir_z));
    right[3] = vecAdd(right[0],scaVecMul(cb.len,dir_z));
    //printVec(right,4);
    
    //set up back and front nod
    back[0] = cb.cnr;
    back[1] = vecAdd(back[0],scaVecMul(cb.len,dir_y));
    back[2] = vecAdd(back[1],scaVecMul(cb.len,dir_z));
    back[3] = vecAdd(back[0],scaVecMul(cb.len,dir_z));
    //printVec(back,4);
    
    front[0] = vecAdd(back[0],scaVecMul(cb.len,dir_x));
    front[1] = vecAdd(front[0],scaVecMul(cb.len,dir_y));
    front[2] = vecAdd(front[1],scaVecMul(cb.len,dir_z));
    front[3] = vecAdd(front[0],scaVecMul(cb.len,dir_z));
    //printVec(front,4);
    
    //declare an array nrml for determining the normal of the new plane
    vec3d nrml[3];
    plane3d plane;
    int result;
    
    //deal with the bottom face
    for(int i=0;i<3;i++) {
        nrml[1] = dir_z;
        switch(i) {
            case 0: // edge determined by btm[0] and btm[1]
                nrml[0] = dir_y;
                break;
            case 1: // edge determined by btm[1] and btm[2]
                nrml[0] = scaVecMul(-1,dir_x);
                break;
            case 2: // edge determined by btm[2] and btm[3]
                nrml[0] = scaVecMul(-1,dir_y);
                break;
            case 3: // edge determined by btm[3] and btm[0]
                nrml[0] = dir_x;
                break;
            default:
                printf("Entered the wrong branch.\n");
        }
        nrml[3] = nrmlzVec(vecAdd(nrml[0],nrml[1]));
        plane.n = nrml[3];
        plane.pt = btm[i];
        result = deterPtPlaneRel(pt,plane);
        if(result == 0) {
            //printf("bottom face: %dth node\n",i);
            return 0;
        }
    }
    
    //deal with the top face
    for(int i=0;i<3;i++) {
        nrml[1] = scaVecMul(-1,dir_z);
        switch(i) {
            case 0: // edge determined by top[0] and top[1]
                nrml[0] = dir_y;
                break;
            case 1: // edge determined by top[1] and top[2]
                nrml[0] = scaVecMul(-1,dir_x);
                break;
            case 2: // edge determined by top[2] and top[3]
                nrml[0] = scaVecMul(-1,dir_y);
                break;
            case 3: // edge determined by top[3] and top[0]
                nrml[0] = dir_x;
                break;
            default:
                printf("Entered the wrong branch.\n");
        }
        nrml[3] = nrmlzVec(vecAdd(nrml[0],nrml[1]));
        plane.n = nrml[3];
        plane.pt = top[i];
        result = deterPtPlaneRel(pt,plane);
        if(result==0) {
            //printf("top face: %dth node\n",i);
            return 0;
        }
    }
    
    //deal with the left face
    for(int i=0;i<3;i++) {
        nrml[1] = dir_y;
        switch(i) {
            case 0: // edge determined by left[0] and left[1]
                nrml[0] = dir_z;
                break;
            case 1: // edge determined by left[1] and left[2]
                nrml[0] = scaVecMul(-1,dir_x);
                break;
            case 2: // edge determined by left[2] and left[3]
                nrml[0] = scaVecMul(-1,dir_z);
                break;
            case 3: // edge determined by left[3] and left[0]
                nrml[0] = dir_x;
                break;
            default:
                printf("Entered the wrong branch.\n");
        }
        nrml[3] = nrmlzVec(vecAdd(nrml[0],nrml[1]));
        plane.n = nrml[3];
        plane.pt = left[i];
        result = deterPtPlaneRel(pt,plane);
        if(result==0) {
            //printf("left face: %dth node\n",i);
            return 0;
        }
    }
    
    //deal with the right face
    for(int i=0;i<3;i++) {
        nrml[1] = scaVecMul(-1,dir_y);
        switch(i) {
            case 0: // edge determined by right[0] and right[1]
                nrml[0] = dir_z;
                break;
            case 1: // edge determined by right[1] and right[2]
                nrml[0] = scaVecMul(-1,dir_x);
                break;
            case 2: // edge determined by right[2] and right[3]
                nrml[0] = scaVecMul(-1,dir_z);
                break;
            case 3: // edge determined by btm[3] and btm[0]
                nrml[0] = dir_x;
                break;
            default:
                printf("Entered the wrong branch.\n");
        }
        nrml[3] = nrmlzVec(vecAdd(nrml[0],nrml[1]));
        plane.n = nrml[3];
        plane.pt = right[i];
        result = deterPtPlaneRel(pt,plane);
        if(result==0) {
            //printf("right face: %dth node\n",i);
            return 0;
        }
    }
    
    //deal with the back face
    for(int i=0;i<3;i++) {
        nrml[1] = dir_x;
        switch(i) {
            case 0: // edge determined by back[0] and back[1]
                nrml[0] = dir_z;
                break;
            case 1: // edge determined by btm[1] and btm[2]
                nrml[0] = scaVecMul(-1,dir_y);
                break;
            case 2: // edge determined by btm[2] and btm[3]
                nrml[0] = scaVecMul(-1,dir_z);
                break;
            case 3: // edge determined by btm[3] and btm[0]
                nrml[0] = dir_y;
                break;
            default:
                printf("Entered the wrong branch.\n");
        }
        nrml[3] = nrmlzVec(vecAdd(nrml[0],nrml[1]));
        plane.n = nrml[3];
        plane.pt = back[i];
        result = deterPtPlaneRel(pt,plane);
        if(result==0) {
            //printf("back face: %dth node\n",i);
            return 0;
        }
    }
    
    //deal with the front face
    for(int i=0;i<3;i++) {
        nrml[1] = scaVecMul(-1,dir_x);
        switch(i) {
            case 0: // edge determined by front[0] and front[1]
                nrml[0] = dir_z;
                break;
            case 1: // edge determined by front[1] and front[2]
                nrml[0] = scaVecMul(-1,dir_y);
                break;
            case 2: // edge determined by front[2] and front[3]
                nrml[0] = scaVecMul(-1,dir_z);
                break;
            case 3: // edge determined by front[3] and front[0]
                nrml[0] = dir_y;
                break;
            default:
                printf("Entered the wrong branch.\n");
        }
        nrml[3] = nrmlzVec(vecAdd(nrml[0],nrml[1]));
        plane.n = nrml[3];
        plane.pt = front[i];
        result = deterPtPlaneRel(pt,plane);
        if(result==0) {
            //printf("front face: %dth node\n",i);
            return 0;
        }
    }
    
    // if not returned 0, then return 1, the point is inside the volume
    return 1;
}

__host__ __device__ int deterPtCubeVtxVolRel(const vec3d pt, const aacb3d cb)
{
    // declare the basis unit vectors
    vec3d dir_x = {1.0,0.0,0.0}, dir_y = {0.0,1.0,0.0}, dir_z = {0.0,0.0,1.0}, 
            nrml[4], tempPt;
    plane3d plane;
    int result;
    // deal with the eight nod in order
    for(int i=0;i<8;i++) {
        switch(i) {
            case 0: //the first vertex
                tempPt = cb.cnr;
                nrml[0] = dir_x;
                nrml[1] = dir_y;
                nrml[2] = dir_z;
                break;
            case 1: //the second vertex
                tempPt = vecAdd(cb.cnr,scaVecMul(cb.len,dir_x));
                nrml[0] = scaVecMul(-1,dir_x);
                nrml[1] = dir_y;
                nrml[2] = dir_z;
                break;
            case 2: //the third vertex
                tempPt = vecAdd(vecAdd(cb.cnr,scaVecMul(cb.len,dir_x)),scaVecMul(cb.len,dir_y));
                nrml[0] = scaVecMul(-1,dir_x);
                nrml[1] = scaVecMul(-1,dir_y);
                nrml[2] = dir_z;
                break;
            case 3: //the fourth vertex
                tempPt = vecAdd(cb.cnr,scaVecMul(cb.len,dir_y));
                nrml[0] = dir_x;
                nrml[1] = scaVecMul(-1,dir_y);
                nrml[2] = dir_z;
                break;
            case 4: //the fifth vertex
                tempPt = vecAdd(cb.cnr,scaVecMul(cb.len,dir_z));
                nrml[0] = dir_x;
                nrml[1] = dir_y;
                nrml[2] = scaVecMul(-1,dir_z);
                break;
            case 5: //the sixth vertex
                tempPt = vecAdd(vecAdd(cb.cnr,scaVecMul(cb.len,dir_z)),scaVecMul(cb.len,dir_x));
                nrml[0] = scaVecMul(-1,dir_x);
                nrml[1] = dir_y;
                nrml[2] = scaVecMul(-1,dir_z);
                break;
            case 6: //the seventh vertex
                tempPt = vecAdd(vecAdd(vecAdd(cb.cnr,scaVecMul(cb.len,dir_z)),
                        scaVecMul(cb.len,dir_x)),scaVecMul(cb.len,dir_y));
                nrml[0] = scaVecMul(-1,dir_x);
                nrml[1] = scaVecMul(-1,dir_y);
                nrml[2] = scaVecMul(-1,dir_z);
                break;
            case 7: //the eighth vertex
                tempPt = vecAdd(vecAdd(cb.cnr,scaVecMul(cb.len,dir_z)),scaVecMul(cb.len,dir_y));
                nrml[0] = dir_x;
                nrml[1] = scaVecMul(-1,dir_y);
                nrml[2] = scaVecMul(-1,dir_z);
                break;
            default:
                printf("safety purpose.\n");
        }
        nrml[3] = nrmlzVec(vecAdd(vecAdd(nrml[0],nrml[1]),nrml[2]));
        plane.n = nrml[3];
        plane.pt = tempPt;
        result = deterPtPlaneRel(pt,plane);
        if(result == 0) {
            return 0;
        }
    }
    return 1;
}

__host__ __device__ int deterPtCubeRel(const vec3d pt, const aacb3d cube)
{
    vec3d cnr_fru = cube.cnr;
    cnr_fru = vecAdd(cnr_fru,scaVecMul(cube.len,{1,0,0}));
    cnr_fru = vecAdd(cnr_fru,scaVecMul(cube.len,{0,1,0}));
    cnr_fru = vecAdd(cnr_fru,scaVecMul(cube.len,{0,0,1}));
    double x_min = cube.cnr.coords[0], y_min = cube.cnr.coords[1], z_min = cube.cnr.coords[2], 
            x_max = cnr_fru.coords[0], y_max = cnr_fru.coords[1], z_max = cnr_fru.coords[2],
            x = pt.coords[0], y = pt.coords[1], z = pt.coords[2];
    if(x >= x_min && x<= x_max && y >= y_min && y<= y_max && z >= z_min && z<= z_max) {
        return 1;
    } else {
        return 0;
    }
}

__host__ __device__ int deterLinePlaneRel(const line_dbl ln, const plane3d pln, double* t)
{
    if(abs(vecDotMul(ln.dir,pln.n))<EPS) {
        //line parallel to plane
        if(abs(vecDotMul(pln.n,vecSub(ln.pt,pln.pt)))<EPS) {
            return 2;
        } else {
            return 0;
        }
    } else {
        double temp = vecDotMul(pln.n,vecSub(pln.pt,ln.pt))/vecDotMul(pln.n,ln.dir);
        *t = temp;
        return 1;
    }
}

__host__ __device__ double triArea(const tri_dbl s)
{
    vec3d vec[2];
    vec[0] = vecSub(s.nod[1],s.nod[0]);
    vec[1] = vecSub(s.nod[2],s.nod[0]);
    return 0.5*vecNorm(vecCrossMul(vec[0],vec[1]));
}

__host__ __device__ double quadArea(const quad_dbl s)
{
    vec3d vec[2];
    vec[0] = vecSub(s.nod[1],s.nod[0]);
    vec[1] = vecSub(s.nod[2],s.nod[0]);
    return vecNorm(vecCrossMul(vec[0],vec[1]));
}

__host__ __device__ plane3d tri2plane(const tri_dbl tri)
{
    plane3d pln;
    pln.pt = tri.nod[0];
    vec3d vec[2];
    vec[0] = vecSub(tri.nod[1],tri.nod[0]);
    vec[1] = vecSub(tri.nod[2],tri.nod[0]);
    pln.n = nrmlzVec(vecCrossMul(vec[0],vec[1]));
    return pln;
}

__host__ __device__ plane3d quad2plane(const quad_dbl qd)
{
    /*get the plane containing a quad*/
    plane3d pln;
    pln.pt = qd.nod[0];
    vec3d vec[2];
    vec[0] = vecSub(qd.nod[1],qd.nod[0]);
    vec[1] = vecSub(qd.nod[2],qd.nod[0]);
    pln.n = nrmlzVec(vecCrossMul(vec[0],vec[1]));
    return pln;
}

__host__ __device__ line_dbl lnSeg2ln(const lnseg3d ls)
{
    line_dbl l;
    l.pt = ls.nod[0];
    l.dir = vecSub(ls.nod[1],ls.nod[0]);
    return l;
}

__host__ __device__ int deterPtTriRel(const vec3d pt, const tri_dbl tri)
{
    /*determine the relationship between a point and a quad on the same plane
     return: 
     1: pt in tri
     0: pt outsidie tri*/
    double area = 0.0;
    vec3d vec[2];
    for(int i=0;i<3;i++) {
        vec[0] = vecSub(tri.nod[i%3],pt);
        vec[1] = vecSub(tri.nod[(i+1)%3],pt);
        area += 0.5*vecNorm(vecCrossMul(vec[0],vec[1]));
    }
    double area_tri = triArea(tri);
    if(abs(area-area_tri)<EPS) {
        return 1; // in
    } else {
        return 0; // out
    }
}

__host__ __device__ int deterPtQuadRel(const vec3d pt, const quad_dbl qd)
{
    /*determine the relationship between a point and a quad on the same plane
     return: 
     1: pt in qd
     0: pt outside qd*/
    double area = 0.0;
    vec3d vec[2];
    for(int i=0;i<4;i++) {
        vec[0] = vecSub(qd.nod[i%4],pt);
        vec[1] = vecSub(qd.nod[(i+1)%4],pt);
        area += 0.5*vecNorm(vecCrossMul(vec[0],vec[1]));
    }
    double area_quad = quadArea(qd);
    if(abs(area-area_quad)<EPS) {
        return 1; // in
    } else {
        return 0; // out
    }
}

__host__ __device__ double rectCoordDet(const vec3d vec[3])
{
    double result, v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z;
    
    v1x = vec[0].coords[0];
    v1y = vec[0].coords[1];
    v1z = vec[0].coords[2];
    
    v2x = vec[1].coords[0];
    v2y = vec[1].coords[1];
    v2z = vec[1].coords[2];
    
    v3x = vec[2].coords[0];
    v3y = vec[2].coords[1];
    v3z = vec[2].coords[2];
    
    result = v1x*(v2y*v3z-v3y*v2z)-v2x*(v1y*v3z-v3y*v1z)+v3x*(v1y*v2z-v2y*v1z);
    
    return result;
}

__host__ __device__ int deterLnLnRel(const line_dbl ln1, const line_dbl ln2, double* t1, double* t2)
{   
    if(abs(vecNorm(vecCrossMul(ln1.dir,ln2.dir)))<EPS) {
        // the two lines are either parallel or the same line
        
        // check if a point on line 1 is on line 2
        vec3d vec = vecSub(ln1.pt,ln2.pt);
        if(vecNorm(vec)<EPS) {
            //the points are the same
            return 2; 
        } 
        else {
            if(vecNorm(vecCrossMul(vec,ln2.dir))<EPS) {
                // vec is a multiple of ln2.dir
                return 2; 
            } 
            else {
                // the two lines are parallel
                return 0;
            }
        }
    } 
    else {
        //the two lines either are skew or intersect
        vec3d pt[4];
        pt[0] = ln1.pt;
        pt[1] = vecAdd(ln1.pt,scaVecMul(1.0,ln1.dir));
        pt[2] = ln2.pt;
        pt[3] = vecAdd(ln2.pt,scaVecMul(1.0,ln2.dir));
        //printVec(pt,4);
        if(vecEqual(pt[0],pt[2]) || vecEqual(pt[0],pt[3]) || 
                vecEqual(pt[1],pt[2]) || vecEqual(pt[1],pt[3])) {
            //the two points on the line is the same point
            if(vecEqual(pt[0],pt[2])) {
                *t1 = 0;
                *t2 = 0;
            } 
            else {
                if(vecEqual(pt[0],pt[3])) {
                    *t1 = 0;
                    *t2 = 1.0;
                } 
                else {
                    if(vecEqual(pt[1],pt[2])) {
                        *t1 = 1.0;
                        *t2 = 0.0;
                    } 
                    else {
                        *t1 = 1.0;
                        *t2 = 1.0;
                    }
                }
            }
            return 1;
        } 
        else {
            //
            vec3d vec[3];
            vec[0] = vecSub(pt[1],pt[0]);
            vec[1] = vecSub(pt[2],pt[0]);
            vec[2] = vecSub(pt[3],pt[0]);
            
            //printf("The determinant is: %f\n",rectCoordDet(vec));
            if(abs(rectCoordDet(vec))>EPS) {
                //skew lines
                return 0;
            } 
            else {
                // the two lines intersects. compute it.
                // first find the valid sub-system
                double coeff1[2], coeff2[2];
                for(int i=0;i<3;i++) {
                    coeff1[0] = ln1.dir.coords[i%3];
                    coeff1[1] = ln1.dir.coords[(i+1)%3];
                    coeff2[0] = ln2.dir.coords[i%3];
                    coeff2[1] = ln2.dir.coords[(i+1)%3];
                    //check the determinant of the current system;
                    double det = coeff1[0]*coeff2[1]-coeff1[1]*coeff2[0];
                    if(abs(det)>EPS) {
                        // get the right-hand side
                        double rhs1[2], rhs2[2];
                        rhs1[0] = ln1.pt.coords[i%3];
                        rhs1[1] = ln1.pt.coords[(i+1)%3];
                        rhs2[0] = ln2.pt.coords[i%3];
                        rhs2[1] = ln2.pt.coords[(i+1)%3];
                        double rhs;
                        rhs = (rhs2[0]-rhs1[0])*coeff2[1]-(rhs2[1]-rhs1[1])*coeff2[0];
                        *t1 = rhs/det;
                        rhs = (rhs2[0]-rhs1[0])*coeff1[1]-(rhs2[1]-rhs1[1])*coeff1[0];
                        *t2 = rhs/det;
                        break;
                    }
                }
                return 1;
            }
        }
        
        
    }
}

__host__ __device__ int deterPtLnRel(const vec3d pt, const line_dbl ln)
{
    /*determines the relation between a point and a line*/
    vec3d vec = vecSub(pt,ln.pt);
    if(vecNorm(vecCrossMul(vec,ln.dir))<EPS) {
        return 1;
    } 
    else {
        return 0;
    }
}

__host__ __device__ int deterPtLnSegRel(const vec3d pt, const lnseg3d lnSeg)
{
    /*determines the relation between a point and a line segment*/
    line_dbl ln = lnSeg2ln(lnSeg);
    if(deterPtLnRel(pt,ln)==0) {
        //point not on the line containing the line segment
        return 0;
    } 
    else {
        double t;
        vec3d vec = vecSub(pt,ln.pt);
        for(int i=0;i<3;i++) {
            if(abs(ln.dir.coords[i])>EPS) {
                t = vec.coords[i]/ln.dir.coords[i];
                break;
            }
        }
        if(t>=0 && t<=1) {
            return 1;
        }
        else {
            return 0;
        }
    }
}

__host__ __device__ int deterLnSegLnSegRel(const lnseg3d seg1, const lnseg3d seg2)
{
    /*determines the relation between two line segments
     seg1: a line segment
     seg2: a line segment
     return: 
     0: no intersection
     1: intersection
     2: infinitely many intersections*/
    line_dbl ln1 = lnSeg2ln(seg1), ln2 = lnSeg2ln(seg2);
    double t1, t2;
    int relLnLn = deterLnLnRel(ln1,ln2,&t1,&t2);
    if(relLnLn==0) {
        // the two lines are skew to each other
        return 0;
    }
    else {
        if(relLnLn==1) {
            // the two lines have one intersection
            if(t1>=0 && t1<=1 && t2>=0 && t2<=1) {
                return 1;
            }
            else {
                return 0;
            }
        }
        else {
            // the two lines are the same line
            if(deterPtLnSegRel(seg1.nod[0],seg2)==0 
                    && deterPtLnSegRel(seg1.nod[1],seg2)==0) {
                // no intersection
                return 0;
            }
            else {
                //determine if one or infinitely many intersection points
                for(int i=0;i<2;i++) {
                    for(int j=0;j<2;j++) {
                        if(vecEqual(seg1.nod[i],seg2.nod[j])) {
                            vec3d vec[2];
                            vec[0] = vecSub(seg1.nod[(i+1)%2],seg1.nod[i]);
                            vec[1] = vecSub(seg2.nod[(j+1)%2],seg1.nod[j]);
                            if(vecDotMul(vec[0],vec[1])<0) {
                                return 1;
                            }
                        }
                    }
                }
                return 2;
            }
        }
    }
}

__host__ __device__ int deterLnSegQuadRel(const lnseg3d lnSeg, const quad_dbl qd)
{
    /*determine if a line segment intersects a quad
     the difference between single intersection and infinitely many intersections 
     is not made.
     0: no intersection
     1: intersection*/
    int flag;
    
    //make a line containing the line segment   
    line_dbl ln = lnSeg2ln(lnSeg);
    
    // define a plane containing the quad
    plane3d pln = quad2plane(qd);
    
    // determine the intersection between the line and the plane
    double t;
    flag = deterLinePlaneRel(ln,pln,&t);
    
    // differentiate between different cases
    if(flag==0) {
        // no intersection between the line and the plane
        return 0;
    } 
    else {
        if(flag==2) {
            // infinitely many intersections between the line and plane
            if(deterPtQuadRel(lnSeg.nod[0],qd)==1 || deterPtQuadRel(lnSeg.nod[1],qd)==1) {
                //oen of the nodes is within the quad
                return 1;
            } 
            else {
                // none of the nodes is within the quad, test if segments intersect
                for(int i=0;i<4;i++) {
                    lnseg3d qdLnSeg;
                    qdLnSeg.nod[0] = qd.nod[i%4];
                    qdLnSeg.nod[1] = qd.nod[(i+1)%4];
                    if(deterLnSegLnSegRel(lnSeg,qdLnSeg)!=0) {
                        // the line segment intersects a quad line segment
                        return 1;
                    }
                }
                return 0;
            }
        }
        else {
            //determines if a point is within a quad
            if(t<0 || t>1) {
                // intersection not on the line segment
                return 0;
            } 
            else {
                vec3d intersection = vecAdd(ln.pt,scaVecMul(t,ln.dir));
                if(deterPtQuadRel(intersection,qd)==1) {
                    // intersection in the quad
                    return 1;
                }
                else {
                    return  0;
                }
            }
            
        }
    }
}

__host__ __device__ int deterLnSegTriRel(const lnseg3d lnSeg, const tri_dbl tri)
{
    /*determine if a line segment intersects a quad
     0: no intersection
     1: intersection*/
    int flag;
    
    //make a line containing the line segment    
    line_dbl ln = lnSeg2ln(lnSeg);
    
    // define a plane containing the triangle
    plane3d pln = tri2plane(tri);
    
    // determine the intersection between the line and the plane
    double t;
    flag = deterLinePlaneRel(ln,pln,&t);
    if(flag==0) {
        // no intersection between the line and the plane
        return 0;
    } 
    else {
        if(flag==2) {
            // infinitely many intersections between the line and plane
            if(deterPtTriRel(lnSeg.nod[0],tri)==1 || deterPtTriRel(lnSeg.nod[1],tri)==1) {
                //oen of the nodes is within the quad
                return 1;
            } 
            else {
                // none of the nodes is within the triangle, test if segments intersect
                for(int i=0;i<3;i++) {
                    lnseg3d triLnSeg;
                    triLnSeg.nod[0] = tri.nod[i%3];
                    triLnSeg.nod[1] = tri.nod[(i+1)%3];
                    if(deterLnSegLnSegRel(lnSeg,triLnSeg)!=0) {
                        // the line segment intersects a trianagle line segment
                        return 1;
                    }
                }
                return 0;
            }
        }
        else {
            //determines if a point is within a quad
            if(t<0 || t>1) {
                // intersection not on the line segment
                return 0;
            } 
            else {
                vec3d intersection = vecAdd(ln.pt,scaVecMul(t,ln.dir));
                if(deterPtTriRel(intersection,tri)==1) {
                    // intersection in the tri
                    return 1;
                }
                else {
                    return  0;
                }
            }
            
        }
    }
}

__host__ __device__ int deterTriCubeInt(const tri_dbl tri, const aacb3d cb)
{
    /*this function determines if a triangle intersects with a cube
     tri: an triangle
     cb: a cube
     return: 
     1: intersection
     0: no intersection*/
    
    //test nodes of the triangle against the cube
    int nodRel[3];
    for(int i=0;i<3;i++) {
        nodRel[i] = deterPtCubeRel(tri.nod[i],cb);
        if(nodRel[i]==1) {
            // node i is in the cube, thus the cube is occupied
            return 1;
        }
    }
    
    //test the intersection between edges of the triangle and the six faces of the cube
    int rel = 0;
    lnseg3d triEdge[3];
    quad_dbl cbFace[6];
    lnseg3d cbDiag[4];
    
    //set up translation vectors
    vec3d dir_x = {1,0,0}, dir_y = {0,1,0}, dir_z = {0,0,1};
    dir_x = scaVecMul(cb.len,dir_x);
    dir_y = scaVecMul(cb.len,dir_y);
    dir_z = scaVecMul(cb.len,dir_z);
    
    //set the edges, faces and diagonals
    for(int i=0;i<3;i++) {
        triEdge[i].nod[0] = tri.nod[i];
        triEdge[i].nod[1] = tri.nod[(i+1)%3];
    }
    
    for(int i=0;i<6;i++) {
        vec3d pt;
        switch(i) {
            case 0: //bottom x-y plane
                pt = cb.cnr;
                cbFace[i].nod[0] = pt;
                cbFace[i].nod[1] = vecAdd(cbFace[i].nod[0],dir_x);
                cbFace[i].nod[2] = vecAdd(cbFace[i].nod[1],dir_y);
                cbFace[i].nod[3] = vecAdd(cbFace[i].nod[2],scaVecMul(-1,dir_x));
                break;
            case 1: //up x-y plane
                pt = vecAdd(pt,dir_z);
                cbFace[i].nod[0] = pt;
                cbFace[i].nod[1] = vecAdd(cbFace[i].nod[0],dir_x);
                cbFace[i].nod[2] = vecAdd(cbFace[i].nod[1],dir_y);
                cbFace[i].nod[3] = vecAdd(cbFace[i].nod[2],scaVecMul(-1,dir_x));
                break;
            case 2: //left y-z plane
                pt = cb.cnr;
                cbFace[i].nod[0] = pt;
                cbFace[i].nod[1] = vecAdd(cbFace[i].nod[0],dir_x);
                cbFace[i].nod[2] = vecAdd(cbFace[i].nod[1],dir_z);
                cbFace[i].nod[3] = vecAdd(cbFace[i].nod[2],scaVecMul(-1,dir_x));
                break;
            case 3: //right y-z plane
                pt = vecAdd(cb.cnr,dir_y);
                cbFace[i].nod[0] = pt;
                cbFace[i].nod[1] = vecAdd(cbFace[i].nod[0],dir_x);
                cbFace[i].nod[2] = vecAdd(cbFace[i].nod[1],dir_z);
                cbFace[i].nod[3] = vecAdd(cbFace[i].nod[2],scaVecMul(-1,dir_x));
                break;
            case 4: //back z-x plane
                pt = cb.cnr;
                cbFace[i].nod[0] = pt;
                cbFace[i].nod[1] = vecAdd(cbFace[i].nod[0],dir_y);
                cbFace[i].nod[2] = vecAdd(cbFace[i].nod[1],dir_z);
                cbFace[i].nod[3] = vecAdd(cbFace[i].nod[2],scaVecMul(-1,dir_y));
                break;
            case 5: //front z-x plane
                pt = vecAdd(cb.cnr,dir_x);
                cbFace[i].nod[0] = pt;
                cbFace[i].nod[1] = vecAdd(cbFace[i].nod[0],dir_y);
                cbFace[i].nod[2] = vecAdd(cbFace[i].nod[1],dir_z);
                cbFace[i].nod[3] = vecAdd(cbFace[i].nod[2],scaVecMul(-1,dir_y));
                break;
            default:
                printf("Should not enter this.\n");
        }
    }
    
    // first diagnonal
    cbDiag[0].nod[0] = cb.cnr;
    cbDiag[0].nod[1] = vecAdd(vecAdd(vecAdd(cbDiag[0].nod[0],dir_x),dir_y),dir_z);
    
    // second diagnonal
    cbDiag[1].nod[0] = vecAdd(cbDiag[0].nod[0],dir_x);
    cbDiag[1].nod[1] = vecAdd(vecAdd(vecAdd(cbDiag[1].nod[0],dir_y),scaVecMul(-1,dir_x)),dir_z);
    
    // third diagnoal
    cbDiag[2].nod[0] = vecAdd(cbDiag[1].nod[0],dir_y);
    cbDiag[2].nod[1] = vecAdd(vecAdd(vecAdd(cbDiag[2].nod[0],scaVecMul(-1,dir_x)),scaVecMul(-1,dir_y)),dir_z);
    
    // fourth diagonal
    cbDiag[3].nod[0] = vecAdd(cbDiag[2].nod[0],scaVecMul(-1,dir_x));
    cbDiag[3].nod[1] = vecAdd(vecAdd(vecAdd(cbDiag[3].nod[0],scaVecMul(-1,dir_y)),dir_x),dir_z);
    
    // determine if any of the three edges of the triangle intersects the faces;
    //printf("Entered diagonal test.\n");
    for(int i=0;i<3;i++) {
        for(int j=0;j<6;j++) {
            rel = deterLnSegQuadRel(triEdge[i],cbFace[j]);
            if(rel==1) {
                return 1;
            }
        }
    }
    
    for(int i=0;i<4;i++) {
        rel = deterLnSegTriRel(cbDiag[i],tri);
        if(rel==1) {
            return 1;
        }
    }
    
    return 0;
}

__global__ void testTriCbInt(const tri_dbl* tri, const int numTri, const aacb3d* cb, 
        const int numCb, int* flag)
{
    /*the global function for testing triangle-cube intersection
     tri: an array of triangles
     numTri: the number of triangles
     cb: an array of cubes
     numCb: the number of cubes
     flag: an array of flags, initialized to zero, of size numCb*/
    int idx_x = blockIdx.x*blockDim.x+threadIdx.x; // triangle index
    int idx_y = blockIdx.y*blockDim.y+threadIdx.y; // cube index
    
    if(idx_x < numTri && idx_y < numCb) {
        if(idx_x==0 && idx_y==0) {
            printf("the first node. cube: (%lf,%lf,%lf),%lf\n",cb[idx_y].cnr.coords[0],
                    cb[idx_y].cnr.coords[1],cb[idx_y].cnr.coords[2],cb[idx_y].len);
        }
        int rel = deterTriCubeInt(tri[idx_x],cb[idx_y]);
        if(idx_x==0 && idx_y==0) {
            printf("Completed determination.\n");
        }
        atomicAdd(&flag[idx_y],rel);
    }
}

__host__ int getTriCbRel(const tri_dbl* tri, const int numTri, const aacb3d* cb, 
        const int numCb, int* flag)
{
    /*voxelize a space into occupance grids
     tri: an array of triangles
     numTri: the number of triangles
     cb: an array of cubes
     numCb: number of cubes
     flag: an array of flags for cube occupancy*/
    printf("Entered getTriCbRel.\n");
    tri_dbl *tri_d;
    CUDA_CALL(cudaMalloc(&tri_d,numTri*sizeof(tri_dbl)));
    CUDA_CALL(cudaMemcpy(tri_d,tri,numTri*sizeof(tri_dbl),cudaMemcpyHostToDevice));
    printf("Allocated and copied memory for triangles\n");
    
    aacb3d *cb_d;
    CUDA_CALL(cudaMalloc(&cb_d,numCb*sizeof(aacb3d)));
    CUDA_CALL(cudaMemcpy(cb_d,cb,numCb*sizeof(aacb3d),cudaMemcpyHostToDevice));
    printf("Allocated and copied memory for cubes\n");
    
    memset(flag,0,numCb*sizeof(int));
    printf("Initialized flag\n");
    int *flag_d;
    CUDA_CALL(cudaMalloc(&flag_d,numCb*sizeof(int)));
    CUDA_CALL(cudaMemcpy(flag_d,flag,numCb*sizeof(int),cudaMemcpyHostToDevice));
    
    printf("Device memory allocated.\n");
    
    int xNumBlocks, xWidth = 1, yNumBlocks, yWidth = 1;
    xNumBlocks = (numTri+xWidth-1)/xWidth;
    yNumBlocks = (numCb+yWidth-1)/yWidth;
    
    dim3 gridLayout, blockLayout;
    gridLayout.x = xNumBlocks;
    gridLayout.y = yNumBlocks;
    
    blockLayout.x = xWidth;
    blockLayout.y = yWidth;
    
    testTriCbInt<<<gridLayout,blockLayout>>>(tri_d,numTri,cb_d,numCb,flag_d);
    HOST_CALL(cudaMemcpy(flag,flag_d,numCb*sizeof(int),cudaMemcpyDeviceToHost));
    
    CUDA_CALL(cudaFree(flag_d));
    CUDA_CALL(cudaFree(cb_d));
    CUDA_CALL(cudaFree(tri_d));
    
    for(int i=0;i<numCb;i++) {
        if(flag[i]!=0) {
            flag[i] = 1;
        }
    }
    
    return EXIT_SUCCESS;
}

void reorgGrid_zyx2xyz(int* grid, const int l)
{
    /*re-organize the voxel grids from the order of significance of z, y, x to x, y, z*/
    int totalNum = pow(8,l), dimNum = pow(2,l);
    int *temp = (int*)malloc(totalNum*sizeof(int));
    memcpy(temp,grid,totalNum*sizeof(int));
    
    // reorganize
    for(int x=0;x<dimNum;x++) {
        for(int y=0;y<dimNum;y++) {
            for(int z=0;z<dimNum;z++) {
                int idx_old = x*dimNum*dimNum+y*dimNum+z;
                int idx_new = z*dimNum*dimNum+y*dimNum+x;
                grid[idx_new] = temp[idx_old];
            }
        }
    }
    free(temp);
}

__host__ __device__ void printCube(const aacb3d cb)
{
    printf("corner: (%lf,%lf,%lf), length: %lf\n",cb.cnr.coords[0],cb.cnr.coords[1],
            cb.cnr.coords[2],cb.len);
}

__host__ __device__ void printTriangle(const tri_dbl tri)
{
    printf("nodes: (%lf,%lf,%lf), (%lf,%lf,%lf), (%lf,%lf,%lf)\n",
            tri.nod[0].coords[0],tri.nod[0].coords[1],tri.nod[0].coords[2],
            tri.nod[1].coords[0],tri.nod[1].coords[1],tri.nod[1].coords[2],
            tri.nod[2].coords[0],tri.nod[2].coords[1],tri.nod[2].coords[2]);
}

__host__ __device__ void PrintVec(const vec2d* vec, const int num)
{
    for(int i=0;i<num;i++) {
        printf("(%lf,%lf)\n",vec[i].coords[0],vec[i].coords[1]);
    }
}

__host__ __device__ void PrintVec(const vec2f* vec, const int num)
{
    for(int i=0;i<num;i++) {
        printf("(%f,%f)\n",vec[i].coords[0],vec[i].coords[1]);
    }
}

__host__ int voxelSpace(const aacb3d sp, const int numEachDim, const vec3d* pt, 
        const tri_elem* elem, const int numElem, int* flag)
{
    /*voxelize the a space of objects composed of triangles
     sp: a cube representing the whole space
     pt: an array of points
     numPt: the number of points
     elem: an array of elements
     numElem: the number of elements
     flag: an array of flags
     octLevel: the level of the octree*/
    
    printf("Entered voxSpace.\n");
    // save all the triangles in a triangle array
    tri_dbl *tri = (tri_dbl*)malloc(numElem*sizeof(tri_dbl));
    for(int i=0;i<numElem;i++) {
        for(int j=0;j<3;j++) {
            tri[i].nod[j] = pt[elem[i].nod[j]];
        }
    }
    printf("Initialized triangles.\n");
    //for(int i=0;i<numElem;i++) {
    //    printf("Current triangle: (%lf,%lf,%f), (%lf,%lf,%lf), (%lf,%lf,%f)\n",
    //            tri[i].nod[0].coords[0],tri[i].nod[0].coords[1],tri[i].nod[0].coords[2],
    //            tri[i].nod[1].coords[0],tri[i].nod[1].coords[1],tri[i].nod[1].coords[2],
    //            tri[i].nod[2].coords[0],tri[i].nod[2].coords[1],tri[i].nod[2].coords[2]);
    //}
    // save all the unit boxes in a cube array
    int numVox = numEachDim*numEachDim*numEachDim;
    memset(flag,0,numVox*sizeof(int));
    
    aacb3d *cb = (aacb3d*)malloc(numVox*sizeof(aacb3d));
    double unitLen = sp.len/numEachDim;
    vec3d dir_x = {unitLen,0,0}, dir_y = {0,unitLen,0}, dir_z = {0,0,unitLen}, 
            xOffset, yOffset, zOffset;
    int idx, rel;
    for(int i=0;i<numEachDim;i++) {
        // z dimension
        zOffset = scaVecMul(i,dir_z);
        for(int j=0;j<numEachDim;j++) {
            // y dimension
            yOffset = scaVecMul(j,dir_y);
            for(int k=0;k<numEachDim;k++) {
                // x dimension
                xOffset = scaVecMul(k,dir_x);
                idx = i*(numEachDim*numEachDim)+j*numEachDim+k;
                cb[idx].cnr = vecAdd(vecAdd(vecAdd(sp.cnr,xOffset),yOffset),zOffset);
                cb[idx].len = unitLen;
            }
        }
    }
    for(int i=0;i<numVox;i++) {
        for(int j=0;j<numElem;j++) {
            rel = deterTriCubeInt(tri[j],cb[i]);
            if(rel==1) {
                flag[i] = 1;
                break;
            }
        }
    }
    free(cb);
    free(tri);
    return EXIT_SUCCESS;
}

__host__ int write_voxels(const int* flag, const int numEachDim, const char* file_path)
{
    FILE *file = fopen(file_path,"w");
    if(file==NULL) {
        printf("Failed to open file.\n");
        return EXIT_FAILURE;
    }
    else {
        int status;
        for(int i=0;i<numEachDim*numEachDim*numEachDim;i++) {
            status = fprintf(file,"%d ",flag[i]);
            if((i+1)%numEachDim == 0) {
                status = fprintf(file,"\n");
            }
            if((i+1)%(numEachDim*numEachDim) == 0) {
                status = fprintf(file,"\n");
            }
            if(status<0) {
                printf("Failed to write the %dth line to file\n",i);
                return EXIT_FAILURE;
            }
        }
        fclose(file);
        return EXIT_SUCCESS;
    }
}

__host__ __device__ vec2d GetMin(const aarect2d rect)
{
    return rect.cnr;
}

__host__ __device__ vec2d GetMax(const aarect2d rect)
{
    vec2d dir_x = {1,0}, dir_y = {0,1};
    vec2d nod = vecAdd(vecAdd(rect.cnr,scaVecMul(rect.len[0],dir_x)),scaVecMul(rect.len[1],dir_y));
    return nod;
}

__host__ __device__ bool IntvlIntvlOvlp(const intvl2d intvl1, const intvl2d intvl2)
{
    /*returns true if the two intervals overlap and false if not*/
    if(intvl1.min<=intvl2.max && intvl2.min<=intvl1.max) {
        return true;
    }
    else {
        return false;
    }
}

__host__ __device__ bool AaRectAaRectOvlp(const aarect2d rect1, const aarect2d rect2)
{
    /*determines if two axis-aligned rectangles overlap
     rect1: the first rectangle
     rect2: the second rectangle
     return: 
     true: the two rectangles overlap
     false: the two rectangles do not overlap*/
    // first check if the projections on the x axis overlap
    vec2d minNod1 = GetMin(rect1), maxNod1 = GetMax(rect1), minNod2 = GetMin(rect2), 
            maxNod2 = GetMax(rect2);
    intvl2d intvl1x = {minNod1.coords[0],maxNod1.coords[0]}, intvl2x = {minNod2.coords[0],maxNod2.coords[0]},
            intvl1y = {minNod1.coords[1],maxNod1.coords[1]}, intvl2y = {minNod2.coords[1],maxNod2.coords[1]};
    
    if(IntvlIntvlOvlp(intvl1x,intvl2x) && IntvlIntvlOvlp(intvl1y,intvl2y)) {
        return true;
    }
    else {
        return false;
    }
}

__host__ __device__ vec3d GetMin(const aarect3d& rect)
{
    return rect.cnr;
}

#ifdef __CUDA_ARCH__

vec3d GetMax(const aarect3d& rect)
{
    vec3d cnr_max = rect.cnr;
    for(int i=0;i<3;i++) {
        cnr_max = vecAdd(cnr_max,scaVecMul(rect.len[i],BASES[i]));
    }
    return cnr_max;
}

#else

vec3d GetMax(const aarect3d& rect)
{   
    vec3d cnr_max = rect.cnr;
    for(int i=0;i<3;i++) {
        cnr_max = vecAdd(cnr_max,scaVecMul(rect.len[i],bases[i]));
    }
    return cnr_max;
}

#endif

__host__ __device__ intvl3d GetInterval(const aarect3d& rect, const vec3d& axis)
{
    vec3d cnrs[2], vertex;
    cnrs[0] = GetMin(rect);
    cnrs[1] = GetMax(rect);
    intvl3d intvl;
    double projection;
    intvl.max = -DBL_MAX;
    intvl.min = DBL_MAX;
    for(int i=0;i<2;i++) {
        vertex.coords[0] = cnrs[i].coords[0];
        for(int j=0;j<2;j++) {
            vertex.coords[1] = cnrs[j].coords[1];
            for(int k=0;k<2;k++) {
                vertex.coords[2] = cnrs[k].coords[2];
                //printVec(&vertex,1);
                projection = vecDotMul(vertex,axis);
                intvl.max = (projection>intvl.max) ? projection : intvl.max;
                intvl.min = (projection<intvl.min) ? projection : intvl.min;
            }
        }
    }
    return intvl;
}

__host__ __device__ bool IntvlIntvlOvlp(const intvl3d& intvl1, const intvl3d& intvl2)
{
    /*returns true if the two intervals overlap and false if not*/
    if(intvl1.min<=intvl2.max && intvl2.min<=intvl1.max) {
        return true;
    }
    else {
        return false;
    }
}

__host__ __device__ intvl3d GetInterval(const tri3d& tri, const vec3d& ax)
{
    intvl3d intvl;
    intvl.min = DBL_MAX;
    intvl.max = -DBL_MAX;
    double projection;
    
    for(int i=0;i<3;i++) {
        projection = vecDotMul(tri.nod[i],ax);
        intvl.max = (projection>intvl.max) ? projection : intvl.max;
        intvl.min = (projection<intvl.min) ? projection : intvl.min;
    }
    
    return intvl;
}

__host__ __device__ bool OverlapOnAxis(const tri3d& tri, const aarect3d& rect, const vec3d& ax)
{
    intvl3d intvl_tri, intvl_rect;
    intvl_tri = GetInterval(tri,ax);
    intvl_rect = GetInterval(rect,ax);
    
    return IntvlIntvlOvlp(intvl_tri,intvl_rect);
}

__host__ __device__ bool OverlapTriangleAARect(const tri3d& tri, const aarect3d& rect)
{
    return true;
}
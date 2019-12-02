/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
#include "geometry.h"
#include <stdio.h>
#include <stdlib.h>

__host__ __device__ int deterPtPlaneRel(const rect_coord_dbl pt, const plane_dbl plane)
{
    rect_coord_dbl vec = rectCoordSub(pt,plane.pt);
    double result = rectDotMul(plane.n,vec);
    if(result>=0) {
        // on the positive side of the plane normal
        return 1;
    } else {
        return 0;
    }
}

__host__ __device__ int deterPtCubeEdgeVolRel(const rect_coord_dbl pt, const cube_dbl cb)
{
    /*determine the relationship between a point and the volume bounded by edge faces
     of a cube*/
    
    //declare two rect_coord_dbl arrays for nod at the bottom and the top face 
    rect_coord_dbl btm[4], top[4], left[4], right[4], back[4], front[4];
    
    // declare the basis unit vectors
    rect_coord_dbl dir_x = {1.0,0.0,0.0}, dir_y = {0.0,1.0,0.0}, dir_z = {0.0,0.0,1.0};
    
    //set up btm and top nod
    btm[0] = cb.cnr;
    btm[1] = rectCoordAdd(btm[0],scaRectMul(cb.len,dir_x));
    btm[2] = rectCoordAdd(btm[1],scaRectMul(cb.len,dir_y));
    btm[3] = rectCoordAdd(btm[0],scaRectMul(cb.len,dir_y));
    //printRectCoord(btm,4);
    
    top[0] = rectCoordAdd(btm[0],scaRectMul(cb.len,dir_z));
    top[1] = rectCoordAdd(top[0],scaRectMul(cb.len,dir_x));
    top[2] = rectCoordAdd(top[1],scaRectMul(cb.len,dir_y));
    top[3] = rectCoordAdd(top[0],scaRectMul(cb.len,dir_y));
    //printRectCoord(top,4);
    
    //set up left and right nod
    left[0] = cb.cnr;
    left[1] = rectCoordAdd(left[0],scaRectMul(cb.len,dir_x));
    left[2] = rectCoordAdd(left[1],scaRectMul(cb.len,dir_z));
    left[3] = rectCoordAdd(left[0],scaRectMul(cb.len,dir_z));
    //printRectCoord(left,4);
    
    right[0] = rectCoordAdd(left[0],scaRectMul(cb.len,dir_y));
    right[1] = rectCoordAdd(right[0],scaRectMul(cb.len,dir_x));
    right[2] = rectCoordAdd(right[1],scaRectMul(cb.len,dir_z));
    right[3] = rectCoordAdd(right[0],scaRectMul(cb.len,dir_z));
    //printRectCoord(right,4);
    
    //set up back and front nod
    back[0] = cb.cnr;
    back[1] = rectCoordAdd(back[0],scaRectMul(cb.len,dir_y));
    back[2] = rectCoordAdd(back[1],scaRectMul(cb.len,dir_z));
    back[3] = rectCoordAdd(back[0],scaRectMul(cb.len,dir_z));
    //printRectCoord(back,4);
    
    front[0] = rectCoordAdd(back[0],scaRectMul(cb.len,dir_x));
    front[1] = rectCoordAdd(front[0],scaRectMul(cb.len,dir_y));
    front[2] = rectCoordAdd(front[1],scaRectMul(cb.len,dir_z));
    front[3] = rectCoordAdd(front[0],scaRectMul(cb.len,dir_z));
    //printRectCoord(front,4);
    
    //declare an array nrml for determining the normal of the new plane
    rect_coord_dbl nrml[3];
    plane_dbl plane;
    int result;
    
    //deal with the bottom face
    for(int i=0;i<3;i++) {
        nrml[1] = dir_z;
        switch(i) {
            case 0: // edge determined by btm[0] and btm[1]
                nrml[0] = dir_y;
                break;
            case 1: // edge determined by btm[1] and btm[2]
                nrml[0] = scaRectMul(-1,dir_x);
                break;
            case 2: // edge determined by btm[2] and btm[3]
                nrml[0] = scaRectMul(-1,dir_y);
                break;
            case 3: // edge determined by btm[3] and btm[0]
                nrml[0] = dir_x;
                break;
            default:
                printf("Entered the wrong branch.\n");
        }
        nrml[3] = nrmlzRectCoord(rectCoordAdd(nrml[0],nrml[1]));
        plane.n = nrml[3];
        plane.pt = btm[i];
        result = deterPtPlaneRel(pt,plane);
        if(result == 0) {
            printf("bottom face: %dth node\n",i);
            return 0;
        }
    }
    
    //deal with the top face
    for(int i=0;i<3;i++) {
        nrml[1] = scaRectMul(-1,dir_z);
        switch(i) {
            case 0: // edge determined by top[0] and top[1]
                nrml[0] = dir_y;
                break;
            case 1: // edge determined by top[1] and top[2]
                nrml[0] = scaRectMul(-1,dir_x);
                break;
            case 2: // edge determined by top[2] and top[3]
                nrml[0] = scaRectMul(-1,dir_y);
                break;
            case 3: // edge determined by top[3] and top[0]
                nrml[0] = dir_x;
                break;
            default:
                printf("Entered the wrong branch.\n");
        }
        nrml[3] = nrmlzRectCoord(rectCoordAdd(nrml[0],nrml[1]));
        plane.n = nrml[3];
        plane.pt = top[i];
        result = deterPtPlaneRel(pt,plane);
        if(result == 0) {
            printf("top face: %dth node\n",i);
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
                nrml[0] = scaRectMul(-1,dir_x);
                break;
            case 2: // edge determined by left[2] and left[3]
                nrml[0] = scaRectMul(-1,dir_z);
                break;
            case 3: // edge determined by left[3] and left[0]
                nrml[0] = dir_x;
                break;
            default:
                printf("Entered the wrong branch.\n");
        }
        nrml[3] = nrmlzRectCoord(rectCoordAdd(nrml[0],nrml[1]));
        plane.n = nrml[3];
        plane.pt = left[i];
        result = deterPtPlaneRel(pt,plane);
        if(result == 0) {
            printf("left face: %dth node\n",i);
            return 0;
        }
    }
    
    //deal with the right face
    for(int i=0;i<3;i++) {
        nrml[1] = scaRectMul(-1,dir_y);
        switch(i) {
            case 0: // edge determined by right[0] and right[1]
                nrml[0] = dir_z;
                break;
            case 1: // edge determined by right[1] and right[2]
                nrml[0] = scaRectMul(-1,dir_x);
                break;
            case 2: // edge determined by right[2] and right[3]
                nrml[0] = scaRectMul(-1,dir_z);
                break;
            case 3: // edge determined by btm[3] and btm[0]
                nrml[0] = dir_x;
                break;
            default:
                printf("Entered the wrong branch.\n");
        }
        nrml[3] = nrmlzRectCoord(rectCoordAdd(nrml[0],nrml[1]));
        plane.n = nrml[3];
        plane.pt = right[i];
        result = deterPtPlaneRel(pt,plane);
        if(result == 0) {
            printf("right face: %dth node\n",i);
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
                nrml[0] = scaRectMul(-1,dir_y);
                break;
            case 2: // edge determined by btm[2] and btm[3]
                nrml[0] = scaRectMul(-1,dir_z);
                break;
            case 3: // edge determined by btm[3] and btm[0]
                nrml[0] = dir_y;
                break;
            default:
                printf("Entered the wrong branch.\n");
        }
        nrml[3] = nrmlzRectCoord(rectCoordAdd(nrml[0],nrml[1]));
        plane.n = nrml[3];
        plane.pt = back[i];
        result = deterPtPlaneRel(pt,plane);
        if(result == 0) {
            printf("back face: %dth node\n",i);
            return 0;
        }
    }
    
    //deal with the front face
    for(int i=0;i<3;i++) {
        nrml[1] = scaRectMul(-1,dir_x);
        switch(i) {
            case 0: // edge determined by front[0] and front[1]
                nrml[0] = dir_z;
                break;
            case 1: // edge determined by front[1] and front[2]
                nrml[0] = scaRectMul(-1,dir_y);
                break;
            case 2: // edge determined by front[2] and front[3]
                nrml[0] = scaRectMul(-1,dir_z);
                break;
            case 3: // edge determined by front[3] and front[0]
                nrml[0] = dir_y;
                break;
            default:
                printf("Entered the wrong branch.\n");
        }
        nrml[3] = nrmlzRectCoord(rectCoordAdd(nrml[0],nrml[1]));
        plane.n = nrml[3];
        plane.pt = front[i];
        result = deterPtPlaneRel(pt,plane);
        if(result == 0) {
            printf("front face: %dth node\n",i);
            return 0;
        }
    }
    
    // if not returned 0, then return 1, the point is inside the volume
    return 1;
}

__host__ __device__ int deterPtCubeVtxVolRel(const rect_coord_dbl pt, const cube_dbl cb)
{
    // declare the basis unit vectors
    rect_coord_dbl dir_x = {1.0,0.0,0.0}, dir_y = {0.0,1.0,0.0}, dir_z = {0.0,0.0,1.0}, 
            nrml[4], tempPt;
    plane_dbl plane;
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
                tempPt = rectCoordAdd(cb.cnr,scaRectMul(cb.len,dir_x));
                nrml[0] = scaRectMul(-1,dir_x);
                nrml[1] = dir_y;
                nrml[2] = dir_z;
                break;
            case 2: //the third vertex
                tempPt = rectCoordAdd(rectCoordAdd(cb.cnr,scaRectMul(cb.len,dir_x)),scaRectMul(cb.len,dir_y));
                nrml[0] = scaRectMul(-1,dir_x);
                nrml[1] = scaRectMul(-1,dir_y);
                nrml[2] = dir_z;
                break;
            case 3: //the fourth vertex
                tempPt = rectCoordAdd(cb.cnr,scaRectMul(cb.len,dir_y));
                nrml[0] = dir_x;
                nrml[1] = scaRectMul(-1,dir_y);
                nrml[2] = dir_z;
                break;
            case 4: //the fifth vertex
                tempPt = rectCoordAdd(cb.cnr,scaRectMul(cb.len,dir_z));
                nrml[0] = dir_x;
                nrml[1] = dir_y;
                nrml[2] = scaRectMul(-1,dir_z);
                break;
            case 5: //the sixth vertex
                tempPt = rectCoordAdd(rectCoordAdd(cb.cnr,scaRectMul(cb.len,dir_z)),scaRectMul(cb.len,dir_x));
                nrml[0] = scaRectMul(-1,dir_x);
                nrml[1] = dir_y;
                nrml[2] = scaRectMul(-1,dir_z);
                break;
            case 6: //the seventh vertex
                tempPt = rectCoordAdd(rectCoordAdd(rectCoordAdd(cb.cnr,scaRectMul(cb.len,dir_z)),
                        scaRectMul(cb.len,dir_x)),scaRectMul(cb.len,dir_y));
                nrml[0] = scaRectMul(-1,dir_x);
                nrml[1] = scaRectMul(-1,dir_y);
                nrml[2] = scaRectMul(-1,dir_z);
                break;
            case 7: //the eighth vertex
                tempPt = rectCoordAdd(rectCoordAdd(cb.cnr,scaRectMul(cb.len,dir_z)),scaRectMul(cb.len,dir_y));
                nrml[0] = dir_x;
                nrml[1] = scaRectMul(-1,dir_y);
                nrml[2] = scaRectMul(-1,dir_z);
                break;
            default:
                printf("safety purpose.\n");
        }
        nrml[3] = nrmlzRectCoord(rectCoordAdd(rectCoordAdd(nrml[0],nrml[1]),nrml[2]));
        plane.n = nrml[3];
        plane.pt = tempPt;
        result = deterPtPlaneRel(pt,plane);
        if(result == 0) {
            return 0;
        }
    }
    return 1;
}

__host__ __device__ int deterPtCubeRel(const rect_coord_dbl pt, const cube_dbl cube)
{
    rect_coord_dbl cnr_fru = cube.cnr;
    cnr_fru = rectCoordAdd(cnr_fru,scaRectMul(cube.len,{1,0,0}));
    cnr_fru = rectCoordAdd(cnr_fru,scaRectMul(cube.len,{0,1,0}));
    cnr_fru = rectCoordAdd(cnr_fru,scaRectMul(cube.len,{0,0,1}));
    double x_min = cube.cnr.coords[0], y_min = cube.cnr.coords[1], z_min = cube.cnr.coords[2], 
            x_max = cnr_fru.coords[0], y_max = cnr_fru.coords[1], z_max = cnr_fru.coords[2],
            x = pt.coords[0], y = pt.coords[1], z = pt.coords[2];
    if(x >= x_min && x<= x_max && y >= y_min && y<= y_max && z >= z_min && z<= z_max) {
        return 1;
    } else {
        return 0;
    }
}

__host__ __device__ int deterLinePlaneRel(const line_dbl ln, const plane_dbl pln, double* t)
{
    if(abs(rectDotMul(ln.dir,pln.n))<EPS) {
        //line parallel to plane
        if(abs(rectDotMul(pln.n,rectCoordSub(ln.pt,pln.pt)))<EPS) {
            return 2;
        } else {
            return 0;
        }
    } else {
        double temp = rectDotMul(pln.n,rectCoordSub(pln.pt,ln.pt))/rectDotMul(pln.n,ln.dir);
        *t = temp;
        return 1;
    }
}

__host__ __device__ double triArea(const tri_dbl s)
{
    rect_coord_dbl vec[2];
    vec[0] = rectCoordSub(s.nod[1],s.nod[0]);
    vec[1] = rectCoordSub(s.nod[2],s.nod[0]);
    return 0.5*rectNorm(rectCrossMul(vec[0],vec[1]));
}

__host__ __device__ double quadArea(const quad_dbl s)
{
    rect_coord_dbl vec[2];
    vec[0] = rectCoordSub(s.nod[1],s.nod[0]);
    vec[1] = rectCoordSub(s.nod[2],s.nod[0]);
    return rectNorm(rectCrossMul(vec[0],vec[1]));
}

__host__ __device__ plane_dbl quad2plane(const quad_dbl qd)
{
    /*get the plane containing a quad*/
    plane_dbl pln;
    pln.pt = qd.nod[0];
    rect_coord_dbl vec[2];
    vec[0] = rectCoordSub(qd.nod[1],qd.nod[0]);
    vec[1] = rectCoordSub(qd.nod[2],qd.nod[0]);
    pln.n = nrmlzRectCoord(rectCrossMul(vec[0],vec[1]));
    return pln;
}

__host__ __device__ line_dbl lnSeg2ln(const ln_seg_dbl ls)
{
    line_dbl l;
    l.pt = ls.nod[0];
    l.dir = rectCoordSub(ls.nod[1],ls.nod[0]);
    return l;
}

__host__ __device__ int deterPtQuadRel(const rect_coord_dbl pt, const quad_dbl qd)
{
    /*determine the relationship between a point and a quad on the same plane*/
    double area = 0.0;
    rect_coord_dbl vec[2];
    for(int i=0;i<4;i++) {
        vec[0] = rectCoordSub(qd.nod[i%4],pt);
        vec[1] = rectCoordSub(qd.nod[(i+1)%4],pt);
        area += 0.5*rectNorm(rectCrossMul(vec[0],vec[1]));
    }
    double area_quad = quadArea(qd);
    if(abs(area-area_quad)<EPS) {
        return 1; // in
    } else {
        return 0; // out
    }
}

__host__ __device__ double rectCoordDet(const rect_coord_dbl vec[3])
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
    if(abs(rectNorm(rectCrossMul(ln1.dir,ln2.dir)))<EPS) {
        // the two lines are either parallel or the same line
        
        // check if a point on line 1 is on line 2
        rect_coord_dbl vec = rectCoordSub(ln1.pt,ln2.pt);
        if(rectNorm(vec)<EPS) {
            //the points are the same
            return 2; 
        } 
        else {
            if(rectNorm(rectCrossMul(vec,ln2.dir))<EPS) {
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
        rect_coord_dbl pt[4];
        pt[0] = ln1.pt;
        pt[1] = rectCoordAdd(ln1.pt,scaRectMul(1.0,ln1.dir));
        pt[2] = ln2.pt;
        pt[3] = rectCoordAdd(ln2.pt,scaRectMul(1.0,ln2.dir));
        //printRectCoord(pt,4);
        if(rectCoordEqual(pt[0],pt[2]) || rectCoordEqual(pt[0],pt[3]) || 
                rectCoordEqual(pt[1],pt[2]) || rectCoordEqual(pt[1],pt[3])) {
            //the two points on the line is the same point
            if(rectCoordEqual(pt[0],pt[2])) {
                *t1 = 0;
                *t2 = 0;
            } 
            else {
                if(rectCoordEqual(pt[0],pt[3])) {
                    *t1 = 0;
                    *t2 = 1.0;
                } 
                else {
                    if(rectCoordEqual(pt[1],pt[2])) {
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
            rect_coord_dbl vec[3];
            vec[0] = rectCoordSub(pt[1],pt[0]);
            vec[1] = rectCoordSub(pt[2],pt[0]);
            vec[2] = rectCoordSub(pt[3],pt[0]);
            
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

__host__ __device__ int deterPtLnRel(const rect_coord_dbl pt, const line_dbl ln)
{
    /*determines the relation between a point and a line*/
    rect_coord_dbl vec = rectCoordSub(pt,ln.pt);
    if(rectNorm(rectCrossMul(vec,ln.dir))<EPS) {
        return 1;
    } 
    else {
        return 0;
    }
}

__host__ __device__ int deterPtLnSegRel(const rect_coord_dbl pt, const ln_seg_dbl lnSeg)
{
    /*determines the relation between a point and a line segment*/
    line_dbl ln = lnSeg2ln(lnSeg);
    if(deterPtLnRel(pt,ln)==0) {
        //point not on the line containing the line segment
        return 0;
    } 
    else {
        double t;
        rect_coord_dbl vec = rectCoordSub(pt,ln.pt);
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

__host__ __device__ int deterLnSegLnSegRel(const ln_seg_dbl seg1, const ln_seg_dbl seg2)
{
    /*determines the relation between two line segments
     0: no intersection
     1: intersection*/
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
                        if(rectCoordEqual(seg1.nod[i],seg2.nod[j])) {
                            rect_coord_dbl vec[2];
                            vec[0] = rectCoordSub(seg1.nod[(i+1)%2],seg1.nod[i]);
                            vec[1] = rectCoordSub(seg2.nod[(j+1)%2],seg1.nod[j]);
                            if(rectDotMul(vec[0],vec[1])<0) {
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

__host__ __device__ int deterLnSegQuadRel(const ln_seg_dbl lnSeg, const quad_dbl qd)
{
    /*determine if a line segment intersects a quad*/
    int flag;
    
    //make a line containing the line segment    
    line_dbl ln = lnSeg2ln(lnSeg);
    
    // define a plane containing the quad
    plane_dbl pln = quad2plane(qd);
    
    // determine the intersection between the line and the plane
    double t;
    flag = deterLinePlaneRel(ln,pln,&t);
    if(flag==0) {
        // no intersection between the line and the plane
        return 0;
    } 
    else {
        if(flag==2) {
            // infinitely many intersections
            if(deterPtQuadRel(lnSeg.nod[0],qd)==1 || deterPtQuadRel(lnSeg.nod[1],qd)==1) {
                //oen of the nodes is within the quad
                return 1;
            } 
            else {
                // none of the nodes is within the quad, test if segments intersect
                for(int i=0;i<4;i++) {
                    ln_seg_dbl qdLnSeg;
                    qdLnSeg.nod[0] = qd.nod[i%4];
                    qdLnSeg.nod[1] = qd.nod[(i+1)%4];
                    line_dbl qdLn = lnSeg2ln(qdLnSeg);
                    double t1, t2;
                    int rel = deterLnLnRel(ln,qdLn,&t1,&t2);
                    if(rel==0) {
                        //lines are skew to each other
                        return 0;
                    } 
                    else {
                        if(rel==1) {
                            //lines have a single intersection
                            if(t1>=0 && t1<=1 && t2>=0 && t2<=1) {
                                //there exists a single intersection
                                return 1;
                            } 
                            else {
                                return 0;
                            }
                        }
                        else {
                            if(deterPtLnSegRel(lnSeg.nod[0],qdLnSeg)==1 || 
                                    deterPtLnSegRel(lnSeg.nod[1],qdLnSeg)==1) {
                                return 1;
                            }
                            else {
                                return 0;
                            }
                        }
                    }
                }
            }
        } 
        else {
            //determines if a point is within a quad
            if(t<0 || t>1) {
                return 0;
            } 
            else {
                rect_coord_dbl intersection = rectCoordAdd(ln.pt,scaRectMul(t,ln.dir));
                if(deterPtQuadRel(intersection,qd)==1) {
                    return 1;
                } 
                else {
                    return  0;
                }
            }
            
        }
    }
    
    return 1;
}




__host__ __device__ int deterTriCubeInt(const rect_coord_dbl nod[3], const cube_dbl cb)
{
    /*this function determines if a triangle intersects with a cube*/
    return 1;
}


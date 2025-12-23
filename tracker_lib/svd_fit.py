#!/usr/bin/python
import os
import math
import subprocess
import array
import numpy as np
import ROOT
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from scipy.optimize import curve_fit

from tracker_lib import utils


### similar to https://stackoverflow.com/questions/2298390/fitting-a-line-in-3d

'''
SVD points = [  [vtx.x,  vtx.y,  vtx.z],
                [cls0.x, cls0.y, cls0.z],
                [cls1.x, cls1.y, cls1.z],
                [cls2.x, cls2.y, cls2.z],
                [cls3.x, cls3.y, cls3.z],
                ...  ]
'''
def calculateSVDchi2(points, errors, direction, centroid):
    r1,r2 = utils.r1r2(direction, centroid)
    x  = points[:,0]
    y  = points[:,1]
    z  = points[:,2]
    ex = errors[:,0]
    ey = errors[:,1]
    ## There are four independent parameters: a 2D offset(x0,y0) and slope (dx,dy).
    ## Each point (hit / vertex) provides two measurements.
    ## ndof = 2*num_points - N_pars = 2*(4 or 5) - 4
    ndof = 2*len(points)-4
    chisq = 0
    for i in range(len(z)):
        xonline,yonline = utils.xyofz(r1,r2,z[i])
        dx = xonline-x[i]
        dy = yonline-y[i]
        chisq += (dx/ex[i])**2 + (dy/ex[i])**2 if(ex[i]>0 and ey[i]>0) else 1e10
    return chisq,ndof


def fit_line_3d_SVD(points):
    """
    Fit a straight line to a set of 3D points using the least-squares method.
    similar to https://stackoverflow.com/questions/2298390/fitting-a-line-in-3d
    """
    # Find the centroid of the points
    centroid = np.mean(points, axis=0)
    # Compute the matrix A
    A = points - centroid
    # Compute the singular value decomposition of A
    U, s, Vt = np.linalg.svd(A)
    # The direction of the line is given by the first column of Vt
    direction = Vt[0]
    # The point on the line that is closest to the centroid is given by the centroid itself
    centroid
    return direction,centroid,s


def fit_3d_SVD(points,errors):
    # Fit a straight line to the points
    direction, centroid, s = fit_line_3d_SVD(points)
    # point = points[0] ### just the ploint where the line comes from
    point = centroid
    # goodness = s[1]
    chisq,ndof = calculateSVDchi2(points,errors,direction,centroid)
    return chisq,ndof,direction,centroid
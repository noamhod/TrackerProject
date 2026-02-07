#!/usr/bin/python
import os
import math
import subprocess
import time
import array
import numpy as np
import ROOT
from scipy.optimize import minimize
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from scipy.optimize import curve_fit
from iminuit import Minuit


from tracker_lib import config, utils


def chi2_vectorized(params, x, y, z, ex, ey, ez):
    ### params: [x0, mx, y0, my], representing: x = x0 + mx*t, y = y0 + my*t, z = t
    ### point on line at z=0 and z=1, direction vector u = (mx, my, 1)
    ux, uy, uz = params[1], params[3], 1.0
    mag_u = np.sqrt(ux**2 + uy**2 + uz**2)
    ### unit direction
    ux /= mag_u
    uy /= mag_u
    uz /= mag_u
    ### vectors from point on line (x0, y0, 0) to data points (xi, yi, zi)
    dx = x - params[0]
    dy = y - params[2]
    dz = z - 0.0
    ### cross product v = (dx, dy, dz) X (ux, uy, uz)
    vx = dy * uz - dz * uy
    vy = dz * ux - dx * uz
    # vz = dx * uy - dy * ux # Usually not needed if errors are only in XY
    ### weighted chi2
    return np.sum((vx / ex)**2 + (vy / ey)**2)



def fit_line_3d_chi2err(x, y, z, ex, ey, ez, guess=None):
    initial_params = [1.0, 0.0, 0.0, 0.0] if guess is None else guess
    ### Define the cost function for Minuit (it looks for function signatures, so we use a wrapper)
    def cost_func(x0, mx, y0, my):
        params = [x0, mx, y0, my]
        return chi2_vectorized(params, x, y, z, ex, ey, ez)
    ### Initialize Minuit (name the parameters for better readability in output)
    m = Minuit(cost_func, x0=initial_params[0], mx=initial_params[1], y0=initial_params[2], my=initial_params[3])
    ### set errors (step sizes) for the numerical gradient
    # m.errors = [0.1, 0.1, 0.1, 0.1]
    m.errors = [abs(p) * 0.1 if p != 0 else 0.01 for p in initial_params] # Use a small fraction of the guess or a physical scale
    ### run the migrad minimizer
    m.migrad()
    ### calculate the reliable uncertainties (Hessian)
    m.hesse()
    ### extract results to match your original return structure
    params = np.array(m.values)
    parerr = np.array(m.errors) # the calculated statistical errors from Hesse
    parcov = m.covariance.tolist()
    chisq = m.fval
    ndof = 2 * len(x) - len(params)
    success = m.valid
    ### return
    return params, parerr, parcov, chisq, ndof, success



def fit_3d_chi2err(points,errors,guess=None):
    cfg = config.Config().map
    x = points[0]
    y = points[1]
    z = points[2]
    ex = errors[0]
    ey = errors[1]
    ez = errors[2]
    params,parerr,parcov,chisq,ndof,success = fit_line_3d_chi2err(x,y,z,ex,ey,ez,guess)
    ### plot the points and the fitted line
    x0,y0,z0 = utils.line(cfg["world"]["z"][0], params)
    x1,y1,z1 = utils.line(cfg["world"]["z"][1], params)
    #TODO: need to check this:
    xm,ym,zm = utils.line((cfg["world"]["z"][1]-cfg["world"]["z"][0])/2., params) #TODO
    centroid  = [xm,ym,zm]                     #TODO
    direction = [x1-x0,y1-y0,z1-z0]            #TODO
    return chisq,ndof,direction,centroid,params,parerr,parcov,success
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


from tracker_lib import config, utils


### similar to https://root.cern.ch/doc/master/line3Dfit_8C_source.html

def distance2(params, xi,yi,zi, exi,eyi,ezi):
    # distance line point is D=|(xp-x0) cross ux|
    # where ux is direction of line and x0 is a point in the line (like t=0)
    xp = ROOT.Math.XYZVector(xi,yi,zi)
    x0 = ROOT.Math.XYZVector(params[0], params[2], 0. )
    x1 = ROOT.Math.XYZVector(params[0] + params[1], params[2] + params[3], 1. )
    u  = ROOT.Math.XYZVector(x1-x0).Unit()
    v  = (xp-x0).Cross(u)
    # d2 = v.Mag2()
    d2 = v.X()*v.X()/(exi*exi) + v.Y()*v.Y()/(eyi*eyi)
    # d2 = v.X()*v.X()/(exi*exi) + v.Y()*v.Y()/(eyi*eyi) + v.Z()*v.Z()/(ezi*ezi)
    # d2 = (v.X()*v.X() + v.Y()*v.Y() + v.Z()*v.Z())/(exi*exi + eyi*eyi + ezi*ezi)
    # d2 = (v.X()*v.X() + v.Y()*v.Y())/(exi*exi + eyi*eyi)
    return d2


def fit_line_3d_chi2err(x,y,z,ex,ey,ez,guess=None):
    cfg = config.Config().map
    ### Define the objective function to minimize (the chi^2 function)
    ### similar to https://root.cern.ch/doc/master/line3Dfit_8C_source.html
    def chi2(params, x,y,z, ex,ey,ez):
        sum = 0
        for i in range(len(x)):
            sum += distance2(params, x[i],y[i],z[i], ex[i],ey[i],ez[i])
        return sum
    ### Perform the chi^2 fit using minimize
    ### https://stackoverflow.com/questions/24767191/scipy-is-not-optimizing-and-returns-desired-error-not-necessarily-achieved-due
    initial_params = [1,0,0,0] if(guess==None) else guess
    result0 = None
    result1 = None
    if(cfg["fit_chi2_fast"]):
        result1 = minimize(chi2, initial_params, method=cfg["fit_chi2_method0"],      args=(x,y,z, ex,ey,ez))
    else:
        result0 = minimize(chi2, initial_params, method=cfg["fit_chi2_method1"][0], args=(x,y,z, ex,ey,ez)) ### first fit to get closer
        result1 = minimize(chi2, result0.x,      method=cfg["fit_chi2_method0"][1], args=(x,y,z, ex,ey,ez)) ### second fit to finish
    ### get the chi^2 value and the number of degrees of freedom
    chisq = result1.fun
    ndof = 2*len(x) - len(initial_params)
    params  = result1.x
    success = result1.success
    # status  = result1.status
    # message = result1.message
    # print(success,status,message)
    return params,chisq,ndof,success


def fit_3d_chi2err(points,errors,guess=None):
    cfg = config.Config().map
    x = points[0]
    y = points[1]
    z = points[2]
    ex = errors[0]
    ey = errors[1]
    ez = errors[2]
    params,chisq,ndof,success = fit_line_3d_chi2err(x,y,z,ex,ey,ez,guess)
    # Plot the points and the fitted line
    x0,y0,z0 = utils.line(cfg["world"]["z"][0], params)
    x1,y1,z1 = utils.line(cfg["world"]["z"][1], params)
    #TODO: need to check this:
    xm,ym,zm = utils.line((cfg["world"]["z"][1]-cfg["world"]["z"][0])/2., params) #TODO
    centroid  = [xm,ym,zm]                     #TODO
    direction = [x1-x0,y1-y0,z1-z0]            #TODO
    return chisq,ndof,direction,centroid,params,success
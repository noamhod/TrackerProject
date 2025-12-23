#!/usr/bin/python
import os
import math
import array
import numpy as np
import ROOT

from tracker_lib import config, objects

'''
SVD points = [  [vtx.x,  vtx.y,  vtx.z],
                [cls0.x, cls0.y, cls0.z],
                [cls1.x, cls1.y, cls1.z],
                [cls2.x, cls2.y, cls2.z],
                [cls3.x, cls3.y, cls3.z], 
                ...  ]

Chi2 points = [ [vtx.x,  cls0.x, cls1.x, cls2.x, cls3.x,...],
                [vtx.y,  cls0.y, cls1.y, cls2.y, cls3.y,...],
                [vtx.z,  cls0.z, cls1.z, cls2.z, cls3.z,...] ]
'''


def Candidate_SVDtoChi2(points,errors):
    clusters_x = []
    clusters_y = []
    clusters_z = []
    errors_x   = []
    errors_y   = []
    errors_z   = []
    for point in points:
        clusters_x.append( point[0] )
        clusters_y.append( point[1] )
        clusters_z.append( point[2] )    
    for error in errors:
        errors_x.append( point[0] )
        errors_y.append( point[1] )
        errors_z.append( point[2] )    
    points = np.array([ clusters_x, clusters_y, clusters_z ])
    errors = np.array([ errors_x,   errors_y,   errors_z ])
    return points,errors


def Candidate_Chi2toSVD(points,errors):
    clusters = []
    clerrors = []
    npoints = len(points[0])
    x = points[0]
    y = points[1]
    z = points[2]
    ex = errors[0]
    ey = errors[1]
    ez = errors[2]
    for i in range(npoints):
        clusters.append( [x[i],y[i],z[i]] )
        clerrors.append( [ex[i],ey[i],ez[i]] )
    points = np.array(clusters)
    errors = np.array(clerrors)
    return points,errors


def SVD_candidate(detectors,clsx,clsy,clsz,clsdx,clsdy,vtx=[],evtx=[],skipdets=[]):
    cfg = config.Config().map
    isvtx = (len(vtx)>0 and len(evtx)>0)
    clusters = [ [cfg["xVtx"], cfg["yVtx"], cfg["zVtx"]] ]  if(isvtx) else []
    clerrors = [ [cfg["exVtx"],cfg["eyVtx"],cfg["ezVtx"]] ] if(isvtx) else []
    for det in detectors:
        if(det in skipdets): continue
        clusters.append( [clsx[det],  clsy[det],  clsz[det]] )
        clerrors.append( [clsdx[det], clsdy[det], cfg["ezCls"]] )
    points = np.array(clusters)
    errors = np.array(clerrors)
    return points,errors


def Chi2_candidate(detectors,clsx,clsy,clsz,clsdx,clsdy,vtx=[],evtx=[],skipdets=[]):
    cfg = config.Config().map
    isvtx = (len(vtx)>0 and len(evtx)>0)
    clusters_x = [cfg["xVtx"]]  if(isvtx) else []
    clusters_y = [cfg["yVtx"]]  if(isvtx) else []
    clusters_z = [cfg["zVtx"]]  if(isvtx) else []
    clerrors_x = [cfg["exVtx"]] if(isvtx) else []
    clerrors_y = [cfg["eyVtx"]] if(isvtx) else []
    clerrors_z = [cfg["ezVtx"]] if(isvtx) else []
    for det in detectors:
        if(det in skipdets): continue
        clusters_x.append( clsx[det] )
        clusters_y.append( clsy[det] )
        clusters_z.append( clsz[det] )
        clerrors_x.append( clsdx[det] )
        clerrors_y.append( clsdy[det] )
        clerrors_z.append( cfg["ezCls"] )
    points = np.array([ clusters_x,clusters_y,clusters_z ])
    errors = np.array([ clerrors_x,clerrors_y,clerrors_z ])
    return points,errors
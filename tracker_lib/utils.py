#!/usr/bin/python
import os
import time
import datetime
import math
import subprocess
import array
import numpy as np
import ROOT
# from ROOT import *
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from scipy.optimize import curve_fit
import glob
from pathlib import Path
import re


from tracker_lib import config


def is_preprocessed():
    cfg = config.Config().map
    return ("preprocessed" in cfg["inputfile"])


def format_run_number(run):
    if(run<0 or run>=10000000):
        print(f"run number {run} is not supported. Quitting.")
        quit()
    if(run<10):                        return f"run_000000{run}"
    if(run>=10 and run<100):           return f"run_00000{run}"
    if(run>=100 and run<1000):         return f"run_0000{run}"
    if(run>=1000 and run<10000):       return f"run_000{run}"
    if(run>=10000 and run<100000):     return f"run_00{run}"
    if(run>=100000 and run<1000000):   return f"run_0{run}"
    if(run>=1000000 and run<10000000): return f"run_{run}" # assume no more than 9,999,999 events...
    return ""


def get_run_from_file(name):
    ## example: name = tree_09_02_2024_21_39_47_Run128.root
    words = name.split("_")
    word = words[-1]
    srun = word.replace("Run","").replace(".root","")
    run = int(srun)
    return run


def make_run_dirs(name):
    print(f"Got input file {name}")
    if(not os.path.isfile(name)):
        print(f"Input file {name} does not exist. Quitting.")
        quit()
    run    = get_run_from_file(name)
    srun   = format_run_number(run)
    paths  = name.split("/")
    infile = paths[-1]
    rundir = ""
    for i in range(len(paths)-1): rundir += paths[i]+"/"
    rundir += srun
    evtdir = rundir+"/event_displays"
    trgdir = rundir+"/beam_quality"
    cfgdir = rundir+"/config_used"
    filecopy = f"{rundir}/{infile}"
    if(not os.path.isdir(rundir)):
        print(f"Making dir {rundir}")
        ROOT.gSystem.Exec(f"/bin/mkdir -p {rundir}")
    if(not os.path.isdir(evtdir)):
        print(f"Making dir {evtdir}")
        ROOT.gSystem.Exec(f"/bin/mkdir -p {evtdir}")
    if(not os.path.isdir(trgdir)):
        print(f"Making dir {trgdir}")
        ROOT.gSystem.Exec(f"/bin/mkdir -p {trgdir}")
    if(not os.path.isdir(cfgdir)):
        print(f"Making dir {cfgdir}")
        ROOT.gSystem.Exec(f"/bin/mkdir -p {cfgdir}")
    # if(not os.path.isfile(filecopy)):
    #     print(f"Copying input file {name} to run dir {rundir}")
    #     ROOT.gSystem.Exec(f"/bin/cp -f {name} {rundir}/")
    print(f"Always(!) copying input file {name} to run dir {rundir}")
    ROOT.gSystem.Exec(f"/bin/cp -f {name} {rundir}/")
    return filecopy


def make_multirun_dir(name,runs):
    print(f"Got input file {name}")
    if(not os.path.isfile(name)):
        print(f"Input file {name} does not exist. Quitting.")
        quit()
    run = get_run_from_file(name)
    if(run not in runs):
        print(f"Input run {run} is not in the run list. Quitting.")
        quit()
    paths  = name.split("/")
    rundir = ""
    for i in range(len(paths)-1): rundir += paths[i]+"/"
    ### make the list of files to be hadded
    infiles = ""
    pklfiles = []
    for r in runs:
        srun = format_run_number(r)
        fname = rundir+srun+"/tree_*_multiprocess_histograms.root "
        pname = rundir+srun+"/tree_*.pkl"
        if(not len(glob.glob(fname))<1):
            print(f"Input file {fname} does not exist. Quitting.")
            quit()
        infiles += fname+" "
        pklfiles.extend( glob.glob(pname) )
    ### get the combined rundir
    runs.sort()
    sruns = format_run_number(runs[0])
    for i,r in enumerate(runs):
        if(i==0): continue
        srun = str(r)
        sruns += ("-"+srun)
    rundir += sruns
    if(not os.path.isdir(rundir)):
        print(f"Making dir {rundir}")
        ROOT.gSystem.Exec(f"/bin/mkdir -p {rundir}")
    ### hadd the file from scratch in that dir
    ftarget = f"{rundir}/tree_multiprocess_histograms.root"
    print(f"hadding input files:")
    ROOT.gSystem.Exec(f"hadd -f {ftarget} {infiles}")
    return ftarget, pklfiles


def get_human_timestamp(timestamp_ms,fmt="%d/%m/%Y, %H:%M:%S"):
    unix_timestamp = timestamp_ms/1000
    human_timestamp = time.strftime(fmt,time.localtime(unix_timestamp))
    return human_timestamp


def get_human_timestamp_ns(timestamp_ns,fmt="%d/%m/%Y, %H:%M:%S"):
    unix_timestamp = timestamp_ns/1e9
    human_timestamp = time.strftime(fmt,time.localtime(unix_timestamp))
    return human_timestamp


def get_run_length(run_start,run_end,fmt="hours"):
    run_start  = run_start/1000
    run_end    = run_end/1000
    run_length = datetime.datetime.fromtimestamp(run_end) - datetime.datetime.fromtimestamp(run_start)
    X = -1
    if(fmt=="hours"): X = 60*60
    if(fmt=="days"):  X = 60*60*24
    run_length_X = round(run_length.total_seconds()/X)
    return run_length_X


def count_active_tandem_layers(objects):
    cfg = config.Config().map
    counter = np.zeros(cfg["layers"])
    for det in cfg["detectors"]:
        tdm = cfg["det2tdm"][det]
        if(counter[tdm]>0): continue
        if(len(objects[det])>0): counter[tdm] = 1
    n_active_tandem_layers = 0
    for n in counter: n_active_tandem_layers += n
    return n_active_tandem_layers
    

def transform_to_real_space(v,det,algn=True):
    cfg = config.Config().map
    ############################
    ### important since first
    ### we need to rotate the
    ### chip frame to the lab
    ### frame and there is no
    ### z-ccoordinate there
    v[2] = 0 ###################
    ############################
    stave = cfg["det2stvchp"][det][0]
    tx = cfg["thetax"][stave]
    ty = cfg["thetay"][stave]
    tz = cfg["thetaz"][stave]
    Rx = [[1,0,0],[0,math.cos(tx),-math.sin(tx)], [0,math.sin(tx),math.cos(tx)]]
    Ry = [[math.cos(ty),0,math.sin(ty)], [0,1,0], [-math.sin(ty),0,math.cos(ty)]]
    Rz = [[math.cos(tz),-math.sin(tz),0], [math.sin(tz),math.cos(tz),0], [0,0,1]]
    ### rotate around x
    vx = [0,0,0]
    vx[0] = Rx[0][0]*v[0]+Rx[0][1]*v[1]+Rx[0][2]*v[2]
    vx[1] = Rx[1][0]*v[0]+Rx[1][1]*v[1]+Rx[1][2]*v[2]
    vx[2] = Rx[2][0]*v[0]+Rx[2][1]*v[1]+Rx[2][2]*v[2]
    ### rotate around z
    vz = [0,0,0]
    vz[0] = Rz[0][0]*vx[0]+Rz[0][1]*vx[1]+Rz[0][2]*vx[2]
    vz[1] = Rz[1][0]*vx[0]+Rz[1][1]*vx[1]+Rz[1][2]*vx[2]
    vz[2] = Rz[2][0]*vx[0]+Rz[2][1]*vx[1]+Rz[2][2]*vx[2]
    ### rotate around y
    vy = [0,0,0]
    vy[0] = Ry[0][0]*vz[0]+Ry[0][1]*vz[1]+Ry[0][2]*vz[2]
    vy[1] = Ry[1][0]*vz[0]+Ry[1][1]*vz[1]+Ry[1][2]*vz[2]
    vy[2] = Ry[2][0]*vz[0]+Ry[2][1]*vz[1]+Ry[2][2]*vz[2]
    ### do the alignment BEFORE the global offsetting
    va = [vy[0],vy[1],vy[2]]
    if(algn): va[0],va[1] = align(det,va[0],va[1]) ### this is the alignment
    ### introduce the offsets of the real space position of the detector (this is not the alignment offests!)
    r = va
    r[0] += (cfg["offsets_x"][det]+cfg["xGlobalOffset"])
    r[1] += (cfg["offsets_y"][det]+cfg["yGlobalOffset"])
    r[2] += (cfg["offsets_z"][det]+cfg["zGlobalOffset"])
    return r


def undo_global_offsets(r,det):
    cfg = config.Config().map
    r[0] -= cfg["xGlobalOffset"]
    r[1] -= cfg["yGlobalOffset"]
    r[2] -= cfg["zGlobalOffset"]
    return r




def yofx(r1,r2,x):
   dx = r2[0]-r1[0]
   dy = r2[1]-r1[1]
   if(dx==0):
      print("ERROR in yofz: dx=0 --> r1[0]=%g,r2[0]=%g, r1[1]=%g,r2[1]=%g" % (r1[0],r2[0],r1[1],r2[1]))
      quit()
   a = dy/dx
   b = r1[1]-a*r1[0]
   y = a*x+b
   return y


def xofz(r1,r2,z):
   dz = r2[2]-r1[2]
   dx = r2[0]-r1[0]
   if(dz==0):
      print("ERROR in xofz: dx=0 --> r1[0]=%g,r2[0]=%g, r1[1]=%g,r2[1]=%g, r1[2]=%g,r2[2]=%g" % (r1[0],r2[0],r1[1],r2[1],r1[2],r2[2]))
      quit()
   a = dx/dz
   b = r1[0]-a*r1[2]
   x = a*z+b
   return x


def yofz(r1,r2,z):
   dz = r2[2]-r1[2]
   dy = r2[1]-r1[1]
   if(dz==0):
      print("ERROR in yofz: dz=0 --> r1[0]=%g,r2[0]=%g, r1[1]=%g,r2[1]=%g, r1[2]=%g,r2[2]=%g" % (r1[0],r2[0],r1[1],r2[1],r1[2],r2[2]))
      quit()
   a = dy/dz
   b = r1[1]-a*r1[2]
   y = a*z+b
   return y


def xyofz(r1,r2,z):
    x = xofz(r1,r2,z)
    y = yofz(r1,r2,z)
    return x,y


def line(t, params):
    # a parametric line is defined from 6 parameters but 4 are independent
    # x0,y0,z0,z1,y1,z1 which are the coordinates of two points on the line
    # can choose z0 = 0 if line not parallel to x-y plane and z1 = 1;
    x = params[0] + params[1]*t
    y = params[2] + params[3]*t
    z = t
    return x,y,z


def get_pars_from_points(kA,kB,zA,zB):
    p1 = (kB-kA)/(zB-zA)
    # p0 = ((kB+kA)-p1*(zB+zA))/2.
    p0 = kA-p1*zA
    return p0,p1

    
def get_pars_from_centroid_and_direction(centroid,direction):
    xA = centroid[0]
    xB = centroid[0]+direction[0]
    yA = centroid[1]
    yB = centroid[1]+direction[1]
    zA = centroid[2]
    zB = centroid[2]+direction[2]
    rA = [xA,yA,zA]
    rB = [xB,yB,zB]
    p0x,p1x = get_pars_from_points(rA[0],rB[0],rA[2],rB[2])
    p0y,p1y = get_pars_from_points(rA[1],rB[1],rA[2],rB[2])
    return [p0x,p1x,p0y,p1y]


def r1r2(direction, centroid):
    r1 = [centroid[0], centroid[1], centroid[2] ]
    r2 = [centroid[0]+direction[0], centroid[1]+direction[1], centroid[2]+direction[2] ]
    return r1,r2


def rotate(theta,x,y):
    xr = x*math.cos(theta)-y*math.sin(theta)
    yr = x*math.sin(theta)+y*math.cos(theta)
    return xr,yr


def align(det,x,y):
    cfg = config.Config().map
    x,y = rotate(cfg["misalignment"][det]["theta"],x,y)
    x = x+cfg["misalignment"][det]["dx"]
    y = y+cfg["misalignment"][det]["dy"]
    return x,y


def res_track2clusterErr(det, detectors, points, errors, direction, centroid):
    cfg = config.Config().map
    r1,r2 = r1r2(direction, centroid)
    x  = points[:,0]
    y  = points[:,1]
    ex = errors[:,0]
    ey = errors[:,1]
    zpoints = points[:,2]
    if(det not in detectors):
        print(f"{det} is not in track detectors. returning -999,-999")
        return -999,-999
    i = detectors.index(det)
    if(cfg["doVtx"]):
        if(len(points)==len(detectors)+1): i = i+1 ### when the vertex is the first point in the points array
        else:
            print("In res_track2clusterErr")
            print(f"Problem with vertex or length of points. Quitting")
            quit()
    z = zpoints[i]
    xonline,yonline = xyofz(r1,r2,z)
    # print(f"det={det}: z={z}.  xfit={xonline:.2E}, xpoint={x[i]:.2E}, dx={xonline-x[i]:.2E}, errx={ex[i]:.2E}.  yfit={yonline:.2E}, ypoint={y[i]:.2E}, dx={yonline-y[i]:.2E}, erry={ey[i]:.2E}")
    dx = (xonline-x[i])/ex[i]
    dy = (yonline-y[i])/ey[i]
    return dx,dy


def res_track2cluster(det, detectors, points, direction, centroid):
    cfg = config.Config().map
    r1,r2 = r1r2(direction, centroid)
    x = points[:,0]
    y = points[:,1]
    zpoints = points[:,2]
    if(det not in detectors):
        print(f"{det} is not in track detectors. returning -999,-999")
        return -999,-999
    i = detectors.index(det)
    if(cfg["doVtx"]):
        if(len(points)==len(detectors)+1): i = i+1 ### when the vertex is the first point in the points array
        else:
            print("In res_track2cluster()")
            print(f"Problem with vertex or length of points. Quitting")
            quit()
    z = zpoints[i]
    xonline,yonline = xyofz(r1,r2,z)
    # print(f"det={det}: z={z}.  xfit={xonline:.2E}, xpoint={x[i]:.2E}, dx={xonline-x[i]:.2E}.  yfit={yonline:.2E}, ypoint={y[i]:.2E}, dx={yonline-y[i]:.2E}")
    dx = xonline-x[i]
    dy = yonline-y[i]
    return dx,dy
    

def res_track2vertex(vertex, direction, centroid):
    r1,r2 = r1r2(direction, centroid)
    z = vertex[2]
    xonline = xofz(r1,r2,z)
    yonline = yofz(r1,r2,z)
    dx = xonline-vertex[0]
    dy = yonline-vertex[1]
    return dx,dy


def get_track_point_at_z(track,z):
    x,y,z = line(z,track.params)
    r = [x,y,z]
    return r


def get_trak_at_det(det,track):
    cfg = config.Config().map
    r0 = [0,0,cfg["rdetectors"][det][2]]
    rT = transform_to_real_space(r0,det)
    xT,yT,zT = line(rT[2],track.params)
    return xT,yT,zT


def get_track_point_at_extremes(track):
    cfg = config.Config().map
    x0,y0,z0 = get_trak_at_det(cfg["det_frst"],track)
    xN,yN,zN = get_trak_at_det(cfg["det_last"],track)
    zW = cfg["zWindow"]
    zF = cfg["zFlangeExit"]
    zD = cfg["zDipoleExit"]
    r0 = get_track_point_at_z(track,z0)
    rN = get_track_point_at_z(track,zN)
    rW = get_track_point_at_z(track,zW)
    rF = get_track_point_at_z(track,zF)
    rD = get_track_point_at_z(track,zD)
    
    return r0,rN,rW,rF,rD
    

def get_pdc_window_bounds():
    cfg = config.Config().map
    xWinL = cfg["xWindow"]-cfg["xWindowWidth"]/2.
    xWinR = cfg["xWindow"]+cfg["xWindowWidth"]/2.
    yWinB = cfg["yWindowMin"]
    yWinT = cfg["yWindowMin"]+cfg["yWindowHeight"]
    return xWinL,xWinR,yWinB,yWinT

def get_dipole_exit_bounds():
    cfg = config.Config().map
    xDipL = cfg["xDipoleExitMin"]
    xDipR = cfg["xDipoleExitMax"]
    yDipB = cfg["yDipoleExitMin"]
    yDipT = cfg["yDipoleExitMax"]
    return xDipL,xDipR,yDipB,yDipT
    
def get_dipole_flange_bounds():
    cfg = config.Config().map
    xFlgL = cfg["xFlangeMin"]
    xFlgR = cfg["xFlangeMax"]
    yFlgB = cfg["yFlangeMin"]
    yFlgT = cfg["yFlangeMax"]
    return xFlgL,xFlgR,yFlgB,yFlgT



def getChips2D():
    cfg = config.Config().map
    chips = {}
    for det in cfg["detectors"]:
        r00 = [-cfg["chipX"]/2.,-cfg["chipY"]/2.,0]
        r11 = [+cfg["chipX"]/2.,+cfg["chipY"]/2.,0]
        rT00 = transform_to_real_space(r00,det)
        rT11 = transform_to_real_space(r11,det)
        xmin = min(rT00[0],rT11[0])
        xmax = max(rT00[0],rT11[0])
        ymin = min(rT00[1],rT11[1])
        ymax = max(rT00[1],rT11[1])
        chips.update({ det: np.array([ [xmin,ymin],
                                       [xmin,ymax],
                                       [xmax,ymax],
                                       [xmax,ymin] ]) })
    return chips


def getChips(translatez=True):
    ### draw the chips: https://stackoverflow.com/questions/67410270/how-to-draw-a-flat-3d-rectangle-in-matplotlib
    cfg = config.Config().map
    L1verts = []
    for det in cfg["detectors"]:
        r00 = [-cfg["chipX"]/2.,-cfg["chipY"]/2.,cfg["rdetectors"][det][2]]
        r11 = [+cfg["chipX"]/2.,+cfg["chipY"]/2.,cfg["rdetectors"][det][2]]
        rT00 = transform_to_real_space(r00,det)
        rT11 = transform_to_real_space(r11,det)
        xmin = min(rT00[0],rT11[0])
        xmax = max(rT00[0],rT11[0])
        ymin = min(rT00[1],rT11[1])
        ymax = max(rT00[1],rT11[1])
        z = rT00[2]
        L1verts.append( np.array([ [xmin,ymin,z],
                                   [xmin,ymax,z],
                                   [xmax,ymax,z],
                                   [xmax,ymin,z] ]) )
    return L1verts
        

def getWindowRealSpace():
    cfg = config.Config().map
    zWindow       = cfg["zWindow"]
    xWindowWidth  = cfg["xWindowWidth"]
    yWindowHeight = cfg["yWindowHeight"]
    xWindow       = cfg["xWindow"]
    yWindowMin    = cfg["yWindowMin"]
    window = np.array([ [xWindow-xWindowWidth/2., yWindowMin,               zWindow],
                        [xWindow-xWindowWidth/2., yWindowMin+yWindowHeight, zWindow],
                        [xWindow+xWindowWidth/2., yWindowMin+yWindowHeight, zWindow],
                        [xWindow+xWindowWidth/2., yWindowMin,               zWindow] ])
    return [window]


def getDipoleRealSpace():
    cfg = config.Config().map
    zDipole    = cfg["zDipoleExit"]
    xMin       = cfg["xDipoleExitMin"]
    xMax       = cfg["xDipoleExitMax"]
    yMin       = cfg["yDipoleExitMin"]
    yMax       = cfg["yDipoleExitMax"]
    dipole = np.array([ [xMin, yMin, zDipole],
                        [xMin, yMax, zDipole],
                        [xMax, yMax, zDipole],
                        [xMax, yMin, zDipole] ])
    return [dipole]


def InitCutflow():
    cfg = config.Config().map
    cutflow = {}
    for cut in cfg["cuts"]: cutflow.update({cut:0})
    return cutflow

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

### pickle files
def getfileslist(directory,pattern,suff):
    path = Path(os.path.expanduser(directory))
    ff = [str(file) for file in path.glob(pattern + '*' + suff)]
    ff.sort(key=natural_keys)
    return ff

### pickle files
def getfiles(tfilenamein):
    words = tfilenamein.split("/")
    directory = ""
    for w in range(len(words)-1):
        directory += words[w]+"/"
    strippedname = words[-1].split(".pkl")[0]
    words = strippedname.split("_")
    pattern = ""
    for w in range(len(words)):
        word = words[w].replace(".root","")
        pattern += word+"_"
    print("directory:",directory)
    print("pattern:",pattern)
    files = getfileslist(directory,pattern,".pkl")
    return files


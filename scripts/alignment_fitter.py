#!/usr/bin/python
import multiprocessing as mp
# from multiprocessing.pool import ThreadPool
import time
import sys
import os
import os.path
import math
import subprocess
import array
import numpy as np
import ROOT
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from scipy.optimize import curve_fit,basinhopping,least_squares
import pickle
from pathlib import Path
import ctypes
import random

import argparse
parser = argparse.ArgumentParser(description='alignment_fitter.py...')
parser.add_argument('-conf', metavar='config file', required=True,  help='full path to config file')
parser.add_argument('-beam', metavar='is beam run?',required=True, help='is this a beam run? [0/1]')
parser.add_argument('-ref',  metavar='reference detectors', required=False,  help='reference detectors (comma separated)')
parser.add_argument('-mult', metavar='multi run?',  required=False, help='is this a multirun? [0/1]')
argus = parser.parse_args()
configfile = argus.conf
isbeamrun  = (int(argus.beam)==1)
refdet     = argus.ref  if(argus.ref  is not None) else ""
ismutirun  = argus.mult if(argus.mult is not None and int(argus.mult)==1) else False

refdet = refdet.split(",") if(refdet!="") else []
print(f"Reference detectors for alignment: {refdet}")


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tracker_lib import config
from tracker_lib import objects, Pixels, Clusters, candidate, selections, hists, errors, counters, noise, utils, svd_fit, chi2_fit



ROOT.gROOT.SetBatch(1)
ROOT.gStyle.SetOptFit(0)
# ROOT.gStyle.SetOptStat(0)

### defined below as global
allhistos = {}


def pass_alignment_selections(track):
    cfg = config.Config().map
    ### require good chi2 range and other cuts
    if(track.chi2ndof<cfg["minchi2align"]): return False
    if(track.chi2ndof>cfg["maxchi2align"]): return False
    ### FOR BEAM ONLY: require pointing to the pdc window and the dipole exit aperture and inclined up as a positron, etc
    if(isbeamrun):
        if(track.maxcls>cfg["cut_maxcls"]):   return False
        if(not selections.pass_geoacc_selection(track)): return False
    return True


def fitSVD(track,dx,dy,theta,refdet=[]):
    cfg = config.Config().map
    clsx  = {}
    clsy  = {}
    clsz  = {}
    clsdx = {}
    clsdy = {}
    
    ### initialize the misalignments per detector
    dX    = {}
    dY    = {}
    Theta = {}
    i = 0
    for det in cfg["detectors"]:
        if(len(refdet)>0 and det in refdet): continue
        dX.update({det:dx[i]})
        dY.update({det:dy[i]})
        Theta.update({det:theta[i]})
        i += 1
    
    ### prepare the track's clusters with misalignments
    # for det in cfg["detectors"]:
    for det in track.detectors:
        x = track.trkcls[det].xTnoGmm
        y = track.trkcls[det].yTnoGmm
        z = track.trkcls[det].zTnoGmm
        ex = track.trkcls[det].xTnoGsizemm if(cfg["use_large_clserr_for_algnmnt"]) else track.trkcls[det].dxTnoGmm
        ey = track.trkcls[det].yTnoGsizemm if(cfg["use_large_clserr_for_algnmnt"]) else track.trkcls[det].dyTnoGmm
        ### only for the non-reference detectors or all detectors?
        if(len(refdet)>0):
            if(det not in refdet):
                x,y = utils.rotate(Theta[det],x,y)
                x = x+dX[det]
                y = y+dY[det]
        else:
            x,y = utils.rotate(Theta[det],x,y)
            x = x+dX[det]
            y = y+dY[det]
        clsx.update({det:x})
        clsy.update({det:y})
        clsz.update({det:z})
        clsdx.update({det:ex})
        clsdy.update({det:ey})
    vtx  = [cfg["xVtx"], cfg["yVtx"],  cfg["zVtx"]]  if(cfg["doVtx"]) else []
    evtx = [cfg["exVtx"],cfg["eyVtx"], cfg["ezVtx"]] if(cfg["doVtx"]) else []
    # points_SVD,errors_SVD = candidate.SVD_candidate(clsx,clsy,clsz,clsdx,clsdy,vtx,evtx)
    points_SVD,errors_SVD = candidate.SVD_candidate(track.detectors,clsx,clsy,clsz,clsdx,clsdy,vtx,evtx)
    chisq_SVD,ndof_SVD,direction_SVD,centroid_SVD = svd_fit.fit_3d_SVD(points_SVD,errors_SVD)
    
    dabs = 0
    dxabs = {}
    dyabs = {}
    # for idet,det in enumerate(cfg["detectors"]):
    for idet,det in enumerate(track.detectors):
        dx = 0 
        dy = 0
        if(cfg["alignmentwerr"]):
            dx,dy = utils.res_track2clusterErr(det,track.detectors,points_SVD,errors_SVD,direction_SVD,centroid_SVD)
        else:
            dx,dy = utils.res_track2cluster(det,track.detectors,points_SVD,direction_SVD,centroid_SVD)

        # ##############
        # idet = cfg["detectors"].index(det)
        # r1,r2 = utils.r1r2(direction_SVD, centroid_SVD)
        # xonline,yonline = utils.xyofz(r1,r2,points_SVD[idet][2])
        # print(f"{det}: z={points_SVD[idet][2]} --> dx={dx:.2E}, dy={dy:.2E}  x={points_SVD[idet][0]}-->fitx={xonline}, y={points_SVD[idet][1]}-->fity={yonline}")
        # ##############
            
        dxabs.update( {det:abs(dx)} )
        dyabs.update( {det:abs(dy)} )
        dabs += math.sqrt(dx*dx + dy*dy)
    
    for det in cfg["detectors"]:
        # dxabs[det] /= len(cfg["detectors"])
        dxabs[det] /= len(track.detectors)
        # dyabs[det] /= len(cfg["detectors"])
        dyabs[det] /= len(track.detectors)

    # dabs /= len(cfg["detectors"])
    dabs /= len(track.detectors)
    
    return chisq_SVD,ndof_SVD,dabs,dxabs,dyabs


def init_params(axes,ndet2align,params):
    cfg = config.Config().map
    dxFinal    = [0]*ndet2align
    dyFinal    = [0]*ndet2align
    thetaFinal = [0]*ndet2align
    nparperdet = -1
    if(axes=="xytheta"):
        nparperdet = 3
        dxFinal    = params[0:ndet2align]
        dyFinal    = params[ndet2align:ndet2align*2]
        thetaFinal = params[ndet2align*2:ndet2align*3]
    elif(axes=="xy"):
        nparperdet = 2
        dxFinal    = params[0:ndet2align]
        dyFinal    = params[ndet2align:ndet2align*2]
    elif(axes=="xtheta"):
        nparperdet = 2
        dxFinal    = params[0:ndet2align]
        thetaFinal = params[ndet2align:ndet2align*2]
    elif(axes=="ytheta"):
        nparperdet = 2
        dyFinal    = params[0:ndet2align]
        thetaFinal = params[ndet2align:ndet2align*2]
    elif(axes=="x"):
        nparperdet = 1
        dxFinal = params[0:ndet2align]
    elif(axes=="y"):
        nparperdet = 1
        dyFinal = params[0:ndet2align]
    elif(axes=="theta"):
        nparperdet = 1
        thetaFinal = params[0:ndet2align]
    else:
        print("Unknown axes combination. Quitting.")
        quit()
    return dxFinal,dyFinal,thetaFinal,nparperdet


def fit_misalignment(events,ndet2align,refdet,axes):
    cfg = config.Config().map
    ### Define the objective function to minimize (the chi^2 function)
    ### similar to https://root.cern.ch/doc/master/line3Dfit_8C_source.html
    def metric_function_to_minimize(params):
        dx,dy,dt,nparperdet = init_params(axes,ndet2align,params)
        sum_dx = 0
        sum_dy = 0
        sum_dabs = 0
        sum_chi2 = 0
        nvalidevents = 0
        nvalidtracks = 0
        for event in events:
            
            tracks = event.tracks if(cfg["cut_allow_shared_clusters"]) else selections.remove_tracks_with_shared_clusters(event.tracks)
            for track in tracks:
                ### require some relevant cuts
                if(not pass_alignment_selections(track)): continue
                
                chisq,ndof,dabs,dX,dY = fitSVD(track,dx,dy,dt,refdet)
                nvalidtracks += 1
                sum_dabs     += dabs

        return sum_dabs/nvalidtracks
    
    nparperdet = -1
    if  (axes=="xytheta"):                                nparperdet = 3
    elif(axes=="xy" or axes=="xtheta" or axes=="ytheta"): nparperdet = 2
    elif(axes=="x"  or axes=="y"      or axes=="theta"):  nparperdet = 1
    else:
        print("Unknown axes combination. Quitting.")
        quit()
    
    ### https://stackoverflow.com/questions/24767191/scipy-is-not-optimizing-and-returns-desired-error-not-necessarily-achieved-due
    initial_params = [0]*(nparperdet*ndet2align)
    dx_range = [(cfg["alignmentbounds"]["dx"]["min"],    cfg["alignmentbounds"]["dx"]["max"])]*ndet2align    if("x"     in axes) else []
    dy_range = [(cfg["alignmentbounds"]["dy"]["min"],    cfg["alignmentbounds"]["dy"]["max"])]*ndet2align    if("y"     in axes) else []
    dt_range = [(cfg["alignmentbounds"]["theta"]["min"], cfg["alignmentbounds"]["theta"]["max"])]*ndet2align if("theta" in axes) else []
    ranges = []
    if("x"     in axes): ranges.extend(dx_range)
    if("y"     in axes): ranges.extend(dy_range)
    if("theta" in axes): ranges.extend(dt_range)
    range_params = tuple(ranges)
    print("initial_params:",initial_params)
    print("range_params:",range_params)
    ### https://stackoverflow.com/questions/52438263/scipy-optimize-gets-trapped-in-local-minima-what-can-i-do
    ### https://stackoverflow.com/questions/25448296/scipy-basin-hopping-minimization-on-function-with-free-and-fixed-parameters
    result = None
    if(cfg["alignmentmethod"]=="SLSQP"):
        result = basinhopping(metric_function_to_minimize, initial_params, niter=cfg["naligniter"], minimizer_kwargs={"method":"SLSQP", "bounds":range_params})
    elif(cfg["alignmentmethod"]=="COBYLA"):
        result = minimize(metric_function_to_minimize, initial_params, method='COBYLA', bounds=range_params, options={'disp': True, 'maxiter':2000})
    elif(cfg["alignmentmethod"]=="Powell"):
        result = minimize(metric_function_to_minimize, initial_params, method='Powell', bounds=range_params, options={'disp': True, 'maxiter':2000})
        # result = minimize(metric_function_to_minimize, initial_params, method='Powell', bounds=range_params, options={'disp': True, 'maxiter':2000, 'ftol':1e-1, 'xtol':1e-1})
    elif(cfg["alignmentmethod"]=="least_squares"):
        lower_bounds = np.array([lo for (lo, hi) in range_params])
        upper_bounds = np.array([hi for (lo, hi) in range_params])
        # result = least_squares(metric_function_to_minimize, initial_params, bounds=(lower_bounds, upper_bounds), verbose=2, xtol=1e-15, ftol=1e-15)
        result = least_squares(metric_function_to_minimize, initial_params, bounds=(lower_bounds, upper_bounds), verbose=2)
    else:
        print(f'alignmentmethod={cfg["alignmentmethod"]} is not supported. Quitting.')
        quit()
    
    ### get the chi^2 value and the number of degrees of freedom
    chisq = result.fun
    params  = result.x
    success = result.success
    return params,chisq,success





if __name__ == "__main__":
    # get the start time
    st = time.time()
    
    
    #############################################
    ### Initialize Config in the main process ###
    config.init_config(configfile, False)
    cfg = config.Config().map
    # config.show_config() # print config once
    print(cfg) # print config once
    #############################################
    
    
    if(len(refdet)>0):
        for det in refdet:
            if(det not in cfg["detectors"]):
                print("Unknown detector:",det," --> quitting")
                quit()
    
    ### get all the files
    tfilenamein = ""
    files = []
    if(ismutirun):
        tfilenamein,files = utils.make_multirun_dir(cfg["inputfile"],cfg["runnums"])
    else:
        tfilenamein = utils.make_run_dirs(cfg["inputfile"])
        files = utils.getfiles(tfilenamein)
    
    ###
    axes       = cfg["axes2align"]
    ndet2align = len(cfg["detectors"])-len(refdet)
    nparperdet = -1
    if  (axes=="xytheta"):                                nparperdet = 3
    elif(axes=="xy" or axes=="xtheta" or axes=="ytheta"): nparperdet = 2
    elif(axes=="x"  or axes=="y"      or axes=="theta"):  nparperdet = 1
    else:
        print("Unknown axes combination. Quitting.")
        quit()
    
    
    ### some histos
    histos = {}
    NscanBins = 50
    absRes    = 0.05
    nResBins  = 50
    nResBins2D = 80
    for det in cfg["detectors"]:
        name = f"dx_{det}"; histos.update( {name:ROOT.TH1D(name,det+";dx [mm];#sum#Deltax [mm]",NscanBins,cfg["alignmentbounds"]["dx"]["min"],cfg["alignmentbounds"]["dx"]["max"])} )
        name = f"dy_{det}"; histos.update( {name:ROOT.TH1D(name,det+";dy [mm];#sum#Deltay [mm]",NscanBins,cfg["alignmentbounds"]["dy"]["min"],cfg["alignmentbounds"]["dy"]["max"])} )
        name = f"dt_{det}"; histos.update( {name:ROOT.TH1D(name,det+";d#theta [rad];#sum#Deltar [mm]",NscanBins,cfg["alignmentbounds"]["theta"]["min"],cfg["alignmentbounds"]["theta"]["max"])} )
        
        if(cfg["isFakeMC"]):
            name = f"dxhist_{det}"; histos.update( {name:ROOT.TH1D(name,";x_{final}-x_{orig} [mm];Tracks",400,cfg["alignmentbounds"]["dx"]["min"],cfg["alignmentbounds"]["dx"]["max"])} )
            name = f"dyhist_{det}"; histos.update( {name:ROOT.TH1D(name,";y_{final}-y_{orig} [mm];Tracks",400,cfg["alignmentbounds"]["dy"]["min"],cfg["alignmentbounds"]["dy"]["max"])} )

        name = f"h_residual_x_{det}"; histos.update( {name:ROOT.TH1D(name,"det+;x_{trk}-x_{cls} [mm];Tracks",nResBins,-absRes*3,+absRes*3) } )
        name = f"h_residual_y_{det}"; histos.update( {name:ROOT.TH1D(name,"det+;y_{trk}-y_{cls} [mm];Tracks",nResBins,-absRes*3,+absRes*3) } )

        name = f"h_residual_x_mid_{det}"; histos.update( {name:ROOT.TH1D(name,det+";x_{trk}-x_{cls} [mm];Tracks",nResBins*2,-absRes*5,+absRes*5) } )
        name = f"h_residual_y_mid_{det}"; histos.update( {name:ROOT.TH1D(name,det+";y_{trk}-y_{cls} [mm];Tracks",nResBins*2,-absRes*5,+absRes*5) } )
        
        name = f"h_residual_xy_{det}";     histos.update( {name:ROOT.TH2D(name,det+";x_{trk}-x_{cls} [mm];y_{trk}-y_{cls} [mm];Tracks",nResBins2D,-absRes*3,+absRes*3, nResBins2D,-absRes*3,+absRes*3) } )
        name = f"h_residual_xy_mid_{det}"; histos.update( {name:ROOT.TH2D(name,det+";x_{trk}-x_{cls} [mm];y_{trk}-y_{cls} [mm];Tracks",nResBins2D,-absRes*5,+absRes*5, nResBins2D,-absRes*5,+absRes*5) } )
        
        # name = f"h_residual_x_full_{det}"; histos.update( {name:ROOT.TH1D(name,det+";x_{trk}-x_{cls} [mm];Tracks",nResBins*2,-absRes*50,+absRes*50) } )
        # name = f"h_residual_y_full_{det}"; histos.update( {name:ROOT.TH1D(name,det+";y_{trk}-y_{cls} [mm];Tracks",nResBins*2,-absRes*50,+absRes*50) } )
        
        name = f"h_response_x_{det}"; histos.update( {name:ROOT.TH1D(name,det+";#frac{x_{trk}-x_{cls}}{#sigma(x_{cls})};Tracks",100,-12.5,+12.5) } )
        name = f"h_response_y_{det}"; histos.update( {name:ROOT.TH1D(name,det+";#frac{y_{trk}-y_{cls}}{#sigma(y_{cls})};Tracks",100,-12.5,+12.5) } )
    
    
    ### save all events
    events = []
    chisq0 = 0
    dabs0  = 0
    dX0    = 0
    dY0    = 0
    allevents = 0
    alltracks = 0
    nuniquetrks = 0
    ngoodtracks = 0
    for fpkl in files:
        suff = str(fpkl).split("_")[-1].replace(".pkl","")
        print(f"Opening file {suff}")
        with open(fpkl,'rb') as handle:
            data = pickle.load(handle)
            for event in data:
                if(allevents%50==0 and allevents>0): print(f"Reading event #{allevents} with {ngoodtracks} good tracks")
                allevents += 1
                alltracks += len(event.tracks)
                evtgoodtracks = 0
                tracks = event.tracks if(cfg["cut_allow_shared_clusters"]) else selections.remove_tracks_with_shared_clusters(event.tracks)
                for track in tracks:
                    
                    ### require some relevant cuts
                    if(not pass_alignment_selections(track)): continue
                    
                    chisq,ndof,dabs,dX,dY = fitSVD(track,[0.]*ndet2align,[0.]*ndet2align,[0.]*ndet2align,refdet)
                    chi2dof = chisq/ndof
                    
                    ### count and proceed
                    if(ngoodtracks%25==0 and ngoodtracks>0): print(f"Added {ngoodtracks} tracks")
                    
                    ngoodtracks += 1
                    evtgoodtracks += 1 
                    chisq0 += chi2dof
                    dabs0  += dabs
                
                if(evtgoodtracks>0):
                    minevt = objects.MinimalEvent(event.trigger,event.tracks)
                    events.append(minevt)
                    # events.append(event)

    if(ngoodtracks<cfg["alignmentmintrks"]):
        print(f'Too few tracks collected ({ngoodtracks}) for the chi2/dof cut of maxchi2align={cfg["maxchi2align"]} --> try to increase it in the config file.')
        print("Quitting")
        quit()
    chisq0 = chisq0/ngoodtracks
    dabs0  = dabs0/ngoodtracks
    print(f"Done collecting {ngoodtracks} tracks (out of {alltracks} ({nuniquetrks} unique) in {allevents} events, or {float(alltracks)/float(allevents)} trks/evt) with chisq0={chisq0} and dabs0={dabs0}. Now going to fit misalignments")
    
    
    
    ###################
    ### fill histos ###
    ###################
    
    for event in events:
        tracks = event.tracks if(cfg["cut_allow_shared_clusters"]) else selections.remove_tracks_with_shared_clusters(event.tracks)
        for itrk,track in enumerate(tracks):
            
            ### require some relevant cuts
            if(not pass_alignment_selections(track)): continue
            # for det in cfg["detectors"]:
            for det in track.detectors:
                dx,dy = utils.res_track2cluster(det,track.detectors,track.points,track.direction,track.centroid)
                histos[f"h_residual_xy_{det}"].Fill(dx,dy)
                histos[f"h_residual_xy_mid_{det}"].Fill(dx,dy)
                histos[f"h_residual_x_{det}"].Fill(dx)
                histos[f"h_residual_y_{det}"].Fill(dy)
                histos[f"h_residual_x_mid_{det}"].Fill(dx)
                histos[f"h_residual_y_mid_{det}"].Fill(dy)
                histos[f"h_response_x_{det}"].Fill(dx/track.trkcls[det].dxTnoGmm)
                histos[f"h_response_y_{det}"].Fill(dy/track.trkcls[det].dyTnoGmm)
                
                if(cfg["isFakeMC"]):
                    xOrig = track.trkcls[det].pixels[0].xOrig
                    yOrig = track.trkcls[det].pixels[0].yOrig
                    xFinal = track.trkcls[det].pixels[0].xFake
                    yFinal = track.trkcls[det].pixels[0].yFake
                    # print(f"Track[{itrk}] in {det}:  xOrig={xOrig}-->xFinal={xFinal}, yOrig={yOrig}-->yFinal={yFinal}")
                    histos[f"dxhist_{det}"].Fill(xFinal-xOrig)
                    histos[f"dyhist_{det}"].Fill(yFinal-yOrig)


    ### save histos
    fOut = ROOT.TFile("scan.root","RECREATE")
    fOut.cd()
    for hname,hist in histos.items(): hist.Write()
    fOut.Write()
    fOut.Close()
    ###################   
    
    
    #######################
    ### Run the fit !!! ###
    #######################
    params,result,success = fit_misalignment(events,ndet2align,refdet,axes)
    
    ### check
    chisq1 = 0
    dabs1  = 0
    dX1    = 0
    dY1    = 0
    allevents1 = 0
    ngoodtracks = 0
    dxFinal,dyFinal,thetaFinal,nparperdet = init_params(axes,ndet2align,params)
    for event in events:
        for track in event.tracks:
            ### require some relevant cuts
            if(not pass_alignment_selections(track)): continue

            chisq,ndof,dabs,dX,dY = fitSVD(track,dxFinal,dyFinal,thetaFinal,refdet)
            chi2dof = chisq/ndof

            ngoodtracks += 1
            chisq1 += chi2dof
            dabs1  += dabs
            
    chisq1 = chisq1/ngoodtracks
    dabs1  = dabs1/ngoodtracks
    
    ### sumarize
    print("\n----------------------------------------")
    print(f"Alignment axes: {axes}")
    if(len(refdet)>0): print(f"Reference detectors: {refdet}")
    else:              print(f"No reference detector")
    print(f"Events used: {len(events)} out of {allevents}")
    print(f"Tracks used: {ngoodtracks}")
    print(f"Success? {success}")
    print(f"chi2: {chisq1} (original: {chisq0})")
    print(f"dabs: {dabs1} (original: {dabs0})")
    print(f"dx final   : {dxFinal}")
    print(f"dy final   : {dyFinal}")
    print(f"theta final: {thetaFinal}")
    print("----------------------------------------\n")
    salignment = "misalignment  = "
    k = 0
    for det in cfg["detectors"]:
        if(det in refdet):
            salignment += f"{det}:dx=0,dy=0,theta=0 "
        else:
            dx = dxFinal[k]    + cfg["misalignment"][det]["dx"]
            dy = dyFinal[k]    + cfg["misalignment"][det]["dy"]
            dt = thetaFinal[k] + cfg["misalignment"][det]["theta"]
            salignment += f"{det}:dx={dx:.2E},dy={dy:.2E},theta={dt:.2E} "
            # salignment += f"{det}:dx={dxFinal[k]:.2E},dy={dyFinal[k]:.2E},theta={thetaFinal[k]:.2E} "
            # salignment += f"{det}:dx={dxf:.2E},dy={dyf:.2E},theta={dtf:.2E} "
            k += 1
    print(salignment)
    
    # get the end time
    et = time.time()
    # get the execution time
    elapsed_time = et - st
    print(f'ֿֿ\nExecution time: {elapsed_time}, seconds')

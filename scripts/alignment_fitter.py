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
from iminuit import Minuit
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
from tracker_lib import objects, Pixels, Clusters, candidate, selections, hists, errors, counters, noise, utils, svd_fit, chi2_fit, mle_fit



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
        if(track.maxcls>cfg["cut_maxcls"]): return False
        if(not selections.pass_geoacc_selection(track)): return False
    return True


def get_selected_tracks(event):
    tracks = []
    for track in event.tracks:
        if(not track.success):                    continue
        if(not pass_alignment_selections(track)): continue
        tracks.append(track)
    tracks = tracks if(cfg["cut_allow_shared_clusters"]) else selections.remove_tracks_with_shared_clusters(tracks)
    return tracks


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



def RefitTrack_Fast(dets, coords, dx_f, dy_f, dt_f, refdet, det_map):
    ### make a working copy to avoid modify the shared data
    working_coords = coords.copy()
    for i, det in enumerate(dets):
        if det not in refdet and det in det_map:
            idx = det_map[det]
            theta = dt_f[idx]
            c, s = np.cos(theta), np.sin(theta)
            x_old, y_old = working_coords[i, 0], working_coords[i, 1]
            working_coords[i, 0] = x_old * c - y_old * s + dx_f[idx]
            working_coords[i, 1] = x_old * s + y_old * c + dy_f[idx]
    
    # Perform fit on working_coords
    x, y, z = working_coords[:, 0], working_coords[:, 1], working_coords[:, 2]
    ex, ey = working_coords[:, 3], working_coords[:, 4]
    ex, ey = working_coords[:, 3], working_coords[:, 4]
    ez = np.zeros_like(ex)
    params  = None
    parerr  = None
    parcov  = None
    chisq   = None
    ndof    = None
    nll     = None
    theta2  = None
    success = None
    if(cfg["fit_method"]=="CHI2"):  params,parerr,parcov,chisq,ndof,success = chi2_fit.fit_line_3d_chi2err(x,y,z,ex,ey,ez)
    elif(cfg["fit_method"]=="MLE"): params,parerr,parcov,nll,chisq,ndof,success = mle_fit.fit_line_3d_mle(x,y,z,ex,ey,ez,fixtheta2=True)
    else: print(f"unsupported version of the fit_method")
    return chisq,ndof,nll,success



def fit_misalignment_fast(events, ndet2align, nparperdet, refdet, axes):
    cfg = config.Config().map
    
    ### map only aligned detectors to 0...ndet2align indices
    aligned_detectors = [d for d in cfg["detectors"] if d not in refdet]
    det_map = {det: i for i, det in enumerate(aligned_detectors)}
    
    ### pre-extract data into numpy arrays to avoid object overhead in the loop (this is a one-time cost)
    track_data = []
    for event in events:
        for track in get_selected_tracks(event):
            t_dets = []
            t_coords = [] # [x, y, z, ex, ey]
            for det in track.detectors:
                t_dets.append(det)
                t_coords.append([
                    track.trkcls[det].xTnoGmm,
                    track.trkcls[det].yTnoGmm,
                    track.trkcls[det].zTnoGmm,
                    track.trkcls[det].xTnoGsizemm if(cfg["use_large_clserr_for_algnmnt"]) else track.trkcls[det].dxTnoGmm,
                    track.trkcls[det].yTnoGsizemm if(cfg["use_large_clserr_for_algnmnt"]) else track.trkcls[det].dyTnoGmm
                ])
            track_data.append((t_dets, np.array(t_coords)))

    ### the cost function
    def metric_function_to_minimize(params):
        dx_f, dy_f, dt_f, _ = init_params(axes,ndet2align,params)
        total_metric = 0.0
        n_tracks = len(track_data)
        for dets, coords in track_data:
            chisq,ndof,nll,success = RefitTrack_Fast(dets, coords, dx_f, dy_f, dt_f, refdet, det_map)
            #########################
            ### important !!! #######
            if(not success):
                total_metric += 1e9 # Huge penalty
                continue
            #########################
            ### collect all results
            if(cfg["fit_method"]=="CHI2"):  total_metric += chisq
            elif(cfg["fit_method"]=="MLE"): total_metric += nll
            else: print("Error: fit_method unknown/not implemented")
            #########################
        return total_metric
    
    ### apply bounds from config
    ranges = []
    if "x"     in axes: ranges.extend([(cfg["alignmentbounds"]["dx"]["min"],    cfg["alignmentbounds"]["dx"]["max"])] * ndet2align)
    if "y"     in axes: ranges.extend([(cfg["alignmentbounds"]["dy"]["min"],    cfg["alignmentbounds"]["dy"]["max"])] * ndet2align)
    if "theta" in axes: ranges.extend([(cfg["alignmentbounds"]["theta"]["min"], cfg["alignmentbounds"]["theta"]["max"])] * ndet2align)
    range_params = tuple(ranges)
    
    #################################################
    #################################################
    #################################################
    minimizer = "scipy" ### or minuit
    params = None
    parerr = None
    parcov = None
    chisq  = None
    valid  = None
    n_params = nparperdet * ndet2align
    param_names = [f"p{i}" for i in range(n_params)]
    initial_values = np.zeros(n_params) ### initial guesses
    if(minimizer=="minuit"):
        ### create the Minuit object
        m = Minuit(metric_function_to_minimize, initial_values)
        for i, name in enumerate(param_names):
            m.var2pos[name] = i
            m.limits[name] = ranges[i]
            m.errors[name] = 0.001 if((i+1)%3==0) else 0.05  ### initial step size 0.1 mrad for thetas and 10 um for x/y
        ### the miminization call
        print("Starting Minuit Migrad Alignment with Hesse errors...")
        m.migrad()  ### the gradient descent
        m.hesse()   ### to get correlations
        params = np.array(m.values)
        parerr = np.array(m.errors) ### the calculated errors from Hesse
        parcov = m.covariance.tolist()
        chisq  = m.fval
        valid  = m.valid
    elif(minimizer=="scipy"):
        print("Starting Powell Alignment with Hesse errors...")
        # result = minimize(metric_function_to_minimize,initial_values,method='Nelder-Mead',options={'xatol':1e-4,'disp':True}) # Tolerance 0.1 micron
        result = minimize(metric_function_to_minimize, initial_values, method='Powell', bounds=range_params, options={'disp': True, 'maxiter':2000})
        params = result.x
        metric = result.fun
        valid  = result.success
        m = Minuit(metric_function_to_minimize, params)
        m.hesse() # This calculates the errors at the minimum found by Nelder-Mead
        parerr = np.array(m.errors) ### the calculated errors from Hesse
    ###########################################
    ###########################################
    ###########################################
    
    return params, parerr, parcov, metric, valid
    

####################################
####################################
####################################


if __name__ == "__main__":
    # get the start time
    st = time.time()
    
    #############################################
    ### Initialize Config in the main process ###
    config.init_config(configfile, False)
    cfg = config.Config().map
    config.show_config(cfg)
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
    print(f"Files:\n{files}")
    
    
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
    histos     = {}
    NscanBins  = 50
    absRes     = 0.05
    nResBins   = 50
    nResBins2D = 80
    for det in cfg["detectors"]:
        name = f"dx_{det}"; histos.update( {name:ROOT.TH1D(name,det+";dx [mm];#sum#Deltax [mm]",NscanBins,cfg["alignmentbounds"]["dx"]["min"],cfg["alignmentbounds"]["dx"]["max"])} )
        name = f"dy_{det}"; histos.update( {name:ROOT.TH1D(name,det+";dy [mm];#sum#Deltay [mm]",NscanBins,cfg["alignmentbounds"]["dy"]["min"],cfg["alignmentbounds"]["dy"]["max"])} )
        name = f"dt_{det}"; histos.update( {name:ROOT.TH1D(name,det+";d#theta [rad];#sum#Deltar [mm]",NscanBins,cfg["alignmentbounds"]["theta"]["min"],cfg["alignmentbounds"]["theta"]["max"])} )
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
    
    ### Correctly map only aligned detectors to 0...ndet2align indices
    aligned_detectors = [d for d in cfg["detectors"] if d not in refdet]
    det_map = {det: i for i, det in enumerate(aligned_detectors)}
    
    ### save all relevant events with only the good tracks
    events      = []
    chisqtot0   = 0
    chisq0      = 0
    nll0        = 0
    allevents   = 0
    alltracks   = 0
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
                
                selected_tracks = get_selected_tracks(event)
                for track in selected_tracks:
                    t_dets = []
                    t_coords = [] # [x, y, z, ex, ey]
                    for det in track.detectors:
                        dx,dy = utils.res_track2cluster(det,track.detectors,track.points,track.direction,track.centroid)
                        histos[f"h_residual_xy_{det}"].Fill(dx,dy)
                        histos[f"h_residual_xy_mid_{det}"].Fill(dx,dy)
                        histos[f"h_residual_x_{det}"].Fill(dx)
                        histos[f"h_residual_y_{det}"].Fill(dy)
                        histos[f"h_residual_x_mid_{det}"].Fill(dx)
                        histos[f"h_residual_y_mid_{det}"].Fill(dy)
                        histos[f"h_response_x_{det}"].Fill(dx/track.trkcls[det].xTnoGsizemm if(cfg["use_large_clserr_for_algnmnt"]) else dx/track.trkcls[det].dxTnoGmm)
                        histos[f"h_response_y_{det}"].Fill(dy/track.trkcls[det].yTnoGsizemm if(cfg["use_large_clserr_for_algnmnt"]) else dy/track.trkcls[det].dyTnoGmm)
                        t_dets.append(det)
                        t_coords.append([track.trkcls[det].xTnoGmm,
                                         track.trkcls[det].yTnoGmm,
                                         track.trkcls[det].zTnoGmm,
                                         track.trkcls[det].xTnoGsizemm if(cfg["use_large_clserr_for_algnmnt"]) else track.trkcls[det].dxTnoGmm,
                                         track.trkcls[det].yTnoGsizemm if(cfg["use_large_clserr_for_algnmnt"]) else track.trkcls[det].dyTnoGmm
                                     ])

                    n_params = nparperdet * ndet2align
                    dx_f, dy_f, dt_f, _ = init_params(axes, ndet2align, np.zeros(n_params))
                    chisq,ndof,nll,success = RefitTrack_Fast(t_dets, np.array(t_coords), dx_f, dy_f, dt_f, refdet, det_map)
                    chi2dof = chisq/ndof
                    
                    ### TODO: this is new
                    if(not success): continue
                    
                    ### count and proceed
                    if(ngoodtracks%25==0 and ngoodtracks>0): print(f"Added {ngoodtracks} tracks")
                    
                    ngoodtracks   += 1
                    evtgoodtracks += 1 
                    chisq0        += chi2dof
                    nll0          += nll if(nll is not None and cfg["fit_method"]=="MLE") else 0
                    chisqtot0     += chisq
                
                if(evtgoodtracks>0):
                    minevt = objects.MinimalEvent(event.trigger,event.tracks)
                    events.append(minevt)

    if(ngoodtracks<cfg["alignmentmintrks"]):
        print(f'Too few tracks collected ({ngoodtracks}) for the chi2/dof cut of maxchi2align={cfg["maxchi2align"]} --> try to increase it in the config file.')
        print("Quitting")
        quit()
    chisq0 = chisq0/ngoodtracks
    nll0   = nll0/ngoodtracks
    print(f"Done collecting {ngoodtracks} tracks (out of {alltracks}) in {allevents} events, or {float(ngoodtracks)/float(allevents):.3f} trks/evt) with chisqtot0={chisqtot0} and nll0={nll0}. Now going to fit misalignments")
    ### save histos
    fOut = ROOT.TFile(tfilenamein.replace(".root","_aligment_scan.root"),"RECREATE")
    fOut.cd()
    for hname,hist in histos.items(): hist.Write()
    fOut.Write()
    fOut.Close()
    ###################   
    
    
    #######################
    ### Run the fit !!! ###
    ### events is already not including the 
    #######################
    params, parerr, parcov, metric, success = fit_misalignment_fast(events,ndet2align,nparperdet,refdet,axes)
    

    ########################
    ### and now check it ###
    ########################
    chisqtot1   = 0
    chisq1      = 0
    nll1        = 0 
    allevents1  = 0
    ngoodtracks = 0
    dxFinal,dyFinal,thetaFinal,nparperdet = init_params(axes,ndet2align,params)
    for event in events:
        selected_tracks = get_selected_tracks(event)
        for track in selected_tracks:
            t_dets = []
            t_coords = [] # [x, y, z, ex, ey]
            for det in track.detectors:
                t_dets.append(det)
                t_coords.append([track.trkcls[det].xTnoGmm,
                                 track.trkcls[det].yTnoGmm,
                                 track.trkcls[det].zTnoGmm,
                                 track.trkcls[det].xTnoGsizemm if(cfg["use_large_clserr_for_algnmnt"]) else track.trkcls[det].dxTnoGmm,
                                 track.trkcls[det].yTnoGsizemm if(cfg["use_large_clserr_for_algnmnt"]) else track.trkcls[det].dyTnoGmm
                             ])
            n_params = nparperdet * ndet2align
            chisq,ndof,nll,success = RefitTrack_Fast(t_dets, np.array(t_coords), dxFinal,dyFinal,thetaFinal, refdet, det_map)
            
            #############################
            ### important!!! ############
            if(not success): continue ###
            #############################
            
            nll1        += nll if(nll is not None and cfg["fit_method"]=="MLE") else 0
            chi2dof     = chisq/ndof
            ngoodtracks += 1
            chisq1      += chi2dof
            chisqtot1   += chisq

    nll1   = nll1/ngoodtracks
    chisq1 = chisq1/ngoodtracks
    
    
    ### sumarize
    print("\n----------------------------------------")
    print(f"Alignment axes: {axes}")
    if(len(refdet)>0): print(f"Reference detectors: {refdet}")
    else:              print(f"No reference detector")
    print(f"Events used: {len(events)} out of {allevents}")
    print(f"Tracks used: {ngoodtracks}")
    print(f"Success?     {success}")
    if(cfg["fit_method"]=="MLE"): print(f"nll:     {nll1:3f} (original: {nll0:3f})")
    print(f"chi2:        {chisqtot1:3f} (original: {chisqtot0:3f})")
    print(f"theta final: {thetaFinal}")
    print(f"Parameters:  {params}")
    print(f"Errors:      {parerr}")
    print("----------------------------------------\n")
    salignment = "misalignment = "
    k = 0
    for det in cfg["detectors"]:
        if(det in refdet):
            salignment += f'{det}:dx={cfg["misalignment"][det]["dx"]:.2E},dy={cfg["misalignment"][det]["dy"]:.2E},theta={cfg["misalignment"][det]["theta"]:.2E} '
        else:
            dx = dxFinal[k]    + cfg["misalignment"][det]["dx"]
            dy = dyFinal[k]    + cfg["misalignment"][det]["dy"]
            dt = thetaFinal[k] + cfg["misalignment"][det]["theta"]
            salignment += f"{det}:dx={dx:.2E},dy={dy:.2E},theta={dt:.2E} "
            k += 1
    print(salignment)
    
    # get the end time
    et = time.time()
    # get the execution time
    elapsed_time = et - st
    print(f'ֿֿ\nExecution time: {elapsed_time}, seconds')

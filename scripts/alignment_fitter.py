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
        if(track.maxcls>cfg["cut_maxcls"]): return False
        if(not selections.pass_geoacc_selection(track)): return False
    return True


def get_selected_tracks(event):
    tracks = []
    for track in event.tracks:
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
    params, parerr, parcov, chisq, ndof, success = chi2_fit.fit_line_3d_chi2err(x, y, z, ex, ey, ez)
    
    # Calculate residuals
    x0, mx, y0, my = params
    fit_x = x0 + mx * z
    fit_y = y0 + my * z
    dx_res = x - fit_x
    dy_res = y - fit_y
    dabs = np.mean(np.sqrt(dx_res**2 + dy_res**2))
    
    return chisq, ndof, dabs, dx_res, dy_res



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
            track_data.append((t_dets, np.array(t_coords), track.params))

    dorefit=True

    def metric_function_to_minimize(params):
        dx_f, dy_f, dt_f, _ = init_params(axes, ndet2align, params)
        total_chisq = 0.0
        n_tracks = len(track_data)
        for dets, coords, trackpars in track_data:
            chisq = 0
            if(dorefit):
                chisq, _, _, _, _ = RefitTrack_Fast(dets, coords, dx_f, dy_f, dt_f, refdet, det_map)
            else:
                x_aligned = coords[:,0].copy()
                y_aligned = coords[:,1].copy()
                z = coords[:,2]
                ex = coords[:,3]
                ey = coords[:,4]
                
                for i, det in enumerate(dets):
                    if det not in refdet and det in det_map:
                        idx = det_map[det]
                        theta = dt_f[idx]
                        c, s = np.cos(theta), np.sin(theta)
                        x_old, y_old = x_aligned[i], y_aligned[i]
                        x_aligned[i] = x_old * c - y_old * s + dx_f[idx]
                        y_aligned[i] = x_old * s + y_old * c + dy_f[idx]
                fitx = trackpars[0]+trackpars[1]*(z+cfg["zGlobalOffset"]) - cfg["xGlobalOffset"]
                fity = trackpars[2]+trackpars[3]*(z+cfg["zGlobalOffset"]) - cfg["yGlobalOffset"]
                res_x = fitx-x_aligned
                res_y = fity-y_aligned
                # print(f"z={z}")
                # print(f"x={x_aligned} --> {fitx}")
                # print(f"y={y_aligned} --> {fity}")
                # print(f"res_x={res_x},   ex={ex}")
                # print(f"res_y={res_y},   ey={ey}")
                chisq = np.sum((res_x**2 / ex**2) + (res_y**2 / ey**2))
            
            total_chisq += chisq
        return total_chisq
    
    ### define parameter names for Minuit
    n_params = nparperdet * ndet2align
    param_names = [f"p{i}" for i in range(n_params)]
    ### initial guesses
    initial_values = np.zeros(n_params)
    ### create the Minuit object
    m = Minuit(metric_function_to_minimize, initial_values)
    for i, name in enumerate(param_names): m.var2pos[name] = i
    
    ### apply bounds from config
    ranges = []
    if "x" in axes: ranges.extend([(cfg["alignmentbounds"]["dx"]["min"], cfg["alignmentbounds"]["dx"]["max"])] * ndet2align)
    if "y" in axes: ranges.extend([(cfg["alignmentbounds"]["dy"]["min"], cfg["alignmentbounds"]["dy"]["max"])] * ndet2align)
    if "theta" in axes: ranges.extend([(cfg["alignmentbounds"]["theta"]["min"], cfg["alignmentbounds"]["theta"]["max"])] * ndet2align)
    for i, name in enumerate(param_names):
        m.limits[name] = ranges[i]
        m.errors[name] = 0.0001 if((i+1)%3==0) else 0.01  ### initial step size 0.1 mrad for thetas and 10 um for x/y
    ### the miminization call
    print("Starting Minuit Migrad Alignment...")
    
    
    # # --- DEBUG SIGN CHECK ---
    # # Manually test the chi2 with a +/- 100 micron shift on ALPIDE_3
    # test_val = 0.05 # 50 microns
    # test_params_pos = np.zeros(n_params)
    # test_params_neg = np.zeros(n_params)
    # # Assuming ALPIDE_0 dx is the 1th param (index 0) when refdet=ALPIDE_0
    # print(f"params={initial_values}")
    # test_params_pos[0] = test_val
    # test_params_neg[0] = -test_val
    # print(f"params_pos={test_params_pos}")
    # print(f"params_neg={test_params_neg}")
    # chi2_zero = metric_function_to_minimize(initial_values)
    # chi2_pos  = metric_function_to_minimize(test_params_pos)
    # chi2_neg  = metric_function_to_minimize(test_params_neg)
    # print(f"DEBUG SIGN CHECK (ALPIDE_3):")
    # print(f"  Chi2 @ 0.00 mm: {chi2_zero}")
    # print(f"  Chi2 @ +0.10 mm: {chi2_pos}")
    # print(f"  Chi2 @ -0.10 mm: {chi2_neg}")
    # if chi2_pos > chi2_zero and chi2_neg > chi2_zero:
    #     print("WARNING: Both directions increase Chi2. Track absorption is likely the culprit.")
    # elif chi2_pos < chi2_zero:
    #     print("RESULT: Positive shift is correct direction.")
    # else:
    #     print("RESULT: Negative shift is correct direction.")
    # # -------------------------
    
    
    m.migrad()  ### the gradient descent
    m.hesse()   ### to get correlations
    params = np.array(m.values)
    parerr = np.array(m.errors) ### the calculated errors from Hesse
    parcov = m.covariance.tolist()
    chisq  = m.fval
    valid  = m.valid
    return params, parerr, parcov, chisq, valid
    

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
    histos = {}
    NscanBins = 50
    absRes    = 0.05
    nResBins  = 50
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
    events = []
    chisqtot0 = 0
    chisq0 = 0
    dabs0  = 0
    dX0    = 0
    dY0    = 0
    allevents = 0
    alltracks = 0
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
                    chisq,ndof,dabs,dX,dY = RefitTrack_Fast(t_dets, np.array(t_coords), dx_f, dy_f, dt_f, refdet, det_map)
                    chi2dof = chisq/ndof
                    
                    ### count and proceed
                    if(ngoodtracks%25==0 and ngoodtracks>0): print(f"Added {ngoodtracks} tracks")
                    
                    ngoodtracks   += 1
                    evtgoodtracks += 1 
                    chisq0 += chi2dof
                    dabs0  += dabs
                    chisqtot0 += chisq
                
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
    print(f"Done collecting {ngoodtracks} tracks (out of {alltracks}) in {allevents} events, or {float(ngoodtracks)/float(allevents):.3f} trks/evt) with chisqtot0={chisqtot0}. Now going to fit misalignments")
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
    params, parerr, parcov, result, success = fit_misalignment_fast(events,ndet2align,nparperdet,refdet,axes)
    

    ########################
    ### and now check it ###
    ########################
    chisqtot1 = 0
    chisq1 = 0
    dabs1  = 0
    dX1    = 0
    dY1    = 0
    allevents1 = 0
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
            chisq,ndof,dabs,dX,dY = RefitTrack_Fast(t_dets, np.array(t_coords), dxFinal,dyFinal,thetaFinal, refdet, det_map)
            
            chi2dof = chisq/ndof
            ngoodtracks += 1
            chisq1 += chi2dof
            dabs1  += dabs
            chisqtot1 += chisq
            
    chisq1 = chisq1/ngoodtracks
    dabs1  = dabs1/ngoodtracks
    
    
    print(f"Parameters: {params}")
    print(f"Errors: {parerr}")
    # print(f"Covariance: {parcov}")
    
    ### sumarize
    print("\n----------------------------------------")
    print(f"Alignment axes: {axes}")
    if(len(refdet)>0): print(f"Reference detectors: {refdet}")
    else:              print(f"No reference detector")
    print(f"Events used: {len(events)} out of {allevents}")
    print(f"Tracks used: {ngoodtracks}")
    print(f"Success? {success}")
    print(f"chi2: {chisqtot1:3f} (original: {chisqtot0:3f})")
    print(f"dabs: {dabs1:4f} (original: {dabs0:4f})")
    print(f"dx final   : {dxFinal}")
    print(f"dy final   : {dyFinal}")
    print(f"theta final: {thetaFinal}")
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
            # salignment += f"{det}:dx={dxFinal[k]:.2E},dy={dyFinal[k]:.2E},theta={thetaFinal[k]:.2E} "
            # salignment += f"{det}:dx={dxf:.2E},dy={dyf:.2E},theta={dtf:.2E} "
            k += 1
    print(salignment)
    
    # get the end time
    et = time.time()
    # get the execution time
    elapsed_time = et - st
    print(f'ֿֿ\nExecution time: {elapsed_time}, seconds')

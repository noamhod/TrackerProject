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
from scipy.optimize import curve_fit,basinhopping
import pickle
from pathlib import Path
import ctypes
import random
from line_profiler import LineProfiler

import argparse
parser = argparse.ArgumentParser(description='serial_analyzer.py...')
parser.add_argument('-conf', metavar='config file', required=True,  help='full path to config file')
parser.add_argument('-mult', metavar='multi run?',  required=False, help='is this a multirun? [0/1]')
parser.add_argument('-wave', metavar='fill waves?', required=False, help='fill wave histos? [0/1]')
parser.add_argument('-eudq', metavar='fill eudaq?', required=False, help='fill eudaq tree? [0/1]')
parser.add_argument('-toys', metavar='fill toys?',  required=False, help='fill toys histos? [0/1]')
argus = parser.parse_args()
configfile = argus.conf
ismutirun  = argus.mult if(argus.mult is not None and int(argus.mult)==1) else False
iswavehst  = argus.wave if(argus.wave is not None and int(argus.wave)==1) else False
weudaqout  = argus.eudq if(argus.eudq is not None and int(argus.eudq)==1) else False
dotoyhsit  = argus.toys if(argus.toys is not None and int(argus.toys)==1) else False
print(f"ismutirun={ismutirun}")
print(f"iswavehst={iswavehst}")
print(f"weudaqout={weudaqout}")
print(f"dotoyhsit={dotoyhsit}")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tracker_lib import config
from tracker_lib import objects, candidate, selections, evtdisp, hists, counters, utils, svd_fit, chi2_fit, mle_fit


ROOT.gROOT.SetBatch(1)
ROOT.gStyle.SetOptFit(0)
ROOT.gStyle.SetOptStat(0)
# ROOT.gStyle.SetPalette(ROOT.kRust)
# ROOT.gStyle.SetPalette(ROOT.kSolar)
# ROOT.gStyle.SetPalette(ROOT.kInvertedDarkBodyRadiator)
ROOT.gStyle.SetPalette(ROOT.kDarkBodyRadiator)
# ROOT.gStyle.SetPalette(ROOT.kRainbow)
ROOT.gStyle.SetPadBottomMargin(0.15)
ROOT.gStyle.SetPadLeftMargin(0.13)
ROOT.gStyle.SetPadRightMargin(0.16)
ROOT.gStyle.SetGridColor(ROOT.kGray)
ROOT.gStyle.SetGridWidth(1)
# ROOT.gStyle.SetImageScaling(2.)

ROOT.gErrorIgnoreLevel = ROOT.kError
# ROOT.gErrorIgnoreLevel = ROOT.kWarning

B  = None
LB = None
mm2m = 1e-3

rnd = ROOT.TRandom()
rnd.SetSeed()


def get_error_graph(name,h0,hh,hl):
    cfg = config.Config().map
    gx, gy, exl, exh, eyl, eyh = [], [], [], [], [], []
    for b in range(1,h0.GetNbinsX()+1):
        x0 = h0.GetXaxis().GetBinCenter(b)
        xd = x0-h0.GetXaxis().GetBinLowEdge(b)
        xu = h0.GetXaxis().GetBinUpEdge(b)-x0
        gx.append(x0)
        exl.append(xd)
        exh.append(xu)
        y0 = h0.GetBinContent(b)
        yh = hh.GetBinContent(b)
        yl = hl.GetBinContent(b)
        if(cfg["dbg"]): print(f"{name} - b={b}:  yl={yl} <<< y0={y0} <<< yh={yh}")
        gy.append(y0)
        eyl.append(y0-yl)
        eyh.append(yh-y0)
    gr = ROOT.TGraphAsymmErrors(len(gx), np.array(gx), np.array(gy), np.array(exl), np.array(exh), np.array(eyl), np.array(eyh))
    gr.SetFillColorAlpha(ROOT.kGray+2,0.3)
    gr.SetLineColor(ROOT.kGray+2)
    gr.SetMarkerStyle(0)
    gr.SetName(name)
    return gr
    

def get_pz_from_fit(theta_yz, err_thet_yz=0):
    cfg = config.Config().map
    phi = theta_yz
    if(err_thet_yz>0):
        e = rnd.Gaus(0,err_thet_yz)
        while(e<-err_thet_yz or e>err_thet_yz): e = rnd.Gaus(0,err_thet_yz)
        phi = theta_yz + e
    pz = (0.3 * B * LB)/math.sin( phi )
    return pz


def get_toy(toy,T,htH,htL,err,hpzH=None,hpzL=None,err_thet_yz=0,fOut=None):
    ### get the misalignment
    e = rnd.Gaus(0,err)
    while(e<-err or e>err): e = rnd.Gaus(0,err)
    ### clone and reset
    ht1 = htH.Clone(f"ht1_{toy}")
    ht1.Reset()
    hp1 = None
    if(hpzH is not None): 
        hp1 = hpzH.Clone(f"hp1_{toy}")
        hp1.Reset()
    
    ### fil the toy histos
    for t in T:
        t1 = t+e
        ht1.Fill(t1)
        if(hp1 is not None):
            p1 = get_pz_from_fit(t1,err_thet_yz)
            hp1.Fill( p1 )
    
    ### check if larger/smaller than the existing and update if so
    for b in range(1,ht1.GetNbinsX()+1):
        yH0 = htH.GetBinContent(b)
        yL0 = htL.GetBinContent(b)
        y1  = ht1.GetBinContent(b)
        if(y1>yH0): htH.SetBinContent(b,y1)
        if(y1<yL0): htL.SetBinContent(b,y1)
    if(fOut is None): del ht1
    ### check if larger/smaller than the existing and update if so
    if(hp1 is not None):
        for b in range(1,hp1.GetNbinsX()+1):
            yH0 = hpzH.GetBinContent(b)
            yL0 = hpzL.GetBinContent(b)
            y1  = hp1.GetBinContent(b)
            if(y1>yH0): hpzH.SetBinContent(b,y1)
            if(y1<yL0): hpzL.SetBinContent(b,y1)
        if(fOut is None): del hp1

    ### write for diagnostics
    if(fOut is not None):
        fOut.cd()
        ht1.Write()
        hp1.Write()
        
    
            

def h1h2max(h1,h2):
    hmax = -1
    y1 = h1.GetMaximum()
    y2 = h2.GetMaximum()
    hmax = y1 if(y1>y2) else y2
    return hmax

def fit1(h,col,xmin,xmax):
    g1 = ROOT.TF1("g1", "gaus", xmin,xmax)
    g1.SetLineColor(col)
    h.Fit(g1,"EMRS")
    chi2dof = g1.GetChisquare()/g1.GetNDF() if(g1.GetNDF()>0) else -1
    print("g1 chi2/Ndof=",chi2dof)
    return g1

# def refit(track):
#     cfg = config.Config().map
#     hough_coords = track.hough_coords
#     clusters = track.trkcls
#     detectors = track.detectors
#     seed_x = {}
#     seed_y = {}
#     seed_z = {}
#     seed_dx = {}
#     seed_dy = {}
#     # for det in cfg["detectors"]:
#     for det in track.detectors:
#         ### first get the track cluster in this det at the EUDAQ coordinates
#         rCLS = [clusters[det].xmm, clusters[det].ymm, clusters[det].zmm]
#         ### then transform (including alignment) to the real space
#         rCLS = transform_to_real_space(rCLS,det,algn=True)
#         seed_x.update({  det : rCLS[0]  })
#         seed_y.update({  det : rCLS[1]  })
#         seed_z.update({  det : rCLS[2]  })
#         seed_dx.update({ det : clusters[det].xTsizemm if(cfg["use_large_clserr_for_algnmnt"]) else clusters[det].dxTmm })
#         seed_dy.update({ det : clusters[det].yTsizemm if(cfg["use_large_clserr_for_algnmnt"]) else clusters[det].dyTmm })
#     ### then prepare for refit
#     vtx  = [cfg["xVtx"],cfg["yVtx"],cfg["zVtx"]]    if(cfg["doVtx"]) else []
#     evtx = [cfg["exVtx"],cfg["eyVtx"],cfg["ezVtx"]] if(cfg["doVtx"]) else []
#     points_SVD, errors_SVD  = candidate.SVD_candidate(seed_x,seed_y,seed_z,seed_dx,seed_dy,vtx,evtx)
#     points_Chi2,errors_Chi2 = candidate.Chi2_candidate(seed_x,seed_y,seed_z,seed_dx,seed_dy,vtx,evtx)
#
#     chisq     = None
#     ndof      = None
#     direction = None
#     centroid  = None
#     params    = None
#     parerr    = None
#     parcov    = None
#     nll       = None
#     theta2    = None
#     success   = None
#     ### SVD fit
#     if(cfg["fit_method"]=="SVD"):
#         chisq,ndof,direction,centroid = svd_fit.fit_3d_SVD(points_SVD,errors_SVD)
#         params = utils.get_pars_from_centroid_and_direction(centroid,direction)
#         success = True
#     ### chi2 fit
#     if(cfg["fit_method"]=="CHI2"):
#         chisq,ndof,direction,centroid,params,parerr,parcov,success = fit_chi2.fit_3d_chi2err(points_Chi2,errors_Chi2,par_guess)
#     ### MLE fit
#     if(cfg["fit_method"]=="MLE"):
#         theta2,nll,chisq,ndof,direction,centroid,params,parerr,parcov,success = fit_mle.fit_3d_mle(points_Chi2,errors_Chi2)
#
#     ### set the track
#     track = objects.Track(detectors,clusters,points_SVD,errors_SVD,chisq,ndof,direction,centroid,params,success,hough_coords)
#     return track




def get_wave(name,z,k,thetamin,thetamax):
    ### rho = k*sin(theta) + z*cos(theta)
    func = ROOT.TF1(f"func_{name}","[1]*sin(x)+[0]*cos(x)",thetamin,thetamax,2)
    func.SetParameter(0,z)
    func.SetParameter(1,k)
    return func

def get_par_lin(theta_k,rho_k): ### theta and rho from Hough transform
    if(math.sin(theta_k)==0):
        print(f"in get_par_lin, sin(theta)=0: quitting.")
        quit()
    if(math.tan(theta_k)==0):
        print(f"in get_par_lin, 1/tan(theta)=0: quitting.")
        quit()
    AK = -1./math.tan(theta_k)
    BK = rho_k/math.sin(theta_k)
    # print(f"theta_k={theta_k}, rho_k={rho_k} --> AK={AK}, BK={BK}")
    return AK,BK

def find_waves_intersect(k1,z1,k2,z2):
    dk = (k1-k2) if(abs(k1-k2)>1e-15) else 1e15*np.sign(k1-k2)
    theta = math.atan2((z2-z1),dk) # the arc tangent of (y/x) in radians
    rho   = k1*math.sin(theta) + z1*math.cos(theta)
    # print(f"k1={k1},z1={z1}, k2={k2},z1={z2} --> theta={theta},rho={rho}")
    return theta,rho
    
def fill_pair(a,b,track,hx,hy):
    pair = [f"ALPIDE_{a}",f"ALPIDE_{b}"]
    rA = [track.trkcls[pair[0]].xTnoGmm,track.trkcls[pair[0]].yTnoGmm,track.trkcls[pair[0]].zTnoGmm]
    rB = [track.trkcls[pair[1]].xTnoGmm,track.trkcls[pair[1]].yTnoGmm,track.trkcls[pair[1]].zTnoGmm]
    thetax,rhox = find_waves_intersect(rA[0],rA[2],rB[0],rB[2])
    thetay,rhoy = find_waves_intersect(rA[1],rA[2],rB[1],rB[2])
    hx.Fill(thetax,rhox)
    hy.Fill(thetay,rhoy)

def k_of_z(z,AK,BK):
    k = AK*z + BK
    # print(f"AK={AK}, BK={BK}, z={z} --> k={k}")
    return k

def get_edges_from_theta_rho_corners(det,theta_x,rho_x,theta_y,rho_y):
    cfg = config.Config().map
    xmin = +1e20
    xmax = -1e20
    ymin = +1e20
    ymax = -1e20
    for i in range(2):
        AX,BX = get_par_lin(theta_x[i],rho_x[i])
        AY,BY = get_par_lin(theta_y[i],rho_y[i])
        zdet = cfg["rdetectors"][det][2]
        XX = k_of_z(zdet,AX,BX)
        YY = k_of_z(zdet,AY,BY)
        # print(f"get_edges_from_theta_rho_corners cornere[i]: eventid={self.eventid}  -->  {det} prediction: x={XX}, y={YY}, z={zdet}")
        xmin = XX if(XX<xmin) else xmin
        xmax = XX if(XX>xmax) else xmax
        ymin = YY if(YY<ymin) else ymin
        ymax = YY if(YY>ymax) else ymax
    return xmin,xmax,ymin,ymax




def book_histos():
    cfg = config.Config().map
    
    ### some histos
    histos = {}
    
    trkarr = hists.GetLogBinning(75,0.5,3000)
    ntrkarr = len(trkarr)-1

    theta1arr = hists.GetLogBinning(100,1e-13,1e-1)
    ntheta1arr = len(theta1arr)-1
    
    theta2arr = hists.GetLogBinning(100,1e-22,1e-3)
    ntheta2arr = len(theta2arr)-1
    
    histos.update({ "hTriggers": ROOT.TH1D("hTriggers",";;Triggers",2,0,2)})
    histos["hTriggers"].GetXaxis().SetBinLabel(1,"All")
    histos["hTriggers"].GetXaxis().SetBinLabel(2,"Good")
    
    histos.update( { "h_nTunnels"             : ROOT.TH1D("h_nTunnels",";N_{tunnels}/Event;Events",250,0,250) } )
    histos.update( { "h_nTunnels_log"         : ROOT.TH1D("h_nTunnels_log",";N_{tunnels}/Event;Events",ntrkarr,trkarr) } )
    histos.update( { "h_nTunnels_full"        : ROOT.TH1D("h_nTunnels_full",";N_{tunnels}/Event;Events",2000,0,20000) } )
    histos.update( { "h_nTunnels_mid"         : ROOT.TH1D("h_nTunnels_mid",";N_{tunnels}/Event;Events",100,0,100) } )
    histos.update( { "h_nTunnels_zoom"        : ROOT.TH1D("h_nTunnels_zoom",";N_{tunnels}/Event;Events",40,0,40) } )
    histos.update( { "h_nSeeds"               : ROOT.TH1D("h_nSeeds",";N_{seeds}/Event;Events",250,0,250) } )
    histos.update( { "h_nSeeds_log"           : ROOT.TH1D("h_nSeeds_log",";N_{seeds}/Event;Events",ntrkarr,trkarr) } )
    histos.update( { "h_nSeeds_full"          : ROOT.TH1D("h_nSeeds_full",";N_{seeds}/Event;Events",2000,0,20000) } )
    histos.update( { "h_nSeeds_mid"           : ROOT.TH1D("h_nSeeds_mid",";N_{seeds}/Event;Events",100,0,100) } )
    histos.update( { "h_nSeeds_zoom"          : ROOT.TH1D("h_nSeeds_zoom",";N_{seeds}/Event;Events",40,0,40) } )
    histos.update( { "h_nTracks"              : ROOT.TH1D("h_nTracks",";N_{tracks}/Event;Events",250,0,250) } )
    histos.update( { "h_nTracks_log"          : ROOT.TH1D("h_nTracks_log",";N_{tracks}/Event;Events",ntrkarr,trkarr) } )
    histos.update( { "h_nTracks_full"         : ROOT.TH1D("h_nTracks_full",";N_{tracks}/Event;Events",2000,0,20000) } )
    histos.update( { "h_nTracks_mid"          : ROOT.TH1D("h_nTracks_mid",";N_{tracks}/Event;Events",100,0,100) } )
    histos.update( { "h_nTracks_zoom"         : ROOT.TH1D("h_nTracks_zoom",";N_{tracks}/Event;Events",40,0,40) } )
    histos.update( { "h_nTracks_btrfly"       : ROOT.TH1D("h_nTracks_btrfly",";N_{tracks}/Event;Events",250,0,250) } )
    histos.update( { "h_nTracks_btrfly_log"   : ROOT.TH1D("h_nTracks_btrfly_log",";N_{tracks}/Event;Events",ntrkarr,trkarr) } )
    histos.update( { "h_nTracks_btrfly_full"  : ROOT.TH1D("h_nTracks_btrfly_full",";N_{tracks}/Event;Events",2000,0,20000) } )
    histos.update( { "h_nTracks_btrfly_mid"   : ROOT.TH1D("h_nTracks_btrfly_mid",";N_{tracks}/Event;Events",100,0,100) } )
    histos.update( { "h_nTracks_btrfly_zoom"  : ROOT.TH1D("h_nTracks_btrfly_zoom",";N_{tracks}/Event;Events",40,0,40) } )
    
    histos.update({ "hChi2DoF_alowshrcls": ROOT.TH1D("hChi2DoF_alowshrcls",";#chi^{2}/N_{DoF};Tracks",200,0,50)})
    histos.update({ "hChi2DoF_zeroshrcls": ROOT.TH1D("hChi2DoF_zeroshrcls",";#chi^{2}/N_{DoF};Tracks",200,0,50)})
    histos.update({ "hChi2DoF_full_alowshrcls": ROOT.TH1D("hChi2DoF_full_alowshrcls",";#chi^{2}/N_{DoF};Tracks",200,0,cfg["cut_chi2dof"])})
    histos.update({ "hChi2DoF_full_zeroshrcls": ROOT.TH1D("hChi2DoF_full_zeroshrcls",";#chi^{2}/N_{DoF};Tracks",200,0,cfg["cut_chi2dof"])})
    histos.update({ "hChi2DoF_mid_alowshrcls": ROOT.TH1D("hChi2DoF_mid_alowshrcls",";#chi^{2}/N_{DoF};Tracks",200,0,200)})
    histos.update({ "hChi2DoF_mid_zeroshrcls": ROOT.TH1D("hChi2DoF_mid_zeroshrcls",";#chi^{2}/N_{DoF};Tracks",200,0,200)})
    histos.update({ "hChi2DoF_small_alowshrcls": ROOT.TH1D("hChi2DoF_small_alowshrcls",";#chi^{2}/N_{DoF};Tracks",100,0,20)})
    histos.update({ "hChi2DoF_small_zeroshrcls": ROOT.TH1D("hChi2DoF_small_zeroshrcls",";#chi^{2}/N_{DoF};Tracks",100,0,20)})
    histos.update({ "hChi2DoF_zoom_alowshrcls": ROOT.TH1D("hChi2DoF_zoom_alowshrcls",";#chi^{2}/N_{DoF};Tracks",200,0,5)})
    histos.update({ "hChi2DoF_zoom_zeroshrcls": ROOT.TH1D("hChi2DoF_zoon_zeroshrcls",";#chi^{2}/N_{DoF};Tracks",200,0,5)})
    
    histos.update({ "hChi2_alowshrcls": ROOT.TH1D("hChi_alowshrcls",";#chi^{2};Tracks",200,0,50*4)})
    histos.update({ "hChi2_zeroshrcls": ROOT.TH1D("hChi_zeroshrcls",";#chi^{2};Tracks",200,0,50*4)})
    histos.update({ "hChi2_full_alowshrcls": ROOT.TH1D("hChi2_full_alowshrcls",";#chi^{2};Tracks",200,0,cfg["cut_chi2dof"]*4)})
    histos.update({ "hChi2_full_zeroshrcls": ROOT.TH1D("hChi2_full_zeroshrcls",";#chi^{2};Tracks",200,0,cfg["cut_chi2dof"]*4)})
    histos.update({ "hChi2_mid_alowshrcls": ROOT.TH1D("hChi2_mid_alowshrcls",";#chi^{2};Tracks",200,0,200*4)})
    histos.update({ "hChi2_mid_zeroshrcls": ROOT.TH1D("hChi2_mid_zeroshrcls",";#chi^{2};Tracks",200,0,200*4)})
    histos.update({ "hChi2_small_alowshrcls": ROOT.TH1D("hChi2_small_alowshrcls",";#chi^{2};Tracks",100,0,20*4)})
    histos.update({ "hChi2_small_zeroshrcls": ROOT.TH1D("hChi2_small_zeroshrcls",";#chi^{2};Tracks",100,0,20*4)})
    histos.update({ "hChi2_zoom_alowshrcls": ROOT.TH1D("hChi2_zoom_alowshrcls",";#chi^{2};Tracks",200,0,5*4)})
    histos.update({ "hChi2_zoom_zeroshrcls": ROOT.TH1D("hChi2_zoon_zeroshrcls",";#chi^{2};Tracks",200,0,5*4)})

    histos.update({ "h_MLE_theta1_linx_before_cuts": ROOT.TH1D("h_MLE_theta1_linx_before_cuts",";#theta_{RMS} [rad];Tracks",100,2e-7,1e-6)})
    histos.update({ "h_MLE_theta2_linx_before_cuts": ROOT.TH1D("h_MLE_theta2_linx_before_cuts",";MLE #theta^{2} [rad^{2}];Tracks",100,5e-14,1e-12)})
    histos.update({ "h_MLE_theta1_linx_after_cuts":  ROOT.TH1D("h_MLE_theta1_linx_after_cuts",";#theta_{RMS} [rad];Tracks",100,2e-7,1e-6)})
    histos.update({ "h_MLE_theta2_linx_after_cuts":  ROOT.TH1D("h_MLE_theta2_linx_after_cuts",";MLE #theta^{2} [rad^{2}];Tracks",100,5e-14,1e-12)})
    
    histos.update({ "h_MLE_theta1_logx_before_cuts": ROOT.TH1D("h_MLE_theta1_logx_before_cuts",";#theta_{RMS} [rad];Tracks",ntheta1arr,theta1arr)})
    histos.update({ "h_MLE_theta2_logx_before_cuts": ROOT.TH1D("h_MLE_theta2_logx_before_cuts",";MLE #theta^{2} [rad^{2}];Tracks",ntheta2arr,theta2arr)})
    histos.update({ "h_MLE_theta1_logx_after_cuts": ROOT.TH1D("h_MLE_theta1_logx_after_cuts",";#theta_{RMS} [rad];Tracks",ntheta1arr,theta1arr)})
    histos.update({ "h_MLE_theta2_logx_after_cuts": ROOT.TH1D("h_MLE_theta2_logx_after_cuts",";MLE #theta^{2} [rad^{2}];Tracks",ntheta2arr,theta2arr)})
    
    histos.update({ "hPf_vs_dExit": ROOT.TH2D("hPf_vs_dExit",";d_{exit} [mm];p(#theta(fit)) [GeV];Tracks",50,0,+35, 50,0,10) })
    histos.update({ "hPd_vs_dExit": ROOT.TH2D("hPd_vs_dExit",";d_{exit} [mm];p(#theta(d_{exit}) [GeV];Tracks",50,0,+35, 50,0,10) })
    histos.update({ "hPr_vs_dExit": ROOT.TH2D("hPr_vs_dExit",";d_{exit} [mm];p(#theta(r) [GeV];Tracks",50,0,+35, 50,0,10) })

    histos.update({ "hPf_vs_thetaf": ROOT.TH2D("hPf_vs_thetaf",";#theta_{yz}(fit) [rad];p(#theta(fit)) [GeV];Tracks",50,0,0.05, 50,0,10) })
    histos.update({ "hPd_vs_thetad": ROOT.TH2D("hPd_vs_thetad",";#theta_{yz}(d_{exit}) [rad];p(#theta(d_{exit})) [GeV];Tracks",50,0,0.05, 50,0,10) })
    histos.update({ "hPr_vs_thetar": ROOT.TH2D("hPr_vs_thetar",";#theta_{yz}(r) [rad];p(#theta(r)) [GeV];Tracks",50,0,0.05, 50,0,10) })

    histos.update({ "hDexit_vs_thetaf": ROOT.TH2D("hDexit_vs_thetaf",";#theta_{yz}(fit) [rad];d_{exit} [mm];Tracks",50,0,0.05, 50,0,+35) })
    histos.update({ "hDexit_vs_thetad": ROOT.TH2D("hDexit_vs_thetad",";#theta_{yz}(d_{exit}) [rad];d_{exit} [mm];Tracks",50,0,0.05, 50,0,+35) })
    histos.update({ "hDexit_vs_thetar": ROOT.TH2D("hDexit_vs_thetar",";#theta_{yz}(r) [rad];d_{exit} [mm];Tracks",50,0,0.05, 50,0,+35) })
    
    histos.update({ "hThetad_vs_thetaf": ROOT.TH2D("hThetad_vs_thetaf",";#theta_{yz}(fit) [rad];#theta(d_{exit}) [rad];Tracks",50,0,0.05, 50,0,0.05) })
    histos.update({ "hThetar_vs_thetaf": ROOT.TH2D("hThetar_vs_thetaf",";#theta_{yz}(fit) [rad];#theta(r) [rad];Tracks",50,0,0.05, 50,0,0.05) })

    histos.update({ "hF_before_cuts": ROOT.TH2D("hF_before_cuts","Dipole flange plane;x [mm];y [mm];Extrapolated Tracks",120,-80,+80, 120,-70,+90) })
    histos.update({ "hF_after_cuts":  ROOT.TH2D("hF_after_cuts","Dipole flange plane;x [mm];y [mm];Extrapolated Tracks",120,-80,+80, 120,-70,+90) })
    
    # histos.update({ "hD_before_cuts": ROOT.TH2D("hD_before_cuts","Dipole exit plane;x [mm];y [mm];Extrapolated Tracks",120,-80,+80, 120,-70,+90) })
    # histos.update({ "hD_after_cuts":  ROOT.TH2D("hD_after_cuts","Dipole exit plane;x [mm];y [mm];Extrapolated Tracks",120,-80,+80, 120,-70,+90) })
    histos.update({ "hD_before_cuts": ROOT.TH2D("hD_before_cuts","Dipole exit plane;x [mm];y [mm];Extrapolated Tracks",200,-80,+80, 200,-70,+90) })
    histos.update({ "hD_after_cuts":  ROOT.TH2D("hD_after_cuts","Dipole exit plane;x [mm];y [mm];Extrapolated Tracks",200,-80,+80, 200,-70,+90) })
    histos.update({ "hD_zoomout_before_cuts": ROOT.TH2D("hD_zoomout_before_cuts","Dipole exit plane;x [mm];y [mm];Extrapolated Tracks",120,-1000,+1000, 120,-1000,+1000) })
    histos.update({ "hD_zoomout_after_cuts":  ROOT.TH2D("hD_zoomout_after_cuts","Dipole exit plane;x [mm];y [mm];Extrapolated Tracks",120,-1000,+1000, 120,-1000,+1000) })
    histos.update({ "hD_zoomin_before_cuts":  ROOT.TH2D("hD_zoomin_before_cuts","Dipole exit plane;x [mm];y [mm];Extrapolated Tracks",200,1.2*cfg["xDipoleExitMin"],1.2*cfg["xDipoleExitMax"], 200,1.1*cfg["yDipoleExitMin"],1.1*cfg["yDipoleExitMax"]) })
    histos.update({ "hD_zoomin_after_cuts":   ROOT.TH2D("hD_zoomin_after_cuts","Dipole exit plane;x [mm];y [mm];Extrapolated Tracks", 200,1.2*cfg["xDipoleExitMin"],1.2*cfg["xDipoleExitMax"], 200,1.1*cfg["yDipoleExitMin"],1.1*cfg["yDipoleExitMax"]) })
    
    
    histos.update({ "hW_before_cuts": ROOT.TH2D("hW_before_cuts","Vacuum window plane;x [mm];y [mm];Extrapolated Tracks",120,-70,+70, 120,50,+190) })
    histos.update({ "hW_after_cuts":  ROOT.TH2D("hW_after_cuts","Vacuum window plane;x [mm];y [mm];Extrapolated Tracks",120,-70,+70, 120,50,+190) })
    
    histos.update({ "hThetaf_yz": ROOT.TH1D("hThetaf_yz",";#theta_{yz}^{trk}(fit) [rad];Tracks",100,0,0.1)})
    histos.update({ "hThetad_yz": ROOT.TH1D("hThetad_yz",";#theta_{yz}(d_{exit}) [rad];Tracks",100,0,0.1)})
    histos.update({ "hThetar_yz": ROOT.TH1D("hThetar_yz",";#theta_{yz}(r) [rad];Tracks",100,0,0.1)})
    
    histos.update({ "hTheta_xz_before_cuts": ROOT.TH1D("hTheta_xz_before_cuts",";#theta_{xz}^{trk} [rad];Tracks",50,-0.015,0.015)})
    histos.update({ "hTheta_xz_after_cuts":  ROOT.TH1D("hTheta_xz_after_cuts",";#theta_{xz}^{trk} [rad];Tracks",50,-0.015,0.015)})
    histos.update({ "hTheta_yz_before_cuts": ROOT.TH1D("hTheta_yz_before_cuts",";#theta_{yz}^{trk} [rad];Tracks",50,0,0.05)})
    histos.update({ "hTheta_yz_after_cuts":  ROOT.TH1D("hTheta_yz_after_cuts",";#theta_{yz}^{trk} [rad];Tracks",50,0,0.05)})
    
    histos.update({ "hTheta_xz_labframe_before_cuts": ROOT.TH1D("hTheta_xzlabframe_before_cuts",";#theta_{xz}^{lab} [rad];Tracks",50,-0.015,0.015)})
    histos.update({ "hTheta_xz_labframe_after_cuts":  ROOT.TH1D("hTheta_xzlabframe_after_cuts",";#theta_{xz}^{lab} [rad];Tracks",50,-0.015,0.015)})
    histos.update({ "hTheta_yz_labframe_before_cuts": ROOT.TH1D("hTheta_yzlabframe_before_cuts",";#theta_{yz}^{lab} [rad];Tracks",50,0,0.05)})
    histos.update({ "hTheta_yz_labframe_after_cuts":  ROOT.TH1D("hTheta_yzlabframe_after_cuts",";#theta_{yz}^{lab} [rad];Tracks",50,0,0.05)})
    
    histos.update({ "hTheta_xz_tru": ROOT.TH1D("hTheta_xz_tru",";#theta_{xz} [rad];Tracks",100,-0.01,0.01)})
    histos.update({ "hTheta_yz_tru": ROOT.TH1D("hTheta_yz_tru",";#theta_{yz} [rad];Tracks",100,0,0.035)})
    
    histos.update({ "hTheta_xz_tru_all": ROOT.TH1D("hTheta_xz_tru_all",";#theta_{xz} [rad];Tracks",100,-0.006,0.006)})
    histos.update({ "hTheta_yz_tru_all": ROOT.TH1D("hTheta_yz_tru_all",";#theta_{yz} [rad];Tracks",100,0,0.035)})
    
    histos.update({ "hdExit":    ROOT.TH1D("hdExit",";d_{exit} [mm];Tracks",120,-70,+90)})
    
    histos.update({ "hTheta_xz_response": ROOT.TH1D("hThetaf_xz_response",";#frac{#theta_{xz}^{rec}-#theta_{xz}^{tru}}{#theta_{xz}^{tru}};Tracks",100,-0.5,0.5)})
    histos.update({ "hTheta_yz_response": ROOT.TH1D("hThetaf_yz_response",";#frac{#theta_{yz}^{rec}-#theta_{yz}^{tru}}{#theta_{yz}^{tru}};Tracks",100,-0.05,0.05)})
    histos.update({ "hD_x_response": ROOT.TH1D("hD_x_response",";#frac{x_{vtx}^{rec}-x_{vtx}^{tru}}{x_{vtx}^{tru}};Tracks",100,-0.5,0.5)})
    histos.update({ "hD_y_response": ROOT.TH1D("hD_y_response",";#frac{y_{vtx}^{rec}-y_{vtx}^{tru}}{y_{vtx}^{tru}};Tracks",100,-0.5,0.5)})
    
    histos.update({ "hPf": ROOT.TH1D("hPf",";p(fit) [GeV];Tracks",100,0,10)})
    histos.update({ "hPd": ROOT.TH1D("hPd",";p(d_{exit}) [GeV];Tracks",100,0,10)})
    histos.update({ "hPr": ROOT.TH1D("hPr",";p(r) [GeV];Tracks",100,0,10)})
    
    histos.update({ "hPf_small": ROOT.TH1D("hPf_small",";p(fit) [GeV];Tracks",50,1.5,4.5)})
    histos.update({ "hPd_small": ROOT.TH1D("hPd_small",";p(d_{exit}) [GeV];Tracks",50,1.5,4.5)})
    histos.update({ "hPr_small": ROOT.TH1D("hPr_small",";p(r) [GeV];Tracks",50,1.5,4.5)})
    
    histos.update({ "hPf_zoom": ROOT.TH1D("hPf_zoom",";p(fit) [GeV];Tracks",40,1.5,3.5)})
    histos.update({ "hPd_zoom": ROOT.TH1D("hPd_zoom",";p(d_{exit}) [GeV];Tracks",40,1.5,3.5)})
    histos.update({ "hPr_zoom": ROOT.TH1D("hPr_zoom",";p(r) [GeV];Tracks",40,1.5,3.5)})    

    thetaxmin = 0     #np.pi/2-cfg["seed_thetax_scale_mid"]*np.pi/2.
    thetaxmax = np.pi #np.pi/2+cfg["seed_thetax_scale_mid"]*np.pi/2.
    thetaymin = 0     #np.pi/2-cfg["seed_thetay_scale_mid"]*np.pi/2.
    thetaymax = np.pi #np.pi/2+cfg["seed_thetay_scale_mid"]*np.pi/2.
    minthetarhobins = 2000
    # nthetarhobins = minthetarhobins if(cfg["seed_nbins_thetarho_mid"]<minthetarhobins) else cfg["seed_nbins_thetarho_mid"]
    nthetarhobins = 500
    histos.update({ "hWaves_zx" : ROOT.TH2D("hWaves_zx",";#theta_{zx};#rho_{zx};",nthetarhobins,thetaxmin,thetaxmax,nthetarhobins,-90,90) })
    histos.update({ "hWaves_zy" : ROOT.TH2D("hWaves_zy",";#theta_{zy};#rho_{zy};",nthetarhobins,thetaymin,thetaymax,nthetarhobins,-90,90) })
    histos.update({ "hWaves_zx_intersections" : ROOT.TH2D("hWaves_zx_intersections",";#theta_{zx};#rho_{zx};",nthetarhobins,thetaxmin,thetaxmax,nthetarhobins,-90,90) })
    histos.update({ "hWaves_zy_intersections" : ROOT.TH2D("hWaves_zy_intersections",";#theta_{zy};#rho_{zy};",nthetarhobins,thetaymin,thetaymax,nthetarhobins,-90,90) })
    
    absRes   = 0.05
    nResBins = 50
    limtnl = {}
    for det in cfg["detectors"]:
        tdm = cfg["det2tdm"][det]
        if  (tdm==0): limtnl.update({det:[0.0,0.35]})
        elif(tdm==1): limtnl.update({det:[0.0,0.50]})
        elif(tdm==2): limtnl.update({det:[0.0,0.65]})
        elif(tdm==3): limtnl.update({det:[0.0,0.80]})
        elif(tdm==4): limtnl.update({det:[0.0,0.95]})
    bintnl = 60
    
    
    histos.update({ "h_cls_absdx": ROOT.TH1D("h_cls_absdx",";dx(detA,detB) [mm];Tracks",100,0,2)})
    histos.update({ "h_cls_absdy": ROOT.TH1D("h_cls_absdy",";dy(detA,detB) [mm];Tracks",100,0,5)})



    name = f"h_residual_alowshrcls_x_sml_inc"; histos.update( { name:ROOT.TH1D(name,det+";x_{trk}-x_{cls} [mm];Tracks",int(nResBins*0.6),-absRes*0.6,+absRes*0.6) } )
    name = f"h_residual_alowshrcls_y_sml_inc"; histos.update( { name:ROOT.TH1D(name,det+";y_{trk}-y_{cls} [mm];Tracks",int(nResBins*0.6),-absRes*0.6,+absRes*0.6) } )
    name = f"h_residual_alowshrcls_x_mid_inc"; histos.update( { name:ROOT.TH1D(name,det+";x_{trk}-x_{cls} [mm];Tracks",nResBins,-absRes*3,+absRes*3) } )
    name = f"h_residual_alowshrcls_y_mid_inc"; histos.update( { name:ROOT.TH1D(name,det+";y_{trk}-y_{cls} [mm];Tracks",nResBins,-absRes*3,+absRes*3) } )
    name = f"h_residual_alowshrcls_x_ful_inc"; histos.update( { name:ROOT.TH1D(name,det+";x_{trk}-x_{cls} [mm];Tracks",nResBins*2,-absRes*5,+absRes*5) } )
    name = f"h_residual_alowshrcls_y_ful_inc"; histos.update( { name:ROOT.TH1D(name,det+";y_{trk}-y_{cls} [mm];Tracks",nResBins*2,-absRes*5,+absRes*5) } )

    name = f"h_response_alowshrcls_x_sml_inc"; histos.update( { name:ROOT.TH1D(name,det+";#frac{x_{trk}-x_{cls}}{#sigma(x_{cls})};Tracks",30,-12.5,+12.5) } )
    name = f"h_response_alowshrcls_y_sml_inc"; histos.update( { name:ROOT.TH1D(name,det+";#frac{y_{trk}-y_{cls}}{#sigma(y_{cls})};Tracks",30,-12.5,+12.5) } )
    name = f"h_response_alowshrcls_x_ful_inc"; histos.update( { name:ROOT.TH1D(name,det+";#frac{x_{trk}-x_{cls}}{#sigma(x_{cls})};Tracks",30,-12.5,+12.5) } )
    name = f"h_response_alowshrcls_y_ful_inc"; histos.update( { name:ROOT.TH1D(name,det+";#frac{y_{trk}-y_{cls}}{#sigma(y_{cls})};Tracks",30,-12.5,+12.5) } )

    name = f"h_residual_zeroshrcls_x_sml_inc"; histos.update( { name:ROOT.TH1D(name,det+";x_{trk}-x_{cls} [mm];Tracks",int(nResBins*0.6),-absRes*0.6,+absRes*0.6) } )
    name = f"h_residual_zeroshrcls_y_sml_inc"; histos.update( { name:ROOT.TH1D(name,det+";y_{trk}-y_{cls} [mm];Tracks",int(nResBins*0.6),-absRes*0.6,+absRes*0.6) } )
    name = f"h_residual_zeroshrcls_x_mid_inc"; histos.update( { name:ROOT.TH1D(name,det+";x_{trk}-x_{cls} [mm];Tracks",nResBins,-absRes*3,+absRes*3) } )
    name = f"h_residual_zeroshrcls_y_mid_inc"; histos.update( { name:ROOT.TH1D(name,det+";y_{trk}-y_{cls} [mm];Tracks",nResBins,-absRes*3,+absRes*3) } )
    name = f"h_residual_zeroshrcls_x_ful_inc"; histos.update( { name:ROOT.TH1D(name,det+";x_{trk}-x_{cls} [mm];Tracks",nResBins*2,-absRes*5,+absRes*5) } )
    name = f"h_residual_zeroshrcls_y_ful_inc"; histos.update( { name:ROOT.TH1D(name,det+";y_{trk}-y_{cls} [mm];Tracks",nResBins*2,-absRes*5,+absRes*5) } )

    name = f"h_response_zeroshrcls_x_sml_inc"; histos.update( { name:ROOT.TH1D(name,det+";#frac{x_{trk}-x_{cls}}{#sigma(x_{cls})};Tracks",30,-5,+5) } )
    name = f"h_response_zeroshrcls_y_sml_inc"; histos.update( { name:ROOT.TH1D(name,det+";#frac{y_{trk}-y_{cls}}{#sigma(y_{cls})};Tracks",30,-5,+5) } )
    name = f"h_response_zeroshrcls_x_ful_inc"; histos.update( { name:ROOT.TH1D(name,det+";#frac{x_{trk}-x_{cls}}{#sigma(x_{cls})};Tracks",30,-12.5,+12.5) } )
    name = f"h_response_zeroshrcls_y_ful_inc"; histos.update( { name:ROOT.TH1D(name,det+";#frac{y_{trk}-y_{cls}}{#sigma(y_{cls})};Tracks",30,-12.5,+12.5) } )
    
    for det in cfg["detectors"]:
        name = f"h_cls_occ_2D_{det}"; histos.update({name : ROOT.TH2D(name,f"{det};x [mm];y [mm];Track clusters",128,-cfg["chipY"]/2.,+cfg["chipY"]/2., 64,-cfg["chipX"]/2.,+cfg["chipX"]/2.) } )
        name = f"h_trk_occ_2D_{det}"; histos.update({name : ROOT.TH2D(name,f"{det};x [mm];y [mm];Tracks",        128,-cfg["chipY"]/2.,+cfg["chipY"]/2., 64,-cfg["chipX"]/2.,+cfg["chipX"]/2.) } )

        name = f"h_cls_occ_2D_{det}_after_cuts"; histos.update({name : ROOT.TH2D(name,f"{det};x [mm];y [mm];Track clusters",128,-cfg["chipY"]/2.,+cfg["chipY"]/2., 64,-cfg["chipX"]/2.,+cfg["chipX"]/2.) } )
        name = f"h_trk_occ_2D_{det}_after_cuts"; histos.update({name : ROOT.TH2D(name,f"{det};x [mm];y [mm];Tracks",        128,-cfg["chipY"]/2.,+cfg["chipY"]/2., 64,-cfg["chipX"]/2.,+cfg["chipX"]/2.) } )
        
        name = f"h_residual_alowshrcls_x_sml_{det}"; histos.update( { name:ROOT.TH1D(name,det+";x_{trk}-x_{cls} [mm];Tracks",int(nResBins*0.6),-absRes*0.6,+absRes*0.6) } )
        name = f"h_residual_alowshrcls_y_sml_{det}"; histos.update( { name:ROOT.TH1D(name,det+";y_{trk}-y_{cls} [mm];Tracks",int(nResBins*0.6),-absRes*0.6,+absRes*0.6) } )
        name = f"h_residual_alowshrcls_x_mid_{det}"; histos.update( { name:ROOT.TH1D(name,det+";x_{trk}-x_{cls} [mm];Tracks",nResBins,-absRes*3,+absRes*3) } )
        name = f"h_residual_alowshrcls_y_mid_{det}"; histos.update( { name:ROOT.TH1D(name,det+";y_{trk}-y_{cls} [mm];Tracks",nResBins,-absRes*3,+absRes*3) } )
        name = f"h_residual_alowshrcls_x_ful_{det}"; histos.update( { name:ROOT.TH1D(name,det+";x_{trk}-x_{cls} [mm];Tracks",nResBins*2,-absRes*5,+absRes*5) } )
        name = f"h_residual_alowshrcls_y_ful_{det}"; histos.update( { name:ROOT.TH1D(name,det+";y_{trk}-y_{cls} [mm];Tracks",nResBins*2,-absRes*5,+absRes*5) } )

        name = f"h_response_alowshrcls_x_sml_{det}"; histos.update( { name:ROOT.TH1D(name,det+";#frac{x_{trk}-x_{cls}}{#sigma(x_{cls})};Tracks",30,-12.5,+12.5) } )
        name = f"h_response_alowshrcls_y_sml_{det}"; histos.update( { name:ROOT.TH1D(name,det+";#frac{y_{trk}-y_{cls}}{#sigma(y_{cls})};Tracks",30,-12.5,+12.5) } )
        name = f"h_response_alowshrcls_x_ful_{det}"; histos.update( { name:ROOT.TH1D(name,det+";#frac{x_{trk}-x_{cls}}{#sigma(x_{cls})};Tracks",30,-12.5,+12.5) } )
        name = f"h_response_alowshrcls_y_ful_{det}"; histos.update( { name:ROOT.TH1D(name,det+";#frac{y_{trk}-y_{cls}}{#sigma(y_{cls})};Tracks",30,-12.5,+12.5) } )
        
        name = f"h_residual_zeroshrcls_x_sml_{det}"; histos.update( { name:ROOT.TH1D(name,det+";x_{trk}-x_{cls} [mm];Tracks",int(nResBins*0.6),-absRes*0.6,+absRes*0.6) } )
        name = f"h_residual_zeroshrcls_y_sml_{det}"; histos.update( { name:ROOT.TH1D(name,det+";y_{trk}-y_{cls} [mm];Tracks",int(nResBins*0.6),-absRes*0.6,+absRes*0.6) } )
        name = f"h_residual_zeroshrcls_x_mid_{det}"; histos.update( { name:ROOT.TH1D(name,det+";x_{trk}-x_{cls} [mm];Tracks",nResBins,-absRes*3,+absRes*3) } )
        name = f"h_residual_zeroshrcls_y_mid_{det}"; histos.update( { name:ROOT.TH1D(name,det+";y_{trk}-y_{cls} [mm];Tracks",nResBins,-absRes*3,+absRes*3) } )
        name = f"h_residual_zeroshrcls_x_ful_{det}"; histos.update( { name:ROOT.TH1D(name,det+";x_{trk}-x_{cls} [mm];Tracks",nResBins*2,-absRes*5,+absRes*5) } )
        name = f"h_residual_zeroshrcls_y_ful_{det}"; histos.update( { name:ROOT.TH1D(name,det+";y_{trk}-y_{cls} [mm];Tracks",nResBins*2,-absRes*5,+absRes*5) } )

        name = f"h_response_zeroshrcls_x_sml_{det}"; histos.update( { name:ROOT.TH1D(name,det+";#frac{x_{trk}-x_{cls}}{#sigma(x_{cls})};Tracks",30,-5,+5) } )
        name = f"h_response_zeroshrcls_y_sml_{det}"; histos.update( { name:ROOT.TH1D(name,det+";#frac{y_{trk}-y_{cls}}{#sigma(y_{cls})};Tracks",30,-5,+5) } )
        name = f"h_response_zeroshrcls_x_ful_{det}"; histos.update( { name:ROOT.TH1D(name,det+";#frac{x_{trk}-x_{cls}}{#sigma(x_{cls})};Tracks",30,-12.5,+12.5) } )
        name = f"h_response_zeroshrcls_y_ful_{det}"; histos.update( { name:ROOT.TH1D(name,det+";#frac{y_{trk}-y_{cls}}{#sigma(y_{cls})};Tracks",30,-12.5,+12.5) } )
        
        name = f"h_residual_zeroshrcls_xy_{det}";    histos.update( { name:ROOT.TH2D(name,det+";x_{trk}-x_{cls} [mm];y_{trk}-y_{cls} [mm];Tracks",nResBins,-absRes*3,+absRes*3, nResBins,-absRes*3,+absRes*3) } )
        name = f"h_residual_zeroshrcls_xy_mid_{det}";histos.update( { name:ROOT.TH2D(name,det+";x_{trk}-x_{cls} [mm];y_{trk}-y_{cls} [mm];Tracks",nResBins,-absRes*5,+absRes*5, nResBins,-absRes*5,+absRes*5) } )
    
        name = f"h_tunnel_width_x_{det}"; histos.update( { name:ROOT.TH1D(name,det+";Tunnel width [mm];Tracks",bintnl,limtnl[det][0],limtnl[det][1]) } )
        name = f"h_tunnel_width_y_{det}"; histos.update( { name:ROOT.TH1D(name,det+";Tunnel width [mm];Tracks",bintnl,limtnl[det][0],limtnl[det][1]) } )
    
    ####################################################
    for hname,hist in histos.items(): hist.Sumw2() #####
    ####################################################
    return histos


def book_shapes():
    cfg = config.Config().map
    
    dipole = ROOT.TPolyLine()
    xMinD = cfg["xDipoleExitMin"]
    xMaxD = cfg["xDipoleExitMax"]
    yMinD = cfg["yDipoleExitMin"]
    yMaxD = cfg["yDipoleExitMax"]    
    dipole.SetNextPoint(xMinD,yMinD)
    dipole.SetNextPoint(xMinD,yMaxD)
    dipole.SetNextPoint(xMaxD,yMaxD)
    dipole.SetNextPoint(xMaxD,yMinD)
    dipole.SetNextPoint(xMinD,yMinD)
    dipole.SetLineColor(ROOT.kBlue)
    dipole.SetLineWidth(1)

    flange = ROOT.TPolyLine()
    xMinF = cfg["xFlangeMin"]
    xMaxF = cfg["xFlangeMax"]
    yMinF = cfg["yFlangeMin"]
    yMaxF = cfg["yFlangeMax"]    
    flange.SetNextPoint(xMinF,yMinF)
    flange.SetNextPoint(xMinF,yMaxF)
    flange.SetNextPoint(xMaxF,yMaxF)
    flange.SetNextPoint(xMaxF,yMinF)
    flange.SetNextPoint(xMinF,yMinF)
    flange.SetLineColor(ROOT.kAzure+1)
    flange.SetLineWidth(1)
    
    window = ROOT.TPolyLine()
    xMinW = -cfg["xWindowWidth"]/2.
    xMaxW = +cfg["xWindowWidth"]/2.
    yMinW = cfg["yWindowMin"]
    yMaxW = cfg["yWindowMin"]+cfg["yWindowHeight"]
    window.SetNextPoint(xMinW,yMinW)
    window.SetNextPoint(xMinW,yMaxW)
    window.SetNextPoint(xMaxW,yMaxW)
    window.SetNextPoint(xMaxW,yMinW)
    window.SetNextPoint(xMinW,yMinW)    
    window.SetLineColor(ROOT.kBlue)
    window.SetLineWidth(1)
    
    return dipole, flange, window






##################################################
##################################################
##################################################


if __name__ == "__main__":
    # get the start time
    st = time.time()
    
    #############################################
    ### Initialize Config in the main process ###
    config.init_config(configfile, False)
    cfg = config.Config().map
    config.show_config(cfg)
    #############################################
    
    
    B  = cfg["fDipoleTesla"]
    LB = cfg["zDipoleLenghMeters"]
    
    
    ### get all the files
    tfilenamein = ""
    files = []
    if(ismutirun):
        tfilenamein,files = utils.make_multirun_dir(cfg["inputfile"],cfg["runnums"])
    else:
        tfilenamein = utils.make_run_dirs(cfg["inputfile"])
        files = utils.getfiles(tfilenamein)
    files = [fx for fx in files if '_BadTriggers' not in fx and os.path.getsize(fx) > 0]
    for f in files: print(f)
    
    runnum = utils.get_run_from_file(cfg["inputfile"])
    
    ispreproc = ("preprocessed" in cfg["inputfile"])
    
    
    ### read production config
    fpklcfgname = tfilenamein.replace("tree_","config_used/tree_").replace(".root","_config.pkl")
    fpklconfig = open(fpklcfgname,'rb')
    prod_cfg = pickle.load(fpklconfig)
    fpklconfig.close()
    ### was it aligned during production?
    isAlignedAtProd = False
    for det in prod_cfg["detectors"]:
        for axis,value in prod_cfg["misalignment"][det].items():
            if(value!=0):
                isAlignedAtProd = True
                break
        if(isAlignedAtProd): break
    ### should we apply misalignemnt here?
    isNon0Mislaignment = False
    for det in cfg["detectors"]:
        for axis,value in cfg["misalignment"][det].items():
            if(value!=0):
                isNon0Mislaignment = True
                break
        if(isNon0Mislaignment): break
    
    
    
    ### bad triggers
    fpkltrgname = tfilenamein.replace("tree_","beam_quality/tree_").replace(".root","_BadTriggers.pkl")
    badtriggers = []
    if(cfg["runtype"]=="beam" and cfg["checkbadtriggers"]):
        fpkltrigger = open(fpkltrgname,'rb')
        badtriggers = pickle.load(fpkltrigger)
        fpkltrigger.close()
        print("\n----------------------------------")
        print(f"Found {len(badtriggers)} bad triggers")
        print("-----------------------------------\n")
    else:
        print("\n----------------------------")
        print(f"Not removing any triggers!")
        print("----------------------------\n")
    nbadtrigs = len(badtriggers)
    
    
    
    ### counters
    counters.init_global_counters()
    Ndet = len(cfg["detectors"])
    
    ### histograms and shapes
    histos = book_histos()
    dipole, flange, window = book_shapes()
    print(f"Done booking histos")
    
    #################################
    ### prepare for eudaq writeup ###
    #################################
    nweudaqtrks = 0
    if(weudaqout):
        ### declare the data tree and its classes
        ROOT.gROOT.ProcessLine("struct pixel  { Int_t ix; Int_t iy; };" )
        ROOT.gROOT.ProcessLine("struct chip   { Int_t chip_id; std::vector<pixel> hits; };" )
        ROOT.gROOT.ProcessLine("struct stave  { Int_t stave_id; std::vector<chip> ch_ev_buffer; };" )
        ROOT.gROOT.ProcessLine("struct event  { Int_t trg_n; Double_t ts_begin; Double_t ts_end; std::vector<stave> st_ev_buffer; };" )
        ### declare the meta-data tree and its classes
        ROOT.gROOT.ProcessLine("struct run_meta_data  { Int_t run_number; Double_t run_start; Double_t run_end; };" )
        ### the main root gile
        fEUDAQout = ROOT.TFile.Open(f"tree_with_HT_selected_tracks_only_Run{runnum}.root", "RECREATE")
        ### data tree
        tEUDAQout = ROOT.TTree("MyTree","")
        eudaq_event = ROOT.event()
        tEUDAQout.Branch("event", eudaq_event)
        ### meta-data tree
        tEUDAQoutMeta = ROOT.TTree("MyTreeMeta","")
        run_meta_data = ROOT.run_meta_data()
        tEUDAQoutMeta.Branch("run_meta_data", run_meta_data)
        ### fill meta-data tree
        run_meta_data.run_number = runnum ### dummy
        run_meta_data.run_start  = -1.    ### dummy
        run_meta_data.run_end    = -1.    ### dummy
        tEUDAQoutMeta.Fill()
        print(f"Done booking tEUDAQ")
    
    
    ### save all events
    nevents = 0
    nalltrk = 0
    nacctrk = 0
    ngodtrk = 0
    nseltrk = 0
    nbtrtrk = 0
    nbadtrigs_actual = 0
    ntrigs_actual = 0
    tracks_triggers_dict = { "all": {"trgs":{"all":0,"good":0},"pix":{"all":0,"good":0},"cls":{"all":0,"good":0},"trks":0,"btrfly":0},
                             "even":{"trgs":{"all":0,"good":0},"pix":{"all":0,"good":0},"cls":{"all":0,"good":0},"trks":0,"btrfly":0},
                             "odd": {"trgs":{"all":0,"good":0},"pix":{"all":0,"good":0},"cls":{"all":0,"good":0},"trks":0,"btrfly":0} }
    
    arr_theta_xz = []
    arr_theta_yz = []
    arr_theta_yz_pass = []
    
    
    print(f"Files:\n{files}")
    stop = False
    for fpkl in files:
        if(stop): break
        suff = str(fpkl).split("_")[-1].replace(".pkl","")
        with open(fpkl,'rb') as handle:
            if(stop): break
            data = pickle.load(handle)
            for ievt,pkl_event in enumerate(data):
                
                
                ########################################
                ### nicely clear per event for eudaq ###
                ########################################
                if(weudaqout):
                    for s in range(eudaq_event.st_ev_buffer.size()):
                        eudaq_event.st_ev_buffer[s].ch_ev_buffer.clear()
                    eudaq_event.st_ev_buffer.clear()
                    eudaq_event.trg_n    = pkl_event.trigger
                    eudaq_event.ts_begin = -1.
                    eudaq_event.ts_end   = -1.
                    ### initialize stave buffer
                    for i in range(len(cfg["staves"])):
                        eudaq_event.st_ev_buffer.push_back( ROOT.stave() )
                ########################################
                
                
                ### check if the first part should be ignored:
                if(pkl_event.trigger<cfg["first2process"]): continue
                if(cfg["nmax2process"]>0 and nevents>cfg["nmax2process"]):
                    stop = True
                    break
                
                
                ### check parity
                iseven = (int(pkl_event.trigger)%2==0)
                
                
                ### calculate the average occupancies
                avgnpix = 0
                avgncls = 0
                if( len(pkl_event.npixels)==len(cfg["detectors"]) and len(pkl_event.nclusters)==len(cfg["detectors"])):
                    for det in cfg["detectors"]:
                        avgnpix += pkl_event.npixels[det]
                        avgncls += pkl_event.nclusters[det]
                    avgnpix /= len(cfg["detectors"])
                    avgncls /= len(cfg["detectors"])
                # else:
                #     continue ### empty event
                #     print("---------------------------------------------------------------------------------------")
                #     print(f"Problem with length of pixels array {len(pkl_event.npixels)} or clusters array {len(pkl_event.nclusters)}")
                #     print("---------------------------------------------------------------------------------------")
                print(f"Reading event #{ievt}, trigger:{pkl_event.trigger}, ts:[{utils.get_human_timestamp_ns(pkl_event.timestamp_bgn)}, {utils.get_human_timestamp_ns(pkl_event.timestamp_end)}]")
                
                
                ### some counters
                tracks_triggers_dict["all"]["trgs"]["all"] += 1
                tracks_triggers_dict["all"]["pix"]["all"]  += avgnpix
                tracks_triggers_dict["all"]["cls"]["all"]  += avgncls
                if(iseven):
                    tracks_triggers_dict["even"]["trgs"]["all"] += 1
                    tracks_triggers_dict["even"]["pix"]["all"]  += avgnpix
                    tracks_triggers_dict["even"]["cls"]["all"]  += avgncls
                else:
                    tracks_triggers_dict["odd"]["trgs"]["all"] += 1
                    tracks_triggers_dict["odd"]["pix"]["all"]  += avgnpix
                    tracks_triggers_dict["odd"]["cls"]["all"]  += avgncls
                ntrigs_actual += 1
                nevents += 1
                histos["hTriggers"].Fill(0.5)
                
                
                ### counters
                counters.counters_x_trg.append( pkl_event.trigger )
                counters.append_global_counters()
                icounter = len(counters.counters_x_trg)-1
                

                ### skip bad triggers...
                if((cfg["runtype"]=="beam" and cfg["checkbadtriggers"]) and (int(pkl_event.trigger) in badtriggers)):
                    nbadtrigs_actual += 1
                    print(f"Skipping bad trigger: {int(pkl_event.trigger)}")
                    continue
                histos["hTriggers"].Fill(1.5)
                
                
                tracks_triggers_dict["all"]["trgs"]["good"] += 1
                tracks_triggers_dict["all"]["pix"]["good"]  += avgnpix
                tracks_triggers_dict["all"]["cls"]["good"]  += avgncls
                if(iseven):
                    tracks_triggers_dict["even"]["trgs"]["good"] += 1
                    tracks_triggers_dict["even"]["pix"]["good"]  += avgnpix
                    tracks_triggers_dict["even"]["cls"]["good"]  += avgncls
                else:
                    tracks_triggers_dict["odd"]["trgs"]["good"] += 1
                    tracks_triggers_dict["odd"]["pix"]["good"]  += avgnpix
                    tracks_triggers_dict["odd"]["cls"]["good"]  += avgncls


                ### check errors
                if(not cfg["isMC"] and not ispreproc):
                    if(len(pkl_event.errors)!=len(cfg["detectors"])): continue
                    nErrors = 0
                    for det in cfg["detectors"]: nErrors += len(pkl_event.errors[det])
                    if(nErrors>0): continue
                
                
                ### check pixels
                n_pixels = 0
                nactivelayers = np.zeros(cfg["layers"], dtype=int)
                for det in cfg["detectors"]:
                    if(det not in pkl_event.npixels):
                        print(f"problem in trigger {pkl_event.trigger} --> {det} not found in pkl_event.npixels. Skipping.")
                        break
                    npix = pkl_event.npixels[det]
                    tdm = cfg["det2tdm"][det]
                    nactivelayers[tdm] = (npix>0)
                    n_pixels += npix
                pass_pixels = (sum(nactivelayers)==cfg["layers"])
                counters.set_global_counter("Pixels/layer",icounter,n_pixels/Ndet)
                if(not pass_pixels): continue
            

                ### check clusters
                n_clusters = 0
                nactivelayers = np.zeros(cfg["layers"], dtype=int)
                for det in cfg["detectors"]:
                    ncls = pkl_event.nclusters[det]
                    tdm = cfg["det2tdm"][det]
                    nactivelayers[tdm] = (ncls>0)
                    n_clusters += ncls
                pass_clusters = (sum(nactivelayers)==cfg["layers"])
                counters.set_global_counter("Clusters/layer",icounter,n_clusters/Ndet)
                if(not pass_clusters): continue
                
                
                histos["h_nTunnels"     ].Fill(pkl_event.ntunnels)
                histos["h_nTunnels_log" ].Fill(pkl_event.ntunnels)
                histos["h_nTunnels_full"].Fill(pkl_event.ntunnels)
                histos["h_nTunnels_mid" ].Fill(pkl_event.ntunnels)
                histos["h_nTunnels_zoom"].Fill(pkl_event.ntunnels)
                

                ### check seeds
                n_seeds = len(pkl_event.seeds)
                counters.set_global_counter("Track Seeds",icounter,n_seeds)
                histos["h_nSeeds"     ].Fill(n_seeds)
                histos["h_nSeeds_log" ].Fill(n_seeds)
                histos["h_nSeeds_full"].Fill(n_seeds)
                histos["h_nSeeds_mid" ].Fill(n_seeds)
                histos["h_nSeeds_zoom"].Fill(n_seeds)
                if(n_seeds==0): continue
        

                ### check tracks
                n_tracks = len(pkl_event.tracks)
                if(n_tracks==0): continue
                

                good_tracks = []
                acceptance_tracks = []
                for track in pkl_event.tracks:
                    # #############
                    # ### test ####
                    # print(f"eventid={ievt}: trigger={pkl_event.trigger}")
                    # print(f"   track:")
                    # for det in track.detectors:
                    #     print(f"      {det}: clsid={track.trkcls[det].CID}: x={track.trkcls[det].xTmm:.3f}, y={track.trkcls[det].yTmm:.3f}, z={track.trkcls[det].zTmm:.3f}")
                    # #############
                    
                    
                    #####################################
                    ### first require successful fit ####
                    #####################################
                    if(not track.success): continue

                    
                    ##################################
                    ### first require max cluster ####
                    ##################################
                    if(track.maxcls>cfg["cut_maxcls"]): continue
                    
                    
                    # #####################
                    # ### pixel ROI cut ###
                    # #####################
                    # inROI = True
                    # for det in cfg["detectors"]:
                    #     for pix in track.trkcls[det].pixels:
                    #         # print(f"{det}: pix={pix.x,pix.y}")
                    #         if(pix.x<cfg["cut_ROI_xmin"] or pix.x>cfg["cut_ROI_xmax"]): inROI = False
                    #         if(pix.y<cfg["cut_ROI_ymin"] or pix.y>cfg["cut_ROI_ymax"]): inROI = False
                    #         if(not inROI): break
                    #     if(not inROI): break
                    # if(not inROI): continue
                    
                    
                    histos["h_MLE_theta1_logx_before_cuts"].Fill(math.sqrt(track.theta2))
                    histos["h_MLE_theta2_logx_before_cuts"].Fill(track.theta2)
                    histos["h_MLE_theta1_linx_before_cuts"].Fill(math.sqrt(track.theta2))
                    histos["h_MLE_theta2_linx_before_cuts"].Fill(track.theta2)
                    
                    
                    ### fill some quantities
                    if(track.chi2ndof<=cfg["cut_chi2dof"] and track.chi2ndof>=cfg["minchi2align"] and selections.pass_geoacc_selection(track,ismultiproc=False)): ##TODO: missing the shared hits cut here...
                        histos["hChi2DoF_alowshrcls"].Fill(track.chi2ndof)
                        histos["hChi2DoF_full_alowshrcls"].Fill(track.chi2ndof)
                        histos["hChi2DoF_mid_alowshrcls"].Fill(track.chi2ndof)
                        histos["hChi2DoF_zoom_alowshrcls"].Fill(track.chi2ndof)
                        histos["hChi2DoF_small_alowshrcls"].Fill(track.chi2ndof)
                        
                        histos["hChi2_alowshrcls"].Fill(track.chisq)
                        histos["hChi2_full_alowshrcls"].Fill(track.chisq)
                        histos["hChi2_mid_alowshrcls"].Fill(track.chisq)
                        histos["hChi2_zoom_alowshrcls"].Fill(track.chisq)
                        histos["hChi2_small_alowshrcls"].Fill(track.chisq)
                        
                        histos["h_cls_absdx"].Fill(abs(track.trkcls["ALPIDE_1"].xTnoGmm-track.trkcls["ALPIDE_0"].xTnoGmm))
                        histos["h_cls_absdx"].Fill(abs(track.trkcls["ALPIDE_2"].xTnoGmm-track.trkcls["ALPIDE_1"].xTnoGmm))
                        histos["h_cls_absdx"].Fill(abs(track.trkcls["ALPIDE_3"].xTnoGmm-track.trkcls["ALPIDE_2"].xTnoGmm))
                        histos["h_cls_absdx"].Fill(abs(track.trkcls["ALPIDE_4"].xTnoGmm-track.trkcls["ALPIDE_3"].xTnoGmm))
                        ##
                        histos["h_cls_absdy"].Fill(abs(track.trkcls["ALPIDE_1"].yTnoGmm-track.trkcls["ALPIDE_0"].yTnoGmm))
                        histos["h_cls_absdy"].Fill(abs(track.trkcls["ALPIDE_2"].yTnoGmm-track.trkcls["ALPIDE_1"].yTnoGmm))
                        histos["h_cls_absdy"].Fill(abs(track.trkcls["ALPIDE_3"].yTnoGmm-track.trkcls["ALPIDE_2"].yTnoGmm))
                        histos["h_cls_absdy"].Fill(abs(track.trkcls["ALPIDE_4"].yTnoGmm-track.trkcls["ALPIDE_3"].yTnoGmm))
                        
                        for det in cfg["detectors"]:
                            if(det not in track.detectors): continue
                            
                            histos[f"h_cls_occ_2D_{det}"].Fill(track.trkcls[det].xTnoGmm,track.trkcls[det].yTnoGmm)
                            xTnoG,yTnoG,zTnoG = utils.get_track_at_det_noG(det,track)
                            histos[f"h_trk_occ_2D_{det}"].Fill(xTnoG,yTnoG)
                            
                            dx,dy = utils.res_track2cluster(det,track.detectors,track.points,track.direction,track.centroid)
                            histos[f"h_residual_alowshrcls_x_sml_inc"].Fill(dx)
                            histos[f"h_residual_alowshrcls_x_mid_inc"].Fill(dx)
                            histos[f"h_residual_alowshrcls_x_ful_inc"].Fill(dx)
                            histos[f"h_residual_alowshrcls_y_sml_inc"].Fill(dy)
                            histos[f"h_residual_alowshrcls_y_mid_inc"].Fill(dy)
                            histos[f"h_residual_alowshrcls_y_ful_inc"].Fill(dy)
                            histos[f"h_response_alowshrcls_x_sml_inc"].Fill(dx/track.trkcls[det].dxTmm)
                            histos[f"h_response_alowshrcls_x_ful_inc"].Fill(dx/track.trkcls[det].dxTmm)
                            histos[f"h_response_alowshrcls_y_sml_inc"].Fill(dy/track.trkcls[det].dyTmm)
                            histos[f"h_response_alowshrcls_y_ful_inc"].Fill(dy/track.trkcls[det].dyTmm)

                            histos[f"h_residual_alowshrcls_x_sml_{det}"].Fill(dx)
                            histos[f"h_residual_alowshrcls_x_mid_{det}"].Fill(dx)
                            histos[f"h_residual_alowshrcls_x_ful_{det}"].Fill(dx)
                            histos[f"h_residual_alowshrcls_y_sml_{det}"].Fill(dy)
                            histos[f"h_residual_alowshrcls_y_mid_{det}"].Fill(dy)
                            histos[f"h_residual_alowshrcls_y_ful_{det}"].Fill(dy)
                            histos[f"h_response_alowshrcls_x_sml_{det}"].Fill(dx/track.trkcls[det].dxTmm)
                            histos[f"h_response_alowshrcls_x_ful_{det}"].Fill(dx/track.trkcls[det].dxTmm)
                            histos[f"h_response_alowshrcls_y_sml_{det}"].Fill(dy/track.trkcls[det].dyTmm)
                            histos[f"h_response_alowshrcls_y_ful_{det}"].Fill(dy/track.trkcls[det].dyTmm)
                            
                            
                    
                    # #################################################
                    # ### refit the track if necessary
                    # if(not isAlignedAtProd and isNon0Mislaignment):
                    #     print(f"Note I'm doing a refit!!")
                    #     track = refit(track)
                    # ### will be the same if misalignment is 0
                    # #################################################
                    
                    # dxarr = [abs(track.trkcls["ALPIDE_1"].xTnoGmm-track.trkcls["ALPIDE_0"].xTnoGmm),
                    #          abs(track.trkcls["ALPIDE_2"].xTnoGmm-track.trkcls["ALPIDE_1"].xTnoGmm),
                    #          abs(track.trkcls["ALPIDE_3"].xTnoGmm-track.trkcls["ALPIDE_2"].xTnoGmm),
                    #          abs(track.trkcls["ALPIDE_4"].xTnoGmm-track.trkcls["ALPIDE_3"].xTnoGmm)
                    #         ]
                    # if(max(dxarr)>0.15): continue
                    
                    
                    #########################
                    ### then require chi2 ###
                    #########################
                    if(track.chi2ndof>cfg["cut_chi2dof"]): continue ### this is the new chi2!
                    if(track.chi2ndof<cfg["minchi2align"]): continue
                    
                    # if(weudaqout):
                    #     if(track.chi2ndof<cfg["minchi2align"]): continue
                    #     if(track.chi2ndof>cfg["maxchi2align"]): continue
                    good_tracks.append(track)
                    ngodtrk += 1
                    
                    ### get the coordinates at extreme points in real space and after tilting the detector
                    r0,rN,rW,rF,rD = utils.get_track_point_at_extremes(track,ismultiproc=False)

                    ### the y distance from y=0 in the dipole exit plane
                    dExit = rD[1]
                    
                    ### calculate the fit angles
                    ### params = [p0x,p1x,p0y,p1y]
                    newdir,newcnt = utils.apply_global_alignment_to_p(track.direction,track.centroid,ismultiproc=False)
                    newpars = utils.get_pars_from_centroid_and_direction(newcnt,newdir)
                    # tan_theta_xz = track.params[1] ### the slope in xz
                    # tan_theta_yz = track.params[3] ### the slope in yz
                    tan_theta_xz = newpars[1] ### the slope in xz
                    tan_theta_yz = newpars[3] ### the slope in yz
                    thetaf_xz = math.atan(tan_theta_xz)
                    thetaf_yz = math.atan(tan_theta_yz)
                    
                    thetaf_yz_labframe = math.atan( (rN[1]-r0[1])/(rN[2]-r0[2]) )
                    thetaf_xz_labframe = math.atan( (rN[0]-r0[0])/(rN[2]-r0[2]) )
                    

                    ### fill histos before cuts
                    histos["hF_before_cuts"].Fill(rF[0],rF[1])
                    histos["hD_before_cuts"].Fill(rD[0],rD[1])
                    histos["hD_zoomin_before_cuts"].Fill(rD[0],rD[1])
                    histos["hD_zoomout_before_cuts"].Fill(rD[0],rD[1])
                    histos["hW_before_cuts"].Fill(rW[0],rW[1])
                    histos["hTheta_xz_before_cuts"].Fill(thetaf_xz)
                    histos["hTheta_yz_before_cuts"].Fill(thetaf_yz)
                    
                    arr_theta_xz.append(thetaf_xz)
                    arr_theta_yz.append(thetaf_yz)
                        
                    histos["hTheta_xz_labframe_before_cuts"].Fill(thetaf_xz_labframe)
                    histos["hTheta_yz_labframe_before_cuts"].Fill(thetaf_yz_labframe)
                                        
                    nalltrk += 1
                    
                    ########################################
                    ### require pointing cuts and others ###
                    ### as defined in the [CUTS] section ###
                    ### of the config file #################
                    ########################################
                    if(not selections.pass_geoacc_selection(track,ismultiproc=False)): continue
                    
                    ### the angle in y-z calculated from d_exit
                    thetad_yz = 2.*math.atan(dExit*mm2m/LB)
                    ### the angle in y-z calculated from the tilted detector extremes
                    thetar_yz = math.atan( (rN[1]-r0[1])/(rN[2]-r0[2]) )
                    
                    ### the momentum magnitudes
                    pf = get_pz_from_fit(thetaf_yz)
                    pd = (0.3 * B * LB)/math.sin( thetad_yz )
                    pr = (0.3 * B * LB)/math.sin( thetar_yz )
                    
                    ### theta_yz passing:
                    arr_theta_yz_pass.append(thetaf_yz)
                    
                    histos["hThetad_vs_thetaf"].Fill(thetaf_yz,thetad_yz)
                    histos["hThetar_vs_thetaf"].Fill(thetaf_yz,thetar_yz)
                    
                    histos["hPf_vs_dExit"].Fill(dExit,pf)
                    histos["hPd_vs_dExit"].Fill(dExit,pd)
                    histos["hPr_vs_dExit"].Fill(dExit,pr)
                    
                    histos["hPf_vs_thetaf"].Fill(thetaf_yz,pf)
                    histos["hPd_vs_thetad"].Fill(thetad_yz,pd)
                    histos["hPr_vs_thetar"].Fill(thetar_yz,pr)
                    
                    histos["hDexit_vs_thetaf"].Fill(thetaf_yz,dExit)
                    histos["hDexit_vs_thetad"].Fill(thetad_yz,dExit)
                    histos["hDexit_vs_thetar"].Fill(thetar_yz,dExit)
                    
                    histos["hF_after_cuts"].Fill(rF[0],rF[1])
                    histos["hD_after_cuts"].Fill(rD[0],rD[1])
                    histos["hD_zoomin_after_cuts"].Fill(rD[0],rD[1])
                    histos["hD_zoomout_after_cuts"].Fill(rD[0],rD[1])
                    histos["hW_after_cuts"].Fill(rW[0],rW[1])
                    
                    histos["hThetaf_yz"].Fill(thetaf_yz)
                    histos["hThetad_yz"].Fill(thetad_yz)
                    histos["hThetar_yz"].Fill(thetar_yz)
                    
                    histos["hTheta_xz_after_cuts"].Fill(thetaf_xz)
                    histos["hTheta_yz_after_cuts"].Fill(thetaf_yz)
                    
                    histos["hTheta_xz_labframe_after_cuts"].Fill(thetaf_xz_labframe)
                    histos["hTheta_yz_labframe_after_cuts"].Fill(thetaf_yz_labframe)
                    
                    histos["hdExit"].Fill(dExit)
                    
                    histos["h_MLE_theta1_logx_after_cuts"].Fill(math.sqrt(track.theta2))
                    histos["h_MLE_theta2_logx_after_cuts"].Fill(track.theta2)
                    histos["h_MLE_theta1_linx_after_cuts"].Fill(math.sqrt(track.theta2))
                    histos["h_MLE_theta2_linx_after_cuts"].Fill(track.theta2)
                    
                    if(pf>0):
                        histos["hPf"].Fill(pf)
                        histos["hPf_small"].Fill(pf)
                        histos["hPf_zoom"].Fill(pf)
                    if(pd>0):
                        histos["hPd"].Fill(pd)
                        histos["hPd_small"].Fill(pd)
                        histos["hPd_zoom"].Fill(pd)
                    if(pr>0):
                        histos["hPr"].Fill(pr)
                        histos["hPr_small"].Fill(pr)
                        histos["hPr_zoom"].Fill(pr)
                    
                    acceptance_tracks.append(track)
                    nacctrk += 1
                    
                    # TODO: FOR PRINTING THE CLUSTERS FOR THE HOUGH-TRANSFORM ILLUSTRATIVE PLOT
                    # for det in track.detectors:
                    #     print(f'"{det}": ({track.trkcls[det].xTnoGmm},{track.trkcls[det].yTnoGmm},{track.trkcls[det].zTnoGmm}),')
                    
                ###################
                ### end of loop ###
                ###################
                    
                
                ### the graph of the good tracks
                counters.set_global_counter("Good Tracks",icounter,len(good_tracks))
                
                ### check for overlaps
                selected_tracks = acceptance_tracks if(cfg["cut_allow_shared_clusters"]) else selections.remove_tracks_with_shared_clusters(acceptance_tracks)
                # if(len(selected_tracks)!=len(acceptance_tracks)): print(f"nacc:{len(acceptance_tracks)} --> nsel={len(selected_tracks)}")
                nseltrk += len(selected_tracks)
                
                histos["h_nTracks"     ].Fill(len(selected_tracks))
                histos["h_nTracks_log" ].Fill(len(selected_tracks))
                histos["h_nTracks_full"].Fill(len(selected_tracks))
                histos["h_nTracks_mid" ].Fill(len(selected_tracks))
                histos["h_nTracks_zoom"].Fill(len(selected_tracks))
                
                
                ### split here to tracks in/out the butterfly cut
                butterfly_tracks_mask = []
                n_butterfly_tracks = 0
                for track in selected_tracks:
                    pass_butterfly = True
                    if(cfg["cut_RoI_btrfly"]): pass_butterfly = selections.tilted_butterfly_RoI_cut(track)
                    butterfly_tracks_mask.append( pass_butterfly )
                    n_butterfly_tracks += pass_butterfly
                    ### fill the track occupancies after the cuts
                    if(pass_butterfly):
                        for det in cfg["detectors"]:
                            if(det not in track.detectors): continue
                            histos[f"h_cls_occ_2D_{det}_after_cuts"].Fill(track.trkcls[det].xTnoGmm,track.trkcls[det].yTnoGmm)
                            xTnoG,yTnoG,zTnoG = utils.get_track_at_det_noG(det,track)
                            histos[f"h_trk_occ_2D_{det}_after_cuts"].Fill(xTnoG,yTnoG)
                
                ##############################################################################
                ### fill the last counter ####################################################
                counters.set_global_counter("Selected Tracks",icounter,n_butterfly_tracks) ###
                ##############################################################################
                histos["h_nTracks_btrfly"     ].Fill(n_butterfly_tracks)
                histos["h_nTracks_btrfly_log" ].Fill(n_butterfly_tracks)
                histos["h_nTracks_btrfly_full"].Fill(n_butterfly_tracks)
                histos["h_nTracks_btrfly_mid" ].Fill(n_butterfly_tracks)
                histos["h_nTracks_btrfly_zoom"].Fill(n_butterfly_tracks)
                nbtrtrk += n_butterfly_tracks
                
                
                ### event displays
                if(cfg["plot_offline_evtdisp"] and len(good_tracks)>0):
                    fevtdisplayname = tfilenamein.replace("tree_","event_displays/").replace(".root",f"_offline_{pkl_event.trigger}.pdf")
                    evtdisp.plot_event(pkl_event.meta.run,pkl_event.meta.start,pkl_event.meta.dur,pkl_event.trigger,fevtdisplayname,pkl_event.clusters,pkl_event.tracks,chi2threshold=cfg["cut_chi2dof"],ismultiproc=False)
                
                
                ### the Hough space (for the tunnel widths)
                hzx = ROOT.TH2D("hzx","",pkl_event.hough_space["zx_xbins"],pkl_event.hough_space["zx_xmin"],pkl_event.hough_space["zx_xmax"],  pkl_event.hough_space["zx_ybins"],pkl_event.hough_space["zx_ymin"],pkl_event.hough_space["zx_ymax"])
                hzy = ROOT.TH2D("hzy","",pkl_event.hough_space["zy_xbins"],pkl_event.hough_space["zy_xmin"],pkl_event.hough_space["zy_xmax"],  pkl_event.hough_space["zy_ybins"],pkl_event.hough_space["zy_ymin"],pkl_event.hough_space["zy_ymax"])
                
                ### count selected tracks
                tracks_triggers_dict["all"]["trks"]              += len(selected_tracks)
                if(iseven): tracks_triggers_dict["even"]["trks"] += len(selected_tracks)
                else:       tracks_triggers_dict["odd"]["trks"]  += len(selected_tracks)
                
                ### count butterfly tracks
                tracks_triggers_dict["all"]["btrfly"]              += n_butterfly_tracks
                if(iseven): tracks_triggers_dict["even"]["btrfly"] += n_butterfly_tracks
                else:       tracks_triggers_dict["odd"]["btrfly"]  += n_butterfly_tracks
                
                
                ### plot some selected tracks
                for itrk,track in enumerate(selected_tracks):
                    ##################################################
                    ### keep only the tracks passing the butterfly ###
                    if(not butterfly_tracks_mask[itrk]): continue ####
                    ##################################################
                    
                    ###########################
                    ### fill the eudaq tree ###
                    ###########################
                    if(weudaqout):
                        nweudaqtrks += 1
                        # print(f"Trigger[{pkl_event.trigger}]: trk[{itrk}] chi2={track.chi2ndof}")
                        for det in track.detectors:
                            stvid = cfg["det2stvchp"][det][0]
                            chpid = cfg["det2stvchp"][det][1]
                            eudaq_event.st_ev_buffer[stvid].ch_ev_buffer.push_back( ROOT.chip() )
                            ichip = eudaq_event.st_ev_buffer[stvid].ch_ev_buffer.size()-1
                            eudaq_event.st_ev_buffer[stvid].ch_ev_buffer[ichip].chip_id = int(chpid)
                            trkpixels = []
                            for pixel in track.trkcls[det].pixels:
                                eudaq_event.st_ev_buffer[stvid].ch_ev_buffer[ichip].hits.push_back( ROOT.pixel() )
                                ihit = eudaq_event.st_ev_buffer[stvid].ch_ev_buffer[ichip].hits.size()-1
                                eudaq_event.st_ev_buffer[stvid].ch_ev_buffer[ichip].hits[ihit].ix = pixel.x
                                eudaq_event.st_ev_buffer[stvid].ch_ev_buffer[ichip].hits[ihit].iy = pixel.y
                                trkpixels.append([pixel.x,pixel.y])
                            # print(f"itrk[{itrk}]: chpid={chpid} --> trkpixels={trkpixels}")
                    ###########################
                    

                    histos["hChi2DoF_zeroshrcls"].Fill(track.chi2ndof)
                    histos["hChi2DoF_full_zeroshrcls"].Fill(track.chi2ndof)
                    histos["hChi2DoF_mid_zeroshrcls"].Fill(track.chi2ndof)
                    histos["hChi2DoF_zoom_zeroshrcls"].Fill(track.chi2ndof)
                    histos["hChi2DoF_small_zeroshrcls"].Fill(track.chi2ndof)
                    
                    histos["hChi2_zeroshrcls"].Fill(track.chisq)
                    histos["hChi2_full_zeroshrcls"].Fill(track.chisq)
                    histos["hChi2_mid_zeroshrcls"].Fill(track.chisq)
                    histos["hChi2_zoom_zeroshrcls"].Fill(track.chisq)
                    histos["hChi2_small_zeroshrcls"].Fill(track.chisq)
                    # for det in cfg["detectors"]:
                    for det in track.detectors:
                        dx,dy = utils.res_track2cluster(det,track.detectors,track.points,track.direction,track.centroid)
                        
                        histos[f"h_residual_zeroshrcls_x_sml_inc"].Fill(dx)
                        histos[f"h_residual_zeroshrcls_x_mid_inc"].Fill(dx)
                        histos[f"h_residual_zeroshrcls_x_ful_inc"].Fill(dx)
                        histos[f"h_residual_zeroshrcls_y_sml_inc"].Fill(dy)
                        histos[f"h_residual_zeroshrcls_y_mid_inc"].Fill(dy)
                        histos[f"h_residual_zeroshrcls_y_ful_inc"].Fill(dy)
                        histos[f"h_response_zeroshrcls_x_sml_inc"].Fill(dx/track.trkcls[det].dxTmm)
                        histos[f"h_response_zeroshrcls_x_ful_inc"].Fill(dx/track.trkcls[det].dxTmm)
                        histos[f"h_response_zeroshrcls_y_sml_inc"].Fill(dy/track.trkcls[det].dyTmm)
                        histos[f"h_response_zeroshrcls_y_ful_inc"].Fill(dy/track.trkcls[det].dyTmm)
                        
                        histos[f"h_residual_zeroshrcls_x_sml_{det}"].Fill(dx)
                        histos[f"h_residual_zeroshrcls_x_mid_{det}"].Fill(dx)
                        histos[f"h_residual_zeroshrcls_x_ful_{det}"].Fill(dx)
                        histos[f"h_residual_zeroshrcls_y_sml_{det}"].Fill(dy)
                        histos[f"h_residual_zeroshrcls_y_mid_{det}"].Fill(dy)
                        histos[f"h_residual_zeroshrcls_y_ful_{det}"].Fill(dy)
                        histos[f"h_response_zeroshrcls_x_sml_{det}"].Fill(dx/track.trkcls[det].dxTmm)
                        histos[f"h_response_zeroshrcls_x_ful_{det}"].Fill(dx/track.trkcls[det].dxTmm)
                        histos[f"h_response_zeroshrcls_y_sml_{det}"].Fill(dy/track.trkcls[det].dyTmm)
                        histos[f"h_response_zeroshrcls_y_ful_{det}"].Fill(dy/track.trkcls[det].dyTmm)
                        
                        histos[f"h_residual_zeroshrcls_xy_{det}"].Fill(dx,dy)
                        histos[f"h_residual_zeroshrcls_xy_mid_{det}"].Fill(dx,dy)
                        
                        ### draw all waves
                        rChip = [track.trkcls[det].xTnoGmm,track.trkcls[det].yTnoGmm,track.trkcls[det].zTnoGmm]
                        
                        if(iswavehst):
                            # xwave = get_wave("xz",rChip[2],rChip[0],0,np.pi)
                            # ywave = get_wave("yz",rChip[2],rChip[1],0,np.pi)
                            xwave = get_wave("xz",rChip[2],rChip[0],0,np.pi)
                            ywave = get_wave("yz",rChip[2],rChip[1],0,np.pi)
                            for btheta in range(1,histos["hWaves_zx"].GetNbinsX()+1):
                                theta = histos["hWaves_zx"].GetXaxis().GetBinCenter(btheta)
                                rhox  = xwave.Eval(theta)
                                rhoy  = ywave.Eval(theta)
                                histos["hWaves_zx"].Fill(theta,rhox)
                                histos["hWaves_zy"].Fill(theta,rhoy)
                            del xwave
                            del ywave
                    
                    if(iswavehst):
                        ### draw only wave intersections
                        pivot_stave = 0
                        for det in track.detectors: pivot_stave += cfg["det2stv"][det]
                        pivot_stave /= len(track.detectors)
                        pivot_stave = 0 if(pivot_stave<0.5) else cfg["layers"]
                        fill_pair( pivot_stave+0, pivot_stave+1, track, histos["hWaves_zx_intersections"], histos["hWaves_zy_intersections"] )
                        fill_pair( pivot_stave+0, pivot_stave+2, track, histos["hWaves_zx_intersections"], histos["hWaves_zy_intersections"] )
                        fill_pair( pivot_stave+0, pivot_stave+3, track, histos["hWaves_zx_intersections"], histos["hWaves_zy_intersections"] )
                        fill_pair( pivot_stave+0, pivot_stave+4, track, histos["hWaves_zx_intersections"], histos["hWaves_zy_intersections"] )
                        fill_pair( pivot_stave+1, pivot_stave+2, track, histos["hWaves_zx_intersections"], histos["hWaves_zy_intersections"] )
                        fill_pair( pivot_stave+1, pivot_stave+3, track, histos["hWaves_zx_intersections"], histos["hWaves_zy_intersections"] )
                        fill_pair( pivot_stave+1, pivot_stave+4, track, histos["hWaves_zx_intersections"], histos["hWaves_zy_intersections"] )
                        fill_pair( pivot_stave+2, pivot_stave+3, track, histos["hWaves_zx_intersections"], histos["hWaves_zy_intersections"] )
                        fill_pair( pivot_stave+2, pivot_stave+4, track, histos["hWaves_zx_intersections"], histos["hWaves_zy_intersections"] )
                        fill_pair( pivot_stave+3, pivot_stave+4, track, histos["hWaves_zx_intersections"], histos["hWaves_zy_intersections"] )
                        ### find the tunnel widths
                        thetax = track.hough_coords[0]
                        rhox   = track.hough_coords[1]
                        thetay = track.hough_coords[2]
                        rhoy   = track.hough_coords[3]
                        bthetax = hzx.GetXaxis().FindBin( thetax )
                        brhox   = hzx.GetXaxis().FindBin( rhox   )
                        bthetay = hzy.GetXaxis().FindBin( thetay )
                        brhoy   = hzy.GetXaxis().FindBin( rhoy   )
                        arr_thetax = [ hzx.GetXaxis().GetBinLowEdge(bthetax), hzx.GetXaxis().GetBinUpEdge(bthetax) ]
                        arr_rhox   = [ hzx.GetYaxis().GetBinLowEdge(brhox),   hzx.GetYaxis().GetBinUpEdge(brhox)   ]
                        arr_thetay = [ hzy.GetXaxis().GetBinLowEdge(bthetay), hzy.GetXaxis().GetBinUpEdge(bthetay) ]
                        arr_rhoy   = [ hzy.GetYaxis().GetBinLowEdge(brhoy),   hzy.GetYaxis().GetBinUpEdge(brhoy)   ]
                        for det in cfg["detectors"]:
                            xmin,xmax,ymin,ymax = get_edges_from_theta_rho_corners(det,arr_thetax,arr_rhox,arr_thetay,arr_rhoy)
                            histos[f"h_tunnel_width_x_{det}"].Fill(xmax-xmin)
                            histos[f"h_tunnel_width_y_{det}"].Fill(ymax-ymin)
                
                
                ### at the end of the pkl_event, clean the Hough space histos
                del hzx
                del hzy
                
                ###########################
                ### fill the eudaq tree ###
                if(weudaqout): tEUDAQout.Fill()
                ###########################
                
                print(f"Event[{nevents-1}], Trigger[{pkl_event.trigger}] --> Good tracks: {len(good_tracks)}, Acceptance tracks: {len(acceptance_tracks)}, Selected tracks: {len(selected_tracks)}, Butterfly tracks: {n_butterfly_tracks}")

    # print(f"Events:{nevents}, Tracks:{nacctrk}")
    print(f"All tracks: {nalltrk}, Good tracks:{ngodtrk}, Accepted tracks:{nacctrk}, Selected tracks:{nseltrk}, Butterfly tracks:{nbtrtrk}  with GoodTriggers:{nevents-nbadtrigs_actual}, Actual triggers: {ntrigs_actual} (with AllTriggers:{nevents} and BadTriggers in the range: {nbadtrigs_actual} (or {nbadtrigs} in the full run))")
    
    if(weudaqout): print(f"\nnweudaqtrks={nweudaqtrks}\n")
    
    ### plot the counters
    counters.counters_x_trg[0] = counters.counters_x_trg[0] if(counters.counters_x_trg[0]!=0 and cfg["first2process"]!=0) else 0
    fmultpdfname = tfilenamein.replace(".root",f"_multiplicities_vs_triggers.pdf")
    counters.plot_counters(fmultpdfname,runnum)


    ### plot the geometry distributions
    foupdfname = tfilenamein.replace(".root",f"_allplots.pdf")

    cnv = ROOT.TCanvas("cnv_dipole_window","",1000,500)
    cnv.Divide(2,1)
    cnv.cd(1)
    ROOT.gPad.SetTicks(1,1)
    histos["hD_zoomout_before_cuts"].Draw("colz")
    dipole.Draw()
    ROOT.gPad.RedrawAxis()
    cnv.cd(2)
    ROOT.gPad.SetTicks(1,1)
    histos["hD_zoomout_after_cuts"].Draw("colz")
    dipole.Draw()
    ROOT.gPad.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{foupdfname}(")
    del cnv

    cnv = ROOT.TCanvas("cnv_dipole_window","",1000,500)
    cnv.Divide(2,1)
    cnv.cd(1)
    ROOT.gPad.SetTicks(1,1)
    ROOT.gPad.SetGridx()
    ROOT.gPad.SetGridy()
    histos["hD_before_cuts"].Draw("colz")
    dipole.Draw()
    ROOT.gPad.RedrawAxis()
    cnv.cd(2)
    ROOT.gPad.SetTicks(1,1)
    ROOT.gPad.SetGridx()
    ROOT.gPad.SetGridy()
    histos["hD_after_cuts"].Draw("colz")
    dipole.Draw()
    ROOT.gPad.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{foupdfname}")
    del cnv
    
    
    glb_dx = cfg["global_corr_dx"]
    glb_tx = cfg["global_corr_thetax"]
    # glb_ty = cfg["global_corr_thetay"]
    glb_dy = cfg["global_corr_dy"]
    cut_strp = cfg["cut_strip"]
    cut_spot = cfg["cut_spot"]
    # noGlobalAlignment      = (glb_tx==0 and glb_ty==0 and glb_dx==0 and glb_dy==0 and cut_strp==False and cut_spot==False)
    # noGlobAlgnWithStrip    = (glb_tx==0 and glb_ty==0 and glb_dx==0 and glb_dy==0 and cut_strp==True  and cut_spot==False) ## strip cut has to be around the orig blob
    # partialGlobalAlignment = (glb_tx!=0 and glb_ty!=0 and glb_dx==0 and glb_dy==0 and cut_strp==False and cut_spot==False)
    # partGlobalAlgnWithStrp = (glb_tx!=0 and glb_ty!=0 and glb_dx==0 and glb_dy==0 and cut_strp==True  and cut_spot==False) ## strip cut has to be around the new blob
    # fullGlobalAlignment    = (glb_tx!=0 and glb_ty!=0 and glb_dx!=0 and glb_dy!=0 and cut_strp==False and cut_spot==False)
    # fullGlobAlgnWithStrip  = (glb_tx!=0 and glb_ty!=0 and glb_dx!=0 and glb_dy!=0 and cut_strp==True  and cut_spot==False) ## strip cut has to be around the new blob
    # fullGlobAlgFullSel     = (glb_tx!=0 and glb_ty!=0 and glb_dx!=0 and glb_dy!=0 and cut_strp==False and cut_spot==True)

    noGlobalAlignment      = (glb_tx==0 and glb_dx==0 and glb_dy==0 and cut_strp==False and cut_spot==False)
    noGlobAlgnWithStrip    = (glb_tx==0 and glb_dx==0 and glb_dy==0 and cut_strp==True  and cut_spot==False) ## strip cut has to be around the orig blob
    partialGlobalAlignment = (glb_tx!=0 and glb_dx==0 and glb_dy==0 and cut_strp==False and cut_spot==False) 
    partGlobalAlgnWithStrp = (glb_tx!=0 and glb_dx==0 and glb_dy==0 and cut_strp==True  and cut_spot==False) ## strip cut has to be around the new blob
    fullGlobalAlignment    = (glb_tx!=0 and glb_dx!=0 and glb_dy!=0 and cut_strp==False and cut_spot==False)
    fullGlobAlgnWithStrip  = (glb_tx!=0 and glb_dx!=0 and glb_dy!=0 and cut_strp==True  and cut_spot==False) ## strip cut has to be around the new blob
    fullGlobAlgFullSel     = (glb_tx!=0 and glb_dx!=0 and glb_dy!=0 and cut_strp==False and cut_spot==True)
    
    algn_label = "NULL"
    if(noGlobalAlignment):      algn_label = "Before global alignment"
    if(noGlobAlgnWithStrip):    algn_label = "Outlier tracks removed"
    if(partialGlobalAlignment): algn_label = "Partial global alignment"
    if(partGlobalAlgnWithStrp): algn_label = "Partial alignment w/o outliers"
    if(fullGlobalAlignment):    algn_label = "Full global alignment"
    if(fullGlobAlgnWithStrip):  algn_label = "Global alignment w/o outliers"
    if(fullGlobAlgFullSel):     algn_label = "Full selection"
    algn_sufix = "NULL"
    if(noGlobalAlignment):      algn_sufix = ""
    if(noGlobAlgnWithStrip):    algn_sufix = "_no_outliers"
    if(partialGlobalAlignment): algn_sufix = "_partial_glob_algn"
    if(partGlobalAlgnWithStrp): algn_sufix = "_partial_glob_algn_no_outliers"
    if(fullGlobalAlignment):    algn_sufix = "_full_glob_algn"
    if(fullGlobAlgnWithStrip):  algn_sufix = "_full_glob_algn_no_outliers"
    if(fullGlobAlgFullSel):     algn_sufix = "_full_selection"
    
    print(f'cfg["xOffset0"]={cfg["xOffset0"]}, cfg["thetax"]={cfg["thetax"]}, cfg["yOffset0"]={cfg["yOffset0"]}, cfg["cut_strip"]={cfg["cut_strip"]}, cfg["cut_spot"]={cfg["cut_spot"]}')
    print(f"noGlobalAlignment={noGlobalAlignment}")
    print(f"noGlobAlgnWithStrip={noGlobAlgnWithStrip}")
    print(f"partialGlobalAlignment={partialGlobalAlignment}")
    print(f"fullGlobalAlignment={fullGlobalAlignment}")
    print(f"algn_label={algn_label}, algn_sufix={algn_sufix}")
    
    
    hDipoleExitNoCuts = histos["hD_before_cuts"].Clone("hDipoleExitNoCuts")
    hDipoleExitNoCuts.SetTitle("Dipole exit plane;x_{LAB} [mm];y_{LAB} [mm];Back-extrapolated tracks")
    ROOT.gStyle.SetPadRightMargin(0.15)
    cnv = ROOT.TCanvas("cnv_dipole_exit_no_cuts","",550,500)
    cnv.SetTicks(1,1)
    cnv.SetGridx()
    cnv.SetGridy()
    if(hDipoleExitNoCuts.GetMaximum()<3): hDipoleExitNoCuts.SetMaximum(3)
    hDipoleExitNoCuts.Draw("colz")
    palette = cnv.GetPrimitive("palette")
    if not palette: palette = hDipoleExitNoCuts.FindObject("palette")
    if palette:
        palette.SetX1NDC(0.86)
        palette.SetX2NDC(0.91)
    cnv.Modified()
    dipole.Draw()
    flange.Draw()
    s = ROOT.TLatex()
    s.SetNDC(1)
    s.SetTextAlign(13)
    s.SetTextColor(ROOT.kBlack)
    s.SetTextFont(22)
    s.SetTextSize(0.045)
    s.DrawLatex(0.15,0.88,f"Run {runnum}")
    #
    s = ROOT.TLatex()
    s.SetNDC(1)
    s.SetTextAlign(13)
    s.SetTextColor(ROOT.kBlack)
    s.SetTextFont(132)
    s.SetTextSize(0.045)
    s.DrawLatex(0.15,0.83,f"#mu_{{x}}={hDipoleExitNoCuts.GetMean(1):.1f} mm, #sigma_{{x}}={hDipoleExitNoCuts.GetStdDev(1):.1f} mm")
    s.DrawLatex(0.15,0.78,f"#mu_{{y}}={hDipoleExitNoCuts.GetMean(2):.1f} mm, #sigma_{{y}}={hDipoleExitNoCuts.GetStdDev(2):.1f} mm")
    #
    s = ROOT.TLatex()
    s.SetNDC(1)
    s.SetTextAlign(13)
    s.SetTextColor(ROOT.kBlack)
    s.SetTextFont(132)
    s.SetTextSize(0.045)
    s.DrawLatex(0.35,0.88,f"({algn_label})")
    #
    s = ROOT.TLatex()
    s.SetNDC(1)
    s.SetTextAlign(13)
    s.SetTextColor(ROOT.kBlue)
    s.SetTextFont(132)
    s.SetTextSize(0.045)
    s.DrawLatex(0.36,0.67,"Dipole aperture")
    #
    s = ROOT.TLatex()
    s.SetNDC(1)
    s.SetTextAlign(13)
    s.SetTextColor(ROOT.kAzure+1)
    s.SetTextFont(132)
    s.SetTextSize(0.045)
    s.DrawLatex(0.13,0.625,"Flange aperture")
    cnv.Update()
    cnv.SaveAs(f'{foupdfname.replace(".pdf","")}_dipole_exit_nocuts{algn_sufix}.pdf')
    del cnv
    print("---------------1")
    
    
    hDipoleExitWithCuts = histos["hD_after_cuts"].Clone("hDipoleExitWithCuts")
    hDipoleExitWithCuts.SetTitle("Dipole exit plane;x_{LAB} [mm];y_{LAB} [mm];Back-extrapolated tracks")
    cnv = ROOT.TCanvas("cnv_dipole_exit_with_cuts","",550,500)
    cnv.cd()
    ROOT.gStyle.SetPadRightMargin(0.15)
    cnv.SetTicks(1,1)
    cnv.SetGridx()
    cnv.SetGridy()
    if(hDipoleExitWithCuts.GetMaximum()<3): hDipoleExitWithCuts.SetMaximum(3)
    hDipoleExitWithCuts.Draw("colz")
    palette = cnv.GetPrimitive("palette")
    if not palette: palette = hDipoleExitWithCuts.FindObject("palette")
    if palette:
        palette.SetX1NDC(0.86)
        palette.SetX2NDC(0.91)
    cnv.Modified()
    dipole.Draw()
    flange.Draw()
    s = ROOT.TLatex()
    s.SetNDC(1)
    s.SetTextAlign(13)
    s.SetTextColor(ROOT.kBlack)
    s.SetTextFont(22)
    s.SetTextSize(0.045)
    s.DrawLatex(0.15,0.88,f"Run {runnum}")
    #
    s = ROOT.TLatex()
    s.SetNDC(1)
    s.SetTextAlign(13)
    s.SetTextColor(ROOT.kBlack)
    s.SetTextFont(132)
    s.SetTextSize(0.045)
    s.DrawLatex(0.15,0.83,f"#mu_{{x}}={hDipoleExitWithCuts.GetMean(1):.1f} mm, #sigma_{{x}}={hDipoleExitWithCuts.GetStdDev(1):.1f} mm")
    s.DrawLatex(0.15,0.78,f"#mu_{{y}}={hDipoleExitWithCuts.GetMean(2):.1f} mm, #sigma_{{y}}={hDipoleExitWithCuts.GetStdDev(2):.1f} mm")
    #
    s = ROOT.TLatex()
    s.SetNDC(1)
    s.SetTextAlign(13)
    s.SetTextColor(ROOT.kBlack)
    s.SetTextFont(132)
    s.SetTextSize(0.045)
    s.DrawLatex(0.35,0.88,f"({algn_label})")
    #
    s = ROOT.TLatex()
    s.SetNDC(1)
    s.SetTextAlign(13)
    s.SetTextColor(ROOT.kBlue)
    s.SetTextFont(132)
    s.SetTextSize(0.045)
    s.DrawLatex(0.36,0.67,"Dipole aperture")
    #
    s = ROOT.TLatex()
    s.SetNDC(1)
    s.SetTextAlign(13)
    s.SetTextColor(ROOT.kAzure+1)
    s.SetTextFont(132)
    s.SetTextSize(0.045)
    s.DrawLatex(0.13,0.625,"Flange aperture")
    cnv.Update()
    cnv.SaveAs(f'{foupdfname.replace(".pdf","")}_dipole_exit_withcuts{algn_sufix}.pdf')
    del cnv
    print("---------------2")
    
    
    
    cnv = ROOT.TCanvas("cnv_dipole_window","",1000,500)
    cnv.Divide(2,1)
    cnv.cd(1)
    ROOT.gPad.SetTicks(1,1)
    ROOT.gPad.SetGridy()
    ROOT.gPad.SetGridx()
    histos["hD_zoomin_before_cuts"].Draw("colz")
    dipole.Draw()
    ROOT.gPad.RedrawAxis()
    cnv.cd(2)
    ROOT.gPad.SetTicks(1,1)
    ROOT.gPad.SetGridy()
    ROOT.gPad.SetGridx()
    histos["hD_zoomin_after_cuts"].Draw("colz")
    dipole.Draw()
    ROOT.gPad.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{foupdfname}")
    del cnv

    cnv = ROOT.TCanvas("cnv_dipole_window","",1000,500)
    cnv.Divide(2,1)
    cnv.cd(1)
    ROOT.gPad.SetTicks(1,1)
    ROOT.gPad.SetGridx()
    ROOT.gPad.SetGridy()
    histos["hF_before_cuts"].Draw("colz")
    flange.Draw()
    ROOT.gPad.RedrawAxis()
    cnv.cd(2)
    ROOT.gPad.SetTicks(1,1)
    ROOT.gPad.SetGridx()
    ROOT.gPad.SetGridy()
    histos["hF_after_cuts"].Draw("colz")
    flange.Draw()
    ROOT.gPad.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{foupdfname}")
    del cnv
    print("---------------3")
    
    cnv = ROOT.TCanvas("cnv_dipole_window","",1000,500)
    cnv.Divide(2,1)
    cnv.cd(1)
    ROOT.gPad.SetTicks(1,1)
    ROOT.gPad.SetGridx()
    ROOT.gPad.SetGridy()
    histos["hW_before_cuts"].Draw("colz")
    window.Draw()
    ROOT.gPad.RedrawAxis()
    cnv.cd(2)
    ROOT.gPad.SetTicks(1,1)
    ROOT.gPad.SetGridx()
    ROOT.gPad.SetGridy()
    histos["hW_after_cuts"].Draw("colz")
    window.Draw()
    ROOT.gPad.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{foupdfname}")
    del cnv
    print("---------------4")
    
    cnv = ROOT.TCanvas("cnv_dipole_window","",1000,500)
    cnv.Divide(2,1)
    cnv.cd(1)
    ROOT.gPad.SetTicks(1,1)
    histos["hTheta_xz_before_cuts"].Draw("hist")
    if(cfg["isMC"] and cfg["isFakeMC"]):
        histos["hTheta_xz_tru"].SetLineColor(ROOT.kRed)
        histos["hTheta_xz_tru"].Draw("hist same")
    cnv.RedrawAxis()
    cnv.cd(2)
    ROOT.gPad.SetTicks(1,1)
    histos["hTheta_xz_after_cuts"].Draw("hist")
    if(cfg["isMC"] and cfg["isFakeMC"]):
        histos["hTheta_xz_tru"].SetLineColor(ROOT.kRed)
        histos["hTheta_xz_tru"].Draw("hist same")
    cnv.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{foupdfname}")
    del cnv
    print("---------------5")
    
    
    grxz = None
    gryz = None
    grpz = None
    if(dotoyhsit):
        ROOT.gStyle.SetErrorX(0.5)
        hxz = histos["hTheta_xz_before_cuts"].Clone("hxz")
        hyz = histos["hTheta_yz_before_cuts"].Clone("hyz")
        hpz = histos["hPf_small"].Clone("hpz")
        hxz.SetBinErrorOption(ROOT.TH1.kPoisson)
        hyz.SetBinErrorOption(ROOT.TH1.kPoisson)
        hxzl = histos["hTheta_xz_before_cuts"].Clone("hxz_down")
        hxzh = histos["hTheta_xz_before_cuts"].Clone("hxz_up")
        hyzl = histos["hTheta_yz_before_cuts"].Clone("hyz_down")
        hyzh = histos["hTheta_yz_before_cuts"].Clone("hyz_up")
        hpzl = histos["hPf_small"].Clone("hpz_down")
        hpzh = histos["hPf_small"].Clone("hpz_up")
        err_xz_rad      = 0.001 ## [rad] ## From step 3 of the local-alignment process in the paper
        err_yz_rad      = 0.001 ## [rad] ## From step 3 of the local-alignment process in the paper
        err_thet_yz_rad = 0.001 ## [rad] ## This is due to the tilt: from the bin width of the θ_yz distribution (i.e., from the uncertainty on θ^max_yz)
        fOutToys = ROOT.TFile(tfilenamein.replace(".root","_toys.root"),"RECREATE")
        hyz.Write()
        hpz.Write()
        for i in range(1000): get_toy(i,arr_theta_xz,htH=hxzh,htL=hxzl,err=err_xz_rad)
        for i in range(1000): get_toy(i,arr_theta_yz,htH=hyzh,htL=hyzl,err=err_yz_rad)
        for i in range(1000): get_toy(i,arr_theta_yz_pass,htH=hyzh,htL=hyzl,err=err_yz_rad,hpzH=hpzh,hpzL=hpzl,err_thet_yz=err_thet_yz_rad)
        fOutToys.Close()
        grxz = get_error_graph("grxz",h0=hxz,hh=hxzh,hl=hxzl)
        gryz = get_error_graph("gryz",h0=hyz,hh=hyzh,hl=hyzl)
        grpz = get_error_graph("grpz",h0=hpz,hh=hpzh,hl=hpzl)
        #
        hxz.GetXaxis().SetTitle("Track #theta_{xz} (LAB frame) [rad]")
        hyz.GetXaxis().SetTitle("Track #theta_{yz} (LAB frame) [rad]")
        hxz.SetMarkerStyle(20)
        hyz.SetMarkerStyle(20)
        hxz.SetMarkerSize(1)
        hyz.SetMarkerSize(1)
        hxz.SetMarkerColor(ROOT.kBlack)
        hyz.SetMarkerColor(ROOT.kBlack)
        hxz.SetLineColor(ROOT.kBlack)
        hyz.SetLineColor(ROOT.kBlack)
        #
        hxz.SetMaximum(420)
        hyz.SetMaximum(250)
        hxz.GetXaxis().SetTitleOffset(1.3)
        hyz.GetXaxis().SetTitleOffset(1.3)
        cnv = ROOT.TCanvas("cnv_dipole_window","",1100,500)
        cnv.Divide(2,1)
        cnv.cd(1)
        ROOT.gPad.SetTicks(1,1)
        hxz.Draw("e1p")
        grxz.Draw("E2 same")
        # grxz.SetMarkerStyle(25)
        # grxz.SetMarkerSize(1)
        # grxz.Draw("e1p same")
        cnv.RedrawAxis()
        s = ROOT.TLatex()
        s.SetNDC(1)
        s.SetTextAlign(13)
        s.SetTextColor(ROOT.kBlack)
        s.SetTextFont(22)
        s.SetTextSize(0.045)
        s.DrawLatex(0.57,0.85,f"Run {runnum}")
        #
        s = ROOT.TLatex()
        s.SetNDC(1)
        s.SetTextAlign(13)
        s.SetTextColor(ROOT.kBlack)
        s.SetTextFont(132)
        s.SetTextSize(0.045)
        s.DrawLatex(0.56,0.80,f"#mu={hxz.GetMean():.3f} rad")
        s.DrawLatex(0.56,0.75,f"#sigma={hxz.GetStdDev():.3f} rad")
        s.DrawLatex(0.56,0.70,f"#theta_{{xz}}^{{max}}={hxz.GetXaxis().GetBinCenter(hxz.GetMaximumBin()):.3f} rad")
        #
        cnv.cd(2)
        ROOT.gPad.SetTicks(1,1)
        hyz.Draw("e1p")
        gryz.Draw("E2 same")
        cnv.RedrawAxis()
        s = ROOT.TLatex()
        s.SetNDC(1)
        s.SetTextAlign(13)
        s.SetTextColor(ROOT.kBlack)
        s.SetTextFont(22)
        s.SetTextSize(0.045)
        s.DrawLatex(0.17,0.85,f"Run {runnum}")
        #
        s = ROOT.TLatex()
        s.SetNDC(1)
        s.SetTextAlign(13)
        s.SetTextColor(ROOT.kBlack)
        s.SetTextFont(132)
        s.SetTextSize(0.045)
        s.DrawLatex(0.16,0.80,f"#mu={hyz.GetMean():.3f} rad")
        s.DrawLatex(0.16,0.75,f"#sigma={hyz.GetStdDev():.3f} rad")
        s.DrawLatex(0.16,0.70,f"#theta_{{yz}}^{{max}}={hyz.GetXaxis().GetBinCenter(hyz.GetMaximumBin()):.3f} rad")
        #
        cnv.Update()
        cnv.SaveAs(f'{foupdfname.replace(".pdf","")}_angles_nocuts.pdf')
        del cnv
        ROOT.gStyle.SetErrorX(0.0)
    print("---------------6")
    
    
    
    
    
    cnv = ROOT.TCanvas("cnv_dipole_window","",1000,500)
    cnv.Divide(2,1)
    cnv.cd(1)
    ROOT.gPad.SetTicks(1,1)
    histos["hTheta_yz_before_cuts"].Draw("hist")
    if(cfg["isMC"] and cfg["isFakeMC"]):
        histos["hTheta_yz_tru"].SetLineColor(ROOT.kRed)
        histos["hTheta_yz_tru"].Draw("hist same")
    cnv.RedrawAxis()
    cnv.cd(2)
    ROOT.gPad.SetTicks(1,1)
    histos["hTheta_yz_after_cuts"].Draw("hist")
    if(cfg["isMC"] and cfg["isFakeMC"]):
        histos["hTheta_yz_tru"].SetLineColor(ROOT.kRed)
        histos["hTheta_yz_tru"].Draw("hist same")
    cnv.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{foupdfname}")
    del cnv
    print("---------------6.1")

    if(cfg["isMC"] and cfg["isFakeMC"]):
        cnv = ROOT.TCanvas("cnv_dipole_window","",1000,500)
        cnv.Divide(2,1)
        cnv.cd(1)
        ROOT.gPad.SetTicks(1,1)
        histos["hTheta_xz_response"].Draw("hist")
        ROOT.gPad.RedrawAxis()
        cnv.cd(2)
        ROOT.gPad.SetTicks(1,1)
        histos["hTheta_yz_response"].Draw("hist")
        ROOT.gPad.RedrawAxis()
        cnv.Update()
        cnv.SaveAs(f"{foupdfname}")
        del cnv
        
        cnv = ROOT.TCanvas("cnv_dipole_window","",1000,500)
        cnv.Divide(2,1)
        cnv.cd(1)
        ROOT.gPad.SetTicks(1,1)
        hTheta_xz_eff = histos["hTheta_xz_tru"].Clone("hTheta_xz_tru_clone") 
        hTheta_xz_eff.Divide(histos["hTheta_xz_tru_all"])    
        hTheta_xz_eff.Draw("hist")
        ROOT.gPad.RedrawAxis()
        cnv.cd(2)
        ROOT.gPad.SetTicks(1,1)
        hTheta_yz_eff = histos["hTheta_yz_tru"].Clone("hTheta_yz_tru_clone") 
        hTheta_yz_eff.Divide(histos["hTheta_yz_tru_all"])
        hTheta_yz_eff.Draw("hist")
        ROOT.gPad.RedrawAxis()
        cnv.Update()
        cnv.SaveAs(f"{foupdfname}")
        del cnv
        
        cnv = ROOT.TCanvas("cnv_dipole_window","",1000,500)
        cnv.Divide(2,1)
        cnv.cd(1)
        ROOT.gPad.SetTicks(1,1)
        histos["hD_x_response"].Draw("hist")
        ROOT.gPad.RedrawAxis()
        cnv.cd(2)
        ROOT.gPad.SetTicks(1,1)
        histos["hD_y_response"].Draw("hist")
        ROOT.gPad.RedrawAxis()
        cnv.Update()
        cnv.SaveAs(f"{foupdfname}")
        del cnv

    cnv = ROOT.TCanvas("cnv_dipole_window","",500,500)
    cnv.SetTicks(1,1)
    histos["hdExit"].Draw("hist")
    cnv.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{foupdfname}")
    del cnv
    print("---------------7")
    
    
    
    
    
    
    
    
    
    # cnv = ROOT.TCanvas("cnv_dipole_window","",1000,500)
    # cnv.Divide(2,1)
    # cnv.cd(1)
    # ROOT.gPad.SetTicks(1,1)
    # histos["hdExit"].Draw("hist")
    # cnv.RedrawAxis()
    # cnv.cd(2)
    # ROOT.gPad.SetTicks(1,1)
    # histos["hThetad_yz"].Draw("hist")
    # cnv.RedrawAxis()
    # cnv.Update()
    # cnv.SaveAs(f"{foupdfname}")
    
    cnv = ROOT.TCanvas("cnv_dipole_window","",1000,500)
    # cnv = ROOT.TCanvas("cnv_dipole_window","",1500,500)
    cnv.Divide(2,1)
    # cnv.Divide(3,1)
    cnv.cd(1)
    ROOT.gPad.SetTicks(1,1)
    histos["hThetaf_yz"].Draw("hist")
    ROOT.gPad.RedrawAxis()
    cnv.cd(2)
    # ROOT.gPad.SetTicks(1,1)
    # histos["hThetad_yz"].Draw("hist")
    # ROOT.gPad.RedrawAxis()
    # cnv.cd(3)
    ROOT.gPad.SetTicks(1,1)
    histos["hThetar_yz"].Draw("hist")
    ROOT.gPad.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{foupdfname}")
    del cnv
    print("---------------8")
    

    cnv = ROOT.TCanvas("cnv_dipole_window","",1000,500)
    # cnv = ROOT.TCanvas("cnv_dipole_window","",1500,500)
    cnv.Divide(2,1)
    # cnv.Divide(3,1)
    cnv.cd(1)
    ROOT.gPad.SetTicks(1,1)
    histos["hPf_small"].Draw("hist")
    cnv.RedrawAxis()
    cnv.cd(2)
    # ROOT.gPad.SetTicks(1,1)
    # histos["hPd_small"].Draw("hist")
    # cnv.RedrawAxis()
    cnv.cd(3)
    ROOT.gPad.SetTicks(1,1)
    histos["hPr_small"].Draw("hist")
    cnv.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{foupdfname}")
    del cnv
    print("---------------9")
    
    
    cnv = ROOT.TCanvas("cnv_dipole_window","",1000,500)
    # cnv = ROOT.TCanvas("cnv_dipole_window","",1500,500)
    cnv.Divide(2,1)
    # cnv.Divide(3,1)
    cnv.cd(1)
    ROOT.gPad.SetTicks(1,1)
    histos["hPf_zoom"].Draw("hist")
    cnv.RedrawAxis()
    cnv.cd(2)
    # ROOT.gPad.SetTicks(1,1)
    # histos["hPd_zoom"].Draw("hist")
    # cnv.RedrawAxis()
    cnv.cd(3)
    ROOT.gPad.SetTicks(1,1)
    histos["hPr_zoom"].Draw("hist")
    cnv.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{foupdfname}")
    del cnv
    print("---------------10")
    
    
    cnv = ROOT.TCanvas("cnv_dipole_window","",1000,500)
    # cnv = ROOT.TCanvas("cnv_dipole_window","",1500,500)
    cnv.Divide(2,1)
    # cnv.Divide(3,1)
    cnv.cd(1)
    ROOT.gPad.SetTicks(1,1)
    histos["hPf"].Draw("hist")
    cnv.RedrawAxis()
    cnv.cd(2)
    # ROOT.gPad.SetTicks(1,1)
    # histos["hPd"].Draw("hist")
    # cnv.RedrawAxis()
    cnv.cd(3)
    ROOT.gPad.SetTicks(1,1)
    histos["hPr"].Draw("hist")
    cnv.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{foupdfname}")
    del cnv
    
    cnv = ROOT.TCanvas("cnv_dipole_window","",500,500)
    cnv.SetTicks(1,1)
    histos["hPf"].Draw("hist")
    cnv.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{foupdfname}")
    del cnv
    print("---------------11")
    
    
    # cnv = ROOT.TCanvas("cnv_dipole_window","",500,500)
    # cnv.SetTicks(1,1)
    # histos["hPd"].Draw("hist")
    # cnv.RedrawAxis()
    # cnv.Update()
    # cnv.SaveAs(f"{foupdfname}")
    
    cnv = ROOT.TCanvas("cnv_dipole_window","",500,500)
    cnv.SetTicks(1,1)
    histos["hPr"].Draw("hist")
    cnv.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{foupdfname}")
    del cnv
    print("---------------12")
    
    
    # cnv = ROOT.TCanvas("cnv_dipole_window","",1000,500)
    # # cnv = ROOT.TCanvas("cnv_dipole_window","",1500,500)
    # cnv.Divide(2,1)
    # # cnv.Divide(3,1)
    # cnv.cd(1)
    # ROOT.gPad.SetTicks(1,1)
    # histos["hPf_vs_dExit"].Draw("colz")
    # ROOT.gPad.RedrawAxis()
    # cnv.cd(2)
    # # ROOT.gPad.SetTicks(1,1)
    # # histos["hPd_vs_dExit"].Draw("colz")
    # # ROOT.gPad.RedrawAxis()
    # # cnv.cd(3)
    # ROOT.gPad.SetTicks(1,1)
    # histos["hPr_vs_dExit"].Draw("colz")
    # ROOT.gPad.RedrawAxis()
    # cnv.Update()
    # cnv.SaveAs(f"{foupdfname}")
    
    cnv = ROOT.TCanvas("cnv_dipole_window","",1000,500)
    # cnv = ROOT.TCanvas("cnv_dipole_window","",1500,500)
    cnv.Divide(2,1)
    # cnv.Divide(3,1)
    cnv.cd(1)
    ROOT.gPad.SetTicks(1,1)
    histos["hPf_vs_thetaf"].Draw("colz")
    ROOT.gPad.RedrawAxis()
    cnv.cd(2)
    # ROOT.gPad.SetTicks(1,1)
    # histos["hPd_vs_thetad"].Draw("colz")
    # ROOT.gPad.RedrawAxis()
    # cnv.cd(3)
    ROOT.gPad.SetTicks(1,1)
    histos["hPr_vs_thetar"].Draw("colz")
    ROOT.gPad.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{foupdfname}")
    del cnv
    print("---------------13")
    
    # cnv = ROOT.TCanvas("cnv_dipole_window","",1500,500)
    # cnv.Divide(3,1)
    # cnv.cd(1)
    # # ROOT.gPad.SetLogy()
    # ROOT.gPad.SetTicks(1,1)
    # histos["hDexit_vs_thetaf"].Draw("colz")
    # ROOT.gPad.RedrawAxis()
    # cnv.cd(2)
    # # ROOT.gPad.SetLogy()
    # ROOT.gPad.SetTicks(1,1)
    # histos["hDexit_vs_thetad"].Draw("colz")
    # ROOT.gPad.RedrawAxis()
    # cnv.cd(3)
    # # ROOT.gPad.SetLogy()
    # ROOT.gPad.SetTicks(1,1)
    # histos["hDexit_vs_thetar"].Draw("colz")
    # ROOT.gPad.RedrawAxis()
    # cnv.Update()
    # cnv.SaveAs(f"{foupdfname}")
    
    cnv = ROOT.TCanvas("cnv_dipole_window","",500,500)
    # cnv = ROOT.TCanvas("cnv_dipole_window","",1000,500)
    # cnv.Divide(2,1)
    # cnv.cd(1)
    # ROOT.gPad.SetTicks(1,1)
    # histos["hThetad_vs_thetaf"].Draw("colz")
    # dipole.Draw()
    # ROOT.gPad.RedrawAxis()
    # cnv.cd(2)
    ROOT.gPad.SetTicks(1,1)
    histos["hThetar_vs_thetaf"].Draw("colz")
    dipole.Draw()
    ROOT.gPad.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{foupdfname}")
    del cnv
    print("---------------14")
    
    
    leg = ROOT.TLegend(0.3,0.8,0.7,0.88)
    leg.SetFillStyle(4000) # will be transparent
    leg.SetFillColor(0)
    leg.SetTextFont(42)
    leg.SetTextSize(0.037)
    leg.SetBorderSize(0)
    leg.AddEntry(histos["hChi2DoF_full_alowshrcls"],"Baseline w/shared clusters","l")
    leg.AddEntry(histos["hChi2DoF_full_zeroshrcls"],"Baseline w/o shared clusters","l")
    
    cnv = ROOT.TCanvas("cnv_dipole_window","",500,500)
    cnv.SetTicks(1,1)
    hmax = h1h2max(histos["hChi2DoF_full_alowshrcls"],histos["hChi2DoF_full_zeroshrcls"])
    histos["hChi2DoF_full_alowshrcls"].SetMinimum(0)
    histos["hChi2DoF_full_zeroshrcls"].SetMinimum(0)
    histos["hChi2DoF_full_alowshrcls"].SetMaximum(1.1*hmax)
    histos["hChi2DoF_full_zeroshrcls"].SetMaximum(1.1*hmax)  
    histos["hChi2DoF_full_alowshrcls"].SetLineColor(ROOT.kBlack)
    histos["hChi2DoF_full_zeroshrcls"].SetLineColor(ROOT.kRed)
    histos["hChi2DoF_full_alowshrcls"].Draw("hist")
    histos["hChi2DoF_full_zeroshrcls"].Draw("hist same")
    leg.Draw("same")
    cnv.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{foupdfname}")
    del cnv
    print("---------------15")
    
    
    cnv = ROOT.TCanvas("cnv_dipole_window","",500,500)
    cnv.SetTicks(1,1)
    hmax = h1h2max(histos["hChi2DoF_mid_alowshrcls"],histos["hChi2DoF_mid_zeroshrcls"])
    histos["hChi2DoF_mid_alowshrcls"].SetMinimum(0)
    histos["hChi2DoF_mid_zeroshrcls"].SetMinimum(0)
    histos["hChi2DoF_mid_alowshrcls"].SetMaximum(1.1*hmax)
    histos["hChi2DoF_mid_zeroshrcls"].SetMaximum(1.1*hmax)  
    histos["hChi2DoF_mid_alowshrcls"].SetLineColor(ROOT.kBlack)
    histos["hChi2DoF_mid_zeroshrcls"].SetLineColor(ROOT.kRed)
    histos["hChi2DoF_mid_alowshrcls"].Draw("hist")
    histos["hChi2DoF_mid_zeroshrcls"].Draw("hist same")
    leg.Draw("same")
    cnv.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{foupdfname}")
    del cnv
    print("---------------16")
    
    
    cnv = ROOT.TCanvas("cnv_dipole_window","",500,500)
    cnv.SetTicks(1,1)
    hmax = h1h2max(histos["hChi2DoF_alowshrcls"],histos["hChi2DoF_zeroshrcls"])
    histos["hChi2DoF_alowshrcls"].SetMaximum(1.1*hmax)
    histos["hChi2DoF_zeroshrcls"].SetMaximum(1.1*hmax)
    histos["hChi2DoF_alowshrcls"].SetMinimum(0)
    histos["hChi2DoF_zeroshrcls"].SetMinimum(0)
    histos["hChi2DoF_alowshrcls"].SetLineColor(ROOT.kBlack)
    histos["hChi2DoF_zeroshrcls"].SetLineColor(ROOT.kRed)
    histos["hChi2DoF_alowshrcls"].Draw("hist")
    histos["hChi2DoF_zeroshrcls"].Draw("hist same")
    leg.Draw("same")
    cnv.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{foupdfname}")
    del cnv
    print("---------------17")
    
    
    cnv = ROOT.TCanvas("cnv_dipole_window","",500,500)
    cnv.SetTicks(1,1)
    hmax = h1h2max(histos["hChi2DoF_small_alowshrcls"],histos["hChi2DoF_small_zeroshrcls"])
    histos["hChi2DoF_small_alowshrcls"].SetMinimum(0)
    histos["hChi2DoF_small_zeroshrcls"].SetMinimum(0)
    histos["hChi2DoF_small_alowshrcls"].SetMaximum(1.1*hmax)
    histos["hChi2DoF_small_zeroshrcls"].SetMaximum(1.1*hmax)
    histos["hChi2DoF_small_alowshrcls"].SetLineColor(ROOT.kBlack)
    histos["hChi2DoF_small_zeroshrcls"].SetLineColor(ROOT.kRed)
    histos["hChi2DoF_small_alowshrcls"].Draw("hist")
    histos["hChi2DoF_small_zeroshrcls"].Draw("hist same")
    leg.Draw("same")
    cnv.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{foupdfname}")
    del cnv
    print("---------------18")
    
    
    cnv = ROOT.TCanvas("cnv_dipole_window","",500,500)
    cnv.SetTicks(1,1)
    hmax = h1h2max(histos["hChi2DoF_zoom_alowshrcls"],histos["hChi2DoF_zoom_zeroshrcls"])
    histos["hChi2DoF_zoom_alowshrcls"].SetMinimum(0)
    histos["hChi2DoF_zoom_zeroshrcls"].SetMinimum(0)
    histos["hChi2DoF_zoom_alowshrcls"].SetMaximum(1.1*hmax)
    histos["hChi2DoF_zoom_zeroshrcls"].SetMaximum(1.1*hmax)
    histos["hChi2DoF_zoom_alowshrcls"].SetLineColor(ROOT.kBlack)
    histos["hChi2DoF_zoom_zeroshrcls"].SetLineColor(ROOT.kRed)
    histos["hChi2DoF_zoom_alowshrcls"].Draw("hist")
    histos["hChi2DoF_zoom_zeroshrcls"].Draw("hist same")
    leg.Draw("same")
    cnv.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{foupdfname}")
    del cnv
    print("---------------19")
    
    
    
    cnv = ROOT.TCanvas("cnv_occupancy_clusters","",1500,500)
    cnv.Divide(5,1)
    for idet,det in enumerate(cfg["detectors"]):
        cnv.cd(idet+1)
        ROOT.gPad.SetTicks(1,1)
        histos[f"h_cls_occ_2D_{det}"].Draw("colz")
        ROOT.gPad.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{foupdfname}")
    del cnv
    print("---------------19.1")
    
    cnv = ROOT.TCanvas("cnv_occupancy_clusters_after","",1500,500)
    cnv.Divide(5,1)
    for idet,det in enumerate(cfg["detectors"]):
        cnv.cd(idet+1)
        ROOT.gPad.SetTicks(1,1)
        histos[f"h_cls_occ_2D_{det}_after_cuts"].Draw("colz")
        ROOT.gPad.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{foupdfname}")
    del cnv
    print("---------------19.11")
    
    cnv = ROOT.TCanvas("cnv_occupancy_clusters","",1500,500)
    cnv.Divide(5,1)
    for idet,det in enumerate(cfg["detectors"]):
        cnv.cd(idet+1)
        ROOT.gPad.SetTicks(1,1)
        histos[f"h_trk_occ_2D_{det}"].Draw("colz")
        ROOT.gPad.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{foupdfname}")
    del cnv
    print("---------------19.2")
    
    cnv = ROOT.TCanvas("cnv_occupancy_clusters_after","",1500,500)
    cnv.Divide(5,1)
    for idet,det in enumerate(cfg["detectors"]):
        cnv.cd(idet+1)
        ROOT.gPad.SetTicks(1,1)
        histos[f"h_trk_occ_2D_{det}_after_cuts"].Draw("colz")
        ROOT.gPad.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{foupdfname}")
    del cnv
    print("---------------19.22")
    
    
    cnv = ROOT.TCanvas("cnv_occupancy_clusters","",1000,500)
    cnv.Divide(2,1)
    cnv.cd(1)
    ROOT.gPad.SetTicks(1,1)
    ROOT.gPad.SetLogy()
    histos[f"h_cls_absdx"].Draw("hist")
    ROOT.gPad.RedrawAxis()
    cnv.cd(2)
    ROOT.gPad.SetTicks(1,1)
    ROOT.gPad.SetLogy()
    histos[f"h_cls_absdy"].Draw("hist")
    ROOT.gPad.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{foupdfname}")
    del cnv
    print("---------------19.3")
    
    cnv = ROOT.TCanvas("cnv_theta2","",1100,500)
    cnv.Divide(2,1)
    cnv.cd(1)
    ROOT.gPad.SetTicks(2,1)
    ROOT.gPad.SetLogx()
    ROOT.gPad.SetLogy()
    histos[f"h_MLE_theta1_logx_before_cuts"].SetLineColor(ROOT.kBlack)
    histos[f"h_MLE_theta1_logx_before_cuts"].Draw("hist")
    histos[f"h_MLE_theta1_logx_after_cuts"].SetLineColor(ROOT.kRed)
    histos[f"h_MLE_theta1_logx_after_cuts"].Draw("hist same")
    ROOT.gPad.RedrawAxis()
    cnv.cd(2)
    ROOT.gPad.SetTicks(2,1)
    ROOT.gPad.SetLogx()
    ROOT.gPad.SetLogy()
    histos[f"h_MLE_theta2_logx_before_cuts"].SetLineColor(ROOT.kBlack)
    histos[f"h_MLE_theta2_logx_before_cuts"].Draw("hist")
    histos[f"h_MLE_theta2_logx_after_cuts"].SetLineColor(ROOT.kRed)
    histos[f"h_MLE_theta2_logx_after_cuts"].Draw("hist same")
    ROOT.gPad.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{foupdfname}")
    del cnv
    print("---------------19.4")
    
    cnv = ROOT.TCanvas("cnv_theta2_lin","",1100,500)
    cnv.Divide(2,1)
    cnv.cd(1)
    ROOT.gPad.SetTicks(2,1)
    # ROOT.gPad.SetLogx()
    # ROOT.gPad.SetLogy()
    histos[f"h_MLE_theta1_linx_before_cuts"].SetLineColor(ROOT.kBlack)
    histos[f"h_MLE_theta1_linx_before_cuts"].Draw("hist")
    histos[f"h_MLE_theta1_linx_after_cuts"].SetLineColor(ROOT.kRed)
    histos[f"h_MLE_theta1_linx_after_cuts"].Draw("hist same")
    ROOT.gPad.RedrawAxis()
    cnv.cd(2)
    ROOT.gPad.SetTicks(2,1)
    # ROOT.gPad.SetLogx()
    # ROOT.gPad.SetLogy()
    histos[f"h_MLE_theta2_linx_before_cuts"].SetLineColor(ROOT.kBlack)
    histos[f"h_MLE_theta2_linx_before_cuts"].Draw("hist")
    histos[f"h_MLE_theta2_linx_after_cuts"].SetLineColor(ROOT.kRed)
    histos[f"h_MLE_theta2_linx_after_cuts"].Draw("hist same")
    ROOT.gPad.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{foupdfname}")
    del cnv
    print("---------------19.5")
    
    
    
    
    cnv = ROOT.TCanvas("cnv_dipole_window","",1500,1000)
    cnv.Divide(3,2)
    for idet,det in enumerate(cfg["detectors"]):
        cnv.cd(idet+1)
        ROOT.gPad.SetTicks(1,1)
        histos[f"h_residual_zeroshrcls_xy_{det}"].Draw("colz")
        ROOT.gPad.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{foupdfname}")
    del cnv
    print("---------------20")
    
    cnv = ROOT.TCanvas("cnv_dipole_window","",2500,500)
    cnv.Divide(5,1)
    for idet,det in enumerate(cfg["detectors"]):
        cnv.cd(idet+1)
        ROOT.gPad.SetTicks(1,1)
        histos[f"h_residual_zeroshrcls_xy_{det}"].Draw("colz")
        ROOT.gPad.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{foupdfname}")
    del cnv
    print("---------------21")
    
    cnv = ROOT.TCanvas("cnv_dipole_window","",1500,1000)
    cnv.Divide(3,2)
    for idet,det in enumerate(cfg["detectors"]):
        cnv.cd(idet+1)
        ROOT.gPad.SetTicks(1,1)
        histos[f"h_residual_zeroshrcls_xy_mid_{det}"].Draw("colz")
        ROOT.gPad.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{foupdfname}")
    del cnv
    print("---------------22")
    
    cnv = ROOT.TCanvas("cnv_dipole_window","",2500,500)
    cnv.Divide(5,1)
    for idet,det in enumerate(cfg["detectors"]):
        cnv.cd(idet+1)
        ROOT.gPad.SetTicks(1,1)
        histos[f"h_residual_zeroshrcls_xy_mid_{det}"].Draw("colz")
        ROOT.gPad.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{foupdfname}")
    del cnv
    print("---------------23")
    

    cnv = ROOT.TCanvas("cnv_dipole_window","",1500,1000)
    cnv.Divide(3,2)
    for idet,det in enumerate(cfg["detectors"]):
        cnv.cd(idet+1)
        ROOT.gPad.SetTicks(1,1)
        #
        histos[f"h_residual_zeroshrcls_x_sml_{det}"].SetMinimum(0)
        histos[f"h_residual_zeroshrcls_x_sml_{det}"].SetMarkerStyle(24)
        histos[f"h_residual_zeroshrcls_x_sml_{det}"].SetMarkerColor(ROOT.kRed)
        histos[f"h_residual_zeroshrcls_x_sml_{det}"].SetLineColor(ROOT.kRed)
        histos[f"h_residual_zeroshrcls_x_sml_{det}"].Draw("e1p")
        xmin = histos[f"h_residual_zeroshrcls_x_sml_{det}"].GetXaxis().GetXmin()
        xmax = histos[f"h_residual_zeroshrcls_x_sml_{det}"].GetXaxis().GetXmax()
        mm2um = 1e3
        func = fit1(histos[f"h_residual_zeroshrcls_x_sml_{det}"],ROOT.kRed,xmin,xmax)
        s = ROOT.TLatex()
        s.SetNDC(1)
        s.SetTextAlign(13)
        s.SetTextColor(ROOT.kBlack)
        s.SetTextFont(22)
        s.SetTextSize(0.045)
        s.DrawLatex(0.17,0.85,ROOT.Form("Mean: %.2f #mum" % (mm2um*func.GetParameter(1))))
        s.DrawLatex(0.17,0.78,ROOT.Form("Sigma: %.2f #mum" % (mm2um*func.GetParameter(2))))
        if(func.GetNDF()>0): s.DrawLatex(0.2,0.71,ROOT.Form("#chi^{2}/N_{DOF}: %.2f" % (func.GetChisquare()/func.GetNDF())))
        #
        ROOT.gPad.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{foupdfname}")
    del cnv
    print("---------------24")
    
    
    cnv = ROOT.TCanvas("cnv_dipole_window","",2500,500)
    cnv.Divide(5,1)
    for idet,det in enumerate(cfg["detectors"]):
        cnv.cd(idet+1)
        ROOT.gPad.SetTicks(1,1)
        #
        histos[f"h_residual_zeroshrcls_x_sml_{det}"].SetMinimum(0)
        histos[f"h_residual_zeroshrcls_x_sml_{det}"].SetMarkerStyle(24)
        histos[f"h_residual_zeroshrcls_x_sml_{det}"].SetMarkerColor(ROOT.kRed)
        histos[f"h_residual_zeroshrcls_x_sml_{det}"].SetLineColor(ROOT.kRed)
        histos[f"h_residual_zeroshrcls_x_sml_{det}"].Draw("e1p")
        xmin = histos[f"h_residual_zeroshrcls_x_sml_{det}"].GetXaxis().GetXmin()
        xmax = histos[f"h_residual_zeroshrcls_x_sml_{det}"].GetXaxis().GetXmax()
        mm2um = 1e3
        func = fit1(histos[f"h_residual_zeroshrcls_x_sml_{det}"],ROOT.kRed,xmin,xmax)
        s = ROOT.TLatex()
        s.SetNDC(1)
        s.SetTextAlign(13)
        s.SetTextColor(ROOT.kBlack)
        s.SetTextFont(22)
        s.SetTextSize(0.045)
        s.DrawLatex(0.17,0.85,ROOT.Form("Mean: %.2f #mum" % (mm2um*func.GetParameter(1))))
        s.DrawLatex(0.17,0.78,ROOT.Form("Sigma: %.2f #mum" % (mm2um*func.GetParameter(2))))
        if(func.GetNDF()>0): s.DrawLatex(0.2,0.71,ROOT.Form("#chi^{2}/N_{DOF}: %.2f" % (func.GetChisquare()/func.GetNDF())))
        #
        ROOT.gPad.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{foupdfname}")
    del cnv
    print("---------------25")
    

    cnv = ROOT.TCanvas("cnv_dipole_window","",1500,1000)
    cnv.Divide(3,2)
    for idet,det in enumerate(cfg["detectors"]):
        cnv.cd(idet+1)
        ROOT.gPad.SetTicks(1,1)
        #
        histos[f"h_residual_alowshrcls_x_mid_{det}"].SetMinimum(0)
        histos[f"h_residual_zeroshrcls_x_mid_{det}"].SetMinimum(0)
        hbmax = histos[f"h_residual_alowshrcls_x_mid_{det}"].GetMaximum()
        hamax = histos[f"h_residual_zeroshrcls_x_mid_{det}"].GetMaximum()
        hmax = hbmax if(hbmax>hamax) else hamax
        hmax *= 1.2
        histos[f"h_residual_alowshrcls_x_mid_{det}"].SetMaximum(hmax)
        histos[f"h_residual_zeroshrcls_x_mid_{det}"].SetMaximum(hmax)
        
        histos[f"h_residual_alowshrcls_x_mid_{det}"].SetMarkerStyle(20)
        histos[f"h_residual_alowshrcls_x_mid_{det}"].SetMarkerColor(ROOT.kBlack)
        histos[f"h_residual_alowshrcls_x_mid_{det}"].SetLineColor(ROOT.kBlack)
        histos[f"h_residual_alowshrcls_x_mid_{det}"].Draw("ep")
        
        histos[f"h_residual_zeroshrcls_x_mid_{det}"].SetMarkerStyle(24)
        histos[f"h_residual_zeroshrcls_x_mid_{det}"].SetMarkerColor(ROOT.kRed)
        histos[f"h_residual_zeroshrcls_x_mid_{det}"].SetLineColor(ROOT.kRed)
        histos[f"h_residual_zeroshrcls_x_mid_{det}"].Draw("ep same")
        
        ROOT.gPad.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{foupdfname}")
    del cnv
    print("---------------26")
    
    
    cnv = ROOT.TCanvas("cnv_dipole_window","",1500,1000)
    cnv.Divide(3,2)
    for idet,det in enumerate(cfg["detectors"]):
        cnv.cd(idet+1)
        ROOT.gPad.SetTicks(1,1)
        
        histos[f"h_residual_alowshrcls_x_ful_{det}"].SetMinimum(0)
        histos[f"h_residual_zeroshrcls_x_ful_{det}"].SetMinimum(0)
        hbmax = histos[f"h_residual_alowshrcls_x_ful_{det}"].GetMaximum()
        hamax = histos[f"h_residual_zeroshrcls_x_ful_{det}"].GetMaximum()
        hmax = hbmax if(hbmax>hamax) else hamax
        hmax *= 1.2
        histos[f"h_residual_alowshrcls_x_ful_{det}"].SetMaximum(hmax)
        histos[f"h_residual_zeroshrcls_x_ful_{det}"].SetMaximum(hmax)
        
        histos[f"h_residual_alowshrcls_x_ful_{det}"].SetMarkerStyle(20)
        histos[f"h_residual_alowshrcls_x_ful_{det}"].SetMarkerColor(ROOT.kBlack)
        histos[f"h_residual_alowshrcls_x_ful_{det}"].SetLineColor(ROOT.kBlack)
        histos[f"h_residual_alowshrcls_x_ful_{det}"].Draw("ep")
        
        histos[f"h_residual_zeroshrcls_x_ful_{det}"].SetMarkerStyle(24)
        histos[f"h_residual_zeroshrcls_x_ful_{det}"].SetMarkerColor(ROOT.kRed)
        histos[f"h_residual_zeroshrcls_x_ful_{det}"].SetLineColor(ROOT.kRed)
        histos[f"h_residual_zeroshrcls_x_ful_{det}"].Draw("ep same")
        
        ROOT.gPad.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{foupdfname}")
    del cnv
    print("---------------27")
    
    
    cnv = ROOT.TCanvas("cnv_dipole_window","",1500,1000)
    cnv.Divide(3,2)
    for idet,det in enumerate(cfg["detectors"]):
        cnv.cd(idet+1)
        ROOT.gPad.SetTicks(1,1)
        
        histos[f"h_residual_zeroshrcls_y_sml_{det}"].SetMinimum(0)
        histos[f"h_residual_zeroshrcls_y_sml_{det}"].SetMarkerStyle(24)
        histos[f"h_residual_zeroshrcls_y_sml_{det}"].SetMarkerColor(ROOT.kRed)
        histos[f"h_residual_zeroshrcls_y_sml_{det}"].SetLineColor(ROOT.kRed)
        histos[f"h_residual_zeroshrcls_y_sml_{det}"].Draw("e1p")
        xmin = histos[f"h_residual_zeroshrcls_y_sml_{det}"].GetXaxis().GetXmin()
        xmax = histos[f"h_residual_zeroshrcls_y_sml_{det}"].GetXaxis().GetXmax()
        mm2um = 1e3
        func = fit1(histos[f"h_residual_zeroshrcls_y_sml_{det}"],ROOT.kRed,xmin,xmax)
        s = ROOT.TLatex()
        s.SetNDC(1)
        s.SetTextAlign(13)
        s.SetTextColor(ROOT.kBlack)
        s.SetTextFont(22)
        s.SetTextSize(0.045)
        s.DrawLatex(0.17,0.85,ROOT.Form("Mean: %.2f #mum" % (mm2um*func.GetParameter(1))))
        s.DrawLatex(0.17,0.78,ROOT.Form("Sigma: %.2f #mum" % (mm2um*func.GetParameter(2))))
        if(func.GetNDF()>0): s.DrawLatex(0.2,0.71,ROOT.Form("#chi^{2}/N_{DOF}: %.2f" % (func.GetChisquare()/func.GetNDF())))
        
        ROOT.gPad.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{foupdfname}")
    del cnv
    print("---------------28")
    
    
    cnv = ROOT.TCanvas("cnv_dipole_window","",2500,500)
    cnv.Divide(5,1)
    for idet,det in enumerate(cfg["detectors"]):
        cnv.cd(idet+1)
        ROOT.gPad.SetTicks(1,1)
        
        histos[f"h_residual_zeroshrcls_y_sml_{det}"].SetMinimum(0)
        histos[f"h_residual_zeroshrcls_y_sml_{det}"].SetMarkerStyle(24)
        histos[f"h_residual_zeroshrcls_y_sml_{det}"].SetMarkerColor(ROOT.kRed)
        histos[f"h_residual_zeroshrcls_y_sml_{det}"].SetLineColor(ROOT.kRed)
        histos[f"h_residual_zeroshrcls_y_sml_{det}"].Draw("e1p")
        xmin = histos[f"h_residual_zeroshrcls_y_sml_{det}"].GetXaxis().GetXmin()
        xmax = histos[f"h_residual_zeroshrcls_y_sml_{det}"].GetXaxis().GetXmax()
        mm2um = 1e3
        func = fit1(histos[f"h_residual_zeroshrcls_y_sml_{det}"],ROOT.kRed,xmin,xmax)
        s = ROOT.TLatex()
        s.SetNDC(1)
        s.SetTextAlign(13)
        s.SetTextColor(ROOT.kBlack)
        s.SetTextFont(22)
        s.SetTextSize(0.045)
        s.DrawLatex(0.17,0.85,ROOT.Form("Mean: %.2f #mum" % (mm2um*func.GetParameter(1))))
        s.DrawLatex(0.17,0.78,ROOT.Form("Sigma: %.2f #mum" % (mm2um*func.GetParameter(2))))
        if(func.GetNDF()>0): s.DrawLatex(0.2,0.71,ROOT.Form("#chi^{2}/N_{DOF}: %.2f" % (func.GetChisquare()/func.GetNDF())))
        
        ROOT.gPad.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{foupdfname}")
    del cnv
    print("---------------29")
    
    
    cnv = ROOT.TCanvas("cnv_dipole_window","",1500,1000)
    cnv.Divide(3,2)
    for idet,det in enumerate(cfg["detectors"]):
        cnv.cd(idet+1)
        ROOT.gPad.SetTicks(1,1)

        histos[f"h_residual_alowshrcls_y_mid_{det}"].SetMinimum(0)
        histos[f"h_residual_zeroshrcls_y_mid_{det}"].SetMinimum(0)
        hbmax = histos[f"h_residual_alowshrcls_y_mid_{det}"].GetMaximum()
        hamax = histos[f"h_residual_zeroshrcls_y_mid_{det}"].GetMaximum()
        hmax = hbmax if(hbmax>hamax) else hamax
        hmax *= 1.2
        histos[f"h_residual_alowshrcls_y_mid_{det}"].SetMaximum(hmax)
        histos[f"h_residual_zeroshrcls_y_mid_{det}"].SetMaximum(hmax)
        
        histos[f"h_residual_alowshrcls_y_mid_{det}"].SetMarkerStyle(20)
        histos[f"h_residual_alowshrcls_y_mid_{det}"].SetMarkerColor(ROOT.kBlack)
        histos[f"h_residual_alowshrcls_y_mid_{det}"].SetLineColor(ROOT.kBlack)
        histos[f"h_residual_alowshrcls_y_mid_{det}"].Draw("ep")
        
        histos[f"h_residual_zeroshrcls_y_mid_{det}"].SetMarkerStyle(24)
        histos[f"h_residual_zeroshrcls_y_mid_{det}"].SetMarkerColor(ROOT.kRed)
        histos[f"h_residual_zeroshrcls_y_mid_{det}"].SetLineColor(ROOT.kRed)
        histos[f"h_residual_zeroshrcls_y_mid_{det}"].Draw("ep same")
        
        ROOT.gPad.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{foupdfname}")
    del cnv
    print("---------------30")
    
    
    cnv = ROOT.TCanvas("cnv_dipole_window","",1500,1000)
    cnv.Divide(3,2)
    for idet,det in enumerate(cfg["detectors"]):
        cnv.cd(idet+1)
        ROOT.gPad.SetTicks(1,1)

        histos[f"h_residual_alowshrcls_y_ful_{det}"].SetMinimum(0)
        histos[f"h_residual_zeroshrcls_y_ful_{det}"].SetMinimum(0)
        hbmax = histos[f"h_residual_alowshrcls_y_ful_{det}"].GetMaximum()
        hamax = histos[f"h_residual_zeroshrcls_y_ful_{det}"].GetMaximum()
        hmax = hbmax if(hbmax>hamax) else hamax
        hmax *= 1.2
        histos[f"h_residual_alowshrcls_y_ful_{det}"].SetMaximum(hmax)
        histos[f"h_residual_zeroshrcls_y_ful_{det}"].SetMaximum(hmax)
        
        histos[f"h_residual_alowshrcls_y_ful_{det}"].SetMarkerStyle(20)
        histos[f"h_residual_alowshrcls_y_ful_{det}"].SetMarkerColor(ROOT.kBlack)
        histos[f"h_residual_alowshrcls_y_ful_{det}"].SetLineColor(ROOT.kBlack)
        histos[f"h_residual_alowshrcls_y_ful_{det}"].Draw("ep")
        
        histos[f"h_residual_zeroshrcls_y_ful_{det}"].SetMarkerStyle(24)
        histos[f"h_residual_zeroshrcls_y_ful_{det}"].SetMarkerColor(ROOT.kRed)
        histos[f"h_residual_zeroshrcls_y_ful_{det}"].SetLineColor(ROOT.kRed)
        histos[f"h_residual_zeroshrcls_y_ful_{det}"].Draw("ep same")
        
        ROOT.gPad.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{foupdfname}")
    del cnv
    print("---------------31")
    
    
    cnv = ROOT.TCanvas("cnv_dipole_window","",1500,1000)
    cnv.Divide(3,2)
    for idet,det in enumerate(cfg["detectors"]):
        cnv.cd(idet+1)
        ROOT.gPad.SetTicks(1,1)
        
        histos[f"h_response_zeroshrcls_x_sml_{det}"].SetMinimum(0)
        histos[f"h_response_zeroshrcls_x_sml_{det}"].SetMarkerStyle(24)
        histos[f"h_response_zeroshrcls_x_sml_{det}"].SetMarkerColor(ROOT.kRed)
        histos[f"h_response_zeroshrcls_x_sml_{det}"].SetLineColor(ROOT.kRed)
        histos[f"h_response_zeroshrcls_x_sml_{det}"].Draw("e1p")
        
        xmin = histos[f"h_response_zeroshrcls_x_sml_{det}"].GetXaxis().GetXmin()
        xmax = histos[f"h_response_zeroshrcls_x_sml_{det}"].GetXaxis().GetXmax()
        func = fit1(histos[f"h_response_zeroshrcls_x_sml_{det}"],ROOT.kRed,xmin,xmax)
        s = ROOT.TLatex()
        s.SetNDC(1)
        s.SetTextAlign(13)
        s.SetTextColor(ROOT.kBlack)
        s.SetTextFont(22)
        s.SetTextSize(0.045)
        s.DrawLatex(0.17,0.85,ROOT.Form("Mean: %.2f" % (func.GetParameter(1))))
        s.DrawLatex(0.17,0.78,ROOT.Form("Sigma: %.2f" % (func.GetParameter(2))))
        if(func.GetNDF()>0): s.DrawLatex(0.2,0.71,ROOT.Form("#chi^{2}/N_{DOF}: %.2f" % (func.GetChisquare()/func.GetNDF())))
        
        ROOT.gPad.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{foupdfname}")
    del cnv
    print("---------------32")
    
    
    cnv = ROOT.TCanvas("cnv_dipole_window","",2500,500)
    cnv.Divide(5,1)
    for idet,det in enumerate(cfg["detectors"]):
        cnv.cd(idet+1)
        ROOT.gPad.SetTicks(1,1)
        
        histos[f"h_response_zeroshrcls_x_sml_{det}"].SetMinimum(0)
        histos[f"h_response_zeroshrcls_x_sml_{det}"].SetMarkerStyle(24)
        histos[f"h_response_zeroshrcls_x_sml_{det}"].SetMarkerColor(ROOT.kRed)
        histos[f"h_response_zeroshrcls_x_sml_{det}"].SetLineColor(ROOT.kRed)
        histos[f"h_response_zeroshrcls_x_sml_{det}"].Draw("e1p")
        
        xmin = histos[f"h_response_zeroshrcls_x_sml_{det}"].GetXaxis().GetXmin()
        xmax = histos[f"h_response_zeroshrcls_x_sml_{det}"].GetXaxis().GetXmax()
        func = fit1(histos[f"h_response_zeroshrcls_x_sml_{det}"],ROOT.kRed,xmin,xmax)
        s = ROOT.TLatex()
        s.SetNDC(1)
        s.SetTextAlign(13)
        s.SetTextColor(ROOT.kBlack)
        s.SetTextFont(22)
        s.SetTextSize(0.045)
        s.DrawLatex(0.17,0.85,ROOT.Form("Mean: %.2f" % (func.GetParameter(1))))
        s.DrawLatex(0.17,0.78,ROOT.Form("Sigma: %.2f" % (func.GetParameter(2))))
        if(func.GetNDF()>0): s.DrawLatex(0.2,0.71,ROOT.Form("#chi^{2}/N_{DOF}: %.2f" % (func.GetChisquare()/func.GetNDF())))
        
        ROOT.gPad.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{foupdfname}")
    del cnv
    print("---------------33")
    
    
    cnv = ROOT.TCanvas("cnv_dipole_window","",1500,1000)
    cnv.Divide(3,2)
    for idet,det in enumerate(cfg["detectors"]):
        cnv.cd(idet+1)
        ROOT.gPad.SetTicks(1,1)

        histos[f"h_response_alowshrcls_x_ful_{det}"].SetMinimum(0)
        histos[f"h_response_zeroshrcls_x_ful_{det}"].SetMinimum(0)
        hbmax = histos[f"h_response_alowshrcls_x_ful_{det}"].GetMaximum()
        hamax = histos[f"h_response_zeroshrcls_x_ful_{det}"].GetMaximum()
        hmax = hbmax if(hbmax>hamax) else hamax
        hmax *= 1.2
        histos[f"h_response_alowshrcls_x_ful_{det}"].SetMaximum(hmax)
        histos[f"h_response_zeroshrcls_x_ful_{det}"].SetMaximum(hmax)
        
        histos[f"h_response_alowshrcls_x_ful_{det}"].SetMarkerStyle(20)
        histos[f"h_response_alowshrcls_x_ful_{det}"].SetMarkerColor(ROOT.kBlack)
        histos[f"h_response_alowshrcls_x_ful_{det}"].SetLineColor(ROOT.kBlack)
        histos[f"h_response_alowshrcls_x_ful_{det}"].Draw("ep")
        
        histos[f"h_response_zeroshrcls_x_ful_{det}"].SetMarkerStyle(24)
        histos[f"h_response_zeroshrcls_x_ful_{det}"].SetMarkerColor(ROOT.kRed)
        histos[f"h_response_zeroshrcls_x_ful_{det}"].SetLineColor(ROOT.kRed)
        histos[f"h_response_zeroshrcls_x_ful_{det}"].Draw("ep same")
        
        ROOT.gPad.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{foupdfname}")
    del cnv
    print("---------------34")
    
    
    cnv = ROOT.TCanvas("cnv_dipole_window","",1500,1000)
    cnv.Divide(3,2)
    for idet,det in enumerate(cfg["detectors"]):
        cnv.cd(idet+1)
        ROOT.gPad.SetTicks(1,1)
        
        histos[f"h_response_zeroshrcls_y_sml_{det}"].SetMinimum(0)
        histos[f"h_response_zeroshrcls_y_sml_{det}"].SetMarkerStyle(24)
        histos[f"h_response_zeroshrcls_y_sml_{det}"].SetMarkerColor(ROOT.kRed)
        histos[f"h_response_zeroshrcls_y_sml_{det}"].SetLineColor(ROOT.kRed)
        histos[f"h_response_zeroshrcls_y_sml_{det}"].Draw("e1p")
        
        xmin = histos[f"h_response_zeroshrcls_y_sml_{det}"].GetXaxis().GetXmin()
        xmax = histos[f"h_response_zeroshrcls_y_sml_{det}"].GetXaxis().GetXmax()
        func = fit1(histos[f"h_response_zeroshrcls_y_sml_{det}"],ROOT.kRed,xmin,xmax)
        s = ROOT.TLatex()
        s.SetNDC(1)
        s.SetTextAlign(13)
        s.SetTextColor(ROOT.kBlack)
        s.SetTextFont(22)
        s.SetTextSize(0.045)
        s.DrawLatex(0.17,0.85,ROOT.Form("Mean: %.2f" % (func.GetParameter(1))))
        s.DrawLatex(0.17,0.78,ROOT.Form("Sigma: %.2f" % (func.GetParameter(2))))
        if(func.GetNDF()>0): s.DrawLatex(0.2,0.71,ROOT.Form("#chi^{2}/N_{DOF}: %.2f" % (func.GetChisquare()/func.GetNDF())))
        
        ROOT.gPad.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{foupdfname}")
    del cnv
    print("---------------35")
    
    
    cnv = ROOT.TCanvas("cnv_dipole_window","",2500,500)
    cnv.Divide(5,1)
    for idet,det in enumerate(cfg["detectors"]):
        cnv.cd(idet+1)
        ROOT.gPad.SetTicks(1,1)
        
        histos[f"h_response_zeroshrcls_y_sml_{det}"].SetMinimum(0)
        histos[f"h_response_zeroshrcls_y_sml_{det}"].SetMarkerStyle(24)
        histos[f"h_response_zeroshrcls_y_sml_{det}"].SetMarkerColor(ROOT.kRed)
        histos[f"h_response_zeroshrcls_y_sml_{det}"].SetLineColor(ROOT.kRed)
        histos[f"h_response_zeroshrcls_y_sml_{det}"].Draw("e1p")
        
        xmin = histos[f"h_response_zeroshrcls_y_sml_{det}"].GetXaxis().GetXmin()
        xmax = histos[f"h_response_zeroshrcls_y_sml_{det}"].GetXaxis().GetXmax()
        func = fit1(histos[f"h_response_zeroshrcls_y_sml_{det}"],ROOT.kRed,xmin,xmax)
        s = ROOT.TLatex()
        s.SetNDC(1)
        s.SetTextAlign(13)
        s.SetTextColor(ROOT.kBlack)
        s.SetTextFont(22)
        s.SetTextSize(0.045)
        s.DrawLatex(0.17,0.85,ROOT.Form("Mean: %.2f" % (func.GetParameter(1))))
        s.DrawLatex(0.17,0.78,ROOT.Form("Sigma: %.2f" % (func.GetParameter(2))))
        if(func.GetNDF()>0): s.DrawLatex(0.2,0.71,ROOT.Form("#chi^{2}/N_{DOF}: %.2f" % (func.GetChisquare()/func.GetNDF())))
        
        ROOT.gPad.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{foupdfname}")
    del cnv
    print("---------------36")
    
    
    cnv = ROOT.TCanvas("cnv_dipole_window","",1500,1000)
    cnv.Divide(3,2)
    for idet,det in enumerate(cfg["detectors"]):
        cnv.cd(idet+1)
        ROOT.gPad.SetTicks(1,1)

        histos[f"h_response_alowshrcls_y_ful_{det}"].SetMinimum(0)
        histos[f"h_response_zeroshrcls_y_ful_{det}"].SetMinimum(0)
        hbmax = histos[f"h_response_alowshrcls_y_ful_{det}"].GetMaximum()
        hamax = histos[f"h_response_zeroshrcls_y_ful_{det}"].GetMaximum()
        hmax = hbmax if(hbmax>hamax) else hamax
        hmax *= 1.2
        histos[f"h_response_alowshrcls_y_ful_{det}"].SetMaximum(hmax)
        histos[f"h_response_zeroshrcls_y_ful_{det}"].SetMaximum(hmax)
        
        histos[f"h_response_alowshrcls_y_ful_{det}"].SetMarkerStyle(20)
        histos[f"h_response_alowshrcls_y_ful_{det}"].SetMarkerColor(ROOT.kBlack)
        histos[f"h_response_alowshrcls_y_ful_{det}"].SetLineColor(ROOT.kBlack)
        histos[f"h_response_alowshrcls_y_ful_{det}"].Draw("ep")
        
        histos[f"h_response_zeroshrcls_y_ful_{det}"].SetMarkerStyle(24)
        histos[f"h_response_zeroshrcls_y_ful_{det}"].SetMarkerColor(ROOT.kRed)
        histos[f"h_response_zeroshrcls_y_ful_{det}"].SetLineColor(ROOT.kRed)
        histos[f"h_response_zeroshrcls_y_ful_{det}"].Draw("ep same")
        
        ROOT.gPad.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{foupdfname}")
    del cnv
    print("---------------37")
    
    

    thetamin_x = cfg["seed_thetax_range_mid"][0]
    thetamax_x = cfg["seed_thetax_range_mid"][1]
    thetamin_y = cfg["seed_thetay_range_mid"][0]
    thetamax_y = cfg["seed_thetay_range_mid"][1]
    rhomin_x = cfg["seed_rhox_range_mid"][0]
    rhomax_x = cfg["seed_rhox_range_mid"][1]
    rhomin_y = cfg["seed_rhoy_range_mid"][0]
    rhomax_y = cfg["seed_rhoy_range_mid"][1]
    
    trxwin = ROOT.TPolyLine()
    trxwin.SetNextPoint(thetamin_x,rhomin_x)
    trxwin.SetNextPoint(thetamin_x,rhomax_x)
    trxwin.SetNextPoint(thetamax_x,rhomax_x)
    trxwin.SetNextPoint(thetamax_x,rhomin_x)
    trxwin.SetNextPoint(thetamin_x,rhomin_x)
    trxwin.SetLineColor(ROOT.kBlue)
    trxwin.SetLineWidth(1)

    trywin = ROOT.TPolyLine()
    trywin.SetNextPoint(thetamin_y,rhomin_y)
    trywin.SetNextPoint(thetamin_y,rhomax_y)
    trywin.SetNextPoint(thetamax_y,rhomax_y)
    trywin.SetNextPoint(thetamax_y,rhomin_y)
    trywin.SetNextPoint(thetamin_y,rhomin_y)
    trywin.SetLineColor(ROOT.kBlue)
    trywin.SetLineWidth(1)
    
    cnv = ROOT.TCanvas("cnv_dipole_window","",1000,1000)
    cnv.Divide(2,2)
    cnv.cd(1)
    ROOT.gPad.SetLogz()
    ROOT.gPad.SetTicks(1,1)
    histos["hWaves_zx"].Draw("colz")
    trxwin.Draw()
    ROOT.gPad.RedrawAxis()
    cnv.cd(2)
    ROOT.gPad.SetLogz()
    ROOT.gPad.SetTicks(1,1)
    histos["hWaves_zy"].Draw("colz")
    trywin.Draw()
    ROOT.gPad.RedrawAxis()
    cnv.cd(3)
    ROOT.gPad.SetLogz()
    ROOT.gPad.SetTicks(1,1)
    histos["hWaves_zx_intersections"].Draw("colz")
    trxwin.Draw()
    ROOT.gPad.RedrawAxis()
    cnv.cd(4)
    ROOT.gPad.SetLogz()
    ROOT.gPad.SetTicks(1,1)
    histos["hWaves_zy_intersections"].Draw("colz")
    trywin.Draw()
    ROOT.gPad.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{foupdfname}")
    del cnv
    print("---------------38")




    leg = ROOT.TLegend(0.2,0.7,0.55,0.8)
    leg.SetFillStyle(4000) # will be transparent
    leg.SetFillColor(0)
    leg.SetTextFont(42)
    leg.SetTextSize(0.037)
    leg.SetBorderSize(0)
    cnv = ROOT.TCanvas("cnv_dipole_window","",1500,1000)
    cnv.Divide(3,2)
    for idet,det in enumerate(cfg["detectors"]):
        cnv.cd(idet+1)
        ROOT.gPad.SetTicks(1,1)
        histos[f"h_tunnel_width_x_{det}"].SetMinimum(0)
        histos[f"h_tunnel_width_y_{det}"].SetMinimum(0)
        hbmax = histos[f"h_tunnel_width_x_{det}"].GetMaximum()
        hamax = histos[f"h_tunnel_width_y_{det}"].GetMaximum()
        hmax = hbmax if(hbmax>hamax) else hamax
        hmax *= 1.2
        histos[f"h_tunnel_width_x_{det}"].SetMaximum(hmax)
        histos[f"h_tunnel_width_y_{det}"].SetMaximum(hmax)
        histos[f"h_tunnel_width_x_{det}"].SetLineColor(ROOT.kBlack)
        histos[f"h_tunnel_width_x_{det}"].Draw("hist")
        histos[f"h_tunnel_width_y_{det}"].SetLineColor(ROOT.kRed)
        histos[f"h_tunnel_width_y_{det}"].Draw("hist same")
        if(idet==0):
            leg.AddEntry(histos[f"h_tunnel_width_x_{det}"],"k=x","l")
            leg.AddEntry(histos[f"h_tunnel_width_y_{det}"],"k=y","l")
        leg.Draw("same")
        
        ROOT.gPad.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{foupdfname})")
    del cnv
    print("---------------39")
    
    
    ### save as root file
    foutrootname = tfilenamein.replace(".root",f"_allplots.root")
    fout = ROOT.TFile(foutrootname,"RECREATE")
    fout.cd()
    for hname,hist in histos.items(): hist.Write()
    if(grxz is not None): grxz.Write()
    if(gryz is not None): gryz.Write()
    if(grpz is not None): grpz.Write()
    fout.Write()
    fout.Close()
    
    ########################
    ### write eudaq file ###
    ########################
    if(weudaqout):
        fEUDAQout.cd()
        tEUDAQout.Write()
        tEUDAQoutMeta.Write()
        fEUDAQout.Write()
        fEUDAQout.Close()
    ########################
    
    
    ### summary of tracking
    # print(f"\nTracks:{nacctrk}, GoodTriggers:{nevents-nbadtrigs}  (with AllTriggers:{nevents} and BadTriggers: {nbadtrigs})")
    print(f"\nAll tracks:{nalltrk}, Accepted tracks:{nacctrk}, Selected tracks:{nseltrk}, Butterfly tracks:{nbtrtrk}, GoodTriggers:{nevents-nbadtrigs_actual} Actual triggers: {ntrigs_actual} (with AllTriggers:{nevents} and BadTriggers in the range: {nbadtrigs_actual} (or {nbadtrigs} in the full run))")
    
    
    tracks_triggers_dict["all"]["pix"]["all"]  /= tracks_triggers_dict["all"]["trgs"]["all"]
    tracks_triggers_dict["all"]["cls"]["all"]  /= tracks_triggers_dict["all"]["trgs"]["all"]
    tracks_triggers_dict["all"]["pix"]["good"] /= tracks_triggers_dict["all"]["trgs"]["good"]
    tracks_triggers_dict["all"]["cls"]["good"] /= tracks_triggers_dict["all"]["trgs"]["good"]

    tracks_triggers_dict["even"]["pix"]["all"]  /= tracks_triggers_dict["even"]["trgs"]["all"]
    tracks_triggers_dict["even"]["cls"]["all"]  /= tracks_triggers_dict["even"]["trgs"]["all"]
    tracks_triggers_dict["even"]["pix"]["good"] /= tracks_triggers_dict["even"]["trgs"]["good"]
    tracks_triggers_dict["even"]["cls"]["good"] /= tracks_triggers_dict["even"]["trgs"]["good"]

    tracks_triggers_dict["odd"]["pix"]["all"]  /= tracks_triggers_dict["odd"]["trgs"]["all"]
    tracks_triggers_dict["odd"]["cls"]["all"]  /= tracks_triggers_dict["odd"]["trgs"]["all"]
    tracks_triggers_dict["odd"]["pix"]["good"] /= tracks_triggers_dict["odd"]["trgs"]["good"]
    tracks_triggers_dict["odd"]["cls"]["good"] /= tracks_triggers_dict["odd"]["trgs"]["good"]
    
    
    
    print(f"\ncounters before: {tracks_triggers_dict}")
    def convert_all_to_ints(d):
        if isinstance(d, dict):
            return {k: convert_all_to_ints(v) for k, v in d.items()}
        elif isinstance(d, float) or isinstance(d, int):
            return int(d)
        else:
            return d
    tracks_triggers_dict = convert_all_to_ints(tracks_triggers_dict)
    print(f"\ncounters: {tracks_triggers_dict}")
    
    
    # get the end time
    et = time.time()
    # get the execution time
    elapsed_time = et - st
    print(f'ֿֿ\nExecution time: {elapsed_time} seconds')

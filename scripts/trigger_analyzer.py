import ROOT
import math
import array
import numpy as np
import pickle
import ctypes
import time
import sys
import copy
import os

from scipy import linalg
from PIL import Image, ImageFilter
from scipy.optimize import minimize, differential_evolution
from scipy.optimize import NonlinearConstraint
import matplotlib
from matplotlib.colors import LogNorm
from matplotlib import rcParams
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from matplotlib.backends.backend_pdf import PdfPages
import scipy.io
from scipy import stats
from scipy.ndimage import gaussian_filter, median_filter
import glob
from pathlib import Path
import re



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

import argparse
parser = argparse.ArgumentParser(description='analyze_triggers.py...')
parser.add_argument('-conf', metavar='config file', required=True,  help='full path to config file')
parser.add_argument('-imin', metavar='first entry', required=False,  help='first entry')
parser.add_argument('-imax', metavar='last entry', required=False,  help='last entry')
parser.add_argument('-hits', metavar='plot hits?', required=False,  help='plot hits?')
argus = parser.parse_args()
configfile = argus.conf
fillhits = (int(argus.hits)==1) if(argus.hits is not None) else False


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tracker_lib import config
from tracker_lib import Pixels, hists, counters, hists, utils



### global
graphs = {}
histos = {}


####
# quads_scan = [0-7005, 7007-9178, 9190-12320, 12330-13150, 13160-18240, 18250-28880]
# quads_scan = [0-1000, 7100-8100, 9200-10200, 12325-13150, 15000-16000, 18300-19300] <<< actual runs
#                 0          1          2            3            4            5
# quad1_scan = [46.421,  44.98254, 40.425,     30.05477,    26.718,      29.86015]
# quad0_scan = [-30.677, -27.9938, -20.38,     -11.56,      -11.56,      -6.659]
# quad2_scan = [-30.6775,-27.994,  -20.3813,   -11.56075,   -3.371,      -6.659]
# m34_scan   = [1,       3,        10,         28,          30,          26]




def getLine(thr,xaxis):
    line = ROOT.TLine(xaxis[0],thr,xaxis[-1],thr)
    line.SetLineColor(ROOT.kViolet)
    line.SetLineStyle(2)
    return line

def trimArr(arr,direction,frac):
    srt_arr = np.sort(arr)
    n = len(arr)
    nTrim = int(frac*n)
    # print(f"nTrim for frac={frac} and n={n} is: {nTrim}")
    trim_arr = srt_arr[:-nTrim] if(direction=="up") else srt_arr[nTrim:]
    return trim_arr

def getThr(name,arr,direction,nsigma=3,frac=0.):
    arr0 = arr.copy()
    if(frac>0.): arr0 = trimArr(arr,direction,frac)
    avg = np.average(arr0)
    std = np.std(arr0)
    thr = 0
    if(direction=="down"): thr = avg - nsigma*std
    elif(direction=="up"): thr = avg + nsigma*std
    else:
        print("Error, direction can be down/up. got {direction}. Quitting.")
        quit()
    nremoved = 0
    for y in arr:
        if(direction=="up"   and y>thr): nremoved+=1
        if(direction=="down" and y<thr): nremoved+=1
    print(f"Threshold for {name} is: thr={thr:.3f} for avg={avg:.3f}, std={std:.5f} and nsigma={nsigma}. Triggers removed: {nremoved} ({(nremoved/len(arr))*100:.2f}%)")
    return thr


def add_graph(gname,x,y,col=ROOT.kBlack):
    graphs.update( {gname:ROOT.TGraph(len(x),x,y)} )
    graphs[gname].SetName(gname)
    graphs[gname].SetBit(ROOT.TGraph.kIsSortedX)
    graphs[gname].SetLineColor(col)
    print(f"{gname}: avg={np.mean(y)}, std={np.std(y)}")
    




### plot matplotlib
def plotter(name, pdf, arrx, arryL, arrsyR, xmin, xmax, xtitle, ytitleL="", ytitleR="", colL="blue", colR="red", logyL=True, logyR=True, lstyleL="-", lstyleR="-"):
    ### first page
    fig = plt.figure(figsize=(25, 5), constrained_layout=True)
    ax1 = plt.subplot(111)
    ax2 = ax1.twinx()
    
    ax1.plot(arrx, arryL, linewidth=0.7, color=colL,alpha=0.7, linestyle=lstyleL)
    for i,arry in enumerate(arrsyR):
        ax2.plot(arrx, arry, linewidth=0.7, color=colR, alpha=1-i*0.15, linestyle=lstyleR)
    
    plt.locator_params(axis='x', nbins=10)
    ax1.xaxis.set_minor_locator(AutoMinorLocator(10))
    ax1.yaxis.set_minor_locator(AutoMinorLocator(10))
    ax1.grid(which='major', linestyle='-', linewidth='0.25', color='gray',alpha=0.5)
    ax1.grid(which='minor', linestyle=':', linewidth='0.25', color='gray',alpha=0.3)
    
    ax1.set_xlim(xmin,xmax)
    ax1.set_xlabel(xtitle, fontsize=18)
    ax1.set_ylabel(ytitleL, fontsize=18, color=colL)
    if(logyL): ax1.set_yscale("log")
    if(len(arrsyR)>0):
        ax2.set_xlim(xmin,xmax)
        ax2.set_xlabel(xtitle, fontsize=18)
        ax2.set_ylabel(ytitleR, fontsize=18, color=colR)
        if(logyR): ax2.set_yscale("log")
    
    # Make tick labels bigger
    ax1.tick_params(axis='both', labelsize=16)
    ax2.tick_params(axis='both', labelsize=16)
    
    plt.title(name, fontsize=20, loc='center')
    
    # # Adjust bottom margin to fit bigger x-axis label
    # fig.subplots_adjust(bottom=0.2)  # increase if still cropped
    
    pdf.savefig(fig)
    plt.close(fig)








    



if __name__ == "__main__":
    
    #############################################
    ### Initialize Config in the main process ###
    config.init_config(configfile, False)
    cfg = config.Config().map
    config.show_config(cfg)
    #############################################
    
    
    ### see https://root.cern/manual/python
    print("---- start loading libs")
    if(os.uname()[1]=="wisett"):
        print("On DAQ PC (linux): must first add DetectorEvent lib:")
        print("export LD_LIBRARY_PATH=$HOME/work/eudaq/lib:$LD_LIBRARY_PATH")
        ROOT.gInterpreter.AddIncludePath('../eudaq/user/stave/module/inc/')
        ROOT.gInterpreter.AddIncludePath('../eudaq/user/stave/hardware/inc/')
        ROOT.gSystem.Load('libeudaq_det_event_dict.so')
    else:
        print("On mac: must first add DetectorEvent lib:")
        detevtlib = cfg["detevtlib"]
        print(f"export LD_LIBRARY_PATH=$PWD/DetectorEvent/{detevtlib}:$LD_LIBRARY_PATH")
        ROOT.gInterpreter.AddIncludePath(f"DetectorEvent/{detevtlib}/")
        ROOT.gSystem.Load('libtrk_event_dict.dylib')
    print("---- finish loading libs")
    
    
    ### make directories, copy the input file to the new basedir and return the path to it
    tfilenamein = utils.make_run_dirs(cfg["inputfile"])
    RunNum = utils.get_run_from_file(tfilenamein)
    
    tfile = ROOT.TFile(tfilenamein,"READ")
    ttree = tfile.Get("MyTree")
    nentries = ttree.GetEntries()
    print(f"Entries in tree: {nentries}")
    imin = int(argus.imin) if(argus.imin is not None) else 0
    imax = int(argus.imax) if(argus.imax is not None) else nentries
    nentries = imax-imin
    print(f"Reading from entry {imin} to {imax} --> corrected entries: {nentries}")
    
    detectors = ["ALPIDE_0","ALPIDE_1","ALPIDE_2","ALPIDE_3","ALPIDE_4"]
    detectorids = [8,6,4,2,0]
    detcol = [ROOT.kBlack, ROOT.kRed, ROOT.kBlue, ROOT.kGreen+2, ROOT.kOrange+1]
    
    hPixMatix = hists.GetPixMatrix()
    pix_x_nbins,pix_x_min,pix_x_max, pix_y_nbins,pix_y_min,pix_y_max = hists.GetPixelMatrixPars()
    for det in cfg["detectors"]:
        histos.update( { "h_pix_occ_2D_"+det : ROOT.TH2D("h_pix_occ_2D_"+det,";x;y;Hits",pix_x_nbins,pix_x_min,pix_x_max, pix_y_nbins,pix_y_min,pix_y_max) } )

    histos.update( { "h_ntrgs"  : ROOT.TH1D("h_ntrgs",";;Triggers",1,0,1) } )
    histos.update( { "h_m34"    : ROOT.TH1D("h_m34",  ";;M_{34} [m]",1,0,1) } )
    histos.update( { "h_q0act"  : ROOT.TH1D("h_q0act",";;Quad0 gradient [kG/m]",1,0,1) } )
    histos.update( { "h_q1act"  : ROOT.TH1D("h_q1act",";;Quad1 gradient [kG/m]",1,0,1) } )
    histos.update( { "h_q2act"  : ROOT.TH1D("h_q2act",";;Quad2 gradient [kG/m]",1,0,1) } )
    histos.update( { "h_time"   : ROOT.TH1D("h_time",";Processing time [ms];Triggers",500,0,50) } )
    histos.update( { "h_radmon" : ROOT.TH1D("h_radmon",";RadMon [mRem/h];Triggers",500,0,10) } )
    
    fltr_trgs = {}
    
    x_trg = np.zeros(nentries)
    x_ent = np.zeros(nentries)
    x_tim = []
    hits_vs_trg = {}
    hits_vs_ent = {}
    fltr_hits_vs_trg = {}
    fltr_hits_vs_ent = {}
    for det in detectors:
        hits_vs_trg.update({ det:np.zeros(nentries) })
        hits_vs_ent.update({ det:np.zeros(nentries) })
        fltr_hits_vs_trg.update({ det:np.zeros(nentries) })
        fltr_hits_vs_ent.update({ det:np.zeros(nentries) })
    
    y_dt          = np.zeros(nentries)
    y_yag         = np.zeros(nentries)
    y_dipole      = np.zeros(nentries)
    y_q0act       = np.zeros(nentries)
    y_q1act       = np.zeros(nentries)
    y_q2act       = np.zeros(nentries)
    y_m12         = np.zeros(nentries)
    y_m34         = np.zeros(nentries)
    y_z_img       = np.zeros(nentries)
    y_z_obj       = np.zeros(nentries)
    y_rad         = np.zeros(nentries)
    y_toro2040    = np.zeros(nentries)
    y_toro2452    = np.zeros(nentries)
    y_toro3163    = np.zeros(nentries)
    y_toro3255    = np.zeros(nentries)
    y_foilm1      = np.zeros(nentries)
    y_foilm2      = np.zeros(nentries)
    y_pmt3060     = np.zeros(nentries)
    y_pmt3070     = np.zeros(nentries)
    y_pmt3179     = np.zeros(nentries)
    y_pmt3350     = np.zeros(nentries)
    y_pmt3360     = np.zeros(nentries)
    y_bpm_pb_3156 = np.zeros(nentries)
    y_bpm_q0_3218 = np.zeros(nentries)
    y_bpm_q1_3265 = np.zeros(nentries)
    y_bpm_q2_3315 = np.zeros(nentries)
    
    ranges = []
    rng = []
    counter = 0
    for ientry,entry in enumerate(ttree):
        if(ientry<imin): continue
        if(ientry>=imax): break
    
        trgn   = entry.event.trg_n
        ts_bgn = entry.event.ts_begin
        ts_end = entry.event.ts_end
        dt     = (ts_end-ts_bgn)/1e6
        
        x_trg[counter] = trgn
        x_ent[counter] = ientry
        x_tim.append( utils.get_human_timestamp_ns(ts_bgn) )
        
        
        y_dt[counter]       = dt
        y_dipole[counter]   = entry.event.epics_frame.espec_dipole_bact
        y_q0act[counter]    = entry.event.epics_frame.espec_quad0_bact
        y_q1act[counter]    = entry.event.epics_frame.espec_quad1_bact
        y_q2act[counter]    = entry.event.epics_frame.espec_quad2_bact
        y_m12[counter]      = entry.event.epics_frame.mcalc_m12
        y_m34[counter]      = entry.event.epics_frame.mcalc_m34
        y_z_img[counter]    = entry.event.epics_frame.mcalc_z_im
        y_z_obj[counter]    = entry.event.epics_frame.mcalc_z_obj
        y_rad[counter]      = entry.event.epics_frame.radm_li20_1_ch01_meas
        y_toro2040[counter] = entry.event.epics_frame.toro_li20_2040_tmit_pc
        y_toro2452[counter] = entry.event.epics_frame.toro_li20_2452_tmit_pc
        y_toro3163[counter] = entry.event.epics_frame.toro_li20_3163_tmit_pc
        y_toro3255[counter] = entry.event.epics_frame.toro_li20_3255_tmit_pc
        y_foilm1[counter]   = entry.event.epics_frame.xps_li20_mc05_m1_rbv
        y_foilm2[counter]   = entry.event.epics_frame.xps_li20_mc05_m2_rbv
        y_yag[counter]      = entry.event.epics_frame.yag_hm_rbv
        y_pmt3060[counter]  = entry.event.epics_frame.pmt_s20_3060 
        y_pmt3070[counter]  = entry.event.epics_frame.pmt_s20_3070 
        y_pmt3179[counter]  = entry.event.epics_frame.pmt_s20_3179 
        y_pmt3350[counter]  = entry.event.epics_frame.pmt_s20_3350 
        y_pmt3360[counter]  = entry.event.epics_frame.pmt_s20_3360 
        y_bpm_pb_3156[counter] = entry.event.epics_frame.bpm_pb_3156_tmit 
        y_bpm_q0_3218[counter] = entry.event.epics_frame.bpm_quad0_3218_tmit 
        y_bpm_q1_3265[counter] = entry.event.epics_frame.bpm_quad1_3265_tmit 
        y_bpm_q2_3315[counter] = entry.event.epics_frame.bpm_quad2_3315_tmit 
    
        ### test!!!
        toro3163 = entry.event.epics_frame.toro_li20_3163_tmit_pc
        toro3255 = entry.event.epics_frame.toro_li20_3255_tmit_pc
        radmon   = entry.event.epics_frame.radm_li20_1_ch01_meas
        pmt3060  = entry.event.epics_frame.pmt_s20_3060
        pmt3070  = entry.event.epics_frame.pmt_s20_3070
        pmt3179  = entry.event.epics_frame.pmt_s20_3179
        pmt3350  = entry.event.epics_frame.pmt_s20_3350
        pmt3360  = entry.event.epics_frame.pmt_s20_3360
        bpm_q0_3218 = entry.event.epics_frame.bpm_quad0_3218_tmit
        bpm_q1_3265 = entry.event.epics_frame.bpm_quad1_3265_tmit
        bpm_q2_3315 = entry.event.epics_frame.bpm_quad2_3315_tmit
        
        histos["h_time"].Fill(dt)
        histos["h_radmon"].Fill(radmon)
        
        allhits = 0
        for ichip in range(entry.event.st_ev_buffer[0].ch_ev_buffer.size()):
            detid = entry.event.st_ev_buffer[0].ch_ev_buffer[ichip].chip_id
            detix = detectorids.index(detid)
            det   = detectors[detix]
            nhits = entry.event.st_ev_buffer[0].ch_ev_buffer[ichip].hits.size()
            hits_vs_trg[det][counter] = nhits
            hits_vs_ent[det][counter] = nhits
            
            fltr_hits_vs_trg[det][counter] = nhits
            fltr_hits_vs_ent[det][counter] = nhits
            
            allhits += nhits
            
            ###############
            ### 2D occupancy:
            if(fillhits):
                for ipix in range(nhits):
                    ix,iy = entry.event.st_ev_buffer[0].ch_ev_buffer[ichip].hits[ipix]
                    histos["h_pix_occ_2D_"+det].Fill(ix,iy)
    
        ### important!!
        counter += 1
    
    
    
    ####################################################################
    ####################################################################
    fmtpltlibname = tfilenamein.replace("tree_","beam_quality/tree_").replace(".root","_matplotloib_trigger_analysis_toroids.pdf")
    with PdfPages(fmtpltlibname) as pdf:
        xarr   = x_trg
        arryL  = hits_vs_trg["ALPIDE_0"]
        arrsyR = [y_toro2040, y_toro2452, y_toro3163, y_toro3255]
        xmin   = xarr[0]
        xmax   = xarr[-1]
        # plotter(name, pdf, arrx, arryL, arrsyR, xmin, xmax, xtitle, ytitleL="", ytitleR="", colL="blue", colR="red", logyL=True, logyR=True, lstyleL="-", lstyleR="-"):
        plotter(f'Toroids in Run {RunNum}', pdf, xarr, arryL, arrsyR, xmin, xmax, xtitle='EUDAQ BX number', ytitleL='Fired pixels in ALPIDE_0', ytitleR="Toroids [pC]", colL="black", colR="red", logyL=True, logyR=False)

    fmtpltlibname = tfilenamein.replace("tree_","beam_quality/tree_").replace(".root","_matplotloib_trigger_analysis_pmts.pdf")
    with PdfPages(fmtpltlibname) as pdf:
        xarr   = x_trg
        arryL  = hits_vs_trg["ALPIDE_0"]
        arrsyR = [y_pmt3060, y_pmt3070, y_pmt3179, y_pmt3350, y_pmt3360]
        xmin   = xarr[0]
        xmax   = xarr[-1]
        # plotter(name, pdf, arrx, arryL, arrsyR, xmin, xmax, xtitle, ytitleL="", ytitleR="", colL="blue", colR="red", logyL=True, logyR=True, lstyleL="-", lstyleR="-"):
        plotter(f'PMTs in Run {RunNum}', pdf, xarr, arryL, arrsyR, xmin, xmax, xtitle='EUDAQ BX number', ytitleL='Fired pixels in ALPIDE_0', ytitleR="PMTs [counts]", colL="black", colR="red", logyL=True, logyR=True)

    fmtpltlibname = tfilenamein.replace("tree_","beam_quality/tree_").replace(".root","_matplotloib_trigger_analysis_bpms.pdf")
    with PdfPages(fmtpltlibname) as pdf:
        xarr   = x_trg
        arryL  = hits_vs_trg["ALPIDE_0"]
        arrsyR = [y_bpm_pb_3156, y_bpm_q0_3218, y_bpm_q1_3265, y_bpm_q2_3315]
        xmin   = xarr[0]
        xmax   = xarr[-1]
        # plotter(name, pdf, arrx, arryL, arrsyR, xmin, xmax, xtitle, ytitleL="", ytitleR="", colL="blue", colR="red", logyL=True, logyR=True, lstyleL="-", lstyleR="-"):
        plotter(f'BPMs in Run {RunNum}', pdf, xarr, arryL, arrsyR, xmin, xmax, xtitle='EUDAQ BX number', ytitleL='Fired pixels in ALPIDE_0', ytitleR="BPMs [#electrons]", colL="black", colR="red", logyL=True, logyR=True)
    
    fmtpltlibname = tfilenamein.replace("tree_","beam_quality/tree_").replace(".root","_matplotloib_trigger_analysis_radmon.pdf")
    with PdfPages(fmtpltlibname) as pdf:
        xarr   = x_trg
        arryL  = hits_vs_trg["ALPIDE_0"]
        arrsyR = [y_rad]
        xmin   = xarr[0]
        xmax   = xarr[-1]
        # plotter(name, pdf, arrx, arryL, arrsyR, xmin, xmax, xtitle, ytitleL="", ytitleR="", colL="blue", colR="red", logyL=True, logyR=True, lstyleL="-", lstyleR="-"):
        plotter(f'RadMon in Run {RunNum}', pdf, xarr, arryL, arrsyR, xmin, xmax, xtitle='EUDAQ BX number', ytitleL='Fired pixels in ALPIDE_0', ytitleR="RadMon [mRem/h]", colL="black", colR="red", logyL=True, logyR=True)
    ####################################################################
    ####################################################################
    # quit()
    
    
    
    
    
    
    
    
    
    
    print(f"beginning:  {x_tim[0]}")
    print(f"end of run: {x_tim[-1]}")
    
    
    lines = {}
    
    ### GOOD FOR RUNS 502 (SHORT BERYLLIUM) AND 503 (LONG BACKGROUND)
    thr_toro2040 = getThr("toro2040",y_toro2040,direction="down",nsigma=1,frac=0.083)
    lines.update({"toro2040":getLine(thr_toro2040,x_trg)})
    thr_toro2452 = getThr("toro2452",y_toro2452,direction="down",nsigma=1,frac=0.083)
    lines.update({"toro2452":getLine(thr_toro2452,x_trg)})
    thr_toro3163 = getThr("toro3163",y_toro3163,direction="down",nsigma=1,frac=0.083)
    lines.update({"toro3163":getLine(thr_toro3163,x_trg)})
    thr_toro3255 = getThr("toro3255",y_toro3255,direction="down",nsigma=1,frac=0.083)
    lines.update({"toro3255":getLine(thr_toro3255,x_trg)})
    thr_pmt3060 = getThr("pmt3060",y_pmt3060,direction="up",nsigma=79.5,frac=0.01)
    lines.update({"pmt3060":getLine(thr_pmt3060,x_trg)})
    thr_pmt3070 = getThr("pmt3070",y_pmt3070,direction="up",nsigma=8.3,frac=0.01)
    lines.update({"pmt3070":getLine(thr_pmt3070,x_trg)})
    thr_pmt3179 = getThr("pmt3179",y_pmt3179,direction="up",nsigma=18,frac=0.01)
    lines.update({"pmt3179":getLine(thr_pmt3179,x_trg)})
    thr_pmt3350 = getThr("pmt3350",y_pmt3350,direction="up",nsigma=130,frac=0.01)
    lines.update({"pmt3350":getLine(thr_pmt3350,x_trg)})
    thr_pmt3360 = getThr("pmt3360",y_pmt3360,direction="up",nsigma=351.5,frac=0.01)
    lines.update({"pmt3360":getLine(thr_pmt3360,x_trg)})
    thr_rad = getThr("rad",y_rad,direction="up",nsigma=1,frac=0.01)
    lines.update({"rad":getLine(thr_rad,x_trg)})
    thr_bpm_pb_3156 = getThr("bpm pb 3156",y_bpm_pb_3156,direction="down",nsigma=1,frac=0.01)
    lines.update({"bpm_pb_3156":getLine(thr_bpm_pb_3156,x_trg)})
    thr_bpm_q0_3218 = getThr("bpm_q0_3218",y_bpm_q0_3218,direction="down",nsigma=1,frac=0.01)
    lines.update({"bpm_q0_3218":getLine(thr_bpm_q0_3218,x_trg)})
    thr_bpm_q1_3265 = getThr("bpm_q1_3265",y_bpm_q1_3265,direction="down",nsigma=1,frac=0.01)
    lines.update({"bpm_q1_3265":getLine(thr_bpm_q1_3265,x_trg)})
    thr_bpm_q2_3315 = getThr("bpm_q2_3315",y_bpm_q2_3315,direction="down",nsigma=1,frac=0.01)
    lines.update({"bpm_q2_3315":getLine(thr_bpm_q2_3315,x_trg)})
    
    
    # ### GOOD FOR RUNS 510 (LONG BERYLLIUM)
    # thr_toro2040 = getThr("toro2040",y_toro2040,direction="down",nsigma=5,frac=0.05)
    # lines.update({"toro2040":getLine(thr_toro2040,x_trg)})
    # thr_toro2452 = getThr("toro2452",y_toro2452,direction="down",nsigma=5,frac=0.05)
    # lines.update({"toro2452":getLine(thr_toro2452,x_trg)})
    # thr_toro3163 = getThr("toro3163",y_toro3163,direction="down",nsigma=5,frac=0.05)
    # lines.update({"toro3163":getLine(thr_toro3163,x_trg)})
    # thr_toro3255 = getThr("toro3255",y_toro3255,direction="down",nsigma=5,frac=0.05)
    # lines.update({"toro3255":getLine(thr_toro3255,x_trg)})
    # thr_pmt3060 = getThr("pmt3060",y_pmt3060,direction="up",nsigma=79.5,frac=0.01)
    # lines.update({"pmt3060":getLine(thr_pmt3060,x_trg)})
    # thr_pmt3070 = getThr("pmt3070",y_pmt3070,direction="up",nsigma=8.3,frac=0.2)
    # lines.update({"pmt3070":getLine(thr_pmt3070,x_trg)})
    # thr_pmt3179 = getThr("pmt3179",y_pmt3179,direction="up",nsigma=18,frac=0.01)
    # lines.update({"pmt3179":getLine(thr_pmt3179,x_trg)})
    # thr_pmt3350 = getThr("pmt3350",y_pmt3350,direction="up",nsigma=130,frac=0.01)
    # lines.update({"pmt3350":getLine(thr_pmt3350,x_trg)})
    # thr_pmt3360 = getThr("pmt3360",y_pmt3360,direction="up",nsigma=351.5,frac=0.01)
    # lines.update({"pmt3360":getLine(thr_pmt3360,x_trg)})
    # thr_rad = getThr("rad",y_rad,direction="up",nsigma=2,frac=0.2)
    # lines.update({"rad":getLine(thr_rad,x_trg)})
    # thr_bpm_pb_3156 = getThr("bpm pb 3156",y_bpm_pb_3156,direction="down",nsigma=10,frac=0.01)
    # lines.update({"bpm_pb_3156":getLine(thr_bpm_pb_3156,x_trg)})
    # thr_bpm_q0_3218 = getThr("bpm_q0_3218",y_bpm_q0_3218,direction="down",nsigma=15,frac=0.01)
    # lines.update({"bpm_q0_3218":getLine(thr_bpm_q0_3218,x_trg)})
    # thr_bpm_q1_3265 = getThr("bpm_q1_3265",y_bpm_q1_3265,direction="down",nsigma=15,frac=0.01)
    # lines.update({"bpm_q1_3265":getLine(thr_bpm_q1_3265,x_trg)})
    # thr_bpm_q2_3315 = getThr("bpm_q2_3315",y_bpm_q2_3315,direction="down",nsigma=15,frac=0.01)
    # lines.update({"bpm_q2_3315":getLine(thr_bpm_q2_3315,x_trg)})
    
    fltr_trgs = {}
    removed_triggers = []
    for i in range(len(x_trg)):
        fail = False
        if(not fail and y_toro2040[i]<thr_toro2040):       fail = True
        if(not fail and y_toro2452[i]<thr_toro2452):       fail = True
        if(not fail and y_toro3163[i]<thr_toro3163):       fail = True
        if(not fail and y_toro3255[i]<thr_toro3255):       fail = True
        if(not fail and y_pmt3060[i]>thr_pmt3060):         fail = True
        if(not fail and y_pmt3070[i]>thr_pmt3070):         fail = True
        if(not fail and y_pmt3179[i]>thr_pmt3179):         fail = True
        if(not fail and y_pmt3350[i]>thr_pmt3350):         fail = True
        if(not fail and y_pmt3360[i]>thr_pmt3360):         fail = True
        if(not fail and y_rad[i]>thr_rad):                 fail = True
        if(not fail and y_bpm_pb_3156[i]<thr_bpm_pb_3156): fail = True
        if(not fail and y_bpm_q0_3218[i]<thr_bpm_q0_3218): fail = True
        if(not fail and y_bpm_q1_3265[i]<thr_bpm_q1_3265): fail = True
        if(not fail and y_bpm_q2_3315[i]<thr_bpm_q2_3315): fail = True
        ##########################
        if(fail):
            trg = int(x_trg[i])
            if(trg not in removed_triggers): removed_triggers.append(trg)
            fltr_trgs.update({trg:i})
            for det in detectors:
                fltr_hits_vs_trg[det][i] = 0
                fltr_hits_vs_ent[det][i] = 0
        ##########################
    print(f'Removed triggers: {len(removed_triggers)}, which is {len(removed_triggers)/len(x_trg)*100:.2f}% of the total')
        
    
    ### check neighbour triggers and remove if failed
    final_fltr_trgs = removed_triggers.copy()
    for fltr_trg,counter in fltr_trgs.items():
        # for i in [-1,+1]:
        for i in [-1]:
            if(counter+i>=0 and counter+i<nentries):
                fltr_trg1 = int(x_trg[counter+i])
                for det in detectors:
                    fltr_hits_vs_trg[det][counter+i] = 0
                    fltr_hits_vs_ent[det][counter+i] = 0
                    if(fltr_trg1 not in final_fltr_trgs): final_fltr_trgs.append( fltr_trg1 )
    final_fltr_trgs = np.array(final_fltr_trgs).astype(int)
    # print(final_fltr_trgs)
    print(f'After neighbours removal, removed {len(final_fltr_trgs)}, which is {len(final_fltr_trgs)/len(x_trg)*100:.2f}% of the total')
    
    pklname = tfilenamein.replace("tree_","beam_quality/tree_").replace(".root","_BadTriggers.pkl")
    fpkl = open(pklname,"wb")
    pickle.dump(final_fltr_trgs, fpkl, protocol=pickle.HIGHEST_PROTOCOL)
    fpkl.close()
    
    
    histos["h_ntrgs"].SetBinContent(1,len(x_trg))
    histos["h_m34"].SetBinContent(1, np.average(y_m34))
    histos["h_q0act"].SetBinContent(1, np.average(y_q0act))
    histos["h_q1act"].SetBinContent(1, np.average(y_q1act))
    histos["h_q2act"].SetBinContent(1, np.average(y_q2act))
    
    
    maxhits = -1
    for det in detectors:
        maxhitsdet = max(hits_vs_trg[det])
        if(maxhitsdet>maxhits): maxhits = maxhitsdet
        add_graph(f"hits_vs_trg_{det}",x_trg,hits_vs_trg[det])
        add_graph(f"hits_vs_ent_{det}",x_trg,hits_vs_ent[det])
        add_graph(f"fltr_hits_vs_trg_{det}",x_trg,fltr_hits_vs_trg[det])
        add_graph(f"fltr_hits_vs_ent_{det}",x_trg,fltr_hits_vs_ent[det])
    
    
    add_graph("dt",x_trg,y_dt,ROOT.kBlack)
    add_graph("dipole",x_trg,y_dipole,ROOT.kBlack)
    add_graph("q0act",x_trg,y_q0act,ROOT.kBlack)
    add_graph("q1act",x_trg,y_q1act,ROOT.kRed)
    add_graph("q2act",x_trg,y_q2act,ROOT.kGreen+1)
    add_graph("m12",x_trg,y_m12,ROOT.kBlack)
    add_graph("m34",x_trg,y_m34,ROOT.kRed)
    add_graph("z_img",x_trg,y_z_img,ROOT.kBlack)
    add_graph("z_obj",x_trg,y_z_obj,ROOT.kRed)
    add_graph("rad",x_trg,y_rad,ROOT.kBlack)
    add_graph("foilm1",x_trg,y_foilm1,ROOT.kBlack)
    add_graph("foilm2",x_trg,y_foilm2,ROOT.kRed)
    add_graph("toro2040",x_trg,y_toro2040,ROOT.kBlack)
    add_graph("toro2452",x_trg,y_toro2452,ROOT.kRed)
    add_graph("toro3163",x_trg,y_toro3163,ROOT.kBlue)
    add_graph("toro3255",x_trg,y_toro3255,ROOT.kGreen+2)
    add_graph("yag",x_trg,y_yag,ROOT.kBlack)
    add_graph("pmt3060",x_trg,y_pmt3060,ROOT.kBlack)
    add_graph("pmt3070",x_trg,y_pmt3070,ROOT.kRed)
    add_graph("pmt3179",x_trg,y_pmt3179,ROOT.kBlue)
    add_graph("pmt3350",x_trg,y_pmt3350,ROOT.kGreen+2)
    add_graph("pmt3360",x_trg,y_pmt3360,ROOT.kOrange+1)
    add_graph("bpm_pb_3156",x_trg,y_bpm_pb_3156,ROOT.kBlack)
    add_graph("bpm_q0_3218",x_trg,y_bpm_q0_3218,ROOT.kBlack)
    add_graph("bpm_q1_3265",x_trg,y_bpm_q1_3265,ROOT.kRed)
    add_graph("bpm_q2_3315",x_trg,y_bpm_q2_3315,ROOT.kBlue)
    
    
    for i,det in enumerate(detectors):
        gname = f"hits_vs_ent_{det}"
        graphs[gname].SetTitle(f"{det}: fired pixels per tree entry;Tree entry;Fired pixels")
        graphs[gname].SetMaximum(maxhits*5)
        graphs[gname].SetMinimum(0.9)
        graphs[gname].GetXaxis().SetLimits(x_trg[0],x_trg[-1])
        graphs[gname].SetLineColor(detcol[i])
    for i,det in enumerate(detectors):
        gname = f"hits_vs_trg_{det}"
        graphs[gname].SetTitle(f"{det}: fired pixels per trigger;Trigger number;Fired pixels")
        graphs[gname].SetMaximum(maxhits*5)
        graphs[gname].SetMinimum(0.9)
        graphs[gname].GetXaxis().SetLimits(x_ent[0],x_ent[-1])
        graphs[gname].SetLineColor(detcol[i])
        
    for i,det in enumerate(detectors):
        gname = f"fltr_hits_vs_ent_{det}"
        graphs[gname].SetTitle(f"{det}: fired pixels per tree entry after filter;Tree entry;Fired pixels after filter")
        graphs[gname].SetMaximum(maxhits*5)
        graphs[gname].SetMinimum(0.9)
        graphs[gname].GetXaxis().SetLimits(x_trg[0],x_trg[-1])
        graphs[gname].SetLineColor(detcol[i])
    for i,det in enumerate(detectors):
        gname = f"fltr_hits_vs_trg_{det}"
        graphs[gname].SetTitle(f"{det}: fired pixels per trigger after filter;Trigger number;Fired pixels after filter")
        graphs[gname].SetMaximum(maxhits*5)
        graphs[gname].SetMinimum(0.9)
        graphs[gname].GetXaxis().SetLimits(x_ent[0],x_ent[-1])
        graphs[gname].SetLineColor(detcol[i])
    
    
    
    ################################################################
    ################################################################
    ################################################################
    
    ftrgname = tfilenamein.replace("tree_","beam_quality/tree_").replace(".root","_trigger_analysis.pdf")
    fhitname = tfilenamein.replace("tree_","beam_quality/tree_").replace(".root","_trigger_analysis_hits.pdf")
    tfo = ROOT.TFile(ftrgname.replace(".pdf",".root"),"RECREATE")
    if not tfo or tfo.IsZombie():
        print("Error: Could not open output file")
        exit()
    tfo.cd()
    for gname,graph in graphs.items(): graph.Write()
    for hname,histo in histos.items(): histo.Write()
    tfo.Write()
    tfo.Close()
    
    
    cnv = ROOT.TCanvas("c1","",1200,500)
    cnv.SetTicks(1,1)
    cnv.SetLogy()
    leg = ROOT.TLegend(0.4,0.2,0.7,0.5)
    leg.SetFillStyle(4000) # will be transparent
    leg.SetFillColor(0)
    leg.SetTextFont(42)
    leg.SetBorderSize(0)
    mg = ROOT.TMultiGraph()
    for i,det in enumerate(detectors):
        gname = f"hits_vs_trg_{det}"
        leg.AddEntry(graphs[gname],f"{det}","l")
        mg.Add(graphs[gname])
    mg.Draw("al")
    leg.Draw("same")
    mg.SetTitle(f";Trigger number;Fired pixels")
    mg.SetMaximum(maxhits*5)
    mg.SetMinimum(0.9)
    mg.GetXaxis().SetLimits(x_trg[0],x_trg[-1])
    cnv.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{ftrgname}(")
    
    
    cnv = ROOT.TCanvas("c2","",1200,500)
    cnv.SetTicks(1,1)
    cnv.SetLogy()
    leg = ROOT.TLegend(0.4,0.2,0.7,0.5)
    leg.SetFillStyle(4000) # will be transparent
    leg.SetFillColor(0)
    leg.SetTextFont(42)
    leg.SetBorderSize(0)
    mg = ROOT.TMultiGraph()
    for i,det in enumerate(detectors):
        gname = f"fltr_hits_vs_trg_{det}"
        leg.AddEntry(graphs[gname],f"{det}","l")
        mg.Add(graphs[gname])
    mg.Draw("al")
    leg.Draw("same")
    mg.SetTitle(f";Trigger number;Fired pixels after filter")
    mg.SetMaximum(maxhits*5)
    mg.SetMinimum(0.9)
    mg.GetXaxis().SetLimits(x_trg[0],x_trg[-1])
    cnv.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{ftrgname}")
    
    cnv = ROOT.TCanvas("c3","",1200,500)
    cnv.SetTicks(1,1)
    mg = ROOT.TMultiGraph()
    mg.Add(graphs["dipole"])
    mg.SetMinimum(0)
    mg.SetMaximum(13.1)
    mg.Draw("al")
    mg.SetTitle(f";Trigger number;Dipole [GeV] (for a 6 mrad deflection at 10 GeV)")
    mg.GetXaxis().SetLimits(x_trg[0],x_trg[-1])
    cnv.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{ftrgname}")
    
    cnv = ROOT.TCanvas("c4","",1200,500)
    cnv.SetTicks(1,1)
    # cnv.SetLogy()
    leg = ROOT.TLegend(0.4,0.2,0.7,0.5)
    leg.SetFillStyle(4000) # will be transparent
    leg.SetFillColor(0)
    leg.SetTextFont(42)
    leg.SetBorderSize(0)
    mg = ROOT.TMultiGraph()
    for q in range(3):
        gname = f"q{q}act"
        leg.AddEntry(graphs[gname],f"Quad{q}","l")
        mg.Add(graphs[gname])
    mg.Draw("al")
    leg.Draw("same")
    mg.SetTitle(f";Trigger number;Quad [kG/m]")
    mg.GetXaxis().SetLimits(x_trg[0],x_trg[-1])
    cnv.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{ftrgname}")
    
    cnv = ROOT.TCanvas("c5","",1200,500)
    cnv.SetTicks(1,1)
    # cnv.SetLogy()
    leg = ROOT.TLegend(0.4,0.2,0.7,0.5)
    leg.SetFillStyle(4000) # will be transparent
    leg.SetFillColor(0)
    leg.SetTextFont(42)
    leg.SetBorderSize(0)
    mg = ROOT.TMultiGraph()
    for mij in ["12","34"]:
        gname = f"m{mij}"
        leg.AddEntry(graphs[gname],f"M{mij}","l")
        mg.Add(graphs[gname])
    mg.Draw("al")
    leg.Draw("same")
    mg.SetTitle(";Trigger number;M_{ij} [m]")
    mg.GetXaxis().SetLimits(x_trg[0],x_trg[-1])
    cnv.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{ftrgname}")
    
    
    cnv = ROOT.TCanvas("c5.1","",1200,500)
    cnv.SetTicks(1,1)
    # cnv.SetLogy()
    leg = ROOT.TLegend(0.4,0.2,0.7,0.5)
    leg.SetFillStyle(4000) # will be transparent
    leg.SetFillColor(0)
    leg.SetTextFont(42)
    leg.SetBorderSize(0)
    mg = ROOT.TMultiGraph()
    for k in ["img","obj"]:
        gname = f"z_{k}"
        leg.AddEntry(graphs[gname],f"z_{k}","l")
        mg.Add(graphs[gname])
    mg.Draw("al")
    leg.Draw("same")
    mg.SetTitle(";Trigger number;z_img/obj [m]")
    mg.GetXaxis().SetLimits(x_trg[0],x_trg[-1])
    cnv.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{ftrgname}")
    
    
    cnv = ROOT.TCanvas("c6","",1200,500)
    cnv.SetTicks(1,1)
    # cnv.SetLogy()
    leg = ROOT.TLegend(0.4,0.2,0.7,0.5)
    leg.SetFillStyle(4000) # will be transparent
    leg.SetFillColor(0)
    leg.SetTextFont(42)
    leg.SetBorderSize(0)
    mg = ROOT.TMultiGraph()
    for t in ["2040","2452","3163","3255"]:
        gname = f"toro{t}"
        leg.AddEntry(graphs[gname],f"Toroid {t}","l")
        mg.Add(graphs[gname])
    mg.Draw("al")
    leg.Draw("same")
    for t in ["2040","2452","3163","3255"]:
        gname = f"toro{t}"
        lines[gname].Draw("same")
    mg.SetTitle(";Trigger number;Toroid charge [pC]")
    mg.GetXaxis().SetLimits(x_trg[0],x_trg[-1])
    # mg.SetMinimum(800)
    cnv.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{ftrgname}")
    
    
    cnv = ROOT.TCanvas("c7","",1200,500)
    cnv.SetTicks(1,1)
    cnv.SetLogy()
    leg = ROOT.TLegend(0.4,0.2,0.7,0.5)
    leg.SetFillStyle(4000) # will be transparent
    leg.SetFillColor(0)
    leg.SetTextFont(42)
    leg.SetBorderSize(0)
    mg = ROOT.TMultiGraph()
    for t in ["3060","3070","3179","3350","3360"]:
        gname = f"pmt{t}"
        leg.AddEntry(graphs[gname],f"PMT {t}","l")
        mg.Add(graphs[gname])
    mg.Draw("al")
    leg.Draw("same")
    for t in ["3060","3070","3179","3350","3360"]:
        gname = f"pmt{t}"
        lines[gname].Draw("same")
    mg.SetTitle(";Trigger number;PMT [counts]")
    mg.GetXaxis().SetLimits(x_trg[0],x_trg[-1])
    cnv.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{ftrgname}")
    
    
    # cnv = ROOT.TCanvas("c8","",1200,500)
    # cnv.SetTicks(1,1)
    # # cnv.SetLogy()
    # leg = ROOT.TLegend(0.4,0.2,0.7,0.5)
    # leg.SetFillStyle(4000) # will be transparent
    # leg.SetFillColor(0)
    # leg.SetTextFont(42)
    # leg.SetBorderSize(0)
    # mg = ROOT.TMultiGraph()
    # for m in ["m1","m2"]:
    #     gname = f"foil{m}"
    #     leg.AddEntry(graphs[gname],f"Foil {m}","l")
    #     mg.Add(graphs[gname])
    # mg.Draw("al")
    # leg.Draw("same")
    # mg.SetTitle(";Trigger number;Foil position [mm]")
    # mg.GetXaxis().SetLimits(x_trg[0],x_trg[-1])
    # cnv.RedrawAxis()
    # cnv.Update()
    # cnv.SaveAs(f"{ftrgname}")
    
    cnv = ROOT.TCanvas("c9","",1200,500)
    cnv.SetTicks(1,1)
    cnv.SetLogy()
    graphs["rad"].Draw("al")
    graphs["rad"].SetTitle(";Trigger number;RadMon [mRem/h]")
    graphs["rad"].GetXaxis().SetLimits(x_trg[0],x_trg[-1])
    # graphs["rad"].SetMaximum(10)
    lines["rad"].Draw("same")
    cnv.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{ftrgname}")
    
    cnv = ROOT.TCanvas("c10","",1200,500)
    cnv.SetTicks(1,1)
    cnv.SetLogy()
    histos["h_radmon"].Draw("hist")
    cnv.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{ftrgname}")
    
    
    cnv = ROOT.TCanvas("c11","",1200,500)
    cnv.SetTicks(1,1)
    cnv.SetLogy()
    graphs["bpm_pb_3156"].Draw("al")
    graphs["bpm_pb_3156"].SetTitle(";Trigger number;BPM PB [#electrons]")
    graphs["bpm_pb_3156"].GetXaxis().SetLimits(x_trg[0],x_trg[-1])
    lines["bpm_pb_3156"].Draw("same")
    cnv.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{ftrgname}")
    
    cnv = ROOT.TCanvas("c12","",1200,500)
    cnv.SetTicks(1,1)
    cnv.SetLogy()
    leg = ROOT.TLegend(0.4,0.2,0.7,0.5)
    leg.SetFillStyle(4000) # will be transparent
    leg.SetFillColor(0)
    leg.SetTextFont(42)
    leg.SetBorderSize(0)
    mg = ROOT.TMultiGraph()
    for t in ["q0_3218","q1_3265","q2_3315"]:
        gname = f"bpm_{t}"
        leg.AddEntry(graphs[gname],f"Quad {t}","l")
        mg.Add(graphs[gname])
    mg.Draw("al")
    leg.Draw("same")
    for t in ["q0_3218","q1_3265","q2_3315"]:
        gname = f"bpm_{t}"
        lines[gname].Draw("same")
    mg.SetTitle(";Trigger number;BPM quads [#electrons]")
    mg.GetXaxis().SetLimits(x_trg[0],x_trg[-1])
    cnv.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{ftrgname}")
    
    
    cnv = ROOT.TCanvas("c13","",1200,500)
    cnv.SetTicks(1,1)
    # cnv.SetLogy()
    graphs["foilm2"].Draw("al")
    graphs["foilm2"].SetTitle(";Trigger number;Foil x [mm]")
    graphs["foilm2"].GetXaxis().SetLimits(x_trg[0],x_trg[-1])
    cnv.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{ftrgname}")
    
    cnv = ROOT.TCanvas("c14","",1200,500)
    cnv.SetTicks(1,1)
    # cnv.SetLogy()
    graphs["dt"].Draw("al")
    graphs["dt"].SetTitle(";Trigger number;Processing time [ms]")
    graphs["dt"].GetXaxis().SetLimits(x_trg[0],x_trg[-1])
    cnv.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{ftrgname}")
    
    cnv = ROOT.TCanvas("c15","",1200,500)
    cnv.SetTicks(1,1)
    cnv.SetLogy()
    histos["h_time"].Draw("hist")
    cnv.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{ftrgname})")
    
    
    cnv = ROOT.TCanvas("cnv","",800,2200)
    cnv.Divide(1,5)
    for idet,det in enumerate(cfg["detectors"]):
        cnv.cd(idet+1)
        ROOT.gPad.SetTicks(1,1)
        histos[f"h_pix_occ_2D_{det}"].SetTitle(f"{det};x [pixels];y [pixels];Pixels/Trigger")
        # for i in range(3): histos[f"h_pix_occ_2D_{det}"].Smooth()
        histos[f"h_pix_occ_2D_{det}"].Scale(1./len(x_trg))
        histos[f"h_pix_occ_2D_{det}"].Draw("colz")
        ROOT.gPad.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{fhitname}")
    
    for det in cfg["detectors"]:
        for i,y in enumerate(fltr_hits_vs_trg[det]):
            if(y>0):
                print(f"First trigger for {det} is: {x_trg[i]}")
                break
        
    
    



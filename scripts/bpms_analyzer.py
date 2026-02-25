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
ROOT.gStyle.SetPadBottomMargin(0.17)
ROOT.gStyle.SetPadLeftMargin(0.13)
ROOT.gStyle.SetPadRightMargin(0.16)

import argparse
parser = argparse.ArgumentParser(description='analyze_triggers.py...')
parser.add_argument('-conf', metavar='config file', required=True,  help='full path to config file')
parser.add_argument('-imin', metavar='first entry', required=False, help='first entry')
parser.add_argument('-imax', metavar='last entry', required=False,  help='last entry')
argus = parser.parse_args()
configfile = argus.conf


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tracker_lib import config
from tracker_lib import Pixels, hists, counters, hists, utils

### global
histos = {}
electrons_in_1nC = 6.2415e6


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
    fpkltrgname = tfilenamein.replace("tree_","beam_quality/tree_").replace(".root","_BadTriggers.pkl") if(not cfg["isMC"] and cfg["runtype"]=="beam") else None
    RunNum = utils.get_run_from_file(tfilenamein)
    
    
    ### load bad triggers from pickle
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
    
    
    tfile = ROOT.TFile(tfilenamein,"READ")
    ttree = tfile.Get("MyTree")
    nentries = ttree.GetEntries()
    print(f"Entries in tree: {nentries}")
    imin = int(argus.imin) if(argus.imin is not None) else 0
    imax = int(argus.imax) if(argus.imax is not None) else nentries
    nentries = imax-imin
    print(f"Reading from entry {imin} to {imax} --> corrected entries: {nentries}")
    
    histos.update( { "h_xy_quad0_3218"  : ROOT.TH2D("h_xy_quad0_3218","Quad_{0} BPM 3218;x [mm];y [mm];Charge per trigger [pC]",100,-1,+1,100,-1,+1) } )
    histos.update( { "h_xy_quad1_3265"  : ROOT.TH2D("h_xy_quad1_3265","Quad_{1} BPM 3265;x [mm];y [mm];Charge per trigger [pC]",100,-1,+1,100,-1,+1) } )
    histos.update( { "h_xy_quad2_3315"  : ROOT.TH2D("h_xy_quad2_3315","Quad_{2} BPM 3315;x [mm];y [mm];Charge per trigger [pC]",100,-1,+1,100,-1,+1) } )
    
    ranges = []
    rng = []
    ntrg = 0
    for ientry,entry in enumerate(ttree):
        if(ientry<imin): continue
        if(ientry>=imax): break
    
        trgn   = entry.event.trg_n
        ts_bgn = entry.event.ts_begin
        ts_end = entry.event.ts_end
        dt     = (ts_end-ts_bgn)/1e6
        
        if((cfg["runtype"]=="beam" and cfg["checkbadtriggers"]) and int(trgn) in badtriggers):
            print(f"Skipping bad trigger: {trgn}")
            continue
        
        ntrg += 1
        
        bpm_q0_3218_tmit = entry.event.epics_frame.bpm_quad0_3218_tmit
        bpm_q1_3265_tmit = entry.event.epics_frame.bpm_quad1_3265_tmit
        bpm_q2_3315_tmit = entry.event.epics_frame.bpm_quad2_3315_tmit
        
        bpm_q0_3218_x = entry.event.epics_frame.bpm_quad0_3218_x
        bpm_q1_3265_x = entry.event.epics_frame.bpm_quad1_3265_x
        bpm_q2_3315_x = entry.event.epics_frame.bpm_quad2_3315_x
        
        bpm_q0_3218_y = entry.event.epics_frame.bpm_quad0_3218_y
        bpm_q1_3265_y = entry.event.epics_frame.bpm_quad1_3265_y
        bpm_q2_3315_y = entry.event.epics_frame.bpm_quad2_3315_y
        
        histos["h_xy_quad0_3218"].Fill(bpm_q0_3218_x,bpm_q0_3218_y, bpm_q0_3218_tmit)
        histos["h_xy_quad1_3265"].Fill(bpm_q1_3265_x,bpm_q1_3265_y, bpm_q1_3265_tmit)
        histos["h_xy_quad2_3315"].Fill(bpm_q2_3315_x,bpm_q2_3315_y, bpm_q2_3315_tmit)

    
    ### normalize to triggers:
    for hname,histo in histos.items():
        histo.Scale(1./float(ntrg))
        histo.Scale(1./electrons_in_1nC)
    
    ftrgname = tfilenamein.replace("tree_","beam_quality/tree_").replace(".root","_bpms_analysis.pdf")
    tfo = ROOT.TFile(ftrgname.replace(".pdf",".root"),"RECREATE")
    if not tfo or tfo.IsZombie():
        print("Error: Could not open output file")
        exit()
    tfo.cd()
    for hname,histo in histos.items(): histo.Write()
    tfo.Write()
    tfo.Close()
    
    
    cnv = ROOT.TCanvas("c1","",1800,500)
    cnv.Divide(3,1)
    cnv.cd(1)
    ROOT.gPad.SetTicks(1,1)
    histos["h_xy_quad0_3218"].Draw("colz")
    s = ROOT.TLatex()
    s.SetNDC(1)
    s.SetTextAlign(13)
    s.SetTextColor(ROOT.kBlack)
    s.SetTextFont(22)
    s.SetTextSize(0.045)
    s.DrawLatex(0.17,0.88,f"Run {RunNum}")
    s = ROOT.TLatex()
    s.SetNDC(1)
    s.SetTextAlign(13)
    s.SetTextColor(ROOT.kBlack)
    s.SetTextFont(132)
    s.SetTextSize(0.045)
    s.DrawLatex(0.17,0.83,f'#mu_{{x}}={histos["h_xy_quad0_3218"].GetMean(1):.3f} mm, #sigma_{{x}}={histos["h_xy_quad0_3218"].GetStdDev(1):.3f} mm')
    s.DrawLatex(0.17,0.78,f'#mu_{{y}}={histos["h_xy_quad0_3218"].GetMean(2):.3f} mm, #sigma_{{y}}={histos["h_xy_quad0_3218"].GetStdDev(2):.3f} mm')
    s.DrawLatex(0.17,0.73,f'Q={histos["h_xy_quad0_3218"].Integral():.1f} pC')
    ROOT.gPad.RedrawAxis()
    
    cnv.cd(2)
    ROOT.gPad.SetTicks(1,1)
    histos["h_xy_quad1_3265"].Draw("colz")
    s = ROOT.TLatex()
    s.SetNDC(1)
    s.SetTextAlign(13)
    s.SetTextColor(ROOT.kBlack)
    s.SetTextFont(22)
    s.SetTextSize(0.045)
    s.DrawLatex(0.17,0.88,f"Run {RunNum}")
    s = ROOT.TLatex()
    s.SetNDC(1)
    s.SetTextAlign(13)
    s.SetTextColor(ROOT.kBlack)
    s.SetTextFont(132)
    s.SetTextSize(0.045)
    s.DrawLatex(0.17,0.83,f'#mu_{{x}}={histos["h_xy_quad1_3265"].GetMean(1):.3f} mm, #sigma_{{x}}={histos["h_xy_quad1_3265"].GetStdDev(1):.3f} mm')
    s.DrawLatex(0.17,0.78,f'#mu_{{y}}={histos["h_xy_quad1_3265"].GetMean(2):.3f} mm, #sigma_{{y}}={histos["h_xy_quad1_3265"].GetStdDev(2):.3f} mm')
    s.DrawLatex(0.17,0.73,f'Q={histos["h_xy_quad1_3265"].Integral():.1f} pC')
    ROOT.gPad.RedrawAxis()
    
    cnv.cd(3)
    ROOT.gPad.SetTicks(1,1)
    histos["h_xy_quad2_3315"].Draw("colz")
    s = ROOT.TLatex()
    s.SetNDC(1)
    s.SetTextAlign(13)
    s.SetTextColor(ROOT.kBlack)
    s.SetTextFont(22)
    s.SetTextSize(0.045)
    s.DrawLatex(0.17,0.88,f"Run {RunNum}")
    s = ROOT.TLatex()
    s.SetNDC(1)
    s.SetTextAlign(13)
    s.SetTextColor(ROOT.kBlack)
    s.SetTextFont(132)
    s.SetTextSize(0.045)
    s.DrawLatex(0.17,0.83,f'#mu_{{x}}={histos["h_xy_quad2_3315"].GetMean(1):.3f} mm, #sigma_{{x}}={histos["h_xy_quad2_3315"].GetStdDev(1):.3f} mm')
    s.DrawLatex(0.17,0.78,f'#mu_{{y}}={histos["h_xy_quad2_3315"].GetMean(2):.3f} mm, #sigma_{{y}}={histos["h_xy_quad2_3315"].GetStdDev(2):.3f} mm')
    s.DrawLatex(0.17,0.73,f'Q={histos["h_xy_quad2_3315"].Integral():.1f} pC')
    ROOT.gPad.RedrawAxis()
    
    cnv.Update()
    cnv.SaveAs(f"{ftrgname}")
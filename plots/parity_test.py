#!/usr/bin/python
import multiprocessing as mp
import time
import datetime
import sys
import os
import os.path
import math
import subprocess
import array
import numpy as np
import ROOT
### Stop ROOT from taking ownership of histograms automatically
ROOT.TH1.AddDirectory(ROOT.kFALSE)

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







def h1h2max(h1,h2):
    hmax = -1
    y1 = h1.GetMaximum()
    y2 = h2.GetMaximum()
    hmax = y1 if(y1>y2) else y2
    return hmax

def get_bkg_histogram(fname,hname):
    f = ROOT.TFile.Open(fname)
    htot  = f.Get(f"{hname}_even")
    htot.Reset()
    heven = f.Get(f"{hname}_even")
    hodd  = f.Get(f"{hname}_odd")
    htot.Add(heven)
    heven.Add(hodd)
    htot.SetLineColor(ROOT.kBlue)
    htot.SetLineStyle(2)
    htot.SetLineWidth(2)
    return htot

def plot_triggers(runs,basepath,runpath,fprfx,fsufx,hname,pdfname,title):
    fname = f"{basepath}/{runpath}{runs[0]}/{fprfx}{runs[0]}{fsufx}"
    f = ROOT.TFile.Open(fname)
    h = f.Get(f"{hname}")
    h.Reset()
    h.SetLineColor(ROOT.kBlack)
    h.SetLineWidth(2)
    h.SetTitle(title)
    
    for run in runs:
        frname = f"{basepath}/{runpath}{run}/{fprfx}{run}{fsufx}"
        fr = ROOT.TFile.Open(frname)
        hr = fr.Get(f"{hname}")
        h.Add(hr)
    
    h.SetMinimum(0.8)
    
    c = ROOT.TCanvas("c", "Comparison", 800, 600)
    c.SetTicks(1,1)
    c.SetLogy()
    h.Draw("TEXT0")

    c.RedrawAxis()
    c.Update()
    c.Draw()
    c.SaveAs(pdfname)

def get_integral(runs,basepath,runpath,fprfx,fsufx,hname):
    sigint_even = 0
    sigint_odd  = 0
    for run in runs:
        frname = f"{basepath}/{runpath}{run}/{fprfx}{run}{fsufx}"
        fr = ROOT.TFile.Open(frname)
        hrEven = fr.Get(f"{hname}_even")
        hrOdd  = fr.Get(f"{hname}_odd")
        sigint_even += hrEven.Integral()
        sigint_odd  += hrOdd.Integral()
    print(f"sigint_even={sigint_even}, sigint_odd={sigint_odd}")
    return sigint_even,sigint_odd


def plot_parity(runs,basepath,runpath,fprfx,fsufx,hname,pdfname,title,rebin,norm=True,hbkg=None):
    fname = f"{basepath}/{runpath}{runs[0]}/{fprfx}{runs[0]}{fsufx}"
    f = ROOT.TFile.Open(fname)
    hEven = f.Get(f"{hname}_even")
    hOdd  = f.Get(f"{hname}_odd")
    hEven.Reset()
    hOdd.Reset()
    hEven.SetLineColor(ROOT.kBlack)
    hOdd.SetLineColor(ROOT.kGreen+2)
    hEven.SetLineWidth(2)
    hOdd.SetLineWidth(2)
    if(rebin>1): hEven.Rebin(rebin)
    if(rebin>1): hOdd.Rebin(rebin)
    hEven.SetTitle(title)
    hOdd.SetTitle(title)
    
    for run in runs:
        frname = f"{basepath}/{runpath}{run}/{fprfx}{run}{fsufx}"
        fr = ROOT.TFile.Open(frname)
        hrEven = fr.Get(f"{hname}_even")
        hrOdd  = fr.Get(f"{hname}_odd")
        if(rebin>1): hrEven.Rebin(rebin)
        if(rebin>1): hrOdd.Rebin(rebin)
        hEven.Add(hrEven)
        hOdd.Add(hrOdd)
    
    c = ROOT.TCanvas("c", "Comparison", 800, 600)
    c.SetTicks(1,1)
    
    int_even = hEven.Integral()
    int_odd  = hOdd.Integral()
    hmax_even = hEven.GetMaximum()
    hmax_odd  = hOdd.GetMaximum()
    
    hmax = h1h2max(hEven,hOdd)
    hEven.SetMaximum(hmax*1.1)
    hOdd.SetMaximum(hmax*1.1)
    
    if(norm):
        hOdd.GetYaxis().SetTitle("Normalized")
        hEven.GetYaxis().SetTitle("Normalized")
        hOdd.DrawNormalized("HIST")
        hEven.DrawNormalized("HIST SAME")
    else:
        hOdd.Draw("HIST")
        hEven.Draw("HIST SAME")
    if(hbkg is not None):
        if(rebin>1): hbkg.Rebin(rebin)
        hbkg.SetLineWidth(2)
        hbkg.SetLineColor(ROOT.kRed)
        if(norm):
            hbkg.GetYaxis().SetTitle("Normalized")
            hbkg.DrawNormalized("HIST SAME")
        else:
            hbkg.Draw("HIST SAME")
        
        
    s = ROOT.TLatex()
    s.SetNDC(1)
    s.SetTextAlign(13)
    s.SetTextColor(ROOT.kBlack)
    s.SetTextFont(132)
    s.SetTextSize(0.045)
    s.DrawLatex(0.5,0.60,f"Even #mu={hEven.GetMean():.2f}#pm{hEven.GetMeanError():.2f}")
    s.DrawLatex(0.5,0.53,f"Odd  #mu={hOdd.GetMean():.2f}#pm{hOdd.GetMeanError():.2f}")
    if(hbkg is not None):
        s.DrawLatex(0.5,0.46,f"Bkg  #mu={hbkg.GetMean():.2f}#pm{hbkg.GetMeanError():.2f}")

    leg = ROOT.TLegend(0.7, 0.75, 0.88, 0.88)
    leg.SetFillStyle(4000) # will be transparent
    leg.SetFillColor(0)
    leg.SetTextFont(42)
    leg.SetTextSize(0.037)
    leg.SetBorderSize(0)
    leg.AddEntry(hEven, "Even", "l")
    leg.AddEntry(hOdd, "Odd", "l")
    if(hbkg is not None): leg.AddEntry(hbkg, "Bkg", "l")
    leg.Draw()

    c.RedrawAxis()
    c.Update()
    c.Draw()
    c.SaveAs(pdfname)


runs = [1162,1163,1165,1166,1167,1168,1169,1170,1171,1172,1173,1174]

basepath = "data/e320_prototype_beam_Jun222026_noalignment/runs"
runpath  = "run_000"
fprfx    = "tree_Run"
fsufx    = "_allplots.root"
hname    = "hTriggers"
pdfname  = "plots/parity_test.pdf("
title    = "Triggers"
rebin    = -1
plot_triggers(runs,basepath,runpath,fprfx,fsufx,hname,pdfname,title)

basepath = "data/e320_prototype_beam_Jun222026_noalignment/runs"
runpath  = "run_000"
fprfx    = "tree_Run"
fsufx    = "_multiprocess_histograms.root"
hname    = "h_nPixels"
pdfname  = "plots/parity_test.pdf"
title    = "All pixels"
rebin    = 2
norm     = True
hbkg = get_bkg_histogram(f"{basepath}/{runpath}1175/{fprfx}1175{fsufx}",hname)
plot_parity(runs,basepath,runpath,fprfx,fsufx,hname,pdfname,title,rebin,norm,hbkg)

basepath = "data/e320_prototype_beam_Jun222026_noalignment/runs"
runpath  = "run_000"
fprfx    = "tree_Run"
fsufx    = "_multiprocess_histograms.root"
hname    = "h_nPixels_btrfly"
pdfname  = "plots/parity_test.pdf"
title    = "Butterfly pixels"
rebin    = -1
norm     = True
hbkg = get_bkg_histogram(f"{basepath}/{runpath}1175/{fprfx}1175{fsufx}",hname)
plot_parity(runs,basepath,runpath,fprfx,fsufx,hname,pdfname,title,rebin,norm,hbkg)

basepath = "data/e320_prototype_beam_Jun222026_noalignment/runs"
runpath  = "run_000"
fprfx    = "tree_Run"
fsufx    = "_multiprocess_histograms.root"
hname    = "h_nClusters"
pdfname  = "plots/parity_test.pdf"
title    = "All clusters"
rebin    = 2
norm     = True
hbkg = get_bkg_histogram(f"{basepath}/{runpath}1175/{fprfx}1175{fsufx}",hname)
plot_parity(runs,basepath,runpath,fprfx,fsufx,hname,pdfname,title,rebin,norm,hbkg)

basepath = "data/e320_prototype_beam_Jun222026_noalignment/runs"
runpath  = "run_000"
fprfx    = "tree_Run"
fsufx    = "_multiprocess_histograms.root"
hname    = "h_nClusters_btrfly"
pdfname  = "plots/parity_test.pdf"
title    = "Butterfly clusters"
rebin    = -1
norm     = True
hbkg = get_bkg_histogram(f"{basepath}/{runpath}1175/{fprfx}1175{fsufx}",hname)
plot_parity(runs,basepath,runpath,fprfx,fsufx,hname,pdfname,title,rebin,norm,hbkg)

basepath = "data/e320_prototype_beam_Jun222026_noalignment/runs"
runpath  = "run_000"
fprfx    = "tree_Run"
fsufx    = "_allplots.root"
hname    = "h_nTracks_btrfly"
pdfname  = "plots/parity_test.pdf"
title    = "Tracks"
rebin    = -1
norm     = False
hbkg = get_bkg_histogram(f"{basepath}/{runpath}1175/{fprfx}1175{fsufx}",hname)
plot_parity(runs,basepath,runpath,fprfx,fsufx,hname,pdfname,title,rebin,norm,hbkg)

# basepath = "data/e320_prototype_beam_Jun222026_noalignment/runs"
# runpath  = "run_000"
# fprfx    = "tree_Run"
# fsufx    = "_allplots.root"
# hname    = "h_MLE_theta1_logx_after_cuts"
# pdfname  = "plots/parity_test.pdf"
# title    = "Tracks"
# rebin    = -1
# plot_parity(runs,basepath,runpath,fprfx,fsufx,hname,pdfname,title,rebin)

# basepath = "data/e320_prototype_beam_Jun222026_noalignment/runs"
# runpath  = "run_000"
# fprfx    = "tree_Run"
# fsufx    = "_allplots.root"
# hname    = "hChi2DoF_zeroshrcls"
# pdfname  = "plots/parity_test.pdf"
# title    = "Tracks"
# rebin    = -1
# plot_parity(runs,basepath,runpath,fprfx,fsufx,hname,pdfname,title,rebin)

basepath = "data/e320_prototype_beam_Jun222026_noalignment/runs"
runpath  = "run_000"
fprfx    = "tree_Run"
fsufx    = "_allplots.root"
hname    = "hChi2DoF_small_zeroshrcls"
pdfname  = "plots/parity_test.pdf)"
title    = "Tracks"
rebin    = -1
plot_parity(runs,basepath,runpath,fprfx,fsufx,hname,pdfname,title,rebin)


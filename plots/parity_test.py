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


xDipoleExitMin = -22.352
xDipoleExitMax = +22.352
yDipoleExitMin = -66.927
yDipoleExitMax = +34.927
dipole = ROOT.TPolyLine()
xMinD = xDipoleExitMin
xMaxD = xDipoleExitMax
yMinD = yDipoleExitMin
yMaxD = yDipoleExitMax  
dipole.SetNextPoint(xMinD,yMinD)
dipole.SetNextPoint(xMinD,yMaxD)
dipole.SetNextPoint(xMaxD,yMaxD)
dipole.SetNextPoint(xMaxD,yMinD)
dipole.SetNextPoint(xMinD,yMinD)
dipole.SetLineColor(ROOT.kBlue)
dipole.SetLineWidth(1)




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

def plot_triggers(histos,runs,basepath,runpath,fprfx,fsufx,hname,pdfname,title):
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
    
    # h.SetMinimum(0.8)
    
    c = ROOT.TCanvas("c", "Comparison", 800, 600)
    c.SetTicks(1,1)
    # c.SetLogy()
    h.Draw("HIST TEXT0")

    c.RedrawAxis()
    c.Update()
    c.Draw()
    c.SaveAs(pdfname)


def plot_parity(histos,runs,basepath,runpath,fprfx,fsufx,hname,pdfname,title,rebin,norm=True,hbkg=None,sht=True):
    fname = f"{basepath}/{runpath}{runs[0]}/{fprfx}{runs[0]}{fsufx}"
    print(fname)
    f = ROOT.TFile.Open(fname)
    print(f"{hname}_even")
    hEven = f.Get(f"{hname}_even")
    hOdd  = f.Get(f"{hname}_odd")
    hSht = None
    if(sht): hSht = f.Get(f"{hname}_shuttered")
    hEven.Reset()
    hOdd.Reset()
    if(sht): hSht.Reset()
    hEven.SetLineColor(ROOT.kBlack)
    hOdd.SetLineColor(ROOT.kGreen+2)
    if(sht): hSht.SetLineColor(ROOT.kRed)
    hEven.SetFillColorAlpha(ROOT.kBlack,0.35)
    hOdd.SetFillColorAlpha(ROOT.kGreen+2,0.35)
    if(sht): hSht.SetFillColorAlpha(ROOT.kRed,0.35)
    hEven.SetLineWidth(2)
    hOdd.SetLineWidth(2)
    if(sht): hSht.SetLineWidth(2)
    if(rebin>1):
        hEven.Rebin(rebin)
        hOdd.Rebin(rebin)
        if(sht): hSht.Rebin(rebin)
    hEven.SetTitle(title)
    hOdd.SetTitle(title)
    if(sht): hSht.SetTitle(title)
    
    for run in runs:
        frname = f"{basepath}/{runpath}{run}/{fprfx}{run}{fsufx}"
        fr = ROOT.TFile.Open(frname)
        hrEven = fr.Get(f"{hname}_even")
        hrOdd  = fr.Get(f"{hname}_odd")
        hrSht  = None
        if(sht): hrSht = fr.Get(f"{hname}_shuttered")
        if(rebin>1):
            hrEven.Rebin(rebin)
            hrOdd.Rebin(rebin)
            if(sht): hrSht.Rebin(rebin)
        hEven.Add(hrEven)
        hOdd.Add(hrOdd)
        if(sht): hSht.Add(hrSht)
    
    c = ROOT.TCanvas("c", "Comparison", 800, 600)
    c.SetTicks(1,1)
    
    if(norm):
        if(sht): hSht.GetYaxis().SetTitle("Normalized")
        hEven.GetYaxis().SetTitle("Normalized")
        hOdd.GetYaxis().SetTitle("Normalized")
        if(sht):
            if(hSht.GetMean()>1 and hSht.GetEntries()>10):
                hSht.DrawNormalized("HIST")
                hOdd.DrawNormalized("HIST SAME")
                hEven.DrawNormalized("HIST SAME")
            else:
                hOdd.DrawNormalized("HIST")
                hEven.DrawNormalized("HIST SAME")
                hSht.DrawNormalized("HIST SAME")
        else:
            hOdd.DrawNormalized("HIST")
            hEven.DrawNormalized("HIST SAME")
    else:
        if(sht):
            if(hSht.GetMean()>1 and hSht.GetEntries()>10):
                hSht.Draw("HIST")
                hOdd.Draw("HIST SAME")
                hEven.Draw("HIST SAME")
            else:
                hOdd.Draw("HIST")
                hEven.Draw("HIST SAME")
                hSht.Draw("HIST SAME")
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
    s.DrawLatex(0.5,0.52,f"Even   #mu={hEven.GetMean():.2f}#pm{hEven.GetMeanError():.2f}")
    s.DrawLatex(0.5,0.45,f"Odd    #mu={hOdd.GetMean():.2f}#pm{hOdd.GetMeanError():.2f}")
    if(sht): s.DrawLatex(0.5,0.38,f"Shut.  #mu={hSht.GetMean():.2f}#pm{hSht.GetMeanError():.2f}")
    if(hbkg is not None):
        s.DrawLatex(0.5,0.46,f"Bkg  #mu={hbkg.GetMean():.2f}#pm{hbkg.GetMeanError():.2f}")
    if(hname=="hChi2DoF_small_zeroshrcls"):
        chi2cut = 3
        b0 = hEven.FindBin(0)
        b1 = hEven.FindBin(chi2cut)
        s.DrawLatex(0.4,0.74,f"Even    N_{{tracks}} with #chi/N_{{DoF}}<{chi2cut}: {int(hEven.Integral(b0,b1))}")
        s.DrawLatex(0.4,0.67,f"Odd     N_{{tracks}} with #chi/N_{{DoF}}<{chi2cut}: {int(hOdd.Integral(b0,b1))}")
        if(sht): s.DrawLatex(0.4,0.60,f"Shut.:  N_{{tracks}} with #chi/N_{{DoF}}<{chi2cut}: {int(hSht.Integral(b0,b1))}")
    

    leg = ROOT.TLegend(0.6, 0.75, 0.88, 0.88)
    leg.SetFillStyle(4000) # will be transparent
    leg.SetFillColor(0)
    leg.SetTextFont(42)
    leg.SetTextSize(0.037)
    leg.SetBorderSize(0)
    leg.AddEntry(hEven, "Even", "f")
    leg.AddEntry(hOdd, "Odd", "f")
    if(sht): leg.AddEntry(hSht, "Shuttered", "f")
    if(hbkg is not None): leg.AddEntry(hbkg, "Bkg", "f")
    leg.Draw()

    c.RedrawAxis()
    c.Update()
    c.Draw()
    c.SaveAs(pdfname)

    histos.update({hOdd.GetName():hOdd.Clone()})
    histos.update({hEven.GetName():hEven.Clone()})
    if(sht): histos.update({hSht.GetName():hSht.Clone()})




def plot_parity_2D(histos,runs,basepath,runpath,fprfx,fsufx,hname,pdfname,dipole=None):
    fname = f"{basepath}/{runpath}{runs[0]}/{fprfx}{runs[0]}{fsufx}"
    f = ROOT.TFile.Open(fname)
    hEven = f.Get(f"{hname}_even")
    hOdd  = f.Get(f"{hname}_odd")
    hEven.Reset()
    hOdd.Reset()
    
    for run in runs:
        frname = f"{basepath}/{runpath}{run}/{fprfx}{run}{fsufx}"
        fr = ROOT.TFile.Open(frname)
        hrEven = fr.Get(f"{hname}_even")
        hrOdd  = fr.Get(f"{hname}_odd")
        hEven.Add(hrEven)
        hOdd.Add(hrOdd)
    
    c = ROOT.TCanvas("c", "Comparison", 1600, 600)
    c.Divide(2,1)

    chi2cut = 3
    bx0 = hEven.GetXaxis().FindBin(-20)
    bx1 = hEven.GetXaxis().FindBin(+20)
    by0 = hEven.GetYaxis().FindBin(-20)
    by1 = hEven.GetYaxis().FindBin(+20)

    c.cd(1)
    ROOT.gPad.SetTicks(1,1)
    ROOT.gPad.SetGridx()
    ROOT.gPad.SetGridy()
    hEven.Draw("col")
    if(dipole is not None): dipole.Draw()
    s = ROOT.TLatex()
    s.SetNDC(1)
    s.SetTextAlign(13)
    s.SetTextColor(ROOT.kBlack)
    s.SetTextFont(132)
    s.SetTextSize(0.045)
    s.DrawLatex(0.3,0.74,f"Even N_{{trk}} in #pm20 mm (both x,y): {int(hEven.Integral(bx0,bx1,by0,by1))}")
    ROOT.gPad.RedrawAxis()
    print(f"hEven.GetEntries()={hEven.GetEntries()}")

    c.cd(2)
    ROOT.gPad.SetTicks(1,1)
    ROOT.gPad.SetGridx()
    ROOT.gPad.SetGridy()
    hOdd.Draw("col")
    if(dipole is not None): dipole.Draw()
    s = ROOT.TLatex()
    s.SetNDC(1)
    s.SetTextAlign(13)
    s.SetTextColor(ROOT.kBlack)
    s.SetTextFont(132)
    s.SetTextSize(0.045)
    s.DrawLatex(0.3,0.74,f"Odd  N_{{trk}} in #pm20 mm (both x,y): {int(hOdd.Integral(bx0,bx1,by0,by1))}")
    ROOT.gPad.RedrawAxis()
    print(f"hOdd.GetEntries()={hOdd.GetEntries()}")
    
    c.Update()
    c.Draw()
    c.SaveAs(pdfname)

    histos.update({hOdd.GetName():hOdd.Clone()})
    histos.update({hEven.GetName():hEven.Clone()})


histos = {}

# runs = [1162,1163,1165,1166,1167,1168,1169,1170,1171,1172,1173,1174]
runs = [1212,1217,1218,1219,
        1220,1221,1223,1226,1227,1228,1229,
        1230,1231,1232,1233,1234,1235,1236,1237,1238,1239,
        1240,1241,1242,1243,1244,1245,1246,1247,1248,1249,
        1250,1251,1252,1253,1254,1255,1256,1257,1258,1259,
        1260,1261,1262,1263,1264,1265,1266,1267,1268,1269,
        1270,1271,1272,1273,1275,1276,1277,1279,1280,1281
]

# basepath = "data/e320_prototype_beam_Jun22026_noalignment/runs"
basepath = "data/e320_prototype_beam_Jun242026_noalignment/runs"
runpath  = "run_000"
fprfx    = "tree_Run"
fsufx    = "_allplots.root"
hname    = "hTriggersParity"
pdfname  = "plots/parity_test.pdf("
title    = "Triggers parity"
rebin    = -1
plot_triggers(histos,runs,basepath,runpath,fprfx,fsufx,hname,pdfname,title)

# basepath = "data/e320_prototype_beam_Jun222026_noalignment/runs"
basepath = "data/e320_prototype_beam_Jun242026_noalignment/runs"
runpath  = "run_000"
fprfx    = "tree_Run"
fsufx    = "_multiprocess_histograms.root"
hname    = "h_nPixels_all"
pdfname  = "plots/parity_test.pdf"
title    = "All pixels (#geq0 pixel in all layers)"
rebin    = 2
norm     = True
hbkg     = None #get_bkg_histogram(f"{basepath}/{runpath}1175/{fprfx}1175{fsufx}",hname)
sht      = True
plot_parity(histos,runs,basepath,runpath,fprfx,fsufx,hname,pdfname,title,rebin,norm,hbkg,sht)

# basepath = "data/e320_prototype_beam_Jun222026_noalignment/runs"
basepath = "data/e320_prototype_beam_Jun242026_noalignment/runs"
runpath  = "run_000"
fprfx    = "tree_Run"
fsufx    = "_multiprocess_histograms.root"
hname    = "h_nPixels_all_btrfly"
pdfname  = "plots/parity_test.pdf"
title    = "Butterfly pixels (#geq0 pixel in all layers)"
rebin    = -1
norm     = True
hbkg     = None #get_bkg_histogram(f"{basepath}/{runpath}1175/{fprfx}1175{fsufx}",hname)
sht      = True
plot_parity(histos,runs,basepath,runpath,fprfx,fsufx,hname,pdfname,title,rebin,norm,hbkg,sht)

# basepath = "data/e320_prototype_beam_Jun222026_noalignment/runs"
basepath = "data/e320_prototype_beam_Jun242026_noalignment/runs"
runpath  = "run_000"
fprfx    = "tree_Run"
fsufx    = "_multiprocess_histograms.root"
hname    = "h_nPixels_hitperlyr"
pdfname  = "plots/parity_test.pdf"
title    = "All pixels (#geq1 pixel in all layers)"
rebin    = 2
norm     = True
hbkg     = None #get_bkg_histogram(f"{basepath}/{runpath}1175/{fprfx}1175{fsufx}",hname)
sht      = True
plot_parity(histos,runs,basepath,runpath,fprfx,fsufx,hname,pdfname,title,rebin,norm,hbkg,sht)

# basepath = "data/e320_prototype_beam_Jun222026_noalignment/runs"
basepath = "data/e320_prototype_beam_Jun242026_noalignment/runs"
runpath  = "run_000"
fprfx    = "tree_Run"
fsufx    = "_multiprocess_histograms.root"
hname    = "h_nPixels_hitperlyr_btrfly"
pdfname  = "plots/parity_test.pdf"
title    = "Butterfly pixels (#geq1 pixel in all layers)"
rebin    = -1
norm     = True
hbkg     = None #get_bkg_histogram(f"{basepath}/{runpath}1175/{fprfx}1175{fsufx}",hname)
sht      = True
plot_parity(histos,runs,basepath,runpath,fprfx,fsufx,hname,pdfname,title,rebin,norm,hbkg,sht)

# basepath = "data/e320_prototype_beam_Jun222026_noalignment/runs"
basepath = "data/e320_prototype_beam_Jun242026_noalignment/runs"
runpath  = "run_000"
fprfx    = "tree_Run"
fsufx    = "_multiprocess_histograms.root"
hname    = "h_nClusters_hitperlyr"
pdfname  = "plots/parity_test.pdf"
title    = "All clusters"
rebin    = 2
norm     = True
hbkg     = None #get_bkg_histogram(f"{basepath}/{runpath}1175/{fprfx}1175{fsufx}",hname)
sht      = True
plot_parity(histos,runs,basepath,runpath,fprfx,fsufx,hname,pdfname,title,rebin,norm,hbkg,sht)

# basepath = "data/e320_prototype_beam_Jun222026_noalignment/runs"
basepath = "data/e320_prototype_beam_Jun242026_noalignment/runs"
runpath  = "run_000"
fprfx    = "tree_Run"
fsufx    = "_multiprocess_histograms.root"
hname    = "h_nClusters_hitperlyr_btrfly"
pdfname  = "plots/parity_test.pdf"
title    = "Butterfly clusters"
rebin    = -1
norm     = True
hbkg     = None #get_bkg_histogram(f"{basepath}/{runpath}1175/{fprfx}1175{fsufx}",hname)
sht      = True
plot_parity(histos,runs,basepath,runpath,fprfx,fsufx,hname,pdfname,title,rebin,norm,hbkg,sht)

# basepath = "data/e320_prototype_beam_Jun222026_noalignment/runs"
basepath = "data/e320_prototype_beam_Jun242026_noalignment/runs"
runpath  = "run_000"
fprfx    = "tree_Run"
fsufx    = "_allplots.root"
hname    = "h_nTracks_btrfly"
pdfname  = "plots/parity_test.pdf"
title    = "Tracks"
rebin    = -1
plot_parity(histos,runs,basepath,runpath,fprfx,fsufx,hname,pdfname,title,rebin,norm=False,hbkg=None,sht=True)

# basepath = "data/e320_prototype_beam_Jun222026_noalignment/runs"
basepath = "data/e320_prototype_beam_Jun242026_noalignment/runs"
runpath  = "run_000"
fprfx    = "tree_Run"
fsufx    = "_allplots.root"
hname    = "hChi2DoF_small_zeroshrcls"
pdfname  = "plots/parity_test.pdf"
title    = "Tracks"
rebin    = -1
plot_parity(histos,runs,basepath,runpath,fprfx,fsufx,hname,pdfname,title,rebin,norm=False,hbkg=None,sht=True)

# basepath = "data/e320_prototype_beam_Jun222026_noalignment/runs"
basepath = "data/e320_prototype_beam_Jun242026_noalignment/runs"
runpath  = "run_000"
fprfx    = "tree_Run"
fsufx    = "_allplots.root"
hname    = "hChi2DoF_zeroshrcls"
pdfname  = "plots/parity_test.pdf"
title    = "Tracks"
rebin    = -1
plot_parity(histos,runs,basepath,runpath,fprfx,fsufx,hname,pdfname,title,rebin,norm=False,hbkg=None,sht=True)

basepath = "data/e320_prototype_beam_Jun242026_noalignment/runs"
runpath  = "run_000"
fprfx    = "tree_Run"
fsufx    = "_allplots.root"
hname    = "hD_before_cuts"
pdfname  = "plots/parity_test.pdf"
plot_parity_2D(histos,runs,basepath,runpath,fprfx,fsufx,hname,pdfname,dipole)

basepath = "data/e320_prototype_beam_Jun242026_noalignment/runs"
runpath  = "run_000"
fprfx    = "tree_Run"
fsufx    = "_allplots.root"
hname    = "hD_after_cuts"
pdfname  = "plots/parity_test.pdf)"
plot_parity_2D(histos,runs,basepath,runpath,fprfx,fsufx,hname,pdfname,dipole)




fout = ROOT.TFile("plots/parity_test.root","RECREATE")
fout.cd()
for hname,hist in histos.items(): hist.Write()
fout.Write()
fout.Close()
    


basepath = "data/e320_prototype_beam_Jun242026_noalignment/runs"
txtfiles = [] # tree_Run1281_alltrks
for run in runs: txtfiles.append( f"{basepath}/{runpath}{run}/{fprfx}{run}_alltrks.txt" )
with open("plots/e320_Jun232026_Jun242026_alltracks.txt", "w") as out:
    for path in txtfiles:
        with open(path, "r") as f:
            content = f.read()
            out.write(content)
            if not content.endswith("\n"):
                out.write("\n")
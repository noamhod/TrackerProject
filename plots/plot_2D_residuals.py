#!/usr/bin/python
import time
import os
import math
import array
import numpy as np
import ROOT

ROOT.gROOT.SetBatch(1)
ROOT.gStyle.SetOptFit(0)
ROOT.gStyle.SetOptStat(0)
# ROOT.gStyle.SetPalette(ROOT.kRust)
# ROOT.gStyle.SetPalette(ROOT.kSolar)
# ROOT.gStyle.SetPalette(ROOT.kInvertedDarkBodyRadiator)
ROOT.gStyle.SetPalette(ROOT.kDarkBodyRadiator)
# ROOT.gStyle.SetPalette(ROOT.kRainbow)
ROOT.gStyle.SetPadTopMargin(0.1)
ROOT.gStyle.SetPadBottomMargin(0.15)
ROOT.gStyle.SetPadLeftMargin(0.18)
ROOT.gStyle.SetPadRightMargin(0.05)
ROOT.gErrorIgnoreLevel = ROOT.kError
# ROOT.gErrorIgnoreLevel = ROOT.kWarning


detectors = ["ALPIDE_0","ALPIDE_1","ALPIDE_2","ALPIDE_3","ALPIDE_4"]

basedir = "data/e320_prototype_beam_Feb2025_no_alignment/runs/run_0000502"
fnamein = f"{basedir}/tree_Run502_allplots.root"
hstname = "h_residual_zeroshrcls_xy_mid"
fIn  = ROOT.TFile(fnamein,"READ")


fnameout = fnamein.replace(".root","")


cnv = ROOT.TCanvas("cnv_dipole_window","",2800,500)
cnv.Divide(5,1)
for idet,det in enumerate(detectors):
    cnv.cd(idet+1)
    ROOT.gPad.SetTicks(1,1)
    name = f"{hstname}_{det}"
    print(f"{name}")
    h = fIn.Get(name).Clone(f"{name}_clone")
    h.GetZaxis().SetTitleOffset(1.3)
    h.GetXaxis().SetTitleSize(0.06)
    h.GetYaxis().SetTitleSize(0.06)
    h.GetZaxis().SetTitleSize(0.06)
    
    h.GetXaxis().SetLabelSize(0.04)
    h.GetYaxis().SetLabelSize(0.04)
    h.GetZaxis().SetLabelSize(0.04)
    
    h.Draw("col")
    ROOT.gPad.RedrawAxis()
cnv.Update()
cnv.SaveAs(f"{fnameout}_2D_residuals.pdf")

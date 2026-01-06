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
ROOT.gStyle.SetPadTopMargin(0.05)
ROOT.gStyle.SetPadBottomMargin(0.1)
ROOT.gStyle.SetPadLeftMargin(0.1)
ROOT.gStyle.SetPadRightMargin(0.03)
ROOT.gErrorIgnoreLevel = ROOT.kError
# ROOT.gErrorIgnoreLevel = ROOT.kWarning



def fit2(h, poln=0):
    x_min = h.GetXaxis().GetXmin()
    x_max = h.GetXaxis().GetXmax()

    # --- 1. Define Formula Strings ---
    # We build two strings: one for the full fit (indices offset by 2)
    # and one for the isolated background (indices starting at 0)
    
    # Base Signal: [0] = Norm, [1] = NDF
    fit_formula = "[0] * ROOT::Math::chisquared_pdf(x, [1])"
    # fit_formula = "[0] * ROOT::Math::chisquared_pdf(x*[1], [1])"
    bkg_formula = ""

    # Helper to build the polynomial string dynamically
    # offset=2 for the full fit, offset=0 for the isolated bkg
    def get_poly_str(order, offset):
        s = ""
        for i in range(order + 1):
            term = f"[{i+offset}]"
            if i > 0: term += f"*x^{i}" if i > 1 else "*x"
            s += f" + {term}"
        return s

    bkg_formula = "0"

    # Logic to append specific background models
    if poln == -1:
        fit_formula += " + [2]*ROOT::Math::erf([3]-x)-[4]*x^0.4"
        bkg_formula  = "[0]*ROOT::Math::erf([1]-x)-[2]*x^0.4"
    elif poln > 0: # Polynomials (0 to 7)
        poly_part_fit = get_poly_str(int(poln), offset=2)
        poly_part_bkg = get_poly_str(int(poln), offset=0)
        fit_formula += poly_part_fit
        bkg_formula = poly_part_bkg.lstrip(" +") # Remove leading " +"

    # --- 2. Create Functions ---
    fit_func = ROOT.TF1("fit_func", fit_formula, x_min, x_max)
    bkg_func = ROOT.TF1("bkg_func", bkg_formula, x_min, x_max)


    # NDF: Guessing 10.0
    fit_func.SetParameter(1, 10.0)
    fit_func.SetParLimits(1, 1.0, 50.0)
    # --- Background Initial Guesses ---
    if(poln==-1):
        fit_func.SetParameter(3, 1 ) ## peak
        fit_func.SetParLimits(3, 0, 2.0)

    # Naming parameters (Safety check added to ensure param exists)
    fit_func.SetParName(0, "Sig_Norm")
    fit_func.SetParName(1, "Chi2_NDF")
    if fit_func.GetNpar() > 2: fit_func.SetParName(2, "Bkg_p0")
    if fit_func.GetNpar() > 3: fit_func.SetParName(3, "Bkg_p1")
    if fit_func.GetNpar() > 4: fit_func.SetParName(4, "Bkg_p2")

    # --- 4. Perform the Fit ---
    print(f"Fitting with model: {fit_formula}")
    fit_result = h.Fit("fit_func", "LEMRS") # L=LogLikelihood, E=Minos, M=More, R=Range, S=Save

    # --- 5. Sync Background Function ---
    # Copy parameters from the Full Fit (starting at index 2) 
    # to the Background Function (starting at index 0)
    for i in range(bkg_func.GetNpar()):
        # Map Full[i+2] -> Bkg[i]
        val = fit_func.GetParameter(i + 2)
        err = fit_func.GetParError(i + 2)
        bkg_func.SetParameter(i, val)
        bkg_func.SetParError(i, err)

    # Stylize for drawing
    fit_func.SetLineColor(ROOT.kRed)
    fit_func.SetLineWidth(2)
    
    bkg_func.SetLineColor(ROOT.kBlue)
    bkg_func.SetLineStyle(2) # Dashed line
    bkg_func.SetLineWidth(2)

    # Calc Chi2/NDF
    ndf = fit_func.GetNDF()
    chi2dof = fit_func.GetChisquare() / ndf if ndf > 0 else -1
    print("chi2/Ndof=", chi2dof)

    return fit_func, bkg_func, fit_result




basedir = "data/e320_prototype_beam_Feb2025/runs/run_0000502"
fnamein = f"{basedir}/tree_Run502_allplots.root"
fIn  = ROOT.TFile(fnamein,"READ")


fnameout = fnamein.replace(".root","")



cnv = ROOT.TCanvas("cnv1","",750,500)
cnv.cd()
ROOT.gPad.SetTicks(1,1)
ROOT.gPad.SetGridx()
ROOT.gPad.SetGridy()
# h = fIn.Get(f"hChi2DoF_zeroshrcls")
h = fIn.Get(f"hChi2DoF_small_zeroshrcls")
h.Sumw2()

# # sfunc,bfunc,frslt = fit2(h,6)
# sfunc,bfunc,frslt = fit2(h,-1)

h.SetMinimum(0)
# h.SetMaximum(300)
h.SetMaximum(160)
h.SetMarkerStyle(20)
h.SetMarkerColor(ROOT.kBlack)
h.SetLineColor(ROOT.kBlack)
h.Draw("e1p")
# sfunc.Draw("same")
# bfunc.Draw("same")
s = ROOT.TLatex()
s.SetNDC(1)
s.SetTextAlign(13)
s.SetTextColor(ROOT.kBlack)
s.SetTextFont(22)
s.SetTextSize(0.06)
ROOT.gPad.RedrawAxis()
cnv.Update()
cnv.SaveAs(f"{fnameout}_chi2.pdf")
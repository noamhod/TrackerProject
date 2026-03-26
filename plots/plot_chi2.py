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
    # fit_formula = "[0] * ROOT::Math::chisquared_pdf(x/[1], [1])"
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
    if poln == -2:
        fit_formula += " + (([2] + [3]*x + [4]*x*x) / (1.0 + TMath::Exp(-1.0 * (x - [5]) / [6])))"
        bkg_formula = "(([0] + [1]*x + [2]*x*x) / (1.0 + TMath::Exp(-1.0 * (x - [3]) / [4])))"
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
    fit_func.SetParLimits(1, 0, 100.0)
    # --- Background Initial Guesses ---
    # if(poln==-2):
        # fit_func.SetParameter(4, 5) ## mean
        # fit_func.SetParLimits(4, 0, 10)
    if(poln==-1):
        fit_func.SetParameter(3, 1 ) ## peak
        fit_func.SetParLimits(3, 0, 2.0)
        
    print(f"Npar = {fit_func.GetNpar()}")

    # Naming parameters (Safety check added to ensure param exists)
    fit_func.SetParName(0, "Sig_Norm")
    fit_func.SetParName(1, "Chi2_NDF")
    if fit_func.GetNpar() > 2: fit_func.SetParName(2, "Bkg_p0")
    if fit_func.GetNpar() > 3: fit_func.SetParName(3, "Bkg_p1")
    if fit_func.GetNpar() > 4: fit_func.SetParName(4, "Bkg_p2")
    if fit_func.GetNpar() > 5: fit_func.SetParName(5, "Bkg_p3")

    # --- 4. Perform the Fit ---
    print(f"Fitting with model: {fit_formula}")
    fit_result = h.Fit("fit_func", "LEMRS") # L=LogLikelihood, E=Minos, M=More, R=Range, S=Save

    # --- 5. Sync Background Function ---
    for i in range(2,fit_func.GetNpar()):
        print(f"setting par[{i}] of fit_func (par[{i-2}] of bkg_func)")
        val = fit_func.GetParameter(i)
        err = fit_func.GetParError(i)
        bkg_func.SetParameter(i-2, val)
        bkg_func.SetParError(i-2, err)

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
    
    
    # This calculates the 95% confidence interval by default (0.95). 
    # It evaluates the fit_func at the center of every bin in h_band, 
    # computes the error using the fit_result covariance matrix, 
    # and sets the bin content and error of h_band.
    # Pass the fit_result pointer if available; otherwise it uses the attached fit.
    # (ROOT 6.x syntax often allows passing just the TF1 if it was just fitted, 
    # but passing the FitResult is safest).
    h_band = h.Clone("band")
    h_band.Reset()
    h_band.SetFillColorAlpha(ROOT.kRed,0.3)
    h_band.SetLineColorAlpha(ROOT.kRed,0.3)
    h_band.SetMarkerSize(0)
    # ROOT.TVirtualFitter.GetFitter().GetConfidenceIntervals(h_band, 0.6827)
    ROOT.TVirtualFitter.GetFitter().GetConfidenceIntervals(h_band, 0.9545)
    
    return fit_func, bkg_func, fit_result, h_band
    



# basedir = "data/e320_prototype_beam_Feb2025/runs/run_0000502"
# fnamein = f"{basedir}/tree_Run502_allplots.root"
# ### fnamein = f"{basedir}/tree_Run502_allplots_iter0_nocuts.root"
# hstname = "hChi2DoF_small_zeroshrcls"
# # hstname = "hChi2_small_zeroshrcls"

# basedir = "data/e320_prototype_beam_Feb2025_no_alignment/runs/run_0000502"
# fnamein = f"{basedir}/tree_Run502_allplots.root"
# hstname = "hChi2DoF_zeroshrcls"

basedir = "data/e320_prototype_beam_Mar2026/runs/run_0000856"
fnamein = f"{basedir}/tree_Run856_allplots.root"
hstname = "hChi2DoF_small_zeroshrcls"
# hstname = "hChi2_small_zeroshrcls"

isnoalgn = ("_no_alignment" in basedir)
fIn  = None
fIn0 = None
fIn1 = None
fIn2 = None
if(isnoalgn):
    fIn0 = ROOT.TFile(fnamein.replace(".root","_nocuts.root"),"READ")
    fIn1 = ROOT.TFile(fnamein.replace(".root","_dkcut.root"),"READ")
    fIn2 = ROOT.TFile(fnamein.replace(".root","_dkspotcut.root"),"READ")
else:
    fIn  = ROOT.TFile(fnamein,"READ")


fnameout = fnamein.replace(".root","") if(not isnoalgn) else fnamein.replace(".root","_evolution")


cnv = ROOT.TCanvas("cnv1","",750,500)
cnv.cd()
ROOT.gPad.SetTicks(1,1)
ROOT.gPad.SetGridx()
ROOT.gPad.SetGridy()
if(isnoalgn):
    h0 = fIn0.Get(f"{hstname}").Clone("nocuts")
    h0.Sumw2()
    h1 = fIn1.Get(f"{hstname}").Clone("dkcut")
    h1.Sumw2()
    h2 = fIn2.Get(f"{hstname}").Clone("dkspotcuts")
    h2.Sumw2()
else:
    h = fIn.Get(f"{hstname}")
    h.Sumw2()

if(isnoalgn):
    h0.GetXaxis().SetTitle("#tilde{#chi}^{2}_{inf}")
    h1.GetXaxis().SetTitle("#tilde{#chi}^{2}_{inf}")
    h2.GetXaxis().SetTitle("#tilde{#chi}^{2}_{inf}")
    h0.SetMinimum(0)
    h1.SetMinimum(0)
    h2.SetMinimum(0)
    if("_small_" in hstname):
        h0.SetMaximum(180)
        h1.SetMaximum(180)
        h2.SetMaximum(180)
    else:
        h0.SetMaximum(310)
        h1.SetMaximum(310)
        h2.SetMaximum(310)
else:
    h.GetXaxis().SetTitle("#tilde{#chi}^{2}")
    h.SetMinimum(0)
    h.SetMaximum(180 if("DoF" in hstname) else 110)

if(isnoalgn):
    h0.SetLineColor(ROOT.kBlack)
    h0.SetFillColorAlpha(ROOT.kBlack,0.35)
    h1.SetLineColor(ROOT.kBlue)
    h1.SetFillColorAlpha(ROOT.kBlue,0.35)
    h2.SetLineColor(ROOT.kRed)
    h2.SetFillColorAlpha(ROOT.kRed,0.35)
else:
    h.SetMarkerStyle(20)
    h.SetMarkerColor(ROOT.kBlack)
    h.SetLineColor(ROOT.kBlack)

if(isnoalgn):
    leg = ROOT.TLegend(0.5,0.7,0.88,0.88)
    leg.SetFillStyle(4000) # will be transparent
    leg.SetFillColor(0)
    leg.SetTextFont(42)
    leg.SetTextSize(0.037)
    leg.SetBorderSize(0)
    leg.AddEntry(h0,"Baseline cuts","f")
    leg.AddEntry(h1,"With residuals cut","f")
    leg.AddEntry(h2,"With spot cut","f")
    h0.Draw("hist")
    h1.Draw("hist same")
    h2.Draw("hist same")
    leg.Draw("same")
else:
    h.Draw("e1p")
    poln = 0
    # sfunc,bfunc,frslt = fit2(h,poln=-2)
    # sfunc,bfunc,frslt = fit2(h,poln=2)
    sfunc,bfunc,frslt,h_band = fit2(h,poln)
    
    leg = ROOT.TLegend(0.5,0.5,0.88,0.88)
    leg.SetFillStyle(4000) # will be transparent
    leg.SetFillColor(0)
    leg.SetTextFont(42)
    leg.SetTextSize(0.037)
    leg.SetBorderSize(0)
    leg.AddEntry(h,"Post alignment data","ep")
    leg.AddEntry(sfunc,"Signal+Background","l")
    leg.AddEntry(h_band,"Fit uncertainty (2#sigma)","f")
    if(poln!=0): leg.AddEntry(bfunc,"Background","l")
    
    h_band.Draw("same e3")
    h.Draw("e1p same")
    sfunc.Draw("same")
    if(poln!=0): bfunc.Draw("same")
    leg.Draw("same")

s = ROOT.TLatex()
s.SetNDC(1)
s.SetTextAlign(13)
s.SetTextColor(ROOT.kBlack)
s.SetTextFont(22)
s.SetTextSize(0.06)
ROOT.gPad.RedrawAxis()
cnv.Update()
cnv.SaveAs(f"{fnameout}_chi2.pdf")
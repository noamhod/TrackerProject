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
ROOT.gStyle.SetPadBottomMargin(0.16)
ROOT.gStyle.SetPadLeftMargin(0.15)
ROOT.gStyle.SetPadRightMargin(0.05)
ROOT.gErrorIgnoreLevel = ROOT.kError
# ROOT.gErrorIgnoreLevel = ROOT.kWarning


def fit1(h,col,xmin,xmax):
    g1 = ROOT.TF1("g1", "gaus", xmin,xmax)
    g1.SetLineColor(col)
    h.Fit(g1,"EMRS")
    chi2dof = g1.GetChisquare()/g1.GetNDF() if(g1.GetNDF()>0) else -1
    print("g1 chi2/Ndof=",chi2dof)
    return g1



detectors = ["ALPIDE_0","ALPIDE_1","ALPIDE_2","ALPIDE_3","ALPIDE_4"]

basedir = "data/e320_prototype_beam_Feb2025/runs/run_0000502"
fnamein = f"{basedir}/tree_Run502_allplots.root"
fIn  = ROOT.TFile(fnamein,"READ")


fnameout = fnamein.replace(".root","")



cnv = ROOT.TCanvas("cnv1","",2000,300)
cnv.Divide(5,1)
for idet,det in enumerate(detectors):
    cnv.cd(idet+1)
    ROOT.gPad.SetTicks(1,1)
    h = fIn.Get(f"h_residual_zeroshrcls_x_sml_{det}")
    h.Sumw2()
    h.SetTitle(f"Residuals in x for {det}")
    h.GetXaxis().SetLabelSize(0.06) ### default ~0.035
    h.GetXaxis().SetTitleSize(0.07) ### default ~0.04
    h.GetYaxis().SetLabelSize(0.06)
    h.GetYaxis().SetTitleSize(0.07)
    h.GetYaxis().SetTitleOffset(1.05)
    h.Scale(1./h.Integral())
    h.SetMinimum(0)
    h.SetMarkerStyle(20)
    h.SetMarkerColor(ROOT.kBlack)
    h.SetLineColor(ROOT.kBlack)
    h.Draw("e1p")
    xmin = h.GetXaxis().GetXmin()
    xmax = h.GetXaxis().GetXmax()
    mm2um = 1e3
    func = fit1(h,ROOT.kBlue,xmin,xmax)
    s = ROOT.TLatex()
    s.SetNDC(1)
    s.SetTextAlign(13)
    s.SetTextColor(ROOT.kBlack)
    s.SetTextFont(22)
    s.SetTextSize(0.06)
    s.DrawLatex(0.2,0.85,ROOT.Form("#mu=%.2f #mum" % (mm2um*func.GetParameter(1))))
    s.DrawLatex(0.2,0.78,ROOT.Form("#sigma=%.2f #mum" % (mm2um*func.GetParameter(2))))
    if(func.GetNDF()>0): s.DrawLatex(0.2,0.71,ROOT.Form("#chi^{2}/N_{DOF}=%.2f" % (func.GetChisquare()/func.GetNDF())))
    ROOT.gPad.RedrawAxis()
cnv.Update()
cnv.SaveAs(f"{fnameout}_residuals_x.pdf")
print("x res finished")


cnv = ROOT.TCanvas("cnv2","",1200,200)
cnv.Divide(5,1)
for idet,det in enumerate(detectors):
    cnv.cd(idet+1)
    ROOT.gPad.SetTicks(1,1)
    h = fIn.Get(f"h_residual_zeroshrcls_y_sml_{det}")
    h.Sumw2()
    h.SetTitle(f"Residuals in y for {det}")
    h.GetXaxis().SetLabelSize(0.06) ### default ~0.035
    h.GetXaxis().SetTitleSize(0.07) ### default ~0.04
    h.GetYaxis().SetLabelSize(0.06)
    h.GetYaxis().SetTitleSize(0.07)
    h.GetYaxis().SetTitleOffset(1.05)
    h.Scale(1./h.Integral())
    h.SetMinimum(0)
    h.SetMarkerStyle(20)
    h.SetMarkerColor(ROOT.kBlack)
    h.SetLineColor(ROOT.kBlack)
    h.Draw("e1p")
    xmin = h.GetXaxis().GetXmin()
    xmax = h.GetXaxis().GetXmax()
    mm2um = 1e3
    func = fit1(h,ROOT.kBlue,xmin,xmax)
    s = ROOT.TLatex()
    s.SetNDC(1)
    s.SetTextAlign(13)
    s.SetTextColor(ROOT.kBlack)
    s.SetTextFont(22)
    s.SetTextSize(0.06)
    s.DrawLatex(0.2,0.85,ROOT.Form("#mu=%.2f #mum" % (mm2um*func.GetParameter(1))))
    s.DrawLatex(0.2,0.78,ROOT.Form("#sigma=%.2f #mum" % (mm2um*func.GetParameter(2))))
    if(func.GetNDF()>0): s.DrawLatex(0.2,0.71,ROOT.Form("#chi^{2}/N_{DOF}=%.2f" % (func.GetChisquare()/func.GetNDF())))
    ROOT.gPad.RedrawAxis()
cnv.Update()
cnv.SaveAs(f"{fnameout}_residuals_y.pdf")
print("y res finished")





cnv = ROOT.TCanvas("cnv3","",2000,300)
cnv.Divide(5,1)
for idet,det in enumerate(detectors):
    cnv.cd(idet+1)
    ROOT.gPad.SetTicks(1,1)
    h = fIn.Get(f"h_response_zeroshrcls_x_sml_{det}")
    h.Sumw2()
    h.SetTitle(f"Pulls in x for {det}")
    h.GetXaxis().SetTitle("(x_{trk}-x_{cls})/#Deltax_{cls}")
    h.GetXaxis().SetLabelSize(0.06) ### default ~0.035
    h.GetXaxis().SetTitleSize(0.07) ### default ~0.04
    h.GetYaxis().SetLabelSize(0.06)
    h.GetYaxis().SetTitleSize(0.07)
    h.GetYaxis().SetTitleOffset(1.05)
    h.Scale(1./h.Integral())
    h.SetMinimum(0)
    h.SetMarkerStyle(20)
    h.SetMarkerColor(ROOT.kBlack)
    h.SetLineColor(ROOT.kBlack)
    h.Draw("e1p")
    xmin = h.GetXaxis().GetXmin()
    xmax = h.GetXaxis().GetXmax()
    mm2um = 1e3
    func = fit1(h,ROOT.kBlue,xmin,xmax)
    s = ROOT.TLatex()
    s.SetNDC(1)
    s.SetTextAlign(13)
    s.SetTextColor(ROOT.kBlack)
    s.SetTextFont(22)
    s.SetTextSize(0.06)
    s.DrawLatex(0.2,0.85,ROOT.Form("#mu=%.2f" % (func.GetParameter(1))))
    s.DrawLatex(0.2,0.78,ROOT.Form("#sigma=%.2f" % (func.GetParameter(2))))
    if(func.GetNDF()>0): s.DrawLatex(0.2,0.71,ROOT.Form("#chi^{2}/N_{DOF}=%.2f" % (func.GetChisquare()/func.GetNDF())))
    ROOT.gPad.RedrawAxis()
cnv.Update()
cnv.SaveAs(f"{fnameout}_pulls_x.pdf")
print("x pulls finished")


cnv = ROOT.TCanvas("cnv4","",1200,200)
cnv.Divide(5,1)
for idet,det in enumerate(detectors):
    cnv.cd(idet+1)
    ROOT.gPad.SetTicks(1,1)
    h = fIn.Get(f"h_response_zeroshrcls_y_sml_{det}")
    h.Sumw2()
    h.SetTitle(f"Pulls in y for {det}")
    h.GetXaxis().SetTitle("(y_{trk}-y_{cls})/#Deltay_{cls}")
    h.GetXaxis().SetLabelSize(0.06) ### default ~0.035
    h.GetXaxis().SetTitleSize(0.07) ### default ~0.04
    h.GetYaxis().SetLabelSize(0.06)
    h.GetYaxis().SetTitleSize(0.07)
    h.GetYaxis().SetTitleOffset(1.05)
    h.Scale(1./h.Integral())
    h.SetMinimum(0)
    h.SetMarkerStyle(20)
    h.SetMarkerColor(ROOT.kBlack)
    h.SetLineColor(ROOT.kBlack)
    h.Draw("e1p")
    xmin = h.GetXaxis().GetXmin()
    xmax = h.GetXaxis().GetXmax()
    mm2um = 1e3
    func = fit1(h,ROOT.kBlue,xmin,xmax)
    s = ROOT.TLatex()
    s.SetNDC(1)
    s.SetTextAlign(13)
    s.SetTextColor(ROOT.kBlack)
    s.SetTextFont(22)
    s.SetTextSize(0.06)
    s.DrawLatex(0.2,0.85,ROOT.Form("#mu=%.2f" % (func.GetParameter(1))))
    s.DrawLatex(0.2,0.78,ROOT.Form("#sigma=%.2f" % (func.GetParameter(2))))
    if(func.GetNDF()>0): s.DrawLatex(0.2,0.71,ROOT.Form("#chi^{2}/N_{DOF}=%.2f" % (func.GetChisquare()/func.GetNDF())))
    ROOT.gPad.RedrawAxis()
cnv.Update()
cnv.SaveAs(f"{fnameout}_pulls_y.pdf")
print("y pulls finished")


#!/usr/bin/python
import sys
import os
import os.path
import math
import time
import subprocess
import array
import numpy as np
import ROOT
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from scipy.optimize import curve_fit
# from skspatial.objects import Line, Sphere
# from skspatial.plotting import plot_3d

import argparse
parser = argparse.ArgumentParser(description='postptrocess_plots.py...')
parser.add_argument('-conf', metavar='config file', required=True,  help='full path to config file')
parser.add_argument('-mult', metavar='multi run?',  required=False, help='is this a multirun? [0/1]')
argus = parser.parse_args()
configfile = argus.conf
ismutirun  = argus.mult if(argus.mult is not None and int(argus.mult)==1) else False

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tracker_lib import config
from tracker_lib import objects, Pixels, Clusters, candidate, hough_seeder, selections, evtdisp, hists, errors, counters, noise, utils, svd_fit, chi2_fit




ROOT.gROOT.SetBatch(1)
ROOT.gStyle.SetOptFit(0)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetPalette(ROOT.kDarkBodyRadiator)
# ROOT.gStyle.SetPalette(ROOT.kRust)
# ROOT.gStyle.SetPalette(ROOT.kRainbow)
ROOT.gStyle.SetPadBottomMargin(0.15)
ROOT.gStyle.SetPadLeftMargin(0.13)
ROOT.gStyle.SetPadRightMargin(0.16)



mm2um = 1000
histos = {}

def gethmax(h,norm=True):
    hmax = 0
    hint = h.Integral()
    for b in range(h.GetNbinsX()+1):
        y = h.GetBinContent(b) if(norm==False) else h.GetBinContent(b)/hint
        hmax = y if(y>hmax) else hmax
    return hmax

def get_chip_lines():
    cfg = config.Config().map
    chips = utils.getChips2D()
    chiplines = {}
    for det in cfg["detectors"]:
        chiplines.update({det:ROOT.TPolyLine()})
        corner0 = None
        for icorner,corner in enumerate(chips[det]):
            chiplines[det].SetNextPoint(corner[0],corner[1])
            if(icorner==0): corner0 = corner
            if(icorner==len(chips[det])-1): chiplines[det].SetNextPoint(corner0[0],corner0[1])
        chiplines[det].SetLineColor(ROOT.kGreen+2)
    return chiplines

def book_histos(tfi,tfo,hprefx_glb,hprefx_det,dets):
    tfo.cd()
    ### global histos
    for hist in hprefx_glb:
        print("From file:",tfilenamein,"getting histogram named:",hist)
        name = hist
        histos.update({name:tfi.Get(hist).Clone(name)})
        histos[name].SetDirectory(0)
    ### per detector histos
    for prefx in hprefx_det:
        for idet,det in enumerate(dets):
            hname = prefx+"_"+det
            hist = det+"/"+hname
            name = hname
            print("From file:",tfilenamein,"getting histogram named:",hist)
            histos.update({name:tfi.Get(hist).Clone(name)})
            if(det in histos[name].GetTitle()): histos[name].SetTitle( det )
            histos[name].SetDirectory(0)

def write_histos(tfo):
    tfo.cd()
    for hname,hist in histos.items():
        hist.Write()

def fit1(h,col,xmin,xmax):
    g1 = ROOT.TF1("g1", "gaus", xmin,xmax)
    # f1 = TF1("f1", "gaus(0)", xmin,xmax)
    g1.SetLineColor(col)
    # f1.SetLineColor(col)
    h.Fit(g1,"EMRS")
    # f1.SetParameter(0,g1.GetParameter(0))
    # f1.SetParameter(1,g1.GetParameter(1))
    # f1.SetParameter(2,g1.GetParameter(2))
    # chi2dof = f1.GetChisquare()/f1.GetNDF() if(f1.GetNDF()>0) else -1
    chi2dof = g1.GetChisquare()/g1.GetNDF() if(g1.GetNDF()>0) else -1
    print("g1 chi2/Ndof=",chi2dof)
    return g1

def fit2(h,col):
    g1 = ROOT.TF1("g1", "gaus", xmin,xmax)
    g2 = ROOT.TF1("g2", "gaus", xmin,xmax)
    f1 = ROOT.TF1("f2", "gaus(0)+gaus(3)", xmin,xmax)
    g1.SetLineColor(col)
    g2.SetLineColor(col)
    f2.SetLineColor(col)
    h.Fit(g1,"EMRS")
    h.Fit(g2,"EMRS")
    f2.SetParameter(0,g1.GetParameter(0))
    f2.SetParameter(1,g1.GetParameter(1))
    f2.SetParameter(2,g1.GetParameter(2))
    f2.SetParameter(3,g2.GetParameter(0))
    f2.SetParameter(4,g2.GetParameter(1))
    f2.SetParameter(5,g2.GetParameter(2))
    chi2dof = f2.GetChisquare()/f2.GetNDF() if(f2.GetNDF()>0) else -1
    print("f2 chi2/Ndof=",chi2dof)
    return f2

def fit3(h,col):
    g1 = ROOT.TF1("g1", "gaus", xmin,xmax)
    g2 = ROOT.TF1("g2", "gaus", xmin,xmax)
    g3 = ROOT.TF1("g3", "gaus", xmin,xmax)
    f3 = ROOT.TF1("f3", "gaus(0)+gaus(3)+gaus(6)", xmin,xmax)
    g1.SetLineColor(col)
    g2.SetLineColor(col)
    g3.SetLineColor(col)
    f3.SetLineColor(col)
    h.Fit(g1,"EMRS")
    h.Fit(g2,"EMRS")
    h.Fit(g3,"EMRS")
    f3.SetParameter(0,g1.GetParameter(0))
    f3.SetParameter(1,g1.GetParameter(1))
    f3.SetParameter(2,g1.GetParameter(2))
    f3.SetParameter(3,g2.GetParameter(0))
    f3.SetParameter(4,g2.GetParameter(1))
    f3.SetParameter(5,g2.GetParameter(2))
    f3.SetParameter(6,g3.GetParameter(0))
    f3.SetParameter(7,g3.GetParameter(1))
    f3.SetParameter(8,g3.GetParameter(2))
    chi2dof = f3.GetChisquare()/f3.GetNDF() if(f3.GetNDF()>0) else -1
    print("f3 chi2/Ndof=",chi2dof)
    return f3


# def plot_2x2_1D_histos(pdf,prefix,dets):
#     for idet,det in enumerate(dets):
#         hname = prefix+"_"+det
#         histos[hname].SetLineColor(ROOT.kBlack)
#         histos[hname].SetMarkerColor(ROOT.kBlack)
#         histos[hname].SetMarkerSize(1)
#         histos[hname].SetMarkerStyle(20)
#         integral = histos[hname].Integral()
#         if(integral>0): histos[hname].Scale(1./integral)
#         histos[hname].SetTitle(det)
#         histos[hname].GetYaxis().SetTitle("Normalized")
#
#     cnv = ROOT.TCanvas("cnv","",1200,1000)
#     cnv.Divide(2,2)
#     for count1,det in enumerate(dets):
#         p = cnv.cd(count1+1)
#         p.SetTicks(1,1)
#         # if("cls_size" in prefix): p.SetLogy()
#
#         hname = prefix+"_"+det
#         histos[hname].Draw("e1p")
#     cnv.SaveAs(pdf)


def plot_2x2_FIT_histos(pdf,prefix,dets,xfitmin,xfitmax):
    for idet,det in enumerate(dets):
        hname = prefix+"_"+det
        histos[hname].SetLineColor(ROOT.kBlack)
        histos[hname].SetMarkerColor(ROOT.kBlack)
        histos[hname].SetMarkerSize(1)
        histos[hname].SetMarkerStyle(20)
        integral = histos[hname].Integral()
        if(integral>0): histos[hname].Scale(1./integral)
        histos[hname].SetTitle(det)
        histos[hname].GetYaxis().SetTitle("Normalized")
    
    cnv = ROOT.TCanvas("cnv","",1200,1000)
    cnv.Divide(2,2)
    for count1,det in enumerate(dets):
        p = cnv.cd(count1+1)
        p.SetTicks(1,1)
        # if("cls_size" in prefix): p.SetLogy()
        
        hname = prefix+"_"+det
        histos[hname].Draw("e1p")
        ### fit
        func = fit1(histos[hname],ROOT.kRed,xfitmin,xfitmax)
        s = ROOT.TLatex()
        s.SetNDC(1);
        s.SetTextAlign(13);
        s.SetTextColor(ROOT.kBlack)
        s.SetTextFont(22)
        s.SetTextSize(0.045)
        if("h_Chi2fit_res_trk2cls" in prefix):
            s.DrawLatex(0.17,0.85,ROOT.Form("Mean: %.2f #mum" % (mm2um*func.GetParameter(1))))
            s.DrawLatex(0.17,0.78,ROOT.Form("Sigma: %.2f #mum" % (mm2um*func.GetParameter(2))))
        if("h_response" in prefix):
            s.DrawLatex(0.17,0.85,ROOT.Form("Mean: %.2f" % (func.GetParameter(1))))
            s.DrawLatex(0.17,0.78,ROOT.Form("Sigma: %.2f" % (func.GetParameter(2))))
        if(func.GetNDF()>0): s.DrawLatex(0.2,0.71,ROOT.Form("#chi^{2}/N_{DOF}: %.2f" % (func.GetChisquare()/func.GetNDF())))
        p.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(pdf)

def plot_1D_histos(pdf,hname,logy,cnvx=500,cnvy=500,drawopt="hist",rebin=-1,addtotitle=""):
    cnv = ROOT.TCanvas("cnv","",cnvx,cnvy)
    cnv.SetTicks(1,1)
    if(logy): cnv.SetLogy()
    if(addtotitle!=""): histos[hname].SetTitle(addtotitle)
    if(rebin>0):        histos[hname].Rebin(rebin)
    histos[hname].Draw(drawopt)
    cnv.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(pdf)

def overlay_1D_histos(pdf,hnames,legnd,cols,logy,cnvx=500,cnvy=500,drawopt="hist",rebin=-1,titles=""):
    cnv = ROOT.TCanvas("cnv","",cnvx,cnvy)
    cnv.SetTicks(1,1)
    if(logy): cnv.SetLogy()
    if("_log" in hnames[0]): cnv.SetLogx()
    
    leg = ROOT.TLegend(0.5,0.6,0.8,0.8)
    leg.SetFillStyle(4000) # will be transparent
    leg.SetFillColor(0)
    leg.SetTextFont(42)
    leg.SetBorderSize(0)
    
    hmax = -1
    for hname in hnames:
        if(titles!=""): histos[hname].SetTitle(titles)
        if(rebin>0):    histos[hname].Rebin(rebin)
        y = gethmax(histos[hname],False)
        if(y>hmax): hmax = y
    for ih,hname in enumerate(hnames):
        histos[hname].SetMaximum(hmax*2)
        histos[hname].SetLineColor(cols[ih])
        histos[hname].SetFillColorAlpha(cols[ih],0.4)
        if(ih==0): histos[hname].Draw(drawopt)
        else:      histos[hname].Draw(drawopt+" same")
        if(len(legnd)): leg.AddEntry(histos[hname],legnd[ih],"f")
    if(len(legnd)): leg.Draw("same")
    cnv.RedrawAxis()
    cnv.SaveAs(pdf)
    
def plot_2x2_1D_histos(pdf,prefix,dets,logy,drawopt="hist",addtotitle=""):
    for idet,det in enumerate(dets):
        hname = prefix+"_"+det
        histos[hname].SetLineColor(ROOT.kBlack)
        if(drawopt=="e1p"):
            histos[hname].SetMarkerColor(ROOT.kBlack)
            histos[hname].SetMarkerSize(1)
            histos[hname].SetMarkerStyle(20)
        title = det
        if(addtotitle!=""): title += " "+addtotitle
        histos[hname].SetTitle(title)
    cnv = ROOT.TCanvas("cnv","",1200,1000)
    cnv.Divide(2,2)
    for count1,det in enumerate(dets):
        p = cnv.cd(count1+1)
        p.SetTicks(1,1)
        if(logy): p.SetLogy()
        hname = prefix+"_"+det
        histos[hname].Draw(drawopt)
        p.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(pdf)

def plot_2x2_2D_histos(pdf,prefix,dets,logz,addtotitle=""):
    for idet,det in enumerate(dets):
        hname = prefix+"_"+det
        title = det
        if(addtotitle!=""): title += " "+addtotitle
        histos[hname].SetTitle(title)
    cnv = ROOT.TCanvas("cnv","",1200,1000)
    cnv.Divide(2,2)
    for count1,det in enumerate(dets):
        p = cnv.cd(count1+1)
        p.SetTicks(1,1)
        if(logz): p.SetLogz()
        hname = prefix+"_"+det
        histos[hname].Draw("colz")
        p.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(pdf)


def plot_2x2_2D_realspace_histos(pdf,prefix,dets,logz,addtotitle=""):
    chiplines = get_chip_lines()
    for idet,det in enumerate(dets):
        hname = prefix+"_"+det
        title = det
        if(addtotitle!=""): title += " "+addtotitle
        histos[hname].SetTitle(title)
    cnv = ROOT.TCanvas("cnv","",1200,1000)
    cnv.Divide(2,2)
    for count1,det in enumerate(dets):
        p = cnv.cd(count1+1)
        p.SetTicks(1,1)
        if(logz): p.SetLogz()
        hname = prefix+"_"+det
        histos[hname].Draw("colz")
        chiplines[det].Draw()
        p.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(pdf)



#####################################################################################
#####################################################################################
#####################################################################################
if __name__ == "__main__":
    #############################################
    ### Initialize Config in the main process ###
    config.init_config(configfile, False)
    cfg = config.Config().map
    # config.show_config() # print config once
    print(cfg) # print config once
    #############################################
    
    
    tfilenamein = ""
    if(ismutirun):
        tfilenamein,pklfiles = utils.make_multirun_dir(cfg["inputfile"],cfg["runnums"])
    else:
        tfilenamein = utils.make_run_dirs(cfg["inputfile"])
        tfilenamein = tfilenamein.replace(".root","_multiprocess_histograms.root")
    tfi = ROOT.TFile( tfilenamein,"READ" )
    
    detectors = cfg["detectors"]

    histprefx_glb = ["h_cutflow", "h_nSeeds","h_nSeeds_log","h_nSeeds_mid", "h_nTracks","h_nTracks_log","h_nTracks_mid", "h_nTracks_success","h_nTracks_success_log","h_nTracks_success_mid", "h_nTracks_goodchi2","h_nTracks_goodchi2_log","h_nTracks_goodchi2_mid", "h_nTracks_selected","h_nTracks_selected_log","h_nTracks_selected_mid", "h_3Dchi2err_full", "h_3Dchi2err_all",  "h_3Dchi2err", "h_3Dchi2err_zoom", "h_3Dchi2err_0to1" ]
    histprefx_det = [ "h_errors", "h_pix_occ_1D", "h_pix_occ_1D_masked", "h_pix_occ_2D", "h_pix_occ_2D_masked", "h_cls_occ_2D", "h_cls_occ_2D_masked", "h_trk_occ_2D", "h_cls_size", "h_cls_size_zoom", "h_Chi2fit_res_trk2cls_pass_x", "h_Chi2fit_res_trk2cls_pass_y", "h_response_x", "h_response_y", "h_response_x_vs_csize", "h_response_y_vs_csize" ]
    
    # get the start time
    tfilenameout = tfilenamein.replace(".root","_postprocessplots.root")
    tfo = ROOT.TFile(tfilenameout,"RECREATE")
    book_histos(tfi,tfo,histprefx_glb,histprefx_det,detectors)
    
    pdf = tfilenameout.replace("root","pdf")
    
    ####### plot
    plot_1D_histos(pdf+"(","h_cutflow",logy=True,cnvx=800,cnvy=500,drawopt="hist text0")

    # plot_1D_histos(pdf,    "h_nSeeds",logy=False,cnvx=500,cnvy=500,drawopt="hist text0",rebin=-1)
    # plot_1D_histos(pdf,    "h_nTracks",logy=False,cnvx=500,cnvy=500,drawopt="hist text0",rebin=-1,addtotitle="Successfully fitted tracks")
    # plot_1D_histos(pdf,    "h_nTracks_goodchi2",logy=False,cnvx=500,cnvy=500,drawopt="hist text0",rebin=-1,addtotitle="Good #chi^{2}/N_{DoF} tracks")

    hnames = ["h_nSeeds_log", "h_nTracks_goodchi2_log", "h_nTracks_selected_log"]
    hlegnd = ["Seeds",         "Good #chi^{2} tracks",    "Selected tracks"] 
    cols   = [ROOT.kBlack,  ROOT.kBlue,         ROOT.kRed]
    overlay_1D_histos(pdf, hnames,hlegnd,cols,logy=True,cnvx=500,cnvy=500,drawopt="hist",rebin=-1,titles="Hough transform based seeding & tracking;N per trigger;Triggers")

    plot_1D_histos(pdf, "h_3Dchi2err_all",logy=True,cnvx=500,cnvy=500,drawopt="hist")
    plot_1D_histos(pdf, "h_3Dchi2err_full",logy=True,cnvx=500,cnvy=500,drawopt="hist")
    plot_1D_histos(pdf, "h_3Dchi2err",logy=True,cnvx=500,cnvy=500,drawopt="hist")
    plot_1D_histos(pdf, "h_3Dchi2err_zoom",logy=True,cnvx=500,cnvy=500,drawopt="hist")
    # plot_1D_histos(pdf, "h_3Dchi2err_0to1",logy=True,cnvx=500,cnvy=500,drawopt="hist")
    
    plot_2x2_FIT_histos(pdf,"h_response_x",detectors,-2.,+2.)
    plot_2x2_FIT_histos(pdf,"h_response_y",detectors,-2.,+2.)
    plot_2x2_2D_histos(pdf,"h_response_x_vs_csize",detectors,logz=False)
    plot_2x2_2D_histos(pdf,"h_response_y_vs_csize",detectors,logz=False)
    
    plot_2x2_1D_histos(pdf,"h_errors",detectors,logy=False,drawopt="hist")
    # plot_2x2_1D_histos(pdf,"h_pix_occ_1D",detectors,logy=True,drawopt="hist",addtotitle="unmasked")
    plot_2x2_1D_histos(pdf,"h_pix_occ_1D_masked",detectors,logy=True,drawopt="hist",addtotitle="Pixels")
    # plot_2x2_2D_histos(pdf,"h_pix_occ_2D",detectors,logz=False,addtotitle="unmasked")
    plot_2x2_2D_histos(pdf,"h_pix_occ_2D_masked",detectors,logz=False,addtotitle="Pixels")
    # plot_2x2_2D_histos(pdf,"h_cls_occ_2D",detectors,logz=False,addtotitle="unmasked")
    plot_2x2_2D_histos(pdf,"h_cls_occ_2D_masked",detectors,logz=False,addtotitle="Clusters")

    plot_2x2_2D_realspace_histos(pdf,"h_trk_occ_2D",detectors,logz=False,addtotitle="Tracks")
    
    plot_2x2_1D_histos(pdf,"h_cls_size",detectors,logy=True,drawopt="e1p")
    plot_2x2_1D_histos(pdf,"h_cls_size_zoom",detectors,logy=False,drawopt="e1p")
    
    plot_2x2_FIT_histos(pdf,    "h_Chi2fit_res_trk2cls_pass_x",detectors,-0.012,+0.012)
    plot_2x2_FIT_histos(pdf+")","h_Chi2fit_res_trk2cls_pass_y",detectors,-0.012,+0.012)
    
    ####### save in root file
    write_histos(tfo)
    tfo.Write()
    tfo.Close()

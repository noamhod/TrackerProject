#!/usr/bin/python
import os
import math
import array
import numpy as np
import ROOT

from tracker_lib import config

pad_mrg_bot = ROOT.gStyle.GetPadBottomMargin()
pad_mrg_lft = ROOT.gStyle.GetPadLeftMargin()
pad_mrg_rgt = ROOT.gStyle.GetPadRightMargin()


COUNTERS      = ["Pixels/layer", "Clusters/layer", "Track Seeds", "Good Tracks", "Selected Tracks"]
counters_cols = [ROOT.kBlack,   ROOT.kBlue,      ROOT.kRed,     ROOT.kOrange+1, ROOT.kGreen+2 ]
counters_mrks = [23,            22,              26,            24,             20 ]

counters_x_trg = array.array('d')
counters_y_val = {}

def init_global_counters():
    for counter in COUNTERS: counters_y_val.update({counter:array.array('d')})

def append_global_counters():
    for counter in COUNTERS:
        counters_y_val[counter].append(0)

def set_global_counter(counter,idx,val):
    counters_y_val[counter][idx] = val


def plot_counters(foutpdfname,runnum):
    gmax = -1e10
    gmin = +1e10
    for i,counter in enumerate(COUNTERS):
        mx = max(counters_y_val[counter])
        mn = min(counters_y_val[counter])
        gmax = mx if(mx>gmax) else gmax
        gmin = mn if(mn<gmin) else gmin
    # gmin = gmin if(gmin>0) else 0.5
    gmin = 0.5
    gmax = gmax*5
    print(f"gmin={gmin}, gmax={gmax}")
    
    graphs = {}
    mg = ROOT.TMultiGraph()
    # leg = ROOT.TLegend(0.0,0.1,1.0,0.9)
    # leg = ROOT.TLegend(0.14,0.13,0.88,0.3)
    leg = ROOT.TLegend(0.06,0.86,0.99,1.02)
    leg.SetNColumns(len(COUNTERS))
    leg.SetFillStyle(4000) # will be transparent
    leg.SetFillColor(0)
    leg.SetTextFont(42)
    leg.SetTextSize(0.045)
    leg.SetBorderSize(0)
    for i,counter in enumerate(COUNTERS):
        counter_name = counter.replace("/","_per_")
        gname = f"{counter}_vs_trg"
        graphs.update( {gname:ROOT.TGraph( len(counters_x_trg), counters_x_trg, counters_y_val[counter] )} )
        graphs[gname].SetBit(ROOT.TGraph.kIsSortedX)
        graphs[gname].GetXaxis().SetLimits(counters_x_trg[0]-2,counters_x_trg[-1]+2)
        graphs[gname].SetLineColor(counters_cols[i])
        graphs[gname].SetMarkerColor(counters_cols[i])
        graphs[gname].SetMarkerStyle(counters_mrks[i])
        graphs[gname].SetMarkerSize(0.8)
        graphs[gname].SetMaximum(gmax)
        graphs[gname].SetMinimum(gmin)
        leg.AddEntry(graphs[gname],f"{counter}","lp")
        mg.Add(graphs[gname])


    # cnv = ROOT.TCanvas("cnv_hits_vs_trg_all","",1200,500)
    # pL = ROOT.TPad("left_pad", "", 0.0, 0.0, 0.8, 1.0)
    # pR = ROOT.TPad("right_pad", "", 0.8, 0.0, 1.0, 1.0)
    # pR.SetMargin(0.0, 0.0, 0.0, 0.0)
    # pL.SetRightMargin(0.05)
    # pL.Draw()
    # pR.Draw()
    #
    # pL.cd()
    # pL.SetTicks(1,1)
    # pL.SetGridx()
    # pL.SetGridy()
    # pL.SetLogy()
    # mg.Draw("ap")
    # mg.SetTitle(f";EUDAQ Trigger Number;Multiplicity")
    # mg.SetMaximum(gmax)
    # mg.SetMinimum(gmin)
    # mg.GetXaxis().SetLimits(counters_x_trg[0],counters_x_trg[-1])
    # pL.RedrawAxis()
    #
    # pR.cd()
    # leg.Draw()
    
    ROOT.gStyle.SetPadBottomMargin(0.1)
    ROOT.gStyle.SetPadLeftMargin(0.08)
    ROOT.gStyle.SetPadRightMargin(0.04)
    
    cnv = ROOT.TCanvas("cnv_hits_vs_trg_all","",1000,500)
    cnv.SetTicks(1,1)
    cnv.SetGridx()
    cnv.SetGridy()
    cnv.SetLogy()
    mg.Draw("alp")
    mg.SetTitle(f";EUDAQ Trigger Number;Multiplicity")
    mg.SetMaximum(gmax)
    mg.SetMinimum(gmin)
    mg.GetXaxis().SetLimits(counters_x_trg[0]-2,counters_x_trg[-1]+2)
    
    mg.GetXaxis().SetTitleSize(1.1*mg.GetXaxis().GetTitleSize())
    mg.GetYaxis().SetTitleSize(1.1*mg.GetYaxis().GetTitleSize())
    # mg.GetXaxis().SetLabelSize(1.1*mg.GetXaxis().GetLabelSize())
    # mg.GetYaxis().SetLabelSize(1.1*mg.GetYaxis().GetLabelSize())
    
    cnv.RedrawAxis()
    leg.Draw()
    
    s = ROOT.TLatex()
    s.SetNDC(1)
    s.SetTextAlign(13)
    s.SetTextColor(ROOT.kBlack)
    s.SetTextFont(22)
    s.SetTextSize(0.045)
    s.DrawLatex(0.15,0.85,f"Run {runnum}")
    
    
    cnv.Update()
    cnv.SaveAs(f"{foutpdfname}")
    cnv.SaveAs(f'{foutpdfname.replace(".pdf",".png")}')
    
    ctr = COUNTERS[-1]
    print(f"Avg+/-Std for {ctr}: {np.mean(counters_y_val[ctr])} +/- {np.std(counters_y_val[ctr])}")
    
    ROOT.gStyle.SetPadBottomMargin(pad_mrg_bot)
    ROOT.gStyle.SetPadLeftMargin(pad_mrg_lft)
    ROOT.gStyle.SetPadRightMargin(pad_mrg_rgt)
    
    print("Done plot counters")
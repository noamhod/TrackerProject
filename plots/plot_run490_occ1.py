import os
import numpy as np
import array
import math
import pickle
import ROOT
import ctypes

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

ROOT.gErrorIgnoreLevel = ROOT.kError
# ROOT.gErrorIgnoreLevel = ROOT.kWarning


fpklcfgname = "quads_impact_fist_chip.pkl"
fpkl = open(fpklcfgname,'rb')
data = pickle.load(fpkl)
for name, arr in data.items():
    print(name)
    # print(arr)
fpkl.close()


x_arr = data["1D_m34_26"]
nbins = len(x_arr)
x_min = 0
x_max = nbins+1
sum_arr = np.sum(x_arr)
print(f"nbins={nbins}, sum_arr={sum_arr}")

hist = ROOT.TH1D("hist", "hist", nbins, x_min, x_max)
for i,v in enumerate(x_arr):
    hist.SetBinContent(i+1, v)

# hist.SetLineColor(ROOT.kBlack)
# hist.SetTitle("Data;pixel global number;Fired pixels (not normalized)")
# f1 = ROOT.TF1("f1", "(1-1/(exp((x-[0])/[1]) + 1))*([2] + [3]*x + [4]*x*x)", x_min,x_max)
# # f1 = ROOT.TF1("f1", "[0]*(1-1/(exp((x-[1])/[2]) + 1)) -[3]*x", x_min,x_max)
# f1.SetLineColor(ROOT.kRed)
# # f1.SetParLimits(0,1e3,10e3)
# # f1.SetParLimits(1,1,5e2)
# # f1.SetParLimits(2,0,15)
# # f1.SetParLimits(3,-1,0)
# # f1.SetParLimits(4,-1,0)
# hist.Fit(f1,"EMRS")
# chi2dof = f1.GetChisquare()/f1.GetNDF() if(f1.GetNDF()>0) else -1
# print("chi2/Ndof=",chi2dof)
# c = ROOT.TCanvas("c", "fit", 2500, 600)
# hist.Draw("hist")
# f1.Draw("same")
# c.SaveAs("fit_result.png")
#
#
# c = ROOT.TCanvas("c", "fit", 2500, 600)
# h1 = hist.Clone("h1")
# h1.Reset()
# h1.SetTitle("Toy MC;pixel global number;Fired pixels (not normalized)")
# for i in range(int(sum_arr)):
#     x = f1.GetRandom()
#     h1.Fill(x)
# h1.Draw("hist")
# f1.Draw("same")
# c.SaveAs("toymc_from_pdf.png")



# h1 = data_gen_hist.createHistogram("h1",x)
pix_x_nbins = 1024+1
pix_x_min  = -0.5
pix_x_max  = 1024+0.5
pix_y_nbins = 512+1
pix_y_min  = -0.5
pix_y_max  = 512+0.5
h2toy = ROOT.TH2D("ToyMC","ToyMC;x pixel number;y pixel number;Pixels",pix_x_nbins,pix_x_min,pix_x_max, pix_y_nbins,pix_y_min,pix_y_max)
h2dat = ROOT.TH2D("ToyMC","Data;x pixel number;y pixel number;Pixels",pix_x_nbins,pix_x_min,pix_x_max, pix_y_nbins,pix_y_min,pix_y_max)
for bx in range(1,h2toy.GetNbinsX()+1):
    for by in range(1,h2toy.GetNbinsY()+1):
        bg = h2toy.GetBin(bx,by)
        # h2toy.SetBinContent(bx,by,h1.GetBinContent(bg))
        h2dat.SetBinContent(bx,by,hist.GetBinContent(bg))

for i in range(int(sum_arr)):
    x_val = ctypes.c_double(0)
    y_val = ctypes.c_double(0)
    h2dat.GetRandom2(x_val,y_val)
    h2toy.Fill(x_val,y_val)
        
c = ROOT.TCanvas("c", "fit", 1200, 1500)
c.Divide(1,2)
c.cd(1)
h2dat.Draw("colz")
c.cd(2)
h2toy.Draw("colz")
c.SaveAs("toymc_2D_from_pdf.png")
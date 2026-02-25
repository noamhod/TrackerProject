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
ROOT.gStyle.SetPadLeftMargin(0.11)
ROOT.gStyle.SetPadRightMargin(0.19)
ROOT.gStyle.SetGridColor(ROOT.kGray)
ROOT.gStyle.SetGridWidth(1)

ROOT.gErrorIgnoreLevel = ROOT.kError
# ROOT.gErrorIgnoreLevel = ROOT.kWarning

m34 = 26


def get_h1(h,namesfx=""):
    npix = h.GetNbinsX()*h.GetNbinsY()
    h1 = ROOT.TH1D(f"{h.GetName()}_1D{namesfx}","",npix,0,npix)
    for bx in range(1,h.GetNbinsX()+1):
        for by in range(1,h.GetNbinsY()+1):
            y = h.GetBinContent(bx,by)
            x = h.GetBin(bx,by)
            h1.SetBinContent(x,y)
    return h1


fpklcfgname = "quads_impact_fist_chip.pkl"
fpkl = open(fpklcfgname,'rb')
data = pickle.load(fpkl)
for name,item in data.items():
    print(name)
fpkl.close()
n = data[f"ntrg_m34_{m34}"]
print(f"N={n} triggers for m34={m34}")


fIn = ROOT.TFile("quads_impact.root", "READ")
h2dat = fIn.Get(f"2D_m34_{m34}").Clone(f"h2D_m34_{m34}")
h2dat_integral = h2dat.Integral()
h2dat.Scale(1./n)
h1dat = get_h1(h2dat)
h1dat.SetTitle("Data for m_{34}="+str(m34)+" m;Global pixel number;Fired pixels per trigger")
h2dat.SetTitle("Data for m_{34}="+str(m34)+" m;#it{x} (pixel number);#it{y} (pixel number);Fired pixels per trigger")
h2dat.GetXaxis().SetLabelSize((1.3/1.7)*h2dat.GetXaxis().GetLabelSize())
h2dat.GetYaxis().SetLabelSize((1.3/1.7)*h2dat.GetYaxis().GetLabelSize())
h2dat.GetZaxis().SetLabelSize((1.3/1.7)*h2dat.GetZaxis().GetLabelSize())
h2dat.GetXaxis().SetTitleSize((1.3/1.7)*h2dat.GetXaxis().GetTitleSize())
h2dat.GetYaxis().SetTitleSize((1.3/1.7)*h2dat.GetYaxis().GetTitleSize())
h2dat.GetZaxis().SetTitleSize((1.3/1.7)*h2dat.GetZaxis().GetTitleSize())

x_min = h2dat.GetXaxis().GetXmin()
x_max = h2dat.GetXaxis().GetXmax()
y_min = h2dat.GetYaxis().GetXmin()
y_max = h2dat.GetYaxis().GetXmax()
# f2 = ROOT.TF2("f2", "([0] + [1]*x + [2]*x^2 + [3]*x^3 + [4]*x^4) + ([5] + [6]*y + [7]*y^2 + [8]*y^3 + [9]*y^4)*(1-1/(exp((y-1)/1) + 1))", x_min,x_max, y_min,y_max)
# f2 = ROOT.TF2("f2", "([0] + [1]*x + [2]*x^2 + [3]*x^3 + [4]*x^4) + ([5] + [6]*y + [7]*y^2 + [8]*y^3 + [9]*y^4)*(1-1/(exp((y-0)/1) + 1))", x_min,x_max, y_min,y_max)
# f2 = ROOT.TF2("f2", "([0] + [1]*x + [2]*x^2 + [3]*x^3 + [4]*x^4) + ([5] + [6]*y + [7]*y^2 + [8]*y^3 + [9]*y^4)*(1-1/(exp((y-0)/0.1) + 1))", x_min,x_max, y_min,y_max)
f2 = ROOT.TF2("f2", "(([0] + [1]*x + [2]*x^2 + [3]*x^3 + [4]*x^4) + ([5] + [6]*y + [7]*y^2 + [8]*y^3 + [9]*y^4))*(1-1/(exp((y-0.5)/0.1) + 1))", x_min,x_max, y_min,y_max)
# f2 = ROOT.TF2("f2", "(([0] + [1]*x + [2]*x^2 + [3]*x^3 + [4]*x^4) + ([5] + [6]*y + [7]*y^2 + [8]*y^3 + [9]*y^4))*(1-1/(exp((y-1)/0.1) + 1))", x_min,x_max, y_min,y_max)
f2.SetLineColor(ROOT.kGray)
f2.SetLineWidth(1)
f2.SetLineStyle(ROOT.kDashed)
f2.SetNpx(1024)
f2.SetNpy(512)
h2dat.Fit(f2,"EMRS")
chi2dof = f2.GetChisquare()/f2.GetNDF() if(f2.GetNDF()>0) else -1
print("chi2/Ndof=",chi2dof)


h2toy = h2dat.Clone(h2dat.GetName()+"_toy")
h2toy.Reset()
h2toy.SetTitle("Toy MC (from fit)")
for i in range(int(h2dat_integral)):
    x_val = ctypes.c_double(0)
    y_val = ctypes.c_double(0)
    f2.GetRandom2(x_val,y_val)
    h2toy.Fill(round(x_val.value),round(y_val.value))
h2toy.Scale(1./n)
h1toy = get_h1(h2toy)
h1toy.SetTitle("Toy MC (from fit);Global pixel number;Fired pixels per trigger")

c = ROOT.TCanvas("c1", "fit", 1500, 400)
c.Divide(2,1)
c.cd(1)
h2dat.Draw("colz")
c.cd(2)
h2toy.Draw("colz")
c.SaveAs("fit_result.png")

c = ROOT.TCanvas("c2", "1d", 1500, 400)
c.Divide(2,1)
c.cd(1)
h1dat.Draw("hist")
c.cd(2)
h1toy.Draw("hist")
c.SaveAs("fit_result_1D.png")

fOut = ROOT.TFile("fit.root","RECREATE")
fOut.cd()
h2dat.Write()
h1dat.Write()
h2toy.Write()
h1toy.Write()
f2.Write()
fOut.Close()


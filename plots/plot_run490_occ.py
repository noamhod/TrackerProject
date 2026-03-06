import os
import numpy as np
import array
import math
import pickle
import ROOT

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


x_arr = data["1D_m34_1"]
x_min = 0
x_max = len(x_arr)+1
nbins = len(x_arr)
sum_arr = np.sum(x_arr)
print(f"nbins={nbins}, sum_arr={sum_arr}")


# 1. Define observable
x = ROOT.RooRealVar("x", "Global pixel index", x_min, x_max)
x.setBins(nbins)

# 2. Convert numpy array -> TH1 -> RooDataHist
hist = ROOT.TH1D("hist", "hist", nbins, x_min, x_max)
for i,v in enumerate(x_arr):
    hist.SetBinContent(i+1, v)
    # hist.SetBinError(i+1,math.sqrt(v))
roodata = ROOT.RooDataHist("roodata", "roodata",ROOT.RooArgList(x),hist)

# 3. Define Gaussian component
mean = ROOT.RooRealVar("mean", "mean of gaussian", 40e3, 30e3, 50e3)
sigma = ROOT.RooRealVar("sigma", "width of gaussian", nbins*0.001, nbins*0.025)
# gauss = ROOT.RooGaussian("gauss", "Gaussian component", x, mean, sigma)
gauss = ROOT.RooBreitWigner("gauss", "Gaussian component", x, mean, sigma)

mean1 = ROOT.RooRealVar("mean1", "mean of gaussian1", 0, 10e3, 30e3)
sigma1 = ROOT.RooRealVar("sigma1", "width of gaussian1", nbins*0.001, nbins*0.1)
gauss1 = ROOT.RooGaussian("gauss1", "Gaussian1 component", x, mean1, sigma1)
# gauss1 = ROOT.RooBreitWigner("gauss1", "Gaussian1 component", x, mean, sigma1)



# 4. Define background (choose exponential or polynomial)
# --- exponential ---
tau = ROOT.RooRealVar("tau", "slope", -100.0, 0.0)   # negative slope
expo = ROOT.RooExponential("expo", "Exponential background", x, tau)

# OR:
# --- linear polynomial ---
a1 = ROOT.RooRealVar("a1", "linear term", -1, +1)
poly = ROOT.RooPolynomial("poly", "Linear background", x, ROOT.RooArgList(a1))

# 5. Define yields (normalizations)
nSig = ROOT.RooRealVar("nSig", "yield of signal", hist.Integral()*0.3, hist.Integral())
nSig1 = ROOT.RooRealVar("nSig1", "yield of signal", 0, hist.Integral()*0.5)
nBkg = ROOT.RooRealVar("nBkg", "yield of background", 0, hist.Integral()*0.5)
# nBkg1 = ROOT.RooRealVar("nBkg1", "yield of background", 0, hist.Integral()*0.2)

# 6. Build total model
# model = ROOT.RooAddPdf("model", "signal + background", ROOT.RooArgList(gauss, expo), ROOT.RooArgList(nSig, nBkg))
# model = ROOT.RooAddPdf("model", "signal + background", ROOT.RooArgList(gauss, poly), ROOT.RooArgList(nSig, nBkg))
model = ROOT.RooAddPdf("model", "signal + background", ROOT.RooArgList(gauss, gauss1, poly), ROOT.RooArgList(nSig, nSig1, nBkg))
# model = ROOT.RooAddPdf("model", "signal + background", ROOT.RooArgList(gauss, gauss1, poly, expo), ROOT.RooArgList(nSig, nSig1, nBkg, nBkg1))

# 7. Fit model to data
# fit_result = model.chi2FitTo(roodata, ROOT.RooFit.Save(),ROOT.RooFit.SumW2Error(True)) # use bin errors (from TH1) instead of Poisson
# fit_result = model.fitTo(roodata, ROOT.RooFit.Save(),ROOT.RooFit.Optimize(True),ROOT.RooFit.Warnings(True),ROOT.RooFit.Strategy(2))
fit_result = model.fitTo(roodata, ROOT.RooFit.Minimizer("Minuit2","migrad"), ROOT.RooFit.Strategy(2),ROOT.RooFit.Extended(True),ROOT.RooFit.Warnings(True), ROOT.RooFit.NumCPU(8),ROOT.RooFit.Save(True),ROOT.RooFit.Optimize(True))

# 8. Plot result
frame = x.frame()
roodata.plotOn(frame, ROOT.RooFit.Name("Data"), ROOT.RooFit.DrawOption("HIST"), ROOT.RooFit.DataError(ROOT.RooAbsData.Poisson))
model.plotOn(frame, ROOT.RooFit.Name("FullPDF"))
# model.plotOn(frame, ROOT.RooFit.Name("Exponent"), ROOT.RooFit.Components("expo"), ROOT.RooFit.LineStyle(ROOT.kDashed), ROOT.RooFit.LineColor(ROOT.kRed))
model.plotOn(frame, ROOT.RooFit.Name("Polynomial"), ROOT.RooFit.Components("poly"), ROOT.RooFit.LineStyle(ROOT.kDashed), ROOT.RooFit.LineColor(ROOT.kRed))
model.plotOn(frame, ROOT.RooFit.Name("Gauss"), ROOT.RooFit.Components("gauss"), ROOT.RooFit.LineStyle(ROOT.kDashed), ROOT.RooFit.LineColor(ROOT.kGreen))
model.plotOn(frame, ROOT.RooFit.Name("Gauss1"), ROOT.RooFit.Components("gauss1"), ROOT.RooFit.LineStyle(ROOT.kDashed), ROOT.RooFit.LineColor(ROOT.kCyan))


c = ROOT.TCanvas("c", "fit", 1500, 600)
# c.SetLogy()
frame.Draw()
leg = ROOT.TLegend(0.65, 0.7, 0.88, 0.88)  # (x1,y1,x2,y2) in NDC
leg.SetFillStyle(4000) # will be transparent
leg.SetFillColor(0)
leg.SetTextFont(42)
leg.SetTextSize(0.037)
leg.SetBorderSize(0)
leg.AddEntry(frame.findObject("Data") , "Data", "lep")
leg.AddEntry(frame.findObject("FullPDF"), "Total fit", "l")
# leg.AddEntry(frame.findObject("Exponent") , "Background", "l")
leg.AddEntry(frame.findObject("Polynomial") , "Background", "l")
leg.AddEntry(frame.findObject("Gauss"), "Gaussian", "l")
leg.AddEntry(frame.findObject("Gauss1"), "Gaussian1", "l")
leg.Draw()
# hist.SetLineColor(ROOT.kAzure+1)
# hist.SetLineWidth(1)
# hist.Draw("hist same")
c.SaveAs("fit_result.png")







# 1. Generate a dataset
# Generate a dataset from the PDF with 5000 events
data_gen = model.generate(ROOT.RooArgSet(x), sum_arr)
data_gen_hist = ROOT.RooDataHist("new_data_hist", "binned data", ROOT.RooArgSet(x), data_gen)

# 2. Plot the results
# Create a RooPlot frame for the observable `x`
frame_gen = x.frame(ROOT.RooFit.Title("Toy MC"))

# Plot the data on the frame
# data_gen.plotOn(frame_gen,ROOT.RooFit.Bins(nbins))
data_gen_hist.plotOn(frame_gen,ROOT.RooFit.DrawOption("HIST"), ROOT.RooFit.DataError(ROOT.RooAbsData.Poisson))

# Plot the fitted PDF on the frame
model.plotOn(frame_gen)

# 5. Display the plot
canvas = ROOT.TCanvas("canvas", "canvas", 1500, 600)
frame_gen.SetMaximum(frame.GetMaximum())
frame_gen.Draw()
canvas.SaveAs("toymc_from_pdf.png")


h1 = data_gen_hist.createHistogram("h1",x)
pix_x_nbins = 1024+1
pix_x_min  = -0.5
pix_x_max  = 1024+0.5
pix_y_nbins = 512+1
pix_y_min  = -0.5
pix_y_max  = 512+0.5
h2 = ROOT.TH2D("h_pix_occ_2D",";x;y;Hits",pix_x_nbins,pix_x_min,pix_x_max, pix_y_nbins,pix_y_min,pix_y_max)
for b in range(1,h1.GetNbinsX()+1):
    z = h1.GetBinContent(b)
    h2.SetBinContent(b,z)
c = ROOT.TCanvas("c", "fit", 1000, 600)
h2.Draw("colz")
c.SaveAs("toymc_2D_from_pdf.png")


    


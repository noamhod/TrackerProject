import ROOT


ROOT.gROOT.SetBatch(1)
ROOT.gStyle.SetOptFit(0)
ROOT.gStyle.SetOptStat(0)
# ROOT.gStyle.SetPalette(ROOT.kRust)
# ROOT.gStyle.SetPalette(ROOT.kSolar)
# ROOT.gStyle.SetPalette(ROOT.kInvertedDarkBodyRadiator)
ROOT.gStyle.SetPalette(ROOT.kDarkBodyRadiator)
# ROOT.gStyle.SetPalette(ROOT.kRainbow)
ROOT.gStyle.SetPadTopMargin(0.03)
ROOT.gStyle.SetPadBottomMargin(0.09)
ROOT.gStyle.SetPadLeftMargin(0.12)
ROOT.gStyle.SetPadRightMargin(0.04)


def hactualmax(h):
    hmax = -1e10
    for b in range(1,h.GetNbinsX()+1):
        y = h.GetBinContent(b)
        hmax = y if(y>hmax) else hmax
    return hmax

def h1h2max(h1,h2):
    hmax = -1
    y1 = h1.GetMaximum()
    y2 = h2.GetMaximum()
    hmax = y1 if(y1>y2) else y2
    return hmax

def h1h2min(h1,h2):
    hmin = 1e10
    for b in range(1,h1.GetNbinsX()+1):
        y1 = h1.GetBinContent(b)
        y2 = h2.GetBinContent(b)
        hmin = y1 if(y1<hmin and y1>0) else hmin
        hmin = y2 if(y2<hmin and y2>0) else hmin
    return hmin


fS = ROOT.TFile("data/e320_prototype_beam_Feb2025/runs/run_0000502/tree_Run502_allplots.root","READ")
fB = ROOT.TFile("data/e320_prototype_beam_Feb2025/runs/run_0000503/tree_Run503_allplots.root","READ")
# fM = ROOT.TFile("generator_plots/generator_502.0_Toy_MC_hPz_zoom.root","READ")
fM = ROOT.TFile("/Users/noamtalhod/GitHub/XsuiteSim/Simulation/Real_data/Xsuite.root","READ")
fF = ROOT.TFile("../../Downloads/sasha/fOut_biased.root","READ")


hTrgS = fS.Get("hTriggers").Clone("TriggerS")
hTrgB = fB.Get("hTriggers").Clone("TriggerB")
nTrgS = hTrgS.GetBinContent(2)
nTrgB = hTrgB.GetBinContent(2)
print(f"nTrgS={nTrgS}, nTrgB={nTrgB}")


legR = ROOT.TLegend(0.58,0.72,0.77,0.94)
legR.SetFillStyle(4000) # will be transparent
legR.SetFillColor(0)
# legR.SetTextFont(42)
legR.SetTextFont(132)
legR.SetTextSize(0.037)
legR.SetBorderSize(0)
hS0 = fS.Get("hPf_zoom").Clone("P_0_S")
hB0 = fB.Get("hPf_zoom").Clone("P_0_B")
hM0 = fM.Get("hPz_zoom").Clone("P_0_M")
hF0 = fF.Get("pz_after").Clone("P_0_F")
hP0 = fF.Get("pz_prod_after").Clone("P_0_P")
print(f'fS.Get("grpz")={fS.Get("grpz")}, is NULL?: {(fS.Get("grpz")==None)}')
grpz0 = fS.Get("grpz").Clone("grpz0")
hS0.GetXaxis().SetTitle("p_{z} [GeV]")
hB0.GetXaxis().SetTitle("p_{z} [GeV]")
hM0.GetXaxis().SetTitle("p_{z} [GeV]")
hF0.GetXaxis().SetTitle("p_{z} [GeV]")
hP0.GetXaxis().SetTitle("p_{z} [GeV]")
# hS0.SetLineColor(ROOT.kBlue)
# hS0.SetLineWidth(2)
# hS0.SetFillColorAlpha(ROOT.kBlue,0.35)
hS0.SetMarkerStyle(20)
hS0.SetMarkerSize(1)
hS0.SetMarkerColor(ROOT.kBlack)
hS0.SetLineColor(ROOT.kBlack)

# hB0.SetLineColor(ROOT.kRed)
# hB0.SetLineWidth(2)
# hB0.SetFillColorAlpha(ROOT.kRed,0.35)
hB0.SetMarkerStyle(24)
hB0.SetMarkerSize(1)
hB0.SetMarkerColor(ROOT.kBlack)
hB0.SetLineColor(ROOT.kBlack)

hM0.SetLineColor(ROOT.kGreen+2)
hM0.SetLineStyle(2)
hM0.SetLineWidth(2)
hF0.SetLineColor(ROOT.kViolet-2)
hF0.SetLineWidth(2)
# hF0.SetFillColorAlpha(ROOT.kViolet-2,0.1)
hP0.SetLineColor(ROOT.kOrange+2)
hP0.SetLineStyle(3)
hP0.SetLineWidth(2)
# legR.AddEntry(hS0,"Run 502 (Be window)","f")

grpz0.SetFillColorAlpha(ROOT.kGray+2,0.3)
grpz0.SetLineColorAlpha(ROOT.kGray+2,0.3)
grpz0.SetMarkerStyle(0)

legR.AddEntry(hS0,"Run 502 (Be win.)","pl")
legR.AddEntry(grpz0,"Systematic Uncertainty","f")
legR.AddEntry(hB0,"Run 503 (Dump-only)","pl")
legR.AddEntry(hM0,"Xsuite MC (as Run 502)","f")
legR.AddEntry(hP0,"GEANT4 (e^{+} from Be win.)","f")
legR.AddEntry(hF0,"GEANT4 (all e^{+}, at detector)","f")


logy = True
cnv = ROOT.TCanvas("cnv","",600,500)
cnv.cd()
ROOT.gPad.SetTicks(1,1)
if(logy): ROOT.gPad.SetLogy()
hS = fS.Get("hPf_small").Clone("P_S")
hB = fB.Get("hPf_small").Clone("P_B")
hM = fM.Get("hPz_small").Clone("P_B")
hF = fF.Get("pz_after").Clone("P_F")
hP = fF.Get("pz_prod_after").Clone("P_P")
grpz = fS.Get("grpz")
hS.Scale(1./nTrgS)
grpz.Scale(1./nTrgS)
hB.Scale(1./nTrgB)
hM.Scale(hactualmax(hS)/hactualmax(hM))
hF.Scale(hactualmax(hS)/hactualmax(hF))
hP.Scale(hactualmax(hS)/hactualmax(hP))
hmax = h1h2max(hS,hB)
hmin = h1h2min(hS,hB)
hS.GetYaxis().SetTitle("Tracks/BX")
hB.GetYaxis().SetTitle("Tracks/BX")
hM.GetYaxis().SetTitle("Tracks/BX")
hP.GetYaxis().SetTitle("Tracks/BX")
hF.GetYaxis().SetTitle("Tracks/BX")
hS.GetXaxis().SetTitle("p_{z} [GeV]")
hB.GetXaxis().SetTitle("p_{z} [GeV]")
hM.GetXaxis().SetTitle("p_{z} [GeV]")
hP.GetXaxis().SetTitle("p_{z} [GeV]")
hF.GetXaxis().SetTitle("p_{z} [GeV]")
# hS.SetLineColor(ROOT.kBlue)
# hS.SetLineWidth(2)
# hS.SetFillColorAlpha(ROOT.kBlue,0.35)
hS.SetMarkerStyle(20)
hS.SetMarkerSize(1)
hS.SetMarkerColor(ROOT.kBlack)
hS.SetLineColor(ROOT.kBlack)

grpz.SetFillColorAlpha(ROOT.kGray+2,0.3)
grpz.SetLineColor(ROOT.kGray+2)
grpz.SetMarkerStyle(0)

# hB.SetLineColor(ROOT.kRed)
# hB.SetLineWidth(2)
# hB.SetFillColorAlpha(ROOT.kRed,0.35)
hB.SetMarkerStyle(24)
hB.SetMarkerSize(1)
hB.SetMarkerColor(ROOT.kBlack)
hB.SetLineColor(ROOT.kBlack)

hM.SetLineColor(ROOT.kGreen+2)
hM.SetLineStyle(2)
hM.SetLineWidth(2)
hF.SetLineColor(ROOT.kViolet-2)
hF.SetLineWidth(2)
# hF.SetFillColorAlpha(ROOT.kViolet-2,0.1)
hP.SetLineColor(ROOT.kOrange+2)
hP.SetLineStyle(3)
hP.SetLineWidth(2)
hS.SetMaximum(5*hmax if(logy) else 3*hmax)
hB.SetMaximum(5*hmax if(logy) else 3*hmax)
hM.SetMaximum(5*hmax if(logy) else 3*hmax)
hP.SetMaximum(5*hmax if(logy) else 3*hmax)
hF.SetMaximum(5*hmax if(logy) else 3*hmax)

hS.SetMinimum(0.5*hmin if(logy) else 0)
hB.SetMinimum(0.5*hmin if(logy) else 0)
hM.SetMinimum(0.5*hmin if(logy) else 0)
hP.SetMinimum(0.5*hmin if(logy) else 0)
hF.SetMinimum(0.5*hmin if(logy) else 0)
hF.Draw("hist")
# hS.Draw("hist same")
hS.Draw("ep same")
grpz.Draw("e2 same")
hB.Draw("ep same")
hM.Draw("hist same")
hP.Draw("hist same")
legR.Draw("same")

s = ROOT.TLatex()
s.SetNDC(1)
s.SetTextAlign(13)
s.SetTextColor(ROOT.kGreen+2)
s.SetTextFont(132)
s.SetTextSize(0.03)
s.DrawLatex(0.68,0.70,f"Xsuite e^{{+}} from Be win.")
s.DrawLatex(0.68,0.67,f" #rightarrow Particles (not tracks)")
s.DrawLatex(0.68,0.64,f" #rightarrow No E-loss and MPS")
s.DrawLatex(0.68,0.61,f" #rightarrow Norm to Run 502 peak")

s = ROOT.TLatex()
s.SetNDC(1)
s.SetTextAlign(13)
s.SetTextColor(ROOT.kOrange+2)
s.SetTextFont(132)
s.SetTextSize(0.03)
s.DrawLatex(0.68,0.55,f"GEANT4 e^{{+}} from Be win.")
s.DrawLatex(0.68,0.52,f" #rightarrow Particles (not tracks)")
s.DrawLatex(0.68,0.49,f" #rightarrow Momentum at production")
s.DrawLatex(0.68,0.46,f" #rightarrow Norm to Run 502 peak")

s = ROOT.TLatex()
s.SetNDC(1)
s.SetTextAlign(13)
s.SetTextColor(ROOT.kViolet-2)
s.SetTextFont(132)
s.SetTextSize(0.03)
s.DrawLatex(0.68,0.40,f"GEANT4 all e^{{+}}, at detector")
s.DrawLatex(0.68,0.37,f" #rightarrow Particles (not tracks)")
s.DrawLatex(0.68,0.34,f" #rightarrow Momentum at first layer")
s.DrawLatex(0.68,0.31,f" #rightarrow Norm to Run 502 peak")



ROOT.gPad.RedrawAxis()
cnv.SaveAs("plot_sigbkg_pz.pdf")
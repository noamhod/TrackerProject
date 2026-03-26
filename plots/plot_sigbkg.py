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

def h1h2max(h1,h2=None):
    hmax = -1
    y1 = h1.GetMaximum()
    y2 = h2.GetMaximum() if(h2 is not None) else 0
    hmax = y1 if(y1>y2) else y2
    return hmax

def h1h2min(h1,h2=None):
    hmin = 1e10
    for b in range(1,h1.GetNbinsX()+1):
        y1 = h1.GetBinContent(b)
        y2 = h2.GetBinContent(b) if(h2 is not None) else 1e10
        hmin = y1 if(y1<hmin and y1>0) else hmin
        hmin = y2 if(y2<hmin and y2>0) else hmin
    return hmin

############################################################################################
# basefile = "data/e320_prototype_beam_Feb2025/runs/run_0000502/tree_Run502_allplots.root" ###
# xstfile = "/Users/noamtalhod/GitHub/XsuiteSim/Simulation/Real_data/Xsuite.root"
# doBkg = True
# doGT4 = True
# doXst = True
# labels = {
#     "hS0":"Run 502 (Be win.)",
#     "grpz0":"Systematic Uncertainty",
#     "hB0":"Run 503 (Dump-only)",
#     "hM0":"Xsuite MC (as Run 502)",
#     "hP0":"GEANT4 (e^{+} from Be win.)",
#     "hF0":"GEANT4 (all e^{+}, at detector)"
# }

basefile = "data/e320_prototype_beam_Mar2026/runs/run_0000856/tree_Run856_allplots.root" ###
xstfile  = "plots/generator_plots/generator_856.0_Toy_MC_hPz_zoom.root"
doBkg = False
doGT4 = False
doXst = True
labels = {
    "hS0":"Run 856 (Al Foil)",
    "grpz0":"Systematic Uncertainty",
    "hB0":"Run XXX (Dump-only)",
    "hM0":"Toy MC (as Run 856)",
    "hP0":"GEANT4 (e^{+} from foil.)",
    "hF0":"GEANT4 (all e^{+}, at detector)"
}

outfile  = basefile.replace("_allplots.root","_plot_sigbkg_pz.pdf")
############################################################################################

fS = ROOT.TFile(basefile,"READ")
fB = ROOT.TFile("data/e320_prototype_beam_Feb2025/runs/run_0000503/tree_Run503_allplots.root","READ") if(doBkg) else None
fM = ROOT.TFile(xstfile,"READ") if(doXst) else None
fF = ROOT.TFile("../../Downloads/sasha/fOut_biased.root","READ") if(doGT4) else None


hTrgS = fS.Get("hTriggers").Clone("TriggerS")
hTrgB = fB.Get("hTriggers").Clone("TriggerB") if(doBkg) else None
nTrgS = hTrgS.GetBinContent(2)
nTrgB = hTrgB.GetBinContent(2) if(doBkg) else 0
print(f"nTrgS={nTrgS}, nTrgB={nTrgB}")


legR = ROOT.TLegend(0.58,0.75,0.77,0.94)
legR.SetFillStyle(4000) # will be transparent
legR.SetFillColor(0)
# legR.SetTextFont(42)
legR.SetTextFont(132)
legR.SetTextSize(0.037)
legR.SetBorderSize(0)
hS0 = fS.Get("hPf_zoom").Clone("P_0_S")
hB0 = fB.Get("hPf_zoom").Clone("P_0_B") if(doBkg) else None
hM0 = fM.Get("hPz_zoom").Clone("P_0_M") if(doXst) else None
hF0 = fF.Get("pz_after").Clone("P_0_F") if(doGT4) else None
hP0 = fF.Get("pz_prod_after").Clone("P_0_P") if(doGT4) else None
# print(f'fS.Get("grpz")={fS.Get("grpz")}, is NULL?: {(fS.Get("grpz")==None)}')
grpz0 = fS.Get("grpz").Clone("grpz0")
hS0.GetXaxis().SetTitle("p_{z} [GeV]")
if(doBkg): hB0.GetXaxis().SetTitle("p_{z} [GeV]")
if(doXst): hM0.GetXaxis().SetTitle("p_{z} [GeV]")
if(doGT4): hF0.GetXaxis().SetTitle("p_{z} [GeV]")
if(doGT4): hP0.GetXaxis().SetTitle("p_{z} [GeV]")
# hS0.SetLineColor(ROOT.kBlue)
# hS0.SetLineWidth(2)
# hS0.SetFillColorAlpha(ROOT.kBlue,0.35)
hS0.SetMarkerStyle(20)
hS0.SetMarkerSize(1)
hS0.SetMarkerColor(ROOT.kBlack)
hS0.SetLineColor(ROOT.kBlack)

if(doBkg):
    # hB0.SetLineColor(ROOT.kRed)
    # hB0.SetLineWidth(2)
    # hB0.SetFillColorAlpha(ROOT.kRed,0.35)
    hB0.SetMarkerStyle(24)
    hB0.SetMarkerSize(1)
    hB0.SetMarkerColor(ROOT.kBlack)
    hB0.SetLineColor(ROOT.kBlack)

if(doXst):
    hM0.SetLineColor(ROOT.kGreen+2)
    hM0.SetLineStyle(2)
    hM0.SetLineWidth(2)
if(doGT4):
    hF0.SetLineColor(ROOT.kViolet-2)
    hF0.SetLineWidth(2)
if(doGT4):
    hP0.SetLineColor(ROOT.kOrange+2)
    hP0.SetLineStyle(3)
    hP0.SetLineWidth(2)

grpz0.SetFillColorAlpha(ROOT.kGray+2,0.3)
grpz0.SetLineColorAlpha(ROOT.kGray+2,0.3)
grpz0.SetMarkerStyle(0)

legR.AddEntry(hS0,labels["hS0"],"pl")
legR.AddEntry(grpz0,labels["grpz0"],"f")
if(doBkg): legR.AddEntry(hB0,labels["hB0"],"pl")
if(doXst): legR.AddEntry(hM0,labels["hM0"],"f")
#if(doGT4): legR.AddEntry(hP0,labels["hP0"],"f")
if(doGT4): legR.AddEntry(hF0,labels["hF0"],"f")


logy = True
cnv = ROOT.TCanvas("cnv","",600,500)
cnv.cd()
ROOT.gPad.SetTicks(1,1)
if(logy): ROOT.gPad.SetLogy()
hS = fS.Get("hPf_small").Clone("P_S")
hB = fB.Get("hPf_small").Clone("P_B") if(doBkg) else None
hM = fM.Get("hPz_small").Clone("P_B") if(doXst) else None
hF = fF.Get("pz_after").Clone("P_F")  if(doGT4) else None
hP = fF.Get("pz_prod_after").Clone("P_P") if(doGT4) else None
grpz = fS.Get("grpz")
hS.Scale(1./nTrgS)
grpz.Scale(1./nTrgS)
if(doBkg): hB.Scale(1./nTrgB)
if(doXst): hM.Scale(hactualmax(hS)/hactualmax(hM))
if(doGT4): hF.Scale(hactualmax(hS)/hactualmax(hF))
if(doGT4): hP.Scale(hactualmax(hS)/hactualmax(hP))
hmax = h1h2max(hS,hB)
hmin = h1h2min(hS,hB)
hS.GetYaxis().SetTitle("Tracks/BX")
if(doBkg): hB.GetYaxis().SetTitle("Tracks/BX")
if(doXst): hM.GetYaxis().SetTitle("Tracks/BX")
if(doGT4): hF.GetYaxis().SetTitle("Tracks/BX")
if(doGT4): hP.GetYaxis().SetTitle("Tracks/BX")
hS.GetXaxis().SetTitle("p_{z} [GeV]")
if(doBkg): hB.GetXaxis().SetTitle("p_{z} [GeV]")
if(doXst): hM.GetXaxis().SetTitle("p_{z} [GeV]")
if(doGT4): hF.GetXaxis().SetTitle("p_{z} [GeV]")
if(doGT4): hP.GetXaxis().SetTitle("p_{z} [GeV]")
hS.SetMarkerStyle(20)
hS.SetMarkerSize(1)
hS.SetMarkerColor(ROOT.kBlack)
hS.SetLineColor(ROOT.kBlack)

grpz.SetFillColorAlpha(ROOT.kGray+2,0.3)
grpz.SetLineColor(ROOT.kGray+2)
grpz.SetMarkerStyle(0)

if(doBkg):
    # hB.SetLineColor(ROOT.kRed)
    # hB.SetLineWidth(2)
    # hB.SetFillColorAlpha(ROOT.kRed,0.35)
    hB.SetMarkerStyle(24)
    hB.SetMarkerSize(1)
    hB.SetMarkerColor(ROOT.kBlack)
    hB.SetLineColor(ROOT.kBlack)

if(doXst):
    hM.SetLineColor(ROOT.kGreen+2)
    hM.SetLineStyle(2)
    hM.SetLineWidth(2)
if(doGT4):
    hF.SetLineColor(ROOT.kViolet-2)
    hF.SetLineWidth(2)
    # hF.SetFillColorAlpha(ROOT.kViolet-2,0.1)
if(doGT4):
    hP.SetLineColor(ROOT.kOrange+2)
    hP.SetLineStyle(3)
    hP.SetLineWidth(2)
hS.SetMaximum(5*hmax if(logy) else 3*hmax)
if(doBkg): hB.SetMaximum(5*hmax if(logy) else 3*hmax)
if(doXst): hM.SetMaximum(5*hmax if(logy) else 3*hmax)
if(doGT4): hF.SetMaximum(5*hmax if(logy) else 3*hmax)
if(doGT4): hP.SetMaximum(5*hmax if(logy) else 3*hmax)

hS.SetMinimum(0.5*hmin if(logy) else 0)
if(doBkg): hB.SetMinimum(0.5*hmin if(logy) else 0)
if(doXst): hM.SetMinimum(0.5*hmin if(logy) else 0)
if(doGT4): hF.SetMinimum(0.5*hmin if(logy) else 0)
if(doGT4): hP.SetMinimum(0.5*hmin if(logy) else 0)

hS.Draw("ep")
grpz.Draw("e2 same")
if(doGT4): hF.Draw("hist same")
if(doBkg): hB.Draw("ep same")
if(doXst): hM.Draw("hist same")
hS.Draw("ep same")
# if(doGT4): hP.Draw("hist same")
legR.Draw("same")

if(doXst):
    s = ROOT.TLatex()
    s.SetNDC(1)
    s.SetTextAlign(13)
    s.SetTextColor(ROOT.kGreen+2)
    s.SetTextFont(132)
    s.SetTextSize(0.03)
    s.DrawLatex(0.68,0.70,f"Xsuite e^{{+}}")
    s.DrawLatex(0.68,0.67,f" #rightarrow Particles (not tracks)")
    s.DrawLatex(0.68,0.64,f" #rightarrow No E-loss and MPS")
    s.DrawLatex(0.68,0.61,f" #rightarrow Norm to Run 502 peak")

# s = ROOT.TLatex()
# s.SetNDC(1)
# s.SetTextAlign(13)
# s.SetTextColor(ROOT.kOrange+2)
# s.SetTextFont(132)
# s.SetTextSize(0.03)
# s.DrawLatex(0.68,0.55,f"GEANT4 e^{{+}} from Be win.")
# s.DrawLatex(0.68,0.52,f" #rightarrow Particles (not tracks)")
# s.DrawLatex(0.68,0.49,f" #rightarrow Momentum at production")
# s.DrawLatex(0.68,0.46,f" #rightarrow Norm to Run 502 peak")

# s = ROOT.TLatex()
# s.SetNDC(1)
# s.SetTextAlign(13)
# s.SetTextColor(ROOT.kViolet-2)
# s.SetTextFont(132)
# s.SetTextSize(0.03)
# s.DrawLatex(0.68,0.40,f"GEANT4 all e^{{+}}, at detector")
# s.DrawLatex(0.68,0.37,f" #rightarrow Particles (not tracks)")
# s.DrawLatex(0.68,0.34,f" #rightarrow Momentum at first layer")
# s.DrawLatex(0.68,0.31,f" #rightarrow Norm to Run 502 peak")

if(doGT4):
    s = ROOT.TLatex()
    s.SetNDC(1)
    s.SetTextAlign(13)
    s.SetTextColor(ROOT.kViolet-2)
    s.SetTextFont(132)
    s.SetTextSize(0.03)
    s.DrawLatex(0.68,0.55,f"GEANT4 all e^{{+}}, at detector")
    s.DrawLatex(0.68,0.52,f" #rightarrow Particles (not tracks)")
    s.DrawLatex(0.68,0.49,f" #rightarrow Momentum at first layer")
    s.DrawLatex(0.68,0.46,f" #rightarrow Norm to Run 502 peak")



ROOT.gPad.RedrawAxis()
cnv.SaveAs(outfile)
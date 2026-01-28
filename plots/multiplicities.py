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




fLoose = ROOT.TFile("data/e320_prototype_beam_Feb2025_no_alignment/runs/run_0000502/tree_Run502_allplots.root","READ")
fTight = ROOT.TFile("data/e320_prototype_beam_Feb2025/runs/run_0000502/tree_Run502_allplots.root","READ")



cols       = [ROOT.kBlack, ROOT.kRed, ROOT.kGreen+2]
types      = ["Tunnels", "Seeds", "Tracks"]
suffs      = ["", "_log", "_full", "_mid", "_zoom"]
loosetight = ["_loose", "_tight"]
histos = {}


for t in types:
    for s in suffs:
        name = f"h_n{t}{s}"
        col = cols[ types.index(t) ]
        for lt in loosetight:
            namelt = f"{name}{lt}"
            fRoot = fLoose if("loose" in lt) else fTight
            histos.update({ f"{namelt}":fRoot.Get(name).Clone(f"{namelt}") })
            histos[f"{namelt}"].SetLineColor( col )
            histos[f"{namelt}"].SetFillColorAlpha( col, 0.35 )
            histos[f"{namelt}"].SetLineWidth( 2 )
            mean = histos[f"{namelt}"].GetMean()
            stdv = histos[f"{namelt}"].GetStdDev()
            print(f"{namelt}: mean={mean:.1f}, stdv={stdv:.1f}")

cnv = ROOT.TCanvas("1","",600,500)
cnv.cd()
cnv.SaveAs("plot_multiplicities.pdf(")
del cnv
for lt in loosetight:
    for s in suffs:
        name = f"{s}{lt}"
        print(name)
        cnv = ROOT.TCanvas(f"{name}","",600,500)
        cnv.cd()
        ROOT.gPad.SetTicks(1,1)
        ROOT.gPad.SetLogy()
        if("log" in s): ROOT.gPad.SetLogx()

        hmax1 = h1h2max(histos[f"h_nTunnels{name}"],histos[f"h_nSeeds{name}"])
        hmax2 = h1h2max(histos[f"h_nTunnels{name}"],histos[f"h_nTracks{name}"])
        hmax3 = h1h2max(histos[f"h_nSeeds{name}"],histos[f"h_nTracks{name}"])
        arrmax = [hmax1,hmax2,hmax3]
        hmax = max(arrmax)*1.5
        
        histos[f"h_nTunnels{name}"].SetMaximum(hmax)
        histos[f"h_nSeeds{name}"].SetMaximum(hmax)
        histos[f"h_nTracks{name}"].SetMaximum(hmax)
        
        histos[f"h_nTunnels{name}"].Draw("hist")
        histos[f"h_nSeeds{name}"].Draw("hist same")
        histos[f"h_nTracks{name}"].Draw("hist same")
        ROOT.gPad.RedrawAxis()
        cnv.SaveAs("plot_multiplicities.pdf")
        del cnv
cnv = ROOT.TCanvas("2","",600,500)
cnv.cd()
cnv.SaveAs("plot_multiplicities.pdf)")
del cnv





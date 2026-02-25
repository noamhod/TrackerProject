import ROOT
import os
import numpy as np
import array
import pickle

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


def get_h1(h,namesfx=""):
    npix = h.GetNbinsX()*h.GetNbinsY()
    h1 = ROOT.TH1D(f"{h.GetName()}_1D{namesfx}","",npix,0,npix)
    for bx in range(1,h.GetNbinsX()+1):
        for by in range(1,h.GetNbinsY()+1):
            y = h.GetBinContent(bx,by)
            x = h.GetBin(bx,by)
            h1.SetBinContent(x,y)
    return h1

def neigbours_mask_2D(h):
    for bx in range(1,h.GetNbinsX()+1):
        for by in range(1,h.GetNbinsY()+1):
            z0 = h.GetBinContent(bx,by)
            avg = 0
            ngbrs = 0
            for ix in [-1,+1]:
                if(bx+ix<1 or bx+ix>h.GetNbinsX()+1): continue
                for jy in [-1,+1]:
                    if(by+jy<1 or by+jy>h.GetNbinsY()+1): continue
                    z = h.GetBinContent(bx+ix,by+jy)
                    avg   += z
                    ngbrs += 1
            avg /= ngbrs
            if(z0>avg*5): h.SetBinContent(bx,by,0)

def asarray1(h):
    nx = h.GetNbinsX()
    arr = np.zeros(nx)
    for bx in range(1,nx+1):
        arr[bx-1] = h.GetBinContent(bx)
    return arr

def asarray2(h):
    nx = h.GetNbinsX()
    ny = h.GetNbinsY()
    arr = np.zeros((nx, ny))
    for bx in range(1,nx+1):
        for by in range(1,ny+1):
            arr[bx-1][by-1] = h.GetBinContent(bx,by)
    return arr
            
def dump_to_pickle(histos,fpklname):
    print(f"Dumping into {fpklname}")
    fpkl = open(fpklname,"wb")
    data = {}
    for hname,hist in histos.items():
        if("2D" in hname):
            arr2D = asarray2(hist)
            data.update( {f"{hname}":arr2D} )
        if("1D" in hname):
            arr1D = asarray1(hist)
            data.update( {f"{hname}":arr1D} )
        if("ntrg_" in hname):
            data.update( {f"{hname}":hist} ) # <-- this is jsut a number, not an integer
    pickle.dump(data, fpkl, protocol=pickle.HIGHEST_PROTOCOL) ### dump to pickle
    fpkl.close()
        


def proj(h2,projmax=False,name=""):
    newname = h2.GetName()+"_proj"
    if(name!=""): newname += f"_{name}"
    if(not projmax):
        p = h2.ProjectionX()
        return p
    ### if projmax:
    p = ROOT.TH1D(newname,h2.GetName(),h2.GetNbinsX(),h2.GetXaxis().GetXmin(),h2.GetXaxis().GetXmax())
    for bx in range(1,h2.GetNbinsX()+1):
        ymax = 0
        for by in range(1,h2.GetNbinsY()+1):
            y = h2.GetBinContent(bx,by)
            ymax = y if(y>ymax) else ymax
        p.SetBinContent(bx,ymax)
    p.SetLineColor(ROOT.kRed)
    p.Scale(0.9*h2.GetYaxis().GetXmax()/p.GetMaximum())
    return p


detectors = ["ALPIDE_0","ALPIDE_1","ALPIDE_2","ALPIDE_3","ALPIDE_4"]
# sufxs = [0,        1,         2,        3,         4,       5        ]
# quad1 = [+46.4210, +44.98254, +40.4250, +30.05477, +26.718, +29.86015]
# quad0 = [-30.6770, -27.99380, -20.3800, -11.56000, -3.3710, -6.659000]
# quad2 = [-30.6775, -27.99400, -20.3813, -11.56075, -3.3710, -6.659000]
# m34   = [1,        3,         10,       28,        30,      26       ]

sufxs = [0,        1,         2,        3        ]
quad1 = [+46.4210, +44.98254, +40.4250, +29.86015]
quad0 = [-30.6770, -27.99380, -20.3800, -6.659000]
quad2 = [-30.6775, -27.99400, -20.3813, -6.659000]
m34   = [1,        3,         10,       26       ]


# data = {}
# for det in detectors:
#     for m in m34:
#         data.update( { f"{det}_M34_{m34}_X":np.zeros(1024+1) } )
#         data.update( { f"{det}_M34_{m34}_Y":np.zeros(512+1) } )




files = []
prefix = "data/e320_prototype_beam_Feb2025/runs/run_0000490/beam_quality"
filename = "tree_Run490_trigger_analysis.root"
for sfx in sufxs:
    path = f"{prefix}_{sfx}/{filename}"
    print(f"path={path}")
    if(not os.path.isfile(path)): 
        files.append( None )
        continue
    files.append( ROOT.TFile(path,"READ")  )
print(f"files={files}")



maxima = list(range(6*6))




f = ROOT.TFile("quads_impact.root","RECREATE")
f.cd()




# maxima2 = list(range(6))
maxima2 = list(range(4))
for isfx,sfx in enumerate(sufxs):
    if(files[sfx] is None): continue
    det = "ALPIDE_0"
    h = files[sfx].Get(f"h_pix_occ_2D_{det}")
    h1 = get_h1(h,f"_{isfx}_dummy")
    n = files[sfx].Get("h_ntrgs").GetBinContent(1)
    h1.Scale(1./n)
    maxima2[isfx] = h1.GetMaximumBin()

# cnv = ROOT.TCanvas("cnv","",250,500)
# p1 = ROOT.TPad("p1","p1",0.0,0.0,0.4,1.0)
# p2 = ROOT.TPad("p2","p2",0.4,0.0,1.0,1.0)
# p1.Draw()
# p2.Draw()
# p1.Divide(1,6)
# p2.Divide(1,6)
# for isfx,sfx in enumerate(sufxs):
#     if(files[sfx] is None): continue
#     ipad = isfx+1
#     p1.cd(ipad)
#     s = ROOT.TLatex()
#     s.SetNDC(1)
#     s.SetTextAlign(13)
#     s.SetTextColor(ROOT.kBlack)
#     s.SetTextFont(22)
#     s.SetTextSize(0.11)
#     s.DrawLatex(0.2,0.90,ROOT.Form("Setting #it{M}_{34} to %.2f m:"   % (m34[isfx])))
#     s.DrawLatex(0.2,0.70,ROOT.Form("#bullet Quad_{0} = %.1f kG/m" % (quad0[isfx])))
#     s.DrawLatex(0.2,0.50,ROOT.Form("#bullet Quad_{1} = %.1f kG/m" % (quad1[isfx])))
#     s.DrawLatex(0.2,0.30,ROOT.Form("#bullet Quad_{2} = %.1f kG/m" % (quad2[isfx])))
#
#     p2.cd(ipad)
#     det = "ALPIDE_0"
#     ROOT.gPad.SetTicks(1,1)
#     print(f"file:{sfx}, detector:{det}")
#     h = files[sfx].Get(f"h_pix_occ_2D_{det}").Clone(f"h_{sfx}_pix_occ_2D_{det}")
#     h.SetTitle(f"{det}: average pixel occupancy over ~1000 BXs")
#     h.GetXaxis().SetTitle("#it{x} (pixel number)")
#     h.GetYaxis().SetTitle("#it{y} (pixel number)")
#     h.GetZaxis().SetTitle("Avierage number of fired pixels/BX")
#     ####################
#     ### "cheap mask" ###
#     bmax = maxima2[isfx]
#     h.SetBinContent(bmax,0)
#     ####################
#     bmax  = h.GetMaximumBin()
#     bxmax = array.array('i', [0])
#     bymax = array.array('i', [0])
#     bzmax = array.array('i', [0])
#     h.GetBinXYZ(bmax, bxmax, bymax, bzmax)
#     print(f"{det}: x={h.GetXaxis().GetBinCenter(bxmax[0])}, y={h.GetYaxis().GetBinCenter(bymax[0])}")
#     n = files[sfx].Get("h_ntrgs").GetBinContent(1)
#     h.Scale(1./n)
#     h.GetXaxis().SetTitleSize(1.5*h.GetXaxis().GetTitleSize())
#     h.GetYaxis().SetTitleSize(1.5*h.GetYaxis().GetTitleSize())
#     h.GetZaxis().SetTitleSize(1.5*h.GetZaxis().GetTitleSize())
#     h.DrawCopy("colz")
#     ROOT.gPad.RedrawAxis()
# cnv.Update()
# cnv.SaveAs("quads_impact_fist_chip.pdf")
# cnv.SaveAs("quads_impact_fist_chip.png")

# cnv = ROOT.TCanvas("cnv","",2000,1000)
cnv = ROOT.TCanvas("cnv","",2000,675)
p1 = ROOT.TPad("p1","p1",0.0,0.0,0.2,1.0)
p2 = ROOT.TPad("p2","p2",0.2,0.0,0.5,1.0)
p3 = ROOT.TPad("p3","p3",0.5,0.0,0.7,1.0)
p4 = ROOT.TPad("p4","p4",0.7,0.0,1.0,1.0)
p1.Draw()
p2.Draw()
p3.Draw()
p4.Draw()
# p1.Divide(1,3)
# p2.Divide(1,3)
# p3.Divide(1,3)
# p4.Divide(1,3)
p1.Divide(1,2)
p2.Divide(1,2)
p3.Divide(1,2)
p4.Divide(1,2)

histos = {}

### even cases
ipad = 1
for isfx,sfx in enumerate(sufxs):
    if(isfx%2!=0): continue
    if(files[sfx] is None): continue
    p1.cd(ipad)
    s = ROOT.TLatex()
    s.SetNDC(1)
    s.SetTextAlign(13)
    s.SetTextColor(ROOT.kBlack)
    s.SetTextFont(22)
    s.SetTextSize(0.11)
    s.DrawLatex(0.2,0.90,ROOT.Form("Setting #it{M}_{34} = %.f m:"   % (m34[isfx])))
    s.DrawLatex(0.2,0.70,ROOT.Form("#bullet Quad_{0} = %.1f kG/m" % (quad0[isfx])))
    s.DrawLatex(0.2,0.50,ROOT.Form("#bullet Quad_{1} = %.1f kG/m" % (quad1[isfx])))
    s.DrawLatex(0.2,0.30,ROOT.Form("#bullet Quad_{2} = %.1f kG/m" % (quad2[isfx])))
    
    p2.cd(ipad)
    det = "ALPIDE_0"
    ROOT.gPad.SetTicks(1,1)
    print(f"file:{sfx}, detector:{det}")
    h = files[sfx].Get(f"h_pix_occ_2D_{det}").Clone(f"h_{sfx}_pix_occ_2D_{det}")
    h.SetTitle(f"{det}: average pixel occupancy over ~1000 BXs")
    h.GetXaxis().SetTitle("#it{x} (pixel number)")
    h.GetYaxis().SetTitle("#it{y} (pixel number)")
    h.GetZaxis().SetTitle("Average number of fired pixels/BX")
    ####################
    ### "cheap mask" ###
    # bmax = maxima2[isfx]
    # h.SetBinContent(bmax,0)
    neigbours_mask_2D(h)
    ####################
    bmax  = h.GetMaximumBin()
    bxmax = array.array('i', [0])
    bymax = array.array('i', [0])
    bzmax = array.array('i', [0])
    h.GetBinXYZ(bmax, bxmax, bymax, bzmax)
    print(f"{det}: x={h.GetXaxis().GetBinCenter(bxmax[0])}, y={h.GetYaxis().GetBinCenter(bymax[0])}")
    h.GetXaxis().SetTitleSize(1.7*h.GetXaxis().GetTitleSize())
    h.GetYaxis().SetTitleSize(1.7*h.GetYaxis().GetTitleSize())
    h.GetZaxis().SetTitleSize(1.7*h.GetZaxis().GetTitleSize())
    # h.GetXaxis().SetTitleOffset(1.2)
    # h.GetYaxis().SetTitleOffset(1.2)
    # h.GetZaxis().SetTitleOffset(1.2)
    h.GetXaxis().SetLabelSize(1.7*h.GetXaxis().GetLabelSize())
    h.GetYaxis().SetLabelSize(1.7*h.GetYaxis().GetLabelSize())
    h.GetZaxis().SetLabelSize(1.7*h.GetZaxis().GetLabelSize())
    histos.update({f"2D_m34_{m34[isfx]}":h.Clone(f"2D_m34_{m34[isfx]}")})
    h1 = get_h1(h)
    h1.GetYaxis().SetTitle("Pixels/Trigger w/masking")

    histos.update({f"1D_m34_{m34[isfx]}":h1.Clone(f"1D_m34_{m34[isfx]}")})
    histos.update({f"ntrg_m34_{m34[isfx]}":n})
    
    n = files[sfx].Get("h_ntrgs").GetBinContent(1)
    h.Scale(1./n)
    h.DrawCopy("colz")
    # h1.Scale(1./n)
    # h1.DrawCopy("hist")
    ROOT.gPad.RedrawAxis()
    ipad += 1
    
    integral = 0
    for bx in range(1,h.GetNbinsX()+1):
        for by in range(1,h.GetNbinsY()+1):
            integral += h.GetBinContent(bx,by)
    print(f"m34={m34[isfx]} m: Integral={integral} for n={n} triggers")

### odd cases
ipad = 1
for isfx,sfx in enumerate(sufxs):
    if(isfx%2==0): continue
    if(files[sfx] is None): continue
    p3.cd(ipad)
    s = ROOT.TLatex()
    s.SetNDC(1)
    s.SetTextAlign(13)
    s.SetTextColor(ROOT.kBlack)
    s.SetTextFont(22)
    s.SetTextSize(0.11)
    s.DrawLatex(0.2,0.90,ROOT.Form("Setting #it{M}_{34} = %.f m:"   % (m34[isfx])))
    s.DrawLatex(0.2,0.70,ROOT.Form("#bullet Quad_{0} = %.1f kG/m" % (quad0[isfx])))
    s.DrawLatex(0.2,0.50,ROOT.Form("#bullet Quad_{1} = %.1f kG/m" % (quad1[isfx])))
    s.DrawLatex(0.2,0.30,ROOT.Form("#bullet Quad_{2} = %.1f kG/m" % (quad2[isfx])))
    
    p4.cd(ipad)
    det = "ALPIDE_0"
    ROOT.gPad.SetTicks(1,1)
    print(f"file:{sfx}, detector:{det}")
    h = files[sfx].Get(f"h_pix_occ_2D_{det}").Clone(f"h_{sfx}_pix_occ_2D_{det}")
    h.SetTitle(f"{det}: average pixel occupancy over ~1000 BXs")
    h.GetXaxis().SetTitle("#it{x} (pixel number)")
    h.GetYaxis().SetTitle("#it{y} (pixel number)")
    h.GetZaxis().SetTitle("Average number of fired pixels/BX")
    ####################
    ### "cheap mask" ###
    # bmax = maxima2[isfx]
    # h.SetBinContent(bmax,0)
    neigbours_mask_2D(h)
    ####################
    bmax  = h.GetMaximumBin()
    bxmax = array.array('i', [0])
    bymax = array.array('i', [0])
    bzmax = array.array('i', [0])
    h.GetBinXYZ(bmax, bxmax, bymax, bzmax)
    print(f"{det}: x={h.GetXaxis().GetBinCenter(bxmax[0])}, y={h.GetYaxis().GetBinCenter(bymax[0])}")
    h.GetXaxis().SetTitleSize(1.7*h.GetXaxis().GetTitleSize())
    h.GetYaxis().SetTitleSize(1.7*h.GetYaxis().GetTitleSize())
    h.GetZaxis().SetTitleSize(1.7*h.GetZaxis().GetTitleSize())
    # h.GetXaxis().SetTitleOffset(1.2)
    # h.GetYaxis().SetTitleOffset(1.2)
    h.GetZaxis().SetTitleOffset(1.2)
    h.GetXaxis().SetLabelSize(1.7*h.GetXaxis().GetLabelSize())
    h.GetYaxis().SetLabelSize(1.7*h.GetYaxis().GetLabelSize())
    h.GetZaxis().SetLabelSize(1.7*h.GetZaxis().GetLabelSize())
    histos.update({f"2D_m34_{m34[isfx]}":h.Clone(f"2D_m34_{m34[isfx]}")})
    h1 = get_h1(h)
    h1.GetYaxis().SetTitle("Pixels/Trigger w/masking")
    
    histos.update({f"1D_m34_{m34[isfx]}":h1.Clone(f"1D_m34_{m34[isfx]}")})
    histos.update({f"ntrg_m34_{m34[isfx]}":n})
    
    n = files[sfx].Get("h_ntrgs").GetBinContent(1)
    h.Scale(1./n)
    h.DrawCopy("colz")
    # h1.Scale(1./n)
    # h1.DrawCopy("hist")
    ROOT.gPad.RedrawAxis()
    ipad += 1

    integral = 0
    for bx in range(1,h.GetNbinsX()+1):
        for by in range(1,h.GetNbinsY()+1):
            integral += h.GetBinContent(bx,by)
    print(f"m34={m34[isfx]} m: Integral={integral} for n={n} triggers")

cnv.Update()
cnv.SaveAs("quads_impact_fist_chip.pdf")
cnv.SaveAs("quads_impact_fist_chip.png")






##########################################
##########################################
##########################################
print(histos)
dump_to_pickle(histos,"quads_impact_fist_chip.pkl")
f.cd()
for hname,hist in histos.items():
    if(type(hist)==float): continue
    hist.Write()
f.Write()
f.Close()
quit()
##########################################
##########################################
##########################################








cnv = ROOT.TCanvas("cnv","",3000,1800)
cnv.Divide(6,6)
ipad = 1
for isfx,sfx in enumerate(sufxs):
    if(files[sfx] is None):
        ipad += (len(detectors)+1)
        continue
    if((ipad-1)%6==0):
        cnv.cd(ipad)
        s = ROOT.TLatex()
        s.SetNDC(1)
        s.SetTextAlign(13)
        s.SetTextColor(ROOT.kBlack)
        s.SetTextFont(22)
        s.SetTextSize(0.17)
        s.DrawLatex(0.2,0.91,ROOT.Form("Q_{0}: %.2f kG/m" % (quad0[isfx])))
        s.DrawLatex(0.2,0.69,ROOT.Form("Q_{1}: %.2f kG/m" % (quad1[isfx])))
        s.DrawLatex(0.2,0.47,ROOT.Form("Q_{2}: %.2f kG/m" % (quad2[isfx])))
        s.DrawLatex(0.2,0.25,ROOT.Form("M_{34}: %.2f m"   % (m34[isfx])))
        ipad += 1
    for det in detectors:
        cnv.cd(ipad)
        ROOT.gPad.SetTicks(1,1)
        print(f"file:{sfx}, detector:{det} pad:{ipad}")
        h = files[sfx].Get(f"h_pix_occ_2D_{det}")        
        h1 = get_h1(h)
        n = files[sfx].Get("h_ntrgs").GetBinContent(1)
        h1.Scale(1./n)
        h1.SetTitle(det)
        h1.GetYaxis().SetTitle("Pixels/Trigger w/o masking")
        h1.DrawCopy("hist")
        maxima[ipad-1] = h1.GetMaximumBin()
        ROOT.gPad.RedrawAxis()
        ipad += 1
cnv.Update()
cnv.SaveAs("quads_impact.pdf(")



cnv = ROOT.TCanvas("cnv","",3000,1800)
cnv.Divide(6,6)
ipad = 1
for isfx,sfx in enumerate(sufxs):
    if(files[sfx] is None):
        ipad += (len(detectors)+1)
        continue
    if((ipad-1)%6==0): 
        cnv.cd(ipad)
        s = ROOT.TLatex()
        s.SetNDC(1)
        s.SetTextAlign(13)
        s.SetTextColor(ROOT.kBlack)
        s.SetTextFont(22)
        s.SetTextSize(0.17)
        s.DrawLatex(0.2,0.91,ROOT.Form("Q_{0}: %.2f kG/m" % (quad0[isfx])))
        s.DrawLatex(0.2,0.69,ROOT.Form("Q_{1}: %.2f kG/m" % (quad1[isfx])))
        s.DrawLatex(0.2,0.47,ROOT.Form("Q_{2}: %.2f kG/m" % (quad2[isfx])))
        s.DrawLatex(0.2,0.25,ROOT.Form("M_{34}: %.2f m"   % (m34[isfx])))
        ipad += 1
    for det in detectors:
        cnv.cd(ipad)
        ROOT.gPad.SetTicks(1,1)
        print(f"file:{sfx}, detector:{det} pad:{ipad}")
        h = files[sfx].Get(f"h_pix_occ_2D_{det}")
        ####################
        ### "cheap mask" ###
        bmax = maxima[ipad-1]
        h.SetBinContent(bmax,0)
        ####################
        h1 = get_h1(h)
        n = files[sfx].Get("h_ntrgs").GetBinContent(1)
        h1.Scale(1./n)
        h1.SetTitle(det)
        h1.GetYaxis().SetTitle("Pixels/Trigger w/masking")
        h1.DrawCopy("hist")
        ROOT.gPad.RedrawAxis()
        ipad += 1
        
        f.cd()
        h.Write()
        h1.Write()
        
cnv.Update()
cnv.SaveAs("quads_impact.pdf")



cnv = ROOT.TCanvas("cnv","",3000,1800)
cnv.Divide(6,6)
ipad = 1
proj_fine = []
for isfx,sfx in enumerate(sufxs):
    if(files[sfx] is None):
        ipad += (len(detectors)+1)
        continue
    if((ipad-1)%6==0):
        cnv.cd(ipad)
        s = ROOT.TLatex()
        s.SetNDC(1)
        s.SetTextAlign(13)
        s.SetTextColor(ROOT.kBlack)
        s.SetTextFont(22)
        s.SetTextSize(0.17)
        s.DrawLatex(0.2,0.91,ROOT.Form("Q_{0}: %.2f kG/m" % (quad0[isfx])))
        s.DrawLatex(0.2,0.69,ROOT.Form("Q_{1}: %.2f kG/m" % (quad1[isfx])))
        s.DrawLatex(0.2,0.47,ROOT.Form("Q_{2}: %.2f kG/m" % (quad2[isfx])))
        s.DrawLatex(0.2,0.25,ROOT.Form("M_{34}: %.2f m"   % (m34[isfx])))
        ipad += 1
    for det in detectors:
        cnv.cd(ipad)
        ROOT.gPad.SetTicks(1,1)
        print(f"file:{sfx}, detector:{det} pad:{ipad}")
        h = files[sfx].Get(f"h_pix_occ_2D_{det}").Clone(f"h_{sfx}_pix_occ_2D_{det}")
        h.SetTitle(det)
        h.GetZaxis().SetTitle("Pixels/Trigger")
        ####################
        ### "cheap mask" ###
        bmax = maxima[ipad-1]
        h.SetBinContent(bmax,0)
        ####################
        bmax  = h.GetMaximumBin()
        bxmax = array.array('i', [0])
        bymax = array.array('i', [0])
        bzmax = array.array('i', [0])
        h.GetBinXYZ(bmax, bxmax, bymax, bzmax)
        print(f"{det}: x={h.GetXaxis().GetBinCenter(bxmax[0])}, y={h.GetYaxis().GetBinCenter(bymax[0])}")
        n = files[sfx].Get("h_ntrgs").GetBinContent(1)
        h.Scale(1./n)
        h.DrawCopy("colz")
        proj_fine.append( proj(h,projmax=True,name="fine") )
        proj_fine[len(proj_fine)-1].Draw("hist same")
        ROOT.gPad.RedrawAxis()
        ipad += 1
f.cd()
for p in proj_fine: p.Write()
        
cnv.Update()
cnv.SaveAs("quads_impact.pdf")



cnv = ROOT.TCanvas("cnv","",3000,1800)
cnv.Divide(6,6)
ipad = 1
proj_coarse = []
for isfx,sfx in enumerate(sufxs):
    if(files[sfx] is None):
        ipad += (len(detectors)+1)
        continue
    if((ipad-1)%6==0):
        cnv.cd(ipad)
        s = ROOT.TLatex()
        s.SetNDC(1)
        s.SetTextAlign(13)
        s.SetTextColor(ROOT.kBlack)
        s.SetTextFont(22)
        s.SetTextSize(0.17)
        s.DrawLatex(0.2,0.91,ROOT.Form("Q_{0}: %.2f kG/m" % (quad0[isfx])))
        s.DrawLatex(0.2,0.69,ROOT.Form("Q_{1}: %.2f kG/m" % (quad1[isfx])))
        s.DrawLatex(0.2,0.47,ROOT.Form("Q_{2}: %.2f kG/m" % (quad2[isfx])))
        s.DrawLatex(0.2,0.25,ROOT.Form("M_{34}: %.2f m"   % (m34[isfx])))
        ipad += 1
    for det in detectors:
        cnv.cd(ipad)
        ROOT.gPad.SetTicks(1,1)
        print(f"file:{sfx}, detector:{det} pad:{ipad}")
        h = files[sfx].Get(f"h_pix_occ_2D_{det}").Clone(f"h_{sfx}_pix_occ_2D_{det}")
        h.SetTitle(det)
        h.GetZaxis().SetTitle("Pixels/Trigger")
        ####################
        ### "cheap mask" ###
        bmax = maxima[ipad-1]
        h.SetBinContent(bmax,0)
        ####################
        #### REBIN TH2 #####
        h.Rebin2D(12,12)
        bmax  = h.GetMaximumBin()
        bxmax = array.array('i', [0])
        bymax = array.array('i', [0])
        bzmax = array.array('i', [0])
        h.GetBinXYZ(bmax, bxmax, bymax, bzmax)
        print(f"{det}: x={h.GetXaxis().GetBinCenter(bxmax[0])}, y={h.GetYaxis().GetBinCenter(bymax[0])}")
        ####################
        n = files[sfx].Get("h_ntrgs").GetBinContent(1)
        h.Scale(1./n)
        h.DrawCopy("colz")
        proj_coarse.append( proj(h,projmax=True,name="coarse") )
        proj_coarse[len(proj_coarse)-1].Draw("hist same")
        ROOT.gPad.RedrawAxis()
        ipad += 1
f.cd()
for p in proj_coarse: p.Write()
        
cnv.Update()
cnv.SaveAs("quads_impact.pdf)")

f.Write()
f.Close()
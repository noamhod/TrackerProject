import ROOT
import math
import array
import numpy as np
import pickle
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


X0 = 511            ### x center
Y0 = 187            ### y center
a  = 500            ### long radius
b  = 30             ### short radius  
t  = 4*(np.pi/180.) ### angle wrt the x axis

def tilted_eliptic_RoI_cut(x, y):
    A = (a*math.sin(t))**2 + (b*math.cos(t))**2
    B = 2*(b**2-a**2)*math.sin(t)*math.cos(t)
    C = (a*math.cos(t))**2 + (b*math.sin(t))**2
    D = -2*A*X0 - B*Y0
    E = -B*X0 - 2*C*Y0
    F = A*(X0**2) + B*X0*Y0 + C*(Y0**2) - (a*b)**2
    elipse = A*(x**2) + B*x*y + C*(y**2) + D*x + E*y + F
    if( elipse>0. ): return False
    return True

def create_ellipse_boundary():
    n_points = 1000
    # Parametric angles from 0 to 2π
    theta = np.linspace(0, 2*np.pi, n_points)
    # Parametric equations for rotated ellipse
    x_points = X0 + a * np.cos(theta) * np.cos(t) - b * np.sin(theta) * np.sin(t)
    y_points = Y0 + a * np.cos(theta) * np.sin(t) + b * np.sin(theta) * np.cos(t)
    return x_points, y_points

if __name__ == "__main__":    
    detectors = ["ALPIDE_0","ALPIDE_1","ALPIDE_2","ALPIDE_3","ALPIDE_4"]
    detectorids = [8,6,4,2,0]
    
    fInName = "test_data/e320_prototype_beam_May2025_10-12/runs/run_0000560/beam_quality/tree_Run560_trigger_analysis.root"
    # fInName = "test_data/e320_prototype_beam_May2025_17-19/runs/run_0000696/beam_quality/tree_Run696_trigger_analysis.root"
    fOutName = fInName.replace(".root","_replot.pdf")
    fIn = ROOT.TFile(fInName,"READ")

    ### get the histos
    histos = {}
    for det in detectors:
        name = f"h_pix_occ_2D_{det}"
        histos.update( { name : fIn.Get(name).Clone(f"{name}_clone") } )
        histos[name].SetDirectory(0)

        name_roi = f"h_pix_occ_2D_{det}_roi"
        histos.update( { name_roi : fIn.Get(name).Clone(f"{name}_clone_roi") } )
        histos[name_roi].SetDirectory(0)
        for ix in range(1,histos[name_roi].GetNbinsX()+1):
            for iy in range(1,histos[name_roi].GetNbinsY()+1):
                x = ix-1
                y = iy-1
                if(not tilted_eliptic_RoI_cut(x,y)):
                    histos[name_roi].SetBinContent(ix,iy,0)
    
    NTRG = fIn.Get("h_ntrgs").GetBinContent(1)
    print(f"NTRG={NTRG}")
    
    
    ### the ellipse line
    x_points, y_points = create_ellipse_boundary()
    # Create TPolyLine
    n_points = len(x_points)
    polyline = ROOT.TPolyLine(n_points)
    # Fill the polyline with points
    for i in range(n_points): polyline.SetPoint(i, x_points[i], y_points[i])
    # Close the ellipse by connecting last point to first
    polyline.SetPoint(n_points-1, x_points[0], y_points[0])
    # Set line properties
    polyline.SetLineColor(ROOT.kBlack)
    polyline.SetLineWidth(1)
    polyline.SetLineStyle(2)
    
    ### used to generate high def png images
    ROOT.gStyle.SetImageScaling(3.)
    
    ### plot
    cnv = ROOT.TCanvas("cnv","",800,2200)
    cnv.Divide(1,5)
    for idet,det in enumerate(detectors):
        cnv.cd(idet+1)
        ROOT.gPad.SetTicks(1,1)
        # for i in range(3): histos[f"h_pix_occ_2D_{det}"].Smooth()
        histos[f"h_pix_occ_2D_{det}"].SetTitle(f"{det};x [pixels];y [pixels];Pixels/Trigger")
        histos[f"h_pix_occ_2D_{det}"].Scale(1./NTRG)
        histos[f"h_pix_occ_2D_{det}"].Draw("colz")
        polyline.Draw("same")
        ROOT.gPad.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f'{fOutName.replace(".pdf",".png")}')
    cnv.SaveAs(f"{fOutName}(")
    
    ### plot just ROI
    cnv = ROOT.TCanvas("cnv","",800,2200)
    cnv.Divide(1,5)
    for idet,det in enumerate(detectors):
        cnv.cd(idet+1)
        ROOT.gPad.SetTicks(1,1)
        # for i in range(3): histos[f"h_pix_occ_2D_{det}_roi"].Smooth()
        histos[f"h_pix_occ_2D_{det}_roi"].SetTitle(f"{det};x [pixels];y [pixels];Pixels/Trigger")
        histos[f"h_pix_occ_2D_{det}_roi"].Scale(1./NTRG)
        histos[f"h_pix_occ_2D_{det}_roi"].Draw("colz")
        ROOT.gPad.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{fOutName})")
    



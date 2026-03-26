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


X0 = 335            ### x center
Y0 = 80             ### y center
a  = 500            ### long radius
b  = 35             ### short radius  
t  = 4*(np.pi/180.) ### angle wrt the x axis
c  = 300            ### NEW: Opening curvature. Smaller c = wider "wings". (Physically similar to the Rayleigh length or beta*)

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



def tilted_butterfly_RoI_cut(x, y, cutoff=False):
    dx = x - X0
    dy = y - Y0
    ### rotate to align with beam axis (x', y') where x' is along the beam, y' is perpendicular (waist)
    x_prime =  dx * math.cos(t) + dy * math.sin(t)
    y_prime = -dx * math.sin(t) + dy * math.cos(t)
    ### check length cutoff (bounded by 'a')
    if(cutoff and abs(x_prime)>a): return False
    ### check hyperbolic width, with the equation: y' < b * sqrt(1 + (x'/c)^2)
    boundary_y = b * math.sqrt(1 + (x_prime/c)**2)
    if(abs(y_prime)>boundary_y):
        return False
    return True


def create_butterfly_lines(h,Npoints=1000):
    ### Solves the tilted hyperbola equation analytically for y given x.
    ### Returns two tuples of arrays: (x, y_upper) and (x, y_lower)
    # Define x range exactly covering the histogram width
    xmin = h.GetXaxis().GetXmin()
    xmax = h.GetXaxis().GetXmax()
    x_arr = np.linspace(xmin,xmax,Npoints)
    ### Precompute constants for the quadratic solution A*dy^2 + B*dy + C = 0, where dy = y - Y0
    S = math.sin(t)
    C = math.cos(t)
    K = (b/c)**2
    ### quadratic coeff A (constant for all x), derived from: (dy*C - dx*S)^2 = b^2 + K*(dx*C + dy*S)^2
    Aq = C**2 - K * S**2
    y_upper = []
    y_lower = []
    for x in x_arr:
        dx = x - X0
        ### quadratic coeffs B and C (depend on x)
        Bq = -2 * dx * S * C * (1 + K)
        Cq = (dx**2) * (S**2 - K * C**2) - b**2
        ### discriminant
        delta = Bq**2 - 4 * Aq * Cq
        if delta >= 0:
            dy1 = (-Bq + math.sqrt(delta)) / (2 * Aq)
            dy2 = (-Bq - math.sqrt(delta)) / (2 * Aq)
            ### sort so we know which is top/bottom
            y_upper.append(Y0 + max(dy1, dy2))
            y_lower.append(Y0 + min(dy1, dy2))
        else:
            ### should not happen for a hyperbola that spans x, but handles cases where beam doesn't exist
            y_upper.append(Y0)
            y_lower.append(Y0)
    return (x_arr, np.array(y_upper)), (x_arr, np.array(y_lower))



if __name__ == "__main__":    
    detectors = ["ALPIDE_0","ALPIDE_1","ALPIDE_2","ALPIDE_3","ALPIDE_4"]
    detectorids = [8,6,4,2,0]
    
    # fInName = "data/e320_prototype_beam_Nov2025/runs/run_0000729/tree_Run729_multiprocess_histograms_notrk.root"
    # fInName = "data/e320_prototype_beam_Mar2026/runs/run_0000872/tree_Run872_multiprocess_histograms_notrk.root"
    fInName = "data/e320_prototype_beam_Mar2026/runs/run_0000872/beam_quality/tree_Run872_trigger_analysis.root"
    fOutName = fInName.replace(".root","_replot.pdf")
    fIn = ROOT.TFile(fInName,"READ")

    ### get the histos
    histos = {}
    for det in detectors:
        name = f"h_pix_occ_2D_{det}"
        if("multiprocess_histograms_notrk" in fInName): histos.update( { name : fIn.Get(f"{det}/{name}").Clone(f"{name}_clone") } )
        else:                                           histos.update( { name : fIn.Get(f"{name}").Clone(f"{name}_clone") } )
        histos[name].SetDirectory(0)

        name_roi = f"h_pix_occ_2D_{det}_roi"
        if("multiprocess_histograms_notrk" in fInName): histos.update( { name_roi : fIn.Get(f"{det}/{name}").Clone(f"{name}_clone_roi") } )
        else:                                           histos.update( { name_roi : fIn.Get(f"{name}").Clone(f"{name}_clone_roi") } )
        histos[name_roi].SetDirectory(0)
        for ix in range(1,histos[name_roi].GetNbinsX()+1):
            for iy in range(1,histos[name_roi].GetNbinsY()+1):
                x = ix-1
                y = iy-1
                # if(not tilted_eliptic_RoI_cut(x,y)):
                if(not tilted_butterfly_RoI_cut(x,y)):
                    histos[name_roi].SetBinContent(ix,iy,0)
    
    NTRG = -1
    if("multiprocess_histograms_notrk" in fInName): NTRG = fIn.Get("h_cutflow").GetBinContent(2) ## BeamQC
    else:                                           NTRG = fIn.Get("h_ntrgs").GetBinContent(1)
    print(f"NTRG={NTRG}")
    
    
    # ### the ellipse line
    # # x_points, y_points = create_ellipse_boundary()
    # # Create TPolyLine
    # n_points = len(x_points)
    # polyline = ROOT.TPolyLine(n_points)
    # # Fill the polyline with points
    # for i in range(n_points): polyline.SetPoint(i, x_points[i], y_points[i])
    # # Close the ellipse by connecting last point to first
    # polyline.SetPoint(n_points-1, x_points[0], y_points[0])
    # # Set line properties
    # polyline.SetLineColor(ROOT.kGreen)
    # polyline.SetLineWidth(1)
    # polyline.SetLineStyle(2)
    
    
    ### Create TWO separate PolyLines (Top and Bottom)
    (x_up, y_up), (x_low, y_low) = create_butterfly_lines( histos[f"h_pix_occ_2D_{detectors[0]}"] )
    n_points = len(x_up)
    
    # Upper Line
    line_up = ROOT.TPolyLine(n_points)
    for i in range(n_points): line_up.SetPoint(i, x_up[i], y_up[i])
    line_up.SetLineColor(ROOT.kGreen)
    line_up.SetLineWidth(1)
    line_up.SetLineStyle(2)
    # Lower Line
    line_low = ROOT.TPolyLine(n_points)
    for i in range(n_points): line_low.SetPoint(i, x_low[i], y_low[i])
    line_low.SetLineColor(ROOT.kGreen)
    line_low.SetLineWidth(1)
    line_low.SetLineStyle(2)
    
    ### used to generate high def png images
    ROOT.gStyle.SetImageScaling(3.)
    
    ### plot
    cnv = ROOT.TCanvas("cnv1","",800,2200)
    cnv.Divide(1,5)
    for idet,det in enumerate(detectors):
        cnv.cd(idet+1)
        ROOT.gPad.SetTicks(1,1)
        # for i in range(3): histos[f"h_pix_occ_2D_{det}"].Smooth()
        histos[f"h_pix_occ_2D_{det}"].SetTitle(f"{det};x [pixels];y [pixels];Pixels/Trigger")
        histos[f"h_pix_occ_2D_{det}"].Scale(1./NTRG)
        histos[f"h_pix_occ_2D_{det}"].Draw("colz")
        # polyline.Draw("same")
        line_low.Draw("same")
        line_up.Draw("same")
        ROOT.gPad.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f'{fOutName.replace(".pdf",".png")}')
    cnv.SaveAs(f"{fOutName}(")
    
    ### plot just ROI
    cnv = ROOT.TCanvas("cnv2","",800,2200)
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
    cnv.SaveAs(f'{fOutName.replace(".pdf","_roi.png")}')
    cnv.SaveAs(f"{fOutName})")
    



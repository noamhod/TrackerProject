#!/usr/bin/python
import os
import math
import array
import numpy as np
import ROOT
from collections import defaultdict

from tracker_lib import config, utils
d2r = np.pi/180.


def tilted_butterfly_PixLevel_cut(x,y):
    cfg = config.Config().map
    X0 = cfg["cut_RoI_btrfly_xcenter"]       ### x center
    Y0 = cfg["cut_RoI_btrfly_ycenter"]       ### y center
    a  = cfg["cut_RoI_btrfly_long_radius"]   ### long radius
    b  = cfg["cut_RoI_btrfly_shrt_radius"]   ### short radius  
    t  = cfg["cut_RoI_btrfly_theta_deg"]*d2r ### angle wrt the x axis
    c  = cfg["cut_RoI_btrfly_theta_curv"]    ### opening curvature. Smaller c = wider "wings". (Physically similar to the Rayleigh length or beta*)
    ### start calculate
    dx = x - X0
    dy = y - Y0
    ### rotate to align with beam axis (x', y') where x' is along the beam, y' is perpendicular (waist)
    x_prime =  dx * math.cos(t) + dy * math.sin(t)
    y_prime = -dx * math.sin(t) + dy * math.cos(t)
    ### check hyperbolic width, with the equation: y' < b * sqrt(1 + (x'/c)^2)
    boundary_y = b * math.sqrt(1 + (x_prime/c)**2)
    if(abs(y_prime)>boundary_y): return False
    return True


def tilted_butterfly_RoI_cut(track):
    cfg = config.Config().map
    X0 = cfg["cut_RoI_btrfly_xcenter"]       ### x center
    Y0 = cfg["cut_RoI_btrfly_ycenter"]       ### y center
    a  = cfg["cut_RoI_btrfly_long_radius"]   ### long radius
    b  = cfg["cut_RoI_btrfly_shrt_radius"]   ### short radius  
    t  = cfg["cut_RoI_btrfly_theta_deg"]*d2r ### angle wrt the x axis
    c  = cfg["cut_RoI_btrfly_theta_curv"]    ### opening curvature. Smaller c = wider "wings". (Physically similar to the Rayleigh length or beta*)
    ### start calculate
    for det,cluster in track.trkcls.items():
        dx = cluster.x - X0
        dy = cluster.y - Y0
        ### rotate to align with beam axis (x', y') where x' is along the beam, y' is perpendicular (waist)
        x_prime =  dx * math.cos(t) + dy * math.sin(t)
        y_prime = -dx * math.sin(t) + dy * math.cos(t)
        ### check hyperbolic width, with the equation: y' < b * sqrt(1 + (x'/c)^2)
        boundary_y = b * math.sqrt(1 + (x_prime/c)**2)
        # print(f"pix xy={pix.x,pix.y}, boundary_y={boundary_y}, y_prime={y_prime}")
        if(abs(y_prime)>boundary_y):
            # print("fail")
            return False
    return True



def tilted_eliptic_RoI_cut(track):
    cfg = config.Config().map
    X0 = cfg["cut_RoI_spot_xcenter"]
    Y0 = cfg["cut_RoI_spot_ycenter"]
    a = cfg["cut_RoI_spot_radius_x"]
    b = cfg["cut_RoI_spot_radius_y"]
    t = cfg["cut_RoI_spot_theta_deg"]*np.pi/180.
    A = (a*math.sin(t))**2 + (b*math.cos(t))**2
    B = 2*(b**2-a**2)*math.sin(t)*math.cos(t)
    C = (a*math.cos(t))**2 + (b*math.sin(t))**2
    D = -2*A*X0 - B*Y0
    E = -B*X0 - 2*C*Y0
    F = A*(X0**2) + B*X0*Y0 + C*(Y0**2) - (a*b)**2
    for det in cfg["detectors"]:
        x = track.trkcls[det].x ### cluster center measured in pixels in the EUDAQ frame
        y = track.trkcls[det].y ### cluster center measured in pixels in the EUDAQ frame 
        elipse = A*(x**2) + B*x*y + C*(y**2) + D*x + E*y + F
        if(elipse>0.): return False
    return True

def spot_cut(x,y):
    cfg = config.Config().map
    CX = cfg["cut_spot_xcenter"]
    CY = cfg["cut_spot_ycenter"]
    RX = cfg["cut_spot_radius_x"]
    RY = cfg["cut_spot_radius_y"]
    X = (x-CX)/RX
    Y = (y-CY)/RY
    X2 = X*X
    Y2 = Y*Y
    if( (X2+Y2)>1. ): return False
    return True

def strip_cut(x,y):
    cfg = config.Config().map
    CX = cfg["cut_strip_xcenter"]
    CY = cfg["cut_strip_ycenter"]
    SX = cfg["cut_strip_xwidth"]
    SY = cfg["cut_strip_ywidth"]
    if( x<CX-SX or x>CX+SX ): return False
    if( y<CY-SY or y>CY+SY ): return False
    return True
    

def pass_dk_at_detector(track,dxrange=[-999,+999],dyrange=[-999,+999]):
    cfg = config.Config().map
    if(not cfg["cut_dk_algn"]): return True
    det = cfg["cut_dk_algn_det"]
    dx,dy = utils.res_track2cluster(det,track.detectors,track.points,track.direction,track.centroid)
    if(dx<cfg["cut_dk_algn_dxmin"] or dx>cfg["cut_dk_algn_dxmax"]): return False
    if(dy<cfg["cut_dk_algn_dymin"] or dy>cfg["cut_dk_algn_dymax"]): return False
    return True




def pass_geoacc_selection(track,ismultiproc=False):
    cfg = config.Config().map
    ## r0: first detector, rN: last detector, rW: window, rD: dipole exit
    r0,rN,rW,rF,rD = utils.get_track_point_at_extremes(track,ismultiproc)
    xWinL,xWinR,yWinB,yWinT = utils.get_pdc_window_bounds()
    xFlgL,xFlgR,yFlgB,yFlgT = utils.get_dipole_flange_bounds()
    xDipL,xDipR,yDipB,yDipT = utils.get_dipole_exit_bounds()
    ### the cuts
    psss_RoI             = ( tilted_eliptic_RoI_cut(track) )                                       if(cfg["cut_RoI_spot"])                else True
    pass_inclination_yz  = ( rN[1]>=r0[1]  and r0[1]>=rW[1]  and rN[1]>=rW[1] )                    if(cfg["cut_allow_negative_yz_slope"]) else True
    pass_window_aperture = ( (rW[0]>=xWinL and rW[0]<=xWinR) and (rW[1]>=yWinB and rW[1]<=yWinT) ) if(cfg["cut_windowaprtr"])             else True 
    pass_flange_aperture = ( (rF[0]>=xFlgL and rF[0]<=xFlgR) and (rF[1]>yFlgB and rF[1]<=yFlgT) )  if(cfg["cut_flangeaprtr"])             else True
    pass_dipole_aperture = ( (rD[0]>=xDipL and rD[0]<=xDipR) and (rD[1]>yDipB and rD[1]<=yDipT) )  if(cfg["cut_dipoleaprtr"])             else True
    pass_dipole_spot     = ( spot_cut(rD[0],rD[1])  )                                              if(cfg["cut_spot"])                    else True
    pass_dipole_strip    = ( strip_cut(rD[0],rD[1]) )                                              if(cfg["cut_strip"])                   else True
    pass_dk_at_det       = ( pass_dk_at_detector(track) )                                          if(cfg["cut_dk_algn"])                 else True
    if(cfg["dbg"]): print(f"psss_RoI={psss_RoI}, pass_inclination_yz={pass_inclination_yz}, pass_window_aperture={pass_window_aperture}, pass_flange_aperture={pass_flange_aperture}, pass_dipole_aperture={pass_dipole_aperture}, pass_dipole_spot={pass_dipole_spot}, pass_dipole_strip={pass_dipole_strip}, pass_dk_at_det={pass_dk_at_det}")
    ### final decision
    return (psss_RoI and
            pass_inclination_yz and
            pass_window_aperture and
            pass_flange_aperture and
            pass_dipole_aperture and
            pass_dipole_spot and
            pass_dipole_strip and
            pass_dk_at_det)




def remove_tracks_with_shared_clusters(tracks):
    cfg = config.Config().map
    clsid_to_trackidx = {}
    for det in cfg["detectors"]: clsid_to_trackidx.update({det:{}})
    for itrk,track in enumerate(tracks):
        for det in track.detectors:
            CID = track.trkcls[det].CID
            if(CID not in clsid_to_trackidx[det]):
                clsid_to_trackidx[det].update({CID:itrk})
            else:
                itrk0 = clsid_to_trackidx[det][CID]
                # print(f"found shared cluster for CID={CID}: itrk1={itrk}(chi2={track.chi2ndof}), itrk2={itrk0}(chi2={tracks[itrk0].chi2ndof})")
                if(tracks[itrk0].chi2ndof>track.chi2ndof):
                    clsid_to_trackidx[det][CID] = itrk

    # ## TODO: this block has to be adapted to the tandem layers concept!!!
    # passing_tracks_idx = []
    # passing_tracks     = []
    # det0 = cfg["detectors"][0]
    # for CID,itrk in clsid_to_trackidx[det0].items():
    #     if(itrk not in passing_tracks_idx):
    #         noccurancees = 1
    #         for i in range(1,len(cfg["detectors"])):
    #             deti = cfg["detectors"][i]
    #             noccurancees += (itrk in clsid_to_trackidx[deti].values())
    #         if(noccurancees!=len(cfg["detectors"])): continue
    #         passing_tracks_idx.append(itrk)
    #         passing_tracks.append(tracks[itrk])
    
    
    passing_tracks_idx = []
    passing_tracks     = []
    for det in cfg["tandemlyrs"][0]: ### track must be in the one of the first tandem layer's chips
        for CID,itrk in clsid_to_trackidx[det].items():
            if(itrk not in passing_tracks_idx):
                passing_tracks_idx.append(itrk)
                passing_tracks.append(tracks[itrk])
    
    return passing_tracks
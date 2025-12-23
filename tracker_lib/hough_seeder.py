#!/usr/bin/python
import os
import math
import subprocess
import array
import numpy as np
from collections import defaultdict
import ROOT

from tracker_lib import config, objects, lookup_table, utils

### based largely on this: https://www.cs.ubc.ca/~lsigal/425_2018W2/Lecture17.pdf
### see also https://www.sciencedirect.com/science/article/pii/S0167865500000441?via%3Dihub

def fwave(theta,k,z):
    rho = k*math.sin(theta) + z*math.cos(theta)
    return rho
    
def fdiff(theta,k1,z1,k2,z2):
    return fwave(theta,k1,z1)-fwave(theta,k2,z2)


class HoughSeeder:
    def __init__(self,clusters,eventid=0):
        cfg = config.Config().map
        ispreproc = utils.is_preprocessed()
        
        ### for not having memory leaks with the TH2D
        self.eventid = eventid
        self.is5lyr = (len(cfg["detectors"])>4)

        ### make the tandem layers
        self.tdmlyr_clusters = np.empty(cfg["layers"], dtype=object)
        for lyr in range(cfg["layers"]): self.tdmlyr_clusters[lyr] = []
        self.set_tdmlyr_clusters(clusters)
        
        nclusters = 0
        for det in cfg["detectors"]: nclusters += len(clusters[det])
        nclusters = int(nclusters/cfg["layers"])
        
        ### other constants
        self.xepsilon = 1e-15
        self.fepsilon = 1e-15
        
        self.theta_x_range = [0,0]
        self.rho_x_range   = [0,0]
        self.theta_y_range = [0,0]
        self.rho_y_range   = [0,0]
        if(nclusters<=cfg["cls_mult_low"]):
            self.theta_x_range = [ cfg["seed_thetax_range_low"][0], cfg["seed_thetax_range_low"][1] ]
            self.rho_x_range   = [ cfg["seed_rhox_range_low"][0],   cfg["seed_rhox_range_low"][1] ]
            self.theta_y_range = [ cfg["seed_thetay_range_low"][0], cfg["seed_thetay_range_low"][1] ]
            self.rho_y_range   = [ cfg["seed_rhoy_range_low"][0],   cfg["seed_rhoy_range_low"][1] ]
        elif(nclusters>cfg["cls_mult_low"]  and nclusters<=cfg["cls_mult_mid"]):
            self.theta_x_range = [ cfg["seed_thetax_range_mid"][0], cfg["seed_thetax_range_mid"][1] ]
            self.rho_x_range   = [ cfg["seed_rhox_range_mid"][0],   cfg["seed_rhox_range_mid"][1] ]
            self.theta_y_range = [ cfg["seed_thetay_range_mid"][0], cfg["seed_thetay_range_mid"][1] ]
            self.rho_y_range   = [ cfg["seed_rhoy_range_mid"][0],   cfg["seed_rhoy_range_mid"][1] ]
        elif(nclusters>cfg["cls_mult_mid"] and nclusters<=cfg["cls_mult_hgh"]):
            self.theta_x_range = [ cfg["seed_thetax_range_hgh"][0], cfg["seed_thetax_range_hgh"][1] ]
            self.rho_x_range   = [ cfg["seed_rhox_range_hgh"][0],   cfg["seed_rhox_range_hgh"][1] ]
            self.theta_y_range = [ cfg["seed_thetay_range_hgh"][0], cfg["seed_thetay_range_hgh"][1] ]
            self.rho_y_range   = [ cfg["seed_rhoy_range_hgh"][0],   cfg["seed_rhoy_range_hgh"][1] ]
        elif(nclusters>cfg["cls_mult_hgh"] and nclusters<=cfg["cls_mult_inf"]):
            self.theta_x_range = [ cfg["seed_thetax_range_inf"][0], cfg["seed_thetax_range_inf"][1] ]
            self.rho_x_range   = [ cfg["seed_rhox_range_inf"][0],   cfg["seed_rhox_range_inf"][1] ]
            self.theta_y_range = [ cfg["seed_thetay_range_inf"][0], cfg["seed_thetay_range_inf"][1] ]
            self.rho_y_range   = [ cfg["seed_rhoy_range_inf"][0],   cfg["seed_rhoy_range_inf"][1] ]
        else: 
            sys.exit(f"In hough_seeder nclusters:{nclusters}>cls_mult_inf, not implemented. exitting")
        self.thetamin_x = self.theta_x_range[0]
        self.thetamax_x = self.theta_x_range[1]
        self.thetamin_y = self.theta_y_range[0]
        self.thetamax_y = self.theta_y_range[1]
        self.rhomin_x = self.rho_x_range[0]
        self.rhomax_x = self.rho_x_range[1]
        self.rhomin_y = self.rho_y_range[0]
        self.rhomax_y = self.rho_y_range[1]
        if(cfg["dbg"]): print(f"theta_x_range={self.thetamin_x,self.thetamax_x}")
        if(cfg["dbg"]): print(f"theta_y_range={self.thetamin_y,self.thetamax_y}")
        if(cfg["dbg"]): print(f"rho_x_range={self.rhomin_x,self.rhomax_x}")
        if(cfg["dbg"]): print(f"rho_y_range={self.rhomin_y,self.rhomax_y}")
        self.nbins_thetax = -1
        self.nbins_thetay = -1
        self.nbins_rhox   = -1
        self.nbins_rhoy   = -1
        if(nclusters<=cfg["cls_mult_low"]):
            self.nbins_thetax = cfg["seed_nbins_thetax_low"]
            self.nbins_thetay = cfg["seed_nbins_thetay_low"]
            self.nbins_rhox   = cfg["seed_nbins_rhox_low"]
            self.nbins_rhoy   = cfg["seed_nbins_rhoy_low"]
        elif(nclusters>cfg["cls_mult_low"] and nclusters<=cfg["cls_mult_mid"]):
            self.nbins_thetax = cfg["seed_nbins_thetax_mid"]
            self.nbins_thetay = cfg["seed_nbins_thetay_mid"]
            self.nbins_rhox   = cfg["seed_nbins_rhox_mid"]
            self.nbins_rhoy   = cfg["seed_nbins_rhoy_mid"]
        elif(nclusters>cfg["cls_mult_mid"] and nclusters<=cfg["cls_mult_hgh"]):
            self.nbins_thetax = cfg["seed_nbins_thetax_hgh"]
            self.nbins_thetay = cfg["seed_nbins_thetay_hgh"]
            self.nbins_rhox   = cfg["seed_nbins_rhox_hgh"]
            self.nbins_rhoy   = cfg["seed_nbins_rhoy_hgh"]
        elif(nclusters>cfg["cls_mult_hgh"] and nclusters<=cfg["cls_mult_inf"]):
            self.nbins_thetax = cfg["seed_nbins_thetax_inf"]
            self.nbins_thetay = cfg["seed_nbins_thetay_inf"]
            self.nbins_rhox   = cfg["seed_nbins_rhox_inf"]
            self.nbins_rhoy   = cfg["seed_nbins_rhoy_inf"]
        else:
            sys.exit(f"In hough_seeder nclusters:{nclusters}>cls_mult_inf, not implemented. exitting")
        # self.minintersections = math.comb(len(cfg["detectors"]),2) ### all pairs out of for detectors w/o repetitions
        self.minintersections = math.comb(cfg["layers"],2) ### all pairs out of for detectors w/o repetitions
        self.nmissintersections = cfg["seed_nmiss_neigbours"] ## how many intersectians we are allowed to miss before searching in the neighbouring cells
        # self.neighbourslist = [ i for i in range(-cfg["seed_nmax_neigbours"],cfg["seed_nmax_neigbours"]+1) if(i!=0) ] ### this will be e.g. [-3,-2,-1,+1,+2,+3] if seed_nmax_neigbours=3
        self.neighbourslist = [ i for i in range(-cfg["seed_nmax_neigbours"],cfg["seed_nmax_neigbours"]+1) ] ### this will be e.g. [-3,-2,-1,0,+1,+2,+3] if seed_nmax_neigbours=3
        if(cfg["dbg"]): print(f"LUT neighbourslist={self.neighbourslist}")

        
        if(cfg["dbg"]): print(f"before waves")        
        ### define the wave parameter space
        self.h2waves_zx = self.define_theta_rho_axes("zxwaves","x",self.nbins_thetax,self.thetamin_x,self.thetamax_x, self.nbins_rhox,self.rhomin_x,self.rhomax_x)
        self.h2waves_zy = self.define_theta_rho_axes("zywaves","y",self.nbins_thetay,self.thetamin_y,self.thetamax_y, self.nbins_rhoy,self.rhomin_y,self.rhomax_y)
        
        if(cfg["dbg"]): print(f"before LUT")
        
        ### allow only positive y-z seeds:
        self.LUT = lookup_table.LookupTable(clusters,eventid)
        
        if(cfg["dbg"]):
            print(f"after LUT")
            print(f"before ACC")
        
        ### the data structure
        self.accumulator = []
        ### accumulator = [0-1{key:val}, 0-2{key:val}, 0-3{key:val}, 0-4{key:val}, 1-2{key:val}, 1-3{key:val}, 1-4{key:val}, 2-3{key:val}, 2-4{key:val}, 3-4{key:val}]
        ### key   = ecoded(brhox,bthetax,brhoy,bthetay)
        ### value = number of times the 4D key in theta-rho-x/y appears
        for ncomb in range(self.minintersections): self.accumulator.append({})
        self.naccumulators = len(self.accumulator)
        # print(f"naccumulators={self.naccumulators}")
        ### fill the accumulator
        if(cfg["dbg"]): print(f"before fill intersections")
        # self.fill_4d_wave_intersections()
        self.fill_4d_wave_intersections()
        ncells_before = 0
        for acc in self.accumulator: ncells_before += len(acc)
        ### get the 4D bin numbers of the good coordinates
        if(cfg["dbg"]): print(f"before get_seed_coordinates")
        self.cells = self.get_seed_coordinates() if(ncells_before>0) else []
        if(cfg["dbg"]): print(f"after get_seed_coordinates with {len(self.cells)} cells")
        
        self.nseeds = 0
        self.ntunnels = 0
        self.nclspertunnel = {}
        if(len(self.cells)<1):
            if(cfg["runtype"]=="beam"): print(f"EventID: {self.eventid}: got zero valid cells in Hough space")
            ##### cleanup!!! #####
            del self.accumulator
            del self.h2waves_zx
            del self.h2waves_zy
            self.LUT.clear_all()
            del self.LUT
            del self.tdmlyr_clusters
            return
        
        if(cfg["dbg"]): print(f"after ACC")
        
        ######################
        ##### cleanup!!! #####
        del self.accumulator
        ######################
        
        if(cfg["dbg"]): print(f"before fill_lut")
        
        ### check the accumulator against the LookupTable
        # self.LUT = LookupTable(clusters,eventid)
        self.LUT.fill_lut(clusters)
        if(cfg["dbg"]): print(f"after fill_lut")
        self.tunnels,self.hough_coords,self.hough_bounds,self.hough_space = self.get_tunnels()
        self.ntunnels = len(self.tunnels)
        for tnl in self.tunnels:
            for det in cfg["detectors"]:
                nclsindet = len(tnl[det])
                if(nclsindet>0): self.nclspertunnel.update({det:nclsindet})
        if(cfg["dbg"]): print(f"after get_tunnels")
        # self.tunnel_nsseds, self.tnlid, self.coord, self.seeds = self.set_seeds(clusters)
        self.tunnel_nsseds, self.tnlid, self.coord, self.seeds = self.set_seeds()
        if(cfg["dbg"]): print(f"after set_seeds")
        self.nseeds = len(self.seeds)
        ######################
        ##### cleanup!!! #####
        del self.h2waves_zx
        del self.h2waves_zy
        self.LUT.clear_all()
        del self.LUT
        del self.tdmlyr_clusters
        ######################
        minSeedsPerTnl = min(self.tunnel_nsseds) if(len(self.tunnel_nsseds)>0)     else -1
        maxSeedsPerTnl = max(self.tunnel_nsseds) if(len(self.tunnel_nsseds)>0)     else -1
        avgSeedsPerTnl = np.mean(self.tunnel_nsseds) if(len(self.tunnel_nsseds)>0) else -1
        stdSeedsPerTnl = np.std(self.tunnel_nsseds)  if(len(self.tunnel_nsseds)>0) else -1
        if(cfg["dbg"]): print(f"eventid={self.eventid}: got {len(self.tunnels)} valid tunnels out of {len(self.cells)} tunnels and a total of {len(self.seeds)} seeds. N seeds per tunnel: min={minSeedsPerTnl}, max={maxSeedsPerTnl}, mean={avgSeedsPerTnl:.3f}+/-{stdSeedsPerTnl:.3f}.")
        
    # def __del__(self):
        # print(f"eventid={self.eventid}: deleted HoughSeeder class")

    def __str__(self):
        return f"Seeder"

    def set_tdmlyr_clusters(self,clusters):
        cfg = config.Config().map
        for det in cfg["detectors"]:
            lyr = cfg["det2tdm"][det]
            for c in clusters[det]:
                self.tdmlyr_clusters[lyr].append(c)

    
    def define_theta_rho_axes(self,name,xy, tbins,tmin,tmax, rbins,rmin,rmax):
        h2 = ROOT.TH2D(f"h2_{name}",";#theta_{xy};#rho_{xy};",tbins,tmin,tmax, rbins,rmin,rmax)
        h2.SetDirectory(0)
        return h2
    

    def find_waves_intersect(self,k1,z1,k2,z2):
        dk = (k1-k2) if(abs(k1-k2)>self.xepsilon) else 1e15*np.sign(k1-k2)
        theta = math.atan2((z2-z1),dk) # the arc tangent of (y/x) in radians
        rho   = k1*math.sin(theta) + z1*math.cos(theta)
        return theta,rho


    def get_detpair(self,CA,CB):
        cfg = config.Config().map
        if(cfg["dbg"]): print(f"CA.TID={CA.TID}, CB.TID={CB.TID}")
        if(CA.TID==0 and CB.TID==1): return 0
        if(CA.TID==0 and CB.TID==2): return 1
        if(CA.TID==0 and CB.TID==3): return 2
        if(CA.TID==0 and CB.TID==4): return 3
        if(CA.TID==1 and CB.TID==2): return 4
        if(CA.TID==1 and CB.TID==3): return 5
        if(CA.TID==1 and CB.TID==4): return 6
        if(CA.TID==2 and CB.TID==3): return 7
        if(CA.TID==2 and CB.TID==4): return 8
        if(CA.TID==3 and CB.TID==4): return 9
        if(cfg["dbg"]): print(f"unknown combination for CA.TID={CA.TID} and CB.TID={CB.TID} - quitting.")
        quit()
        return -1


    def encode_key(self, brhox, bthetax, brhoy, bthetay):
        ### Encode a unique integer key from 4 integer indices using different bin counts per axis.
        ### It looks like only three dividers appear in the nested expression, but in fact,
        ### that’s exactly what’s needed — the fourth one (the first dimension’s) doesn’t need to
        ### appear explicitly as a divider because it’s the outermost stride.
        return ((((brhox * self.nbins_thetax + bthetax) * self.nbins_rhoy + brhoy) * self.nbins_thetay + bthetay))


    def decode_key(self, encoded_key):
        ### Decode the integer key back into (brhox, bthetax, brhoy, bthetay).
        bthetay = encoded_key % self.nbins_thetay
        encoded_key //= self.nbins_thetay
        brhoy = encoded_key % self.nbins_rhoy
        encoded_key //= self.nbins_rhoy
        bthetax = encoded_key % self.nbins_thetax
        encoded_key //= self.nbins_thetax
        brhox = encoded_key
        return (brhox, bthetax, brhoy, bthetay)
        

    def getbin(self,thetax,rhox,thetay,rhoy):
        cfg = config.Config().map
        if(cfg["dbg"]): print(f"thetax={thetax}, rhox={rhox}, thetay={thetay}, rhoy={rhoy}")
        if(cfg["dbg"]): print(f"histo thetax_bounds={self.h2waves_zx.GetXaxis().GetXmin(),self.h2waves_zx.GetXaxis().GetXmax()} with {self.h2waves_zx.GetNbinsX()} bins,  rhox ybounds={self.h2waves_zx.GetYaxis().GetXmin(),self.h2waves_zx.GetYaxis().GetXmax()} with {self.h2waves_zx.GetNbinsY()} bins")
        if(cfg["dbg"]): print(f"histo thetay bounds={self.h2waves_zy.GetXaxis().GetXmin(),self.h2waves_zy.GetXaxis().GetXmax()} with {self.h2waves_zy.GetNbinsX()} bins,  rhoy ybounds={self.h2waves_zy.GetYaxis().GetXmin(),self.h2waves_zy.GetYaxis().GetXmax()} with {self.h2waves_zy.GetNbinsY()} bins")
        bin_thetax = self.h2waves_zx.GetXaxis().FindBin(thetax) if(thetax>=self.thetamin_x and thetax<self.thetamax_x) else -1
        bin_rhox   = self.h2waves_zx.GetYaxis().FindBin(rhox)   if(rhox>=self.rhomin_x     and rhox<self.rhomax_x)     else -1 
        bin_thetay = self.h2waves_zy.GetXaxis().FindBin(thetay) if(thetay>=self.thetamin_y and thetay<self.thetamax_y) else -1
        bin_rhoy   = self.h2waves_zy.GetYaxis().FindBin(rhoy)   if(rhoy>=self.rhomin_y     and rhoy<self.rhomax_y)     else -1
        # print(f"bin_thetax,bin_rhox,bin_thetay,bin_rhoy={bin_thetax,bin_rhox,bin_thetay,bin_rhoy}")
        # print(f"thetax,rhox,thetay,rhoy={thetax,rhox,thetay,rhoy}")
        valid = (bin_thetax>=0 and bin_rhox>=0 and bin_thetay>=0 and bin_rhoy>=0)
        return valid,bin_thetax,bin_rhox,bin_thetay,bin_rhoy        
    

    def fill_pair_vectorized(self, lyrA, lyrB, detpair_idx):
        cfg = config.Config().map
        clustersA = self.tdmlyr_clusters[lyrA]
        clustersB = self.tdmlyr_clusters[lyrB]
        
        if not clustersA or not clustersB: return

        # 1. Extract coordinates
        zA = np.array([c.zTnoGmm for c in clustersA])
        xA = np.array([c.xTnoGmm for c in clustersA])
        yA = np.array([c.yTnoGmm for c in clustersA])
        zB = np.array([c.zTnoGmm for c in clustersB])
        xB = np.array([c.xTnoGmm for c in clustersB])
        yB = np.array([c.yTnoGmm for c in clustersB])

        # 2. Vectorized Analytic Intersection
        dKx = xA[:, np.newaxis] - xB
        dKy = yA[:, np.newaxis] - yB
        dZ  = zB - zA[:, np.newaxis]
        
        thetax = np.arctan2(dZ, dKx)
        rhox   = xA[:, np.newaxis] * np.sin(thetax) + zA[:, np.newaxis] * np.cos(thetax)
        thetay = np.arctan2(dZ, dKy)
        rhoy   = yA[:, np.newaxis] * np.sin(thetay) + zA[:, np.newaxis] * np.cos(thetay)

        # 3. Vectorized ROOT-Parity Binning
        btx = np.floor((thetax - self.thetamin_x) / (self.thetamax_x - self.thetamin_x) * self.nbins_thetax).astype(int) + 1
        brx = np.floor((rhox - self.rhomin_x) / (self.rhomax_x - self.rhomin_x) * self.nbins_rhox).astype(int) + 1
        bty = np.floor((thetay - self.thetamin_y) / (self.thetamax_y - self.thetamin_y) * self.nbins_thetay).astype(int) + 1
        bry = np.floor((rhoy - self.rhomin_y) / (self.rhomax_y - self.rhomin_y) * self.nbins_rhoy).astype(int) + 1

        # 4. Vectorized Valid Intersection Mask
        valid = (btx >= 1) & (btx <= self.nbins_thetax) & \
                (brx >= 1) & (brx <= self.nbins_rhox) & \
                (bty >= 1) & (bty <= self.nbins_thetay) & \
                (bry >= 1) & (bry <= self.nbins_rhoy)

        # 5. NEW: Vectorized Vertical Inclination Cut
        if not cfg["seed_allow_negative_vertical_inclination"]:
            # Replicating the slope calculation from your LUT: Ay = tan(theta)
            # This is applied as a mask across the entire matrix simultaneously
            slope_mask = -np.tan(thetay) >= 0
            valid &= slope_mask

        # 6. Populate Dictionaries and Wave Histograms
        valid_indices = np.argwhere(valid)
        for iA, iB in valid_indices:
            key = self.encode_key(int(brx[iA, iB]), int(btx[iA, iB]), 
                                  int(bry[iA, iB]), int(bty[iA, iB]))
            
            self.accumulator[detpair_idx][key] = self.accumulator[detpair_idx].get(key, 0) + 1
            
            # # Fill wave histograms for final seeding parity
            # self.h2waves_zx.Fill(thetax[iA, iB], rhox[iA, iB])
            # self.h2waves_zy.Fill(thetay[iA, iB], rhoy[iA, iB])

    def fill_4d_wave_intersections(self):
        # Detector pairs mapping to your self.accumulator indices
        pairs = [
            (0, 1, 0), (0, 2, 1), (0, 3, 2), (0, 4, 3), # Pairs with Layer 0
            (1, 2, 4), (1, 3, 5), (1, 4, 6),           # Pairs with Layer 1
            (2, 3, 7), (2, 4, 8),                      # Pairs with Layer 2
            (3, 4, 9)                                  # Pairs with Layer 3
        ]

        for lA, lB, acc_idx in pairs:
            self.fill_pair_vectorized(lA, lB, acc_idx)


    # def fill_accumulator(self,bdetpair,brhox,bthetax,brhoy,bthetay):
    #     key = self.encode_key(brhox,bthetax,brhoy,bthetay)
    #     self.accumulator[bdetpair][key] = self.accumulator[bdetpair].get(key,0)+1
    #
    #
    # def get_pair(self,CA,CB):
    #     cfg = config.Config().map
    #     if(cfg["dbg"]): print(f"In eventid={self.eventid}:  CA={CA.det}.{CA.CID}, CB={CB.det}.{CB.CID}")
    #     # thetax,rhox = self.find_waves_intersect(CA.xTmm,CA.zTmm,CB.xTmm,CB.zTmm)
    #     # thetay,rhoy = self.find_waves_intersect(CA.yTmm,CA.zTmm,CB.yTmm,CB.zTmm)
    #     thetax,rhox = self.find_waves_intersect(CA.xTnoGmm,CA.zTnoGmm,CB.xTnoGmm,CB.zTnoGmm)
    #     thetay,rhoy = self.find_waves_intersect(CA.yTnoGmm,CA.zTnoGmm,CB.yTnoGmm,CB.zTnoGmm)
    #     # if(cfg["dbg"]):
    #     #     print(f"normal: CAxz={CA.xTnoGmm,CA.zTnoGmm} CBxz={CB.xTnoGmm,CB.zTnoGmm}")
    #     #     print(f"normal: CAyz={CA.yTnoGmm,CA.zTnoGmm} CByz={CB.yTnoGmm,CB.zTnoGmm}")
    #     #     z1 = (CA.zTnoGmm-self.zlabmin)/self.Lz
    #     #     z2 = (CB.zTnoGmm-self.zlabmin)/self.Lz
    #     #     x1 = (CA.xTnoGmm-self.xlabmin)/self.Lx
    #     #     x2 = (CB.xTnoGmm-self.xlabmin)/self.Lx
    #     #     y1 = (CA.yTnoGmm-self.ylabmin)/self.Ly
    #     #     y2 = (CB.yTnoGmm-self.ylabmin)/self.Ly
    #     #     print(f"rescaled: CAxz={x1,z1} CBxz={x2,z2}")
    #     #     print(f"rescaled: CAyz={y1,z1} CByz={y2,z2}")
    #     valid,bthetax,brhox,bthetay,brhoy = self.getbin(thetax,rhox,thetay,rhoy)
    #     if(not cfg["seed_allow_negative_vertical_inclination"]):
    #         AY,BY = self.LUT.get_par_lin(thetay,rhoy)
    #         if(AY<0.): return
    #     detpair = self.get_detpair(CA,CB)
    #     # print(f"detpair={detpair}")
    #     if(cfg["dbg"]): print(f"in get_pair: eventid={self.eventid}  detpair={detpair}  valid={valid}  -->  bthetax={bthetax}, brhox={brhox}, bthetay={bthetay}, brhoy={brhoy}")
    #     # print(f"detpair={detpair}: thetax={thetax}, rhox={rhox}, thetay={thetay}, rhoy={rhoy}")
    #     if(valid): self.fill_accumulator(detpair,brhox,bthetax,brhoy,bthetay)
    #     self.h2waves_zx.Fill(thetax,rhox)
    #     self.h2waves_zy.Fill(thetay,rhoy)
    #
    #
    # def fill_4d_wave_intersections(self):
    #     for c0 in self.tdmlyr_clusters[0]:
    #         for c1 in self.tdmlyr_clusters[1]:
    #             self.get_pair(c0,c1)
    #     for c0 in self.tdmlyr_clusters[0]:
    #         for c2 in self.tdmlyr_clusters[2]:
    #             self.get_pair(c0,c2)
    #     for c0 in self.tdmlyr_clusters[0]:
    #         for c3 in self.tdmlyr_clusters[3]:
    #             self.get_pair(c0,c3)
    #     for c0 in self.tdmlyr_clusters[0]:
    #         for c4 in self.tdmlyr_clusters[4]:
    #             self.get_pair(c0,c4)
    #     for c1 in self.tdmlyr_clusters[1]:
    #         for c2 in self.tdmlyr_clusters[2]:
    #             self.get_pair(c1,c2)
    #     for c1 in self.tdmlyr_clusters[1]:
    #         for c3 in self.tdmlyr_clusters[3]:
    #             self.get_pair(c1,c3)
    #     for c1 in self.tdmlyr_clusters[1]:
    #         for c4 in self.tdmlyr_clusters[4]:
    #             self.get_pair(c1,c4)
    #     for c2 in self.tdmlyr_clusters[2]:
    #         for c3 in self.tdmlyr_clusters[3]:
    #             self.get_pair(c2,c3)
    #     for c2 in self.tdmlyr_clusters[2]:
    #         for c4 in self.tdmlyr_clusters[4]:
    #             self.get_pair(c2,c4)
    #     for c3 in self.tdmlyr_clusters[3]:
    #         for c4 in self.tdmlyr_clusters[4]:
    #             self.get_pair(c3,c4)
    
    
    def search_in_neighbours(self,encoded_key):
        cfg = config.Config().map
        neigbours_vals = 0
        ### search in a 4D cube with size +-neighbours 
        ### where for example neighbours is: [-5,-4,-3,-2,-1,0,+1,+2,+3,+4,+5]
        key = self.decode_key(encoded_key)
        # print(f"in search_in_neighbours: key={key}")
        ### d0,d1,d2,,d3 are the brhox,bthetax,brhoy,bthetay
        for d0 in self.neighbourslist:
            for d1 in self.neighbourslist:
                for d2 in self.neighbourslist:
                    for d3 in self.neighbourslist:
                        if(d0==0 and d1==0 and d2==0 and d3==0): continue
                        nighbourkey = self.encode_key(key[0]+d0, key[1]+d1, key[2]+d2, key[3]+d3)
                        if(cfg["dbg"]): print(f"d0={d0}, d1={d1}, d2={d2}, d3={d3} --> nighbourkey={nighbourkey} --> decodednegkey={ self.decode_key(nighbourkey) }")
                        for detpair in range(self.naccumulators): ### loop over all detector-pairs
                            neigbours_vals += (self.accumulator[detpair].get(nighbourkey,0)>0)
        return neigbours_vals


    def get_seed_coordinates(self):
        cfg = config.Config().map
        cells = []
        ### accumulator = [0-1{key:val}, 0-2{key:val}, 0-3{key:val}, 0-4{key:val}, 1-2{key:val}, 1-3{key:val}, 1-4{key:val}, 2-3{key:val}, 2-4{key:val}, 3-4{key:val}]
        ### key   = ecoded(brhox,bthetax,brhoy,bthetay)
        ### value = number of times the 4D key in theta-rho-x/y appears
        # if(cfg["dbg"] and cfg["runtype"]!="beam"): print(f"accumulator: {self.accumulator}")
        if(cfg["dbg"]): print(f"accumulator: {self.accumulator}")
        ### check the index with the most occurances
        index_of_most_frequent_key = -1
        # First pass: count occurrences
        key_counts = defaultdict(int)
        for d in self.accumulator:
            for key in d:  # only one key per dict
                key_counts[key] += 1
        # Find the key with the highest count
        most_common_key = max(key_counts, key=key_counts.get)
        # Second pass: find first index of most common key
        for idx, d in enumerate(self.accumulator):
            if most_common_key in d:
                index_of_most_frequent_key = idx
                break

        ### start by looping on all keys of the detector pair with the most repetitions
        for key,val in self.accumulator[index_of_most_frequent_key].items():
            nintersections = (val>0)
            if(cfg["dbg"]): print(f"key={key}, val={val} --> nintersections={nintersections}")

            for detpair in range(1,self.naccumulators):
                nintersections += (self.accumulator[detpair].get(key,0)>0)
                if(cfg["dbg"]): print(f"key={key} detpair={detpair}: nintersections={nintersections}")
            if(cfg["dbg"]): print(f"Final: nintersections={nintersections}, self.minintersections={self.minintersections}")
            if(nintersections>=self.minintersections):
                cells.append(key)

            ### if too low:
            if(cfg["seed_allow_neigbours"] and (nintersections<self.minintersections and nintersections>=(self.minintersections-self.nmissintersections))):
                if(cfg["dbg"]): print(f"Trying to recover more intersections than the available {nintersections}")
                nintersections += self.search_in_neighbours(key)
                if(nintersections>=self.minintersections):
                    cells.append(key)
            if(cfg["dbg"] and cfg["runtype"]!="beam"): print(f"Final nintersections={nintersections}")
            ### otherwise don't bother
        if(cfg["dbg"]): print(f"cumulator sizes: {len(self.accumulator[0]),len(self.accumulator[1]),len(self.accumulator[2]),len(self.accumulator[3]),len(self.accumulator[4]),len(self.accumulator[5]),len(self.accumulator[6]),len(self.accumulator[7]),len(self.accumulator[8]),len(self.accumulator[9])}, good cells: {len(cells)}")
        return cells        
    
    
    
    def get_tunnels(self):
        cfg = config.Config().map
        if(cfg["dbg"]): print(f"in get tunnels with {len(self.cells)}")
        tunnels      = []
        hough_coords = []
        hough_bounds = []
        hough_space  = {
            "zx_xbins":self.h2waves_zx.GetNbinsX(), "zx_xmin":self.h2waves_zx.GetXaxis().GetXmin(), "zx_xmax":self.h2waves_zx.GetXaxis().GetXmax(),
            "zx_ybins":self.h2waves_zx.GetNbinsY(), "zx_ymin":self.h2waves_zx.GetYaxis().GetXmin(), "zx_ymax":self.h2waves_zx.GetYaxis().GetXmax(),
            "zy_xbins":self.h2waves_zy.GetNbinsX(), "zy_xmin":self.h2waves_zy.GetXaxis().GetXmin(), "zy_xmax":self.h2waves_zy.GetXaxis().GetXmax(),
            "zy_ybins":self.h2waves_zy.GetNbinsY(), "zy_ymin":self.h2waves_zy.GetYaxis().GetXmin(), "zy_ymax":self.h2waves_zy.GetYaxis().GetXmax()
        }

        for icell,cell in enumerate(self.cells):
            (brhox,bthetax,brhoy,bthetay) = self.decode_key(cell)

            central_thetax = self.h2waves_zx.GetXaxis().GetBinCenter(bthetax)
            central_rhox   = self.h2waves_zx.GetYaxis().GetBinCenter(brhox)
            central_thetay = self.h2waves_zy.GetXaxis().GetBinCenter(bthetay)
            central_rhoy   = self.h2waves_zy.GetYaxis().GetBinCenter(brhoy)

            thetax = [ self.h2waves_zx.GetXaxis().GetBinLowEdge(bthetax), self.h2waves_zx.GetXaxis().GetBinUpEdge(bthetax) ]
            rhox   = [ self.h2waves_zx.GetYaxis().GetBinLowEdge(brhox),   self.h2waves_zx.GetYaxis().GetBinUpEdge(brhox)   ]
            thetay = [ self.h2waves_zy.GetXaxis().GetBinLowEdge(bthetay), self.h2waves_zy.GetXaxis().GetBinUpEdge(bthetay) ]
            rhoy   = [ self.h2waves_zy.GetYaxis().GetBinLowEdge(brhoy),   self.h2waves_zy.GetYaxis().GetBinUpEdge(brhoy)   ]

            valid,tunnel = self.LUT.clusters_in_tunnel(thetax,rhox,thetay,rhoy)
            if(cfg["dbg"]):
                print(f"Center: central_thetax,central_rhox,central_thetay,central_rhoy={central_thetax,central_rhox,central_thetay,central_rhoy}")
                print(f"Bounds: thetax,rhox,thetay,rhoy={thetax,rhox,thetay,rhoy}")
                print(f"valid,tunnel={valid,tunnel}")

            if(valid):
                tunnels.append( tunnel )
                hough_coords.append( (central_thetax,central_rhox,central_thetay,central_rhoy) )
                hough_bounds.append( (thetax,rhox,thetay,rhoy) )
            # print(f"Cell[{icell}]: valid?{valid} --> tunnel={tunnel}")
        return tunnels,hough_coords,hough_bounds,hough_space
    
    
    
    def set_seeds(self):
        cfg = config.Config().map
        tunnel_nsseds = [1]*len(self.tunnels)
        seeds = []
        tnlid = []
        coord = []
        if(cfg["dbg"]): print(f"len(self.tunnels)={len(self.tunnels)}")
        for itnl,tunnel in enumerate(self.tunnels):
            candidate = []
            if(cfg["dbg"]): print(f"tunnel={tunnel}")
            tnldets = []
            for det,tnlcls in tunnel.items():
                if(len(tnlcls)>0):
                    tnldets.append(det)
            if(cfg["dbg"]): print(f"len(tnldets)={len(tnldets)}")
            if(len(tnldets)>cfg["layers"]):
                print("TODO: in set_seeds(), got more layers than implemented.")
                # quit()
            det0 = tnldets[0]
            det1 = tnldets[1]
            det2 = tnldets[2]
            det3 = tnldets[3]
            det4 = tnldets[4]
            if(cfg["dbg"]): print(f"dets of tunnel: {tnldets}, tunnel={tunnel}")
            n0 = len(tunnel[det0])
            n1 = len(tunnel[det1])
            n2 = len(tunnel[det2])
            n3 = len(tunnel[det3])
            n4 = len(tunnel[det4])
            tunnel_nsseds[itnl] = n0*n1*n2*n3*n4
            for c0 in tunnel[det0]:
                for c1 in tunnel[det1]:
                    for c2 in tunnel[det2]:
                        for c3 in tunnel[det3]:
                            for c4 in tunnel[det4]:
                                seeds.append( {det0:c0,det1:c1,det2:c2,det3:c3,det4:c4} )
                                tnlid.append( itnl )
                                coord.append( self.hough_coords[itnl] )
        return tunnel_nsseds,tnlid,coord,seeds

#!/usr/bin/python
import os
import math
import subprocess
import array
import numpy as np
import ROOT

from tracker_lib import config, utils

class LookupTable:
    def __init__(self,clusters,eventid=0):
        cfg = config.Config().map
        self.eventid = eventid
        self.LUT = {}
        # self.AXS = {}
        ncls = 0
        for det in cfg["detectors"]: ncls += len(clusters[det])
        ncls = int(ncls/len(cfg["detectors"]))
        
        self.nbinsx = -1
        self.nbinsy = -1
        if(ncls<cfg["cls_mult_low"]):
            self.nbinsx = cfg["lut_nbinsx_low"]
            self.nbinsy = cfg["lut_nbinsy_low"]
        elif(ncls>=cfg["cls_mult_low"] and ncls<cfg["cls_mult_mid"]):
            self.nbinsx = cfg["lut_nbinsx_mid"]
            self.nbinsy = cfg["lut_nbinsy_mid"]
        elif(ncls>=cfg["cls_mult_mid"] and ncls<cfg["cls_mult_hgh"]):
            self.nbinsx = cfg["lut_nbinsx_hgh"]
            self.nbinsy = cfg["lut_nbinsy_hgh"]
        elif(ncls>=cfg["cls_mult_hgh"] and ncls<cfg["cls_mult_inf"]):
            self.nbinsx = cfg["lut_nbinsx_inf"]
            self.nbinsy = cfg["lut_nbinsy_inf"]
        else:
            sys.exit(f"In lookup_table ncls:ncls>cls_mult_inf, not implemented. exitting")
        
        self.chipXmin = -( cfg["chipX"]*(1.+cfg["lut_scaleX"]) )/2.
        self.chipXmax = +( cfg["chipX"]*(1.+cfg["lut_scaleX"]) )/2.
        self.chipYmin = -( cfg["chipY"]*(1.+cfg["lut_scaleY"]) )/2.
        self.chipYmax = +( cfg["chipY"]*(1.+cfg["lut_scaleY"]) )/2.
        
        self.z_positions = {} # NEW: Store Z per detector
        self.axs_meta = {}
        
        ### call in the constructor:
        self.init_axs()
        self.init_lut()
    
    # def __del__(self):
        # print(f"eventid={self.eventid}: deleted LookupTable class")


    def init_axs(self):
        cfg = config.Config().map
        for det in cfg["detectors"]:
            rmin = [self.chipXmin,self.chipYmin,0]
            rmax = [self.chipXmax,self.chipYmax,0]
            rTmin = utils.transform_to_real_space(rmin,det)
            rTmax = utils.transform_to_real_space(rmax,det)
            rTnoGmin = utils.undo_global_offsets(rTmin,det)
            rTnoGmax = utils.undo_global_offsets(rTmax,det)
            xmin = min(rTnoGmin[0],rTnoGmax[0])
            xmax = max(rTnoGmin[0],rTnoGmax[0])
            ymin = min(rTnoGmin[1],rTnoGmax[1])
            ymax = max(rTnoGmin[1],rTnoGmax[1])
            if(cfg["dbg"]): print(f"in init_axs for eventid={self.eventid} and {det}: xmin,xmax,ymin,ymax={xmin,xmax,ymin,ymax}, bins={self.nbinsx,self.nbinsy}")
            # self.AXS.update({ det:ROOT.TH2D(f"lut_{det}_{self.eventid}",";x;y;Clusters",self.nbinsx,xmin,xmax, self.nbinsy,ymin,ymax) })
            # self.AXS[det].SetDirectory(0)
            # Extract Z once
            r0 = [0, 0, cfg["rdetectors"][det][2]]
            rT = utils.transform_to_real_space(r0, det)
            rTnoG = utils.undo_global_offsets(rT, det)
            self.z_positions[det] = rTnoG[2]
            
            # Store metadata for vectorized binning math
            self.axs_meta[det] = {
                'xmin': xmin, 'xstep': (xmax - xmin) / self.nbinsx,
                'ymin': ymin, 'ystep': (ymax - ymin) / self.nbinsy,
                'z': rTnoG[2], 'stride': self.nbinsx + 2
            }

        
    def init_lut(self):
        cfg = config.Config().map
        for det in cfg["detectors"]:
            self.LUT.update({ det:{} })


    def find_alignment_bounds(self):
        cfg = config.Config().map
        xmax = 0
        ymax = 0
        for key1 in cfg["misalignment"]:
            for key2 in cfg["misalignment"][key1]:
                d = abs(cfg["misalignment"][key1][key2])
                xmax = d if(key2=="dx" and d>xmax) else xmax
                ymax = d if(key2=="dy" and d>ymax) else ymax
        # print(f"In lookup table: alignment modifier to x is {xmax} and to y is {ymax}")
        return xmax,ymax

    
    # def fill_lut(self,clusters):
    #     cfg = config.Config().map
    #     for det in cfg["detectors"]:
    #         for clsidx,cluster in enumerate(clusters[det]):
    #             bx = self.AXS[det].GetXaxis().FindBin(cluster.xTnoGmm)
    #             by = self.AXS[det].GetYaxis().FindBin(cluster.yTnoGmm)
    #             if(cfg["dbg"]): print(f"In LUT: eventid={self.eventid}  {det}  --> cluster x={cluster.xTnoGmm}, y={cluster.yTnoGmm}, z={cluster.zTnoGmm}  -->  bx={bx}, by={by}")
    #             validx = (cluster.xTnoGmm>=self.AXS[det].GetXaxis().GetXmin() and cluster.xTnoGmm<self.AXS[det].GetXaxis().GetXmax())
    #             validy = (cluster.yTnoGmm>=self.AXS[det].GetYaxis().GetXmin() and cluster.yTnoGmm<self.AXS[det].GetYaxis().GetXmax())
    #             if(not validx or not validy):
    #                 print(f"in fill_lut: validx={validx}, validy={validy} with x={cluster.xTnoGmm}, y={cluster.yTnoGmm}, z={cluster.zTnoGmm}  and xlim=[{self.AXS[det].GetXaxis().GetXmin(),self.AXS[det].GetXaxis().GetXmax()}] and ylim=[{self.AXS[det].GetYaxis().GetXmin(),self.AXS[det].GetYaxis().GetXmax()}]")
    #                 print("please increase the lut scale. quitting")
    #                 quit()
    #             axsbin = self.AXS[det].FindBin(cluster.xTnoGmm,cluster.yTnoGmm)
    #             # print(f"In LUT: eventid={self.eventid}  {det}  -->  axsbin={axsbin}")
    #             if(axsbin in self.LUT[det]): self.LUT[det][axsbin].append(clsidx)
    #             else:                        self.LUT[det].update( {axsbin:[clsidx]} )
    
    def fill_lut(self,clusters):
        cfg = config.Config().map
        for det in cfg["detectors"]:
            for clsidx,cluster in enumerate(clusters[det]):
                # Fast analytical FindBin
                bx = int((cluster.xTnoGmm - self.axs_meta[det]['xmin']) / self.axs_meta[det]['xstep']) + 1
                by = int((cluster.yTnoGmm - self.axs_meta[det]['ymin']) / self.axs_meta[det]['ystep']) + 1
                # Check bounds exactly as FindBin does
                if 1 <= bx <= self.nbinsx and 1 <= by <= self.nbinsy:
                    axsbin = bx + self.axs_meta[det]['stride'] * by
                    if axsbin in self.LUT[det]: self.LUT[det][axsbin].append(clsidx)
                    else:                       self.LUT[det][axsbin] = [clsidx]
    
    
    # def remove_from_lut(self,det,x,y,clsidx):
    #     axsbin = self.AXS[det].FindBin(x,y)
    #     if(axsbin in self.LUT[det]): self.LUT[det][axsbin].remove(clsidx)
    #
    #
    # def get_par_lin(self,theta_k,rho_k): ### theta and rho from Hough transform
    #     if(math.sin(theta_k)==0):
    #         print(f"in get_par_lin, sin(theta)=0: quitting.")
    #         quit()
    #     if(math.tan(theta_k)==0):
    #         print(f"in get_par_lin, tan(theta)=0: quitting.")
    #         quit()
    #     AK = -1./math.tan(theta_k)
    #     BK = rho_k/math.sin(theta_k)
    #     # print(f"theta_k={theta_k}, rho_k={rho_k} --> AK={AK}, BK={BK}")
    #     return AK,BK
    #
    #
    # def k_of_z(self,z,AK,BK):
    #     k = AK*z + BK
    #     # print(f"AK={AK}, BK={BK}, z={z} --> k={k}")
    #     return k


    def get_edges_vectorized(self, det, theta_x, rho_x, theta_y, rho_y):
        """Vectorized version of corner prediction."""
        zdet = self.z_positions[det]
        
        # Calculate AK and BK for all corners at once
        # theta_x/y and rho_x/y are lists of [min, max]
        tx = np.array(theta_x)
        rx = np.array(rho_x)
        ty = np.array(theta_y)
        ry = np.array(rho_y)
        
        # AK = -1/tan(theta), BK = rho/sin(theta)
        ax = -1.0 / np.tan(tx)
        bx = rx / np.sin(tx)
        ay = -1.0 / np.tan(ty)
        by = ry / np.sin(ty)
        
        # Track prediction: XX = AX*z + BX
        xx = ax * zdet + bx
        yy = ay * zdet + by
        
        return xx.min(), xx.max(), yy.min(), yy.max()
    
    
    # def clusters_in_tunnel(self, theta_x, rho_x, theta_y, rho_y):
    #     cfg = config.Config().map
    #     tunnel = {}
    #     planes = 0
    #
    #     for det in cfg["detectors"]:
    #         # 1. Use vectorized corner prediction
    #         xmin, xmax, ymin, ymax = self.get_edges_vectorized(det, theta_x, rho_x, theta_y, rho_y)
    #
    #         # 2. Faster bin range finding (avoiding repeated ROOT axis calls)
    #         axis_x = self.AXS[det].GetXaxis()
    #         axis_y = self.AXS[det].GetYaxis()
    #
    #         xbinmin = axis_x.FindBin(xmin) if xmin >= axis_x.GetXmin() else 1
    #         xbinmax = axis_x.FindBin(xmax) if xmax <  axis_x.GetXmax() else self.nbinsx
    #         ybinmin = axis_y.FindBin(ymin) if ymin >= axis_y.GetYaxis().GetXmin() else 1
    #         ybinmax = axis_y.FindBin(ymax) if ymax <  axis_y.GetYaxis().GetXmax() else self.nbinsy
    #
    #         clsidx_in_tnl = []
    #         det_lut = self.LUT[det]
    #
    #         # 3. Optimized bin loop
    #         # Fetch the GetBin method reference to speed up calls inside the loop
    #         get_bin_method = self.AXS[det].GetBin
    #
    #         for bx in range(xbinmin, xbinmax + 1):
    #             for by in range(ybinmin, ybinmax + 1):
    #                 axsbin = get_bin_method(bx, by)
    #                 if axsbin in det_lut:
    #                     # Direct extend is faster than a loop
    #                     clsidx_in_tnl.extend(det_lut[axsbin])
    #
    #         tunnel[det] = clsidx_in_tnl
    #         planes += (len(clsidx_in_tnl) > 0)
    #
    #     valid = (planes >= cfg["layers"])
    #     return valid, tunnel

    def clusters_in_tunnel(self, theta_x, rho_x, theta_y, rho_y):
        cfg = config.Config().map
        tunnel = {}
        planes = 0
        
        # Pre-solve geometry
        ax = -1.0 / np.tan(theta_x)
        bx_p = np.array(rho_x) / np.sin(theta_x)
        ay = -1.0 / np.tan(theta_y)
        by_p = np.array(rho_y) / np.sin(theta_y)

        for det in cfg["detectors"]:
            meta = self.axs_meta[det]
            z = meta['z']
            
            # Project corners
            xx = ax * z + bx_p
            yy = ay * z + by_p
            
            # Calculate bin ranges
            # Use np.clip to stay within 1 to nbins range
            x_bins = np.floor((xx - meta['xmin']) / meta['xstep']).astype(int) + 1
            y_bins = np.floor((yy - meta['ymin']) / meta['ystep']).astype(int) + 1
            
            xbinmin, xbinmax = max(1, x_bins.min()), min(self.nbinsx, x_bins.max())
            ybinmin, ybinmax = max(1, y_bins.min()), min(self.nbinsy, y_bins.max())
            
            clsidx_in_tnl = []
            det_lut = self.LUT[det]
            stride = meta['stride']
            
            for bx in range(xbinmin, xbinmax + 1):
                base_idx = bx
                for by in range(ybinmin, ybinmax + 1):
                    axsbin = base_idx + stride * by
                    if axsbin in det_lut:
                        clsidx_in_tnl.extend(det_lut[axsbin])
            
            tunnel[det] = clsidx_in_tnl
            planes += (len(clsidx_in_tnl) > 0)
            
        valid = (planes >= cfg["layers"])
        return valid, tunnel


    # def get_edges_from_theta_rho_corners(self,det,theta_x,rho_x,theta_y,rho_y):
    #     cfg = config.Config().map
    #     xmin = +1e20
    #     xmax = -1e20
    #     ymin = +1e20
    #     ymax = -1e20
    #     r0 = [0,0,cfg["rdetectors"][det][2]]
    #     rT = utils.transform_to_real_space(r0,det)
    #     rTnoG = utils.undo_global_offsets(rT,det)
    #     zdet = rTnoG[2] ### in rescaled space
    #     ## loop on the bounds of theta and rho
    #     for i in range(2):
    #         AX,BX = self.get_par_lin(theta_x[i],rho_x[i])
    #         AY,BY = self.get_par_lin(theta_y[i],rho_y[i])
    #         ### get the xy coordinates in the normal space ###
    #         XX = self.k_of_z(zdet,AX,BX)
    #         YY = self.k_of_z(zdet,AY,BY)
    #         if(cfg["dbg"]): print(f"get_edges_from_theta_rho_corners corner[i]: eventid={self.eventid}  -->  {det} prediction: x={XX}, y={YY}, z={zdet}")
    #         xmin = XX if(XX<xmin) else xmin
    #         xmax = XX if(XX>xmax) else xmax
    #         ymin = YY if(YY<ymin) else ymin
    #         ymax = YY if(YY>ymax) else ymax
    #     if(cfg["dbg"]): print(f"In get_edges_from_theta_rho_corners: xmin,xmax,ymin,ymax={xmin,xmax,ymin,ymax}")
    #     return xmin,xmax,ymin,ymax
    #
    #
    # def clusters_in_tunnel(self,theta_x,rho_x,theta_y,rho_y):
    #     cfg = config.Config().map
    #     if(cfg["dbg"]): print(f"clusters_in_tunnel: eventid={self.eventid}  -->  theta_x={theta_x}, rho_x={rho_x}, theta_y={theta_y}, rho_y={rho_y}")
    #     tunnel = {}
    #     planes = 0
    #     for det in cfg["detectors"]:
    #         xmin,xmax,ymin,ymax = self.get_edges_from_theta_rho_corners(det,theta_x,rho_x,theta_y,rho_y)
    #         xbinmin = self.AXS[det].GetXaxis().FindBin(xmin) if(xmin>=self.AXS[det].GetXaxis().GetXmin()) else 1
    #         xbinmax = self.AXS[det].GetXaxis().FindBin(xmax) if(xmax< self.AXS[det].GetXaxis().GetXmax()) else self.nbinsx
    #         ybinmin = self.AXS[det].GetYaxis().FindBin(ymin) if(ymin>=self.AXS[det].GetYaxis().GetXmin()) else 1
    #         ybinmax = self.AXS[det].GetYaxis().FindBin(ymax) if(ymax< self.AXS[det].GetYaxis().GetXmax()) else self.nbinsy
    #         if(cfg["dbg"]): print(f"clusters_in_tunnel: eventid={self.eventid}  -->  {det}: xrange={xmin,xmax}  xbinrange={xbinmin,xbinmax},  yrange={ymin,ymax}  ybinrange={ybinmin,ybinmax}")
    #         clsidx_in_tnl = []
    #         for bx in range(xbinmin,xbinmax+1):
    #             for by in range(ybinmin,ybinmax+1):
    #                 axsbin = self.AXS[det].GetBin(bx,by)
    #                 # print(f"clusters_in_tunnel: eventid={self.eventid}  -->  {det}:  bx/y={bx,by}  axsbin={axsbin}")
    #                 if(axsbin in self.LUT[det]):
    #                     for c in self.LUT[det][axsbin]:
    #                         clsidx_in_tnl.append(c)
    #         tunnel.update( {det:clsidx_in_tnl} )
    #         if(cfg["dbg"]): print(f"clusters_in_tunnel: eventid={self.eventid}  -->  {det}: tunnel={tunnel[det]}")
    #         planes += (len(clsidx_in_tnl)>0)
    #     valid = (planes>=cfg["layers"])
    #     return valid,tunnel
    
    def clear_all(self):
        del self.LUT
        # del self.AXS
    
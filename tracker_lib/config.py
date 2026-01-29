#!/usr/bin/python
import os
import math
import array
import numpy as np
import sys
import configparser


### config file looks like that:
# [SECTION_NAME]
# key1 = value1
# key2 = value2
class Config:
    _instance = None
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        if(cls._instance is None):
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance

    def __init__(self, fname=None, doprint=False):
        if(self._initialized): return
        if(fname is None):
            if(not hasattr(self, 'map')):
                print("Error: Config accessed before initialization.")
                sys.exit(1)
            return
            
        self.fname = fname
        self.doprint = doprint
        self.configurator = configparser.RawConfigParser()
        self.configurator.optionxform = str ### preserve case sensitivity
        self.map = {} ### the config map
        self.set(fname, doprint)
        self.check_inputs()
        self._initialized = True

        
    def read(self,fname):
        if(self.doprint): print("Reading configuration from: ",fname)
        self.configurator.read(fname)
        
    def getF(self,section,var):
        expr = dict(self.configurator.items(section))[var]
        if(not expr.isnumeric()):
            return float(eval(expr))
        return float(dict(self.configurator.items(section))[var])
       
    def getI(self,section,var):
        expr = dict(self.configurator.items(section))[var]
        if(not expr.isnumeric()):
            return int(eval(expr))
        return int(dict(self.configurator.items(section))[var])
    
    def getB(self,section,var):
        return True if(int(dict(self.configurator.items(section))[var])==1) else False
    
    def getS(self,section,var):
        return str(dict(self.configurator.items(section))[var])

    def getArrS(self,section,var):
        s = self.getS(section,var)
        return s.split(" ")
    
    def getArrI(self,section,var):
        s = self.getS(section,var).split(" ")
        i = [int(x) for x in s]
        return i
        
    def getArrF(self,section,var):
        s = self.getS(section,var).split(" ")
        f = [float(x) for x in s]
        return f
    
    def getMapI2S(self,section,var):
        s = self.getS(section,var).split(" ")
        m = {}
        for x in s:
            x = x.split(":")
            name = x[0]
            sval = x[1]
            m.update({int(sval):name})
        return m

    def getMapI2F(self,section,var):
        s = self.getS(section,var).split(" ")
        m = {}
        for x in s:
            x = x.split(":")
            idx = int(x[0])
            flt = float(x[1])
            m.update({idx:flt})
        return m

    def getMapS2T(self,section,var):
        s = self.getS(section,var).split(" ")
        m = {}
        for x in s:
            x = x.split(":")
            name = x[0]
            sarr = x[1].split(",")
            tpl  = (int(sarr[0]),int(sarr[1]))
            m.update({name:tpl})
        return m
    
    def getMapI2ArrS(self,section,var):
        s = self.getS(section,var).split(" ")
        m = {}
        for x in s:
            x = x.split(":")
            key  = int(x[0])
            sarr = x[1].split(",")
            m.update({key:sarr})
        return m
    
    def getMapS2ArrF(self,section,var):
        s = self.getS(section,var).split(" ")
        m = {}
        for x in s:
            x = x.split(":")
            name = x[0]
            sarr = x[1].split(",")
            farr = [float(n) for n in sarr]
            m.update({name:farr})
        return m
        
    def getMap2MapF(self,section,var):
        s = self.getS(section,var).split(" ")
        m = {}
        for x in s:
            x = x.split(":")
            name = x[0]
            sarr = x[1].split(",")
            ff = {}
            for ss in sarr:
                v = ss.split("=")
                ff.update({v[0]:float(v[1])})
            m.update({name:ff})
        return m
    
    def add(self,name,var):
        self.map.update( {name:var} )
    
    def set(self,fname,doprint=False):
        ### read
        self.read(fname)
        ### set
        self.add("dbg",  self.getB('RUN','dbg'))
        self.add("isMC", self.getB('RUN','isMC'))
        self.add("isFakeMC", self.getB('RUN','isFakeMC'))
        self.add("doVtx", self.getB('RUN','doVtx'))
        self.add("runtype", self.getS('RUN','runtype'))
        self.add("checkbadtriggers", self.getB('RUN','checkbadtriggers'))
        self.add("iszerosuppressed", self.getB('RUN','iszerosuppressed'))
        self.add("skipclustering", self.getB('RUN','skipclustering'))
        self.add("skiptracking", self.getB('RUN','skiptracking'))
        self.add("doprintout", self.getB('RUN','doprintout'))
        self.add("saveprimitive", self.getB('RUN','saveprimitive'))
        hfilesufx = "_multiprocess_histograms"
        if(self.map["skiptracking"]): hfilesufx += "_notrk"
        self.add("hfilesufx", hfilesufx)
        self.add("nmax2process", self.getI('RUN','nmax2process'))
        self.add("first2process", self.getI('RUN','first2process'))
        if(self.map["first2process"]<=0): self.map["first2process"] = 0
        self.add("nCPU", self.getI('RUN','nCPU'))
        self.add("nprintout", self.getI('RUN','nprintout'))
        self.add("skipmasking", self.getB('RUN','skipmasking'))
        self.add("inputfile", self.getS('RUN','inputfile'))
        self.add("detevtlib", self.getS('RUN','detevtlib'))
        
        self.add("runnums", self.getArrI('MULTIRUN','runnums'))

        self.add("npix_x", self.getI('CHIP','npix_x'))
        self.add("npix_y", self.getI('CHIP','npix_y'))
        self.add("pix_x",  self.getF('CHIP','pix_x'))
        self.add("pix_y",  self.getF('CHIP','pix_y'))
        self.add("chipX",  self.map["npix_x"]*self.map["pix_x"])
        self.add("chipY",  self.map["npix_y"]*self.map["pix_y"])
        ### x and y are swapped in the lab space
        self.add("npix_xlab", self.map["npix_y"])
        self.add("npix_ylab", self.map["npix_x"])
        self.add("pix_xlab",  self.map["pix_y"])
        self.add("pix_ylab",  self.map["pix_x"])
        self.add("chipXlab", self.map["chipY"]) 
        self.add("chipYlab", self.map["chipX"])
        
        self.add("cls_mult_low", self.getF('CLUSTERSMULT','cls_mult_low'))
        self.add("cls_mult_mid", self.getF('CLUSTERSMULT','cls_mult_mid'))
        self.add("cls_mult_hgh", self.getF('CLUSTERSMULT','cls_mult_hgh'))
        self.add("cls_mult_inf", self.getF('CLUSTERSMULT','cls_mult_inf'))
        
        self.add("seed_allow_negative_vertical_inclination", self.getB('SEED','seed_allow_negative_vertical_inclination'))
        self.add("seed_allow_neigbours", self.getB('SEED','seed_allow_neigbours'))
        self.add("seed_nmax_neigbours",  self.getI('SEED','seed_nmax_neigbours'))
        self.add("seed_nmiss_neigbours", self.getI('SEED','seed_nmiss_neigbours'))
        
        self.add("seed_thetax_range_low", self.getArrF('SEED','seed_thetax_range_low'))
        self.add("seed_thetax_range_mid", self.getArrF('SEED','seed_thetax_range_mid'))
        self.add("seed_thetax_range_hgh", self.getArrF('SEED','seed_thetax_range_hgh'))
        self.add("seed_thetax_range_inf", self.getArrF('SEED','seed_thetax_range_inf'))
        self.add("seed_rhox_range_low", self.getArrF('SEED','seed_rhox_range_low'))
        self.add("seed_rhox_range_mid", self.getArrF('SEED','seed_rhox_range_mid'))
        self.add("seed_rhox_range_hgh", self.getArrF('SEED','seed_rhox_range_hgh'))
        self.add("seed_rhox_range_inf",  self.getArrF('SEED','seed_rhox_range_inf'))
        self.add("seed_thetay_range_low", self.getArrF('SEED','seed_thetay_range_low'))
        self.add("seed_thetay_range_mid", self.getArrF('SEED','seed_thetay_range_mid'))
        self.add("seed_thetay_range_hgh", self.getArrF('SEED','seed_thetay_range_hgh'))
        self.add("seed_thetay_range_inf", self.getArrF('SEED','seed_thetay_range_inf'))
        self.add("seed_rhoy_range_low", self.getArrF('SEED','seed_rhoy_range_low'))
        self.add("seed_rhoy_range_mid", self.getArrF('SEED','seed_rhoy_range_mid'))
        self.add("seed_rhoy_range_hgh", self.getArrF('SEED','seed_rhoy_range_hgh'))
        self.add("seed_rhoy_range_inf", self.getArrF('SEED','seed_rhoy_range_inf'))
        
        self.add("seed_nbins_thetax_low", self.getI('SEED','seed_nbins_thetax_low'))
        self.add("seed_nbins_thetay_low", self.getI('SEED','seed_nbins_thetay_low'))
        self.add("seed_nbins_rhox_low",   self.getI('SEED','seed_nbins_rhox_low'))
        self.add("seed_nbins_rhoy_low",   self.getI('SEED','seed_nbins_rhoy_low'))
        self.add("seed_nbins_thetax_mid", self.getI('SEED','seed_nbins_thetax_mid'))
        self.add("seed_nbins_thetay_mid", self.getI('SEED','seed_nbins_thetay_mid'))
        self.add("seed_nbins_rhox_mid",   self.getI('SEED','seed_nbins_rhox_mid'))
        self.add("seed_nbins_rhoy_mid",   self.getI('SEED','seed_nbins_rhoy_mid'))
        self.add("seed_nbins_thetax_hgh", self.getI('SEED','seed_nbins_thetax_hgh'))
        self.add("seed_nbins_thetay_hgh", self.getI('SEED','seed_nbins_thetay_hgh'))
        self.add("seed_nbins_rhox_hgh",   self.getI('SEED','seed_nbins_rhox_hgh'))
        self.add("seed_nbins_rhoy_hgh",   self.getI('SEED','seed_nbins_rhoy_hgh'))
        self.add("seed_nbins_thetax_inf", self.getI('SEED','seed_nbins_thetax_inf'))
        self.add("seed_nbins_thetay_inf", self.getI('SEED','seed_nbins_thetay_inf'))
        self.add("seed_nbins_rhox_inf",   self.getI('SEED','seed_nbins_rhox_inf'))
        self.add("seed_nbins_rhoy_inf",   self.getI('SEED','seed_nbins_rhoy_inf'))

        
        self.add("lut_nbinsx_low", self.getI('LUT','lut_nbinsx_low'))
        self.add("lut_nbinsy_low", self.getI('LUT','lut_nbinsy_low'))
        self.add("lut_nbinsx_mid", self.getI('LUT','lut_nbinsx_mid'))
        self.add("lut_nbinsy_mid", self.getI('LUT','lut_nbinsy_mid'))
        self.add("lut_nbinsx_hgh", self.getI('LUT','lut_nbinsx_hgh'))
        self.add("lut_nbinsy_hgh", self.getI('LUT','lut_nbinsy_hgh'))
        self.add("lut_nbinsx_inf",  self.getI('LUT','lut_nbinsx_inf'))
        self.add("lut_nbinsy_inf",  self.getI('LUT','lut_nbinsy_inf'))
        self.add("lut_scaleX", self.getF('LUT','lut_scaleX'))
        self.add("lut_scaleY", self.getF('LUT','lut_scaleY'))
        
        self.add("xVtx", self.getF('VTX','xVtx'))
        self.add("yVtx", self.getF('VTX','yVtx'))
        self.add("zVtx", self.getF('VTX','zVtx'))
        self.add("exVtx", self.getF('VTX','exVtx'))
        self.add("eyVtx", self.getF('VTX','exVtx'))
        self.add("ezVtx", self.getF('VTX','exVtx'))

        self.add("ezCls", self.getF('CLUSTER','ezCls'))
        self.add("allow_diagonals", self.getB('CLUSTER','allow_diagonals'))

        self.add("worldbounds", self.getMapS2ArrF('WORLD','worldbounds'))
        world = {}
        for axis,bound in self.map["worldbounds"].items():
            bounds = [ bound[0], bound[1] ]
            world.update( {axis:bounds} )            
        self.add("world", world)

        self.add("pTrim", self.getF('NOISE','pTrim'))
        self.add("zeroSupp", self.getB('NOISE','zeroSupp'))
        self.add("nSigma", self.getF('NOISE','nSigma'))

        self.add("staves", self.getArrI('DETECTOR','staves'))
        self.add("layers", self.getI('DETECTOR','layers'))
        self.add("detectors", self.getArrS('DETECTOR','detectors'))
        self.add("det2stvchp", self.getMapS2T('DETECTOR','det2stvchp'))
        self.add("rdetectors", self.getMapS2ArrF('DETECTOR','rdetectors'))
        self.add("tandemlyrs", self.getMapI2ArrS('DETECTOR','tandemlyrs'))
        
        zmin=+1e20
        zmax=-1e20
        frstdet = ""
        lastdet = ""
        for det,r in self.map["rdetectors"].items():
            if(r[2]<zmin):
                zmin = r[2]
                frstdet = det
            if(r[2]>zmax):
                zmax = r[2]
                lastdet = det
        self.add("det_frst", frstdet)
        self.add("det_last", lastdet)
        
        stvchps = list(self.map["det2stvchp"].values())
        self.add("stvchps", stvchps)
        stvchp2det = {}
        for det,stvchp in self.map["det2stvchp"].items(): stvchp2det.update({stvchp:det})
        self.add("stvchp2det", stvchp2det)

        stv2det = {}
        det2stv = {}
        for s in self.map["staves"]: stv2det.update({s:[]})
        for stvchp,det in stvchp2det.items():
            stv = stvchp[0]
            stv2det[stv].append(det)
            det2stv.update({det:stv})
        self.add("stv2det", stv2det)
        self.add("det2stv", det2stv)
        
        det2tdm = {}
        for tdm,dets in self.map["tandemlyrs"].items():
            detA = dets[0]
            det2tdm.update({detA:tdm})
            if(len(dets)>1):
                detB = dets[1]
                det2tdm.update({detB:tdm})
        self.add("det2tdm", det2tdm)
        
        self.add("use_large_clserr_for_algnmnt", self.getB('ALIGNMENT','use_large_clserr_for_algnmnt'))
        self.add("misalignment", self.getMap2MapF('ALIGNMENT','misalignment'))
        self.add("minchi2align", self.getF('ALIGNMENT','minchi2align'))
        self.add("maxchi2align", self.getF('ALIGNMENT','maxchi2align'))
        self.add("axes2align", self.getS('ALIGNMENT','axes2align'))
        self.add("naligniter", self.getI('ALIGNMENT','naligniter'))
        self.add("alignmentbounds", self.getMap2MapF('ALIGNMENT','alignmentbounds'))
        self.add("alignmentmethod", self.getS('ALIGNMENT','alignmentmethod'))
        self.add("alignmentwerr", self.getB('ALIGNMENT','alignmentwerr'))
        self.add("alignmentmintrks", self.getI('ALIGNMENT','alignmentmintrks'))

        self.add("global_corr_thetax", self.getF('GLOBALALIGNMENT','global_corr_thetax')*np.pi/180. )
        self.add("global_corr_thetay", self.getF('GLOBALALIGNMENT','global_corr_thetay')*np.pi/180. )
        self.add("global_corr_thetaz", self.getF('GLOBALALIGNMENT','global_corr_thetaz')*np.pi/180. )
        self.add("global_corr_dx", self.getF('GLOBALALIGNMENT','global_corr_dx'))
        self.add("global_corr_dy", self.getF('GLOBALALIGNMENT','global_corr_dy'))
        self.add("global_corr_dz", self.getF('GLOBALALIGNMENT','global_corr_dz'))
        
        
        self.add("zWindow",       self.getF('WINDOW','zWindow'))
        self.add("xWindow",       self.getF('WINDOW','xWindow'))
        self.add("yWindowMin",    self.getF('WINDOW','yWindowMin'))
        self.add("xWindowWidth",  self.getF('WINDOW','xWindowWidth'))
        self.add("yWindowHeight", self.getF('WINDOW','yWindowHeight'))
        
        self.add("Rpipe", self.getF('BEAMPIPE','Rpipe'))
        self.add("yMidWin2PipeCenter", self.getF('BEAMPIPE','yMidWin2PipeCenter'))
        self.add("yZero2PipeTop", self.getF('BEAMPIPE','yZero2PipeTop'))
        
        self.add("fDipoleTesla", self.getF('DIPOLE','fDipoleTesla'))
        self.add("zDipoleLenghMeters", self.getF('DIPOLE','zDipoleLenghMeters'))
        self.add("zDipoleExit", self.getF('DIPOLE','zDipoleExit'))
        self.add("xDipoleExitMin", self.getF('DIPOLE','xDipoleExitMin'))
        self.add("xDipoleExitMax", self.getF('DIPOLE','xDipoleExitMax'))
        self.add("yDipoleExitMin", self.getF('DIPOLE','yDipoleExitMin'))
        self.add("yDipoleExitMax", self.getF('DIPOLE','yDipoleExitMax'))
        self.add("zFlangeExit", self.getF('DIPOLE','zFlangeExit'))
        self.add("xFlangeMin",  self.getF('DIPOLE','xFlangeMin'))
        self.add("xFlangeMax",  self.getF('DIPOLE','xFlangeMax'))
        self.add("yFlangeMin",  self.getF('DIPOLE','yFlangeMin'))
        self.add("yFlangeMax",  self.getF('DIPOLE','yFlangeMax'))
        
        thetax = self.getMapI2F('TRANSFORMATIONS','thetax')
        thetay = self.getMapI2F('TRANSFORMATIONS','thetay')
        thetaz = self.getMapI2F('TRANSFORMATIONS','thetaz')
        for istv in thetax.keys(): thetax[istv] = thetax[istv]*np.pi/180. ### can be different for different staves!
        for istv in thetay.keys(): thetay[istv] = thetay[istv]*np.pi/180. ### can be different for different staves!
        for istv in thetaz.keys(): thetaz[istv] = thetaz[istv]*np.pi/180. ### can be different for different staves!
        self.add("thetax", thetax)
        self.add("thetay", thetay)
        self.add("thetaz", thetaz)
        self.add("xOffset0", self.getF('TRANSFORMATIONS','xOffset0'))
        self.add("yOffset0", self.getF('TRANSFORMATIONS','yOffset0'))
        self.add("zOffset0", self.getF('TRANSFORMATIONS','zOffset0'))
        # self.add("yBoxBot2WinBot", self.getF('TRANSFORMATIONS','yBoxBot2WinBot'))
        self.add("yPipeTop2BoxBot", self.getF('TRANSFORMATIONS','yPipeTop2BoxBot'))
        self.add("yMidChip2BoxBot", self.getF('TRANSFORMATIONS','yMidChip2BoxBot'))
        self.add("zWin2Box", self.getF('TRANSFORMATIONS','zWin2Box'))
        self.add("zBox2chip", self.getF('TRANSFORMATIONS','zBox2chip'))
        
        xGlobalOffset = self.map["xOffset0"]
        self.add("xGlobalOffset", xGlobalOffset)
        yGlobalOffset = self.map["yOffset0"]+self.map["yZero2PipeTop"]+self.map["yPipeTop2BoxBot"]+self.map["yMidChip2BoxBot"]
        self.add("yGlobalOffset", yGlobalOffset)
        zGlobalOffset = self.map["zOffset0"]+self.map["zWin2Box"]+self.map["zBox2chip"]
        self.add("zGlobalOffset", zGlobalOffset)
        
        offsets_x = {}
        offsets_y = {}
        offsets_z = {}
        for det in self.map["detectors"]:
            offsets_x.update( {det:self.map["rdetectors"][det][0]} )
            offsets_y.update( {det:self.map["rdetectors"][det][1]} )
            offsets_z.update( {det:self.map["rdetectors"][det][2]} )
        self.add("offsets_x", offsets_x)
        self.add("offsets_y", offsets_y)
        self.add("offsets_z", offsets_z)
        
        self.add("fit_method",       self.getArrS('FIT','fit_method'))
        self.add("fit_chi2_fast",    self.getB('FIT',   'fit_chi2_fast'))
        self.add("fit_chi2_method0", self.getS('FIT',   'fit_chi2_method0'))
        self.add("fit_chi2_method1", self.getArrS('FIT','fit_chi2_method1'))
        
        self.add("cuts", self.getArrS('CUTS','cuts'))
        self.add("cut_windowaprtr", self.getB('CUTS','cut_windowaprtr'))
        self.add("cut_flangeaprtr", self.getB('CUTS','cut_flangeaprtr'))
        self.add("cut_dipoleaprtr", self.getB('CUTS','cut_dipoleaprtr'))
        self.add("cut_chi2dof", self.getF('CUTS','cut_chi2dof'))
        self.add("cut_dk_algn",       self.getB('CUTS','cut_dk_algn'))
        self.add("cut_dk_algn_det",   self.getS('CUTS','cut_dk_algn_det'))
        self.add("cut_dk_algn_dxmin", self.getF('CUTS','cut_dk_algn_dxmin'))
        self.add("cut_dk_algn_dxmax", self.getF('CUTS','cut_dk_algn_dxmax'))
        self.add("cut_dk_algn_dymin", self.getF('CUTS','cut_dk_algn_dymin'))
        self.add("cut_dk_algn_dymax", self.getF('CUTS','cut_dk_algn_dymax'))        
        self.add("cut_ROI_xmin", self.getF('CUTS','cut_ROI_xmin'))
        self.add("cut_ROI_xmax", self.getF('CUTS','cut_ROI_xmax'))
        self.add("cut_ROI_ymin", self.getF('CUTS','cut_ROI_ymin'))
        self.add("cut_ROI_ymax", self.getF('CUTS','cut_ROI_ymax'))
        self.add("cut_RoI_spot",           self.getB('CUTS','cut_RoI_spot'))
        self.add("cut_RoI_spot_xcenter",   self.getF('CUTS','cut_RoI_spot_xcenter'))
        self.add("cut_RoI_spot_ycenter",   self.getF('CUTS','cut_RoI_spot_ycenter'))
        self.add("cut_RoI_spot_radius_x",  self.getF('CUTS','cut_RoI_spot_radius_x'))
        self.add("cut_RoI_spot_radius_y",  self.getF('CUTS','cut_RoI_spot_radius_y'))
        self.add("cut_RoI_spot_theta_deg", self.getF('CUTS','cut_RoI_spot_theta_deg'))
        self.add("cut_RoI_btrfly", self.getB('CUTS','cut_RoI_btrfly'))
        self.add("cut_RoI_btrfly_xcenter", self.getF('CUTS','cut_RoI_btrfly_xcenter'))
        self.add("cut_RoI_btrfly_ycenter", self.getF('CUTS','cut_RoI_btrfly_ycenter'))
        self.add("cut_RoI_btrfly_long_radius", self.getF('CUTS','cut_RoI_btrfly_long_radius'))
        self.add("cut_RoI_btrfly_shrt_radius", self.getF('CUTS','cut_RoI_btrfly_shrt_radius'))
        self.add("cut_RoI_btrfly_theta_deg", self.getF('CUTS','cut_RoI_btrfly_theta_deg'))
        self.add("cut_RoI_btrfly_theta_curv", self.getF('CUTS','cut_RoI_btrfly_theta_curv'))
        self.add("cut_maxcls", self.getF('CUTS','cut_maxcls'))
        self.add("cut_allow_shared_clusters", self.getB('CUTS','cut_allow_shared_clusters'))
        self.add("cut_allow_negative_yz_slope", self.getB('CUTS','cut_allow_negative_yz_slope'))
        self.add("cut_spot",          self.getB('CUTS','cut_spot'))
        self.add("cut_spot_radius_x", self.getF('CUTS','cut_spot_radius_x'))
        self.add("cut_spot_radius_y", self.getF('CUTS','cut_spot_radius_y'))
        self.add("cut_spot_xcenter",  self.getF('CUTS','cut_spot_xcenter'))
        self.add("cut_spot_ycenter",  self.getF('CUTS','cut_spot_ycenter'))
        self.add("cut_strip",         self.getB('CUTS','cut_strip'))
        self.add("cut_strip_xcenter", self.getF('CUTS','cut_strip_xcenter'))
        self.add("cut_strip_ycenter", self.getF('CUTS','cut_strip_ycenter'))
        self.add("cut_strip_xwidth",  self.getF('CUTS','cut_strip_xwidth'))
        self.add("cut_strip_ywidth",  self.getF('CUTS','cut_strip_ywidth'))
        
        self.add("plot_online_evtdisp", self.getB('PLOT','plot_online_evtdisp'))
        self.add("plot_offline_evtdisp", self.getB('PLOT','plot_offline_evtdisp'))
    
        if(doprint):
            print("Configuration map:")
            for key,val in self.map.items():
                print(f"{key}: {val}")
            print("")

    def error(self,msg):
        sys.exit(msg)
    
    def is_non0_misalignment(self):
        non0_misalignment = False
        for key1 in self.map["misalignment"]:
            for key2 in self.map["misalignment"][key1]:
                if(self.map["misalignment"][key1][key2]!=0):
                    non0_misalignment = True
                    break
                if(non0_misalignment):
                    break
        return non0_misalignment
        
    def check_inputs(self):
        print(f"Checking config file integrity...")
        # if(self.map["use_large_clserr_for_algnmnt"] and self.is_non0_misalignment()):
        #     self.error(f"use_large_clserr_for_algnmnt can be 1 only if all misalignment parameters are set to zero")
        # if(not self.map["use_large_clserr_for_algnmnt"] and not self.is_non0_misalignment()):
        #     self.error(f"use_large_clserr_for_algnmnt can be 0 only if misalignment parameters are not all set to zero")
        # if(self.map["use_large_clserr_for_algnmnt"] and not self.map["seed_allow_negative_vertical_inclination"]):
        #     self.error(f"use_large_clserr_for_algnmnt must not be 1 if seed_allow_negative_vertical_inclination is 0")
        if(self.map["use_large_clserr_for_algnmnt"] and self.map["maxchi2align"]>1000):
            self.error(f'use_large_clserr_for_algnmnt must not be 1 if maxchi2align is>1000 (it is set to {self.map["maxchi2align"]})')
        if(self.map["use_large_clserr_for_algnmnt"] and self.map["fit_method"][0]!="SVD"):
            self.error(f'if use_large_clserr_for_algnmnt is 1 then fit_method must be SVD (it is set to {self.map["fit_method"]})')
        if(self.map["use_large_clserr_for_algnmnt"] and self.map["maxchi2align"]!=self.map["cut_chi2dof"]):
            self.error(f'if use_large_clserr_for_algnmnt is 1 then maxchi2align must be equal to cut_chi2dof')
        if(self.map["use_large_clserr_for_algnmnt"] and self.map["minchi2align"]==0):
            self.error(f'if use_large_clserr_for_algnmnt is 1 then minchi2align must be >0')
        # if(not self.map["use_large_clserr_for_algnmnt"] and self.map["cut_chi2dof"]>25):
        #     self.error(f'if use_large_clserr_for_algnmnt is 0 then cut_chi2dof should be smaller than 25 (it is set to {self.map["cut_chi2dof"]}) or you can adjust the condition in check_inputs()')
        if(self.map["use_large_clserr_for_algnmnt"] and self.map["minchi2align"]>=self.map["maxchi2align"]):
            self.error(f'if use_large_clserr_for_algnmnt is 1 then minchi2align cannot be greater than or equal to maxchi2align')
        print(f"Config file integrity check passed!")

    def __str__(self):
        print("Configuration map:")
        for key,val in self.map.items():
            print(f"{key}: {val}")
        print("")

############################################
############################################
############################################


def init_config(fname, show):
    """Entry point for the main script to start the singleton."""
    return Config(fname, show)
    
def show_config(cfg):
    print("----- Configuration map -----")
    for key,val in cfg.items():
        print(f"{key}: {val}")
    print("-----------------------------")
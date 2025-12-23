#!/usr/bin/python
import os
import math
import array
import numpy as np
import ROOT

from tracker_lib import config, utils

_s12_  = math.sqrt(12.)
_1s12_ = 1./_s12_

class Hit:
    # Explicitly define allowed attributes to save memory
    __slots__ = ('x', 'y', 'q', 'xmm', 'ymm', 'zmm', 'xTmm', 'yTmm', 'zTmm')
    def __init__(self,det,x,y,q=-1,xOrig=0,yOrig=0,xFake=0,yFake=0,Azx=0,Bzx=0,Azy=0,Bzy=0,Vx=0,Vy=0,Vz=0):
        cfg = config.Config().map
        self.x = x
        self.y = y
        self.q = q
        self.xmm = self.x*cfg["pix_x"]-cfg["chipX"]/2.
        self.ymm = self.y*cfg["pix_y"]-cfg["chipY"]/2.
        self.zmm = cfg["rdetectors"][det][2]
        ####################################
        ### trasformed to the real space ###
        ### includes the alignmet
        r0 = [self.xmm,self.ymm,self.zmm]
        rT0 = utils.transform_to_real_space(r0,det) 
        self.xTmm = rT0[0]
        self.yTmm = rT0[1]
        self.zTmm = rT0[2]
        
    def __str__(self):
        return f"Pixel: x={self.x}, y={self.y}, q={self.q}, r=({self.xmm,self.ymm,self.zmm}) [mm]"

class Cls:
    # Explicitly define allowed attributes to save memory
    __slots__ = ('det','SID','CID','DID','TID','pixels',
                'n','x','y','dx','dy','nx','ny','dxmm','dymm','xsizemm','ysizemm','xmm','ymm','zmm',
                'xTmm','yTmm','zTmm','dxTmm','dyTmm','xTsizemm','yTsizemm',
                'xTnoGmm','yTnoGmm','zTnoGmm','dxTnoGmm','dyTnoGmm','xTnoGsizemm','yTnoGsizemm')
    def __init__(self,det,pixels,CID):
        cfg = config.Config().map
        self.det = det
        self.SID = cfg["det2stvchp"][det][0] ### stave ID
        self.CID = CID ## chip ID
        self.DID = cfg["detectors"].index(det)
        self.TID = cfg["det2tdm"][det]
        self.pixels = pixels
        self.n = len(pixels)
        self.x,self.y,self.dx,self.dy,self.nx,self.ny = self.build(pixels) 
        self.dxmm = self.dx*cfg["pix_x"]
        self.dymm = self.dy*cfg["pix_y"]
        self.xsizemm = self.nx*cfg["pix_x"]/2.
        self.ysizemm = self.ny*cfg["pix_y"]/2.
        self.xmm = self.x*cfg["pix_x"]-cfg["chipX"]/2. ### original x
        self.ymm = self.y*cfg["pix_y"]-cfg["chipY"]/2. ### original y
        self.zmm  = cfg["rdetectors"][det][2]
        ####################################
        ### trasformed to the real space ###
        ### includes the alignmet and    ###
        ### the global displacement      ###
        r0 = [self.xmm,self.ymm,self.zmm]
        rT0 = utils.transform_to_real_space(r0,det) 
        self.xTmm = rT0[0]
        self.yTmm = rT0[1]
        self.zTmm = rT0[2] 
        self.dxTmm = self.dymm
        self.dyTmm = self.dxmm
        self.xTsizemm = self.ysizemm
        self.yTsizemm = self.xsizemm
        ####################################
        ### trasformed to the real space ###
        ### includes the alignmet but    ###
        ### but NOT the global displacements 
        rTnoG = utils.undo_global_offsets(rT0,det)
        self.xTnoGmm = rTnoG[0]
        self.yTnoGmm = rTnoG[1]
        self.zTnoGmm = rTnoG[2] 
        self.dxTnoGmm = self.dymm
        self.dyTnoGmm = self.dxmm
        self.xTnoGsizemm = self.ysizemm
        self.yTnoGsizemm = self.xsizemm
    ###    
    def build(self,pixels):
        if(self.n<1):
            print(f"cannot build a cluster from n={self.n} pixels. quitting.")
            quit()
        mu_x = 0
        mu_y = 0
        mu_x2 = 0
        mu_y2 = 0
        xmin = +1e10
        xmax = -1e10
        ymin = +1e10
        ymax = -1e10
        for pixel in pixels:
            mu_x  += pixel.x
            mu_y  += pixel.y
            mu_x2 += pixel.x**2
            mu_y2 += pixel.y**2
            xmin = pixel.x if(pixel.x<xmin) else xmin
            ymin = pixel.y if(pixel.y<ymin) else ymin
            xmax = pixel.x if(pixel.x>xmax) else xmax
            ymax = pixel.y if(pixel.y>ymax) else ymax
        nx = xmax-xmin if(xmax>xmin) else 1
        ny = ymax-ymin if(ymax>ymin) else 1
        mu_x  = mu_x/self.n
        mu_y  = mu_y/self.n
        mu_x2 = mu_x2/self.n
        mu_y2 = mu_y2/self.n
        varx  = mu_x2-mu_x**2
        vary  = mu_y2-mu_y**2
        se_x  = _1s12_/math.sqrt(self.n)
        se_y  = _1s12_/math.sqrt(self.n)
        return mu_x,mu_y,se_x,se_y,nx,ny
    def __str__(self):
        # for p in self.pixels: print(p)
        return f"Cluster: xy={self.x,self.y} [pixels], r={self.xmm,self.ymm,self.zmm} [mm], size={self.n}"

class MCparticle:
    # Explicitly define allowed attributes to save memory
    __slots__ = ('pdg', 'pos1', 'pos2')
    def __init__(self,det,pdg,loc_start,loc_end):
        cfg = config.Config().map
        self.pdg = pdg
        self.pos1 = ROOT.Math.XYZPoint( loc_start.X()-cfg["pix_x"]*cfg["npix_x"]/2., loc_start.Y()-cfg["pix_y"]*cfg["npix_y"]/2., cfg["rdetectors"][det][2] )
        self.pos2 = ROOT.Math.XYZPoint( loc_end.X()-cfg["pix_x"]*cfg["npix_x"]/2.,   loc_end.Y()-cfg["pix_y"]*cfg["npix_y"]/2.,   cfg["rdetectors"][det][2] )
    def __str__(self):
        return f"MCparticle: pdg={self.pdg}, pos1=({self.pos1.X(),self.pos1.Y(),self.pos1.Z()}), pos2=({self.pos2.X(),self.loc_end.Y(),self.pos2.Z()})"

class FakeMCparticle:
    # Explicitly define allowed attributes to save memory
    __slots__ = ('slp', 'itp', 'vtx')
    def __init__(self,slp,itp,vtx):
        self.slp = slp
        self.itp = itp
        self.vtx = vtx
    def __str__(self):
        return f"FakeMCparticle: slp={self.slp}, itp={self.itp}, vtx={vtx}"

class TrackSeed:
    # Explicitly define allowed attributes to save memory
    __slots__ = ('detectors', 'clsids', 'tunnelid', 'hough_coords', 'x','y','z','dx','dy','xsize','ysize')
    def __init__(self,seed,tunnelid,hough_coords,clusters):
        self.detectors    = list(seed.keys())
        self.clsids       = seed
        self.tunnelid     = tunnelid
        self.hough_coords = hough_coords
        self.x  = {}
        self.y  = {}
        self.z  = {}
        self.dx = {}
        self.dy = {}
        self.xsize = {}
        self.ysize = {}
        for det,icls in seed.items():
            self.x.update({  det:clusters[det][icls].xTmm  })
            self.y.update({  det:clusters[det][icls].yTmm  })
            self.z.update({  det:clusters[det][icls].zTmm  })
            self.dx.update({ det:clusters[det][icls].dxTmm })
            self.dy.update({ det:clusters[det][icls].dyTmm })
            self.xsize.update({ det:clusters[det][icls].xTsizemm })
            self.ysize.update({ det:clusters[det][icls].yTsizemm })
    def __str__(self):
        return f"TrackSeed: "

class Track:
    # Explicitly define allowed attributes to save memory
    __slots__ = ('detectors', 'trkcls', 'points', 'errors', 'chisq', 'ndof', 'chi2ndof',
                 'direction', 'centroid', 'params', 'success', 'hough_coords', 'theta', 'phi', 'maxcls')
    def __init__(self,detectors,trkcls,points,errors,chisq,ndof,direction,centroid,params,success,hough_coords={}):
        self.detectors = detectors
        self.trkcls = trkcls
        self.points = points
        self.errors = errors
        self.chisq = chisq
        self.ndof = ndof
        self.chi2ndof = chisq/ndof if(ndof>0) else 99999
        self.direction = direction
        self.centroid = centroid
        self.params = params
        self.success = success
        self.hough_coords = hough_coords
        self.theta,self.phi = self.angles(direction)
        self.maxcls = self.max_cls_size()
    def angles(self,direction):
        dx = direction[0]
        dy = direction[1]
        dz = direction[2]
        theta = np.arctan(np.sqrt(dx*dx+dy*dy)/dz)
        phi   = np.arctan(dy/dx)
        return theta,phi
    def max_cls_size(self):
        maxcls = 0
        for det,cl in self.trkcls.items():
            if(cl.n>maxcls): maxcls = cl.n
        return maxcls
    def __str__(self):
        return f"Track: chisq={self.chisq}, ndof={self.ndof}, chi2ndof={self.chi2ndof}"

class Meta:
    # Explicitly define allowed attributes to save memory
    __slots__ = ('run', 'start', 'end', 'dur')
    def __init__(self,run,start,end,dur):
        self.run = run
        self.start = start
        self.end = end
        self.dur = dur
    def __str__(self):
        return f"Meta: "
        
class Magnets:
    # Explicitly define allowed attributes to save memory
    __slots__ = ('ThetaB', 'dipole', 'quad0','quad1','quad2','m12','m34','zobj','zimg','xcor')
    def __init__(self,dipole_in_GeV,quad0,quad1,quad2,m12,m34,zobj,zimg,xcor):
        cfg = config.Config().map
        self.ThetaB = 0.006 ### mrad
        self.dipole = self.get_dipole(dipole_in_GeV)
        self.quad0 = quad0
        self.quad1 = quad1
        self.quad2 = quad2
        self.m12   = m12
        self.m34   = m34
        self.zobj  = zobj
        self.zimg  = zimg
        self.xcor  = xcor
    def get_dipole(self,Dipole_settings_in_GeV):
        cfg = config.Config().maps
        B_in_Tesla = Dipole_settings_in_GeV*math.sin(self.ThetaB)/(0.3*cfg["zDipoleLenghMeters"])
        return B_in_Tesla
    def __str__(self):
        return f"Magnets: Dipole={self.dipole} [T], Q0={self.quad0} [kG/m], Q1={self.quad1} [kG/m], Q2={self.quad2} [kG/m], M12={self.m12}, M34={self.m34}, XCOR={self.xcor}"

class Event:
    # Explicitly define allowed attributes to save memory
    __slots__ = ('saveprimitive', 'meta', 'trigger', 'timestamp_bgn', 'timestamp_end', 'magnets', 'errors',
                 'pixels', 'npixels', 'clusters', 'nclusters', 'ntunnels', 'hough_space',
                 'seeds', 'tracks', 'misalignment')
    def __init__(self,meta,trigger,timestamp_bgn,timestamp_end,magnets,saveprimitive=True):
        cfg = config.Config().map 
        self.saveprimitive   = saveprimitive
        self.meta            = meta
        self.trigger         = trigger
        self.timestamp_bgn   = timestamp_bgn
        self.timestamp_end   = timestamp_end
        self.magnets         = magnets
        self.errors          = {}
        self.pixels          = {}
        self.npixels         = {}
        self.clusters        = {}
        self.nclusters       = {}
        self.ntunnels        = -1
        self.hough_space     = {} 
        self.seeds           = []
        self.tracks          = []
        self.misalignment    = cfg["misalignment"]
    def __str__(self):
        return f"Event: npixels={self.npixels}, tracks={self.tracks}"
    def set_event_errors(self,errors):
        self.errors = errors
    def set_event_pixels(self,pixels):
        cfg = config.Config().map
        for det in cfg["detectors"]: self.npixels.update({det:len(pixels[det])})
        self.pixels = pixels.copy() if(self.saveprimitive) else {}
    def set_event_clusters(self,clusters):
        cfg = config.Config().map
        for det in cfg["detectors"]: self.nclusters.update({det:len(clusters[det])})
        self.clusters = clusters if(self.saveprimitive) else {}
    def set_ntunnels(self,ntunnels):
        self.ntunnels = ntunnels
    def set_event_seeds(self,seeds,hough_space={}):
        self.seeds = seeds
        self.hough_space = hough_space
    def set_event_tracks(self,tracks):
        self.tracks = tracks
        
class MinimalEvent:
    def __init__(self,trigger,tracks):
        self.trigger = trigger
        self.tracks  = tracks

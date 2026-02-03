import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import time
import pickle
import multiprocessing
from matplotlib.animation import FuncAnimation, PillowWriter
import math
import statistics
import ROOT
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
ROOT.gStyle.SetGridColor(ROOT.kGray)
ROOT.gStyle.SetGridWidth(1)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tracker_lib import bremss as br


import argparse
parser = argparse.ArgumentParser(description='generator.py...')
parser.add_argument('-mag', metavar='magnets settings (run 502 or run 490)', required=True,  help='magnets settings (run 502 or run 490)')
parser.add_argument('-gen', metavar='particles to generate', required=True,  help='particles to generate')
parser.add_argument('-acc', metavar='require full acceptance?', required=False,  help='require full acceptance?')
parser.add_argument('-mlt', metavar='multi processing?', required=False,  help='multi processing?')
parser.add_argument('-shw', metavar='show plots?', required=False,  help='show plots?')
parser.add_argument('-gif', metavar='do gif?', required=False,  help='do gif?')
argus = parser.parse_args()
MagnetsSettings = float(argus.mag)
magset = [502, 490.0,490.1,490.2,490.5, 729]
if(MagnetsSettings not in magset):
    print(f"Unsupported magnets settings run: {MagnetsSettings}")
    quit()
Nparticles = int(argus.gen)
if(Nparticles<=0):
    print(f"Unsupported Nparticles: {Nparticles}")
    quit()
fullacc = True if(argus.acc is not None and argus.acc=="1") else False
mltprc  = True if(argus.mlt is not None and argus.mlt=="1") else False
doshw   = True if(argus.shw is not None and argus.shw=="1") else False
dogif   = True if(argus.gif is not None and argus.gif=="1") else False



plt.rcParams['image.cmap'] = 'afmhot'
# plt.rcParams['image.cmap'] = 'copper'
plt.rcParams['text.usetex'] = True

# Convert units
m_to_cm  = 1e2
m_to_mm  = 1e3
m_to_um  = 1e6
cm_to_mm = 1e1
cm_to_um = 1e4
cm_to_m  = 1e-2
mm_to_m  = 1e-3
mm_to_cm = 1e-1
mm_to_um = 1e3
um_to_mm = 1e-3
um_to_cm = 1e-4
um_to_m  = 1e-6
kG_to_T  = 0.1
GeV_to_kgms   = 5.39e-19
GeV_to_kg     = 1.8e-27
GeV_to_kgm2s2 = 1.6e-10

# Physical constants
c   = 299792458  # speed of light in m/s
c2  = c*c
e   = 1.602176634e-19  # elementary charge in C
m_e = 9.1093837015e-31  # electron/positron mass in kg
m_p = 1.67262192e-27 # proton/antiproton mass in kg


##################################
######### configurations #########
##################################
fx0     = 0*um_to_m ### TODO??? ### this is where the beam is shot from
fy0     = 0*um_to_m ### TODO??? ### this is where the beam is shot from
# fx0     = 0*um_to_m ### TODO??? ### this is where the beam is shot from
# fy0     = 0*um_to_m ### TODO??? ### this is where the beam is shot from
fz0     = -200*cm_to_m ### fixed, just has to be before the Be window ### this is where the beam is shot from
fsigmax = 50*um_to_m ## beam sigma
fsigmay = 50*um_to_m ## beam sigma
fsigmaz = 150*um_to_m ## beam sigma
zAL     = +30 ### the aluminum foil, cm
zBe     = -84 ### the beryllium window, cm
Z0      = zBe if(MagnetsSettings==502) else zAL
Z0_m    = Z0*cm_to_m
MM      = m_e ## kg, positron
QQ      = +1  ## unit charge, positron
mGeV    = (MM*c2)/GeV_to_kgm2s2 ## GeV
E_GeV   = 10 # GeV, energy of primary partticles
Emin    = 1e-2 #1 ## GeV
Emax    = 10 ## GeV
smearT  = False #True
smearP  = True
truncate_smear_pT = False
smear_sigma_T_um  = 1e-4 ## um
smear_sigma_P_GeV = 1e-4 ## GeV
ZMAX    = 18 ## METERES
tmax    = ZMAX / (0.99 * c) ### time range for propagation (seconds): approximate time to travel 18 meters (last detector is at ~18 meters, relativistic particles going ~c)
t_span  = (0, tmax)
max_dt  = 1e-9
dy_det  = 0 # cm
X0 = None
t_cm = None
if(MagnetsSettings==502):
    X0 = 35.3 # cm for Beryllium
    t_cm = 0.005  # cm (50 µm)
else:
    X0 = 8.897 # cm for Aluminum
    t_cm = 0.01  # cm (100 µm)
E_vals, photons, eplus = br.build_pdfs(Emin,E_GeV,X0,t_cm)
rng = np.random.default_rng(123)

### quads
quadsvals = {502:[-7.637,28.55,-7.637],
             490.0:[-30.68,46.42,-30.68],
             490.1:[-27.99,44.98,-27.99],
             490.2:[-20.38,40.42,-20.38],
             490.5:[-6.66,28.86,-6.66],
             729:[-24.93,36.95,-24.93]
}
### dipoles
dipolesvals = {502:  {"dipole":[0.219,0,0],"xcorr":[0,+0.026107,0]},
               490.0:{"dipole":[0.219,0,0],"xcorr":[0,+0.026107,0]},
               490.1:{"dipole":[0.219,0,0],"xcorr":[0,+0.026107,0]},
               490.2:{"dipole":[0.219,0,0],"xcorr":[0,+0.026107,0]},
               490.5:{"dipole":[0.219,0,0],"xcorr":[0,+0.026107,0]},
               729:  {"dipole":[0.175,0,0],"xcorr":[0,-0.0278,0]}
}

### misalignments
magsetdelt = {"quad0":[0,0],   "quad1":[0,0],   "quad2":[0,0],   "xcorr":[0,0],   "dipole":[0,0]} ### x,y displacement, cm
magsetangl = {"quad0":[0,0,0], "quad1":[0,0,0], "quad2":[0,0,0], "xcorr":[0,0,0], "dipole":[0,0,0]} ### 3D rotation, degrees
detangl    = [0,0,0]
for name,angles in magsetangl.items():
    for i in range(3):
        angles[i] = angles[i]*np.pi/180.
for i in range(3):
    detangl[i] = detangl[i]*np.pi/180.

##################################
##################################
##################################

### chip
npix_x = 1024
npix_y = 512
pix_x  = 0.02924
pix_y  = 0.02688
chipXmm = npix_x*pix_x
chipYmm = npix_y*pix_y
chipXcm = chipXmm*mm_to_cm
chipYcm = chipYmm*mm_to_cm
chipXm  = chipXmm*mm_to_m
chipYm  = chipYmm*mm_to_m
pix_x_nbins = npix_x+1
pix_x_min   = -0.5
pix_x_max   = npix_x+0.5
pix_y_nbins = npix_y+1
pix_y_min   = -0.5
pix_y_max   = npix_y+0.5
rebin2D     = 12
rebin2D_mid = 4
hPixelMatrix1 = ROOT.TH1D("h_pix1",";Pixel;Hits",npix_x*npix_y,1,npix_x*npix_y+1)
hPixelMatrix2 = ROOT.TH2D("h_pix2",";pixel-x;pixel-y;Hits",pix_x_nbins,pix_x_min,pix_x_max, pix_y_nbins,pix_y_min,pix_y_max)
hPixelMatrix1_middle = ROOT.TH1D("h_pix1_middle",";Pixel;Hits",int(npix_x/rebin2D_mid)*int(npix_y/rebin2D_mid),1,int(npix_x/rebin2D_mid)*int(npix_y/rebin2D_mid)+1)
hPixelMatrix2_middle = ROOT.TH2D("h_pix2_middle",";pixel-x;pixel-y;Hits",int(npix_x/rebin2D_mid)+1,pix_x_min,pix_x_max, int(npix_y/rebin2D_mid)+1,pix_y_min,pix_y_max)
hPixelMatrix1_coarse = ROOT.TH1D("h_pix1_coarse",";Pixel;Hits",int(npix_x/rebin2D)*int(npix_y/rebin2D),1,int(npix_x/rebin2D)*int(npix_y/rebin2D)+1)
hPixelMatrix2_coarse = ROOT.TH2D("h_pix2_coarse",";pixel-x;pixel-y;Hits",int(npix_x/rebin2D)+1,pix_x_min,pix_x_max, int(npix_y/rebin2D)+1,pix_y_min,pix_y_max)
hPz_full  = ROOT.TH1D("hPz_full", ";p_{z} [GeV];Particles",100,0,10)
hPz_small = ROOT.TH1D("hPz_small",";p_{z} [GeV];Particles",50,1.5,4.5)
hPz_zoom  = ROOT.TH1D("hPz_zoom", ";p_{z} [GeV];Particles",40,1.5,3.5)

hPx_full  = ROOT.TH1D("hPx_full", ";p_{x} [GeV];Particles",100,-1e-2,+1e-2)
hPx_small = ROOT.TH1D("hPx_small",";p_{x} [GeV];Particles",50, -1e-3,+1e-3)
hPx_zoom  = ROOT.TH1D("hPx_zoom", ";p_{x} [GeV];Particles",40, -1e-4,+1e-4)

hPy_full  = ROOT.TH1D("hPy_full", ";p_{y} [GeV];Particles",100,-1e-2,+1e-2)
hPy_small = ROOT.TH1D("hPy_small",";p_{y} [GeV];Particles",50, -1e-3,+1e-3)
hPy_zoom  = ROOT.TH1D("hPy_zoom", ";p_{y} [GeV];Particles",40, -1e-4,+1e-4)

# Define detector x range
# detector_x_center_cm = -1.0 # cm
detector_x_center_cm = 0. if(MagnetsSettings!=729) else +0.4 # cm
detector_x_center_m = detector_x_center_cm*cm_to_m

# Define detector y range
detector_y_center_cm = 5.165 + 0.1525 + 3.685 + dy_det # cm
detector_y_center_m  = detector_y_center_cm*cm_to_m

# Calculate detector z position
detector_z_base_cm = 1363 + 303.2155 + 11.43 + 1.05  # cm
detector_z_base_m  = detector_z_base_cm*cm_to_m
detector_z_base_mm = detector_z_base_cm*cm_to_mm


def spot_cut(x,y,xcent,ycent,xrad,yrad):
    CX = xcent
    CY = ycent
    RX = xrad
    RY = yrad
    X = (x-CX)/RX
    Y = (y-CY)/RY
    X2 = X*X
    Y2 = Y*Y
    if( (X2+Y2)>1. ): return False
    return True

def Rot3D(u,thetax=0,thetay=0,thetaz=0):
    Rx = [[1,0,0],[0,math.cos(thetax),-math.sin(thetax)], [0,math.sin(thetax),math.cos(thetax)]]
    Ry = [[math.cos(thetay),0,math.sin(thetay)], [0,1,0], [-math.sin(thetay),0,math.cos(thetay)]]
    Rz = [[math.cos(thetaz),math.sin(thetaz),0], [-math.sin(thetaz),math.cos(thetaz),0], [0,0,1]]
    ### rotate around x
    vx = [0,0,0]
    vx[0] = Rx[0][0]*u[0]+Rx[0][1]*u[1]+Rx[0][2]*u[2]
    vx[1] = Rx[1][0]*u[0]+Rx[1][1]*u[1]+Rx[1][2]*u[2]
    vx[2] = Rx[2][0]*u[0]+Rx[2][1]*u[1]+Rx[2][2]*u[2]
    ### rotate around y
    vy = [0,0,0]
    vy[0] = Ry[0][0]*vx[0]+Ry[0][1]*vx[1]+Ry[0][2]*vx[2]
    vy[1] = Ry[1][0]*vx[0]+Ry[1][1]*vx[1]+Ry[1][2]*vx[2]
    vy[2] = Ry[2][0]*vx[0]+Ry[2][1]*vx[1]+Ry[2][2]*vx[2]
    ### rotate around z
    vz = [0,0,0]
    vz[0] = Rz[0][0]*vy[0]+Rz[0][1]*vy[1]+Rz[0][2]*vy[2]
    vz[1] = Rz[1][0]*vy[0]+Rz[1][1]*vy[1]+Rz[1][2]*vy[2]
    vz[2] = Rz[2][0]*vy[0]+Rz[2][1]*vy[1]+Rz[2][2]*vy[2]
    ### result
    return vz



# Define magnet elements
class Element:
    def __init__(self, name, x_min, x_max, y_min, y_max, z_min, z_max, angles=[]):
        self.name  = name
        self.x_min = x_min * cm_to_m
        self.x_max = x_max * cm_to_m
        self.y_min = y_min * cm_to_m
        self.y_max = y_max * cm_to_m
        self.z_min = z_min * cm_to_m
        self.z_max = z_max * cm_to_m
        self.hits = []
        
        if(len(angles)==3):
            umin = [self.x_min,self.y_min,self.z_min]
            umax = [self.x_max,self.y_max,self.z_max]
            vmin = Rot3D(umin,thetax=angles[0],thetay=angles[1],thetaz=angles[2])
            vmax = Rot3D(umax,thetax=angles[0],thetay=angles[1],thetaz=angles[2])
            self.x_min = vmin[0]
            self.y_min = vmin[1]
            self.z_min = vmin[2]
            self.x_max = vmax[0]
            self.y_max = vmax[1]
            self.z_max = vmax[2]
        
    def is_inside(self, x, y, z):
        return (self.x_min <= x <= self.x_max and 
                self.y_min <= y <= self.y_max and 
                self.z_min <= z <= self.z_max)
                
    def get_vertices(self):
        vertices = np.array([
            [self.x_min, self.y_min, self.z_min],
            [self.x_max, self.y_min, self.z_min],
            [self.x_max, self.y_max, self.z_min],
            [self.x_min, self.y_max, self.z_min],
            [self.x_min, self.y_min, self.z_max],
            [self.x_max, self.y_min, self.z_max],
            [self.x_max, self.y_max, self.z_max],
            [self.x_min, self.y_max, self.z_max]
        ])
        return vertices
    
    def record_hit(self, particle_id, x, y, z, px, py, pz):
        self.hits.append({
            'particle_id': particle_id,
            'x': x, 'y': y, 'z': z,
            'px': px, 'py': py, 'pz': pz
        })
        
    def plot_element(self, ax, col, alpha=0.2):
        vertices = self.get_vertices()
        
        # List of sides' vertices
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],  # Bottom face
            [vertices[4], vertices[5], vertices[6], vertices[7]],  # Top face
            [vertices[0], vertices[1], vertices[5], vertices[4]],  # Front face
            [vertices[2], vertices[3], vertices[7], vertices[6]],  # Back face
            [vertices[1], vertices[2], vertices[6], vertices[5]],  # Right face
            [vertices[0], vertices[3], vertices[7], vertices[4]]   # Left face
        ]
        
        # Plot sides
        for face in faces:
            face = np.array(face)
            ax.plot_surface(
                face[:, 0].reshape(2, 2),
                face[:, 1].reshape(2, 2),
                face[:, 2].reshape(2, 2),
                color=col, alpha=alpha
            )


class Dipole(Element):
    def __init__(self, name, x_min, x_max, y_min, y_max, z_min, z_max, B_x,B_y,B_z, angles=[]):
        super().__init__(name, x_min, x_max, y_min, y_max, z_min, z_max, angles)
        self.name = name
        self.B_x = B_x  # Tesla
        self.B_y = B_y  # Tesla
        self.B_z = B_z  # Tesla
        # self.hits = []
        self.angles = angles
        
        if(len(self.angles)==3):
            uB = [self.B_x,self.B_y,self.B_z]
            vB = Rot3D(uB,thetax=self.angles[0],thetay=self.angles[1],thetaz=self.angles[2])
            self.B_x = vB[0]
            self.B_y = vB[1]
            self.B_z = vB[2]
        
    def field(self, x, y, z):
        if self.is_inside(x, y, z):
            return np.array([self.B_x, self.B_y, self.B_z])
        else:
            return np.array([0, 0, 0])


class Quadrupole(Element):
    def __init__(self, name, x_min, x_max, y_min, y_max, z_min, z_max, gradient, angles=[]):
        super().__init__(name, x_min, x_max, y_min, y_max, z_min, z_max, angles)
        self.name = name
        self.gradient = gradient * kG_to_T  # Tesla/m
        # self.hits = []
        self.angles = angles

    def field(self, x, y, z):
        if self.is_inside(x, y, z):
            ### calculate the center of the magnet volume
            x_center = (self.x_min + self.x_max) / 2
            y_center = (self.y_min + self.y_max) / 2
            ### field relative to the magnetic center
            B_x = self.gradient * (y - y_center)
            B_y = self.gradient * (x - x_center)
            B_z = 0
            if(len(self.angles)==3):
                uB = [B_x,B_y,0]
                vB = Rot3D(uB,thetax=self.angles[0],thetay=self.angles[1],thetaz=self.angles[2])
                B_x = vB[0]
                B_y = vB[1]
                B_z = vB[2]
            
            return np.array([B_x, B_y, B_z])
        else:
            return np.array([0, 0, 0])


class Detector(Element):
    def __init__(self, name, x_min, x_max, y_min, y_max, z_min, z_max, angles=[]):
        super().__init__(name, x_min, x_max, y_min, y_max, z_min, z_max, angles)
        self.name = name
        self.z_pos = z_min * cm_to_m
        # self.hits = []  # To store particle hits
        self.angles = angles
        
    def field(self, x, y, z):
        # Detectors don't generate magnetic fields
        return np.array([0, 0, 0])
        
    def plot_element(self, ax, col, alpha=0.4):
        # Override to plot as a thin rectangular plane
        vertices = self.get_vertices()
        
        # Create a rectangular plane
        x = [vertices[0][0], vertices[1][0], vertices[2][0], vertices[3][0], vertices[4][0]]
        y = [vertices[0][1], vertices[1][1], vertices[2][1], vertices[3][1], vertices[4][1]]
        z = [vertices[0][2], vertices[1][2], vertices[2][2], vertices[3][2], vertices[4][2]]
        
        L1verts = []
        L1verts.append( np.array([ [x[0],y[0],z[0]],
                                   [x[1],y[1],z[1]],
                                   [x[2],y[2],z[2]],
                                   [x[3],y[3],z[3]] ]) )
        ax.add_collection3d(Poly3DCollection(L1verts, facecolors=col, linewidths=0.5, edgecolors=col, alpha=.20))
        
    def print_hits(self,initial_states):
        if not self.hits:
            print(f"Detector at z={self.z_pos:.2f} m: No hits recorded")
            return
        
        print(f"Detector at z={self.z_pos:.2f} m hits:")
        for hit in self.hits:
            pid = hit['particle_id']
            xx  = hit['x']
            yy  = hit['y']
            pz  = initial_states[pid][5]/GeV_to_kgms
            print(f"  Particle {pid}: x={xx:.6f} m, y={yy:.6f} m (pz={pz:.2f} GeV)")


class Beampipe(Element):
    def __init__(self, name, inner_radius_cm, outer_radius_cm, z_min_cm, z_max_cm, angles=[]):
        # For bounding box, use outer radius to define x,y limits
        super().__init__(name, -outer_radius_cm, outer_radius_cm, 
                         -outer_radius_cm, outer_radius_cm, 
                         z_min_cm, z_max_cm, angles)
        self.name = name
        self.inner_radius = inner_radius_cm * cm_to_m
        self.outer_radius = outer_radius_cm * cm_to_m
        self.z_min_pipe = z_min_cm * cm_to_m
        self.z_max_pipe = z_max_cm * cm_to_m
        # self.hits = []
        
    def field(self, x, y, z):
        # Beampipe doesn't generate magnetic fields
        return np.array([0, 0, 0])
        
    def is_inside_pipe_material(self, x, y, z):
        """Check if point is inside the pipe material (between inner and outer radius)"""
        if not (self.z_min_pipe <= z <= self.z_max_pipe):
            return False
        
        r = np.sqrt(x**2 + y**2)
        return self.inner_radius <= r <= self.outer_radius
    
    def is_outside_vacuum(self, x, y, z):
        """Check if point is outside the vacuum region (beyond inner radius)"""
        if not (self.z_min_pipe <= z <= self.z_max_pipe):
            return False
        
        r = np.sqrt(x**2 + y**2)
        return r > self.inner_radius
        
    def plot_element(self, ax, color='gray', alpha=0.3):
        """Plot beampipe as a semi-transparent cylinder"""
        # Create cylindrical surface
        z_pipe = np.linspace(self.z_min_pipe, self.z_max_pipe, 50)
        theta = np.linspace(0, 2*np.pi, 50)
        
        # Create meshgrid for outer surface
        Z_outer, THETA_outer = np.meshgrid(z_pipe, theta)
        X_outer = self.outer_radius * np.cos(THETA_outer)
        Y_outer = self.outer_radius * np.sin(THETA_outer)
        
        # Create meshgrid for inner surface  
        X_inner = self.inner_radius * np.cos(THETA_outer)
        Y_inner = self.inner_radius * np.sin(THETA_outer)
        
        # Plot outer surface
        ax.plot_surface(X_outer, Y_outer, Z_outer, 
                       color=color, alpha=alpha, linewidth=0)
        
        # Plot inner surface (slightly more transparent)
        ax.plot_surface(X_inner, Y_inner, Z_outer, 
                       color=color, alpha=alpha*0.5, linewidth=0)
        
        # Add end caps
        r_cap = np.linspace(self.inner_radius, self.outer_radius, 20)
        theta_cap = np.linspace(0, 2*np.pi, 50)
        R_cap, THETA_cap = np.meshgrid(r_cap, theta_cap)
        X_cap = R_cap * np.cos(THETA_cap)
        Y_cap = R_cap * np.sin(THETA_cap)
        
        # Front cap
        Z_cap_front = np.full_like(X_cap, self.z_min_pipe)
        ax.plot_surface(X_cap, Y_cap, Z_cap_front, 
                       color=color, alpha=alpha, linewidth=0)
        
        # Back cap  
        Z_cap_back = np.full_like(X_cap, self.z_max_pipe)
        ax.plot_surface(X_cap, Y_cap, Z_cap_back, 
                       color=color, alpha=alpha, linewidth=0)



########################################################################
########################################################################

def GenerateGaussianBeam(E_GeV,mass_GeV,charge,mks=False):
    fbeamfocus  = 0
    lf          = E_GeV/mass_GeV
    femittancex = 50e-3*mm_to_m/lf ### mm-rad
    femittancey = 50e-3*mm_to_m/lf ### mm-rad
    fbetax      = (fsigmax**2)/femittancex
    fbetay      = (fsigmay**2)/femittancey
    ### z
    z0     = np.random.normal(fz0,fsigmaz)
    zdrift = z0 - fbeamfocus ### correct drift distance for x, y distribution. Forces the beam to pass through the IP (i.e. focuesd at z=0)
    ### x
    sigmax  = fsigmax * np.sqrt(1.0 + (zdrift/fbetax)**2)
    x0      = np.random.normal(fx0, sigmax)
    meandx  = x0*zdrift / (zdrift**2 + fbetax**2)
    sigmadx = np.sqrt( femittancex*fbetax / (zdrift**2 + fbetax**2) )
    dx0     = np.random.normal(meandx, sigmadx)
    ### y
    sigmay  = fsigmay * np.sqrt(1.0 + (zdrift/fbetay)**2)
    y0      = np.random.normal(fy0, sigmay)
    meandy  = y0*zdrift / (zdrift**2 + fbetay**2)
    sigmady = np.sqrt( femittancey*fbetay / (zdrift**2 + fbetay**2) )
    dy0     = np.random.normal(meandy, sigmady)
    ### p
    pz = np.sqrt( (E_GeV**2 - mass_GeV**2)/ (dx0**2 + dy0**2 + 1.0) )
    px = dx0*pz
    py = dy0*pz
    pz0 = pz*GeV_to_kgms # kg*m/s
    px0 = px*GeV_to_kgms # kg*m/s
    py0 = py*GeV_to_kgms # kg*m/s
    mass_kg = mass_GeV*GeV_to_kgm2s2/c2 # kg
    ### state
    state_mks = [x0,y0,z0, px0,py0,pz0, mass_kg,charge] ### [x[m],y[m],z[m], px[kg*m/s],py[kg*m/s],pz[kg*m/s], m[kg],q[unit]]
    state_nat = [x0,y0,z0, px,py,pz, mass_GeV,charge]   ### [x[m],y[m],z[m], px[GeV],py[GeV],pz[GeV], m[GeV],q[unit]]
    return state_mks if(mks) else state_nat


def propagate_state_in_vacuum_to_z(state, z):
    if(z==state[2]): return state
    x0 = state[0]
    y0 = state[1]
    z0 = state[2]
    px = state[3]
    py = state[4]
    pz = state[5]
    m  = state[6]
    q  = state[7]
    pxz = np.sqrt(px**2 + pz**2)
    pyz = np.sqrt(py**2 + pz**2)
    thetax = np.arcsin(px/pxz)
    thetay = np.arcsin(py/pyz)
    x = x0 + np.tan(thetax)*(z-z0)
    y = y0 + np.tan(thetay)*(z-z0)
    state_at_z = [x,y,z, px,py,pz, m,q]
    return state_at_z


# def truncated_exp_NK(a,b,how_many=1):
#     a = -np.log(a)
#     b = -np.log(b)
#     rands = np.exp(-(np.random.rand(how_many)*(b-a) + a))
#     return rands[0] if(how_many==1) else rands

def truncated_exp_NK(aa, bb, slope=1.0, how_many=1):
    '''
    Sample from a power-law-like distribution between [aa, bb], with controllable slope.
    slope=1 -> exp distribution
    slope<1 -> shallower (power-law)
    slope>1 -> steeper (power-law)
    '''
    aa, bb = float(aa), float(bb)
    if slope == 1:
        r = np.random.rand(how_many)
        samples = np.exp(-(r * (-np.log(bb) + np.log(aa)) - np.log(aa)))
    else:
        r = np.random.rand(how_many)
        samples = ((bb**(1-slope) - aa**(1-slope)) * r + aa**(1-slope))**(1/(1-slope))
    return samples[0] if how_many == 1 else samples


def simulate_secondary_production(primary_state,q=+1,Emin=0.5,Emax=5,smear_T=False,smear_pT=False,truncate_pT=False):    
    x      = primary_state[0]
    y      = primary_state[1]
    z      = primary_state[2]
    px     = primary_state[3]
    py     = primary_state[4]
    pz     = primary_state[5]
    mass   = primary_state[6]
    ### smear trasverse position
    if(smear_T and smear_sigma_T_um>0):
        x = x + np.random.normal(0,smear_sigma_T_um*um_to_m)
        y = y + np.random.normal(0,smear_sigma_T_um*um_to_m)
    ### smear trasverse momenta
    if(smear_pT):
        if(truncate_pT):
            dpx = np.random.normal(0,smear_sigma_P_GeV)
            while(abs(dpx)>smear_sigma_P_GeV):
                dpx = np.random.normal(0,smear_sigma_P_GeV)
            px = px + dpx
            dpy = np.random.normal(0,smear_sigma_P_GeV)
            while(abs(dpy)>smear_sigma_P_GeV):
                dpy = np.random.normal(0,smear_sigma_P_GeV)
            py = py + dpy
        else:
            px = px + np.random.normal(0,smear_sigma_P_GeV)
            py = py + np.random.normal(0,smear_sigma_P_GeV)
    ### sample energy from exponential
    # E = truncated_exp_NK(Emin,Emax,slope=0.3) if(Emax>Emin) else Emin # GeV
    E = br.sample_from_pdf_on_bins(E_vals, eplus, nsamples=1, rng=rng)
    while(E[0]<Emin or E[0]>Emax): E = br.sample_from_pdf_on_bins(E_vals, eplus, nsamples=1, rng=rng)
    E = E[0]
    
    ### assume the x-y momemnta staty the same and correct the z momentum
    pz = np.sqrt( E**2 - mass**2 - px**2 - py**2 ) # GeV
    secondary_state = [x,y,z, px,py,pz, mass, q]
    return secondary_state


def state_GeV_to_kgms(state):
    state_mks = [0]*len(state)
    state_mks[0] = state[0]
    state_mks[1] = state[1]
    state_mks[2] = state[2]
    state_mks[3] = state[3]*GeV_to_kgms # kg*m/s
    state_mks[4] = state[4]*GeV_to_kgms # kg*m/s
    state_mks[5] = state[5]*GeV_to_kgms # kg*m/s
    state_mks[6] = state[6]*GeV_to_kgm2s2/c2 # kg
    state_mks[7] = state[7]
    return state_mks

########################################################################
########################################################################

# Create the detector objects
detectors = []
for i in range(5):
    zpos = detector_z_base_cm + i ### detectors are spaced by 1 cm
    detector = Detector(
        name=f"ALPIDE_{i}",
        x_min=detector_x_center_cm-chipYcm/2., x_max=detector_x_center_cm+chipYcm/2.,
        y_min=detector_y_center_cm-chipXcm/2., y_max=detector_y_center_cm+chipXcm/2.,
        z_min=zpos, z_max=zpos, ### 0 width...
        angles=detangl
    )
    detectors.append(detector)

# Create magnetic elements
quad0 = Quadrupole(
    name="quad0",
    x_min=-2.4610+magsetdelt["quad0"][0], x_max=2.4610+magsetdelt["quad0"][0], # cm
    y_min=-2.4610+magsetdelt["quad0"][1], y_max=2.4610+magsetdelt["quad0"][1], # cm
    z_min=367.33336, z_max=464.6664, # cm
    gradient=quadsvals[MagnetsSettings][0], # kG/m
    angles=magsetangl["quad0"]
)
quad1 = Quadrupole(
    name="quad1",
    x_min=-2.4610+magsetdelt["quad1"][0], x_max=2.4610+magsetdelt["quad1"][0], # cm
    y_min=-2.4610+magsetdelt["quad1"][1], y_max=2.4610+magsetdelt["quad1"][1], # cm
    z_min=590.3336, z_max=687.6664, # cm
    # gradient=+28.55 if(MagnetsSettings==502) else +46.42 # kG/m
    gradient=quadsvals[MagnetsSettings][1], # kG/m
    angles=magsetangl["quad1"]
)
quad2 = Quadrupole(
    name="quad2",
    x_min=-2.4610+magsetdelt["quad2"][0], x_max=2.4610+magsetdelt["quad2"][0], # cm
    y_min=-2.4610+magsetdelt["quad2"][1], y_max=2.4610+magsetdelt["quad2"][1], # cm
    z_min=812.3336, z_max=909.6664, # cm
    gradient=quadsvals[MagnetsSettings][2], # kG/m
    angles=magsetangl["quad2"]
)
xcorr = Dipole(
    name="xcorr",
    x_min=-10.795+magsetdelt["xcorr"][0], x_max=+10.795+magsetdelt["xcorr"][0], # cm 
    y_min=-4.6990+magsetdelt["xcorr"][1], y_max=+4.6990+magsetdelt["xcorr"][1], # cm
    z_min=987.779, z_max=1011.15, # cm
    # B_x=0, B_y=+0.026107, B_z=0,  # Tesla
    B_x=dipolesvals[MagnetsSettings]["xcorr"][0], B_y=dipolesvals[MagnetsSettings]["xcorr"][1], B_z=dipolesvals[MagnetsSettings]["xcorr"][2], # Tesla
    angles=magsetangl["xcorr"]
)
dipole = Dipole(
    name="dipole",
    x_min=-2.2352+magsetdelt["dipole"][0], x_max=2.2352+magsetdelt["dipole"][0], # cm
    y_min=-6.6927+magsetdelt["dipole"][1], y_max=3.4927+magsetdelt["dipole"][1], # cm
    z_min=1260.34, z_max=1351.78, # cm
    # B_x=0.219, B_y=0, B_z=0,  # Tesla
    B_x=dipolesvals[MagnetsSettings]["dipole"][0], B_y=dipolesvals[MagnetsSettings]["dipole"][1], B_z=dipolesvals[MagnetsSettings]["dipole"][2],  # Tesla
    angles=magsetangl["dipole"]
)
flange = Element(
    name="flange",
    x_min=-2.2352, x_max=2.2352,
    y_min=-6.3752, y_max=3.1752,
    z_min=1377.784, z_max=1377.785
)

# Create beampipe from z=0 to entrance of dipole:
beampipe_inner_radius_cm = 2.   # 2 cm inner radius 
beampipe_outer_radius_cm = 2.2  # 2.2 cm outer radius
beampipe_z_start_cm = 0.0       # Start at IP
beampipe_z_end_cm = dipole.z_min * m_to_cm - 0.0  # End 0 cm before dipole entrance
beampipe = Beampipe(
    name="beampipe",
    inner_radius_cm=beampipe_inner_radius_cm,
    outer_radius_cm=beampipe_outer_radius_cm, 
    z_min_cm=beampipe_z_start_cm,
    z_max_cm=beampipe_z_end_cm,
)


### collect all elements
magnets  = [quad0, quad1, quad2, xcorr, dipole]
elements = magnets + [flange] + [beampipe] + detectors
# for element in elements:
#     if(element.name=="beampipe"): continue
#     print(f"{element.name}: angles={element.angles}")



########################################################################
########################################################################
########################################################################
### linear mapping from real space to pixel space
def real_to_pixel_coords(x_real, y_real, xmin, xmax, ymin, ymax):
    xpixelmin = 0
    xpixelmax = npix_y-1
    nxpixels  = npix_y
    ypixelmin = 0
    ypixelmax = npix_x-1
    nypixels  = npix_x
    x_real = np.asarray(x_real)
    y_real = np.asarray(y_real)
    x_pixel = xpixelmin + (x_real - xmin) * (xpixelmax - xpixelmin) / (xmax - xmin)
    y_pixel = ypixelmin + (y_real - ymin) * (ypixelmax - ypixelmin) / (ymax - ymin)
    return x_pixel, y_pixel

### convert to integer indices and clip to valid range
def real_to_pixel_indices(x_real, y_real, xmin, xmax, ymin, ymax):
    xpixelmin = 0
    xpixelmax = npix_y-1
    nxpixels  = npix_y
    ypixelmin = 0
    ypixelmax = npix_x-1
    nypixels  = npix_x
    x_pixel, y_pixel = real_to_pixel_coords(x_real, y_real, xmin, xmax, ymin, ymax)
    x_indices = np.clip(np.round(x_pixel).astype(int), xpixelmin, xpixelmax-1)
    y_indices = np.clip(np.round(y_pixel).astype(int), ypixelmin, ypixelmax-1)
    return x_indices, y_indices

### transform to EUDAQ space
def transform_pixel_indices(x_indices, y_indices, rotate_90_cw=True, mirror_x=True, flip_y=True):
    xpixelmin = 0
    xpixelmax = npix_y-1
    nxpixels  = npix_y
    ypixelmin = 0
    ypixelmax = npix_x-1
    nypixels  = npix_x
    x_trans   = x_indices.copy()
    y_trans   = y_indices.copy()
    # step 1: Rotate 90 degrees clockwise (x,y) -> (y, -x)
    # After rotation: new_x = y, new_y = max_x - x
    if(rotate_90_cw):
        temp_x = y_trans.copy()
        temp_y = (xpixelmax - 1) - x_trans
        x_trans = temp_x
        y_trans = temp_y
        # Update pixel boundaries after rotation (dimensions swap)
        xpixelmin, xpixelmax, nxpixels, ypixelmin, ypixelmax, nypixels = \
            ypixelmin, ypixelmax, nypixels, xpixelmin, xpixelmax, nxpixels
    # # step 2: mirror x-axis around its center
    # if(mirror_x):
    #     x_center = (xpixelmin + xpixelmax - 1) / 2
    #     x_trans = 2 * x_center - x_trans
    #     x_trans = np.clip(x_trans, xpixelmin, xpixelmax - 1).astype(int)
    # # step 3: flip y-axis around its center
    # if(flip_y):
    #     y_center = (ypixelmin + ypixelmax) / 2
    #     y_trans = 2 * y_center - y_trans
    #     y_trans = np.clip(y_trans, ypixelmin, ypixelmax).astype(int)
    return x_trans, y_trans

def hist2d_to_1d_root_style(hist2d_result,transform=True):
    counts, xedges, yedges, image = hist2d_result
    nbinsx, nbinsy = counts.shape
    bin_numbers = []
    occupancy_counts = []
    for i in range(nbinsx):        # X bins (rows)
        for j in range(nbinsy):    # Y bins (columns) - inner loop
            # ROOT-style global bin number: ny * binx + biny
            # ROOT bins are 1-indexed, but we'll use 0-indexed for simplicity
            global_bin = nbinsy * i + j
            occupancy = counts[i, j]
            bin_numbers.append(global_bin)
            occupancy_counts.append(occupancy)
    return np.array(bin_numbers), np.array(occupancy_counts)


########################################################################
########################################################################
########################################################################


# To calculate total magnetic field at a point
def total_field(position):
    x, y, z = position
    field = np.zeros(3)
    # for element in elements: field += element.field(x, y, z)
    for magnet in magnets: field += magnet.field(x, y, z)
    return field


def record_hits(element,z_element,trajectory,particle_id):
    for i in range(len(trajectory.t) - 1):
        z1, z2 = trajectory.y[2][i], trajectory.y[2][i+1]
        
        # If particle trajectory crosses the detector plane
        if (z1 <= z_element <= z2) or (z2 <= z_element <= z1):
            # Linear interpolation to find position at detector plane
            t1, t2 = trajectory.t[i], trajectory.t[i+1]
            fraction = (z_element - z1) / (z2 - z1) if z2 != z1 else 0
            
            t_hit = t1 + fraction * (t2 - t1)
            x_hit = trajectory.y[0][i] + fraction * (trajectory.y[0][i+1] - trajectory.y[0][i])
            y_hit = trajectory.y[1][i] + fraction * (trajectory.y[1][i+1] - trajectory.y[1][i])
            
            px_hit = trajectory.y[3][i] + fraction * (trajectory.y[3][i+1] - trajectory.y[3][i])
            py_hit = trajectory.y[4][i] + fraction * (trajectory.y[4][i+1] - trajectory.y[4][i])
            pz_hit = trajectory.y[5][i] + fraction * (trajectory.y[5][i+1] - trajectory.y[5][i])
            
            # Check if hit is within detector surface
            if (element.x_min <= x_hit <= element.x_max and element.y_min <= y_hit <= element.y_max):
                element.record_hit(particle_id, x_hit, y_hit, z_element, px_hit, py_hit, pz_hit)



# Define the equations of motion for a charged particle in a magnetic field
def particle_motion(t, state):
    ### state vector: [x[m],y[m],z[m], px[kg*m/s],py[kg*m/s],pz[kg*m/s], m[kg],q[unit]]
    x,y,z, px,py,pz, m,q = state
    q = int(q)
    dm_dt = 0
    dq_dt = 0
    
    position = np.array([x, y, z])
    momentum = np.array([px, py, pz])
    
    ### relativistic velocity
    p_mag = np.linalg.norm(momentum)
    gamma = np.sqrt(1 + (p_mag/(m*c))**2)  ### relativistic factor
    velocity = momentum / (gamma * m)
    
    # Rate of change of position is velocity
    dx_dt = velocity[0]
    dy_dt = velocity[1]
    dz_dt = velocity[2]
    
    ### magnetic field at current position
    B = total_field(position)
    if(np.linalg.norm(B)<1e-6): return [dx_dt,dy_dt,dz_dt, 0,0,0, dm_dt,dq_dt]
    
    ### lorentz force: F = q(v × B)
    force = (q * e) * np.cross(velocity, B)    
    ### rate of change of momentum is force
    dpx_dt = force[0]
    dpy_dt = force[1]
    dpz_dt = force[2]
    
    # print(f"t={t:.2e},  r={x:.2e},{y:.2e},{z:.2e},  p={px:.2e},{py:.2e},{pz:.2e},  p={p_mag:.2e},  gam={gamma:.2e},  v={velocity[0]:.2e},{velocity[1]:.2e},{velocity[2]:.2e},  m={m:.2e}, q={q:}, B={B[0]:.2e},{B[1]:.2e},{B[2]:.2e},  dr/dt={dx_dt:.2e},{dy_dt:.2e},{dz_dt:.2e},  dp_dt={dpx_dt:.2e},{dpy_dt:.2e},{dpz_dt:.2e}")
    
    return [dx_dt,dy_dt,dz_dt, dpx_dt,dpy_dt,dpz_dt, dm_dt,dq_dt]




# Add this function to replace your existing collision_event function
def collision_event(t, state, elements_to_check=None, beampipe=None):
    '''
    Event function to detect when a particle hits walls of magnet elements or beampipe.
    Returns zero when a collision occurs.
    
    Parameters:
    -----------
    t : float
        Current time value
    state : array 
        Current state [x,y,z, px,py,pz, m,q]
    elements_to_check : list, optional
        List of elements to check for collisions
    beampipe : Beampipe, optional
        Beampipe object to check for collisions
        
    Returns:
    --------
    float
        Distance from nearest wall (negative inside boundaries, zero at boundary, positive outside)
    '''
    if elements_to_check is None:
        elements_to_check = [quad0, quad1, quad2, xcorr, dipole] 
    
    x, y, z = state[0], state[1], state[2]
    
    # Initialize distance to a large value
    min_distance = float('inf')
    
    # Check beampipe collision first (most restrictive)
    if beampipe is not None and beampipe.z_min_pipe <= z <= beampipe.z_max_pipe:
        r = np.sqrt(x**2 + y**2)
        # Distance to inner wall of beampipe (negative if outside vacuum)
        distance_to_vacuum_boundary = beampipe.inner_radius - r
        min_distance = min(min_distance, distance_to_vacuum_boundary)
    
    # Check magnet element collisions
    for element in elements_to_check:
        ### skip if not in z-range of element (with small margin)
        if(not (element.z_min - 0.0001 <= z <= element.z_max + 0.0001)): continue
            
        # Calculate distances to each wall
        dx_min = x - element.x_min
        dx_max = element.x_max - x
        dy_min = y - element.y_min
        dy_max = element.y_max - y
        
        # Find minimum distance to wall
        distances = [dx_min, dx_max, dy_min, dy_max]
        element_min_distance = min(distances)
        
        # Update minimum distance if this element's wall is closer
        min_distance = min(min_distance, element_min_distance)
    
    # If not near any element, return a large positive value
    if min_distance == float('inf'):
        return 1.0
        
    return min_distance

# Set terminal attribute
collision_event.terminal = True





def propagate_particle_with_collision(particle_id, initial_state, t_span, beampipe=None, max_step=1e-10):
    '''
    Propagate particle through beamline with collision detection including beampipe.

    Parameters:
    -----------
    particle_id : int, Unique identifier for the particle
    initial_state : array, Initial state [x0,y0,z0, px0,py0,pz0, m,q]
    t_span : tuple, (t_start, t_end) for integration
    beampipe : Beampipe, optional, Beampipe object for collision detection
    max_step : float, optional, Maximum step size for integrator

    Returns:
    --------
    solution : OdeSolution
        Solution object from solve_ivp
    collision_info : dict or None
        Information about collision if it occurred, None otherwise
    '''
    # Use solve_ivp with event detection including beampipe
    solution = solve_ivp(
        particle_motion,
        t_span,
        initial_state,
        method='RK45',
        events=lambda t, state: collision_event(t, state, beampipe=beampipe),
        max_step=max_step,
        rtol=1e-8,
        atol=1e-10
    )

    # Check for detector crossings
    for det in detectors: record_hits(det, det.z_pos, solution, particle_id)
    # Record hits on other elements
    # record_hits(flange, flange.z_min, solution, particle_id)
    record_hits(flange, flange.z_max, solution, particle_id)
    # record_hits(dipole, dipole.z_min, solution, particle_id)
    record_hits(dipole, dipole.z_max, solution, particle_id)
    # record_hits(xcorr,  xcorr.z_min,  solution, particle_id)
    record_hits(xcorr,  xcorr.z_max,  solution, particle_id)
    # record_hits(quad0,  quad0.z_min,  solution, particle_id)
    record_hits(quad0,  quad0.z_max,  solution, particle_id)
    # record_hits(quad1,  quad1.z_min,  solution, particle_id)
    record_hits(quad1,  quad1.z_max,  solution, particle_id)
    # record_hits(quad2,  quad2.z_min,  solution, particle_id)
    record_hits(quad2,  quad2.z_max,  solution, particle_id)

    # Record beampipe hits if it exists
    if beampipe is not None: record_hits(beampipe, beampipe.z_max_pipe, solution, particle_id)

    # Check if collision occurred
    collision_info = None
    if solution.t_events[0].size > 0:
        # Collision occurred
        collision_time = solution.t_events[0][0]
        collision_state = np.array([
            np.interp(collision_time, solution.t, solution.y[0]),
            np.interp(collision_time, solution.t, solution.y[1]),
            np.interp(collision_time, solution.t, solution.y[2]),
            np.interp(collision_time, solution.t, solution.y[3]),
            np.interp(collision_time, solution.t, solution.y[4]),
            np.interp(collision_time, solution.t, solution.y[5])
        ])

        # Determine which element was hit
        collision_element = None
        element_name = "unknown"

        # Check if beampipe was hit first
        if (beampipe is not None and
            beampipe.z_min_pipe <= collision_state[2] <= beampipe.z_max_pipe):
            r = np.sqrt(collision_state[0]**2 + collision_state[1]**2)
            if abs(r - beampipe.inner_radius) < 1e-6:
                collision_element = beampipe
                element_name = "Beampipe"

        # If not beampipe, check other elements
        if collision_element is None:
            for element in [quad0, quad1, quad2, xcorr, dipole]:
                if (element.z_min <= collision_state[2] <= element.z_max):
                    if (abs(collision_state[0] - element.x_min) < 1e-6 or
                        abs(collision_state[0] - element.x_max) < 1e-6 or
                        abs(collision_state[1] - element.y_min) < 1e-6 or
                        abs(collision_state[1] - element.y_max) < 1e-6):
                        collision_element = element
                        break

            if   collision_element == quad0:  element_name = "Quadrupole 0"
            elif collision_element == quad1:  element_name = "Quadrupole 1"
            elif collision_element == quad2:  element_name = "Quadrupole 2"
            elif collision_element == xcorr:  element_name = "Xcorr"
            elif collision_element == dipole: element_name = "Dipole"
            elif collision_element == flange: element_name = "Flange"

        collision_info = {
            "time": collision_time,
            "position": collision_state[:3],
            "momentum": collision_state[3:],
            "element": element_name
        }

    return solution, collision_info


def get_state_at_z(solution, collision, z_target, check_collision=False):
    ### check if there is NO collision first:
    if(check_collision and collision is not None): return []
    ### get the trajectory coordinates
    x  = solution.y[0]
    y  = solution.y[1]
    z  = solution.y[2]
    px = solution.y[3]
    py = solution.y[4]
    pz = solution.y[5]
    m  = solution.y[6]
    q  = solution.y[7]
    ### Loop over each segment and look for a crossing
    for i in range(len(z)-1):
        z1, z2 = z[i], z[i+1]
        if(z1<=z_target<=z2) or (z2<=z_target<=z1): ### TODO: should be only if z1<=z_target<=z2, no ?
            ### avoid division by zero
            if(z2==z1): frac = 0.0
            else:       frac = (z_target-z1)/(z2 - z1)
            x_interp = x[i]+ frac * (x[i+1]-x[i])
            y_interp = y[i]+ frac * (y[i+1]-y[i])
            px_interp = px[i]+ frac * (px[i+1]-px[i])
            py_interp = py[i]+ frac * (py[i+1]-py[i])
            pz_interp = pz[i]+ frac * (pz[i+1]-pz[i])
            state = [x_interp,y_interp,z_target, px_interp,py_interp,pz_interp, m,q]
            return state


### TODO this function is needed for multiprocess runs ONLY
def refill_detector_hits(initial_states, trajectories):
    if(len(initial_states)!=len(trajectories)):
        raise ValueError(f"number of initial_states {len(initial_states)} is not the same as number of trajectories {len(trajectories)}")
    ### remove old hits
    for element in elements: element.hits = []
    ### refill the hits
    for pid,trajectory in enumerate(trajectories):
        for element in elements:
            if(element in detectors):
                record_hits(element, element.z_pos ,trajectory,pid)
            elif(element in magnets): 
                record_hits(element,element.z_min,trajectory,pid)
                record_hits(element,element.z_max,trajectory,pid)
            else:
                record_hits(element,element.z_max,trajectory,pid)
    
        

def plot_system(particle_trajectories, particle_collisions, initial_states, pdfname, nmaxtrks=100, ax=None):
    
    isAxNone = (ax is None)
    
    # fig = plt.figure(figsize=(16, 10))
    # ax = fig.add_subplot(111, projection='3d')
    if ax is None:
        fig = plt.figure(figsize=(16, 10))
        ax = fig.add_subplot(111, projection='3d')

    # Plot magnetic elements
    quad0.plot_element(ax,  'cyan')
    quad1.plot_element(ax,  'purple')
    quad2.plot_element(ax,  'cyan')
    xcorr.plot_element(ax,  'orange')
    dipole.plot_element(ax, 'red')
    flange.plot_element(ax, 'black')
    beampipe.plot_element(ax, 'gray', alpha=0.3)

    # Plot detector planes
    detector_colors = ['green', 'green', 'green', 'green', 'green']
    for i, detector in enumerate(detectors):
        detector.plot_element(ax, detector_colors[i % len(detector_colors)])

    # Plot particle trajectories with collision handling
    collision_count = 0
    for i, (trajectory, collision) in enumerate(zip(particle_trajectories, particle_collisions)):
        if i >= nmaxtrks:
            break
            
        x, y, z = trajectory.y[0], trajectory.y[1], trajectory.y[2]
        
        if collision is not None:
            # Find the index where collision occurred
            collision_z = collision['position'][2]
            collision_idx = None
            
            # Find the closest point to collision in trajectory
            for j, z_val in enumerate(z):
                if z_val >= collision_z:
                    collision_idx = j
                    break
            
            if collision_idx is not None:
                # Plot only up to collision point
                ax.plot(x[:collision_idx+1], y[:collision_idx+1], z[:collision_idx+1], 'r-', linewidth=1.5, alpha=0.8)  # Red for collided particles
                # Mark collision point?
                # ax.scatter(collision['position'][0], collision['position'][1], collision['position'][2], c='red', s=20, marker='x')
                collision_count += 1
            else:
                # Fallback: plot entire trajectory in red
                ax.plot(x, y, z, 'r-', linewidth=1.0, alpha=0.6)
                collision_count += 1
        else:
            # No collision: plot entire trajectory in blue
            ax.plot(x, y, z, 'b-', linewidth=1.0, alpha=0.8)

    if(isAxNone): print(f"Plotted {collision_count} collided trajectories out of {min(nmaxtrks, len(particle_trajectories))}")

    # Set axis labels and limits
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(0, 18)

    # Add legend for elements
    beampipe_patch  = mpatches.Patch(color='gray',   alpha=0.3, label='Beampipe')
    quad0_patch     = mpatches.Patch(color='cyan',   alpha=0.5, label='Quad0')
    quad1_patch     = mpatches.Patch(color='purple', alpha=0.5, label='Quad1')
    quad2_patch     = mpatches.Patch(color='cyan',   alpha=0.5, label='Quad2')
    xcorr_patch     = mpatches.Patch(color='orange', alpha=0.5, label='XCorr')
    dipole_patch    = mpatches.Patch(color='red',    alpha=0.5, label='Dipole')
    flange_patch    = mpatches.Patch(color='black',  alpha=0.5, label='Flange')
    collision_patch = mpatches.Patch(color='red',    alpha=0.8, label='Collided particles')
    free_patch      = mpatches.Patch(color='blue',   alpha=0.8, label='Free particles')
    detector_patch  = mpatches.Patch(color='green',  alpha=0.5, label='Detector')

    handles, labels = ax.get_legend_handles_labels()
    handles.extend([beampipe_patch, quad0_patch, quad1_patch, quad2_patch, xcorr_patch, dipole_patch, flange_patch, collision_patch, free_patch, detector_patch])
    ax.legend(handles=handles, loc='center left')
    ax.set_title(f'Particle Propagation (MagSet={MagnetsSettings})')

    if(isAxNone):
        plt.tight_layout()
        plt.savefig(f"{pdfname}_tracks.pdf")
        if(doshw): plt.show()
        return fig


def plot_gif(pdfname):
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')

    def update(angle):
        ax.cla()  # Clear the 3D axis for redrawing
        plot_system(
            particle_trajectories,
            particle_collisions,
            initial_states,
            pdfname=None,
            nmaxtrks=100,
            ax=ax
        )
        ax.view_init(elev=30, azim=angle)

    print(f"Starting tracks rotating animation")
    ani = FuncAnimation(
        fig,
        update,
        frames=range(0, 360, 5), ### Rotate full circle in 5 degree steps
        interval=100  ### milliseconds per frame
    )
    ### save as animated GIF using Pillow
    ani.save(f"{pdfname}_tracks_rotation.gif", writer=PillowWriter(fps=10))
    print(f"Rotating animation saved in {pdfname}_tracks_rotation.gif")


def plot_scatter(pdfname):
    fig, axs = plt.subplots(1, 5, figsize=(10, 4), sharex=True, sharey=True, tight_layout=True)
    for i, detector in enumerate(detectors):
        rect = plt.Rectangle(
            (detector.x_min, detector.y_min),
            detector.x_max - detector.x_min,
            detector.y_max - detector.y_min,
            fill=False, edgecolor='gray'
        )
        axs[i].add_patch(rect)

        # Plot hits
        hit_x = [hit['x'] for hit in detector.hits]
        hit_y = [hit['y'] for hit in detector.hits]
        particle_ids = [hit['particle_id'] for hit in detector.hits]

        for j, (x, y, pid) in enumerate(zip(hit_x, hit_y, particle_ids)): axs[i].plot(x, y, 'o', markersize=1)

        axs[i].set_xlim(detector.x_min * 1.2, detector.x_max * 1.2)
        axs[i].set_ylim(detector.y_min * 0.9, detector.y_max * 1.1)
        axs[i].set_xlabel('X [m]')
        axs[i].set_ylabel('Y [m]')
        axs[i].set_title(f'ALPIDE_{i}')
        axs[i].grid(True)

    plt.tight_layout()
    plt.savefig(f"{pdfname}_scatter.pdf")
    if(doshw): plt.show()

    return fig


#################################################################################
#################################################################################
#################################################################################


def collect_errors(error):
    ### https://superfastpython.com/multiprocessing-pool-error-callback-functions-in-python/
    print(f'Error: {error}', flush=True)




#####################
### main function ###
#####################
if __name__ == "__main__":
    
    print(f"Run multiprocessing: {mltprc}")
    print(f"MagnetsSettings: {MagnetsSettings}")

    pdfname = f"plots/generator_plots/generator_{MagnetsSettings}"
    
    #####################################################
    #####################################################
    #####################################################
    # initial conditions for particles, format: [x0,y0,z0, px0,py0,pz0, m,q]
    # momentum in units of GeV/c and convert to kg*m/s
    ### generate
    initial_states = []
    PZ0 = []
    for i in range(Nparticles):
        ### generate the primary beam
        electron_state = GenerateGaussianBeam(E_GeV,mGeV,-QQ)
        ### propagate the primary to the foil in vacuum
        electron_state_at_foil = propagate_state_in_vacuum_to_z(electron_state,Z0_m)
        ### smear at the foil to get the secondary
        positron_state_at_foil = simulate_secondary_production(electron_state_at_foil,q=QQ,Emin=Emin,Emax=Emax,smear_T=smearT,smear_pT=smearP,truncate_pT=truncate_smear_pT)
        ### convert to mks
        positron_state_at_foil_mks = state_GeV_to_kgms(positron_state_at_foil)
        # positron_state_at_foil_mks = state_GeV_to_kgms(electron_state_at_foil)
        ### fill the initial states list
        initial_states.append(positron_state_at_foil_mks)
        ### fill the PZ0 list for book-keeping
        PZ0.append(positron_state_at_foil[5])
    #####################################################
    #####################################################
    #####################################################
    
    # Propagate particles and check for collisions (including beampipe)
    particle_trajectories = []
    particle_collisions   = []


    #########################
    ### Run the propagation
    #########################
    if(not mltprc):
        ##########################
        ### no multiprocessing ###
        ##########################
        for i,state in enumerate(initial_states):
            solution, collision = propagate_particle_with_collision(i, state, t_span, beampipe=beampipe, max_step=max_dt)
            particle_trajectories.append(solution)
            particle_collisions.append(collision)
            if(i%100==0): print(f"done propagating particle {i}")
    else:
        ############################
        ### with multiprocessing ###
        ############################
        num_cores = 10 # Number of cores to activate
        pool = multiprocessing.Pool(processes=num_cores) # Create the pool

        # This list will hold the AsyncResult objects. Each AsyncResult corresponds to one submitted task
        async_results = []
        ### loop on the initial states
        for i,state in enumerate(initial_states):
            result = pool.apply_async(propagate_particle_with_collision, args=(i, state, t_span, beampipe, max_dt), error_callback=collect_errors)
            async_results.append(result)
            if(i%100==0): print(f"done propagating particle {i}")
        # After submitting all tasks:
        pool.close() # No more tasks can be added to the pool
        
        # Collect results from each submitted task. The loop here is over the 'async_results' list, which has N elements
        for i,async_res in enumerate(async_results):
            # .get() blocks until the task associated with this async_res is complete. It retrieves the tuple (obj1_result, obj2_result) returned by my_function_per_task.
            solution, collision = async_res.get()
            # APPEND the individual results for THIS task to the main lists. This doesn't overwrite; it adds a new element to the end of the list.
            particle_trajectories.append(solution)
            particle_collisions.append(collision)
        # Wait for all worker processes to finish and terminate
        pool.join()
        
        ### re-record the hits
        print(f"starting refill:")
        st = time.time()
        refill_detector_hits(initial_states, particle_trajectories)
        et = time.time()
        elapsed_time = et - st
        print('end time:', elapsed_time, 'seconds')
    ##########################
    
    
    
    ############################
    ### Now do some plotting ###
    ############################
    
    
    
    ### Plot the system with particle trajectories
    main_fig = plot_system(particle_trajectories, particle_collisions, initial_states, pdfname, nmaxtrks=100)
    
    ### make the gif:
    if(dogif): plot_gif(pdfname)

    ### Plot the hits as scatter plot (using detector.hits)
    scat_fig = plot_scatter(pdfname)



    ### check acceptance
    npivots = 0
    initial_state_nhits = np.zeros(len(initial_states))
    list_good_tracks    = []
    for pivot in detectors[0].hits:
        pivot_pid = pivot['particle_id']
        npivots +=1
        nhits   = 1
        for i,detector in enumerate(detectors):
            if(i==0): continue
            for hit in detector.hits:
                if(hit['particle_id']==pivot_pid):
                    nhits += 1
                    break
        
        ### acceptance
        if(nhits!=len(detectors)): continue
        
        ### main selection cut...
        pass_dipole_spot_cut = True
        for point in dipole.hits:
            if(point['particle_id'] != pivot_pid): continue
            if(abs(point['z']-dipole.z_max)>1e-6): continue
            X = point['x']*m_to_mm
            Y = point['y']*m_to_mm
            if(not spot_cut(X,Y,-2.6,10.6,3,7)):
                pass_dipole_spot_cut = False
                break
        if(not pass_dipole_spot_cut): continue
        
        ### good track
        list_good_tracks.append( pivot_pid )
        initial_state_nhits[pivot_pid] = nhits
    print(f"Got {len(list_good_tracks)} tracks with {len(detectors)} hits out of {npivots} pivot points at ALPIDE_0")       


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
        scale = 0.9*h2.GetYaxis().GetXmax()/p.GetMaximum() if(p.GetMaximum()!=0) else 1.
        p.Scale(scale)
        return p
    
    
    
    
    # ### occupancy fine
    # TH1_F = {}
    # TH2_F = {}
    # PJF = {}
    # for i,detector in enumerate(detectors):
    #     det = f"ALPIDE_{i}"
    #     TH1_F.update({det:hPixelMatrix1.Clone(f"{hPixelMatrix1.GetName()} {det}")})
    #     TH1_F[det].SetTitle(det)
    #     TH2_F.update({det:hPixelMatrix2.Clone(f"{hPixelMatrix2.GetName()} {det}")})
    #     TH2_F[det].SetTitle(det)
    #     for hit in detector.hits:
    #         XX  = np.array(hit['x']*m_to_mm)
    #         YY  = np.array(hit['y']*m_to_mm)
    #         XP,YP = real_to_pixel_indices(XX,YY, detectors[0].x_min*m_to_mm,detectors[0].x_max*m_to_mm, detectors[0].y_min*m_to_mm,detectors[0].y_max*m_to_mm)
    #         XR,YR = transform_pixel_indices(XP,YP)
    #         TH2_F[det].Fill(XR,YR)
    #         bglob = TH2_F[det].FindBin(XR,YR)
    #         TH1_F[det].AddBinContent(bglob,1)
    # cnv1 = ROOT.TCanvas("cnv11","",1000,1500)
    # cnv1.Divide(2,5)
    # cnv1.cd(1)
    # ROOT.gPad.SetTicks(1,1)
    # TH2_F["ALPIDE_0"].Draw("col")
    # PJF.update({"ALPIDE_0": proj(TH2_F["ALPIDE_0"],projmax=True,name="fine")})
    # PJF["ALPIDE_0"].Draw("hist same")
    # ROOT.gPad.RedrawAxis()
    # cnv1.cd(2)
    # ROOT.gPad.SetTicks(1,1)
    # TH1_F["ALPIDE_0"].Draw("hist")
    # ROOT.gPad.RedrawAxis()
    # cnv1.cd(3)
    # ROOT.gPad.SetTicks(1,1)
    # TH2_F["ALPIDE_1"].Draw("col")
    # PJF.update({"ALPIDE_1": proj(TH2_F["ALPIDE_1"],projmax=True,name="fine")})
    # PJF["ALPIDE_1"].Draw("hist same")
    # ROOT.gPad.RedrawAxis()
    # cnv1.cd(4)
    # ROOT.gPad.SetTicks(1,1)
    # TH1_F["ALPIDE_1"].Draw("hist")
    # ROOT.gPad.RedrawAxis()
    # cnv1.cd(5)
    # ROOT.gPad.SetTicks(1,1)
    # TH2_F["ALPIDE_2"].Draw("col")
    # PJF.update({"ALPIDE_2": proj(TH2_F["ALPIDE_2"],projmax=True,name="fine")})
    # PJF["ALPIDE_2"].Draw("hist same")
    # ROOT.gPad.RedrawAxis()
    # cnv1.cd(6)
    # ROOT.gPad.SetTicks(1,1)
    # TH1_F["ALPIDE_2"].Draw("hist")
    # ROOT.gPad.RedrawAxis()
    # cnv1.cd(7)
    # ROOT.gPad.SetTicks(1,1)
    # TH2_F["ALPIDE_3"].Draw("col")
    # PJF.update({"ALPIDE_3": proj(TH2_F["ALPIDE_3"],projmax=True,name="fine")})
    # PJF["ALPIDE_3"].Draw("hist same")
    # ROOT.gPad.RedrawAxis()
    # cnv1.cd(8)
    # ROOT.gPad.SetTicks(1,1)
    # TH1_F["ALPIDE_3"].Draw("hist")
    # ROOT.gPad.RedrawAxis()
    # cnv1.cd(9)
    # ROOT.gPad.SetTicks(1,1)
    # TH2_F["ALPIDE_4"].Draw("col")
    # PJF.update({"ALPIDE_4": proj(TH2_F["ALPIDE_4"],projmax=True,name="fine")})
    # PJF["ALPIDE_4"].Draw("hist same")
    # ROOT.gPad.RedrawAxis()
    # cnv1.cd(10)
    # ROOT.gPad.SetTicks(1,1)
    # TH1_F["ALPIDE_4"].Draw("hist")
    # ROOT.gPad.RedrawAxis()
    # cnv1.Update()
    # cnv1.SaveAs(f"{pdfname}_ROOT_occupancy_fine.pdf")
    # for name,h in TH1_F.items(): del h
    # for name,h in TH2_F.items(): del h
    # for name,h in PJF.items(): del h
    # del TH1_F
    # del TH2_F
    # del PJF
    # del cnv1
    #
    #
    #
    # ### occupancy middle
    # TH1_M = {}
    # TH2_M = {}
    # PJM = {}
    # for i,detector in enumerate(detectors):
    #     det = f"ALPIDE_{i}"
    #     TH1_M.update({det:hPixelMatrix1_middle.Clone(f"{hPixelMatrix1_middle.GetName()} {det}")})
    #     TH1_M[det].SetTitle(det)
    #     TH2_M.update({det:hPixelMatrix2_middle.Clone(f"{hPixelMatrix2_middle.GetName()} {det}")})
    #     TH2_M[det].SetTitle(det)
    #     for hit in detector.hits:
    #         XX  = np.array(hit['x']*m_to_mm)
    #         YY  = np.array(hit['y']*m_to_mm)
    #         XP,YP = real_to_pixel_indices(XX,YY, detectors[0].x_min*m_to_mm,detectors[0].x_max*m_to_mm, detectors[0].y_min*m_to_mm,detectors[0].y_max*m_to_mm)
    #         XR,YR = transform_pixel_indices(XP,YP)
    #         TH2_M[det].Fill(XR,YR)
    #         bglob = TH2_M[det].FindBin(XR,YR)
    #         TH1_M[det].AddBinContent(bglob,1)
    # cnv2 = ROOT.TCanvas("cnv22","",1000,1500)
    # cnv2.Divide(2,5)
    # cnv2.cd(1)
    # ROOT.gPad.SetTicks(1,1)
    # TH2_M["ALPIDE_0"].Draw("col")
    # PJM.update({"ALPIDE_0": proj(TH2_M["ALPIDE_0"],projmax=True,name="middle")})
    # PJM["ALPIDE_0"].Draw("hist same")
    # ROOT.gPad.RedrawAxis()
    # cnv2.cd(2)
    # ROOT.gPad.SetTicks(1,1)
    # TH1_M["ALPIDE_0"].Draw("hist")
    # ROOT.gPad.RedrawAxis()
    # cnv2.cd(3)
    # ROOT.gPad.SetTicks(1,1)
    # TH2_M["ALPIDE_1"].Draw("col")
    # PJM.update({"ALPIDE_1": proj(TH2_M["ALPIDE_1"],projmax=True,name="middle")})
    # PJM["ALPIDE_1"].Draw("hist same")
    # ROOT.gPad.RedrawAxis()
    # cnv2.cd(4)
    # ROOT.gPad.SetTicks(1,1)
    # TH1_M["ALPIDE_1"].Draw("hist")
    # ROOT.gPad.RedrawAxis()
    # cnv2.cd(5)
    # ROOT.gPad.SetTicks(1,1)
    # TH2_M["ALPIDE_2"].Draw("col")
    # PJM.update({"ALPIDE_2": proj(TH2_M["ALPIDE_2"],projmax=True,name="middle")})
    # PJM["ALPIDE_2"].Draw("hist same")
    # ROOT.gPad.RedrawAxis()
    # cnv2.cd(6)
    # ROOT.gPad.SetTicks(1,1)
    # TH1_M["ALPIDE_2"].Draw("hist")
    # ROOT.gPad.RedrawAxis()
    # cnv2.cd(7)
    # ROOT.gPad.SetTicks(1,1)
    # TH2_M["ALPIDE_3"].Draw("col")
    # PJM.update({"ALPIDE_3": proj(TH2_M["ALPIDE_3"],projmax=True,name="middle")})
    # PJM["ALPIDE_3"].Draw("hist same")
    # ROOT.gPad.RedrawAxis()
    # cnv2.cd(8)
    # ROOT.gPad.SetTicks(1,1)
    # TH1_M["ALPIDE_3"].Draw("hist")
    # ROOT.gPad.RedrawAxis()
    # cnv2.cd(9)
    # ROOT.gPad.SetTicks(1,1)
    # TH2_M["ALPIDE_4"].Draw("col")
    # PJM.update({"ALPIDE_4": proj(TH2_M["ALPIDE_4"],projmax=True,name="middle")})
    # PJM["ALPIDE_4"].Draw("hist same")
    # ROOT.gPad.RedrawAxis()
    # cnv2.cd(10)
    # ROOT.gPad.SetTicks(1,1)
    # TH1_M["ALPIDE_4"].Draw("hist")
    # ROOT.gPad.RedrawAxis()
    # cnv2.Update()
    # cnv2.SaveAs(f"{pdfname}_ROOT_occupancy_middle.pdf")
    # for name,h in TH1_M.items(): del h
    # for name,h in TH2_M.items(): del h
    # for name,h in PJM.items(): del h
    # del TH1_M
    # del TH2_M
    # del PJM
    # del cnv2
    #
    # print("here 1")
    #
    # ### occupancy coarse
    # TH1_C = {}
    # TH2_C = {}
    # PJC = {}
    # for i,detector in enumerate(detectors):
    #     det = f"ALPIDE_{i}"
    #     TH1_C.update({det:hPixelMatrix1_coarse.Clone(f"{hPixelMatrix1_coarse.GetName()} {det}")})
    #     TH1_C[det].SetTitle(det)
    #     TH2_C.update({det:hPixelMatrix2_coarse.Clone(f"{hPixelMatrix2_coarse.GetName()} {det}")})
    #     TH2_C[det].SetTitle(det)
    #
    #     print(f"here 1 {det}")
    #
    #     for hit in detector.hits:
    #         XX  = np.array(hit['x']*m_to_mm)
    #         YY  = np.array(hit['y']*m_to_mm)
    #         XP,YP = real_to_pixel_indices(XX,YY, detectors[0].x_min*m_to_mm,detectors[0].x_max*m_to_mm, detectors[0].y_min*m_to_mm,detectors[0].y_max*m_to_mm)
    #         XR,YR = transform_pixel_indices(XP,YP)
    #         TH2_C[det].Fill(XR,YR)
    #         bglob = TH2_C[det].FindBin(XR,YR)
    #         TH1_C[det].AddBinContent(bglob,1)
    #     print(f"here 2 {det}")
    #
    # print("here 2.1")
    # cnv3 = ROOT.TCanvas("cnv33","",1000,1500)
    # cnv3.Divide(2,5)
    # cnv3.cd(1)
    # ROOT.gPad.SetTicks(1,1)
    # TH2_C["ALPIDE_0"].Draw("col")
    # PJC.update({"ALPIDE_0": proj(TH2_C["ALPIDE_0"],projmax=True,name="coarse")})
    # PJC["ALPIDE_0"].Draw("hist same")
    # ROOT.gPad.RedrawAxis()
    # print("here 2.2")
    # cnv3.cd(2)
    # ROOT.gPad.SetTicks(1,1)
    # TH1_C["ALPIDE_0"].Draw("hist")
    # ROOT.gPad.RedrawAxis()
    # print("here 2.3")
    # cnv3.cd(3)
    # ROOT.gPad.SetTicks(1,1)
    # TH2_C["ALPIDE_1"].Draw("col")
    # PJC.update({"ALPIDE_1": proj(TH2_C["ALPIDE_1"],projmax=True,name="coarse")})
    # PJC["ALPIDE_1"].Draw("hist same")
    # ROOT.gPad.RedrawAxis()
    # print("here 2.4")
    # cnv3.cd(4)
    # ROOT.gPad.SetTicks(1,1)
    # TH1_C["ALPIDE_1"].Draw("hist")
    # ROOT.gPad.RedrawAxis()
    # print("here 2.5")
    # cnv3.cd(5)
    # ROOT.gPad.SetTicks(1,1)
    # TH2_C["ALPIDE_2"].Draw("col")
    # PJC.update({"ALPIDE_2": proj(TH2_C["ALPIDE_2"],projmax=True,name="coarse")})
    # PJC["ALPIDE_2"].Draw("hist same")
    # ROOT.gPad.RedrawAxis()
    # print("here 2.6")
    # cnv3.cd(6)
    # ROOT.gPad.SetTicks(1,1)
    # TH1_C["ALPIDE_2"].Draw("hist")
    # ROOT.gPad.RedrawAxis()
    # print("here 2.7")
    # cnv3.cd(7)
    # ROOT.gPad.SetTicks(1,1)
    # TH2_C["ALPIDE_3"].Draw("col")
    # PJC.update({"ALPIDE_3": proj(TH2_C["ALPIDE_3"],projmax=True,name="coarse")})
    # PJC["ALPIDE_3"].Draw("hist same")
    # ROOT.gPad.RedrawAxis()
    # print("here 2.8")
    # cnv3.cd(8)
    # ROOT.gPad.SetTicks(1,1)
    # TH1_C["ALPIDE_3"].Draw("hist")
    # ROOT.gPad.RedrawAxis()
    # print("here 2.10")
    # cnv3.cd(9)
    # ROOT.gPad.SetTicks(1,1)
    # TH2_C["ALPIDE_4"].Draw("col")
    # PJC.update({"ALPIDE_4": proj(TH2_C["ALPIDE_4"],projmax=True,name="coarse")})
    # PJC["ALPIDE_4"].Draw("hist same")
    # ROOT.gPad.RedrawAxis()
    # print("here 2.11")
    # cnv3.cd(10)
    # ROOT.gPad.SetTicks(1,1)
    # TH1_C["ALPIDE_4"].Draw("hist")
    # ROOT.gPad.RedrawAxis()
    # cnv3.Update()
    # cnv3.SaveAs(f"{pdfname}_ROOT_occupancy_coarse.pdf")
    # print("here 2.12")
    # for name,h in TH1_C.items(): del h
    # for name,h in TH2_C.items(): del h
    # for name,h in PJC.items(): del h
    # del TH1_C
    # del TH2_C
    # del PJC
    # del cnv3
    #
    #
    #
    # print("here 3")




    ### plot the hits COARSELY 2D:
    fig, axs = plt.subplots(1, 5, figsize=(10, 3.5), sharex=True, sharey=True, tight_layout=True)
    P0 = []
    P0_X = []
    P0_Y = []
    hOcc = []
    for i,detector in enumerate(detectors):
        X = []
        Y = []
        P = []
        P_X = []
        P_Y = []
        for hit in detector.hits:
            pid = hit['particle_id']
            xx  = hit['x']
            yy  = hit['y']
            zz  = hit['z']
            px  = initial_states[pid][3]
            py  = initial_states[pid][4]
            pz  = initial_states[pid][5]
            X.append(xx*m_to_mm)
            Y.append(yy*m_to_mm) 
            if(fullacc and pid not in list_good_tracks): continue
            P.append(pz/GeV_to_kgms)
            P_X.append(px/GeV_to_kgms)
            P_Y.append(py/GeV_to_kgms)
        # hOcc.append( axs[i].hist2d(X, Y, bins=(128,256),range=[[detectors[0].x_min*m_to_mm,detectors[0].x_max*m_to_mm],[detectors[0].y_min*m_to_mm,detectors[0].y_max*m_to_mm]], rasterized=True) )
        hOcc.append( axs[i].hist2d(X, Y, bins=(64,128),range=[[detectors[0].x_min*m_to_mm,detectors[0].x_max*m_to_mm],[detectors[0].y_min*m_to_mm,detectors[0].y_max*m_to_mm]], rasterized=True) )        
        ## h, xedges, yedges, image = ax.hist2d(...)
        max_idx = np.unravel_index(np.argmax(hOcc[i][0]), hOcc[i][0].shape) # Find the index of the maximum value in the 2D array
        tot_counts = hOcc[i][0].sum()
        max_counts = hOcc[i][0].max()
        x_of_max   = hOcc[i][1][max_idx[0]]
        y_of_max   = hOcc[i][2][max_idx[1]]
        # print(f"{detector.name}: x={statistics.mean(X):.3f}+-{statistics.stdev(X):.3f} mm with total counts={tot_counts} and max counts={max_counts} at (xmax,ymax)=({x_of_max:.2f},{y_of_max:.2f})")
        if(i==0):
            P0 = P
            P0_X = P_X
            P0_Y = P_Y
        axs[i].set_xlabel('X [mm]')
        axs[i].set_ylabel('Y [mm]')
        axs[i].set_title(f'ALPIDE_{i}')
        plt.locator_params(axis='x', nbins=10)
        plt.locator_params(axis='y', nbins=10)
        axs[i].xaxis.set_minor_locator(AutoMinorLocator(10))
        axs[i].yaxis.set_minor_locator(AutoMinorLocator(10))
    plt.tight_layout()
    plt.savefig(f"{pdfname}_occupancy_coarse1.pdf")
    if(doshw): plt.show()
    
    
    ### root histogram of the energies
    for p in P0:
        hPz_full.Fill(p)
        hPz_small.Fill(p)
        hPz_zoom.Fill(p)
    for p in P0_X:
        hPx_full.Fill(p)
        hPx_small.Fill(p)
        hPx_zoom.Fill(p)
    for p in P0_Y:
        hPy_full.Fill(p)
        hPy_small.Fill(p)
        hPy_zoom.Fill(p)
    fOut = ROOT.TFile(f"{pdfname}_Toy_MC_hPz_zoom.root","RECREATE")
    fOut.cd()
    hPz_full.Write()
    hPz_small.Write()
    hPz_zoom.Write()
    hPx_full.Write()
    hPx_small.Write()
    hPx_zoom.Write()
    hPy_full.Write()
    hPy_small.Write()
    hPy_zoom.Write()
    fOut.Close()


    ### plot the hits FINELY 2D:
    fig, axs = plt.subplots(1, 5, figsize=(10, 3.5), sharex=True, sharey=True, tight_layout=True)
    hOcc = []
    for i,detector in enumerate(detectors):
        X = []
        Y = []
        P = []
        for hit in detector.hits:
            pid = hit['particle_id']
            xx  = hit['x']
            yy  = hit['y']
            zz  = hit['z']
            pz  = initial_states[pid][5]
            X.append(xx*m_to_mm)
            Y.append(yy*m_to_mm)
            if(fullacc and pid not in list_good_tracks): continue
            P.append(pz/GeV_to_kgms)
        hOcc.append( axs[i].hist2d(X, Y, bins=(npix_y+1,npix_x+1),range=[[detectors[0].x_min*m_to_mm,detectors[0].x_max*m_to_mm],[detectors[0].y_min*m_to_mm,detectors[0].y_max*m_to_mm]], rasterized=True) )
        axs[i].set_xlabel('X [mm]')
        axs[i].set_ylabel('Y [mm]')
        axs[i].set_title(f'ALPIDE_{i}')
        plt.locator_params(axis='x', nbins=10)
        plt.locator_params(axis='y', nbins=10)
        axs[i].xaxis.set_minor_locator(AutoMinorLocator(10))
        axs[i].yaxis.set_minor_locator(AutoMinorLocator(10))
    plt.tight_layout()
    plt.savefig(f"{pdfname}_occupancy_fine.pdf")
    if(doshw): plt.show()
    
    
    ### plot the hits COARSELY 2D, transformed to EUDAQ (but just shape wise):
    # fig, axs = plt.subplots(1, 5, figsize=(10,3.5), sharex=True, sharey=True, tight_layout=True)
    fig, axs = plt.subplots(5, 1, figsize=(3.5,10), sharex=True, sharey=True, tight_layout=True)
    hOcc = []
    for i,detector in enumerate(detectors):
        X = []
        Y = []
        P = []
        for hit in detector.hits:
            pid = hit['particle_id']
            xx  = hit['x']
            yy  = hit['y']
            zz  = hit['z']
            pz  = initial_states[pid][5]
            X.append(xx*m_to_mm)
            Y.append(yy*m_to_mm)
            if(fullacc and pid not in list_good_tracks): continue
            P.append(pz/GeV_to_kgms)
        XPIX,YPIX = real_to_pixel_indices(X, Y, detectors[0].x_min*m_to_mm,detectors[0].x_max*m_to_mm, detectors[0].y_min*m_to_mm,detectors[0].y_max*m_to_mm)
        XPIXR,YPIXR = transform_pixel_indices(XPIX,YPIX)
        
        # hOcc.append( axs[i].hist2d(XPIXR, YPIXR, bins=(128,256),range=[[0,1023],[0,511]], rasterized=True) )
        hOcc.append( axs[i].hist2d(XPIXR, YPIXR, bins=(256,128),range=[[0,1023],[0,511]], rasterized=True) )
        axs[i].set_xlabel('x [pixels]')
        axs[i].set_ylabel('y [pixels]')
        axs[i].set_title(f'ALPIDE_{i}')
        plt.locator_params(axis='x', nbins=10)
        plt.locator_params(axis='y', nbins=10)
        axs[i].xaxis.set_minor_locator(AutoMinorLocator(10))
        axs[i].yaxis.set_minor_locator(AutoMinorLocator(10))
    plt.tight_layout()
    plt.savefig(f"{pdfname}_occupancy_trnsfrmd.pdf")
    if(doshw): plt.show()


    fig, axs = plt.subplots(5, 1, figsize=(10, 10), sharex=True, sharey=True, tight_layout=True)
    h1Occ = []
    for i,hist2D in enumerate(hOcc):
        bin_numbers, occupancy_counts = hist2d_to_1d_root_style(hist2D)
        # Get dimensions for the histogram
        counts, _, _, _ = hist2D
        total_bins = counts.size
        axs[i].plot(bin_numbers, occupancy_counts, linewidth=1, rasterized=True)
        axs[i].set_xlabel('Global Pixel Number')
        axs[i].set_ylabel('Occupancy')
        axs[i].set_title(f'ALPIDE_{i}')
        axs[i].grid(True, alpha=0.3)
        axs[i].set_xlim(-0.5, total_bins - 0.5)
        plt.locator_params(axis='x', nbins=10)
        axs[i].xaxis.set_minor_locator(AutoMinorLocator(10))
        axs[i].yaxis.set_minor_locator(AutoMinorLocator(10))
        axs[i].grid(True,linewidth=0.25,alpha=0.25)
    plt.tight_layout()
    plt.savefig(f"{pdfname}_occupancy_1D.pdf")
    if(doshw): plt.show()
    
    
    
    
    
    ###############################################
    ###############################################
    ###############################################
    ########### diploe exit plots #################
    ### dipole line
    dipoleTPL = ROOT.TPolyLine()
    dipoleTPL.SetNextPoint(dipole.x_min*m_to_mm,dipole.y_min*m_to_mm)
    dipoleTPL.SetNextPoint(dipole.x_min*m_to_mm,dipole.y_max*m_to_mm)
    dipoleTPL.SetNextPoint(dipole.x_max*m_to_mm,dipole.y_max*m_to_mm)
    dipoleTPL.SetNextPoint(dipole.x_max*m_to_mm,dipole.y_min*m_to_mm)
    dipoleTPL.SetNextPoint(dipole.x_min*m_to_mm,dipole.y_min*m_to_mm)
    dipoleTPL.SetLineColor(ROOT.kBlue)
    dipoleTPL.SetLineWidth(1)
    flangeTPL = ROOT.TPolyLine()
    flangeTPL.SetNextPoint(flange.x_min*m_to_mm,flange.y_min*m_to_mm)
    flangeTPL.SetNextPoint(flange.x_min*m_to_mm,flange.y_max*m_to_mm)
    flangeTPL.SetNextPoint(flange.x_max*m_to_mm,flange.y_max*m_to_mm)
    flangeTPL.SetNextPoint(flange.x_max*m_to_mm,flange.y_min*m_to_mm)
    flangeTPL.SetNextPoint(flange.x_min*m_to_mm,flange.y_min*m_to_mm)
    flangeTPL.SetLineColor(ROOT.kAzure+1)
    flangeTPL.SetLineWidth(1)
    # histD_entrance = ROOT.TH2D("histD_entrance","Dipole entrance plane;x [mm];y [mm];particles",120,-80,+80, 120,-70,+90)
    # histD_exit     = ROOT.TH2D("histD_exit",    "Dipole exit plane;x [mm];y [mm];particles",    120,-80,+80, 120,-70,+90)
    ztitle = "Particles in full acceptance" if(fullacc) else "Particles"
    histD_entrance = ROOT.TH2D("histD_entrance",f"Dipole entrance plane;x_{{LAB}} [mm];y_{{LAB}} [mm];{ztitle}",200,-80,+80, 200,-70,+90)
    histD_exit     = ROOT.TH2D("histD_exit",    f"Dipole exit plane;x_{{LAB}} [mm];y_{{LAB}} [mm];{ztitle}",    200,-80,+80, 200,-70,+90)
    
    
    def getrect():
        ### the dipole exit rectngles:
        rectD1 = plt.Rectangle(
            (dipole.x_min, dipole.y_min),
            dipole.x_max - dipole.x_min,
            dipole.y_max - dipole.y_min,
            fill=False, edgecolor='blue'
        )
        rectD2 = plt.Rectangle(
            (dipole.x_min, dipole.y_min),
            dipole.x_max - dipole.x_min,
            dipole.y_max - dipole.y_min,
            fill=False, edgecolor='blue'
        )
        return rectD1, rectD2
    
    ### plot the exit plane at the dipole faces
    fig, axs = plt.subplots(1, 2, figsize=(9, 5), tight_layout=True)
    rectD1,rectD2 = getrect()
    X = []
    Y = []
    for point in dipole.hits:
        pid = point['particle_id']
        if(fullacc and pid not in list_good_tracks): continue
        if(abs(point['z']-dipole.z_min)>1e-6):       continue
        X.append(point['x'])
        Y.append(point['y'])
        histD_entrance.Fill(point['x']*m_to_mm,point['y']*m_to_mm)
    hDentzoom = axs[0].hist2d(X, Y, bins=(200,200),range=[[dipole.x_min*1.2,dipole.x_max*1.2],[dipole.y_min*1.1,dipole.y_max*1.1]], rasterized=True)
    axs[0].set_xlabel('X [m]')
    axs[0].set_ylabel('Y [m]')
    axs[0].set_title(f'Dipole entrance')
    axs[0].add_patch(rectD1)
    plt.locator_params(axis='x', nbins=10)
    plt.locator_params(axis='y', nbins=10)
    axs[0].xaxis.set_minor_locator(AutoMinorLocator(10))
    axs[0].yaxis.set_minor_locator(AutoMinorLocator(10))
    axs[0].grid(True,linewidth=0.25,alpha=0.25,which='major')
    
    rectD1,rectD2 = getrect()
    X = []
    Y = []
    for point in dipole.hits:
        pid = point['particle_id']
        if(fullacc and pid not in list_good_tracks): continue
        if(abs(point['z']-dipole.z_max)>1e-6):       continue
        X.append(point['x'])
        Y.append(point['y'])
        histD_exit.Fill(point['x']*m_to_mm,point['y']*m_to_mm)
    hDextzoom = axs[1].hist2d(X, Y, bins=(200,200),range=[[dipole.x_min*1.2,dipole.x_max*1.2],[dipole.y_min*1.1,dipole.y_max*1.1]], rasterized=True)
    axs[1].set_xlabel('X [m]')
    axs[1].set_ylabel('Y [m]')
    axs[1].set_title(f'Dipole exit')
    axs[1].add_patch(rectD1)
    plt.locator_params(axis='x', nbins=10)
    plt.locator_params(axis='y', nbins=10)
    axs[1].xaxis.set_minor_locator(AutoMinorLocator(10))
    axs[1].yaxis.set_minor_locator(AutoMinorLocator(10))
    axs[1].grid(True,linewidth=0.25,alpha=0.25,which='major')
    
    plt.tight_layout()
    plt.savefig(f"{pdfname}_dipole_exit_zoom.pdf")
    if(doshw): plt.show()
    
    
    ############ root
    cnv = ROOT.TCanvas("cnv_dipole_entrance","",550,500)
    cnv.SetTicks(1,1)
    cnv.SetGridx()
    cnv.SetGridy()
    histD_entrance.Draw("colz")
    histD_entrance.GetZaxis().SetTitleOffset(1.3)
    dipoleTPL.Draw()
    flangeTPL.Draw()
    s = ROOT.TLatex()
    s.SetNDC(1)
    s.SetTextAlign(13)
    s.SetTextColor(ROOT.kBlack)
    s.SetTextFont(22)
    s.SetTextSize(0.045)
    s.DrawLatex(0.17,0.88,"Toy MC")
    #
    s = ROOT.TLatex()
    s.SetNDC(1)
    s.SetTextAlign(13)
    s.SetTextColor(ROOT.kBlack)
    s.SetTextFont(132)
    s.SetTextSize(0.045)
    s.DrawLatex(0.17,0.83,f"#mu_{{x}}={histD_entrance.GetMean(1):.1f} mm, #sigma_{{x}}={histD_entrance.GetStdDev(1):.1f} mm")
    s.DrawLatex(0.17,0.78,f"#mu_{{y}}={histD_entrance.GetMean(2):.1f} mm, #sigma_{{y}}={histD_entrance.GetStdDev(2):.1f} mm")
    #
    s = ROOT.TLatex()
    s.SetNDC(1)
    s.SetTextAlign(13)
    s.SetTextColor(ROOT.kBlack)
    s.SetTextFont(132)
    s.SetTextSize(0.045)
    s.DrawLatex(0.35,0.88,"(Global alignment: as designed)")
    #
    s = ROOT.TLatex()
    s.SetNDC(1)
    s.SetTextAlign(13)
    s.SetTextColor(ROOT.kBlue)
    s.SetTextFont(132)
    s.SetTextSize(0.045)
    s.DrawLatex(0.367,0.69,"Dipole aperture")
    #
    s = ROOT.TLatex()
    s.SetNDC(1)
    s.SetTextAlign(13)
    s.SetTextColor(ROOT.kAzure+1)
    s.SetTextFont(132)
    s.SetTextSize(0.045)
    s.DrawLatex(0.15,0.65,"Flange aperture")
    ROOT.gPad.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{pdfname}_ROOT_dipole_entrance.pdf")
    
    
    cnv = ROOT.TCanvas("cnv_dipole_exit","",550,500)
    cnv.SetTicks(1,1)
    cnv.SetGridx()
    cnv.SetGridy()
    histD_exit.GetZaxis().SetTitleOffset(1.3)
    histD_exit.Draw("colz")
    dipoleTPL.Draw()
    flangeTPL.Draw()
    s = ROOT.TLatex()
    s.SetNDC(1)
    s.SetTextAlign(13)
    s.SetTextColor(ROOT.kBlack)
    s.SetTextFont(22)
    s.SetTextSize(0.045)
    s.DrawLatex(0.17,0.88,"Toy MC")
    #
    s = ROOT.TLatex()
    s.SetNDC(1)
    s.SetTextAlign(13)
    s.SetTextColor(ROOT.kBlack)
    s.SetTextFont(132)
    s.SetTextSize(0.045)
    s.DrawLatex(0.17,0.83,f"#mu_{{x}}={histD_exit.GetMean(1):.1f} mm, #sigma_{{x}}={histD_exit.GetStdDev(1):.1f} mm")
    s.DrawLatex(0.17,0.78,f"#mu_{{y}}={histD_exit.GetMean(2):.1f} mm, #sigma_{{y}}={histD_exit.GetStdDev(2):.1f} mm")
    #
    s = ROOT.TLatex()
    s.SetNDC(1)
    s.SetTextAlign(13)
    s.SetTextColor(ROOT.kBlack)
    s.SetTextFont(132)
    s.SetTextSize(0.045)
    s.DrawLatex(0.35,0.88,"(Global alignment: as designed)")
    #
    s = ROOT.TLatex()
    s.SetNDC(1)
    s.SetTextAlign(13)
    s.SetTextColor(ROOT.kBlue)
    s.SetTextFont(132)
    s.SetTextSize(0.045)
    s.DrawLatex(0.367,0.69,"Dipole aperture")
    #
    s = ROOT.TLatex()
    s.SetNDC(1)
    s.SetTextAlign(13)
    s.SetTextColor(ROOT.kAzure+1)
    s.SetTextFont(132)
    s.SetTextSize(0.045)
    s.DrawLatex(0.15,0.65,"Flange aperture")
    ROOT.gPad.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{pdfname}_ROOT_dipole_exit.pdf")
    
    
    
    
    ### plot the entrance and exit plane at the dipole:
    fig, axs = plt.subplots(1, 2, figsize=(9, 5), tight_layout=True)
    rectD1,rectD2 = getrect()
    X = []
    Y = []
    for point in dipole.hits:
        pid = point['particle_id']
        if(fullacc and pid not in list_good_tracks): continue
        if(abs(point['z']-dipole.z_min)>1e-6):       continue
        X.append(point['x'])
        Y.append(point['y'])
    hDent = axs[0].hist2d(X, Y, bins=(120,120),range=[[-80*mm_to_m,+80*mm_to_m],[-70*mm_to_m,+90*mm_to_m]], rasterized=True)
    axs[0].set_xlabel('X [m]')
    axs[0].set_ylabel('Y [m]')
    axs[0].set_title(f'Dipole entrance')
    axs[0].add_patch(rectD2)
    plt.locator_params(axis='x', nbins=10)
    plt.locator_params(axis='y', nbins=10)
    axs[0].xaxis.set_minor_locator(AutoMinorLocator(10))
    axs[0].yaxis.set_minor_locator(AutoMinorLocator(10))
    axs[0].grid(True,linewidth=0.25,alpha=0.25,which='major')
    
    rectD1,rectD2 = getrect()
    X = []
    Y = []
    for point in dipole.hits:
        pid = point['particle_id']
        if(fullacc and pid not in list_good_tracks): continue
        if(abs(point['z']-dipole.z_max)>1e-6):       continue
        X.append(point['x'])
        Y.append(point['y'])
    hDext = axs[1].hist2d(X, Y, bins=(120,120),range=[[-80*mm_to_m,+80*mm_to_m],[-70*mm_to_m,+90*mm_to_m]], rasterized=True)
    axs[1].set_xlabel('X [m]')
    axs[1].set_ylabel('Y [m]')
    axs[1].set_title(f'Dipole exit')
    axs[1].add_patch(rectD2)
    plt.locator_params(axis='x', nbins=10)
    plt.locator_params(axis='y', nbins=10)
    axs[1].xaxis.set_minor_locator(AutoMinorLocator(10))
    axs[1].yaxis.set_minor_locator(AutoMinorLocator(10))
    axs[1].grid(True,linewidth=0.25,alpha=0.25,which='major')
    
    plt.tight_layout()
    plt.savefig(f"{pdfname}_dipole_exit.pdf")
    if(doshw): plt.show()
    
    
    
    
    
    ### plot the quads entrance and exit
    fig, axs = plt.subplots(2, 3, figsize=(9, 8), sharex=True, sharey=True, tight_layout=True)
    
    X0 = []
    Y0 = []
    for point in quad0.hits:
        pid = point['particle_id']
        if(fullacc and pid not in list_good_tracks): continue
        if(abs(point['z']-quad0.z_min)>1e-6):        continue
        X0.append(point['x'])
        Y0.append(point['y'])
    hq0ent = axs[0][0].hist2d(X0, Y0, bins=(150,150),range=[[quad0.x_min*1.5,quad0.x_max*1.5],[quad0.y_min*1.5,quad0.y_max*1.5]], rasterized=True)
    axs[0][0].set_xlim(quad0.x_min * 1.5, quad0.x_max * 1.5)
    axs[0][0].set_ylim(quad0.y_min * 1.5, quad0.y_max * 1.5)
    axs[0][0].set_xlabel('X [m]')
    axs[0][0].set_ylabel('Y [m]')
    axs[0][0].set_title(f'Quad0 entrance')
    plt.locator_params(axis='x', nbins=10)
    plt.locator_params(axis='y', nbins=10)
    axs[0][0].xaxis.set_minor_locator(AutoMinorLocator(10))
    axs[0][0].yaxis.set_minor_locator(AutoMinorLocator(10))
    # axs[0][0].grid(True,linewidth=0.25,alpha=0.25)
    X1 = []
    Y1 = []
    for point in quad1.hits:
        pid = point['particle_id']
        if(fullacc and pid not in list_good_tracks): continue
        if(abs(point['z']-quad1.z_min)>1e-6):        continue
        X1.append(point['x'])
        Y1.append(point['y'])
    hq1ent = axs[0][1].hist2d(X1, Y1, bins=(150,150),range=[[quad1.x_min*1.5,quad1.x_max*1.5],[quad1.y_min*1.5,quad1.y_max*1.5]], rasterized=True)
    axs[0][1].set_xlim(quad1.x_min * 1.5, quad1.x_max * 1.5)
    axs[0][1].set_ylim(quad1.y_min * 1.5, quad1.y_max * 1.5)
    axs[0][1].set_xlabel('X [m]')
    axs[0][1].set_ylabel('Y [m]')
    axs[0][1].set_title(f'Quad1 entrance')
    plt.locator_params(axis='x', nbins=10)
    plt.locator_params(axis='y', nbins=10)
    axs[0][1].xaxis.set_minor_locator(AutoMinorLocator(10))
    axs[0][1].yaxis.set_minor_locator(AutoMinorLocator(10))
    # axs[0][1].grid(True,linewidth=0.25,alpha=0.25)
    X2 = []
    Y2 = []
    for point in quad2.hits:
        pid = point['particle_id']
        if(fullacc and pid not in list_good_tracks): continue
        if(abs(point['z']-quad2.z_min)>1e-6):        continue
        X2.append(point['x'])
        Y2.append(point['y'])
    hq2ent = axs[0][2].hist2d(X2, Y2, bins=(150,150),range=[[quad2.x_min*1.5,quad2.x_max*1.5],[quad2.y_min*1.5,quad2.y_max*1.5]], rasterized=True)
    axs[0][2].set_xlim(quad2.x_min * 1.5, quad2.x_max * 1.5)
    axs[0][2].set_ylim(quad2.y_min * 1.5, quad2.y_max * 1.5)
    axs[0][2].set_xlabel('X [m]')
    axs[0][2].set_ylabel('Y [m]')
    axs[0][2].set_title(f'Quad2 entrance')
    plt.locator_params(axis='x', nbins=10)
    plt.locator_params(axis='y', nbins=10)
    axs[0][2].xaxis.set_minor_locator(AutoMinorLocator(10))
    axs[0][2].yaxis.set_minor_locator(AutoMinorLocator(10))
    # axs[0][2].grid(True,linewidth=0.25,alpha=0.25)
    
    X0 = []
    Y0 = []
    for point in quad0.hits:
        pid = point['particle_id']
        if(fullacc and pid not in list_good_tracks): continue
        if(abs(point['z']-quad0.z_max)>1e-6):        continue
        X0.append(point['x'])
        Y0.append(point['y'])
    hq0ext = axs[1][0].hist2d(X0, Y0, bins=(150,150),range=[[quad0.x_min*1.5,quad0.x_max*1.5],[quad0.y_min*1.5,quad0.y_max*1.5]], rasterized=True)
    axs[1][0].set_xlim(quad0.x_min * 1.5, quad0.x_max * 1.5)
    axs[1][0].set_ylim(quad0.y_min * 1.5, quad0.y_max * 1.5)
    axs[1][0].set_xlabel('X [m]')
    axs[1][0].set_ylabel('Y [m]')
    axs[1][0].set_title(f'Quad0 exit')
    plt.locator_params(axis='x', nbins=10)
    plt.locator_params(axis='y', nbins=10)
    axs[1][0].xaxis.set_minor_locator(AutoMinorLocator(10))
    axs[1][0].yaxis.set_minor_locator(AutoMinorLocator(10))
    # axs[1][0].grid(True,linewidth=0.25,alpha=0.25)
    X1 = []
    Y1 = []
    for point in quad1.hits:
        pid = point['particle_id']
        if(fullacc and pid not in list_good_tracks): continue
        if(abs(point['z']-quad1.z_max)>1e-6):        continue
        X1.append(point['x'])
        Y1.append(point['y'])
    hq1ext = axs[1][1].hist2d(X1, Y1, bins=(150,150),range=[[quad1.x_min*1.5,quad1.x_max*1.5],[quad1.y_min*1.5,quad1.y_max*1.5]], rasterized=True)
    axs[1][1].set_xlim(quad1.x_min * 1.5, quad1.x_max * 1.5)
    axs[1][1].set_ylim(quad1.y_min * 1.5, quad1.y_max * 1.5)
    axs[1][1].set_xlabel('X [m]')
    axs[1][1].set_ylabel('Y [m]')
    axs[1][1].set_title(f'Quad1 exit')
    plt.locator_params(axis='x', nbins=10)
    plt.locator_params(axis='y', nbins=10)
    axs[1][1].xaxis.set_minor_locator(AutoMinorLocator(10))
    axs[1][1].yaxis.set_minor_locator(AutoMinorLocator(10))
    # axs[1][1].grid(True,linewidth=0.25,alpha=0.25)
    X2 = []
    Y2 = []
    for point in quad2.hits:
        pid = point['particle_id']
        if(fullacc and pid not in list_good_tracks): continue
        if(abs(point['z']-quad2.z_max)>1e-6):        continue
        X2.append(point['x'])
        Y2.append(point['y'])
    hq2ext = axs[1][2].hist2d(X2, Y2, bins=(150,150),range=[[quad2.x_min*1.5,quad2.x_max*1.5],[quad2.y_min*1.5,quad2.y_max*1.5]], rasterized=True)
    axs[1][2].set_xlim(quad2.x_min * 1.5, quad2.x_max * 1.5)
    axs[1][2].set_ylim(quad2.y_min * 1.5, quad2.y_max * 1.5)
    axs[1][2].set_xlabel('X [m]')
    axs[1][2].set_ylabel('Y [m]')
    axs[1][2].set_title(f'Quad2 exit')
    plt.locator_params(axis='x', nbins=10)
    plt.locator_params(axis='y', nbins=10)
    axs[1][2].xaxis.set_minor_locator(AutoMinorLocator(10))
    axs[1][2].yaxis.set_minor_locator(AutoMinorLocator(10))
    # axs[1][2].grid(True,linewidth=0.25,alpha=0.25)
    
    plt.tight_layout()
    plt.savefig(f"{pdfname}_quads_exit.pdf")
    if(doshw): plt.show()
    
    
    
    
    
    ### plot the divergence
    fig, axs = plt.subplots(1, 2, figsize=(9, 5), tight_layout=True)
    XX = []
    PX = []
    YY = []
    PY = []
    for pid,state in enumerate(initial_states):
        ### state = [XX,YY,ZZ, PX,PY,PZ, MM,QQ]
        xx = state[0]
        px = state[3]/GeV_to_kgms
        yy = state[1]
        py = state[4]/GeV_to_kgms
        XX.append( xx )
        PX.append( px )
        YY.append( yy )
        PY.append( py )
        
    hdivx = axs[0].hist2d(XX, PX, bins=(200,200), rasterized=True)
    axs[0].set_xlabel('X [m]')
    axs[0].set_ylabel('PX [GeV]')
    plt.locator_params(axis='x', nbins=10)
    plt.locator_params(axis='y', nbins=10)
    axs[0].xaxis.set_minor_locator(AutoMinorLocator(10))
    axs[0].yaxis.set_minor_locator(AutoMinorLocator(10))
    axs[0].grid(True,linewidth=0.25,alpha=0.25)

    hdivy = axs[1].hist2d(YY, PY, bins=(200,200), rasterized=True)
    axs[1].set_xlabel('Y [m]')
    axs[1].set_ylabel('PY [GeV]')
    plt.locator_params(axis='x', nbins=10)
    plt.locator_params(axis='y', nbins=10)
    axs[1].xaxis.set_minor_locator(AutoMinorLocator(10))
    axs[1].yaxis.set_minor_locator(AutoMinorLocator(10))
    axs[1].grid(True,linewidth=0.25,alpha=0.25)

    fig.suptitle(f"z = {Z0} [cm]", fontsize=16) # Add overall title

    plt.tight_layout()
    plt.savefig(f"{pdfname}_divergence.pdf")
    if(doshw): plt.show()
    
    
    
    
    
    
    
    ## example how to get the x,y at any point along the trajectory
    ### plot the initial position from the solution directly
    z_target = Z0_m ### Be window or Al foil
    XX = []
    YY = []
    PX = []
    PY = []
    for t,trajectory in enumerate(particle_trajectories):
        collision = particle_collisions[t]
        state = get_state_at_z(trajectory,collision,z_target)
        if(len(state)>0):
            XX.append(state[0])
            YY.append(state[1])
            PX.append(state[3]/GeV_to_kgms)
            PY.append(state[4]/GeV_to_kgms)
    ### plot the divergence
    fig, axs = plt.subplots(1, 2, figsize=(9, 5), tight_layout=True)
    hdivx = axs[0].hist2d(XX, PX, bins=(200,200), rasterized=True)
    axs[0].set_xlabel('X [m]')
    axs[0].set_ylabel('PX [GeV]')
    plt.locator_params(axis='x', nbins=10)
    plt.locator_params(axis='y', nbins=10)
    axs[0].xaxis.set_minor_locator(AutoMinorLocator(10))
    axs[0].yaxis.set_minor_locator(AutoMinorLocator(10))
    axs[0].grid(True,linewidth=0.25,alpha=0.25)

    hdivy = axs[1].hist2d(YY, PY, bins=(200,200), rasterized=True)
    axs[1].set_xlabel('Y [m]')
    axs[1].set_ylabel('PY [GeV]')
    plt.locator_params(axis='x', nbins=10)
    plt.locator_params(axis='y', nbins=10)
    axs[1].xaxis.set_minor_locator(AutoMinorLocator(10))
    axs[1].yaxis.set_minor_locator(AutoMinorLocator(10))
    axs[1].grid(True,linewidth=0.25,alpha=0.25)

    fig.suptitle(f"From trajectory at z = {z_target} [m]", fontsize=16) # Add overall title

    plt.tight_layout()
    plt.savefig(f"{pdfname}_divergence_at_z0_from_trajectory.pdf")
    if(doshw): plt.show()
    
    
    
    ## example how to get the x,y at any point along the trajectory
    ### plot the position just before the first quad from the solution directly
    z_target = quad0.z_min-0.001 ### 
    XX = []
    YY = []
    PX = []
    PY = []
    for t,trajectory in enumerate(particle_trajectories):
        collision = particle_collisions[t]
        state = get_state_at_z(trajectory,collision,z_target)
        if(len(state)>0):
            XX.append(state[0])
            YY.append(state[1])
            PX.append(state[3]/GeV_to_kgms)
            PY.append(state[4]/GeV_to_kgms)
    ### plot the divergence
    fig, axs = plt.subplots(1, 2, figsize=(9, 5), tight_layout=True)
    hdivx = axs[0].hist2d(XX, PX, bins=(200,200), rasterized=True)
    axs[0].set_xlabel('X [m]')
    axs[0].set_ylabel('PX [GeV]')
    plt.locator_params(axis='x', nbins=10)
    plt.locator_params(axis='y', nbins=10)
    axs[0].xaxis.set_minor_locator(AutoMinorLocator(10))
    axs[0].yaxis.set_minor_locator(AutoMinorLocator(10))
    axs[0].grid(True,linewidth=0.25,alpha=0.25)

    hdivy = axs[1].hist2d(YY, PY, bins=(200,200), rasterized=True)
    axs[1].set_xlabel('Y [m]')
    axs[1].set_ylabel('PY [GeV]')
    plt.locator_params(axis='x', nbins=10)
    plt.locator_params(axis='y', nbins=10)
    axs[1].xaxis.set_minor_locator(AutoMinorLocator(10))
    axs[1].yaxis.set_minor_locator(AutoMinorLocator(10))
    axs[1].grid(True,linewidth=0.25,alpha=0.25)

    fig.suptitle(f"From trajectory at z = {z_target} [m]", fontsize=16) # Add overall title

    plt.tight_layout()
    plt.savefig(f"{pdfname}_divergence_at_quad0zmin_from_trajectory.pdf")
    if(doshw): plt.show()
    
    
    
    
    
    
    z_target = quad0.z_min-0.001 ### 
    XX = []
    YY = []
    PX = []
    PY = []
    for state in initial_states:
        states_at_z_vac = propagate_state_in_vacuum_to_z(state,z_target)
        XX.append(states_at_z_vac[0])
        YY.append(states_at_z_vac[1])
        PX.append(states_at_z_vac[3]/GeV_to_kgms)
        PY.append(states_at_z_vac[4]/GeV_to_kgms)
    ### plot the divergence
    fig, axs = plt.subplots(1, 2, figsize=(9, 5), tight_layout=True)
    hdivx = axs[0].hist2d(XX, PX, bins=(200,200), rasterized=True)
    axs[0].set_xlabel('X [m]')
    axs[0].set_ylabel('PX [GeV]')
    plt.locator_params(axis='x', nbins=10)
    plt.locator_params(axis='y', nbins=10)
    axs[0].xaxis.set_minor_locator(AutoMinorLocator(10))
    axs[0].yaxis.set_minor_locator(AutoMinorLocator(10))
    axs[0].grid(True,linewidth=0.25,alpha=0.25)

    hdivy = axs[1].hist2d(YY, PY, bins=(200,200), rasterized=True)
    axs[1].set_xlabel('Y [m]')
    axs[1].set_ylabel('PY [GeV]')
    plt.locator_params(axis='x', nbins=10)
    plt.locator_params(axis='y', nbins=10)
    axs[1].xaxis.set_minor_locator(AutoMinorLocator(10))
    axs[1].yaxis.set_minor_locator(AutoMinorLocator(10))
    axs[1].grid(True,linewidth=0.25,alpha=0.25)

    fig.suptitle(f"From initial state at z = {z_target} [m]", fontsize=16) # Add overall title

    plt.tight_layout()
    plt.savefig(f"{pdfname}_divergence_at_quad0zmin_from_initstate.pdf")
    if(doshw): plt.show()
    
    
    
    
    
    
    
    
    
    
    
    ### plot the energies
    fig, axs = plt.subplots(1, 2, figsize=(6, 3), tight_layout=True)
    # hpz0 = axs[0].hist(PZ0, bins=50,range=(Emin,Emax), rasterized=True)
    # hp0  = axs[1].hist(P0,  bins=50,range=(1.5,4.5), rasterized=True)
    hpz0 = axs[0].hist(PZ0, bins=50,range=(Emin,Emax), rasterized=True)
    hp0  = axs[1].hist(P0,  bins=50,range=(Emin,Emax), rasterized=True)

    axs[0].set_xlim(Emin,Emax)
    axs[0].set_yscale("log")
    axs[0].set_xlabel('E [GeV]')
    axs[0].set_ylabel('Particles')
    axs[0].set_title(f'Generated')
    plt.locator_params(axis='x', nbins=10)
    axs[0].xaxis.set_minor_locator(AutoMinorLocator(10))
    axs[0].yaxis.set_minor_locator(AutoMinorLocator(10))
    axs[0].grid(True,linewidth=0.25,alpha=0.25)
    
    axs[1].set_xlim(Emin,Emax)
    axs[1].set_yscale("log")
    axs[1].set_xlabel('E [GeV]')
    axs[1].set_ylabel('Particles')
    axs[1].set_title(f'In Acceptance')
    plt.locator_params(axis='x', nbins=10)
    axs[1].xaxis.set_minor_locator(AutoMinorLocator(10))
    # axs[1].yaxis.set_minor_locator(AutoMinorLocator(10))
    axs[1].grid(True,linewidth=0.25,alpha=0.25)
    
    plt.tight_layout()
    plt.savefig(f"{pdfname}_energy.pdf")
    if(doshw): plt.show()
    
    
    # ### save config to pickle
    # data = {
    #     "initial_states": initial_states,
    #     "particle_trajectories": particle_trajectories,
    #     "particle_collisions": particle_collisions,
    #     "dipole": dipole,
    #     "quad0": quad0,
    #     "quad1": quad1,
    #     "quad2": quad2,
    #     "beampipe": beampipe
    # }
    # for i,detector in enumerate(detectors): data.update( {f"ALPIDE_{i}":detector} )
    # fpklname = "generator.pkl"
    # fpkl = open(fpklname,'wb')
    # pickle.dump(data,fpkl,protocol=pickle.HIGHEST_PROTOCOL) ### dump to pickle
    # fpkl.close()

#!/usr/bin/python
import os
import math
import subprocess
import time
import array
import numpy as np
import ROOT
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from scipy.optimize import curve_fit


from tracker_lib import config, objects, utils


def plot_event(run,start,duration,evt,fname,clusters,tracks,chi2threshold=1.,showtrkcls=False,showallcls=False,showwindow=True,showpipe=True,ismultiproc=False):
    cfg = config.Config().map
    # if(len(tracks)<1 and): return ###TODO
    
    ### turn interactive plotting off
    plt.ioff()
    matplotlib.use('Agg')
    ### define the plot
    # fig = plt.figure(figsize=(15,15),frameon=False,constrained_layout=True)
    fig = plt.figure(figsize=(15,15),frameon=False)
    plt.title(f"Run{run}, {start}, ~{duration}[h], Trig:{evt}", fontdict=None, loc='center', pad=None)
    plt.box(False)
    plt.axis('off')
    plt.subplots_adjust(wspace=0, hspace=-0.01)
    
    ## the views
    ax1 = fig.add_subplot(221, projection='3d', facecolor='none')
    ax2 = fig.add_subplot(222, projection='3d', facecolor='none')
    ax3 = fig.add_subplot(223, projection='3d', facecolor='none')
    ax4 = fig.add_subplot(224, projection='3d', facecolor='none')
    ax1.set_xlabel("x [mm]")
    ax1.set_ylabel("y [mm]")
    ax1.set_zlabel("z [mm]")
    ax2.set_xlabel("x [mm]")
    ax2.set_ylabel("y [mm]")
    ax2.set_zlabel("z [mm]")
    ax3.set_xlabel("x [mm]")
    ax3.set_ylabel("y [mm]")
    ax3.set_zlabel("z [mm]")
    ax4.set_xlabel("x [mm]")
    ax4.set_ylabel("y [mm]")
    ax4.set_zlabel("z [mm]")
    
    ### avoid ticks and lables for projections
    ax2.zaxis.set_label_position('none')
    ax2.zaxis.set_ticks_position('none')
    ax3.xaxis.set_label_position('none')
    ax3.xaxis.set_ticks_position('none')
    ax4.yaxis.set_label_position('none')
    ax4.yaxis.set_ticks_position('none')
        
    ### the chips
    L1verts = utils.getChips() ### TODO: add here the global offsets and rotations via apply_global_alignment_to_r if(not ismultiproc)
    ax1.add_collection3d(Poly3DCollection(L1verts, facecolors='green', linewidths=0.5, edgecolors='g', alpha=.20))
    ax2.add_collection3d(Poly3DCollection(L1verts, facecolors='green', linewidths=0.5, edgecolors='g', alpha=.20))
    ax3.add_collection3d(Poly3DCollection(L1verts, facecolors='green', linewidths=0.5, edgecolors='g', alpha=.20))
    ax4.add_collection3d(Poly3DCollection(L1verts, facecolors='green', linewidths=0.5, edgecolors='g', alpha=.20))
    ax1.set_box_aspect((1, 1, 1))
    ax2.set_box_aspect((1, 1, 1))
    ax3.set_box_aspect((1, 1, 1))
    ax4.set_box_aspect((1, 1, 1))
    
    ### the window
    if(showwindow):
        window = utils.getWindowRealSpace()
        ax1.add_collection3d(Poly3DCollection(window, facecolors='gray', linewidths=0.5, edgecolors='k', alpha=.20))
        ax2.add_collection3d(Poly3DCollection(window, facecolors='gray', linewidths=0.5, edgecolors='k', alpha=.20))
        ax3.add_collection3d(Poly3DCollection(window, facecolors='gray', linewidths=0.5, edgecolors='k', alpha=.20))
        ax4.add_collection3d(Poly3DCollection(window, facecolors='gray', linewidths=0.5, edgecolors='k', alpha=.20))
    
    ### print ALL clusters
    if(showallcls):
        clsx = []
        clsy = []
        clsz = []
        for det in cfg["detectors"]:
            # stave = cfg["det2stvchp"][det][0]
            for cluster in clusters[det]:
                ### TODO: add here the global offsets and rotations via apply_global_alignment_to_r if(not ismultiproc)
                clsx.append( cluster.xTmm )
                clsy.append( cluster.yTmm )
                clsz.append( cluster.zTmm )
        ax1.scatter(clsx,clsy,clsz,s=0.9,c='k',marker='o',alpha=0.3)
        ax2.scatter(clsx,clsy,clsz,s=0.9,c='k',marker='o',alpha=0.3)
        ax3.scatter(clsx,clsy,clsz,s=0.9,c='k',marker='o',alpha=0.3)
        ax4.scatter(clsx,clsy,clsz,s=0.9,c='k',marker='o',alpha=0.3)

    ### then the track
    goodtrk = 0
    trkcol = 'r'
    linewidth = 0.1
    for track in tracks:
        
        if(track.chi2ndof>chi2threshold): continue
        
        goodtrk += 1
        
        ### Plot the points and the fitted line
        # xFrst,yFrst,zFrst = utils.get_trak_at_det(cfg["det_frst"],track)
        # xLast,yLast,zLast = utils.get_trak_at_det(cfg["det_last"],track)
        r0,rN,rW,rF,rD = utils.get_track_point_at_extremes(track,ismultiproc=False)
        xFrst=r0[0]
        yFrst=r0[1]
        zFrst=r0[2]
        xLast=rN[0]
        yLast=rN[1]
        zLast=rN[2]
        
        ### TODO: add here the global offsets and rotations via apply_global_alignment_to_p if(not ismultiproc)
        ### TODO: to get the new params like:
        ### TODO: newdir,newcnt = utils.apply_global_alignment_to_p(track.direction,track.centroid,ismultiproc=False)
        ### TODO: newpars = utils.get_pars_from_centroid_and_direction(newcnt,newdir)
        
        xw,yw,zw = utils.line(cfg["world"]["z"][0], track.params) ### window
        xd,yd,zd = utils.line(cfg["world"]["z"][1]*0.55, track.params) ### dump direction
        
        if(yLast>=yFrst): ### consistent with positrons coming from the IP magnets
            trkcol = 'red'
            linewidth = 0.4
        else: ### otherwise
            trkcol = 'orange' 
            linewidth = 0.3
        
        # plot only the tracks clusters
        if(showtrkcls):
            ax1.scatter(clsx,clsy,clsz,s=0.92,c='r',marker='o')
            ax2.scatter(clsx,clsy,clsz,s=0.92,c='r',marker='o')
            ax3.scatter(clsx,clsy,clsz,s=0.92,c='r',marker='o')
            ax4.scatter(clsx,clsy,clsz,s=0.92,c='r',marker='o')

        ### plot the tracks lines in the detector volume only
        ax1.plot([xFrst, xLast], [yFrst, yLast], [zFrst, zLast], c=trkcol, linewidth=linewidth)
        ax2.plot([xFrst, xLast], [yFrst, yLast], [zFrst, zLast], c=trkcol, linewidth=linewidth)
        ax3.plot([xFrst, xLast], [yFrst, yLast], [zFrst, zLast], c=trkcol, linewidth=linewidth)
        ax4.plot([xFrst, xLast], [yFrst, yLast], [zFrst, zLast], c=trkcol, linewidth=linewidth)
        ### plot the extrapolated tracks lines to the window direction
        ax1.plot([xLast, xw], [yLast, yw], [zLast, zw], c=trkcol, linewidth=linewidth, linestyle='dashed')
        ax2.plot([xLast, xw], [yLast, yw], [zLast, zw], c=trkcol, linewidth=linewidth, linestyle='dashed')
        ax3.plot([xLast, xw], [yLast, yw], [zLast, zw], c=trkcol, linewidth=linewidth, linestyle='dashed')
        ax4.plot([xLast, xw], [yLast, yw], [zLast, zw], c=trkcol, linewidth=linewidth, linestyle='dashed')
        ### plot the extrapolated tracks lines to the dump direction
        ax1.plot([xd, xLast], [yd, yLast], [zd, zLast], c=trkcol, linewidth=linewidth, linestyle='dashed')
        ax2.plot([xd, xLast], [yd, yLast], [zd, zLast], c=trkcol, linewidth=linewidth, linestyle='dashed')
        ax3.plot([xd, xLast], [yd, yLast], [zd, zLast], c=trkcol, linewidth=linewidth, linestyle='dashed')
        ax4.plot([xd, xLast], [yd, yLast], [zd, zLast], c=trkcol, linewidth=linewidth, linestyle='dashed')
    
    ### add beampipe
    if(showpipe):
        us = np.linspace(0, 2.*np.pi, 100)
        zs = np.linspace(cfg["world"]["z"][0],cfg["world"]["z"][1], 100)
        us, zs = np.meshgrid(us,zs)
        Radius = cfg["Rpipe"]
        xs = Radius * np.cos(us)
        ys = Radius * np.sin(us)
        ys = ys-cfg["Rpipe"]+cfg["yWindowMin"]
        # yMidWindow = cfg["yWindowMin"]+cfg["yWindowHeight"]/2.
        # ys = ys + (yMidWindow-cfg["yMidWin2PipeCenter"])
        # ax1.plot_surface(xs, ys, zs, color='b',alpha=0.3)
        ax2.plot_surface(xs, ys, zs, color='b',alpha=0.3)
        ax3.plot_surface(xs, ys, zs, color='b',alpha=0.3)
        # ax4.plot_surface(xs, ys, zs, color='b',alpha=0.3)
    
    ## world limits
    ax1.set_xlim(cfg["world"]["x"])
    ax1.set_ylim(cfg["world"]["y"])
    ax1.set_zlim(cfg["world"]["z"])
    
    ax2.set_xlim(cfg["world"]["x"])
    ax2.set_ylim(cfg["world"]["y"])
    ax2.set_zlim(cfg["world"]["z"])
    
    ax3.set_xlim(cfg["world"]["x"])
    ax3.set_ylim(cfg["world"]["y"])
    ax3.set_zlim(cfg["world"]["z"])
    
    ax4.set_xlim(cfg["world"]["x"])
    ax4.set_ylim(cfg["world"]["y"])
    ax4.set_zlim(cfg["world"]["z"])
    
    # ### add some text to ax1
    # stracks = "tracks" if(goodtrk>1) else "track"
    # ax1.text(+15,-15,0,f"{goodtrk} {stracks}", fontsize=7)
    # for det in cfg["detectors"]:
    #     z = cfg["rdetectors"][det][2]
    #     n = len(clusters[det])
    #     ax1.text(-30,-20,z,f"{det}", fontsize=7)
    #     ax1.text(+15,+10,z,f"{n} clusters", fontsize=7)

    ### change view of the 2nd plot: 270 is xz view, 0 is yz view, and -90 is xy view
    ax1.elev = 40
    ax1.azim = 230
    ### x-y view:
    ax2.elev = 90
    ax2.azim = 270
    ### y-z view:
    ax3.elev = 0
    ax3.azim = 0
    ### x-z view:
    ax4.elev = 0
    ax4.azim = 270

    ### finish
    # plt.title(f"Run {run}, Start: {start}, Duration: ~{duration} [h], Event {evt}", fontdict=None, pad=None, x=0.1, y=0.6)
    plt.savefig(fname)
    plt.close(fig)
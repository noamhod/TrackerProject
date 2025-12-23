#!/usr/bin/python
import sys
import os
import os.path
import math
import time
import subprocess
import array
import numpy as np
import ROOT
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from scipy.optimize import curve_fit
# from skspatial.objects import Line, Sphere
# from skspatial.plotting import plot_3d

import argparse
parser = argparse.ArgumentParser(description='noise_analyzer.py...')
parser.add_argument('-conf', metavar='config file', required=True,  help='full path to config file')
argus = parser.parse_args()
configfile = argus.conf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tracker_lib import config
from tracker_lib import objects, Pixels, Clusters, hists, errors, counters, noise, utils


ROOT.gErrorIgnoreLevel = ROOT.kError

ROOT.gROOT.SetBatch(1)
ROOT.gStyle.SetOptFit(0)
# ROOT.gStyle.SetOptStat(0)

    
#####################################################################################
#####################################################################################
#####################################################################################


def GetTree(tfilename):
    cfg = config.Config().map
    tfile = ROOT.TFile(tfilename,"READ")
    ttree = None
    if(not cfg["isMC"]): ttree = tfile.Get("MyTree")
    else:
        if(cfg["isCVRroot"]): ttree = tfile.Get("Pixel")
        else:                 ttree = tfile.Get("tt")
    print("Events in tree:",ttree.GetEntries())
    if(cfg["nmax2process"]>0): print("Will process only",cfg["nmax2process"],"events")
    return tfile,ttree


def RunNoiseScan(tfilename,tfnoisename):
    cfg = config.Config().map
    tfilenoise = ROOT.TFile(tfnoisename,"RECREATE")
    tfilenoise.cd()
    h1D_noise       = {}
    h2D_noise       = {}
    for det in cfg["detectors"]:
        h1D_noise.update( { det:ROOT.TH1D("h_noisescan_pix_occ_1D_"+det,";Pixel;Hits",cfg["npix_x"]*cfg["npix_y"],1,cfg["npix_x"]*cfg["npix_y"]+1) } )
        h2D_noise.update( { det:ROOT.TH2D("h_noisescan_pix_occ_2D_"+det,";Pixel;Hits",cfg["npix_x"]+1,-0.5,cfg["npix_x"]+0.5, cfg["npix_y"]+1,-0.5,cfg["npix_y"]+0.5) } )

    ### get the tree
    tfile,ttree = GetTree(tfilename)
    
    nprocevents = 0
    for ievt,evt in enumerate(ttree):
        if(cfg["nmax2process"]>0 and nprocevents>cfg["nmax2process"]): break

        ### check for errors
        nerrors,errors = errors.check_errors(evt)
        if(nerrors>0):
            print(f"Skipping event {ievt} due to errors: {errors}")
            continue

        ### get the pixels
        n_active_tandem_layers, n_active_staves, n_active_chips, pixels = Pixels.get_all_pixles(evt,h2D_noise)
        
        for det in cfg["detectors"]:
            for pix in pixels[det]:
                i = h2D_noise[det].FindBin(pix.x,pix.y)
                h1D_noise[det].AddBinContent(i,1)
                h2D_noise[det].Fill(pix.x,pix.y)
        if(nprocevents%10000==0 and nprocevents>0): print("event:",nprocevents)
        nprocevents += 1
    ### finish
    tfilenoise.Write()
    tfilenoise.Close()
    print("Noise scan histos saved in:",tfnoisename)




#############################################################################
#############################################################################
#############################################################################
if __name__ == "__main__":
    ### get the start time
    st = time.time()
    
    ### see https://root.cern/manual/python
    print("---- start loading libs")
    if(os.uname()[1]=="wisett"):
        print("On DAQ PC (linux): must first add DetectorEvent lib:")
        print("export LD_LIBRARY_PATH=$HOME/work/eudaq/lib:$LD_LIBRARY_PATH")
        ROOT.gInterpreter.AddIncludePath('../eudaq/user/stave/module/inc/')
        ROOT.gInterpreter.AddIncludePath('../eudaq/user/stave/hardware/inc/')
        ROOT.gSystem.Load('libeudaq_det_event_dict.so')
    else:
        print("On mac: must first add DetectorEvent lib:")
        detevtlib = cfg["detevtlib"]
        print(f"export LD_LIBRARY_PATH=$PWD/DetectorEvent/{detevtlib}:$LD_LIBRARY_PATH")
        ROOT.gInterpreter.AddIncludePath(f"DetectorEvent/{detevtlib}/")
        ROOT.gSystem.Load('libtrk_event_dict.dylib')
    print("---- finish loading libs")
    
    
    #############################################
    ### Initialize Config in the main process ###
    config.init_config(configfile, False)
    cfg = config.Config().map
    # config.show_config() # print config once
    print(cfg) # print config once
    #############################################
    
    
    ### make directories, copy the input file to the new basedir and return the path to it
    tfilenamein = utils.make_run_dirs(cfg["inputfile"])
    # tfilenamein = cfg["inputfile"]
    
    ### noise
    tfnoisename = tfilenamein.replace(".root","_noise.root")
    isnoisefile = os.path.isfile(os.path.expanduser(tfnoisename))
    print("Running on:",tfilenamein)
    print("Noise run file exists?:",isnoisefile)
    if(isnoisefile):
        redonoise = input("Noise file exists - do you want to rederive it?[y/n]:")
        if(redonoise=="y" or redonoise=="Y"):
            RunNoiseScan(tfilenamein,tfnoisename)
            masked = noise.GetNoiseMask(tfnoisename)
        else:
            print("Option not understood - please try again.")
    else:
        RunNoiseScan(tfilenamein,tfnoisename)
        masked = noise.GetNoiseMask(tfnoisename)
    print("###################################")
    print("### FINISHED RUNNING NOISE SCAN ###")
    print("###################################")
    quit()
    
    ### get the end time and the execution time
    et = time.time()
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')

#!/usr/bin/python
import multiprocessing as mp
import time
import datetime
import sys
import os
import os.path
import math
import subprocess
import array
import numpy as np
import ROOT
### Stop ROOT from taking ownership of histograms automatically
ROOT.TH1.AddDirectory(ROOT.kFALSE)


from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from scipy.optimize import curve_fit
import pickle
import shelve

import argparse
parser = argparse.ArgumentParser(description='serial_analyzer.py...')
parser.add_argument('-conf', metavar='config file', required=True,  help='full path to config file')
parser.add_argument('-dbg',  metavar='debug with single proc?', required=False,  help='debug with single proc?[0/1]')
argus = parser.parse_args()
configfile = argus.conf
debug = True if(argus.dbg is not None and argus.dbg=="1") else False


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tracker_lib import config, objects, Pixels, Clusters, candidate, hough_seeder, selections, evtdisp, hists, errors, counters, noise, utils, svd_fit, chi2_fit


ROOT.gROOT.SetBatch(1)
ROOT.gStyle.SetOptFit(0)
# ROOT.gStyle.SetOptStat(0)

###############################################################
###############################################################
###############################################################

### defined below as global
allhistos = {}



def GetTree(tfilename):
    tfile = ROOT.TFile(tfilename,"READ")
    ttree = tfile.Get("MyTree")
    nevents = ttree.GetEntries()
    return tfile,ttree,nevents


def analyze(configfile,tfilenamein,irange,evt_range,masked,badtrigs):
    lock = mp.Lock()
    lock.acquire()
    
    
    ### initialize the Config singleton locally in the worker process
    cfg_obj = config.Config(configfile, doprint=False)
    cfg = cfg_obj.map
    
    
    ### important
    sufx = "_"+str(irange)
    
    
    ### important
    ispreproc = utils.is_preprocessed()
    
    
    ### the metadata:
    tfmeta = ROOT.TFile(tfilenamein,"READ")
    tmeta = tfmeta.Get("MyTreeMeta")
    runnumber = -1
    starttime = -1
    endtime   = -1
    duration  = -1
    if(tmeta is not None):
        try:
            nmeta = tmeta.GetEntries()
            tmeta.GetEntry(0)
            runnumber = tmeta.run_meta_data.run_number
            ts_start  = tmeta.run_meta_data.run_start
            starttime = utils.get_human_timestamp(ts_start)
            if(nmeta>1): tmeta.GetEntry(nmeta-1)
            ts_end    = tmeta.run_meta_data.run_end
            endtime   = utils.get_human_timestamp(ts_end)
            duration  = utils.get_run_length(ts_start,ts_end)
        except:
            print("Problem with Meta tree.")
            runnumber = utils.get_run_from_file(tfilenamein) #TODO: can also be taken from the event tree itself later
    meta = objects.Meta(runnumber,starttime,endtime,duration)
    # tfmeta.Close()
    if(cfg["dbg"]): print("Got the meta tree")
    
    
    ### open the pickle:
    if(not cfg["skiptracking"]):
        picklename = tfilenamein.replace(".root","_"+str(irange)+".pkl")
        fpickle = open(os.path.expanduser(picklename),"wb")
    if(cfg["dbg"]): print("Opened the pickle")

    
    
    ### histos
    histos = hists.book_histos()
    if(cfg["dbg"]): print("Done booking histos")
    
    
    ### get the tree
    tfile,ttree,neventsall = GetTree(tfilenamein)
    if(cfg["dbg"]): print("Got the TTree")
    
    
    ### needed below
    hPixMatix = hists.GetPixMatrix()
    if(cfg["dbg"]): print("Defined the pixel matrix")
    
    
    ### start the event loop
    ievt_start = evt_range[0]
    ievt_end   = evt_range[-1]
    eventslist = []
    for ievt in range(ievt_start,ievt_end+1):
        ### get the event
        ttree.GetEntry(ievt)
        if(cfg["dbg"]): print(f"Got entry {ievt} from tree")
        
        ### get the trigger number and time stamps
        trigger         = ttree.event.trg_n
        timestamp_begin = ttree.event.ts_begin
        timestamp_end   = ttree.event.ts_end
        
        ### get the magnets' state
        magnets = None
        if("E320" in cfg["inputfile"]):
            magnets = objects.Magnets(ttree.event.epics_frame.espec_dipole_bact,
                              ttree.event.epics_frame.espec_quad0_bact,
                              ttree.event.epics_frame.espec_quad1_bact,
                              ttree.event.epics_frame.espec_quad2_bact,
                              ttree.event.epics_frame.mcalc_m12,
                              ttree.event.epics_frame.mcalc_m34,
                              ttree.event.epics_frame.mcalc_z_obj,
                              ttree.event.epics_frame.mcalc_z_im,
                              ttree.event.epics_frame.espec_xcor_bact)
            if(cfg["dbg"]): print("Done setting magnets")
            

        ### append the envent no-matter-what:
        eventslist.append( objects.Event(meta,trigger,timestamp_begin,timestamp_end,magnets,saveprimitive=cfg["saveprimitive"]) )
        out_event_index = len(eventslist)-1
        if(cfg["dbg"]): print("Start appending event list")
        

        ### all events...
        histos["h_events"].Fill(0.5)
        histos["h_cutflow"].Fill( cfg["cuts"].index("All") )


        ### skip bad triggers...
        if((cfg["runtype"]=="beam" and cfg["checkbadtriggers"])):
            if(int(trigger) in badtrigs):
                print(f"Skipping bad trigger: {trigger}")
                continue
        histos["h_cutflow"].Fill( cfg["cuts"].index("BeamQC") )
        if(cfg["dbg"]): print("Passed good trigger")

        
        ### check event errors
        if(not cfg["isMC"] and not ispreproc):
            nerrs,errs = errors.check_errors(ttree)
            eventslist[out_event_index].set_event_errors(errs)
            if(nerrs>0):
                wgt = 1./float(len(cfg["detectors"]))
                for det in cfg["detectors"]:
                    for err in errs[det]:
                        b = errs.ERRORS.index(err)+1
                        histos["h_errors"].AddBinContent(b,wgt)
                        histos["h_errors_"+det].AddBinContent(b)
                continue
        histos["h_cutflow"].Fill( cfg["cuts"].index("0Err") )
        if(cfg["dbg"]): print("Passed DAQ errors")


        ### get the pixels
        n_active_tandem_layers, n_active_staves, n_active_chips, pixels = Pixels.get_all_pixels(ttree,hPixMatix,pix_matrix_max_frac=1)
        npixperdet = 0
        histos["h_nStaves"].Fill(n_active_staves)
        histos["h_nDetectors"].Fill(n_active_chips)
        histos["h_nTandemLayers"].Fill(n_active_tandem_layers)
        sprnt = f"ievt={ievt}:"
        for det in cfg["detectors"]:
            npixperdet += len(pixels[det])/len(cfg["detectors"])
            sprnt += f" Npixels[{det}]={len(pixels[det])},"
            hists.fillPixOcc(det,pixels[det],masked[det],histos) ### fill pixel occupancy
        if(cfg["dbg"]): print(sprnt)
        

        ### if skip clustering?
        if(cfg["skipclustering"]): continue
        
        
        if(cfg["dbg"] and n_active_tandem_layers>3):
            print(f"in tandem")
            chips = ""
            for det in cfg["detectors"]:
                for pix in pixels[det]:
                    i = histos["h_pix_occ_2D_"+det].FindBin(pix.x,pix.y)
                    if(i not in masked):
                        if(det not in chips):
                            chips += f" {det}"
            print(f"trigger={trigger}:  n_active_tandem_layers={n_active_tandem_layers}, n_active_chips={n_active_chips}: chips={chips}")
        
        
        ### non-empty events
        if(n_active_chips==0): continue  ### CUT!!!
        histos["h_cutflow"].Fill( cfg["cuts"].index("Non-empty") )
        
        
        ### all layers are active
        if(n_active_tandem_layers<cfg["layers"]): continue  ### CUT!!!
        histos["h_cutflow"].Fill( cfg["cuts"].index("N_{hits/lyr}>0") )
        
        
        ### spatial ROI cut
        # ROI = { "ix":{"min":cfg["cut_ROI_xmin"],"max":cfg["cut_ROI_xmax"]}, "iy":{"min":cfg["cut_ROI_ymin"],"max":cfg["cut_ROI_ymax"]} }
        # n_active_tandem_layers, n_active_staves, n_active_chips, pixels = Pixels.get_all_pixels(ttree,hPixMatix,ROI)
        # sprnt = f"ievt={ievt} in_ROI_chips={n_active_chips} -->"
        # for det in cfg["detectors"]:
        #     sprnt += f" Npixels[{det}]={len(pixels[det])},"
        # if(cfg["doprintout"]): print(sprnt)
        # if(n_active_tandem_layers<cfg["layers"]): continue  ### CUT!!!
        histos["h_cutflow"].Fill( cfg["cuts"].index("N_{hits/lyr}^{ROI}>0") )
        
        
        ### get the non-noisy pixels but this will get emptied during clustering so also keep a duplicate
        pixels_save = {}
        for det in cfg["detectors"]:
            if(cfg["skipmasking"]):
                pixels_save.update({det:pixels[det].copy()})
            else:
                pixels_wo_noise = noise.getGoodPixels(det,pixels[det],masked[det],hPixMatix[det])
                pixels_save.update({det:pixels_wo_noise.copy()})
        eventslist[out_event_index].set_event_pixels(pixels_save)
        

        ### run clustering
        clusters = {}
        nclusters = 0
        nclsperdet = 0
        sprnt = f"ievt={ievt}:"
        for det in cfg["detectors"]:
            det_clusters = Clusters.BFS_GetAllClusters(pixels[det],det)
            clusters.update( {det:det_clusters} )
            hists.fillClsHists(det,clusters[det],masked[det],histos)
            if(len(det_clusters)>0): nclusters += 1
            nclsperdet += len(det_clusters)/len(cfg["detectors"])
            sprnt += f" Nclusters[{det}]={len(det_clusters)},"
        if(cfg["dbg"]): print(sprnt)
        eventslist[out_event_index].set_event_clusters(clusters)
        n_active_tandem_layers = utils.count_active_tandem_layers(clusters)
        ### at least one cluster per layer
        if(n_active_tandem_layers<cfg["layers"]): continue ### CUT!!!
        histos["h_cutflow"].Fill( cfg["cuts"].index("N_{cls/lyr}>0") )
        
        


        #####################################
        if(cfg["skiptracking"]): continue ###
        #####################################
        

        ### run the seeding
        seeder = hough_seeder.HoughSeeder(clusters,ievt)
        ###################
        
        if(cfg["dbg"]): print(f"after seeder with seeder.nseeds={seeder.nseeds}")
        
        nTunnels = seeder.ntunnels
        eventslist[out_event_index].set_ntunnels(nTunnels)
        # nClsPerTnl = seeder.nclspertunnel
        nSeeds = seeder.nseeds

        histos["h_nTunnels"].Fill(nTunnels)
        histos["h_nTunnels_log"].Fill(nTunnels if(nTunnels>0) else 0.11)
        histos["h_nTunnels_full"].Fill(nTunnels)
        histos["h_nTunnels_mid"].Fill(nTunnels)
        histos["h_nTunnels_zoom"].Fill(nTunnels)
        histos["h_nSeeds"].Fill(nSeeds)
        histos["h_nSeeds_log"].Fill(nSeeds if(nSeeds>0) else 0.11)
        histos["h_nSeeds_full"].Fill(nSeeds)
        histos["h_nSeeds_mid"].Fill(nSeeds)
        histos["h_nSeeds_zoom"].Fill(nSeeds)
        if(nSeeds<1):
            del seeder
            continue ### CUT!!!
        histos["h_cutflow"].Fill( cfg["cuts"].index("N_{seeds}>0") )
    
        
        if(cfg["dbg"]): print(f"nSeeds={nSeeds}")
        
        
        ### prepare the clusters for the fit
        seeds = []
        hough_space = None
        if(nSeeds>0):
            hough_space = seeder.hough_space
            for iseed,seed in enumerate(seeder.seeds):
                if(cfg["dbg"]): print(f"iseed={iseed}")
                tunnelid     = seeder.tnlid[iseed]
                hough_coords = seeder.coord[iseed] ### the Hough-Transform coordinates
                if(cfg["dbg"]): print(f"before TrackSeed")
                trkseed      = objects.TrackSeed(seed,tunnelid,hough_coords,clusters)
                if(cfg["dbg"]): print(f"after TrackSeed")
                seeds.append(trkseed)
            if(cfg["dbg"]): print(f"after filling seeds")
            del seeder
            if(cfg["dbg"]): print(f"after deleting seeder")
        eventslist[out_event_index].set_event_seeds(seeds,hough_space)
        if(cfg["dbg"]): print(f"after set_event_seeds")


        ### get the event tracks
        vtx  = [cfg["xVtx"],cfg["yVtx"],cfg["zVtx"]]    if(cfg["doVtx"]) else []
        evtx = [cfg["exVtx"],cfg["eyVtx"],cfg["ezVtx"]] if(cfg["doVtx"]) else []
        

        ### loop over all seeds:
        tracks = []
        n_tracks            = 0
        n_successful_tracks = 0
        n_goodchi2_tracks   = 0
        n_selected_tracks   = 0
        n_btterfly_tracks   = 0
        for iseed,seed in enumerate(seeds):

            if(cfg["dbg"]): print(f"starting iseed={iseed}")
            
            ### get the points
            xclserr = seed.xsize if(cfg["use_large_clserr_for_algnmnt"]) else seed.dx
            yclserr = seed.ysize if(cfg["use_large_clserr_for_algnmnt"]) else seed.dy
            
            
            ### do the fit
            chisq       = None
            ndof        = None
            direction   = None
            centroid    = None
            params      = None
            paramerr    = None   
            paramcov    = None   
            success     = None
            par_guess   = None
            cand_points = None
            cand_errors = None
            if(cfg["dbg"]): print(f"starting track fitting")
            ### svd fit
            if(cfg["fit_method"]=="SVD"):
                points,pnterrs  = candidate.SVD_candidate(seed.detectors,seed.x,seed.y,seed.z,xclserr,yclserr,vtx,evtx)
                chisq,ndof,direction,centroid = svd_fit.fit_3d_SVD(points,pnterrs)
                params = utils.get_pars_from_centroid_and_direction(centroid,direction)
                success = True
                cand_points = points
                cand_errors = pnterrs
                if(cfg["dbg"]): print(f"SVD: success,chisq,ndof,direction,centroid={success,chisq,ndof,direction,centroid}")
            ### chi2 fit
            if(cfg["fit_method"]=="CHI2"):
                points,pnterrs = candidate.Chi2_candidate(seed.detectors,seed.x,seed.y,seed.z,xclserr,yclserr,vtx,evtx)
                chisq,ndof,direction,centroid,params,paramerr,paramcov,success = chi2_fit.fit_3d_chi2err(points,pnterrs,par_guess)
                cand_points,cand_errors = candidate.Candidate_Chi2toSVD(points,pnterrs)
                # print(f"params={params} --> paramerr={paramerr} --> paramcov={paramcov}")
                if(cfg["dbg"]): print(f"CHI2: success,chisq,ndof,direction,centroid={success,chisq,ndof,direction,centroid}")
            ### prepae the track clusters
            trkcls = {}
            for det,icls in seed.clsids.items(): trkcls.update({det:clusters[det][icls]})
            ### set the track
            track = objects.Track(seed.detectors,trkcls,cand_points,cand_errors,chisq,ndof,direction,centroid,params,paramerr,paramcov,success,seed.hough_coords)
            tracks.append(track)
            n_tracks += 1
        
            
            if(cfg["dbg"]): print(f"after track fitting")
            
            ### require good chi2, pointing to the pdc window, inclined up as a positron
            chi2ndof = chisq/ndof if(ndof>0) else 99999
            pass_fit       = (success and chi2ndof<=cfg["cut_chi2dof"])
            pass_selection = (pass_fit and selections.pass_geoacc_selection(track,ismultiproc=True))
            pass_butterfly = (pass_selection and selections.tilted_butterfly_RoI_cut(track)) if(cfg["cut_RoI_btrfly"]) else pass_selection
            if(success):        n_successful_tracks += 1
            if(pass_fit):       n_goodchi2_tracks   += 1
            if(pass_selection): n_selected_tracks   += 1
            if(pass_butterfly): n_btterfly_tracks   += 1

            if(cfg["dbg"]): print(f"after track calculation")

            histos["h_3Dchi2err"].Fill(chi2ndof)
            histos["h_3Dchi2err_all"].Fill(chi2ndof)
            histos["h_3Dchi2err_full"].Fill(chi2ndof)
            histos["h_3Dchi2err_zoom"].Fill(chi2ndof)
            histos["h_3Dchi2err_0to1"].Fill(chi2ndof)
            histos["h_Chi2_phi"].Fill(track.phi)
            histos["h_Chi2_theta"].Fill(track.theta)
            if(abs(np.sin(track.theta))>1e-10): histos["h_Chi2_theta_weighted"].Fill( track.theta,abs(1/(2*np.pi*np.sin(track.theta))) )
            
            if(cfg["dbg"]): print(f"after fill hists")
            
            ### Chi2 track to cluster residuals
            hists.fill_trk2cls_residuals(seed.detectors,cand_points,direction,centroid,chi2ndof,"h_Chi2fit_res_trk2cls",histos)
            hists.fill_trk2cls_residuals(seed.detectors,cand_points,direction,centroid,chi2ndof,"h_Chi2fit_res_trk2cls_pass",histos,chi2threshold=cfg["cut_chi2dof"])
            if(cfg["dbg"]): print(f"after track fill_trk2cls_residuals")
            ### response (residuals over cluster error)
            nxs = []
            nys = []
            for det,clstr in trkcls.items():
                nxs.append(clstr.nx)
                nys.append(clstr.ny)
            if(cfg["dbg"]): print(f"after trkcls")
            hists.fill_trk2cls_response(seed.detectors,cand_points,cand_errors,direction,centroid,nxs,nys,chi2ndof,"h_response",histos,chi2threshold=cfg["cut_chi2dof"])
            if(cfg["dbg"]): print(f"after track fill_trk2cls_response")
            ### fit points occupancy
            if(pass_selection): hists.fillFitOcc(seed.detectors,params,"h_trk_occ_2D", "h_trk_3D",histos)
            if(cfg["dbg"]): print(f"after track fillFitOcc")
            ### track to vertex residuals
            if(cfg["doVtx"]): hists.fill_trk2vtx_residuals(seed.detectors,vtx,direction,centroid,"h_Chi2fit_res_trk2vtx",histos)
            if(cfg["dbg"]): print(f"after track fill_trk2vtx_residuals")
        
        eventslist[out_event_index].set_event_tracks(tracks)
        
        histos["h_nTracks"].Fill( n_tracks )
        histos["h_nTracks_log"].Fill( n_tracks if(n_tracks>0) else 0.11 )
        histos["h_nTracks_mid"].Fill( n_tracks )
        histos["h_nTracks_full"].Fill( n_tracks )
        histos["h_nTracks_zoom"].Fill( n_tracks )
        histos["h_nTracks_success"].Fill( n_successful_tracks )
        histos["h_nTracks_success_log"].Fill( n_successful_tracks if(n_successful_tracks>0) else 0.11 )
        histos["h_nTracks_success_full"].Fill( n_successful_tracks )
        histos["h_nTracks_success_mid"].Fill( n_successful_tracks )
        histos["h_nTracks_success_zoom"].Fill( n_successful_tracks )
        histos["h_nTracks_goodchi2"].Fill( n_goodchi2_tracks )
        histos["h_nTracks_goodchi2_log"].Fill( n_goodchi2_tracks if(n_goodchi2_tracks>0) else 0.11 )
        histos["h_nTracks_goodchi2_full"].Fill( n_goodchi2_tracks )
        histos["h_nTracks_goodchi2_mid"].Fill( n_goodchi2_tracks )
        histos["h_nTracks_goodchi2_zoom"].Fill( n_goodchi2_tracks )
        histos["h_nTracks_selected"].Fill( n_selected_tracks )
        histos["h_nTracks_selected_log"].Fill( n_selected_tracks if(n_selected_tracks>0) else 0.11 )
        histos["h_nTracks_selected_full"].Fill( n_selected_tracks )
        histos["h_nTracks_selected_mid"].Fill( n_selected_tracks )
        histos["h_nTracks_selected_zoom"].Fill( n_selected_tracks )
        histos["h_nTracks_butterfly"].Fill( n_btterfly_tracks )
        histos["h_nTracks_butterfly_log"].Fill( n_btterfly_tracks if(n_btterfly_tracks>0) else 0.11 )
        histos["h_nTracks_butterfly_full"].Fill( n_btterfly_tracks )
        histos["h_nTracks_butterfly_mid"].Fill( n_btterfly_tracks )
        histos["h_nTracks_butterfly_zoom"].Fill( n_btterfly_tracks )
        
        print(f"Eventid={ievt}: Pix/det={int(npixperdet)}, Cls/det={int(nclsperdet)} -->  Tunnels={nTunnels}, Seeds={nSeeds}, AllTracks={n_tracks}, Success={n_successful_tracks}, GoodChi2={n_goodchi2_tracks}, Selected={n_selected_tracks}, Butterfly={n_btterfly_tracks}")
        
        if(n_successful_tracks<1): continue
        histos["h_cutflow"].Fill( cfg["cuts"].index("Fitted") )
        
        # #############
        # ### test ####
        # for itrk,track in enumerate(tracks):
        #     print(f"eventid={ievt}: trigger={trigger}")
        #     print(f"   track[{itrk}]:")
        #     for det in track.detectors:
        #         print(f"      {det}: clsid={track.trkcls[det].CID}: x={track.trkcls[det].xTnoGmm:.3f}, y={track.trkcls[det].yTnoGmm:.3f}, z={track.trkcls[det].zTnoGmm:.3f}")
        # #############
        
        ### plot everything which is fitted but the function will only put the track line if it passes the chi2 cut
        if(cfg["plot_online_evtdisp"]):
            fevtdisplayname = tfilenamein.replace("tree_","event_displays/").replace(".root",f"_{trigger}.pdf")
            showcls = (cfg["runtype"]=="cosmics")
            # showcls = True ###TODO: remove
            showwin = (cfg["runtype"]!="cosmics")
            showpip = ("E320" in cfg["inputfile"])
            evtdisp.plot_event(runnumber,starttime,duration,trigger,fevtdisplayname,clusters,tracks,chi2threshold=cfg["cut_chi2dof"],showtrkcls=showcls,showallcls=showcls,showwindow=showwin,showpipe=showpip,ismultiproc=True)
        
        if(n_goodchi2_tracks<1): continue ### CUT!!!
        histos["h_cutflow"].Fill( cfg["cuts"].index("#chi^{2}/N_{DoF}#leqX") )
        
        if(n_selected_tracks<1): continue ### CUT!!!
        histos["h_cutflow"].Fill( cfg["cuts"].index("Geo") )
        
        if(n_btterfly_tracks<1): continue ### CUT!!!
        histos["h_cutflow"].Fill( cfg["cuts"].index("Btrfly") )
    
    ###########    
    ### end ###
    ###########
    
    ### dump event to file
    if(not cfg["skiptracking"]):
        pickle.dump(eventslist, fpickle, protocol=pickle.HIGHEST_PROTOCOL) ### dump to pickle
        fpickle.close()    
    
    tfile.Close()
    del tfile
    
    print(f"Worker {irange} is done!")
    lock.release()
    return histos


def collect_errors(error):
    ### https://superfastpython.com/multiprocessing-pool-error-callback-functions-in-python/
    print(f'Error: {error}', flush=True)

def collect_histos(histos):
    ### https://www.machinelearningplus.com/python/parallel-processing-python/
    global allhistos ### defined above!!!
    for name,hist in allhistos.items():
        hist.Add(histos[name])
        del histos[name]


if __name__ == "__main__":
    # get the start time
    st = time.time()
    
    #############################################
    ### Initialize Config in the main process ###
    config.init_config(configfile, False)
    cfg = config.Config().map
    config.show_config(cfg)
    #############################################
    
    
    # We only need the TFile for the FINAL write, not during the loop
    # We book the "master" histograms in memory first
    allhistos = hists.book_histos()
    
    
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
    
    
    ###############################################################
    ###############################################################
    ###############################################################
    ispreproc = utils.is_preprocessed()
    if(cfg["isMC"] or ispreproc):
        # print("Building the classes for MC")
        ### declare the data tree and its classes
        if(cfg["isMC"]): ROOT.gROOT.ProcessLine("struct pixel  { Int_t ix; Int_t iy; Float_t xOrig; Float_t yOrig; Float_t xFake; Float_t yFake; Float_t Azx; Float_t Bzx; Float_t Azy; Float_t Bzy; Float_t Vx; Float_t Vy; Float_t Vz; };" )
        else:            ROOT.gROOT.ProcessLine("struct pixel  { Int_t ix; Int_t iy; };" )
        ROOT.gROOT.ProcessLine("struct chip   { Int_t chip_id; std::vector<pixel> hits; };" )
        ROOT.gROOT.ProcessLine("struct stave  { Int_t stave_id; std::vector<chip> ch_ev_buffer; };" )
        ROOT.gROOT.ProcessLine("struct event  { Int_t trg_n; Double_t ts_begin; Double_t ts_end; std::vector<stave> st_ev_buffer; };" )
        ### declare the meta-data tree and its classes
        ROOT.gROOT.ProcessLine("struct run_meta_data  { Int_t run_number; Double_t run_start; Double_t run_end; };" )
    
    
    
    
    ### make directories, copy the input file to the new basedir and return the path to it
    tfilenamein = utils.make_run_dirs(cfg["inputfile"])
    fpkltrgname = tfilenamein.replace("tree_","beam_quality/tree_").replace(".root","_BadTriggers.pkl") if(not cfg["isMC"] and cfg["runtype"]=="beam") else None
    fpklcfgname = tfilenamein.replace("tree_","config_used/tree_").replace(".root","_config.pkl")
    
    
    ### save config to pickle
    fpklconfig = open(fpklcfgname,'wb')
    pickle.dump(cfg,fpklconfig,protocol=pickle.HIGHEST_PROTOCOL) ### dump to pickle
    fpklconfig.close()


    ### load bad triggers from pickle
    badtriggers = []
    if(cfg["runtype"]=="beam" and cfg["checkbadtriggers"]):
        fpkltrigger = open(fpkltrgname,'rb')
        badtriggers = pickle.load(fpkltrigger)
        fpkltrigger.close()
        print("\n----------------------------------")
        print(f"Found {len(badtriggers)} bad triggers")
        print("-----------------------------------\n")
    else:
        print("\n----------------------------")
        print(f"Not removing any triggers!")
        print("----------------------------\n")
    
    
    ### masking business
    masked = {}
    if(cfg["skipmasking"]):
        print("\n----------------------------")
        print("Skipping/ignoring noise mask")
        print("----------------------------\n")
        masked = noise.GetNoiseMaskEmpty()
    else:
        tfnoisename = tfilenamein.replace(".root","_noise.root")
        masked = noise.GetNoiseMask(tfnoisename)
        
    
    ### meta data:
    tfmeta = ROOT.TFile(tfilenamein,"READ")
    tmeta = tfmeta.Get("MyTreeMeta")
    if(tmeta is not None):
        try:
            nmeta = tmeta.GetEntries()
            tmeta.GetEntry(0)
            ts_starttime = tmeta.run_meta_data.run_start
            print( f"\nRun start:  {utils.get_human_timestamp(ts_starttime)}" )
            if(nmeta>1): tmeta.GetEntry(nmeta-1)
            ts_endtime = tmeta.run_meta_data.run_end
            print( f"Run end:    {utils.get_human_timestamp(ts_endtime)}" )
            print( f"Run duration [h]: {utils.get_run_length(ts_starttime,ts_endtime)}" )
        except:
            print("Problem with Meta tree, continuing without it.")
    
    
    #############################
    ### configure the workers ###
    #############################
    nCPUs = mp.cpu_count()
    print(f'nCPUs available: {nCPUs}')
    print(f'nCPUs configured: {cfg["nCPU"]}')
    if(cfg["nCPU"]<1):
        print("nCPU config cannot be <1, quitting")
        quit()
    elif(cfg["nCPU"]>=1 and cfg["nCPU"]<=nCPUs):
        nCPUs = cfg["nCPU"]
    else:
        print(f"nCPU config cannot be greater than {nCPUs}, quitting")
        quit()
    ### Ensure all workers are started cleanly, avoiding the fork/CoW issues.
    mp.set_start_method('spawn', force=True)
    ### Create a pool of workers    
    # pool = mp.Pool(nCPUs)
    pool = mp.Pool(nCPUs, maxtasksperchild=1)
    
    
    ####################################
    ### configure the processes loop ###
    ####################################
    print(f"\nStarting the loop with tree file {tfilenamein}:")
    tfile0,ttree0,nevents0 = GetTree(tfilenamein)
    firstevent  = cfg["first2process"]
    max2process = cfg["nmax2process"]
    print(f"Events in tree: {nevents0}, Starting in event: {firstevent}, Processing maximum {max2process} events")
    nevents = nevents0-firstevent
    if(max2process>0 and max2process<=nevents):
        nevents = max2process
        print(f"Going to analyze only {nevents} events out of the {nevents0} available in the tree")
    else:
        print(f'config nmax2process={max2process} --> will analyze all events in the tree:{nevents}')
    bundle = nCPUs
    fullrange = range(firstevent,firstevent+nevents)
    print(fullrange)
    ranges = np.array_split(fullrange,bundle) if(nevents>=bundle) else [range(firstevent,firstevent+nevents)]
    for irng,rng in enumerate(ranges): print(f"Range[{irng}]: {rng[0]},...,{rng[-1]}")
    
    
    ############################
    ### submit the processes ###
    ############################
    for irng,rng in enumerate(ranges):
        print(f"Submitting range[{irng}]: {rng[0]},...,{rng[-1]}")
        if(debug):
            histos = analyze(tfilenamein,irng,rng,masked,badtriggers)
        else:
            pool.apply_async(analyze, args=(configfile,tfilenamein,irng,rng,masked,badtriggers), callback=collect_histos, error_callback=collect_errors)
    ### Wait for all the workers to finish
    pool.close()
    pool.join()


    #######################
    ### post processing ###
    #######################
    ### cluster mean size vs position
    hname = "h_csize_vs_trupos"
    hnewname = hname.replace("csize","mean")
    hdenname = hname.replace("csize","ntrks")
    allhistos.update( {hnewname:allhistos[hname].Clone(hnewname)} )
    allhistos[hnewname].Divide(allhistos[hdenname])
    for det in cfg["detectors"]:
        hname = "h_csize_vs_trupos_"+det
        hnewname = hname.replace("csize","mean")
        hdenname = hname.replace("csize","ntrks")
        allhistos.update( {hnewname:allhistos[hname].Clone(hnewname)} )
        allhistos[hnewname].Divide(allhistos[hdenname])
    for j in range(1,6):
        strcsize = str(j) if(j<5) else "n"
        hname = "h_csize_"+strcsize+"_vs_trupos"
        hnewname = hname.replace("csize","mean")
        hdenname = hname.replace("csize","ntrks")
        allhistos.update( {hnewname:allhistos[hname].Clone(hnewname)} )
        allhistos[hnewname].Divide(allhistos[hdenname])
        for det in cfg["detectors"]:
            hname = "h_csize_"+strcsize+"_vs_trupos_"+det
            hnewname = hname.replace("csize","mean")
            hdenname = hname.replace("csize","ntrks")
            allhistos.update( {hnewname:allhistos[hname].Clone(hnewname)} )
            allhistos[hnewname].Divide(allhistos[hdenname])
    
    
    #########################
    ### the output histos ###
    #########################
    tfilenameout = tfilenamein.replace(".root",f'{cfg["hfilesufx"]}.root')
    tfo = ROOT.TFile(tfilenameout,"RECREATE")
    tfo.cd()
    for det in cfg["detectors"]:
        tfo.mkdir(det)
        tfo.cd()
    tfo.cd()
    for name,h in allhistos.items():
        isdet  = False
        detdir = ""
        for det in cfg["detectors"]:
            if(det in name):
                isdet  = True
                detdir = det
        if(isdet):
            tfo.cd(detdir)
            h.Write()
            tfo.cd()
        else:
            tfo.cd()
            h.Write()
    tfo.Close()
    
    
    # get the end time
    et = time.time()
    # get the execution time
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')

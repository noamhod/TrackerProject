#!/usr/bin/python
import os
import math
import array
import numpy as np
import ROOT


from tracker_lib import config, utils
from tracker_lib.objects import Hit


# ispreproc = ("preprocessed" in cfg["inputfile"])
# if(cfg["isMC"] or ispreproc):
#     # print("Building the classes for MC")
#     ### declare the data tree and its classes
#     if(cfg["isMC"]): ROOT.gROOT.ProcessLine("struct pixel  { Int_t ix; Int_t iy; Float_t xOrig; Float_t yOrig; Float_t xFake; Float_t yFake; Float_t Azx; Float_t Bzx; Float_t Azy; Float_t Bzy; Float_t Vx; Float_t Vy; Float_t Vz; };" )
#     else:            ROOT.gROOT.ProcessLine("struct pixel  { Int_t ix; Int_t iy; };" )
#     ROOT.gROOT.ProcessLine("struct chip   { Int_t chip_id; std::vector<pixel> hits; };" )
#     ROOT.gROOT.ProcessLine("struct stave  { Int_t stave_id; std::vector<chip> ch_ev_buffer; };" )
#     ROOT.gROOT.ProcessLine("struct event  { Int_t trg_n; Double_t ts_begin; Double_t ts_end; std::vector<stave> st_ev_buffer; };" )


def get_all_pixels(evt,hPixMatrix,ROI={}):
    cfg = config.Config().map

    ispreproc = utils.is_preprocessed()


    pixels = {det: [] for det in cfg["detectors"]}
    raws   = {det: [] for det in cfg["detectors"]}
    ids2d  = {det: [] for det in cfg["detectors"]}

    n_active_staves = 0
    n_active_chips  = 0

    staves = evt.event.st_ev_buffer

    for istv in range(staves.size()):
        staveid  = staves[istv].stave_id
        chips    = staves[istv].ch_ev_buffer
        isactivestave = True
        for ichp in range(chips.size()):
            chipid = chips[ichp].chip_id
            stvchp = (staveid,chipid)
            if(stvchp not in cfg["stvchps"]): continue
            detector = cfg["stvchp2det"][stvchp]
            nhits    = chips[ichp].hits.size()
            if(nhits>0.05*cfg["npix_x"]*cfg["npix_y"]):
                print(f"Event {evt.event.trg_n} has too many pixels ({nhits}) in {det} --> skipping")
                continue
            if(nhits<1 and cfg["iszerosuppressed"]):
                print(f"Problem with zero-suppression in stvchp={stvchp} --> detector={detector}, nhits={nhits}")
            n_active_chips += (nhits>0)
            for ipix in range(nhits):
                ix = -1
                iy = -1
                if(not cfg["isMC"] and not ispreproc):
                    ### EUDAQ
                    ix,iy = chips[ichp].hits[ipix]
                if(ispreproc):
                    ### preprocessed EUDAQ
                    ix = chips[ichp].hits[ipix].ix
                    iy = chips[ichp].hits[ipix].iy
                if(cfg["isMC"] and not cfg["isFakeMC"]):
                    ### AllPix converted to EUDAQ
                    ix = chips[ichp].hits[ipix].ix
                    iy = chips[ichp].hits[ipix].iy

                if(len(ROI)>0):
                    if(("ix" in ROI) and (ix<ROI["ix"]["min"] or ix>ROI["ix"]["max"])): continue
                    if(("iy" in ROI) and (iy<ROI["iy"]["min"] or iy>ROI["iy"]["max"])): continue
                raw = hPixMatrix[detector].FindBin(ix,iy)
                id2d = (ix,iy)
                if(id2d not in ids2d[detector]):
                    ids2d[detector].append(id2d)
                    raws[detector].append(raw)
                    pixels[detector].append( Hit(detector,ix,iy,raw) if(not cfg["isFakeMC"]) else Hit(detector,ix,iy,raw,xOrig,yOrig,xFake,yFake,Azx,Bzx,Azy,Bzy,Vx,Vy,Vz) )
            n_active_staves += (isactivestave)

        tdm_counter = np.zeros(cfg["layers"],dtype=int)
        for det in cfg["detectors"]:
            tdm = cfg["det2tdm"][det]
            if(tdm_counter[tdm]>0): continue
            if(len(pixels[det])>0): tdm_counter[tdm] = int(1)
        n_active_tandem_layers = 0
        for n in tdm_counter: n_active_tandem_layers += n

    return n_active_tandem_layers,n_active_staves,n_active_chips,pixels






# def get_all_pixels(evt, hPixMatrix, ROI={}):
#     ### Optimized pixel reader using NumPy vectorization to minimize Python-C++ overhead.
#
#     # 1. Access the singleton configuration locally
#     cfg = config.Config().map
#
#     # 2. Identify data source characteristics
#     ispreproc = utils.is_preprocessed()
#     is_mc = cfg.get("isMC", False)
#     is_fake_mc = cfg.get("isFakeMC", False)
#
#     # 3. Initialize containers
#     pixels = {det: [] for det in cfg["detectors"]}
#     ids2d = {det: set() for det in cfg["detectors"]} # Use set for O(1) lookups
#     n_active_staves = 0
#     n_active_chips = 0
#
#     # 4. Access the main stave buffer
#     st_buffer = evt.event.st_ev_buffer
#
#     for stave in st_buffer:
#         staveid = stave.stave_id
#         is_active_stave = False
#
#         for chip in stave.ch_ev_buffer:
#             chipid = chip.chip_id
#             stvchp = (staveid, chipid)
#
#             # Skip chips not defined in current geometry
#             if(stvchp not in cfg["stvchps"]): continue
#
#             detector = cfg["stvchp2det"][stvchp]
#             nhits = chip.hits.size()
#
#             # Occupancy protection: skip noisy frames
#             if(nhits > 0.05 * cfg["npix_x"] * cfg["npix_y"]):
#                 print(f"Event {evt.event.trg_n} high occupancy ({nhits}) in {detector} -> skipping chip")
#                 continue
#
#             if(nhits>0):
#                 n_active_chips += 1
#                 is_active_stave = True
#
#                 # OPTIMIZATION: Convert C++ hit vector to NumPy array in one call
#                 # This is much faster than looping over 'ipix' in Python
#                 hits_array = np.array(chip.hits)
#
#                 for i in range(nhits):
#                     # Access hit data; format depends on C++ struct definition
#                     ix, iy = hits_array[i]
#
#                     # Apply Region of Interest (ROI) cuts
#                     if ROI:
#                         if "ix" in ROI and (ix < ROI["ix"]["min"] or ix > ROI["ix"]["max"]): continue
#                         if "iy" in ROI and (iy < ROI["iy"]["min"] or iy > ROI["iy"]["max"]): continue
#
#                     # Ensure uniqueness for this detector in this event
#                     if((ix, iy) not in ids2d[detector]):
#                         ids2d[detector].add((ix, iy))
#                         raw_bin = hPixMatrix[detector].FindBin(ix, iy)
#
#                         # Instantiate Hit with relevant metadata
#                         if not is_fake_mc:
#                             pixels[detector].append(Hit(detector, ix, iy, raw_bin))
#
#         if(is_active_stave):
#             n_active_staves += 1
#
#     # 5. Calculate active tandem layers
#     tdm_counter = np.zeros(cfg["layers"], dtype=int)
#     for det in cfg["detectors"]:
#         if(pixels[det]):
#             tdm = cfg["det2tdm"][det]
#             tdm_counter[tdm] = 1
#
#     return sum(tdm_counter), n_active_staves, n_active_chips, pixels
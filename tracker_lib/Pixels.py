#!/usr/bin/python
import os
import math
import array
import numpy as np
import ROOT


from tracker_lib import config, utils
from tracker_lib.objects import Hit


def get_all_pixels(evt,hPixMatrix,ROI={},pix_matrix_max_frac=1):
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
            if(nhits>pix_matrix_max_frac*cfg["npix_x"]*cfg["npix_y"]):
                print(f"Event {evt.event.trg_n} has too many pixels ({nhits}) in {detector} --> skipping")
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
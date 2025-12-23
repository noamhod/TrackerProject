#!/usr/bin/python
import os
import math
import array
import numpy as np
import ROOT

from tracker_lib import config


ERRORS = ["BUSY_VIOLATION", "INCOMPLETE", "STROBE_EXTENDED", "BUSY_TRANSITION",
          "END_OF_RUN", "OVERFLOW", "TIMEOUT", "HEADER_ERROR",
          "DECODER_10B8B_ERROR", "EVENT_OVERSIZE_ERROR"]

def check_errors(evt):
    cfg = config.Config().map
    errors  = {}
    nerrors = 0
    if(cfg["isMC"]): return nerrors,errors
    
    for det in cfg["detectors"]: errors.update({det:[]})
    staves = evt.event.st_ev_buffer
    for istv in range(staves.size()):
        staveid  = staves[istv].stave_id
        chips    = staves[istv].ch_ev_buffer
        for ichp in range(chips.size()):
            chipid   = chips[ichp].chip_id
            stvchp = (staveid,chipid)
            if(stvchp not in cfg["stvchps"]): continue
            detector = cfg["stvchp2det"][stvchp]
            if( chips[ichp].is_busy_violation ):     errors[detector].append("BUSY_VIOLATION")
            if( chips[ichp].is_flushed_incomplete ): errors[detector].append("INCOMPLETE")
            if( chips[ichp].is_strobe_extended ):    errors[detector].append("STROBE_EXTENDED")
            if( chips[ichp].is_busy_transition ):    errors[detector].append("BUSY_TRANSITION")
            if( chips[ichp].end_of_run ):            errors[detector].append("END_OF_RUN")
            if( chips[ichp].overflow ):              errors[detector].append("OVERFLOW")
            if( chips[ichp].timeout ):               errors[detector].append("TIMEOUT")
            if( chips[ichp].header_error ):          errors[detector].append("HEADER_ERROR")
            if( chips[ichp].decoder_10b8b_error ):   errors[detector].append("DECODER_10B8B_ERROR")
            if( chips[ichp].event_oversize_error ):  errors[detector].append("EVENT_OVERSIZE_ERROR")
            nerrors += (len(errors[detector])>0)
    return nerrors,errors
import time
import sys
import copy
import os

import numpy as np
from scipy import linalg
import matplotlib
from matplotlib.colors import LogNorm
from matplotlib import rcParams
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from matplotlib.backends.backend_pdf import PdfPages
import scipy.io
from   scipy import stats
import pickle
from scipy.ndimage import gaussian_filter, median_filter
import ROOT


import argparse
parser = argparse.ArgumentParser(description='....py...')
parser.add_argument('-set',  metavar='epics dataset name',  required=False,  help='epics dataset name')
parser.add_argument('-run',  metavar='alpide run number',   required=False,  help='alpide run number')
parser.add_argument('-pix',  metavar='read & plot pixels?', required=False,  help='read & plot pixels?')
argus = parser.parse_args()
epics_dataset = argus.set
alpide_run    = argus.run
dopix = True if(argus.pix is not None and argus.pix=="1") else False

match_key = f'{epics_dataset}-RUN_{alpide_run}'


### EPICS time offest
NYEARS   = 20
NDAYS    = 5
NHOURS   = 0
NMINUTES = 1.569509
NSECONDS = NYEARS*365*24*60*60 + NDAYS*24*60*60 + NHOURS*60*60 + NMINUTES*60
print(f"EPICS time offset [s]: {NSECONDS} for NYEARS={NYEARS}, NDAYS={NDAYS}, NHOURS={NHOURS} and NMINUTES={NMINUTES}")


def get_human_timestamp_epics(timestamp_s,fmt="%d/%m/%Y, %H:%M:%S"):
    timestamp_s += NSECONDS
    unix_timestamp = timestamp_s
    milliseconds = int((timestamp_s % 1) * 1000)
    human_timestamp = time.strftime(fmt,time.localtime(unix_timestamp))
    # return human_timestamp
    return f"{human_timestamp}.{milliseconds:03d}"

def get_human_timestamp_eudaq(timestamp_ns,fmt="%d/%m/%Y, %H:%M:%S"):
    unix_timestamp = timestamp_ns/1e9
    milliseconds = int((timestamp_ns % 1e9) / 1e6)
    human_timestamp = time.strftime(fmt,time.localtime(unix_timestamp))
    # return human_timestamp
    return f"{human_timestamp}.{milliseconds:03d}"


def eudaq_parity(trgid):
    return "even" if(trgid%2==0) else "odd"
def epics_parity(trgid):
    return "even" if(((trgid-10)/36)%2==0) else "odd"




### runs can be seen in: https://docs.google.com/spreadsheets/d/1Mux0J1XHzrhXqqxtls9xFwqnpD6LTtEOCWjSFn42w4E/edit?usp=sharing
epics_index_range = {
    "E320_13130-RUN_690":[0,-1],
    "E320_13132-RUN_691":[2004,3001],
    "E320_13133-RUN_692":[0,1000],
    "E320_13133-RUN_693":[4004,5000],
    "TEST_13134-RUN_694":[0,60],
    "E320_13139-RUN_696":[0,999],
    "E320_13158-RUN_701":[0,19],
    "E320_13165-RUN_702":[0,100],
    
}
eudaq_firstindex_map = {
    "E320_13130-RUN_690":11178,
    "E320_13132-RUN_691":18868,
    "E320_13133-RUN_692":9992,
    "E320_13133-RUN_693":650,
    "TEST_13134-RUN_694":1524,
    "E320_13139-RUN_696":27134,
    "E320_13158-RUN_701":4760,
    "E320_13165-RUN_702":3369,
}





### SET INITIAL PARAMETERS FOR ALL IMAGES IN THE DATA SET
main_path   = "../test_data/e320_prototype_beam_May2025_17-19/matlab"
matlab_file = f"{main_path}/{epics_dataset}.mat" 

mat_data    = scipy.io.loadmat(matlab_file, simplify_cells=True)
data_struct = mat_data["data_struct"]
mat_scalars = data_struct["scalars"]
scalar_cidx	= mat_scalars["common_index"]

pulseID		= data_struct["pulseID"]
SLAC_time	= pulseID["SLAC_time"]
scalar_PID	= pulseID["scalar_PID"]

### Lists of the matlab file
BSA_List_S20 = mat_scalars["BSA_List_S20"]

### PVs
TORO_2040 = BSA_List_S20['TORO_LI20_2040_TMIT']
TORO_2452 = BSA_List_S20['TORO_LI20_2452_TMIT']
TORO_3163 = BSA_List_S20['TORO_LI20_3163_TMIT']
TORO_3255 = BSA_List_S20['TORO_LI20_3255_TMIT']
PMT_3060  = BSA_List_S20['PMT_LI20_3060_QDCRAW']
PMT_3070  = BSA_List_S20['PMT_LI20_3070_QDCRAW']
PMT_3179  = BSA_List_S20['PMT_LI20_3179_QDCRAW']
PMT_3350  = BSA_List_S20['PMT_LI20_3350_QDCRAW']
PMT_3360  = BSA_List_S20['PMT_LI20_3360_QDCRAW']
BPMS_3156 = BSA_List_S20['BPMS_LI20_3156_TMIT']
BPMS_3218 = BSA_List_S20['BPMS_LI20_3218_TMIT']
BPMS_3265 = BSA_List_S20['BPMS_LI20_3265_TMIT']
BPMS_3315 = BSA_List_S20['BPMS_LI20_3315_TMIT']

data = {
    "SLAC_time":SLAC_time,
    "scalar_PID":scalar_PID,
    "TORO_2040":TORO_2040,
    "TORO_2452":TORO_2452,
    "TORO_3163":TORO_3163,
    "TORO_3255":TORO_3255,
    "PMT_3060" :PMT_3060,
    "PMT_3070" :PMT_3070,
    "PMT_3179" :PMT_3179,
    "PMT_3350" :PMT_3350,
    "PMT_3360" :PMT_3360,
    "BPMS_3156":BPMS_3156,
    "BPMS_3218":BPMS_3218,
    "BPMS_3265":BPMS_3265,
    "BPMS_3315":BPMS_3315,
}

print(f"epics_index_range for key={match_key}: {epics_index_range[match_key]}")
epics_firstindex = epics_index_range[match_key][0]
epics_lastindex  = epics_index_range[match_key][1] if(epics_index_range[match_key][1]!=-1) else len(data["SLAC_time"])-1
print(f"len(SLAC_time)={len(SLAC_time)} --> epics_lastindex={epics_lastindex}")



### trim to the range
newdata = {}
for name,arr in data.items():
    print(f"{name}: len(arr)={len(arr)}")
    arr = arr[epics_firstindex:epics_lastindex+1] ### +1 to include the last index!
    newdata.update({name:arr})
print(f'len(data["SLAC_time"])={len(data["SLAC_time"])}')
print(f'len(newdata["SLAC_time"])={len(newdata["SLAC_time"])}')






##################################################
##################################################
ALPIDES = {}
if(dopix):
    ROOT.gROOT.ProcessLine("struct pixel  { Int_t ix; Int_t iy; };" )
    ROOT.gROOT.ProcessLine("struct chip   { Int_t chip_id; std::vector<pixel> hits; };" )
    ROOT.gROOT.ProcessLine("struct stave  { Int_t stave_id; std::vector<chip> ch_ev_buffer; };" )
    ROOT.gROOT.ProcessLine("struct event  { Int_t trg_n; Double_t ts_begin; Double_t ts_end; std::vector<stave> st_ev_buffer; };" )
    
    detectors = ["ALPIDE_0","ALPIDE_1","ALPIDE_2","ALPIDE_3","ALPIDE_4"]
    planes    = [8,6,4,2,0]
    plane2det = {8:"ALPIDE_0", 6:"ALPIDE_1", 4:"ALPIDE_2", 2:"ALPIDE_3", 0:"ALPIDE_4"}
    
    tfile = ROOT.TFile(f"../test_data/e320_prototype_beam_May2025_17-19/runs/run_0000{alpide_run}/tree_Run{alpide_run}.root","READ")
    ttree = tfile.Get("MyTree")
    nevents = ttree.GetEntries()
    print(f"nevents={nevents}")
    doprint    = False
    pixelsloop = False
    
    ### holder of the pixels
    pixels = {}
    for det in detectors:
        pixels.update({det:np.zeros(nevents)})
    
    ### read the pixels
    trigger_number = np.zeros(nevents, dtype=np.uint64)
    trigger_timbeg = np.zeros(nevents)
    trigger_timend = np.zeros(nevents)
    for ievt in range(nevents):
        ### get the event
        ttree.GetEntry(ievt)
        ### get the metadata
        trigger_number[ievt] = ttree.event.trg_n
        trigger_timbeg[ievt] = ttree.event.ts_begin
        trigger_timend[ievt] = ttree.event.ts_end
        ### get the staves
        staves = ttree.event.st_ev_buffer
        ### get the chips
        for istv in range(staves.size()):
            staveid  = staves[istv].stave_id
            chips    = staves[istv].ch_ev_buffer
            for ichp in range(chips.size()):
                chipid = chips[ichp].chip_id
                if(chipid not in planes): continue
                detector = plane2det[chipid]
                nhits    = chips[ichp].hits.size()
                if(doprint): print(f"chipid: {chipid} det: {detector} --> npixels: {nhits}")
                pixels[detector][ievt] = nhits
                if(pixelsloop):
                    for ipix in range(nhits):
                        ix,iy = chips[ichp].hits[ipix]
                        if(doprint): print(f"   pixel[{ipix}]: ({ix},{iy})")
    
    eudaq_firstindex = eudaq_firstindex_map[match_key]
    # eudaq_lastindex  = eudaq_firstindex + (epics_lastindex-epics_firstindex)
    eudaq_lastindex  = eudaq_firstindex + len(newdata["SLAC_time"])

    print(f'epics_firstindex={epics_firstindex}, epics_lastindex={epics_lastindex} --> len={len(newdata["SLAC_time"])}')
    print(f"eudaq_firstindex={eudaq_firstindex}, eudaq_lastindex={eudaq_lastindex} --> len={eudaq_lastindex-eudaq_firstindex}")
    for det in detectors: ALPIDES.update( {det:pixels[det][eudaq_firstindex:eudaq_lastindex]} )
    data.update({"ALPIDES":ALPIDES})

    data.update({"ALPIDES_trg_number":trigger_number[eudaq_firstindex:eudaq_lastindex]})
    data.update({"ALPIDES_trg_timbeg":trigger_timbeg[eudaq_firstindex:eudaq_lastindex]})
    data.update({"ALPIDES_trg_timend":trigger_timend[eudaq_firstindex:eudaq_lastindex]})
##################################################
##################################################



### save everything in pickle file
fpklname = f"{match_key}.pkl"
fpkl = open(fpklname,"wb")
pickle.dump(data, fpkl, protocol=pickle.HIGHEST_PROTOCOL) ### dump to pickle
fpkl.close()





### plot
def plotter(name, pdf, arrx, arrsyL, arrsyR, xmin, xmax, xtitle, ytitleL="", ytitleR="", colL="blue", colR="red", logyL=True, logyR=True, lstyleL="-", lstyleR="-", points=[]):
    ### first page
    fig = plt.figure(figsize=(25, 5))
    ax1 = plt.subplot(111)
    ax2 = ax1.twinx()
    
    for i,arry in enumerate(arrsyL): ax1.plot(arrx, arry, linewidth=0.7, color=colL,alpha=1-i*0.15, linestyle=lstyleL)
    for i,arry in enumerate(arrsyR): ax2.plot(arrx, arry, linewidth=0.7, color=colR, alpha=0.8-i*0.15, linestyle=lstyleR)
    
    plt.locator_params(axis='x', nbins=10)
    ax1.xaxis.set_minor_locator(AutoMinorLocator(10))
    ax1.yaxis.set_minor_locator(AutoMinorLocator(10))
    ax1.grid(which='major', linestyle='-', linewidth='0.25', color='gray',alpha=0.5)
    ax1.grid(which='minor', linestyle=':', linewidth='0.25', color='gray',alpha=0.3)
    
    ax1.set_xlim(xmin,xmax)
    ax1.set_xlabel(xtitle)
    ax1.set_ylabel(ytitleL, color=colL)
    if(logyL): ax1.set_yscale("log")
    if(len(arrsyR)>0):
        ax2.set_xlim(xmin,xmax)
        ax2.set_xlabel(xtitle)
        ax2.set_ylabel(ytitleR, color=colR)
        if(logyR): ax2.set_yscale("log")
    
    if(len(points)>0):
        for point in points:
            # ax1.plot(arrx[point], arrsyL[0][point], marker='o', color="black")
            # ax1.annotate(f"{arrsyL[0][point]}", (arrx[point], arrsyL[0][point]), textcoords="offset points", xytext=(0,10), ha='center')
            if(point+1<len(arrsyL[0])):
                ax1.plot(arrx[point+1], arrsyL[0][point+1], marker='x', color="black")
                ax1.annotate(f"{arrsyL[0][point+1]}", (arrx[point+1], arrsyL[0][point+1]), textcoords="offset points", xytext=(0,10), ha='center')
            if(point-1>=0):
                ax1.plot(arrx[point-1], arrsyL[0][point-1], marker='x', color="black")
                ax1.annotate(f"{arrsyL[0][point-1]}", (arrx[point-1], arrsyL[0][point-1]), textcoords="offset points", xytext=(0,10), ha='center')
            ax1.plot(arrx[point], arrsyL[0][point], marker='o', color="black")
    
    plt.title(name, loc='center')
    
    pdf.savefig(fig)
    plt.close(fig)


def plot_time_diff(name,pdf,tEPICS,tEUDAQ):
    dt = np.subtract(tEPICS+NSECONDS, tEUDAQ/1e9)

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5), tight_layout=True)
    hdt = ax.hist(dt, bins=250, range=(-0.05,+0.05), rasterized=True)

    ax.set_xlim(-0.05,+0.05)
    ax.set_xlabel('Time difference (EPICS-EUDAQ) [s]')
    ax.set_ylabel('Triggers')
    ax.set_title(name)
    plt.locator_params(axis='x', nbins=10)
    ax.xaxis.set_minor_locator(AutoMinorLocator(10))
    ax.yaxis.set_minor_locator(AutoMinorLocator(10))
    ax.grid(True,linewidth=0.25,alpha=0.25)

    plt.tight_layout()
    
    pdf.savefig(fig)
    plt.close(fig)
    
    return dt


def plot_PID_mod36(name,pdf,PIDs):
    print(f"about to check {len(PIDs)} PIDs for {name}")
    offset = 10
    mult = 36
    mod  = np.zeros( len(PIDs) )
    pid0 = mod[0]
    wa = []
    for i,pid in enumerate(PIDs):
        ### check wraparounds:
        if(i>0):
            dpid = pid-pid0
            if(abs(dpid)>mult):
                print(f"Wraparound at pid[{i}]={pid} --> (pid-{offset})/{mult}={(pid-offset)/mult}")
                wa.append(i)
            pid0 = pid
        ### fill the parity
        n = int(pid-offset)
        if(n%mult==0): continue
        else:
            print(f"PID-{offset} is not multiple of 36: {n/36}")
            mod[i] = 1.5
    
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5), tight_layout=True)
    hmod = ax.hist(mod, bins=2, range=(0,2), rasterized=True)

    ax.set_xlim(0,2)
    ax.set_xlabel('PID%36')
    ax.set_ylabel('Triggers')
    ax.set_title(name)
    plt.locator_params(axis='x', nbins=10)
    ax.xaxis.set_minor_locator(AutoMinorLocator(10))
    ax.yaxis.set_minor_locator(AutoMinorLocator(10))
    ax.grid(True,linewidth=0.25,alpha=0.25)

    plt.tight_layout()
    
    pdf.savefig(fig)
    plt.close(fig)

    return wa



match_name = f'EPICS_DAQ_{epics_dataset}__ALPIDE_RUN_{alpide_run}'
with PdfPages(f'{match_name}.pdf') as pdf:
    
    xmin0 = data["SLAC_time"][0]
    xmax0 = data["SLAC_time"][-1]
    xarr0 = data["SLAC_time"]
    
    xmin = newdata["SLAC_time"][0]
    xmax = newdata["SLAC_time"][-1]
    xarr = newdata["SLAC_time"]
    
    arrsyL = []
    arrsyR = []
    for name,arr in newdata.items():
        if("TORO" in name): arrsyL.append(arr)
        if("PMT"  in name): arrsyR.append(arr)
    plotter(match_name, pdf, xarr, arrsyL, arrsyR, xmin, xmax, xtitle='SLAC time [?]', ytitleL='Toroids [pC]', ytitleR='PMTs [coutns]', colL="deepskyblue", colR="red")
    
    
    arrsyL = []
    arrsyR = []
    for name,arr in newdata.items():
        if("TORO" in name): arrsyL.append(arr)
        if("BPMS" in name): arrsyR.append(arr)
    plotter(match_name, pdf, xarr, arrsyL, arrsyR, xmin, xmax, xtitle='SLAC time [?]', ytitleL='Toroids [pC]', ytitleR='BPMs [#electrons]', colL="deepskyblue", colR="green")
    
    
    if(dopix):
        arrsyL = []
        arrsyR = []
        for name,arr in newdata.items():
            if("PMT"  in name): arrsyL.append(arr)
        for name,arr in ALPIDES.items(): arrsyR.append(arr)
        plotter(match_name, pdf, xarr, arrsyL, arrsyR, xmin, xmax, xtitle='SLAC time [?]', ytitleL='PMTs [coutns]', ytitleR='Pixels', colL="red", colR="black")
        
        arrsyL = []
        arrsyR = []
        for name,arr in newdata.items():
            if("BPMS"  in name): arrsyL.append(arr)
        for name,arr in ALPIDES.items(): arrsyR.append(arr)
        plotter(match_name, pdf, xarr, arrsyL, arrsyR, xmin, xmax, xtitle='SLAC time [?]', ytitleL='BPMs [#electrons]', ytitleR='Pixels', colL="green", colR="black")
        
        arrsyL = []
        arrsyR = []
        for name,arr in newdata.items():
            if("TORO"  in name): arrsyL.append(arr)
        for name,arr in ALPIDES.items(): arrsyR.append(arr)
        plotter(match_name, pdf, xarr, arrsyL, arrsyR, xmin, xmax, xtitle='SLAC time [?]', ytitleL='Toroids [pC]', ytitleR='Pixels', colL="deepskyblue", colR="black")
        
        
        dt = plot_time_diff(match_name,pdf,newdata["SLAC_time"],data["ALPIDES_trg_timbeg"])
        print(f"Mean of delta t is: {np.mean(dt)}")
        wraparound = plot_PID_mod36(match_name,pdf,data["scalar_PID"])
        
        for i in wraparound:
            print(f'[{i}]:')
            j = i-1
            if(j>0):
                pid = data["scalar_PID"][j]
                parity = ((pid-10)/36)%2
                print(f'  pid[{j}]={pid} --> p={parity}')
            pid = data["scalar_PID"][i]
            parity = ((pid-10)/36)%2
            print(f'  pid[{i}]={data["scalar_PID"][i]} --> p={parity}')
            j = i+1
            if(j<len(data["scalar_PID"])-1):
                pid = data["scalar_PID"][j]
                parity = ((pid-10)/36)%2
                print(f'  pid[{j}]={data["scalar_PID"][j]} --> p={parity}')


        arrsyL = [ data["scalar_PID"] ]
        arrsyR = []
        for name,arr in data.items():
            if("PMT"  in name): arrsyR.append(arr)
        plotter(match_name, pdf, xarr0, arrsyL, arrsyR, xmin0, xmax0, xtitle='SLAC time [?]', ytitleL='EPICS Pulse ID', ytitleR='PMTs [counts]', colL="blue", colR="red", lstyleL="-", lstyleR="-",  logyL=False, logyR=True, points=wraparound)
        



### time ranges for EUDAQ
print(f'EUDAQ trigger time begin: { get_human_timestamp_eudaq( data["ALPIDES_trg_timbeg"][0] ) }')
print(f'EUDAQ trigger time end:   { get_human_timestamp_eudaq( data["ALPIDES_trg_timbeg"][-1] ) }')
### time ranges for EPICS
print(f'EPICS trigger time begin: { get_human_timestamp_epics( newdata["SLAC_time"][0] ) }')
print(f'EPICS trigger time end:   { get_human_timestamp_epics( newdata["SLAC_time"][-1] ) }')




### trigger numbers for EUDAQ
eudaq_parity_begin = eudaq_parity( data["ALPIDES_trg_number"][0] )
eudaq_parity_end   = eudaq_parity( data["ALPIDES_trg_number"][-1] )
print(f'EUDAQ trigger range length: { len(data["ALPIDES_trg_number"]) }')
print(f'EUDAQ trigger begin: { data["ALPIDES_trg_number"][0] } --> parity:{ eudaq_parity_begin }')
print(f'EUDAQ trigger end:   { data["ALPIDES_trg_number"][-1] } --> parity:{ eudaq_parity_end }')
### trigger numbers for EPICS
epics_parity_begin = epics_parity(newdata["scalar_PID"][0])
epics_parity_end   = epics_parity(newdata["scalar_PID"][-1])
print(f'EPICS trigger range length: { len(newdata["scalar_PID"]) }')
print(f'EPICS trigger begin: { newdata["scalar_PID"][0] } --> parity:{ epics_parity_begin }')
print(f'EPICS trigger end:   { newdata["scalar_PID"][-1] } --> parity:{ epics_parity_end }')


epics_time_firstindex = newdata["SLAC_time"][0]+NSECONDS
eudaq_time_firstindex = data["ALPIDES_trg_timbeg"][0]/1e9
print(f'EPICS time of first index: {epics_time_firstindex} [s]')
print(f'EUDAC time of first index: {eudaq_time_firstindex} [s]')
#!/usr/bin/python
import sys
import os
import os.path
import ROOT



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
ROOT.gROOT.SetBatch(1)
ROOT.gStyle.SetOptFit(0)
# ROOT.gStyle.SetOptStat(0)


import argparse
parser = argparse.ArgumentParser(description='submit_multirun.py...')
parser.add_argument('-conf', metavar='config file', required=True,  help='full path to config file')
parser.add_argument('-muproc', metavar='run multiproc_analyzer?', required=True,  help='run multiproc_analyzer?[0/1]')
parser.add_argument('-pickle', metavar='run pickle_analyzer?', required=True,  help='run pickle_analyzer?[0/1]')
argus = parser.parse_args()
configfile = argus.conf
run_muproc = True if(argus.muproc=="1") else False
run_pickle = True if(argus.pickle=="1") else False





configfile = "conf/config_beam_data_E320_prototype_tracker_Jun242026_noalignment.txt"

runs = [1212,1217,1218,1219,
        1220,1221,1223,1226,1227,1228,1229,
        1230,1231,1232,1233,1234,1235,1236,1237,1238,1239,
        1240,1241,1242,1243,1244,1245,1246,1247,1248,1249,
        1250,1251,1252,1253,1254,1255,1256,1257,1258,1259,
        1260,1261,1262,1263,1264,1265,1266,1267,1268,1269,
        1270,1271,1272,1273,1275,1276,1277,1279,1280,1281]

def find_blocks(fcnf):
    blocks = {}
    with open(fcnf, "r") as f:
        for iline,line in enumerate(f):
            if(line.startswith("# ### begin ")):
                run = int(line.split(" ")[-1])
                blocks.update({run:[iline+1,0]})
            if(line.startswith(f"# ### end ")):
                run = int(line.split(" ")[-1])
                if(run not in blocks):
                    print("Reached end of {run} but did not get its beginning. Quitting")
                    quit()
                blocks[run][1] = iline+1
    
    rblocks = {}
    for run,block in blocks.items():
        rblocks.update({(block[0],block[1]):run})
    
    return blocks, rblocks


def find_run(iline,rblocks):
    for block,run in rblocks.items():
        if(iline>=block[0] and iline<=block[1]):
            return run
    return -1


def check_file(fcnf,rblocks):
    runtested = []
    with open(fcnf, "r") as f:
        for iline,line in enumerate(f):
            run = find_run(iline,rblocks)
            if(run==-1): continue
            if run not in runtested:
                runtested.append(run)
                if(line.startswith("# ")):
                    print(f"run {run} is commented")
                else:
                    print(f"run {run} is uncommented --> all runs should be commented before we start")
                    return False
    return True


##################################
##################################
##################################

if __name__ == "__main__":
    
    blocks,rblocks = find_blocks(configfile)
    # print(blocks)
    
    if(not check_file(configfile,rblocks)):
        print("Comment all runs before starting")
        quit()
    
    for run in runs:
        configfilerun = configfile.replace(".txt",f"_Run{run}.txt")
        block = blocks[run]
        with open(configfilerun, "w") as out:
            with open(configfile, "r") as f:
                for iline,line in enumerate(f):
                    if(iline>=block[0]-1 and iline<=block[1]):
                        newline = line.removeprefix("# ")
                        out.write(newline)
                    else:
                        out.write(line)
        if(run_muproc): ROOT.gSystem.Exec(f"python scripts/multiproc_analyzer.py -conf {configfilerun}")
        if(run_pickle): ROOT.gSystem.Exec(f"python scripts/pickle_analyzer.py -conf {configfilerun}")
        # ROOT.gSystem.Exec(f"rm -f {configfilerun}")
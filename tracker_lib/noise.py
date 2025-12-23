#!/usr/bin/python
import os
import math
import array
import numpy as np
import ROOT

from tracker_lib import config, objects


def trimNoise(h1D,pTrim,zeroSuppression=False):
    arr = []
    for b in range(h1D.GetNbinsX()+1):
        y = h1D.GetBinContent(b)
        if(zeroSuppression and y<0): continue
        arr.append(y)
    arr.sort()
    npixels = len(arr)
    nTrim = int(pTrim*npixels)
    print("nTrim",nTrim)
    avg = np.average(arr[:-nTrim]) if(npixels and nTrim>0) else -1
    std = np.std(arr[:-nTrim])     if(npixels and nTrim>0) else -1
    print("avg,std=",avg,std)
    return avg,std


def getNoiseThreshold(h1D,pTrim,nsigma,zeroSuppression=False):
    avg,std = trimNoise(h1D,pTrim,zeroSuppression)
    if(avg<0 and std<0):
        print("Warning: no trimming")
        return avg,std,0
    threshold = int(avg+nsigma*std) if(std>0) else int(avg+1)
    return avg,std,threshold


### get noisy pixels:
def getPixels2Mask(h1D,h2D,threshold):
    masked = {}
    for bx in range(h2D.GetNbinsX()+1):
        for by in range(h2D.GetNbinsY()+1):
            x = int(h2D.GetXaxis().GetBinCenter(bx))
            y = int(h2D.GetYaxis().GetBinCenter(by))
            hpix = h2D.GetBinContent(bx,by)
            ipix = h2D.FindBin(x,y)
            if(hpix>threshold):
                masked.update({ipix:[x,y]})
    return masked


def isNoise(det,pixels,masked):
    isnoise = False
    for pix in pixels:
        ipix = histos["h_pix_occ_2D_"+det].FindBin(pix.x,pix.y)
        if(ipix in masked):
            isnoise = True
            break
    return isnoise


def getGoodPixels(det,pixels,masked,hPixMatix):
    goodpixels = []
    for pix in pixels:
        ipix = hPixMatix.FindBin(pix.x,pix.y)
        if(ipix in masked): continue
        goodpixels.append(pix)
    return goodpixels


def GetNoiseMask(tfnoisename):
    cfg = config.Config().map
    print("Reading noise scan histos from:",tfnoisename)
    tfilenoise = ROOT.TFile(tfnoisename,"READ")
    noise_threshold = {}
    masked          = {}
    h1D_noise       = {}
    h2D_noise       = {}
    # ttxtnoisename = tfnoisename.replace(".root",".txt")
    # fnoisetxt = open(ttxtnoisename,"w")
    for det in cfg["detectors"]:
        noise_threshold.update( {det:-1} )
        h1D_noise.update( { det:tfilenoise.Get("h_noisescan_pix_occ_1D_"+det) } )
        h2D_noise.update( { det:tfilenoise.Get("h_noisescan_pix_occ_2D_"+det) } )
        avg,std,threshold = getNoiseThreshold(h1D_noise[det],cfg["pTrim"],cfg["nSigma"],cfg["zeroSupp"])
        print(det,": avg,std:",avg,std,"--> threshold:",threshold,"(pTrim=",cfg["pTrim"],")")
        # fnoisetxt.write(det,": avg,std:",avg,std,"--> threshold:",threshold,"(pTrim=",pTrim,")")
        noise_threshold[det] = threshold if(threshold>noise_threshold[det]) else noise_threshold[det]
        print("Final noise threshold for",det,"is:",noise_threshold[det])
        # fnoisetxt.write("Final noise threshold for",det,"is:",noise_threshold[det])
        masked.update( {det:getPixels2Mask(h1D_noise[det],h2D_noise[det],noise_threshold[det])} )
        print("Masked pixels for threshold of",noise_threshold[det],"in",det,"is:",masked[det])
        # fnoisetxt.write("Masked pixels for threshold of",noise_threshold[det],"in",det,"is:",masked[det])
    return masked


def GetNoiseMaskEmpty():
    cfg = config.Config().map
    masked = {}
    for det in cfg["detectors"]: masked.update( {det:{}} )
    return masked

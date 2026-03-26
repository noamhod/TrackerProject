#ifndef CHIPREGS_H
#define CHIPREGS_H

struct ALPIDERegs {
    // general
    int fChipId;
    int fEnabled;       
    int fEnabledWithBB; 
    int fReceiver;
    int fControlInterface;
    
    // DACs registers
    int fITHR;
    int fIDB;
    int fVCASN;
    int fVCASN2;
    int fVCLIP;
    int fVRESETD;
    int fVCASP;
    int fVPULSEL;
    int fVPULSEH;
    int fIBIAS;
    int fVRESETP;
    int fVTEMP;
    int fIAUX2;
    int fIRESET;

    // Control register settings
    int fReadoutMode; // 0 = triggered, 1 = continuous (influences busy handling)
    int fEnableClustering;
    int fMatrixReadoutSpeed;
    int fSerialLinkSpeed;
    int fEnableSkewingGlobal;
    int fEnableSkewingStartRO;
    int fEnableClockGating;
    int fEnableCMUReadout;

    // FROMU configuration register 1
    int fMEBMask;
    int fInternalStrobeGen;
    int fBusyMonitor;
    int fTestPulseMode;
    int fEnableTestStrobe;
    int fEnableRotatePulseLines;
    int fTriggerDelay; 

    // Fromu settings
    int fStrobeDuration;
    int fStrobeDelay;  // delay from pulse to strobe if generated internally
    int fStrobeGap;    // gap between subsequent strobes in sequencer mode
    int fPulseDuration;
    
    // Buffer current settings
    int fDclkReceiver;
    int fDclkDriver;
    int fMclkReceiver;
    int fDctrlReceiver;
    int fDctrlDriver;

    // CMU / DMU settings
    int fPreviousId;
    int fInitialToken;
    int fDisableManchester;
    int fEnableDdr;
    
    // DTU settings
    int fPllPhase;
    int fPllStages;
    int fChargePump;
    int fDtuDriver;
    int fDtuPreemp;

    // Scans
    int fScanThrIthr;
    int fScanThrDv;
    int fScanFhr;
};

#endif

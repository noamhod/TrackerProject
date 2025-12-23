#ifndef MOSAICREGS_H
#define MOSAICREGS_H

struct MOSAICRegs {
    char IPAddress[30];
    int NumberOfControlInterfaces; 
    int TCPPort; 
    int ControlInterfacePhase; 
    int RunCtrlAFThreshold; 
    int RunCtrlLatMode; 
    int RunCtrlTimeout; 
    int pollDataTimeout; 
    int ManchesterDisable; 
    int MasterSlave; 
    int SpeedMode;
    int Inverted; 
};

#endif
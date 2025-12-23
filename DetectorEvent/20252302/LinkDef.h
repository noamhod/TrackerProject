#ifdef __CLING__
#pragma link C++ class tuple<int,int>+;
#pragma link C++ class tuple<double,double>+;
#pragma link C++ class chip_event+;
#pragma link C++ class vector<chip_event>+;
#pragma link C++ class tlu_event+;
#pragma link C++ class stave_event+;
#pragma link C++ class detector_event+;
#pragma link C++ class detector_event_tlu+;
#pragma link C++ class ALPIDERegs+;
#pragma link C++ class chip_run_meta+;
#pragma link C++ class vector<chip_run_meta>+;
#pragma link C++ class MOSAICRegs+;
#pragma link C++ class stave_run_meta+;
#pragma link C++ class std::vector<stave_run_meta>+;
#pragma link C++ class run_meta+;
#pragma link C++ class EpicsFrame+;
#endif
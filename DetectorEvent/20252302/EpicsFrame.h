#ifndef EpicsFrame_h
#define EpicsFrame_h

#include <cstdint>

struct EpicsFrame {
    //--------------------------------------------
    
    // timestamp in nsec
    std::uint64_t epics_time = 0;

    //--------------------------------------------

    // pulse id
    std::uint64_t pulse_id = 0;
    // run number
    std::uint64_t run_n = 0;

    //--------------------------------------------

    // ESPEC_Q0 BACT
    double espec_quad0_bact = 0;
    // ESPEC_Q1 BACT
    double espec_quad1_bact = 0;
    // ESPEC_Q2 BACT
    double espec_quad2_bact = 0;
    // ESPEC_DIPOLE BACT
    double espec_dipole_bact = 0;
    // ESPEC_XCOR BACT
    double espec_xcor_bact = 0;

    //--------------------------------------------

    // ESPEC_Q0 BDES
    double espec_quad0_bdes = 0;
    // ESPEC_Q1 BDES
    double espec_quad1_bdes = 0;
    // ESPEC_Q2 BDES
    double espec_quad2_bdes = 0;
    // ESPEC_DIPOLE BDES
    double espec_dipole_bdes = 0;

    //--------------------------------------------

    // magnet calc Energy in GeV
    double mcalc_e_gev = 0;
    // magnet calc Z Object in m
    double mcalc_z_obj = 0;
    // magnet calc Z Image in m
    double mcalc_z_im = 0;
    // magnet calc M12
    double mcalc_m12 = 0;
    // magnet calc M34
    double mcalc_m34 = 0;


    //--------------------------------------------

    // Energy_BPM_2445_X
    double energy_bpm_2445_x = 0;
    // Energy_BPM_2445_Y
    double energy_bpm_2445_y = 0;

    //--------------------------------------------

    // BPM_PB_3156_X
    double bpm_pb_3156_x = 0;
    // BPM_PB_3156_Y
    double bpm_pb_3156_y = 0;
    // BPM_PB_3156_TMIT
    std::uint64_t bpm_pb_3156_tmit = 0;

    //--------------------------------------------

    // BPM_Q0_3218_X
    double bpm_quad0_3218_x = 0;
    // BPM_Q0_3218_Y
    double bpm_quad0_3218_y = 0;
    // BPM_Q0_3218_TMIT
    std::uint64_t bpm_quad0_3218_tmit = 0;

    //--------------------------------------------

    // BPM_Q1_3265_X
    double bpm_quad1_3265_x = 0;
    // BPM_Q1_3265_Y
    double bpm_quad1_3265_y = 0;
    // BPM_Q1_3265_TMIT
    std::uint64_t bpm_quad1_3265_tmit = 0;

    //--------------------------------------------

    // BPM_Q2_3315_X
    double bpm_quad2_3315_x = 0;
    // BPM_Q2_3315_Y
    double bpm_quad2_3315_y = 0;
    // BPM_Q2_3315_TMIT
    std::uint64_t bpm_quad2_3315_tmit = 0;

    //--------------------------------------------

    // PMT_S20_3060
    double pmt_s20_3060 = 0;
    // PMT_S20_3070
    double pmt_s20_3070 = 0;
    // PMT_S20_3179
    double pmt_s20_3179 = 0;
    // PMT_S20_3350
    double pmt_s20_3350 = 0;
    // PMT_S20_3360
    double pmt_s20_3360 = 0;

    //--------------------------------------------
    
    // RADM:LI20:1:CH01:MEAS
    double radm_li20_1_ch01_meas = 0;
    // XPS:LI20:MC05:M1.RBV
    double xps_li20_mc05_m1_rbv = 0;
    // XPS:LI20:MC05:M2.RBV
    double xps_li20_mc05_m2_rbv = 0;

    //--------------------------------------------
    
    // TORO:LI20:1988:TMIT_PC
    double toro_li20_1988_tmit_pc = 0;
    // TORO:LI20:2040:TMIT_PC
    double toro_li20_2040_tmit_pc = 0;
    // TORO:LI20:2452:TMIT_PC
    double toro_li20_2452_tmit_pc = 0;
    // TORO:LI20:3163:TMIT_PC
    double toro_li20_3163_tmit_pc = 0;
    // TORO:LI20:3255:TMIT_PC
    double toro_li20_3255_tmit_pc = 0;


    //--------------------------------------------

    // Vertical YAG motor position in mm
    double yag_vm_rbv = 0;
    // Horizontal YAG motor position
    double yag_hm_rbv = 0;

    //--------------------------------------------

    // laser timing relative to the beam
    double laser_beam_time = 0;
};

#endif

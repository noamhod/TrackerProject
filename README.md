# TrackerProject

Prerequisits:
- have ROOT and python3 installed
- On mac: have the DetectorEvent header and lib installed (to read the EUDAQ data files)
  - it is also included in this rep for convenience, under DetectorEvent/ for different dates
- on the DAQ PC:
  - it is already compiled in work/eudaq/lib/libeudaq_det_event_dict.so

Setup:
- On mac: Setup ROOT
- On the DAQ PC it is already setup
- On mac: `export LD_LIBRARY_PATH=$PWD/DetectorEvent/DATEDIR:$LD_LIBRARY_PATH` (DATEDIR is the specific date where the lib is valid for)
- On the DAQ PC: `export LD_LIBRARY_PATH=$HOME/work/eudaq/lib:$LD_LIBRARY_PATH`
- put data files somewhere with enough space
  - there's a dir called `data` with example data already
  - the data from eudaq on the DAQ PC it is in: $HOME/work/eudaq/user/stave/misc/run_data/
- copy the run data file e.g. to the `data` dir (can be whatever you want)
- change config file as needed (see examples in the conf/ dir)
  - particularly change the path to the input file to wherever you copied it
  - it is assumed that the file name starts with `tree_` and ends with `RunX.root` where `X` is the EUDAQ run number between 0 and 1000000

Quick start, assuming that the detector is aligned already (but read below...):
- If need noize masking run:  `python3 noise_analyzer.py -conf conf/config_file_name.txt`
- If need trigger filter run: `python3 trigger_analyzer.py -conf conf/config_file_name.txt`
- Then run the analysis:      `python3 multiproc_analyzer.py -conf conf/config_file_name.txt`
- Summarize:                  `python3 postproc_analyzer.py -conf conf/config_file_name.txt`
- Finally, check the `event_displays` dir

Directories and input business:
- Once you run any of the scripts listed above (and below), several dirs and files will be created in the same dir where the input file from the config is:
  - a dedicated run dir, e.g. `run_0000046` if the EUDAQ run number is 46
  - a subdir for the event displays, e.g. `run_0000046/event_displays` if the EUDAQ run number is 46
  - an input file will copy placed in `run_0000046/` if the EUDAQ run number is 46

Run noise scan:
- if you want to process only part of the events, set the "nmax2process" parameter as needed
- to process all events check that the "nmax2process" parameter is larger than what you have in data
- `python3 noise_analyzer.py -conf conf/config_file_name.txt`

Run analysis:
- run noise scan (see above)
- run trigger filtering (see above)
- if you want to process only part of the events, set the "nmax2process" / "nmax2processMP" parameter as needed
- to process all events check that the "nmax2process" / "nmax2processMP" parameter is larger than what you have in data
- run analysis serially OR in parallel:
  - `python3 serial_analyzer.py -conf conf/config_file_name.txt`
  - `python3 multiproc_analyzer.py -conf conf/config_file_name.txt`
- look at the histograms in the new root file

Run alignment with cosmics:
The process currently uses only one track per event so better do it with cosmics. The process is somewhat cyclic, that is, you first run the analysis with no alignment corrections, select only the good-chi2 tracks to do the alignment fit while removing the large outliers ones, run the alignment fit with this subset of tracks, update the alignment parameters and rerun the analysis to see the behaviour after the correction is applied. For the first step, it is important to use the "largest" cluster errors, so the chi2 distribution without correction lies in a reasonable range. For the cosmics runs, it is also important to loosen up the seeding algorithm requirements (mostly the rho-theta space scale parameters, the tunnel width parameters and the `seed_allow_negative_vertical_inclination` parameter). In the end of the process after applying the correction, the chi2 histogram should be peaking at ~1 and the response histograms should have a mean consistent with 0 and a sigma consistent with 1.
- step 1:
  - set all `misalignment` parameters set to 0 in the config file
  - set the `use_large_clserr_for_algnmnt` parameter to 1
  - get the tracks `python3 multiproc_analyzer.py -conf conf/config_file_name.txt`
  - look at the chi2 histogram in the root file and adjust the `minchi2align` and `maxchi2align` values to remove the large/small outliers
  - do not include too many tracks (~2k is the upper limit)
- step 2: aligning wrt e.g. ALPIDE_0 or all at once by adjusting the parameters in the config:`axes2align` and `naligniter`:
  - [option A.1] `python3 alignment_fitter.py -conf conf/config_file_name.txt -ref ALPIDE_0` or
  - [option A.2] `python3 alignment_fitter.py -conf conf/config_file_name.txt`
- step 3: choosing the fit strategy
  - [option B.1] if the `axes2align` parameter equals to `xytheta` then the fit will be FULLY-SIMULTANEOUS in 3D
  - [option B.2] if the `axes2align` parameter equals to `xy`, `xtheta` or `ytheta` then the fit will be SEMI-SIMULTANEOUS in 2D
  - [option B.3] if the `axes2align` parameter equals to `x`, `y` or `theta` then the fit will be SEQUENTIAL (i.e. non-simultaneous) in 1D
  - Notes:
    - if e.g. `-det ALPIDE_0` was used in option A.1 then you need to keep all `misalignment` parameters of `ALPIDE_0` fixed to 0 in the config file always
	 - it is advised to use one reference detector and align only the N-1 planes with respect to that with either the sequential or the simultaneous fit
	 - if the fit is SEMI-SIMULTANEOUS or SEQUENTIAL, you need to repeat steps 2-5 for all axes (e.g. `axes2align=x`->`axes2align=y`->`axes2align=theta`)
- step 4: put the non-zero resulting misalignment values in the config file for the relevant detectors
- step 5: run step 1 again, but with the `use_large_clserr_for_algnmnt` parameter set to 0, with the new `misalignment` parameters and with the `cut_chi2dof` cut loosened.
- step 6: check the residuals and the chi2 histograms

Plot some postprocessing histos and fit the residuals:
- step 1: run the alignment as discussed above
- step 2: `python3 postproc_analyzer.py -conf conf/config_file_name.txt`
- step 3: look at the pdf file created in the dir where the root file is (as listed in the config file)

Run the postprocessing plotting as above but for a few similar runs combined:
- provide the list of runs in the config file in field named `runnums` under the `MULTIRUN` area
- run step 2 as in the section above, but with the `-mult 1` flag enabled:
  - `python3 postproc_analyzer.py -conf conf/config_file_name.txt -mult 1`
  - the inputfile from the config has to point to one of the runs in the runs list but it will be otherwise not used.
  - the histograms files belonging to the runs in the list will be hadded
     - the histogram file of each run must exist individually in its dedicated dir (see "Run analysis" section)
	  - the plotting will be done with the result as an input

#!/bin/bash


# -- Pathes to task performance files outputs from WP3 performance_evaluation.py
# cls task level perf files after joining ADCP results
spcls_file=/pathto/cls/SP/pred_per-task_performances_NOUPLOAD_wADCP.csv
mpcls_file=/pathto/cls/MP/pred_per-task_performances_NOUPLOAD_wADCP.csv
spclsaux_file=/pathto/clsaux/SP/pred_per-task_performances_NOUPLOAD_wADCP.csv
mpclsaux_file=/pathto/clsaux/MP/pred_per-task_performances_NOUPLOAD_wADCP.csv

#reg task level perf files
# ! hybrid task pef files may vary
spreg_file=/pathto/regr_cens/SP/pred_per-task_performances_NOUPLOAD.csv
mpreg_file=/pathto/regr_cens/MP/pred_per-task_performances_NOUPLOAD.csv
sphyb_file=/pathto/regr_cens/SP/sp_pred_reg_per-task_performances_NOUPLOAD.csv
mphyb_file=/pathto/regr_cens/MP/mp_pred_reg_per-task_performances_NOUPLOAD.csv

# assay subset files
alive_assay=/pathto/alive_assay.csv
first_line_safety_panel=/pathto/first_line_safety_panel.csv


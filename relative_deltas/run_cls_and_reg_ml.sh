#!/bin/bash


#### EDIT HERE

relative_deltas_script=/db/melloddy/repos/performance_evaluation/relative_deltas/relative_deltas.py

# -- Pathes to task performance files outputs from WP3 performance_evaluation.py
# cls task level perf files after joining ADCP results
spcls_file=/pathto/cls/SP/pred_per-task_performances_NOUPLOAD_wADCP.csv
mpcls_file=pathto/cls/MP/pred_per-task_performances_NOUPLOAD_wADCP.csv
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

# -- workdir NEEDS TO BE ABSOLUTE path
# this will be the folder into which all results will be saved
wd=$PWD/results
mkdir -p $wd


# -- environment related 

# submissionse specify the your submission prefix here
# if singularity in use: 
# prefix="singularity exec $SINGULARITY_PATH" 

# else
prefix=""

## END EDITS do not edit below


# set  dictionary
declare -A files
files['spcls']=$spcls_file
files['spclsaux']=$spclsaux_file
files['mpcls']=$mpcls_file
files['mpclsaux']=$mpclsaux_file

files['spreg']=$spreg_file
files['sphyb']=$sphyb_file
files['mpreg']=$mpreg_file
files['mphyb']=$mphyb_file


cls_pairs=("spcls_mpcls" "spcls_mpclsaux" "spcls_spclsaux" "spclsaux_mpclsaux" "mpcls_mpclsaux" "spclsaux_mpcls")
reg_pairs=("spreg_mpreg" "spreg_mphyb"    "spreg_sphyb"    "sphyb_mphyb"       "mpreg_mphyb"    "sphyb_mpreg")
types=("absolute" "relative_improve" "improve_to_perfect")
percen=0.1

outdir=$wd/classification
for pair in ${cls_pairs[@]}; do
    baseline=$(echo $pair | cut -d'_' -f1)
    compare=$(echo $pair | cut -d'_' -f2)

    echo ">> $pair"
    echo " baseline: $baseline -> ${files[$baseline]}"
    echo " compare : $compare -> ${files[$compare]}"

    for type in ${types[@]}; do
        echo " type : $type"
	$prefix python $relative_deltas_script --type $type \
	--baseline ${files[$baseline]} \
	--compared ${files[$compare]} \
	--outdir $outdir/$pair/$type \
	--subset $alive_assay $first_line_safety_panel \
	--delta_topn $percen \
	--baseline_topn $percen
    done
done


outdir=$wd/regression
for pair in ${reg_pairs[@]}; do
    baseline=$(echo $pair | cut -d'_' -f1)
    compare=$(echo $pair | cut -d'_' -f2)

    echo ">> $pair"
    echo " baseline: $baseline -> ${files[$baseline]}"
    echo " compare : $compare -> ${files[$compare]}"

    for type in ${types[@]}; do
        echo " type : $type"
        $prefix python $relative_deltas_script --type $type \
        --baseline ${files[$baseline]} \
        --compared ${files[$compare]} \
        --outdir $outdir/$pair/$type \
        --subset $alive_assay $first_line_safety_panel \
        --delta_topn $percen \
        --baseline_topn $percen
    done
done


#!/bin/bash
#cls task level perf files after joining ADCP results
#file names might be variable depending on how you ran the perf eval codes.
clssp_file="/pathto/cls/SP/pred_per-task_performances_NOUPLOAD_wADCP.csv"
clsmp_file="/pathto/cls/MP/pred_per-task_performances_NOUPLOAD_wADCP.csv"
clsauxsp_file="/pathto/clsaux/SP/pred_per-task_performances_NOUPLOAD_wADCP.csv"
clsauxmp_file="/pathto/clsaux/MP/pred_per-task_performances_NOUPLOAD_wADCP.csv"

#reg task level perf files
regsp_file="/pathto/regr_cens/SP/pred_per-task_performances_NOUPLOAD.csv"
regmp_file="/pathto/regr_cens/MP/pred_per-task_performances_NOUPLOAD.csv"
hybsp_file="/pathto/regr_cens/SP/pred_per-task_performances_NOUPLOAD.csv"
hybmp_file="/pathto/regr_cens/MP/pred_per-task_performances_NOUPLOAD.csv"

#please specify the your submission prefix here
prefix="singularity exec $SINGULARITY_PATH"




types=("absolute" "relative_improve" "improve_to_perfect")
percen=10
#clsmp-clssp 1
for type in ${types[@]}; do
	
	$prefix python relative_deltas.py --type $type \
	--baseline $clssp_file \
	--compared $clsmp_file \
	--outdir "clsmp-clssp_$type" \
	--subset "alive_assay.csv" "first_line_safety_panel.csv" \
	--delta_topn $percen \
	--baseline_topn $percen
	
done

#clsauxmp-clssp 2
for type in ${types[@]}; do
	
	$prefix python relative_deltas.py --type $type \
	--baseline $clssp_file \
	--compared $clsauxmp_file \
	--outdir "clsauxmp-clssp_$type" \
	--subset "alive_assay.csv" "first_line_safety_panel.csv" \
	--delta_topn $percen \
	--baseline_topn $percen
done

#clsauxsp-clssp 3
for type in ${types[@]}; do
	
	$prefix  python relative_deltas.py --type $type \
	--baseline $clssp_file \
	--compared $clsauxsp_file \
	--outdir "clsauxsp-clssp_$type" \
	--subset "alive_assay.csv" "first_line_safety_panel.csv" \
	--delta_topn $percen \
	--baseline_topn $percen

done

#clsauxmp-clsauxsp 4
for type in ${types[@]}; do
	
	$prefix python relative_deltas.py --type $type \
	--baseline $clsauxsp_file \
	--compared $clsauxmp_file \
	--outdir "clsauxmp-clsauxsp_$type" \
	--subset "alive_assay.csv" "first_line_safety_panel.csv" \
	--delta_topn $percen \
	--baseline_topn $percen
done

#clauxsmp-clsmp 5
for type in ${types[@]}; do
	
	$prefix python relative_deltas.py --type $type \
	--baseline $clsmp_file \
	--compared $clsauxmp_file \
	--outdir "clsauxmp-clsmp_$type" \
	--subset "alive_assay.csv" "first_line_safety_panel.csv" \
	--delta_topn $percen \
	--baseline_topn $percen
done

#clsmp-clsauxsp 6
for type in ${types[@]}; do
	
	$prefix python relative_deltas.py --type $type \
	--baseline $clsauxsp_file \
	--compared $clsmp_file \
	--outdir "clsmp-clsauxsp_$type" \
	--subset "alive_assay.csv" "first_line_safety_panel.csv" \
	--delta_topn $percen \
	--baseline_topn $percen
done

#regression delta calc
#regmp-regsp 1
for type in ${types[@]}; do
	
	$prefix python relative_deltas.py --type $type \
	--baseline $regsp_file \
	--compared $regmp_file \
	--outdir "regmp-regsp_$type" \
	--subset "alive_assay.csv" "first_line_safety_panel.csv" \
	--delta_topn $percen \
	--baseline_topn $percen
	
done

#hybmp-regsp 2
for type in ${types[@]}; do
	
	$prefix python relative_deltas.py --type $type \
	--baseline $regsp_file \
	--compared $hybmp_file \
	--outdir "hybmp-regsp_$type" \
	--subset "alive_assay.csv" "first_line_safety_panel.csv" \
	--delta_topn $percen \
	--baseline_topn $percen
done

#hybsp-regsp 3
for type in ${types[@]}; do
	
	$prefix  python relative_deltas.py --type $type \
	--baseline $regsp_file \
	--compared $hybsp_file \
	--outdir "hybsp-regsp_$type" \
	--subset "alive_assay.csv" "first_line_safety_panel.csv" \
	--delta_topn $percen \
	--baseline_topn $percen

done

#hybmp-hybsp 4
for type in ${types[@]}; do
	
	$prefix python relative_deltas.py --type $type \
	--baseline $hybsp_file \
	--compared $hybmp_file \
	--outdir "hybmp-hybsp_$type" \
	--subset "alive_assay.csv" "first_line_safety_panel.csv" \
	--delta_topn $percen \
	--baseline_topn $percen
done

#hybmp-regmp 5
for type in ${types[@]}; do
	
	$prefix python relative_deltas.py --type $type \
	--baseline $regmp_file \
	--compared $hybmp_file \
	--outdir "hybmp-regmp_$type" \
	--subset "alive_assay.csv" "first_line_safety_panel.csv" \
	--delta_topn $percen \
	--baseline_topn $percen
done

#regmp-hybsp 6
for type in ${types[@]}; do
	
	$prefix python relative_deltas.py --type $type \
	--baseline $hybsp_file \
	--compared $regmp_file \
	--outdir "regmp-hybsp_$type" \
	--subset "alive_assay.csv" "first_line_safety_panel.csv" \
	--delta_topn $percen \
	--baseline_topn $percen
done


#!/bin/bash


#### EDIT HERE

# -- pyscript and file pathes
relative_deltas_script=/pathto/repo/performance_evaluation/relative_deltas/relative_deltas.py
fileloc=file_locations.sh

# -- Define path to OUTPUTs directory , needs to be absolute path
master_outdir=$PWD/results
mkdir -p $master_outdir

# -- Environment related 

# submission prefix if singularity in use: 
# prefix="singularity exec $SINGULARITY_PATH" 

# else
prefix=""

#### END EDITS do not edit below

# load in all file pathes
source $fileloc


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

# classification
sharedir=$master_outdir/to_share/classification
privatedir=$master_outdir/private/classification
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
	--outdir $sharedir/$pair/$type \
	--subset $alive_assay $first_line_safety_panel \
	--delta_topn $percen \
	--baseline_topn $percen
       
       # move the task level deltas to no upload folder
       mkdir -p $privatedir/$pair/$type
       mv $sharedir/$pair/$type/NOUPLOAD/* $privatedir/$pair/$type
       rm -rf $sharedir/$pair/$type/NOUPLOAD/
 
    done
done

# regression
sharedir=$master_outdir/to_share/regression
privatedir=$master_outdir/private/regression
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
        --outdir $sharedir/$pair/$type \
        --subset $alive_assay $first_line_safety_panel \
        --delta_topn $percen \
        --baseline_topn $percen

       # move the task level deltas to no upload folder
       mkdir -p $privatedir/$pair/$type
       mv $sharedir/$pair/$type/NOUPLOAD/* $privatedir/$pair/$type
       rm -rf $sharedir/$pair/$type/NOUPLOAD/



    done
done


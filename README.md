# Year 2 Performance Evaluation Script for the IMI Project MELLODDY

Performance evaluation scripts from the Single- and Multi-pharma outputs.
These evaluation scripts allow us to verify whether predictions performed using a model generated (referred to as single-pharma model hereafter) during a federated run can be reproduced when executed locally (prediction step) within the pharma partner's IT infrastructure. 
Second the evaluation assesses whether the predictions from the federated model improve over predictions from the single-pharma model.

## Requirements
On your local IT infrastructure you'd need 

1. Python 3.6 or higher
2. Local Conda installation (e.g. miniconda)
3. Data prep (>= 2.1.2) and Sparsechem (suggested >= 0.8.2) installation
4. melloddy_pipeline environment from WP1 code: https://git.infra.melloddy.eu/wp1/data_prep & Sparsechem installed into that environment


## Year 2 run analysis (performance_evaluation.py): single- vs. multi-partner evaluation




### CLS vs. CLSAUX comparison

```
$  python performance_evaluation_cls_clsaux.py -h
usage: performance_evaluation_cls_clsaux.py [-h] --y_cls Y_CLS --y_clsaux Y_CLSAUX --task_id_mapping TASK_ID_MAPPING --folding_cls FOLDING_CLS --folding_clsaux FOLDING_CLSAUX --weights_cls WEIGHTS_CLS --weights_clsaux
                                            WEIGHTS_CLSAUX --pred_cls PRED_CLS --pred_clsaux PRED_CLSAUX [--validation_fold VALIDATION_FOLD] --outfile OUTFILE

Computes statistical significance between a cls and a clsaux classification models

optional arguments:
  -h, --help            show this help message and exit
  --y_cls Y_CLS         Path to <...>/matrices/cls/cls_T10_y.npz
  --y_clsaux Y_CLSAUX   Path to <...>/matrices/clsaux/clsaux_T10_y.npz
  --task_id_mapping TASK_ID_MAPPING
                        CSV file with headers 'task_id_cls' and 'task_id_clsaux'
  --folding_cls FOLDING_CLS
                        Path to <...>/matrices/cls/cls_T11_fold_vector.npy
  --folding_clsaux FOLDING_CLSAUX
                        Path to <...>/matrices/clsaux/clsaux_T11_fold_vector.npy
  --weights_cls WEIGHTS_CLS
                        Path to <...>/matrices/clsaux/cls_weights.csv
  --weights_clsaux WEIGHTS_CLSAUX
                        Path to <...>/matrices/clsaux/clsaux_weights.csv
  --pred_cls PRED_CLS   Path to the predictions exported from platform of a cls model
  --pred_clsaux PRED_CLSAUX
                        Path to the predictions exported from platform of a clsaux model
  --validation_fold VALIDATION_FOLD
                        Validation fold to use
  --outfile OUTFILE     Name of the output file
```

#### Step 0 Create CLS-CLSAUX mapping file

Instructions to create the CLS & CLSAUX mapping file (thanks Lukas & Lina for the suggestions):

1. Read in `cls_weights.csv` and `T10c_cont.csv` (stored in MELLODDY-TUNER output folder  "results" from your cls preparation). Relevant columns in `T10c_cont.csv`: `cont_classification_task_id`, `input_assay_id`, `threshold`

2. Merge both files based on `task_id` and `cont_classification_task_id` 

3. drop duplicates

4. Repeat steps 1-3 with `clsaux_weights.csv` and `T10c_cont.csv` (from clsaux preparation)

5. Merge again the two resulting files (one from cls, one from clsaux) based on `input_assay_id`  and `threshold`. Caution: The threshold col can lead to mismatches due to given decimal places of the values. 
Recommendation: round thresholds to a given number of decimal places. Add `_cls` and `_clsaux` as suffixes like:
```
mapping_table = pd.merge(mapping_cls_unique, mapping_clsaux_unique, on=["input_assay_id", "threshold"], suffixes=["_cls", "_clsaux"])
```


#### Step 1. Generate the require file for this analysis

Ensure you have generated a CSV file (instructions in Step 0 above) with headers 'task_id_cls' and 'task_id_clsaux' that map identical tasks overlapping between CLS and CLSAUX models.

NB: Tasks that are unique to CLSAUX should not be in this file.

#### Step 2. Run the script.

For each of the comparisons (where overlapping epochs between CLS vs. CLSAUX are available - i.e. NOT epoch 20):

1. CLS MP (optimal MP epoch) vs.CLSAUX MP (optimal MP epoch)
2. CLS SP (optimal SP epoch) vs.CLSAUX SP (optimal SP epoch)
3. CLS MP (optimal MP epoch) vs.CLSAUX SP (optimal SP epoch)
4. CLS SP (optimal SP epoch) vs.CLSAUX MP (optimal MP epoch)

To do this, run the script as, e.g:
```
python performance_evaluation_cls_clsaux.py \
        --y_cls cls_T10_y.npz \
        --y_clsaux clsaux_T10_y.npz \
        --folding_cls cls_T11_fold_vector.npy \
        --folding_clsaux clsaux_T11_fold_vector.npy \
        --weights_cls cls_weights.csv \
        --weights_clsaux clsaux_weights.csv \
        --validation_fold 0 \
        --pred_cls $cls_preds \
        --pred_clsaux $clsaux_preds \
        --task_id_mapping cls_clsaux_mapping.csv \
        --outfile {sp/mp}cls_vs_{sp/mp}clsaux
```


#### Step 3. Report the summary

Only report the file {args.outfile}_summary_to_report.csv to the pharma only box folder here: https://jjcloud.box.com/s/p3jy16qgr5u59w4f3rpzvdjxzmxza37a

#### Step 4. Update Monday.com

Report progress here: https://melloddy.monday.com/boards/259897343/pulses/1526596124


------


### Evaluate models

```
$  python performance_evaluation.py -h
usage: performance_evaluation.py [-h] [--y_cls Y_CLS] [--y_clsaux Y_CLSAUX] [--y_regr Y_REGR] [--y_censored_regr Y_CENSORED_REGR]
                                    [--y_cls_single_partner Y_CLS_SINGLE_PARTNER] [--y_clsaux_single_partner Y_CLSAUX_SINGLE_PARTNER]
                                    [--y_regr_single_partner Y_REGR_SINGLE_PARTNER] [--y_cls_multi_partner Y_CLS_MULTI_PARTNER]
                                    [--y_clsaux_multi_partner Y_CLSAUX_MULTI_PARTNER] [--y_regr_multi_partner Y_REGR_MULTI_PARTNER] [--folding_cls FOLDING_CLS]
                                    [--folding_clsaux FOLDING_CLSAUX] [--folding_regr FOLDING_REGR] [--t8c_cls T8C_CLS] [--t8c_clsaux T8C_CLSAUX] [--t8r_regr T8R_REGR]
                                    [--weights_cls WEIGHTS_CLS] [--weights_clsaux WEIGHTS_CLSAUX] [--weights_regr WEIGHTS_REGR] [--run_name RUN_NAME]
                                    [--significance_analysis {0,1}] [--verbose {0,1}] [--validation_fold {0,1,2,3,4} [{0,1,2,3,4} ...]]
                                    [--aggr_binning_scheme_perf AGGR_BINNING_SCHEME_PERF [AGGR_BINNING_SCHEME_PERF ...]]
                                    [--aggr_binning_scheme_perf_delta AGGR_BINNING_SCHEME_PERF_DELTA [AGGR_BINNING_SCHEME_PERF_DELTA ...]]

MELLODDY Year 2 Performance Evaluation

optional arguments:
  -h, --help            show this help message and exit
  --y_cls Y_CLS         Classification activity file (npz) (e.g. cls_T10_y.npz)
  --y_clsaux Y_CLSAUX   Aux classification activity file (npz) (e.g. cls_T10_y.npz)
  --y_regr Y_REGR       Activity file (npz) (e.g. reg_T10_y.npz)
  --y_censored_regr Y_CENSORED_REGR
                        Censored activity file (npz) (e.g. reg_T10_censor_y.npz)
  --y_cls_single_partner Y_CLS_SINGLE_PARTNER
                        Yhat cls prediction output from an single-partner run (e.g. <single pharma dir>/<cls_prefix>-class.npy)
  --y_clsaux_single_partner Y_CLSAUX_SINGLE_PARTNER
                        Yhat clsaux prediction from an single-partner run (e.g. <single pharma dir>/<clsaux_prefix>-class.npy)
  --y_regr_single_partner Y_REGR_SINGLE_PARTNER
                        Yhat regr prediction from an single-partner run (e.g. <single pharma dir>/<regr_prefix>-regr.npy)
  --y_cls_multi_partner Y_CLS_MULTI_PARTNER
                        Classification prediction output for comparison (e.g. pred from the multi-partner run)
  --y_clsaux_multi_partner Y_CLSAUX_MULTI_PARTNER
                        Classification w/ aux prediction output for comparison (e.g. pred from the multi-partner run)
  --y_regr_multi_partner Y_REGR_MULTI_PARTNER
                        Regression prediction output for comparison (e.g. pred from the multi-partner run)
  --folding_cls FOLDING_CLS
                        Folding file (npy) (e.g. cls_T11_fold_vector.npy)
  --folding_clsaux FOLDING_CLSAUX
                        Folding file (npy) (e.g. cls_T11_fold_vector.npy)
  --folding_regr FOLDING_REGR
                        Folding file (npy) (e.g. reg_T11_fold_vector.npy)
  --t8c_cls T8C_CLS     T8c file for classification in the results_tmp/classification folder
  --t8c_clsaux T8C_CLSAUX
                        T8c file for classification w/ auxiliary in the results_tmp/classification folder
  --t8r_regr T8R_REGR   T8r file for regression in the results_tmp/regression folder
  --weights_cls WEIGHTS_CLS
                        CSV file with columns task_id and weight (e.g. cls_weights.csv)
  --weights_clsaux WEIGHTS_CLSAUX
                        CSV file with columns task_id and weight (e.g cls_weights.csv)
  --weights_regr WEIGHTS_REGR
                        CSV file with columns task_id and weight (e.g. reg_weights.csv)
  --run_name RUN_NAME   Run name directory for results from this output (timestemp used if not specified)
  --significance_analysis {0,1}
                        Run significant analysis (1 = Yes, 0 = No sig. analysis
  --verbose {0,1}       Verbosity level: 1 = Full; 0 = no output
  --validation_fold {0,1,2,3,4} [{0,1,2,3,4} ...]
                        Validation fold to used to calculate performance
  --aggr_binning_scheme_perf AGGR_BINNING_SCHEME_PERF [AGGR_BINNING_SCHEME_PERF ...]
                        Shared aggregated binning scheme for performances
  --aggr_binning_scheme_perf_delta AGGR_BINNING_SCHEME_PERF_DELTA [AGGR_BINNING_SCHEME_PERF_DELTA ...]
                        Shared aggregated binning scheme for delta performances

```


#### Step 1.1. Retrieve the MP and SP cls, clsaux & reg outputs from substra and decompress.
E.g. for each model type (cls,clsaux and reg) and each variant (different cp's) do:
```
gunzip -c <cls-hash>.tar.gz | tar xvf -
gunzip -c <clsaux-hash>.tar.gz | tar xvf -
gunzip -c <reg-hash>.tar.gz | tar xvf -
```

#### Step 1.2. Select the optimal cls & clsaux mp models.
Identify the optimal performing model based on the AUCPR reported in perf/perf.json.

REMOVE the other models & predictions in pred/pred and export/model.pth folders.

E.g: a very crude example could be:
```
$ head  <clsaux_models_to_compre*>/perf/perf.json
==> <model1>/perf/perf.json <==
{"all": 0.6}
==> <model2>/perf/perf.json <==
{"all": 0.7}
==> <model3>/perf/perf.json <==
{"all": 0.8}    # <==== this would be the model to use in this crude example

$ rm model1/export/model.pth && rm model1/pred/pred
$ rm model2/export/model.pth && rm model2/pred/pred
```

#### Step 1.3. Select only 20 epoch reg model for SP & MP.
Use ONLY the Epoch 20 (SP&MP) models for reg.

I.e: Do NOT select model based on corr (as arbitrated here: https://jjcloud.box.com/s/xrhdrba9ocwz8nq150wdaarib86lr037)
```
gunzip -c <reg-hash-epoch20>.tar.gz | tar xvf -
```

#### Step 2. Run the performance evaluation code:

Run the code supplying the best/epoch 20 MP models to the script.

E.g. for a clsaux model:
```
python performance_evaluation.py --y_clsaux <clsaux_dir>/clsaux_T10_y.npz --y_clsaux_single_partner <best_sp_model>/pred/pred --y_clsaux_multi_partner <best_mp_model>/pred/pred --folding_clsaux <clsaux_dir>/clsaux_T11_fold_vector.npy --t8c_clsaux <clsaux_dir>/T8c.csv --weights_clsaux <clsaux_dir>/clsaux_weights.csv --validation_fold 0 --run_name slurm_y2_test
```

Output should look something like:

```
[INFO]: === WP3 Y2 Performance evaluation script for npy and pred files ===
[INFO]: Wrote input params to 'slurm_y2_test/run_params.json'


=======================================================================================================================================
Evaluating clsaux performance
=======================================================================================================================================

[INFO]: Loading clsaux: <clsaux_dir>/clsaux_T10_y.npz
[INFO]: Loading (pred) output for: <sp_model>/pred/pred
[INFO]: Loading (pred) output for: <mp_model>/pred/pred

[INFO]: === Calculating significance ===
[INFO]: === Calculating <sp_model>/pred/pred performance ===
[INFO]: Wrote per-task report to: slurm_y2_test/clsaux/SP/pred_per-task_performances.csv
[INFO]: Wrote per-task binned performance report to: slurm_y2_test/clsaux/SP/pred_binned_per-task_performances.csv
[INFO]: Wrote per-assay report to: slurm_y2_test/clsaux/SP/pred_per-assay_performances.csv
[INFO]: Wrote per-assay binned report to: slurm_y2_test/clsaux/SP/pred_binned_per-task_performances.csv
[INFO]: Wrote global report to: slurm_y2_test/clsaux/SP/pred_global_performances.csv

[INFO]: === Calculating <mp_model>/pred/pred performance ===
[INFO]: Wrote per-task report to: slurm_y2_test/clsaux/MP/pred_per-task_performances.csv
[INFO]: Wrote per-task binned performance report to: slurm_y2_test/clsaux/MP/pred_binned_per-task_performances.csv
[INFO]: Wrote per-assay report to: slurm_y2_test/clsaux/MP/pred_per-assay_performances.csv
[INFO]: Wrote per-assay binned report to: slurm_y2_test/clsaux/MP/pred_binned_per-task_performances.csv
[INFO]: Wrote global report to: slurm_y2_test/clsaux/MP/pred_global_performances.csv

[INFO]: Wrote per-task delta report to: slurm_y2_test/clsaux/deltas/deltas_per-task_performances.csv
[INFO]: Wrote binned performance per-task delta report to: slurm_y2_test/clsaux/deltas/deltas_binned_per-task_performances.csv
[INFO]: Wrote per-assay delta report to: slurm_y2_test/clsaux/deltas/deltas_per-assay_performances.csv
[INFO]: Wrote binned performance per-assay delta report to: slurm_y2_test/clsaux/deltas/deltas_binned_per-task_performances.csv
[INFO]: Wrote significance performance report to: slurm_y2_test_cls/clsaux/deltas/delta_significance.csv
[INFO]: Wrote per-assay significance report to: slurm_y2_test_cls/clsaux/deltas/delta_significance.csv
[INFO]: Wrote sp significance report to: slurm_y2_test_cls/clsaux/deltas/delta_sp_significance.csv
[INFO]: Wrote per-assay sp significance report to: slurm_y2_test_cls/clsaux/deltas/delta_per-assay_sp_significance.csv
[INFO]: Wrote mp significance report to: slurm_y2_test_cls/clsaux/deltas/delta_mp_significance.csv
[INFO]: Wrote per-assay mp significance report to: slurm_y2_test_cls/clsaux/deltas/delta_per-assay_mp_significance.csv


[INFO]: Run name 'slurm_y2_test' is finished.
[INFO]: Performance evaluation took 482.36725 seconds.
```

The following files are created:

```
  derisk_test #name of the run (timestamp used if not defined)
  ├── <clsaux>
  │   ├── deltas # the folder for sp vs. mp deltas 
  │   │   ├── delta_mp_significance.csv #mp significance of deltas between sp & mp // to be reported to WP3
  │   │   ├── delta_sp_significance.csv #sp significance of deltas between sp & mp // to be reported to WP3
  │   │   ├── delta_per-assay_mp_significance.csv #mp significance of deltas between sp & mp per-assay type  // to be reported to WP3
  │   │   ├── delta_per-assay_sp_significance.csv #sp significance of deltas between sp & mp per-assay type  // to be reported to WP3
  │   │   ├── deltas_binned_per-assay_performances.csv	#binned assay aggregated deltas between sp & mp  // to be reported to WP3
  │   │   ├── deltas_binned_per-task_performances.csv	#binned deltas across all tasks between sp & mp // to be reported to WP3
  │   │   ├── deltas_global_performances.csv	#global aggregated deltas between sp & mp  // to be reported to WP3
  │   │   ├── deltas_per-assay_performances.csv	#assay aggregated deltas between sp & mp // to be reported to WP3
  │   │   └── deltas_per-task_performances.csv	#deltas between sp & mp
  │   ├── sp #e.g. the folder of single-partner prediction results
  │   │   ├── pred_binned_per-assay_performances.csv	#binned sp assay aggregated performances
  │   │   ├── pred_binned_per-task_performances.csv	#binned sp performances
  │   │   ├── pred_global_performances.csv	#sp global performance
  │   │   ├── pred_per-assay_performances.csv	#sp assay aggregated performances
  │   │   └── pred_per-task_performances.csv	#sp performances
  │   └── mp #e.g. the folder of multi-partner predictions results
  │   │   ├── pred_binned_per-assay_performances.csv	#binned mp assay aggregated performances
  │   │   ├── pred_binned_per-task_performances.csv	#binned mp performances
  │   │   ├── pred_global_performances.csv	#mp global performance
  │   │   ├── pred_per-assay_performances.csv	#mp assay aggregated performances
  │   │   └── pred_per-task_performances.csv	#mp performances
  └── run_params.json #json with the runtime parameters

```

#### Step 3. Upload the files to the "Performance metric evaluation" folder in the WP3 pharma only box (https://jjcloud.box.com/s/7k4x3i73c6o9m8sqpe90u636lczjben1):

Expected files are:
```
1.) deltas_binned_per-assay_performances.csv	
2.) deltas_binned_per-task_performances.csv	
3.) deltas_global_performances.csv
4.) deltas_per-assay_performances.csv
5.) delta_mp_significance.csv (will not be available for regr / regr_cens)
6.) delta_sp_significance.csv (will not be available for regr / regr_cens)
7.) delta_per-assay_mp_significance.csv (will not be available for regr / regr_cens)
8.) delta_per-assay_sp_significance.csv (will not be available for regr / regr_cens)
```

N.B: Ensure you do not upload:
```
deltas_per-task_performances.csv
```

N.B: Regression will also output both regr and regr_cens folders - please  upload BOTH sets of relevant files to the WP3 pharma only box (https://jjcloud.box.com/s/7k4x3i73c6o9m8sqpe90u636lczjben1).

N.B: Regression does not calculate significance, so expect those files not to be present in regr / regr-cens outputs.

#### Step 4. Update status on Monday.com:
Please update your status on Monday.com when you have uploaded your files: https://melloddy.monday.com/boards/259897343/pulses/1236182296.

Additional analysis may be run after the D3.8 report to the IMI. 

------


## === De-risk analysis below here ===


## De-risk of 2 epoch MP models (cls/clsaux)

#### Step 1. Retrieve the cls & clsaux output from substra and decompress
```
gunzip -c <cls-hash>.tar.gz | tar xvf - && gunzip -c <clsaux-hash>.tar.gz | tar xvf -
```

#### Step 2. Create SparseChem predictions on-premise with the outputted model

##### 2a. Fix both cls & clsaux hyperparameter.json (by removing model_type) & setup dirs:

```/
sed -i 's/, "model_type": "federated"//g' <2epoch_mp_cls_dir>/export/hyperparameters.json 
sed -i 's/, "model_type": "federated"//g' <2epoch_mp_clsaux_dir>/export/hyperparameters.json 
mkdir derisk_cls derisk_clsaux
```

##### 2b. Predict the validation fold using the MP model output from substra (load your conda env), e.g.:

for cls:
```
python <sparsechem_dir>/examples/chembl/predict.py \
  --x <cls_dir>/cls_T11_x.npz \
  --y_class <cls_dir>/cls_T10_y.npz \
  --folding <cls_dir>/cls_T11_fold_vector.npy \
  --predict_fold 0 \
  --conf <2epoch_mp_cls_dir>/export/hyperparameters.json \
  --model <2epoch_mp_cls_dir>/export/model.pth \
  --dev cuda:0 \
  --outprefix "derisk_cls/pred"
```
and clsaux:
```
python <sparsechem_dir>/examples/chembl/predict.py \
  --x <clsaux_dir>/clsaux_T11_x.npz \
  --y_class <clsaux_dir>/clsaux_T10_y.npz \
  --folding <clsaux_dir>/clsaux_T11_fold_vector.npy \
  --predict_fold 0 \
  --conf <2epoch_mp_clsaux_dir>/export/hyperparameters.json \
  --model <2epoch_mp_clsaux_dir>/export/model.pth \
  --dev cuda:0 \
  --outprefix "derisk_clsaux/pred"
  ```
  
#### Step 3. Run the derisk script
```
python performance_evaluation_derisk.py \ 
  --y_cls cls/cls_T10_y.npz \ 
  --y_cls_onpremise derisk_cls/pred-class.npy \ 
  --y_cls_substra <2epoch_mp_cls_dir>/pred/pred \ 
  --folding_cls cls/cls_T11_fold_vector.npy \
  --t8c_cls cls/T8c.csv \
  --weights_cls cls/cls_weights.csv \ 
  --perf_json_cls <2epoch_mp_cls_dir>/perf/perf.json \ 
  --y_clsaux clsaux/clsaux_T10_y.npz \ 
  --y_clsaux_onpremise derisk_clsaux/pred-class.npy \ 
  --y_clsaux_substra <2epoch_mp_clsaux_dir>/pred/pred \ 
  --folding_clsaux clsaux/clsaux_T11_fold_vector.npy \ 
  --t8c_clsaux clsaux/T8c.csv \ 
  --weights_clsaux clsaux/clsaux_weights.csv \ 
  --perf_json_clsaux <2epoch_mp_clsaux_dir>/perf/perf.json \ 
  --validation_fold 0 \ 
  --run_name derisk_2epoch_cls_clsaux
```

Output should look something like this:
```
=======================================================================================================================================
=======================================================================================================================================
De-risking cls performance
=======================================================================================================================================
=======================================================================================================================================

[INFO]: Loading cls: cls/cls_T10_y.npz
[INFO]: Loading (npy) predictions for: derisk_cls/pred-class.npy
[INFO]: Loading (pred) output for: <2epoch_mp_cls_dir>/pred/pred

=======================================================================================================================================
[DERISK-CHECK #1]: PASSED! yhats close between 'pred-class' and 'pred' (tol:1e-05)
Spearmanr rank correlation coefficient of the 'pred-class' and 'pred' yhats = SpearmanrResult(correlation=0.9999999999999996, pvalue=0.0)
=======================================================================================================================================

[INFO]: === Calculating derisk_cls/pred-class.npy performance ===
[INFO]: Wrote per-task report to: <derisk_run>/cls/pred-class/pred-class_per-task_performances.csv
[INFO]: Wrote per-task binned performance report to: <derisk_run>/cls/pred-class/pred-class_binned_per-task_performances.csv
[INFO]: Wrote per-assay report to: <derisk_run>/cls/pred-class/pred-class_per-assay_performances.csv
[INFO]: Wrote per-assay binned report to: <derisk_run>/cls/pred-class/pred-class_binned_per-task_performances.csv
[INFO]: Wrote global report to: <derisk_run>/cls/pred-class/pred-class_global_performances.csv

[INFO]: === Calculating <2epoch_mp_cls_dir>/pred/pred performance ===
[INFO]: Wrote per-task report to: <derisk_run>/cls/pred/pred_per-task_performances.csv
[INFO]: Wrote per-task binned performance report to: <derisk_run>/cls/pred/pred_binned_per-task_performances.csv
[INFO]: Wrote per-assay report to: <derisk_run>/cls/pred/pred_per-assay_performances.csv
[INFO]: Wrote per-assay binned report to: <derisk_run>/cls/pred/pred_binned_per-task_performances.csv

=======================================================================================================================================
[DERISK-CHECK #2]: SKIPPED! substra does not report individual task performances
=======================================================================================================================================

[INFO]: Wrote global report to: <derisk_run>/cls/pred/pred_global_performances.csv

=======================================================================================================================================
[DERISK-CHECK #3]: FAILED! global reported performance metrics and global calculated performance metrics NOT close (tol:1e-05)
Calculated:<removed>
Reported:<removed>
=======================================================================================================================================

=======================================================================================================================================
[DERISK-CHECK #4]: PASSED! delta between local & substra assay_type aggregated performances close to 0 across all metrics (tol:1e-05)
=======================================================================================================================================

[INFO]: Wrote per-task delta report to: <derisk_run>/cls/deltas/deltas_per-task_performances.csv
[INFO]: Wrote binned performance per-task delta report to: <derisk_run>/cls/deltas/deltas_binned_per-task_performances.csv
[INFO]: Wrote per-assay delta report to: <derisk_run>/cls/deltas/deltas_per-assay_performances.csv
[INFO]: Wrote binned performance per-assay delta report to: <derisk_run>/cls/deltas/deltas_binned_per-task_performances.csv

=======================================================================================================================================
[DERISK-CHECK #5]: PASSED! delta performance between global local & global substra performances close to 0 across all metrics (tol:1e-05)
=======================================================================================================================================

=======================================================================================================================================
=======================================================================================================================================
De-risking clsaux performance
=======================================================================================================================================
=======================================================================================================================================

[INFO]: Loading clsaux: clsaux/clsaux_T10_y.npz
[INFO]: Loading (npy) predictions for: derisk_clsaux/pred-class.npy
[INFO]: Loading (pred) output for: <2epoch_mp_clsaux_dir>/pred/pred

=======================================================================================================================================
[DERISK-CHECK #1]: PASSED! yhats close between 'pred-class' and 'pred' (tol:1e-05)
Spearmanr rank correlation coefficient of the 'pred-class' and 'pred' yhats = SpearmanrResult(correlation=1.0, pvalue=0.0)
=======================================================================================================================================

[INFO]: === Calculating derisk_clsaux/pred-class.npy performance ===
[INFO]: Wrote per-task report to: <derisk_run>/clsaux/pred-class/pred-class_per-task_performances.csv
[INFO]: Wrote per-task binned performance report to: <derisk_run>/clsaux/pred-class/pred-class_binned_per-task_performances.csv
[INFO]: Wrote per-assay report to: <derisk_run>/clsaux/pred-class/pred-class_per-assay_performances.csv
[INFO]: Wrote per-assay binned report to: <derisk_run>/clsaux/pred-class/pred-class_binned_per-task_performances.csv
[INFO]: Wrote global report to: <derisk_run>/clsaux/pred-class/pred-class_global_performances.csv

[INFO]: === Calculating <2epoch_mp_clsaux_dir>/pred/pred performance ===
[INFO]: Wrote per-task report to: <derisk_run>/clsaux/pred/pred_per-task_performances.csv
[INFO]: Wrote per-task binned performance report to: <derisk_run>/clsaux/pred/pred_binned_per-task_performances.csv
[INFO]: Wrote per-assay report to: <derisk_run>/clsaux/pred/pred_per-assay_performances.csv
[INFO]: Wrote per-assay binned report to: <derisk_run>/clsaux/pred/pred_binned_per-task_performances.csv

=======================================================================================================================================
[DERISK-CHECK #2]: SKIPPED! substra does not report individual task performances
=======================================================================================================================================

[INFO]: Wrote global report to: <derisk_run>/clsaux/pred/pred_global_performances.csv

=======================================================================================================================================
[DERISK-CHECK #3]: FAILED! global reported performance metrics and global calculated performance metrics NOT close (tol:1e-05)
Calculated:<removed>
Reported:<removed>
=======================================================================================================================================

=======================================================================================================================================
[DERISK-CHECK #4]: PASSED! delta between local & substra assay_type aggregated performances close to 0 across all metrics (tol:1e-05)
=======================================================================================================================================

[INFO]: Wrote per-task delta report to: <derisk_run>/clsaux/deltas/deltas_per-task_performances.csv
[INFO]: Wrote binned performance per-task delta report to: <derisk_run>/clsaux/deltas/deltas_binned_per-task_performances.csv
[INFO]: Wrote per-assay delta report to: <derisk_run>/clsaux/deltas/deltas_per-assay_performances.csv
[INFO]: Wrote binned performance per-assay delta report to: <derisk_run>/clsaux/deltas/deltas_binned_per-task_performances.csv

=======================================================================================================================================
[DERISK-CHECK #5]: PASSED! delta performance between global local & global substra performances close to 0 across all metrics (tol:1e-05)
=======================================================================================================================================

[INFO]: Run name '<derisk_run>' is finished.
[INFO]: Performance evaluation de-risk took 1101.3661 seconds.
```

The file 'derisk_summary.csv' has a concise summary of de-risk checks



### De-risk MP reg models

#### Step 1. Retrieve a regression output from substra and decompress
```
gunzip -c <reg-hash>.tar.gz | tar xvf -
```

#### Step 2. Create SparseChem predictions on-premise with the outputted model

##### 2a. Fix reg hyperparameter.json (by removing model_type) & setup dirs:

```
sed -i 's/, "model_type": "federated"//g' <mp_reg_dir>/export/hyperparameters.json 
mkdir derisk_reg
```

##### 2b. Predict the validation fold using the MP model output from substra (load your conda env), e.g.:
```
python <sparsechem_dir>/sparsechem/examples/chembl/predict.py \
  --x <reg_dir>/reg_T11_x.npz \
  --y_regr <reg_dir>/reg_T10_y.npz \
  --folding <reg_dir>/reg_T11_fold_vector.npy \
  --predict_fold 2 \
  --conf <mp_reg_dir>/export/hyperparameters.json \
  --model <mp_reg_dir>/export/model.pth \
  --dev cuda:0 \
  --inverse_normalization 1 \
  --outprefix "derisk_reg/pred"
```

  
#### Step 3. Run the derisk script, e.g.:
```
python performance_evaluation_derisk.py \ 
  --y_regr reg/reg_T10_y.npz \ 
  --y_regr_onpremise derisk_reg/pred-regr.npy \ 
  --y_regr_substra <mp_reg_dir>/pred/pred \ 
  --folding_regr reg/reg_T11_fold_vector.npy \
  --t8r_reg reg/T8r.csv \
  --weights_regr reg/reg_weights.csv \ 
  --perf_json_regr <mp_reg_dir>/perf/perf.json \ 
  --validation_fold 2 \ 
  --run_name derisk_<epoch>_reg
```

Report output here: https://jjcloud.box.com/s/ok3k2p6ugbr7nt189b9y2qw2fbmakiqd



### Delta between on-premise and substra models

#### Step 1. Train a local model with the exact same hyperparameters as it was trianed on the platform, e.g.:
```
python $train \
  --x $data_path/cls/cls_T11_x.npz \
  --y $data_path/cls/cls_T10_y.npz \
  --folding $data_path/cls/cls_T11_fold_vector.npy \
  --weights_class $data_path/cls/cls_weights.csv \
  --hidden_sizes $hidden_sizes \
  --weight_decay 0 \
  --last_dropout $dropout \
  --middle_dropout $dropout \
  --last_non_linearity relu \
  --non_linearity relu \
  --input_transform binarize \
  --lr 0.001 \
  --lr_alpha 0.3 \
  --lr_steps 10 \
  --epochs 20 \
  --normalize_loss 100_000 \
  --eval_frequency 1 \
  --batch_ratio 0.02 \
  --fold_va 0 \
  --verbose 1 \
  --save_model 1 
```
#### Step 2. check the performance here:
```
import sparsechem as sc
res = sc.load_results('models/sc_run_h2500_ldo0.8_wd0.0_lr0.001_lrsteps10_ep20_fva0_fteNone.json')
print(res['validation']['classification_agg'])   
```

#### Step 3. Run the derisk workflow as usual and you will get some files in the output folder like:

```
cat derisk_sp_20epoch_cls/cls/pred/pred_global_performances.csv
```

The aucpr values needs to be compared for cls, or coeff for reg 



### Minimum working example

#### Step 1. Download data

##### 1a. Download the MELLODDY_tuner (2.1.2) processed public dataset from here: https://jjcloud.box.com/s/g5c5efl864h5aah7chp3ot0d0urwg646

##### 1b. Extract the 'cls.tar.gz', 'reg.tar.gz' and 'tmp_files_wo_aux.tar.gz' folders:
```
gunzip -c cls.tar.gz | tar xvf - && gunzip -c reg.tar.gz | tar xvf - && gunzip -c cls.tar.gz | tar xvf - && gunzip -c tmp_files_wo_aux.tar.gz | tar xvf -
```
##### 1c. Move t8c and t8r to the corresponding 'cls' and 'reg' directory, respectively:
```
mv results_tmp/classification/T8c.csv cls/ && mv results_tmp/regression/T8r.csv reg/
```

#### Step 2. Train onpremise models (& emulate a substra output model)

##### 2a. Train a 2 epoch classification (cls) model with sparsechem:
```
python <sparsechem_dir>/examples/chembl/train.py \
  --x cls/cls_T11_x.npz \
  --y cls/cls_T10_y.npz \
  --folding cls/cls_T11_fold_vector.npy \
  --fold_va 2 \
  --fold_te 0 \
  --batch_ratio    0.02 \
  --hidden_sizes   800 \
  --last_dropout   0.4 \
  --middle_dropout 0.4 \
  --weight_decay   1e-6 \
  --epochs         2 \
  --lr             1e-3 \
  --lr_steps	   10 \
  --lr_alpha	   0.3 \
  --normalize_loss 100_000 \
  --eval_frequency 1 \
  --output_dir cls \
  --verbose 1
```

##### 2b. Train a 2 epoch regression (reg) model with sparsechem:
```
python <sparsechem_dir>/examples/chembl/train.py \
  --x reg/reg_T11_x.npz \
  --y_regr reg/reg_T10_y.npz \
  --folding reg/reg_T11_fold_vector.npy \
  --fold_va 2 \
  --fold_te 0 \
  --batch_ratio    0.02 \
  --hidden_sizes   400 \
  --last_dropout   0.2 \
  --middle_dropout 0.2 \
  --weight_decay   0.0 \
  --epochs         2 \
  --lr             1e-3 \
  --lr_steps	   10 \
  --lr_alpha	   0.3 \
  --output_dir reg \
  --verbose 1
```

#### Step 3. Predict onpremise models (& an emulated substra output prediction)

##### 3a. Predict the classification validation fold:
```
python <sparsechem_dir>/examples/chembl/predict.py \
  --x cls/cls_T11_x.npz \
  --y_class cls/cls_T10_y.npz \
  --folding cls/cls_T11_fold_vector.npy \
  --predict_fold 2 \
  --conf cls/sc_run_h800_ldo0.4_wd1e-06_lr0.001_lrsteps10_ep2_fva2_fte0.json \
  --model cls/sc_run_h800_ldo0.4_wd1e-06_lr0.001_lrsteps10_ep2_fva2_fte0.pt \
  --outprefix "cls/pred"
```

##### 3b. Predict the regression validation fold:
```
python <sparsechem_dir>/sparsechem/examples/chembl/predict.py \
  --x reg/reg_T11_x.npz \
  --y_regr reg/reg_T10_y.npz \
  --folding reg/reg_T11_fold_vector.npy \
  --predict_fold 2 \
  --conf reg/sc_run_h400_ldo0.2_wd0.0_lr0.001_lrsteps10_ep2_fva2_fte0.json \
  --model reg/sc_run_h400_ldo0.2_wd0.0_lr0.001_lrsteps10_ep2_fva2_fte0.pt \
  --dev cuda:0 \
  --inverse_normalization \
  --outprefix "reg/pred"
```

##### 3c. Emulate substra (.npy) output predictions (NB: the 'real-life' output will be 'pred' files):
```
cp cls/pred-class.npy cls/pred-class-copy.npy && cp reg/pred-regr.npy reg/pred-regr-copy.npy
```


The directories should now look like:
```
$ tree -l reg/ cls/
  reg/
  ├── boards
  │   └── sc_run_h400_ldo0.2_wd0.0_lr0.001_lrsteps10_ep2_fva2_fte0
  │       └── events.out.<removed>
  ├── pred-regr-copy.npy
  ├── pred-regr.npy
  ├── reg_T10_censor_y.npz
  ├── reg_T10_y.npz
  ├── reg_T11_fold_vector.npy
  ├── reg_T11_x.npz
  ├── reg_weights.csv
  ├── sc_run_h400_ldo0.2_wd0.0_lr0.001_lrsteps10_ep2_fva2_fte0.json
  ├── sc_run_h400_ldo0.2_wd0.0_lr0.001_lrsteps10_ep2_fva2_fte0.pt
  └── T8r.csv
  cls/
  ├── boards
  │   └── sc_run_h800_ldo0.4_wd1e-06_lr0.001_lrsteps10_ep2_fva2_fte0
  │       └── events.out.<removed>
  ├── cls_T10_y.npz
  ├── cls_T11_fold_vector.npy
  ├── cls_T11_x.npz
  ├── cls_weights.csv
  ├── pred-class-copy.npy
  ├── pred-class.npy
  ├── sc_run_h800_ldo0.4_wd1e-06_lr0.001_lrsteps10_ep2_fva2_fte0.json
  ├── sc_run_h800_ldo0.4_wd1e-06_lr0.001_lrsteps10_ep2_fva2_fte0.pt
  └── T8c.csv

  4 directories, 21 files
```

#### Step 4. Run the derisk script

```
python performance_evaluation_derisk.py \ 
  --y_cls cls/cls_T10_y.npz \ 
  --y_cls_onpremise cls/pred-class.npy \ 
  --y_cls_substra cls/pred-class-copy.npy \ 
  --folding_cls cls/cls_T11_fold_vector.npy \ 
  --t8c_cls cls/T8c.csv \ 
  --weights_cls cls/cls_weights.csv \ 
  --perf_json_cls cls/sc_run_h800_ldo0.4_wd1e-06_lr0.001_lrsteps10_ep2_fva2_fte0.json \ 
  --y_regr reg/reg_T10_y.npz \ 
  --y_regr_onpremise reg/pred-regr.npy \ 
  --y_regr_substra reg/pred-regr-copy.npy \ 
  --folding_regr reg/reg_T11_fold_vector.npy \ 
  --t8r_reg reg/T8r.csv \ 
  --weights_regr reg/reg_weights.csv \ 
  --perf_json_regr reg/sc_run_h400_ldo0.2_wd0.0_lr0.001_lrsteps10_ep2_fva2_fte0.json \ 
  --validation_fold 2 \ 
  --run_name derisk_test
```

The terminal output should look like:
```
[INFO]: === WP3 Y2 Performance evaluation de-risk script for npy and pred files ===
[INFO]: Namespace(aggr_binning_scheme_perf=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], aggr_binning_scheme_perf_delta=[-0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2], folding_cls='cls/cls_T11_fold_vector.npy', folding_clsaux=None, folding_regr='reg/reg_T11_fold_vector.npy', perf_json_cls='cls/sc_run_h800_ldo0.4_wd1e-06_lr0.001_lrsteps10_ep2_fva2_fte0.json', perf_json_clsaux=None, perf_json_regr='reg/sc_run_h400_ldo0.2_wd0.0_lr0.001_lrsteps10_ep2_fva2_fte0.json', run_name='derisk_test', t8c_cls='cls/T8c.csv', t8c_clsaux=None, t8r_regr='reg/T8r.csv', validation_fold=[2], verbose=1, weights_cls='cls/cls_weights.csv', weights_clsaux=None, weights_regr='reg/reg_weights.csv', y_cls='cls/cls_T10_y.npz', y_cls_onpremise='cls/pred-class.npy', y_cls_substra='cls/pred-class-copy.npy', y_clsaux=None, y_clsaux_onpremise=None, y_clsaux_substra=None, y_regr='reg/reg_T10_y.npz', y_regr_onpremise='reg/pred-regr.npy', y_regr_substra='reg/pred-regr-copy.npy')
[INFO]: Wrote input params to 'derisk_test/run_params.json'

=======================================================================================================================================
=======================================================================================================================================
De-risking cls performance
=======================================================================================================================================
=======================================================================================================================================

[INFO]: Loading cls: cls/cls_T10_y.npz
[INFO]: Loading (npy) predictions for: cls/pred-class.npy
[INFO]: Loading (npy) predictions for: cls/pred-class-copy.npy

=======================================================================================================================================
[DERISK-CHECK #1]: PASSED! yhats close between 'pred-class' and 'pred-class-copy' (tol:1e-05)
Spearmanr rank correlation coefficient of the 'pred-class' and 'pred-class-copy' yhats = SpearmanrResult(correlation=1.0, pvalue=0.0)
=======================================================================================================================================

[INFO]: === Calculating cls/pred-class.npy performance ===
[INFO]: Wrote per-task report to: derisk_test/cls/pred-class/pred-class_per-task_performances.csv
[INFO]: Wrote per-task binned performance report to: derisk_test/cls/pred-class/pred-class_binned_per-task_performances.csv
[INFO]: Wrote per-assay report to: derisk_test/cls/pred-class/pred-class_per-assay_performances.csv
[INFO]: Wrote per-assay binned report to: derisk_test/cls/pred-class/pred-class_binned_per-task_performances.csv
[INFO]: Wrote global report to: derisk_test/cls/pred-class/pred-class_global_performances.csv

[INFO]: === Calculating cls/pred-class-copy.npy performance ===
[INFO]: Wrote per-task report to: derisk_test/cls/pred-class-copy/pred-class-copy_per-task_performances.csv
[INFO]: Wrote per-task binned performance report to: derisk_test/cls/pred-class-copy/pred-class-copy_binned_per-task_performances.csv
[INFO]: Wrote per-assay report to: derisk_test/cls/pred-class-copy/pred-class-copy_per-assay_performances.csv
[INFO]: Wrote per-assay binned report to: derisk_test/cls/pred-class-copy/pred-class-copy_binned_per-task_performances.csv

=======================================================================================================================================
[DERISK-CHECK #2]: FAILED! calculated metrics for one or more individual tasks differ to reported performances (tol:1e-05) 			
['roc_auc_score', 'auc_pr', 'avg_prec_score', 'p_f1_max', 'kappa', 'p_kappa_max', 'bceloss'] are the reported metrics with different performances 			
['f1_max', 'kappa_max'] are identical 			
Check the output of derisk_test/cls/pred-class-copy/pred-class-copy_closeto_reported_performances.csv for details
=======================================================================================================================================

[INFO]: Wrote reported vs. calculated performance delta to: derisk_test/cls/pred-class-copy_delta-reported_performances.csv
[INFO]: Wrote global report to: derisk_test/cls/pred-class-copy/pred-class-copy_global_performances.csv

=======================================================================================================================================
[DERISK-CHECK #3]: FAILED! global reported performance metrics and global calculated performance metrics NOT close (tol:1e-05) 							
Calculated:   roc_auc_score    auc_pr  avg_prec_score    f1_max  p_f1_max  kappa  kappa_max  p_kappa_max   bceloss
0       0.691302  0.597414        0.604776  0.638678  0.584623    0.0   0.339309     0.601179  0.714289
Reported:   roc_auc_score    auc_pr  avg_prec_score    f1_max  p_f1_max     kappa  kappa_max  p_kappa_max   bceloss
0       0.689219  0.591262        0.603295  0.646287  0.369019  0.106348   0.354664      0.41959  0.526552
=======================================================================================================================================

=======================================================================================================================================
[DERISK-CHECK #4]: PASSED! delta between local & substra assay_type aggregated performances close to 0 across all metrics (tol:1e-05)
=======================================================================================================================================

[INFO]: Wrote per-task delta report to: derisk_test/cls/deltas/deltas_per-task_performances.csv
[INFO]: Wrote binned performance per-task delta report to: derisk_test/cls/deltas/deltas_binned_per-task_performances.csv
[INFO]: Wrote per-assay delta report to: derisk_test/cls/deltas/deltas_per-assay_performances.csv
[INFO]: Wrote binned performance per-assay delta report to: derisk_test/cls/deltas/deltas_binned_per-task_performances.csv

=======================================================================================================================================
[DERISK-CHECK #5]: PASSED! delta performance between global local & global substra performances close to 0 across all metrics (tol:1e-05)
=======================================================================================================================================

=======================================================================================================================================
=======================================================================================================================================
De-risking regr performance
=======================================================================================================================================
=======================================================================================================================================

[INFO]: Loading regr: reg/reg_T10_y.npz
[INFO]: Loading (npy) predictions for: reg/pred-regr.npy
[INFO]: Loading (npy) predictions for: reg/pred-regr-copy.npy

=======================================================================================================================================
[DERISK-CHECK #1]: PASSED! yhats close between 'pred-regr' and 'pred-regr-copy' (tol:1e-05)
Spearmanr rank correlation coefficient of the 'pred-regr' and 'pred-regr-copy' yhats = SpearmanrResult(correlation=0.9999999999999998, pvalue=0.0)
=======================================================================================================================================

[INFO]: === Calculating reg/pred-regr.npy performance ===
[INFO]: Wrote per-task report to: derisk_test/regr/pred-regr/pred-regr_per-task_performances.csv
[INFO]: Wrote per-task binned performance report to: derisk_test/regr/pred-regr/pred-regr_binned_per-task_performances.csv
[INFO]: Wrote per-assay report to: derisk_test/regr/pred-regr/pred-regr_per-assay_performances.csv
[INFO]: Wrote per-assay binned report to: derisk_test/regr/pred-regr/pred-regr_binned_per-task_performances.csv
[INFO]: Wrote global report to: derisk_test/regr/pred-regr/pred-regr_global_performances.csv

[INFO]: === Calculating reg/pred-regr-copy.npy performance ===
[INFO]: Wrote per-task report to: derisk_test/regr/pred-regr-copy/pred-regr-copy_per-task_performances.csv
[INFO]: Wrote per-task binned performance report to: derisk_test/regr/pred-regr-copy/pred-regr-copy_binned_per-task_performances.csv
[INFO]: Wrote per-assay report to: derisk_test/regr/pred-regr-copy/pred-regr-copy_per-assay_performances.csv
[INFO]: Wrote per-assay binned report to: derisk_test/regr/pred-regr-copy/pred-regr-copy_binned_per-task_performances.csv

=======================================================================================================================================
[DERISK-CHECK #2]: PASSED! calculated and reported metrics are the same across individual tasks 			
['rmse', 'rmse_uncen', 'rsquared', 'corrcoef'] are identical
=======================================================================================================================================

[INFO]: Wrote reported vs. calculated performance delta to: derisk_test/regr/pred-regr-copy_delta-reported_performances.csv
[INFO]: Wrote global report to: derisk_test/regr/pred-regr-copy/pred-regr-copy_global_performances.csv

=======================================================================================================================================
[DERISK-CHECK #3]: FAILED! global reported performance metrics and global calculated performance metrics NOT close (tol:1e-05) 							
Calculated:           rmse    rmse_uncen  rsquared  corrcoef
0  2.107307e+08  2.107307e+08 -4.838897  0.056098
Reported:           rmse    rmse_uncen  rsquared  corrcoef
0  1.040116e+08  1.040116e+08 -6.438911  0.081194
=======================================================================================================================================

=======================================================================================================================================
[DERISK-CHECK #4]: PASSED! delta between local & substra assay_type aggregated performances close to 0 across all metrics (tol:1e-05)
=======================================================================================================================================

[INFO]: Wrote per-task delta report to: derisk_test/regr/deltas/deltas_per-task_performances.csv
[INFO]: Wrote binned performance per-task delta report to: derisk_test/regr/deltas/deltas_binned_per-task_performances.csv
[INFO]: Wrote per-assay delta report to: derisk_test/regr/deltas/deltas_per-assay_performances.csv
[INFO]: Wrote binned performance per-assay delta report to: derisk_test/regr/deltas/deltas_binned_per-task_performances.csv

=======================================================================================================================================
[DERISK-CHECK #5]: PASSED! delta performance between global local & global substra performances close to 0 across all metrics (tol:1e-05)
=======================================================================================================================================

[INFO]: Run name 'derisk_test' is finished.
[INFO]: Performance evaluation de-risk took 73.012389 seconds.
```

The following files are created, e.g.:
```
  derisk_test #name of the run (timestamp used if not defined)
  ├── <cls/reg>
  │   ├── deltas
  │   │   ├── deltas_binned_per-assay_performances.csv	#binned assay aggregated deltas between pred-class & pred-class-copy
  │   │   ├── deltas_binned_per-task_performances.csv	#binned deltas across all tasks between pred-class & pred-class-copy
  │   │   ├── deltas_global_performances.csv	#global aggregated deltas between pred-class & pred-class-copy
  │   │   ├── deltas_per-assay_performances.csv	#assay aggregated deltas between pred-class & pred-class-copy
  │   │   └── deltas_per-task_performances.csv	#deltas between pred-class & pred-class-copy
  │   ├── pred-<class/regr> #e.g. the onprmise emulated predictions
  │   │   ├── pred-class_binned_per-assay_performances.csv	#binned pred-class assay aggregated performances
  │   │   ├── pred-class_binned_per-task_performances.csv	#binned pred-class performances
  │   │   ├── pred-class_global_performances.csv	#pred-class global performance
  │   │   ├── pred-class_per-assay_performances.csv	#pred-class assay aggregated performances
  │   │   └── pred-class_per-task_performances.csv	#pred-class performances
  │   └── pred-<class/regr>-copy #e.g. the substra emulated predictions
  │       ├── pred-class-copy_binned_per-assay_performances.csv	 #binned pred-class-copy assay aggregated performances
  │       ├── pred-class-copy_binned_per-task_performances.csv	#binned pred-class-copy performances
  │       ├── pred-class-copy_closeto_reported_performances.csv	#clarify when pred-class-copy is close to reported performances across tasks
  │       ├── pred-class-copy_global_performances.csv	#pred-class-copy global performance
  │       ├── pred-class-copy_per-assay_performances.csv	#pred-class-copy assay aggregated performances
  │       └── pred-class-copy_per-task_performances.csv	#pred-class-copy performance
  ├── derisk_summary.csv	#summary of which derisks pass/fail
  └── run_params.json #json with the runtime parameters
```

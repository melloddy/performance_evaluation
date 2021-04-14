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



## De-risk analysis (performance_evaluation_derisk.py): on-premise vs. substra output evaluation
```
$ python performance_evaluation.py -h
  
  optional arguments:
    -h, --help            show this help message and exit
    --y_cls Y_CLS         Classification activity file (npz) (e.g. cls_T10_y.npz)
    --y_clsaux Y_CLSAUX   Aux classification activity file (npz) (e.g. cls_T10_y.npz)
    --y_regr Y_REGR       Activity file (npz) (e.g. reg_T10_y.npz)
    --y_cls_onpremise Y_CLS_ONPREMISE
                          Yhat cls prediction output from an onpremise run (e.g. <single pharma dir>/<cls_prefix>-class.npy)
    --y_clsaux_onpremise Y_CLSAUX_ONPREMISE
                          Yhat clsaux prediction from an onpremise run (e.g. <single pharma dir>/<clsaux_prefix>-class.npy)
    --y_regr_onpremise Y_REGR_ONPREMISE
                          Yhat regr prediction from an onpremise run (e.g. <single pharma dir>/<regr_prefix>-regr.npy)
    --y_cls_substra Y_CLS_SUBSTRA
                          Classification prediction output for comparison (e.g. pred from the substra platform)
    --y_clsaux_substra Y_CLSAUX_SUBSTRA
                          Classification w/ aux prediction output for comparison (e.g. pred from the substra platform)
    --y_regr_substra Y_REGR_SUBSTRA
                          Regression prediction output for comparison (e.g. pred from the substra platform)
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
    --perf_json_cls PERF_JSON_CLS
                          Reported json performances for classification model (i.e. sc_run<cls-params>.json)
    --perf_json_clsaux PERF_JSON_CLSAUX
                          Reported json performances for aux classification model (i.e. sc_run<clsaux-params>.json)
    --perf_json_regr PERF_JSON_REGR
                          Reported json performances for regression model (i.e. sc_run<reg-params>.json)
    --run_name RUN_NAME   Run name directory for results from this output
    --verbose {0,1}       Verbosity level: 1 = Full; 0 = no output
    --validation_fold {0,1,2,3,4} [{0,1,2,3,4} ...]
                          Validation fold to used to calculate performance
    --aggr_binning_scheme_perf AGGR_BINNING_SCHEME_PERF [AGGR_BINNING_SCHEME_PERF ...]
                          (Comma separated) Shared aggregated binning scheme for performances
    --aggr_binning_scheme_perf_delta AGGR_BINNING_SCHEME_PERF_DELTA [AGGR_BINNING_SCHEME_PERF_DELTA ...]
                          (Comma separated) Shared aggregated binning scheme for delta performances
  
```


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
python performance_evaluation.py \ 
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

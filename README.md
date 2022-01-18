# Year 3 Performance Evaluation Script for the IMI Project MELLODDY

Performance evaluation scripts from the Single- and Multi-pharma outputs.
These evaluation scripts allow us to verify whether predictions performed using a model generated (referred to as single-pharma model hereafter) during a federated run can be reproduced when executed locally (prediction step) within the pharma partner's IT infrastructure. 
Second the evaluation assesses whether the predictions from the federated model improve over predictions from the single-pharma model.

## Requirements
On your local IT infrastructure you'd need 

1. Python 3.6 or higher
2. Local Conda installation (e.g. miniconda)
3. Data prep (3.0.2) and Sparsechem (suggested **0.9.5**) installation
4. melloddy_tuner environment from WP1 code: https://git.infra.melloddy.eu/wp1/data_prep & Sparsechem installed into that environment


## === De-risk analysis ===


## De-risk of 20 epoch performance derisk models (cls/reg)

### 1. De-risk of globally reported subtra performance output & onpremise calculated metrics

#### Step 1. Retrieve the 20 epoch cls & reg derisk output from substra and decompress
```
gunzip -c <cls-hash>.tar.gz | tar xvf - && gunzip -c <reg-hash>.tar.gz | tar xvf -
```

The performance JSONs are not named with their respective metrics. Use the following to create new JSONs with the name of each metric for each of the cls/reg derisk runs:

```
import json
from shutil import copyfile as cp
import glob


for eachrun in glob.glob('*/'):

	with open(f'{eachrun}/metadata.json') as json_data:
		metadata = json.load(json_data)
		metrics = metadata["metrics"]


	for m in metrics:
		metric = metadata["metrics"][m]["metadata"]["metrics_name"]
		cp(f"{eachrun}/perf/%s-perf.json" %m, f"{eachrun}/perf/%s-perf.json" %metric)
```


#### Step 2. Create SparseChem predictions on-premise with the outputted model

Predict the validation fold (PH1=4 or PH2=0) using SparseChem (SC) version 0.9.5 and the PH2 model output from substra (load your conda env), e.g.:

for cls:
```
python <sparsechem_dir>/examples/chembl/predict.py \
  --x <cls_dir>/cls_T11_x.npz \
  --y_class <cls_dir>/cls_T10_y.npz \
  --folding <cls_dir>/cls_T11_fold_vector.npy \
  --predict_fold 0 \
  --conf <2epoch_derisk_cls_dir>/export/hyperparameters.json \
  --model <2epoch_derisk_cls_dir>/export/model.pth \
  --dev cuda:0 \
  --outprefix "derisk_cls/pred"
```
and reg:
```
python <sparsechem_dir>/examples/chembl/predict.py \
  --x <reg_dir>/clsaux_T11_x.npz \
  --y_regr <reg_dir>/clsaux_T10_y.npz \
  --folding <reg_dir>/clsaux_T11_fold_vector.npy \
  --predict_fold 0 \
  --conf <2epoch_derisk_reg_dir>/export/hyperparameters.json \
  --model <2epoch_derisk_reg_dir>/export/model.pth \
  --dev cuda:0 \
  --inverse_normalization 1 \
  --outprefix "derisk_regr/pred"
```
  
#### Step 3. Run the derisk script
```
python development_performance_evaluation_derisk.py \
  --y_regr_substra <20epoch_derisk_regr_dir>/pred/pred.json \
  --y_regr_onpremise derisk_regr/pred-regr.npy \
  --t8r_regr regr/T8r.csv \
  --weights_regr regr/reg_weights.csv \
  --y_regr regr/reg_T10_y.npz \
  --folding_regr regr/reg_T11_fold_vector.npy \
  --perf_json_regr <20epoch_derisk_regr_dir>/perf/r_squared-perf.json \
  --validation_fold 0
  --run_name derisk_20epoch_regr

python development_performance_evaluation_derisk.py \
  --y_cls_substra <20epoch_derisk_cls_dir>/pred/pred.json \
  --y_cls_onpremise derisk_cls/predall-cls.npy \
  --t8c_cls cls/T8c.csv \
  --weights_cls cls/cls_weights.csv \
  --y_cls cls/cls_T10_y.npz \
  --folding_cls cls_T11_fold_vector.npy \
  --perf_json_cls <20epoch_derisk_cls_dir>/perf/aucpr-perf.json \
  --validation_fold 0
  --run_name derisk_20epoch_cls
```

or both cls and reg run together with:

```
python development_performance_evaluation_derisk.py \
  --y_regr_substra <20epoch_derisk_regr_dir>/pred/pred.json \
  --y_regr_onpremise derisk_regr/pred-regr.npy \
  --t8r_regr regr/T8r.csv \
  --weights_regr regr/reg_weights.csv \
  --y_regr regr/reg_T10_y.npz \
  --folding_regr regr/reg_T11_fold_vector.npy \
  --perf_json_regr <20epoch_derisk_regr_dir>/perf/r_squared-perf.json \
  --y_cls_substra <20epoch_derisk_cls_dir>/pred/pred.json \
  --y_cls_onpremise hybpredall-cls.npy \
  --t8c_cls cls/T8c.csv \
  --weights_cls cls/cls_weights.csv \
  --y_cls cls/cls_T10_y.npz \
  --folding_cls hyb_T11_fold_vector.npy \
  --perf_json_cls <20epoch_derisk_cls_dir>/perf/aucpr-perf.json \
  --validation_fold 0
  --run_name derisk_20epoch_cls_regr
```

Output should look something like this:
```
[INFO]: === WP3 Y3 Performance evaluation de-risk script for npy and pred files ===

[INFO]: Namespace{..}
[INFO]: Run name is 'derisk_20epoch_regr'

[INFO]: Wrote input params to 'derisk_20epoch_regr/run_params.json'

=======================================================================================================================================
=======================================================================================================================================
De-risking regr performance
=======================================================================================================================================
=======================================================================================================================================

[INFO]: Loading regr: reg_T10_y.npz
[INFO]: Loading (pred) predictions for: pred
[INFO]: Loading (npy) predictions for: derisk_regr/pred-regr.npy

=======================================================================================================================================
[DERISK-CHECK #1]: PASSED! yhats close between 'pred' and 'pred-regr.npy' (tol:1e-05)
Spearmanr rank correlation coefficient of the 'pred' and 'pred-regr.npy' yhats = SpearmanrResult(correlation=1.0, pvalue=0.0)
=======================================================================================================================================

[INFO]: === Calculating pred performance ===
[INFO]: Wrote per-task report to: derisk_20epoch_regr/regr/pred/pred_per-task_performances.csv
[INFO]: Wrote per-task binned performance report to: derisk_20epoch_regr/regr/pred/pred_binned_per-task_performances.csv
[INFO]: Wrote per-assay report to: derisk_20epoch_regr/regr/pred/pred_per-assay_performances.csv
[INFO]: Wrote per-assay binned report to: derisk_20epoch_regr/regr/pred/pred_binned_per-task_performances.csv
[INFO]: Wrote global report to: derisk_20epoch_regr/regr/pred/pred_global_performances.csv

[INFO]: === Calculating pred-regr.npy performance ===
[INFO]: Wrote per-task report to: derisk_20epoch_regr/regr/pred-regr/pred-regr_per-task_performances.csv
[INFO]: Wrote per-task binned performance report to: derisk_20epoch_regr/regr/pred-regr/pred-regr_binned_per-task_performances.csv
[INFO]: Wrote per-assay report to: derisk_20epoch_regr/regr/pred-regr/pred-regr_per-assay_performances.csv
[INFO]: Wrote per-assay binned report to: derisk_20epoch_regr/regr/pred-regr/pred-regr_binned_per-task_performances.csv
[INFO]: Wrote global report to: derisk_20epoch_regr/regr/pred-regr/pred-regr_global_performances.csv

=======================================================================================================================================
[DERISK-CHECK #2]: SKIPPED! substra does not report individual task performances
=======================================================================================================================================

[INFO]: Wrote reported vs. calculated performance delta to: derisk_20epoch_regr/regr/pred_delta-reported_performances.csv
[INFO]: Wrote global report to: derisk_20epoch_regr/regr/pred-regr/pred-regr_global_performances.csv

=======================================================================================================================================
[DERISK-CHECK #3]: PASSED! global reported performance metrics and global calculated performance metrics close (tol:1e-05) 					
=======================================================================================================================================

[INFO]: Wrote per-task delta report to: derisk_20epoch_regr/regr/deltas/deltas_per-task_performances.csv
[INFO]: Wrote binned performance per-task delta report to: derisk_20epoch_regr/regr/deltas/deltas_binned_per-task_performances.csv

=======================================================================================================================================
[DERISK-CHECK #4]: PASSED! delta between local calculated assay_type aggregated & substra performances close to 0 across all metrics (tol:0.001)
Reported:
                   rmse  rmse_uncen  rsquared  corrcoef
assay_type                                             
ADME                0.0         0.0       0.0       0.0
NON-CATALOG-PANEL   0.0         0.0       0.0       0.0
OTHER               0.0         0.0       0.0       0.0
=======================================================================================================================================

[INFO]: Wrote per-assay delta report to: derisk_20epoch_regr/regr/deltas/deltas_per-assay_performances.csv
[INFO]: Wrote binned performance per-assay delta report to:  derisk_20epoch_regr/regr/deltas/deltas_binned_per-task_performances.csv

=======================================================================================================================================
[DERISK-CHECK #5]: PASSED! delta performance between calculated global local & global substra performances close to 0 across all metrics (tol:1e-05)
Reported:
   rmse  rmse_uncen  rsquared  corrcoef
0   0.0         0.0       0.0       0.0
=======================================================================================================================================

[INFO]: Run name ' derisk_20epoch_regr' is finished.

[INFO]: Performance evaluation de-risk took 93.612643 seconds.
```

The file structure should look like:
```
derisk_test #name of the run (timestamp used if not defined)
  ├── <cls/reg>
  │   ├── deltas
  │   │   ├── deltas_binned_per-assay_performances.csv	#binned assay aggregated deltas between pred-onpremise & pred-substra
  │   │   ├── deltas_binned_per-task_performances.csv	#binned deltas across all tasks between pred-onpremise & pred-substra
  │   │   ├── deltas_global_performances.csv	#global aggregated deltas between pred-onpremise & pred-substra
  │   │   ├── deltas_per-assay_performances.csv	#assay aggregated deltas between pred-onpremise & pred-substra
  │   │   └── deltas_per-task_performances.csv	#deltas between pred-onpremise & pred-substra
  │   ├── pred-<class/regr> #e.g. the onprmise predictions
  │   │   ├── pred-onpremise_binned_per-assay_performances.csv	#binned pred-onpremise assay aggregated performances
  │   │   ├── pred-onpremise_binned_per-task_performances.csv	#binned pred-onpremise performances
  │   │   ├── pred-onpremise_global_performances.csv	#pred-onpremise global performance
  │   │   ├── pred-onpremise_per-assay_performances.csv	#pred-onpremise assay aggregated performances
  │   │   └── pred-onpremise_per-task_performances.csv	#pred-onpremise performances
  │   └── pred-<class/regr>-copy #e.g. the substra predictions
  │       ├── pred-substra_binned_per-assay_performances.csv	 #binned pred-substra assay aggregated performances
  │       ├── pred-substra_binned_per-task_performances.csv	#binned pred-substra performances
  │       ├── pred-substra_closeto_reported_performances.csv	#clarify when pred-substra is close to reported performances across tasks
  │       ├── pred-substra_global_performances.csv	#pred-substra global performance
  │       ├── pred-substra_per-assay_performances.csv	#pred-substra assay aggregated performances
  │       └── pred-substra_per-task_performances.csv	#pred-substra performance
  ├── derisk_summary.csv	#summary of which derisks pass/fail - TO BE UPLOADED TO BOX!
  └── run_params.json #json with the runtime parameters
```

The file 'derisk_summary.csv' has a concise summary of de-risk checks

#### Step 3. Report the derisk

Report output here: https://jjcloud.box.com/s/8vgkicq6ky0vx1sy83a3b6ljge7z7nuu


### 2. Delta between on-premise and substra for identically trained models

#### Step 1. Train a local model with the exact same hyperparameters as it was trianed on the platform, e.g.:
```
python $train \
  --x $data_path/cls/cls_T11_x.npz \
  --y $data_path/cls/cls_T10_y.npz \
  --folding $data_path/cls/cls_T11_fold_vector.npy \
  --weights_class $data_path/cls/cls_weights.csv \
  --hidden_sizes $hidden_sizes \
  --weight_decay $weight_decay \
  --dropouts_trunk $middle_dropout \
  --last_non_linearity relu \
  --non_linearity relu \
  --input_transform binarize \
  --lr $lr \
  --lr_alpha $lra \
  --lr_steps $lrs \
  --epochs 20 \
  --normalize_loss 100_000 \
  --eval_frequency 1 \
  --batch_ratio 0.02 \
  --fold_va 0 \
  --verbose 1 \
  --profile 1 \
  --save_model 1
```
#### Step 2. check the performance here:
```
import sparsechem as sc
res = sc.load_results('models/sc_run_h{..}_ldo{..}_wd{..}_lr{..}_lrsteps{..}_ep20_fva0_fteNone.json')
print(res['validation']['classification_agg'])   
```

#### Step 3. Run the derisk workflow as usual and you will get some files in the output folder like:

```
cat derisk_20epoch_cls/cls/pred/pred_global_performances.csv
```

The aucpr values needs to be compared for cls, or coeff for reg 

Report output here: https://jjcloud.box.com/s/8vgkicq6ky0vx1sy83a3b6ljge7z7nuu


## CLI

```

MELLODDY Year 3 Performance Evaluation De-risk

optional arguments:
  -h, --help            show this help message and exit
  --y_cls Y_CLS         Classification activity file (npz) (e.g.
                        cls_T10_y.npz)
  --y_clsaux Y_CLSAUX   Aux classification activity file (npz) (e.g.
                        cls_T10_y.npz)
  --y_regr Y_REGR       Activity file (npz) (e.g. reg_T10_y.npz)
  --y_censored_regr Y_CENSORED_REGR
                        Censored activity file (npz) (e.g.
                        reg_T10_censor_y.npz)
  --y_hyb_cls Y_HYB_CLS
                        Activity file (npz) (e.g. hyb_cls_T10_y.npz)
  --y_hyb_regr Y_HYB_REGR
                        Activity file (npz) (e.g. hyb_reg_T10_y.npz)
  --y_censored_hyb Y_CENSORED_HYB
                        Censored activity file (npz) (e.g.
                        hyb_reg_T10_censor_y.npz)
  --y_cls_onpremise Y_CLS_ONPREMISE
                        Yhat cls prediction output from an onpremise run (e.g.
                        <single pharma dir>/<cls_prefix>-class.npy)
  --y_clsaux_onpremise Y_CLSAUX_ONPREMISE
                        Yhat clsaux prediction from an onpremise run (e.g.
                        <single pharma dir>/<clsaux_prefix>-class.npy)
  --y_regr_onpremise Y_REGR_ONPREMISE
                        Yhat regr prediction from an onpremise run (e.g.
                        <single pharma dir>/<regr_prefix>-regr.npy)
  --y_hyb_cls_onpremise Y_HYB_CLS_ONPREMISE
                        Yhat hyb cls prediction from an onpremise run (e.g.
                        <single pharma dir>/<hyb_cls_prefix>-cls.npy)
  --y_hyb_regr_onpremise Y_HYB_REGR_ONPREMISE
                        Yhat hyb regr prediction from an onpremise run (e.g.
                        <single pharma dir>/<hyb_regr_prefix>-regr.npy)
  --y_cls_substra Y_CLS_SUBSTRA
                        Classification prediction output for comparison (e.g.
                        pred from the substra platform)
  --y_clsaux_substra Y_CLSAUX_SUBSTRA
                        Classification w/ aux prediction output for comparison
                        (e.g. pred from the substra platform)
  --y_regr_substra Y_REGR_SUBSTRA
                        Regression prediction output for comparison (e.g. pred
                        from the substra platform)
  --y_hyb_cls_substra Y_HYB_CLS_SUBSTRA
                        Yhat hyb cls prediction output for comparison (e.g.
                        <single pharma dir>/<hyb_cls_prefix>-cls.npy)
  --y_hyb_regr_substra Y_HYB_REGR_SUBSTRA
                        Yhat hyb regr prediction output for comparison (e.g.
                        <single pharma dir>/<hyb_regr_prefix>-regr.npy)
  --folding_cls FOLDING_CLS
                        Folding file (npy) (e.g. cls_T11_fold_vector.npy)
  --folding_clsaux FOLDING_CLSAUX
                        Folding file (npy) (e.g. cls_T11_fold_vector.npy)
  --folding_regr FOLDING_REGR
                        Folding file (npy) (e.g. reg_T11_fold_vector.npy)
  --folding_hyb FOLDING_HYB
                        Folding file (npy) (e.g. hyb_T11_fold_vector.npy)
  --t8c_cls T8C_CLS     T8c file for classification in the
                        results_tmp/classification folder
  --t8c_clsaux T8C_CLSAUX
                        T8c file for classification w/ auxiliary in the
                        results_tmp/classification folder
  --t8r_regr T8R_REGR   T8r file for regression in the results_tmp/regression
                        folder
  --weights_cls WEIGHTS_CLS
                        CSV file with columns task_id and weight (e.g.
                        cls_weights.csv)
  --weights_clsaux WEIGHTS_CLSAUX
                        CSV file with columns task_id and weight (e.g
                        cls_weights.csv)
  --weights_regr WEIGHTS_REGR
                        CSV file with columns task_id and weight (e.g.
                        reg_weights.csv)
  --weights_hyb_cls WEIGHTS_HYB_CLS
                        CSV file with columns task_id and weight (e.g.
                        hyb_cls_weights.csv)
  --weights_hyb_regr WEIGHTS_HYB_REGR
                        CSV file with columns task_id and weight (e.g.
                        hyb_reg_weights.csv)
  --perf_json_cls PERF_JSON_CLS
                        Reported json performances for classification model
                        (i.e. sc_run<cls-params>.json)
  --perf_json_clsaux PERF_JSON_CLSAUX
                        Reported json performances for aux classification
                        model (i.e. sc_run<clsaux-params>.json)
  --perf_json_regr PERF_JSON_REGR
                        Reported json performances for regression model (i.e.
                        sc_run<reg-params>.json)
  --perf_json_hyb_cls PERF_JSON_HYB_CLS
                        Reported json performances for regression model (i.e.
                        sc_run<reg-params>.json)
  --perf_json_hyb_regr PERF_JSON_HYB_REGR
                        Reported json performances for regression model (i.e.
                        sc_run<reg-params>.json)
  --run_name RUN_NAME   Run name directory for results from this output
                        (timestemp used if not specified)
  --verbose {0,1}       Verbosity level: 1 = Full; 0 = no output
  --validation_fold {0,1,2,3,4} [{0,1,2,3,4} ...]
                        Validation fold to used to calculate performance
  --aggr_binning_scheme_perf AGGR_BINNING_SCHEME_PERF [AGGR_BINNING_SCHEME_PERF ...]
                        Shared aggregated binning scheme for performances
  --aggr_binning_scheme_perf_delta AGGR_BINNING_SCHEME_PERF_DELTA [AGGR_BINNING_SCHEME_PERF_DELTA ...]
                        Shared aggregated binning scheme for delta
                        performances

```

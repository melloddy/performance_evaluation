# Year 3 Performance Evaluation Script for the IMI Project MELLODDY

Performance evaluation scripts from the Single- and Multi-pharma outputs.
These evaluation scripts allow us to verify whether predictions performed using a model generated (referred to as single-pharma model hereafter) during a federated run can be reproduced when executed locally (prediction step) within the pharma partner's IT infrastructure. 
Second the evaluation assesses whether the predictions from the federated model improve over predictions from the single-pharma model.

## Requirements
On your local IT infrastructure you'd need 

1. Python 3.6 or higher
2. Local Conda installation (e.g. miniconda)
3. Data prep (3.0.2) and Sparsechem (suggested **0.9.5** when using platform derived models, but will also work with **0.9.6** for local runs) installation
4. melloddy_tuner environment from WP1 code: https://git.infra.melloddy.eu/wp1/data_prep & Sparsechem installed into that environment
5. Statsmodels is required for the ECDF functionaility, e.g. via `conda install statsmodels==0.13.2`

## Instructions for the main (primary performance reporting) Y3 performance evaluation report

The following instructions are required to complete the task: https://melloddy.monday.com/boards/259897343/pulses/2592726539 

### Step 1. Run the model selection code

First, determine the correct compute plans (CPs) to use for the evaluation report.

#### 1.1. Extract the model_selection scripts

* Extract the scripts located here:https://git.infra.melloddy.eu/wp3/performance_evaluation/-/tree/year3/model_selection) to the directory contining all compute plans (or at least all *MP* plans)
* Run the bash scripts

#### 1.2. Locate optimal PH2 MP CPs for this analysis

* Consult the *opt* output files, which can be used to obtain the optimal PH1-3 MP CPs for each of the CLS/HYB/REG/CLSAUX models.
* *N.B: we are only interested in PH2 for this specific task*

#### 1.3. Decompress the MP models

* For each of the models, extract the contents of the MP *PH2* tarballs and ensure these are easily accessible for step 2 (the perf evaluation) 

#### 1.4. Prepare SP models

* Identify and locate the optimal *SP* models for the CLS/REG/HYB/CLSAUX models as per the SP_model selection procedure here: https://git.infra.melloddy.eu/wp3/performance_evaluation/-/tree/year3/SP_model_selection#performance-evaluation
* The PH2 SparseChem models should be trained to the optimal epochs as would be available from platform - i.e. 5,10,15 ... etc)

### Step 2. Run the performance evaluation code

In the following sections we will run the performance evaluation script (https://git.infra.melloddy.eu/wp3/performance_evaluation/-/blob/year3/performance_evaluation.py) for each of the *CLS/REG/CLSAUX* models and the *HYB* models identified in Step 1 above. 

Run the performance evaluation script on six combinations of SP and MP models:
1. SP CLS vs MP CLS
2. SP CLSAUX vs MP CLSAUX
3. Optimal Classification model: SP opt(CLS,CLSAUX) vs MP opt(CLS,CLSAUX)
4. SP REG vs MP REG
5. SP HYB vs MP HYB
6. Optimal Regression model: SP opt(REG,HYB) vs MP opt(REG,HYB)

Note: SP opt(CLS,CLSAUX) would be the overall best SP model for classification out of all available CLS and CLSAUX models.

Intstructions in more detail:

#### Step 2.1. SP vs. MP CLS/REG/CLSAUX model evaluation

Run the performance evaluation script for each of the optimal CLS/REG/CLSAUX models like so:

```
python performance_evaluation.py \
 --y_cls <cls_dir>/cls_T10_y.npz \
 --y_cls_multi_partner <y_cls_multi_partner-hash>/pred/pred.json  \
 --y_cls_single_partner <y_cls_single_partner-hash>/pred/pred.json  \
 --t8c_cls <cls_dir>/T8c.csv \
 --weights_cls <cls_dir>/cls_weights.csv \
 --folding_cls <cls_dir>/cls_T11_fold_vector.npy \
 --y_regr <regr_dir>/reg_T10_y.npz \
 --y_censored_regr <regr_dir>/reg_T10_censor_y.npz.npz \
 --y_regr_multi_partner <y_regr_multi_partner-hash>/pred/pred.json  \
 --y_regr_single_partner <y_regr_single_partner-hash>/pred/pred.json  \
 --t8r_regr <regr_dir>/T8r.csv \
 --weights_regr <regr_dir>/reg_weights.csv \
 --folding_regr <regr_dir>/reg_T11_fold_vector.npy \
 --y_clsaux <clsaux_dir>/clsaux_T10_y.npz \
 --y_clsaux_multi_partner <y_clsaux_multi_partner-hash>/pred/pred.json  \
 --y_clsaux_single_partner <y_clsaux_single_partner-hash>/pred/pred.json  \
 --t8c_clsaux <clsaux_dir>/T8c.csv \
 --weights_clsaux <clsaux_dir>/clsaux_weights.csv \
 --folding_clsaux <clsaux_dir>/clsaux_T11_fold_vector.npy \
 --validation_fold 0 \
 --use_venn_abers \
 --output_task_sensitive_files 0 \
 --run_name sp_vs_mp__optimal_cls_clsaux_reg
```

#### Step 2.2. SP vs. MP HYB model evaluation

* NB: The HYB command line arguments are only useful for comparing MP pred.json files. 
* I.e. when comparing HYB models with locally generated predictions (as required for this task) the following steps MUST be follwed.

##### Step 2.2.a Prepare MP HYB pred.json preds for evaluation (split the MP pred.json first)

We are comparing MP models to SP, so the MP Hybrid models need to be split into CLS/REG portions (we are only interested in the REG portion), using the following code https://(git.infra.melloddy.eu/wp3/performance_evaluation/-/issues/27), thanks to Noe:

```
import pandas as pd
import numpy as np
import torch

# load the pred.json from a hybrid model 
Yhat_hyb = torch.load("/path/to/pred.json").tocsc()

# load cls and reg task weights
cls_tw = pd.read_csv("/path/to/matrices/cls/cls_weights.csv")
reg_tw = pd.read_csv("/path/to/matrices/reg/reg_weights.csv")

cls_tasks = cls_tw['task_id'].values
reg_tasks = reg_tw['task_id'].values + np.max(cls_tasks) + 1
            
Yhat_reg = Yhat_hyb[:, reg_tasks]
Yhat_cls = Yhat_hyb[:, cls_tasks]

np.save("pred_cls.npy", Yhat_cls)
np.save("pred_reg.npy", Yhat_reg)
```

Note the location of the new *MP* pred_reg.npy which will provided to the performance evaluation code in *Step 2.2.c*

##### Step 2.2.b Generate local SP HYB predictions for evaluation:

The following should be used to generate the SP prediction, e.g (in a SparseChem 0.9.6 environment):

```
python $predict \
  --x $data_path/hyb/hyb_T11_x.npz \
  --y_regr $data_path/hyb/hyb_reg_T10_y.npz \
  --inverse_normalization 1 \
  --conf models/hyb_local_phase2.json \
  --model models/hyb_local_phase2.pt \
  --dev cpu \
  --outprefix yhats_ph2_fva0
```

This produces the following files:

```
pred-class.npy
pred-reg.npy ### this file only
```

Note the location of the new *SP* pred-reg.npy which will provided to the performance evaluation code in *Step 2.2.c*


##### Step 2.2.c. Run the performance evaluation script on the HYB-REG models like so:

```
python performance_evaluation.py \
 --y_regr <regr_dir>/reg_T10_y.npz \ 
 --y_censored_regr <regr_dir>/reg_T10_censor_y.npz.npz \
 --y_regr_multi_partner <mp_dir_output_from_step2a>/pred_reg.npy  \
 --y_regr_single_partner <sp_dir_output_from_step2b>/pred_reg.npy  \
 --t8r_regr <regr_dir>/T8r.csv \
 --weights_regr <reg_dir>/reg_weights.csv \
 --folding_regr <reg_dir>/reg_T11_fold_vector.npy \
 --validation_fold 0 \
 --output_task_sensitive_files 0 \
 --run_name sp_vs_mp__optimal_hyb
```

### Step 3. Report status

#### Step 3.1. Report the output to Box: https://jjcloud.box.com/s/usumy2oedamyu5xbcengmqc7eds5retk (note different to initial testing Box folder)

* Now we need to upload the output (*minus the files revealing task level information*) to Box
* N.B: Running the latest code with ```--output_task_sensitive_files 0``` (also the default setting) means that no sensitive performance evaluation files are produced -> all contents of delta folder can be uploaded
* If you used ```--output_task_sensitive_files 1```, you need to ensure to only upload the following files to box:  

```
deltas_cdfMP-cdfSP.csv
deltas_cdfMP-cdfSP_assay_type.csv
delta_best_100_assays.csv
deltas_global_cdf.csv
deltas_per-assay_cdf.csv
deltas_binned_per-task_performances.csv
deltas_per-assay_performances.csv
deltas_binned_per-assay_performances.csv
deltas_global_performances.csv
tasks_perf_bin_flipped.csv
```

• I.e.: take care to *NOT* upload the *_NOUPLOAD.csv files: 

```
deltas_per-task_performances_NOUPLOAD.csv
tasks_perf_bin_count_NOUPLOAD.csv
pred_per-task_performances_NOUPLOAD.csv
```


#### Step 3.2. Update Monday.com task here: https://melloddy.monday.com/boards/259897343/pulses/2592726539



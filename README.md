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

## Instructions for the main (primary performance reporting) Y3 performance evaluation report

### Step 1. Run the model selection code

First, determine the correct compute plans (CPs) to use for the evaluation report.

1. Extract the model_selection scripts (https://git.infra.melloddy.eu/wp3/performance_evaluation/-/tree/year3/model_selection) to the directory contining all *MP* compute plans and run the bash scripts.

2. Consult the *opt* output files, to obtain the optimal PH1-3 MP CPs for each of the CLS/HYB/REG/CLSAUX models

3. For each of the models, extract the contents of the MP PH2 tarballs and ensure these are easily accessible for step 2 (the perf evaluation) 

4. Identify and locate the optimal *SP* models for the CLS/REG/HYB/CLSAUX models as per the SP_model selection procedure here: https://git.infra.melloddy.eu/wp3/performance_evaluation/-/tree/year3/SP_model_selection#performance-evaluation (PH2 SparseChem models trained to the optimal epochs as would be available from platform - i.e. 5,10,15 ... etc)

### Step 2. Run the performance evaluation code

Run the performance evaluation script (https://git.infra.melloddy.eu/wp3/performance_evaluation/-/blob/year3/performance_evaluation.py) for each of the CLS/HYB/REG/CLSAUX models identified in Step 1 above.

#### Step 2.1. CLS/REG/CLSAUX model evaluation

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
 --folding_regr <regr_dir>/reg_T11_fold_vector.npy.npy \
 --y_clsaux <clsaux_dir>/clsaux_T10_y.npz \
 --y_clsaux_multi_partner <y_clsaux_multi_partner-hash>/pred/pred.json  \
 --y_clsaux_single_partner <y_clsaux_single_partner-hash>/pred/pred.json  \
 --t8c_clsaux <clsaux_dir>/T8c.csv \
 --weights_clsaux <clsaux_dir>/clsaux_weights.csv \
 --folding_clsaux <clsaux_dir>/clsaux_T11_fold_vector.npy \
 --validation_fold 0 \
 --run_name sp_vs_mp__optimal_cls_clsaux_reg
```

#### Step 2.2a. HYB model evaluation (split the pred.json first)

Hybrid models need to be split into CLS/REG portions (we are only interested in the REG portion), using the following code https://(git.infra.melloddy.eu/wp3/performance_evaluation/-/issues/27), thanks to Noe:

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


#### Step 2.2b. Run the performance evaluation script on the HYB-REG models like so:

```
python performance_evaluation.py \
 --y_regr <regr_dir>/reg_T10_y.npz \ 
 --y_censored_regr <regr_dir>/reg_T10_censor_y.npz.npz \
 --y_regr_multi_partner <mp_dir_output_from_step2a>/pred_reg.npy  \
 --y_regr_single_partner <sp_dir_output_from_step2a>/pred_reg.npy  \
 --t8r_regr <regr_dir>/T8r.csv \
 --weights_regr <reg_dir>/reg_weights.csv \
 --folding_regr <reg_dir>/reg_T11_fold_vector.npy \
 --validation_fold 0 \
 --run_name sp_vs_mp__optimal_hyb
```

### Step 3. Report the output and update Monday.com

#### Step 3.1. Report the output to box: https://az.app.box.com/folder/160809892275

NB: take care to not upload the files: 

```
deltas_per-task_performances.csv
pred_per-task_performances.csv
```

or any *per-task* RUN1/RUN2 files that may contain your performance on a per-task level. 

#### Step 3.2. Update Monday.com task here: https://melloddy.monday.com/boards/259897343/pulses/2592726539



# Year 2 Performance Evaluation Script for the IMI Project MELLODDY

Performance evaluation scripts from the Single- and Multi-pharma outputs.
These evaluation scripts allow us to verify whether predictions performed using a model generated (referred to as single-pharma model hereafter) during a federated run can be reproduced when executed locally (prediction step) within the pharma partner's IT infrastructure. 
Second the evaluation assesses whether the predictions from the federated model improve over predictions from the single-pharma model.

## Requirements
On your local IT infrastructure you'd need 

1. Python 3.6 or higher
2. Local Conda installation (e.g. miniconda)
3. Data prep and Sparsechem (suggested >= 0.8.2) installation
4. melloddy_pipeline environment from WP1 code: https://git.infra.melloddy.eu/wp1/data_prep


Alternatively you can install the combined enrionment in environment_melloddy_combined.yml using `conda env create -f development/environment_melloddy_combined.yml`

# Example 1: De-risk analysis (on-premise vs. single-partner substra output evaluation)

## Build onpremise model (with your local sparsechem)
1. Train a model with sparsechem using the same input data as for the run to de-risk (sparsechem/examples/chembl/train.py)
2. Choose the same hyperparameters as the model to de-risk from the federated system:

## De-risk script (performance_evaluation_derisk.py)

```
python performance_evaluation.py -h
usage: performance_evaluation.py [-h]
				[--y_cls Y_CLS]
				[--y_clsaux Y_CLSAUX]
				[--y_regr Y_REGR]
				[--y_cls_onpremise Y_CLS_ONPREMISE]
				[--y_clsaux_onpremise Y_CLSAUX_ONPREMISE]
				[--y_regr_onpremise Y_REGR_ONPREMISE]
				[--y_cls_substra Y_CLS_SUBSTRA]
				[--y_clsaux_substra Y_CLSAUX_SUBSTRA]
				[--y_regr_substra Y_REGR_SUBSTRA]
				[--folding_cls FOLDING_CLS]
				[--folding_clsaux FOLDING_CLSAUX]
				[--folding_regr FOLDING_REGR]
				[--t8c_cls T8C_CLS]
				[--t8c_clsaux T8C_CLSAUX]
				[--t8r_regr T8R_REGR]
				[--weights_cls WEIGHTS_CLS]
				[--weights_clsaux WEIGHTS_CLSAUX]
				[--weights_regr WEIGHTS_REGR]
				[--perf_json_cls PERF_JSON_CLS]
				[--perf_json_clsaux PERF_JSON_CLSAUX]
				[--perf_json_regr PERF_JSON_REGR]
				[--filename FILENAME]
				[--verbose {0,1}]
				[--validation_fold {0,1,2,3,4} [{0,1,2,3,4} ...]]
				[--aggr_binning_scheme_perf AGGR_BINNING_SCHEME_PERF [AGGR_BINNING_SCHEME_PERF ...]]
				[--aggr_binning_scheme_perf_delta AGGR_BINNING_SCHEME_PERF_DELTA [AGGR_BINNING_SCHEME_PERF_DELTA ...]]

MELLODDY Year 2 Performance Evaluation De-risk


```
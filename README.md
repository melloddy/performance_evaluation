# Performance Evaluation Script for the IMI Project MELLODDY

Performance evaluation scripts from the Single- and Multi-pharma outputs

## Requirements

1. Python 3.6 or higher
2. Local Conda installation (e.g. miniconda)
3. Git installation
4. melloddy_tuner environment from WP1 code: https://git.infra.melloddy.eu/wp1/data_prep
5. sparsechem version 0.6.1: https://git.infra.melloddy.eu/wp2/sparsechem/-/tree/v0.6.1 (with sparse-predict functionality) installation from WP2 code. This is required to generate *sparse* on-premise predictions (not required to run the performance_evaluation[_derisk/pred].py code)


Alternatively you can install the combined enrionment in environment_melloddy_combined.yml using `conda env create -f development/environment_melloddy_combined.yml`

# Example 1: De-risk analysis (on-premise vs. single-partner substra output evaluation)

## Setup

1. Download the substra output
2. Use the model located in your Single_pharma_run/medias/subtuple/{pharma-hash}/export/single_model.pth to create on-premise *sparse* y-hat predictions for the in-house dataset. E.g.:

```python sparsechem/examples/chembl/predict.py --x x.npy --y y.npy --outfile onpremise_y_hat.npy --folding folding.npy --conf {pharma-hash}/export/hyperparameters.json --model {pharma-hash}/export/single_model_.pth --predict_fold 1```

3. Locate the Sinlge-pharma "pred" & "perf.json" from the Single_pharma_run/medias/subtuple/{pharma-hash}/pred/ folder 
4. Provide the script with the y-hat sparse prediction (onpremise_y_hat.npy) from step 2, the pred, perf.json and task_mapping file

## Analysis script (performance_evaluation_derisk.py)

```
python performance_evaluation_derisk.py -h
usage: performance_evaluation_derisk.py [-h] --y_true_all Y_TRUE_ALL
                                        --y_pred_onpremise Y_PRED_ONPREMISE
                                        --y_pred_substra Y_PRED_SUBSTRA
                                        --folding FOLDING
                                        --substra_performance_report
                                        SUBSTRA_PERFORMANCE_REPORT --task_map
                                        TASK_MAP [--filename FILENAME]
                                        [--use_venn_abers] [--verbose {0,1}]

Calculate Performance Metrics

optional arguments:
  -h, --help            show this help message and exit
  --y_true_all Y_TRUE_ALL
                        Activity file (npy) (i.e. from files_4_ml/)
  --y_pred_onpremise Y_PRED_ONPREMISE
                        Yhat prediction output from onpremise run (<single
                        pharma dir>/y_hat.npy)
  --y_pred_substra Y_PRED_SUBSTRA
                        Pred prediction output from substra platform
                        (./Single-pharma-
                        run/substra/medias/subtuple/<pharma_hash>/pred/pred)
  --folding FOLDING     LSH Folding file (npy) (i.e. from files_4_ml/)
  --substra_performance_report SUBSTRA_PERFORMANCE_REPORT
                        JSON file with global reported performance from
                        substra platform (i.e. ./Single-pharma-run/substra/med
                        ias/subtuple/<pharma_hash>/pred/perf.json)
  --task_map TASK_MAP   Taskmap from MELLODDY_tuner output of single run (i.e.
                        from results/weight_table_T3_mapped.csv)
  --filename FILENAME   Filename for results from this output
  --use_venn_abers      Toggle to turn on Venn-ABERs code
  --verbose {0,1}       Verbosity level: 1 = Full; 0 = no output


```

## Running the de-risk code
```
python performance_evaluation_derisk.py --y_true_all pharma_partners/pharma_y_partner_1.npy --y_pred_substra Single_pharma_run-1/medias/subtuple/c4f1c9b9d44fea66f9b856d346a0bb9aa5727e587185e87daca170f239a70029/pred/pred --folding pharma_partners/folding_partner_1.npy --substra_performance_report Single_pharma_run-1/medias/subtuple/c4f1c9b9d44fea66f9b856d346a0bb9aa5727e587185e87daca170f239a70029/pred/perf.json --filename derisk_test --task_map pharma_partners/weight_table_T3_mapped.csv --y_pred_onpremise y_hat1.npy
```

The output should look something like:
```
on-premise_per-task_performances_derisk.csv
on-premise_per-assay_performances_derisk.csv
on-premise_global_performances_derisk.csv
substra_per-task_performances_derisk.csv
substra_per-assay_performances_derisk.csv
substra_global_performances_derisk.csv
deltas_per-task_performances_derisk.csv
deltas_per-assay_performances_derisk.csv
deltas_global_performances_derisk.csv
```

Any problem in the substra output (derisk errors) will be reported like this:
```
ERROR! (Phase 2 de-risk output check): there is problem in the substra platform, yhats not close (tol:1e-05)
WARNING! (Phase 2 de-risk output check): calculated per-task deltas are not zeros (tol:1e-05)
WARNING! (Phase 2 de-risk output check): Global aggregation metric check shows there is a mistake in the aggregated metrics or in the performance reported by the substra platform (tol:1e-05)

```

in addition to the assertion check already performed for the reported perf.json auc_pr performance by substra and the calculated auc_pr from the substra output, which would be output as:

```
AssertionError: Reported performance in <substra-path>/pred/perf.json (<reported AUC PR>) does not match calculated performance for substra (<calculated AUC PR>)
```

-----

# Example 2: Pred file analysis (Single-pharma pred vs. Multi-pharma pred file analysis)

## Pred file evaluation Setup

1. Download the substra output
2. Locate the Sinlge-pharma "pred" & "perf.json" from the Single_pharma_run/medias/subtuple/{pharma-hash}/pred/ folder 
3. Locate the Multi-pharma "pred" & "perf.json" from the Multi_pharma_run/medias/subtuple/{pharma-hash}/pred/ folder 
4. Provide the script with the single- and multi-pharma pred files from step 2/3, the perf.json and task_mapping file


## Pred analysis script (performance_evaluation_pred_files.py)

```
python performance_evaluation_pred_files.py -h
usage: performance_evaluation_pred_files.py [-h] [--y_true_all Y_TRUE_ALL]
        --y_pred_single Y_PRED_SINGLE
        --y_pred_multi Y_PRED_MULTI
        [--folding FOLDING]
        [--task_weights TASK_WEIGHTS]
        [--single_performance_report SINGLE_PERFORMANCE_REPORT]
        --multi_performance_report
        MULTI_PERFORMANCE_REPORT
        [--filename FILENAME]
        [--verbose {0,1}]
        [--task_map_single TASK_MAP_SINGLE]
        [--task_map_multi TASK_MAP_MULTI]

Calculate Performance Metrics

optional arguments:
  -h, --help      show this help message and exit
  --y_true_all Y_TRUE_ALL
      Activity file (npy) (i.e. from files_4_ml/)
  --y_pred_single Y_PRED_SINGLE
      Yhat prediction output from single-pharma run
      (./Single-pharma-
      run/substra/medias/subtuple/<pharma_hash>/pred/pred)
      or (<single pharma dir>/y_hat.npy)
  --y_pred_multi Y_PRED_MULTI
      Yhat prediction output from multi-pharma run (./Multi-
      pharma-
      run/substra/medias/subtuple/<pharma_hash>/pred/pred)
      or (<multi pharma dir>/y_hat.npy)
  --folding FOLDING     LSH Folding file (npy) (i.e. from files_4_ml/)
  --task_weights TASK_WEIGHTS
      CSV file with columns task_id and weight (i.e.
      files_4_ml/T9_red.csv)
  --single_performance_report SINGLE_PERFORMANCE_REPORT
      JSON file with global reported single-pharma
      performance (i.e. ./Single-pharma-run/substra/medias/s
      ubtuple/<pharma_hash>/pred/perf.json)
  --multi_performance_report MULTI_PERFORMANCE_REPORT
      JSON file with global reported multi-pharma
      performance (i.e. ./Multi-pharma-run/substra/medias/su
      btuple/<pharma_hash>/pred/perf.json)
  --filename FILENAME   Filename for results from this output
  --verbose {0,1} Verbosity level: 1 = Full; 0 = no output
  --task_map_single TASK_MAP_SINGLE
      Taskmap from MELLODDY_tuner output of single run
      (results/weight_table_T3_mapped.csv)
  --task_map_multi TASK_MAP_MULTI
      Taskmap from MELLODDY_tuner output of single run
      (results/weight_table_T3_mapped.csv)

```

## Running the pred analysis code
```
python performance_evaluation_pred_files.py --y_true_all pharma_partners/pharma_y_partner_1.npy --y_pred_multi Multi_pharma_run-1/medias/subtuple/374d81d50d0df484bfa40708f270225780aa36dd15a366eb0691e89496653212/pred/pred --y_pred_single Single_pharma_run-1/medias/subtuple/c4f1c9b9d44fea66f9b856d346a0bb9aa5727e587185e87daca170f239a70029/pred/pred --single_performance_report Single_pharma_run-1/medias/subtuple/c4f1c9b9d44fea66f9b856d346a0bb9aa5727e587185e87daca170f239a70029/pred/perf.json --multi_performance_report Multi_pharma_run-1/medias/subtuple/374d81d50d0df484bfa40708f270225780aa36dd15a366eb0691e89496653212/pred/perf.json --filename pred_compare --task_map_single pharma_partners/weight_table_T3_mapped.csv --task_map_multi pharma_partners/weight_table_T3_mapped.csv --folding pharma_partners/folding_partner_1.npy 
```




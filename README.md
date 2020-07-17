# Performance Evaluation Script for the IMI Project MELLODDY

Performance evaluation scripts from the Single- and Multi-pharma outputs

## Requirements

1. Python 3.6 or higher
2. Local Conda installation (e.g. miniconda)
3. Git installation
4. melloddy_tuner environment from WP1 code: https://git.infra.melloddy.eu/wp1/data_prep
5. sparsechem version 0.6.1: https://git.infra.melloddy.eu/wp2/sparsechem/-/tree/v0.6.1 (with sparse-predict functionality) installation from WP2 code: https://git.infra.melloddy.eu/wp2/sparsechem

## Example Setup (substra evalulation for single- multi-partner evaluation)

1. Download the splitted ChEMBL data from here: https://az.app.box.com/file/665317784561
2. Identify your pharma partner ID from here: https://az.app.box.com/file/665327503987 and use *_partner_<your_company_ID>.npy from Step 1 for inputs for the y, folding and weights
3. Download the testnet production run output from here: https://az.app.box.com/folder/115772041696 for your corresponding partner ID
4. Provide the script with the file names of the prediction "pred" and "perf.json" from the Single- and Multi- pharma files from step 3

## CLI of performance_evaluation_pred_files.py (for substra evalulation for single- multi-partner evaluation)

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

## Example input for (substra evalulation for single- multi-partner evaluation)
```
python performance_evaluation_pred_files.py --y_true_all pharma_partners/pharma_y_partner_1.npy --y_pred_multi Multi_pharma_run-1/medias/subtuple/374d81d50d0df484bfa40708f270225780aa36dd15a366eb0691e89496653212/pred/pred --y_pred_single Single_pharma_run-1/medias/subtuple/c4f1c9b9d44fea66f9b856d346a0bb9aa5727e587185e87daca170f239a70029/pred/pred --single_performance_report Single_pharma_run-1/medias/subtuple/c4f1c9b9d44fea66f9b856d346a0bb9aa5727e587185e87daca170f239a70029/pred/perf.json --multi_performance_report Multi_pharma_run-1/medias/subtuple/374d81d50d0df484bfa40708f270225780aa36dd15a366eb0691e89496653212/pred/perf.json --filename pred_compare --task_map_single pharma_partners/weight_table_T3_mapped.csv --task_map_multi pharma_partners/weight_table_T3_mapped.csv --folding pharma_partners/folding_partner_1.npy 
```




## Example Setup (de-risk analysis)

1. Download the splitted ChEMBL data from here: https://az.app.box.com/folder/115772041696
2. For this example we will use pharma 1 since the yhat output/reported performances will match the substra 'pred' file
3. Download the testnet production run output from here: https://app.box.com/folder/117114150538
4. Train on-premise model (e.g. for pharma 1 ```python <sparsechem-sparse-predict>/examples/chembl/train.py --x pharma_partners/pharma_x_partner_1.npy --y pharma_partners/pharma_y_partner_1.npy --task_weights pharma_partners/weights_1   --folding pharma_partners/folding_partner_1.npy   --fold_va 0   --batch_ratio    0.02   --hidden_sizes   400   --last_dropout   0.2   --middle_dropout 0.2   --weight_decay   0.0   --epochs   20   --lr  1e-3  --lr_steps 10 --lr_alpha  0.3 --filename chembl_1```)
6. Create on-premise y-hat predictions with same sparseness as the input (e.g. for pharma 1 ```python <sparsechem-sparse-predict>/examples/chembl/predict.py     --x pharma_partners/pharma_x_partner_1.npy --y pharma_partners/pharma_y_partner_1.npy     --outfile y_hat1.npy     --conf ../../sparsechem/models/chembl_1.json     --model ../../sparsechem/models/chembl_1.pt     --dev cpu --folding pharma_partners/folding_partner_1.npy  --predict_fold 1```) 
7. Provide the script with the sparse prediction y_hat1.npy from step 6. and the "pred" & ""perf.json" from the Single-pharma substra output i.e. two headings shown below...

## CLI for performance_evaluation_derisk.py (substra evalulation for single- multi-partner evaluation)

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

## Example input (for substra evalulation for single- multi-partner evaluation)
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
ERROR! (Phase 2 de-risk output check): calculated per-task deltas are not zeros (tol:1e-05)
ERROR! (Phase 2 de-risk output check): Globel aggregation metric check shows there is a mistake in the aggregated metrics or in the performance reported by the substra platform (tol:1e-05)

```

in addition to the assertion check already performed for the reported perf.json auc_pr performance by substra and the calculated auc_pr from the substra output, which would be output as:

```
AssertionError: Reported performance in <substra-path>/pred/perf.json (<reported AUC PR>) does not match calculated performance for substra (<calculated AUC PR>)
```

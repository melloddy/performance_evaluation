# Performance Evaluation Script for the IMI Project MELLODDY

Performance evaluation scripts from the Single- and Multi-pharma outputs

## Requirements

1. Python 3.6 or higher
2. Local Conda installation (e.g. miniconda)
3. Git installation
4. melloddy_tuner environment from WP1 code

## Setup

1. Download the splitted ChEMBL data from here: https://az.app.box.com/file/665317784561
2. Identify your pharma partner ID from here: https://az.app.box.com/file/665327503987 and use *_partner_<your_company_ID>.npy from Step 1 for inputs for the y, folding and weights
3. Download the testnet production run output from here: https://az.app.box.com/folder/115772041696 for your corresponding partner ID
4. Provide the script with the file names of the prediction "pred" and "perf.json" from the Single- and Multi- pharma files from step 3

## Input

```
python performance_evaluation.py -h
usage: performance_evaluation.py [-h] [--y_true_all Y_TRUE_ALL]
                                 [--y_pred_single Y_PRED_SINGLE]
                                 [--y_pred_multi Y_PRED_MULTI]
                                 [--folding FOLDING]
                                 [--task_weights TASK_WEIGHTS]
                                 [--single_performance_report SINGLE_PERFORMANCE_REPORT]
                                 [--multi_performance_report MULTI_PERFORMANCE_REPORT]
                                 [--filename FILENAME] [--verbose {0,1}]

Calculate Performance Metrics

optional arguments:
  -h, --help            show this help message and exit
  --y_true_all Y_TRUE_ALL
                        Activity file (npy) (i.e. from files_4_ml/)
  --y_pred_single Y_PRED_SINGLE
                        Yhat prediction output from single-pharma run
                        (./Single-pharma-
                        run/substra/medias/subtuple/<pharma_hash>/pred/pred)
  --y_pred_multi Y_PRED_MULTI
                        Yhat prediction output from multi-pharma run (./Multi-
                        pharma-
                        run/substra/medias/subtuple/<pharma_hash>/pred/pred)
  --folding FOLDING     LSH Folding file (npy) (i.e. from files_4_ml/)
  --task_weights TASK_WEIGHTS
                        CSV file with columns task_id and weight (i.e. from
                        files_4_ml/)
  --single_performance_report SINGLE_PERFORMANCE_REPORT
                        JSON file with global reported single-pharma
                        performance (i.e. ./Single-pharma-run/substra/medias/s
                        ubtuple/<pharma_hash>/pred/perf.json)
  --multi_performance_report MULTI_PERFORMANCE_REPORT
                        JSON file with global reported multi-pharma
                        performance (i.e. ./Multi-pharma-run/substra/medias/su
                        btuple/<pharma_hash>/pred/perf.json)
  --filename FILENAME   Filename for results from this output
  --verbose {0,1}       Verbosity level: 1 = Full; 0 = no output
```



## Running the performance evaluation script
i.e. for pharma 1, the following would be run:

> python performance_evaluation.py --y_true_all pharma_partners/pharma_y_partner_3.npy --y_pred_single Single-pharma-run/substra/medias/subtuple/29e625fafef2822df68e8907b612fe81143847ceb4061d7da1502c7e0f0f939d/pred/pred --y_pred_multi Multi-pharma-run/substra/medias/subtuple/29e625fafef2822df68e8907b612fe81143847ceb4061d7da1502c7e0f0f939d/pred/pred --folding pharma_partners/folding_partner_3.npy --task_weights pharma_partners/weights_3 --single_performance_report Single-pharma-run/substra/medias/subtuple/29e625fafef2822df68e8907b612fe81143847ceb4061d7da1502c7e0f0f939d/pred/perf.json --multi_performance_report Multi-pharma-run/substra/medias/subtuple/29e625fafef2822df68e8907b612fe81143847ceb4061d7da1502c7e0f0f939d/pred/perf.json 

## Output from the script

`$ ls global_deltas.csv  local_deltas.csv  pred_global_performances.csv  pred_local_performances.csv`

## Comparison with reference performance evaulation results

The script automatically compares the reported performance metrics with the calculated metrics generated here.

Discrepancies are reported at command line.
### CLS vs. CLSAUX comparison

```
$  python performance_evaluation_cls_clsaux.py -h
usage: performance_evaluation_cls_clsaux.py [-h] --y_cls Y_CLS --y_clsaux Y_CLSAUX --folding_cls FOLDING_CLS --folding_clsaux FOLDING_CLSAUX --weights_cls WEIGHTS_CLS --weights_clsaux WEIGHTS_CLSAUX --t10c_cls T10C_CLS --t10c_clsaux T10C_CLSAUX --pred_cls PRED_CLS --pred_clsaux PRED_CLSAUX [--validation_fold VALIDATION_FOLD] --outfile OUTFILE

Computes statistical significance between a cls and a clsaux classification models

optional arguments:
  -h, --help            show this help message and exit
  --y_cls Y_CLS         Path to <...>/matrices/cls/cls_T10_y.npz
  --y_clsaux Y_CLSAUX   Path to <...>/matrices/clsaux/clsaux_T10_y.npz
  --folding_cls FOLDING_CLS
                        Path to <...>/matrices/cls/cls_T11_fold_vector.npy
  --folding_clsaux FOLDING_CLSAUX
                        Path to <...>/matrices/clsaux/clsaux_T11_fold_vector.npy
  --weights_cls WEIGHTS_CLS
                        Path to <...>/matrices/clsaux/cls_weights.csv
  --weights_clsaux WEIGHTS_CLSAUX
                        Path to <...>/matrices/clsaux/clsaux_weights.csv
  --t10c_cls T10C_CLS   Path to <...>/matrices/cls/T10c_cont.csv
  --t10c_clsaux T10C_CLSAUX
                        Path to <...>/matrices/clsaux/T10c_cont.csv
  --pred_cls PRED_CLS   Path to the predictions exported from platform of a cls model
  --pred_clsaux PRED_CLSAUX
                        Path to the predictions exported from platform of a clsaux model
  --validation_fold VALIDATION_FOLD
                        Validation fold to use
  --outfile OUTFILE     Name of the output file
```

#### Overview

This script ingests the `T10_c` and `cls_weights.csv` files from CLS and CLSAUX runs to create a CLS & CLSAUX mapping. Next, any significant change in performance upon including auxiliary data is computed.

#### Step 1. Locate the required SP/MP runs for the CLS and CLSAUX models.

Locate each of the best CLS/CLSAUX runs for the MP and SP runs. 

#### Step 2. Run the script.
Then run the following comparisons:

1. CLS MP vs. CLSAUX MP
2. CLS SP vs. CLSAUX SP
3. CLS MP vs. CLSAUX SP
4. CLS SP vs. CLSAUX MP

To do this, run the script as, e.g:
```
python <perf_eval_dir>/performance_evaluation_cls_clsaux.py \
    --validation_fold 0 --y_cls cls/cls_T10_y.npz \
    --y_clsaux cls/clsaux_T10_y.npz \
    --folding_cls cls/cls_T11_fold_vector.npy \
    --folding_clsaux cls/clsaux_T11_fold_vector.npy  \
    --weights_cls cls/cls_weights.csv \
    --weights_clsaux cls/clsaux_weights.csv \
    --t10c_cls cls/T10c_cont.csv \
    --t10c_clsaux cls/T10c_cont_aux.csv \
    --pred_cls $cls_preds/pred.json \
    --pred_clsaux $clsaux_preds/pred.json \
    --outfile $clsaux_preds_vs_$clsaux_preds
```


#### Step 3. Report the summary

Only report the file {args.outfile}_summary_to_report.csv to the pharma only box folder.

#### Step 4. Update Monday.com

Report progress on Monday.com


------

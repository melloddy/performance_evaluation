```
usage: relative_deltas.py [-h] --type {baseline,perfection,normal} --baseline BASELINE --compared COMPARED --outdir OUTDIR [-v]

Computes (absolute and) relative deltas

optional arguments:
  -h, --help            show this help message and exit
  --type {baseline,perfection,absolute}
                        type of relative delta to compute:
                        absolute: (compared - baseline)
                        relative_improve: (compared - baseline)/baseline
                        improve_to_perfect:(compared - baseline)/(perfect_val - baseline)
  --baseline BASELINE   *per-task_performances_NOUPLOAD.csv file containing task level performances of baseline: produced by WP3 performance_evaluation.py code
  --compared COMPARED   *per-task_performances_NOUPLOAD.csv file containing task level performances of model to compare: produced by WP3 performance_evaluation.py code
  --outdir OUTDIR       output directory into which the resultig files will be saved.
  -v, --verbose         verbosity


```
## Types of deltas

| key word | delta description |
|---|---|
| absolute | compared - baseline |
| relative_improve | (compared - baseline) / baseline  |
| improve_to_perfect | (compared - baseline) / (perfect_value - baseline)  | 


## Example using SPCLS as baseline and SPCLSAUX as compared

```bash

python relative_deltas.py --type baseline \
                          --baseline cls/SP/pred_per-task_performances_NOUPLOAD.csv \
                          --compared clsaux/SP/pred_per-task_performances_NOUPLOAD.csv \
                          --outdir clsaux/deltas_relative_SPCLS 

# Note : the input task level performances are the actual performance metrics, not the deltas

```

## Outputs

```
deltas_relative_SPCLS
├── deltas_global_performances.csv             # global relative deltas (for each metrics)
├── deltas_per-assay_performances.csv          # per assay type relative deltas (for each metrics)
└── deltas_per-task_performances_NOUPLOAD.csv  # per task relative deltas (for each metrics)

```



# NEEDED: Full list of command line calls needed for partners to run. 

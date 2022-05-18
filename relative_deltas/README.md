```
usage: relative_deltas.py [-h] --type {baseline,perfection,normal} --baseline BASELINE --compared COMPARED [--subset SUBSET [SUBSET ...]] --outdir OUTDIR [-v]

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
  --subset SUBSET [SUBSET ...]
                        selection of csv files (w/ header: input_assay_id) containing the subsets of input assays IDs for which to calculate performances, e.g. 'alive', or 'virtual safety panel' lists
  --outdir OUTDIR       output directory into which the resultig files will be saved.
  -v, --verbose         verbosity


```
## Types of deltas

| key word | delta description |
|---|---|
| absolute | compared - baseline |
| relative_improve | (compared - baseline) / baseline  |
| improve_to_perfect | (compared - baseline) / (perfect_value - baseline)  | 

Convention applied for delta improvement to perfection: <br>
 - if compared and baseline are both = perfect performance, delta improve_to_perfect = 0
 - if baseline has perfect perforance and compared has worst performance, delta improve_to_perfect = compared - baseline (i.e. identic to absolute delta)

## Example using SPCLS as baseline and SPCLSAUX as compared , using a subset of alive assays

```bash

python relative_deltas.py --type relative_improve \
                          --baseline cls/SP/pred_per-task_performances_NOUPLOAD.csv \
                          --compared clsaux/SP/pred_per-task_performances_NOUPLOAD.csv \
                          --subset alive_assays.csv \
                          --outdir clsaux/deltas_relative_SPCLS 

# Note : the input task level performances are the actual performance metrics, not the deltas

```

## Outputs

```
deltas_relative_SPCLS
├── deltas_global_performances_alive_assays.csv             # global relative deltas (for each metrics) for alive assay subset
├── deltas_per-assay_performances_alive_assays.csv          # per assay type relative deltas (for each metrics) for alive assay subset
└── deltas_per-task_performances_NOUPLOAD_alive_assays.csv  # per task relative deltas (for each metrics) for alive assay subset
├── deltas_global_performances.csv             # global relative deltas (for each metrics)
├── deltas_per-assay_performances.csv          # per assay type relative deltas (for each metrics)
└── deltas_per-task_performances_NOUPLOAD.csv  # per task relative deltas (for each metrics)

```



# NEEDED: Full list of command line calls needed for partners to run. 

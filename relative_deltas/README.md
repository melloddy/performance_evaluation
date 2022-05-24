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

## An example to run the code in which SPCLS and SPCLSAUX are chosen as baseline and compared respectively

```bash
#In this example, subset results on alive assay will be also computed 
python relative_deltas.py --type relative_improve \
                          --baseline cls/SP/pred_per-task_performances_NOUPLOAD.csv \
                          --compared clsaux/SP/pred_per-task_performances_NOUPLOAD.csv \
                          --subset alive_assay.csv \
                          --outdir clsaux/deltas_relative_SPCLS 

# Note : the input task level performances are the actual performance metrics, not the deltas

```

## Outputs for the example above

```
deltas_relative_SPCLS
├── deltas_global_performances_alive_assay.csv             # global relative deltas (for each metrics) for alive assay subset
├── deltas_per-assay_performances_alive_assay.csv          # per assay type relative deltas (for each metrics) for alive assay subset
└── deltas_per-task_performances_NOUPLOAD_alive_assay.csv  # per task relative deltas (for each metrics) for alive assay subset
├── deltas_global_performances.csv             # global relative deltas (for each metrics)
├── deltas_per-assay_performances.csv          # per assay type relative deltas (for each metrics)
└── deltas_per-task_performances_NOUPLOAD.csv  # per task relative deltas (for each metrics)

```

# The usage of pipeline for the final performance evaluation
To calculate deltas for 6 settings for classification and 6 settings for regression, relative_deltas.py has been integrated in to a pipeline. Below, we will explain how this pipeline works and what it does.
## Pre-requisite
The user of the pipeline need to run both conformal prediction efficiency codes [year-3](https://git.infra.melloddy.eu/wp1/entropy_cp_ad/-/tree/master/year3) and performance evaluation codes, performance_evaluation.py [year-3](https://git.infra.melloddy.eu/wp3/performance_evaluation). For latter, we need output task level performances by setting --output_task_sensitive_files 1.

## Step 1: Extraction of the taskids and conformal prediction efficiency (cpe) values
This will be done by running the ipynb put [here](https://git.infra.melloddy.eu/wp1/entropy_cp_ad/-/blob/master/year3/setup/analysis/ad_result_gathering.ipynb).To run this ipynb, using the same environment as the one used for the main conformal prediction efficiency codes will be nice. The resulting file will have 6 columns containing task_id_cls, task_id_clsaux, cpe_sp_cls, cpe_sp_clsaux, cpe_mp_cls, cpe_mp_clsaux.

## Step 2: Joining of conformal prediction efficiencies to task level performance evaluation
This step will be do by running the ipynb put [here]().

### Step 2-1: Opening the notebook
Let's activate your environment where you can use jupyter notebook and pandas. Let's open the ipynb file.

### Step 2-2: Filling the input files
Let us fill in the input file. First, we will fill the output folder for cls and clsaux respectively.In the perf eval output folder architecture, there is a folder whose subfolder is MP/SP/deltas. We will select that folder for cls and clsaux respectively in the 2nd cell of the ipynb.

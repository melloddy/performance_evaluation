```
usage: relative_deltas.py [-h] --type {relative_improve,improve_to_perfect,absolute} --baseline BASELINE --compared COMPARED [--subset SUBSET [SUBSET ...]] [--baseline_topn BASELINE_TOPN [BASELINE_TOPN ...]] [--delta_topn DELTA_TOPN [DELTA_TOPN ...]] --outdir OUTDIR [-v]

Computes (absolute and) relative deltas

optional arguments:
  -h, --help            show this help message and exit
  --type {relative_improve,improve_to_perfect,absolute}
                        type of relative delta to compute:
                        absolute: (compared - baseline)
                        relative_improve: (compared - baseline)/baseline
                        improve_to_perfect:(compared - baseline)/(perfect_val - baseline)
  --baseline BASELINE   *per-task_performances_NOUPLOAD.csv file containing task level performances of baseline: produced by WP3 performance_evaluation.py code
  --compared COMPARED   *per-task_performances_NOUPLOAD.csv file containing task level performances of model to compare: produced by WP3 performance_evaluation.py code
  --subset SUBSET [SUBSET ...]
                        selection of csv files (w/ header: input_assay_id) containing the subsets of input assays IDs for which to calculate performances, e.g. 'alive', or 'virtual safety panel' lists
  --baseline_topn BASELINE_TOPN [BASELINE_TOPN ...]
  --delta_topn DELTA_TOPN [DELTA_TOPN ...]
  --outdir OUTDIR       output directory into which the resultig files will be saved.
  -v, --verbose         verbosity
```
## Types of deltas

| key word | delta description |
|---|---|
| absolute | compared - baseline |
| relative_improve | (compared - baseline) / baseline  |
| improve_to_perfect | (compared - baseline) / (perfect_value - baseline)  |

Convention applied to delta relative to baseline:  
 - if baseline perf = 0 , delta relative to baseline = perf of compared


Convention applied to delta improvement to perfection: <br>
 - if compared and baseline are both = perfect performance, delta improve_to_perfect = 0
 - if baseline has perfect perforance and compared has worst performance, delta improve_to_perfect = compared - baseline (i.e. identic to absolute delta)

## An example code execution

In this example we will compare SPCLSAUX to the baseline SPCLS.<br>
Here we also request for the aggregate/CDF perfs to be computed over three subsets: 
 - subset of assays (alive_assay.csv)
 - TOP 10% subset of tasks relative to the baseline
 - TOP 10% subset of tasks relative to the delta (defined by `--type`)

```bash
#In this example, subset results on alive assay will be also computed 
python relative_deltas.py --type relative_improve \
                          --baseline cls/SP/pred_per-task_performances_NOUPLOAD.csv \
                          --compared clsaux/SP/pred_per-task_performances_NOUPLOAD.csv \
                          --subset alive_assay.csv \
                          --delta_topn 0.1 \
                          --baseline_topn 0.1 \
                          --outdir spcls_mpclsaux/relative_improve

# Note : the input task level performances are the actual performance metrics, not the deltas

```

## Outputs for the example above

```
spcls_mpclsaux/
└── relative_improve
    ├── cdf
    │   ├── _alive_assay_baseline-topn_0.1_cdfbaseline-cdfcompared_assay_type.csv
    │   ├── _alive_assay_baseline-topn_0.1_cdfbaseline-cdfcompared.csv
    │   ├── _alive_assay_cdfbaseline-cdfcompared_assay_type.csv
    │   ├── _alive_assay_cdfbaseline-cdfcompared.csv
    │   ├── _alive_assay_delta-topn_0.1_auc_pr_cal_cdfbaseline-cdfcompared_assay_type.csv
    │   ├── _alive_assay_delta-topn_0.1_auc_pr_cal_cdfbaseline-cdfcompared.csv
    │   ├── _alive_assay_delta-topn_0.1_auc_pr_cdfbaseline-cdfcompared_assay_type.csv
    │   ├── _alive_assay_delta-topn_0.1_auc_pr_cdfbaseline-cdfcompared.csv
    │   ├── _alive_assay_delta-topn_0.1_efficiency_overall_cdfbaseline-cdfcompared_assay_type.csv
    │   ├── _alive_assay_delta-topn_0.1_efficiency_overall_cdfbaseline-cdfcompared.csv
    │   ├── _alive_assay_delta-topn_0.1_roc_auc_score_cdfbaseline-cdfcompared_assay_type.csv
    │   ├── _alive_assay_delta-topn_0.1_roc_auc_score_cdfbaseline-cdfcompared.csv
    │   ├── _baseline-topn_0.1_cdfbaseline-cdfcompared_assay_type.csv
    │   ├── _baseline-topn_0.1_cdfbaseline-cdfcompared.csv
    │   ├── _cdfbaseline-cdfcompared_assay_type.csv
    │   ├── _cdfbaseline-cdfcompared.csv
    │   ├── _delta-topn_0.1_auc_pr_cal_cdfbaseline-cdfcompared_assay_type.csv
    │   ├── _delta-topn_0.1_auc_pr_cal_cdfbaseline-cdfcompared.csv
    │   ├── _delta-topn_0.1_auc_pr_cdfbaseline-cdfcompared_assay_type.csv
    │   ├── _delta-topn_0.1_auc_pr_cdfbaseline-cdfcompared.csv
    │   ├── _delta-topn_0.1_efficiency_overall_cdfbaseline-cdfcompared_assay_type.csv
    │   ├── _delta-topn_0.1_efficiency_overall_cdfbaseline-cdfcompared.csv
    │   ├── _delta-topn_0.1_roc_auc_score_cdfbaseline-cdfcompared_assay_type.csv
    │   └── _delta-topn_0.1_roc_auc_score_cdfbaseline-cdfcompared.csv
    ├── deltas_global_performances_alive_assay_baseline-topn_0.1_auc_pr_cal.csv
    ├── deltas_global_performances_alive_assay_baseline-topn_0.1_auc_pr.csv
    ├── deltas_global_performances_alive_assay_baseline-topn_0.1_efficiency_overall.csv
    ├── deltas_global_performances_alive_assay_baseline-topn_0.1_roc_auc_score.csv
    ├── deltas_global_performances_alive_assay.csv
    ├── deltas_global_performances_alive_assay_delta-topn_0.1_auc_pr_cal.csv
    ├── deltas_global_performances_alive_assay_delta-topn_0.1_auc_pr.csv
    ├── deltas_global_performances_alive_assay_delta-topn_0.1_efficiency_overall.csv
    ├── deltas_global_performances_alive_assay_delta-topn_0.1_roc_auc_score.csv
    ├── deltas_global_performances_baseline-topn_0.1_auc_pr_cal.csv
    ├── deltas_global_performances_baseline-topn_0.1_auc_pr.csv
    ├── deltas_global_performances_baseline-topn_0.1_efficiency_overall.csv
    ├── deltas_global_performances_baseline-topn_0.1_roc_auc_score.csv
    ├── deltas_global_performances.csv
    ├── deltas_global_performances_delta-topn_0.1_auc_pr_cal.csv
    ├── deltas_global_performances_delta-topn_0.1_auc_pr.csv
    ├── deltas_global_performances_delta-topn_0.1_efficiency_overall.csv
    ├── deltas_global_performances_delta-topn_0.1_roc_auc_score.csv
    ├── deltas_per-assay_performances_alive_assay_baseline-topn_0.1_auc_pr_cal.csv
    ├── deltas_per-assay_performances_alive_assay_baseline-topn_0.1_auc_pr.csv
    ├── deltas_per-assay_performances_alive_assay_baseline-topn_0.1_efficiency_overall.csv
    ├── deltas_per-assay_performances_alive_assay_baseline-topn_0.1_roc_auc_score.csv
    ├── deltas_per-assay_performances_alive_assay.csv
    ├── deltas_per-assay_performances_alive_assay_delta-topn_0.1_auc_pr_cal.csv
    ├── deltas_per-assay_performances_alive_assay_delta-topn_0.1_auc_pr.csv
    ├── deltas_per-assay_performances_alive_assay_delta-topn_0.1_efficiency_overall.csv
    ├── deltas_per-assay_performances_alive_assay_delta-topn_0.1_roc_auc_score.csv
    ├── deltas_per-assay_performances_baseline-topn_0.1_auc_pr_cal.csv
    ├── deltas_per-assay_performances_baseline-topn_0.1_auc_pr.csv
    ├── deltas_per-assay_performances_baseline-topn_0.1_efficiency_overall.csv
    ├── deltas_per-assay_performances_baseline-topn_0.1_roc_auc_score.csv
    ├── deltas_per-assay_performances.csv
    ├── deltas_per-assay_performances_delta-topn_0.1_auc_pr_cal.csv
    ├── deltas_per-assay_performances_delta-topn_0.1_auc_pr.csv
    ├── deltas_per-assay_performances_delta-topn_0.1_efficiency_overall.csv
    ├── deltas_per-assay_performances_delta-topn_0.1_roc_auc_score.csv
    └── NOUPLOAD
        ├── deltas_per-task_performances_NOUPLOAD_alive_assay_baseline-topn_0.1_auc_pr_cal.csv
        ├── deltas_per-task_performances_NOUPLOAD_alive_assay_baseline-topn_0.1_auc_pr.csv
        ├── deltas_per-task_performances_NOUPLOAD_alive_assay_baseline-topn_0.1_efficiency_overall.csv
        ├── deltas_per-task_performances_NOUPLOAD_alive_assay_baseline-topn_0.1_roc_auc_score.csv
        ├── deltas_per-task_performances_NOUPLOAD_alive_assay.csv
        ├── deltas_per-task_performances_NOUPLOAD_alive_assay_delta-topn_0.1_auc_pr_cal.csv
        ├── deltas_per-task_performances_NOUPLOAD_alive_assay_delta-topn_0.1_auc_pr.csv
        ├── deltas_per-task_performances_NOUPLOAD_alive_assay_delta-topn_0.1_efficiency_overall.csv
        ├── deltas_per-task_performances_NOUPLOAD_alive_assay_delta-topn_0.1_roc_auc_score.csv
        ├── deltas_per-task_performances_NOUPLOAD_baseline-topn_0.1_auc_pr_cal.csv
        ├── deltas_per-task_performances_NOUPLOAD_baseline-topn_0.1_auc_pr.csv
        ├── deltas_per-task_performances_NOUPLOAD_baseline-topn_0.1_efficiency_overall.csv
        ├── deltas_per-task_performances_NOUPLOAD_baseline-topn_0.1_roc_auc_score.csv
        ├── deltas_per-task_performances_NOUPLOAD.csv
        ├── deltas_per-task_performances_NOUPLOAD_delta-topn_0.1_auc_pr_cal.csv
        ├── deltas_per-task_performances_NOUPLOAD_delta-topn_0.1_auc_pr.csv
        ├── deltas_per-task_performances_NOUPLOAD_delta-topn_0.1_efficiency_overall.csv
        └── deltas_per-task_performances_NOUPLOAD_delta-topn_0.1_roc_auc_score.csv
```

# The usage of pipeline for the final performance evaluation
To calculate deltas for 6 settings for classification and 6 settings for regression, relative_deltas.py has been integrated in to a pipeline. Below, we will explain how this pipeline works and what it does.
## Pre-requisite
The user of the pipeline need to run both conformal prediction efficiency codes [year-3](https://git.infra.melloddy.eu/wp1/entropy_cp_ad/-/tree/master/year3) and performance evaluation codes, performance_evaluation.py [year-3](https://git.infra.melloddy.eu/wp3/performance_evaluation). For latter, we need output task level performances by setting --output_task_sensitive_files 1.

## Step 1: Extraction of the taskids and conformal prediction efficiency (cpe) values
This will be done by running the ipynb put [here](https://git.infra.melloddy.eu/wp1/entropy_cp_ad/-/blob/master/year3/setup/analysis/ad_result_gathering.ipynb).To run this ipynb, using the same environment as the one used for the main conformal prediction efficiency codes will be nice. The resulting file will have 6 columns containing task_id_cls, task_id_clsaux, cpe_sp_cls, cpe_sp_clsaux, cpe_mp_cls, cpe_mp_clsaux.

## Step 2: Joining of conformal prediction efficiencies to task level performance evaluation
This step will be do by running the ipynb put [here](https://git.infra.melloddy.eu/wp3/performance_evaluation/-/blob/year3/relative_deltas/Joiner_perfeval_adcp_ml.ipynb).

### Step 2-1: Opening the notebook
Let's activate your environment where you can use jupyter notebook and pandas. Let's open the ipynb file.

### Step 2-2: Filling the input files
Let us fill in the input file. First, we will fill the output folder for cls and clsaux respectively.In the perf eval output folder architecture, there is a folder whose subfolder is MP/SP/deltas. We will select that folder for cls and clsaux respectively in the 2nd cell of the ipynb.

Second, we will input the file path to the ad_summary_phase2_commcat2.csv generated in the Step 1, using path_adcp_result in the 2nd cell of the ipynb.

### Step 2-3: Running the notebook.
Then we can run the notebook, which will give you 4 files whose names finish with wADCP.csv; 1 each in cls/SP, cls/MP, clsaux/SP, and clsaux/MP folders. In these files, the ADCP results were added under the column name 'efficiency_overall'.

## Step 3: Calculate the relative deltas
### Step 3-1: Getting the latest year3 performance evaluation repository
Let's clone or pull [the latest year3 performance evaluation repository](https://git.infra.melloddy.eu/wp3/performance_evaluation).
### Step 3-2: Setup task performance file locations
Please decide your working directory and copy `file_location.sh` there.<br>
Fill-in the pathes to the task-level performance files of CLS/CLSAUX (produced above with an ADCP result column) and of REG/HYB models in `file_location.sh` for both SP and MP. (On the repo, you can find this shell script [here](https://git.infra.melloddy.eu/wp3/performance_evaluation/-/blob/year3/relative_deltas/file_locations.sh).)<br>
You will also need the list of "alive" assays and the list of first line safety panel assays (one column data frame with an input_assay_id column).<br>


### Step 3-3: Run the relative_deltas.py script over desired comparisons
Copy bash script `run_cls_and_reg_ml.sh` provided [here](https://git.infra.melloddy.eu/wp3/performance_evaluation/-/blob/year3/relative_deltas/run_cls_and_reg_ml.sh) into the working directory (specified in the Step 3-2).<br>
In the shell script, let's fill-in paths of `relative_deltas.py` and `file_location.sh`. Then please specify the output folder. If you need some specific prefix to run python code, please specify this prefix in the prefix variable. Then, the editing of this shell script is done. <br>
Run the bash script to excute all comparisons: 

```bash
bash run_cls_and_reg_ml.sh
```
This will produce, for all comparisons listed [here](https://git.infra.melloddy.eu/wp3/performance_evaluation/-/blob/year3/relative_deltas/run_cls_and_reg_ml.sh#L41-42), the CDFs, the deltas (absolute, relative to baseline and relative to perfection) for full sets of tasks as well as subsets.<br>

The specified output folder will contain two folders:
 - private -> contains task level deltas to keep home
 - to_share -> contains aggregate performances for all considered sets of tasks 

### Step 4 - Share content of "to_share" folder
TBD

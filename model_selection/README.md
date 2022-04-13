# Model selection script

Set of scripts to select the optimal set (phase1, phase2, phase3) of MP models per dataset

**Workflow**:
- Loops over final model list
- Selects list of epochs available across all phases
- Selects optimal epoch in phase 1 by extracting metrics files from tarball
- Writes out csv files with optimal epoch per model and overall optimal model

## Instructions
- Copy python code, bash scripts, and model list to directory with model tar.gz files
- Activate python env
- For each dataset, execute the corresponding bash script, e.g.:
```
./run_model_selection_CLS.bash
```
- Use phase2 models listed in *perf_opt_$DATASET.csv* for further performance evaluation anaylsis


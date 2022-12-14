# Guidelines for local SP model selection

## Description

Model fusion and performance evaluation requires access to single partner (SP) models trained and selected analogously to the multi partner (MP) models trained on platform.
The following steps for local SP training & selection should ensure equal treatment and a fair MP-SP comparison:

## Instructions

### HP selection

Based on HP scans you've performed at the beginning of the year 3 run, select:

**For CLS**:
- the best HP combination (it was used on the platform)
- 2 diverse, but reasonably good HP combinations

**For CLSAUX**:
- the best HP combination (it was used on the platform)
- 6 reasonably good, diverse HP combinations

**For REG**:
- the best HP combination (it was used on the platform)
- 5 reasonably good diverse HP combinations

**For HYB**:
- the best HP combination (it was used on the platform)
- 10 reasonably good diverse HP combinations

### PH1 training

Train local SP models in **Phase 1** with the following maximum epoch:

- CLS: 20
- CLSAUX: 30
- REG / HYB: 50

For REG perform training using the best HP combination 5 times (just 5 independent sparsechem runs), for HYB - 3 times. As a result you should have 3 SP models for CLS, 7 for CLSAUX, 10 for REG and 13 for HYB.

### PH1 selection

For each **Phase 1** model (including all repeats of the best HP combinations), select optimal epoch considering model output frequency on platform (select from epochs 5, 10, 15, 20, 25, 30, 35, 40, 45, 50).
Note that in SparseChem Epoch 0 is actually Epoch 1 (e.g. Epoch 15 on platform would be Epoch 14 in SparseChem)

### PH2 training

Train local **Phase 2** model with max. epoch equaling the optimal epoch selected as described above. Make sure to set `--save_model 1`.

### PH1 re-training

Train local **Phase 1** model with max. epoch equaling the optimal epoch selected as described above (model output in SparseChem is limited to the final epoch so re-training is required if in the initial maximum epoch is not equal the optimal epoch). Make sure to set `--save_model 1`.

## Phase Specification in SparseChem

### Phase 1 = Validation Fold Evaluation

```
  --fold_va 4 \ 
  --fold_te 0 \ 
```

### Phase 2 = Test Fold Evaluation

```
  --fold_va 0 \
```

## Model Analysis

### Performance Evaluation

Use Phase 2 models for running the main performance evluation script.

### Model Fusion

Use Phase 1 and corresponding Phase 2 models both trained up to the optimal epoch for model fusion analysis.

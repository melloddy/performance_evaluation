This notebook assumes you have exectuted the CLS_vs_CLSAUX performance analysis [here](https://git.infra.melloddy.eu/wp3/performance_evaluation/-/tree/year3/CLS_vs_CLSAUX) and that you have task level performance of the different comparisons (including significance results). <br>
It also assumes you have task level performance deltas (including significance results) produced by [performance_evaluation.py](https://git.infra.melloddy.eu/wp3/performance_evaluation/-/blob/year3/performance_evaluation.py) for the following comparison:
- SPCLSAUX-MPCLSAUX
- SPCLS_MPCLS


Required for the ansalysis are task level performance improvements significance results for: 
```python
config = {
    "SPCLS_SPCLSAUX":"< path to task performance with significance results from cls_vs_clsaux : spcls_vs_spclsaux >",
    "MPCLS_MPCLSAUX":"< path to task performance with significance results from cls_vs_clsaux : mpcls_vs_mpclsaux >",
    "MPCLS_SPCLSAUX":"< path to task performance with significance results from cls_vs_clsaux : mpcls_vs_spclsaux >" ,
    "SPCLS_MPCLSAUX":"< path to task performance with significance results from cls_vs_clsaux : spcls_vs_mpclsaux >",
    "SPCLS_MPCLS":"< path to performance_evluation.py task perf deltas of comparison SPCLS-MPCLS cls/deltas/deltas_per-task_performances_NOUPLOAD.csv > ",
    "SPCLSAUX_MPCLSAUX":"< path to performance_evluation.py task perf deltas of comparison SPCLSAUX-MPCLSAUX clsaux/deltas/deltas_per-task_performances_NOUPLOAD.csv >"
}

```

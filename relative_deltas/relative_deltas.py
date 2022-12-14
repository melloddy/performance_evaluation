import os
import argparse
import pandas as pd
import numpy as np

class SmartFormatter(argparse.HelpFormatter):

    def _split_lines(self, text, width):
        if text.startswith('R|'):
            return text[2:].splitlines()  
        return argparse.HelpFormatter._split_lines(self, text, width)

parser = argparse.ArgumentParser(description="Computes (absolute and) relative deltas", formatter_class=SmartFormatter)
parser.add_argument("--type",
                    type=str,
                    help="R|type of relative delta to compute:\n"
                    "absolute: (compared - baseline)\n"
                    "relative_improve: (compared - baseline)/baseline\n"
                    "improve_to_perfect:(compared - baseline)/(perfect_val - baseline)",
                    choices=["relative_improve", "improve_to_perfect", "absolute"],
                    required=True)

parser.add_argument("--baseline",
                    type=str,
                    help="*per-task_performances_NOUPLOAD.csv file containing task level performances of baseline: produced by WP3 performance_evaluation.py code",
                    required=True)

parser.add_argument("--compared",
                    type=str,
                    help="*per-task_performances_NOUPLOAD.csv file containing task level performances of model to compare: produced by WP3 performance_evaluation.py code",
                    required=True)

parser.add_argument("--subset",
                    type=str,
                    help="selection of csv files (w/ header: input_assay_id) containing the subsets of input assays IDs for which to calculate performances, e.g. 'alive', or 'virtual safety panel' lists",
                    default=[],
                    nargs='+')

parser.add_argument("--baseline_topn",
                    type=float,
                    help="",
                    default=[],
                    nargs='+')

parser.add_argument("--delta_topn",
                    type=float,
                    help="",
                    default=[],
                    nargs='+')

parser.add_argument("--outdir",
                    type=str,
                    help="output directory into which the resultig files will be saved.",
                    required=True)

parser.add_argument("-v", "--verbose", 
                    action="store_true", 
                    help="verbosity", 
                    default=False)
                    
parser.add_argument("--version", 
                    help="version of this script",
                    default="0.5", choices=["0.5"])
args = parser.parse_args()



def load_task_perfs():
    """ Load, sanitize, lighten up, merges """
    
    cls_metrics = ['roc_auc_score', 'auc_pr', 'auc_pr_cal']
    reg_metrics = ['rsquared', 'corrcoef', 'rmse_uncen', 'rmse']
        
    assert 'deltas' not in args.baseline.split(os.sep)[-1], "deltas detected in filename: deltas not allowed here, the script needs task level actual performance values" 
    assert 'deltas' not in args.compared.split(os.sep)[-1], "deltas detected in filename: deltas not allowed here, the script needs task level actual performance values" 
    
    baseline_task_perf = pd.read_csv(args.baseline)
    compared_task_perf = pd.read_csv(args.compared)
    
    assert baseline_task_perf.shape[0] > 1, f"There seems to be only one record in {args.baseline}, need task level performances from *per-task_performances_NOUPLOAD.csv"
    assert compared_task_perf.shape[0] > 1, f"There seems to be only one record in {args.baseline}, need task level performances from *per-task_performances_NOUPLOAD.csv"
        
    if args.verbose:
        print(f"Loaded baseline  : {args.baseline}")
        print(f"Loaded comparison: {args.compared}")
    
    assert baseline_task_perf.shape[0] == compared_task_perf.shape[0], "baseline and compared do not have the same number of tasks..."
        
    if args.subset:
        for filename in args.subset:
            if filename is not None:
                sub = pd.read_csv(filename)
                subset_in_perf = np.unique(np.isin(sub['input_assay_id'].unique(), baseline_task_perf['input_assay_id'].unique()), return_counts=True)
                counts = {subset_in_perf[0][i]:subset_in_perf[1][i] for i in range(len(subset_in_perf[0]))}
            
                if args.verbose:     
                    print(f"\nSubset : {filename.split(os.sep)[-1]}")
                    print(f"{counts[True]:>8} assays w/ performance")
                    try: print(f"{counts[False]:>8} assays w/o performance\n")
                    except KeyError: pass 
                assert counts[True] > 0, f"Could not identify any input_assay_ids in task perf from subset {filename}"
            
    
    if 'rsquared' in baseline_task_perf.columns: 
        assert 'rsquared' in compared_task_perf.columns, "compared task performance does not contain regression performance metrics"
        
        
        # set the assay types (no catalog type)
        assay_type_reg = {'OTHER':'OTHER',
                          'ADME':'ADME',
                          'NON-CATALOG-PANEL':'PANEL',
                          'CATALOG-PANEL':'PANEL'}
        
        baseline_task_perf['assay_type'] == baseline_task_perf['assay_type'].map(assay_type_reg)
        compared_task_perf['assay_type'] == compared_task_perf['assay_type'].map(assay_type_reg)
        
        metrics = reg_metrics
        cols2use = ['input_assay_id', f'cont_regression_task_id', 'assay_type'] + metrics
        merged = baseline_task_perf[cols2use].merge(compared_task_perf[cols2use], on=['input_assay_id', 'cont_regression_task_id', 'assay_type'], suffixes=('_baseline', '_compared'))
        assert merged.shape[0] == baseline_task_perf.shape[0], "Not able to match tasks between baseline and compared... Are your input_assay_ids consistent between both?"
        
    else:
        assert 'auc_pr' in compared_task_perf.columns, "compared task performance does not contain classification performance metrics"
        if 'efficiency_overall' in baseline_task_perf.columns and 'efficiency_overall' in compared_task_perf.columns: 
            cls_metrics.append('efficiency_overall')
        else:
            print("\nWARNING: the metric 'efficiency_overall' was not found in both compared and baseline task performance: it will be ignored")
        
        metrics = cls_metrics
        cols2use = ['input_assay_id', 'threshold', f'cont_classification_task_id', 'assay_type'] + metrics
        
        # verif that tasks can match (i.e. cls vs clsaux situation)
        merged = baseline_task_perf[cols2use].merge(compared_task_perf[cols2use], on=['input_assay_id', 'threshold', 'assay_type'], suffixes=('_baseline', '_compared')).reset_index(drop=True)
        assert merged.shape[0] == baseline_task_perf.shape[0], "Not able to match tasks between baseline and compared... Are your input_assay_ids consistent between both?"
    
    return merged, metrics


def compute_task_deltas(df_, metrics, subset=None):
    """ from the merged dataframe returned above, computes the relative (or absolute) task deltas"""
    
    df = df_.copy()
    if subset is not None: 
        df=df.query('input_assay_id in @subset').copy()
    
    if args.type == 'absolute':
        for m in metrics:
            df[m] = df[f'{m}_compared'] - df[f'{m}_baseline']
    
    elif args.type == 'relative_improve':
        for m in metrics:
            df[m] = (df[f'{m}_compared'] - df[f'{m}_baseline']) / df[f'{m}_baseline']
            
            # convention : if baseline = 0 , relative delta = perf of compared
            zero_baseline = df.loc[df[f'{m}_baseline']==0].copy()
            if zero_baseline.shape[0] > 0: 
                not_zero_baseline = df.loc[df[f'{m}_baseline']!=0]
                zero_baseline[m] = zero_baseline[f'{m}_compared'] 
                
                df = pd.concat([zero_baseline, not_zero_baseline], ignore_index=False).sort_index()
            
            if m != 'efficiency_overall':
                assert ~df[m].isna().all(), f"detected NaN in relative_improve delta of {m}"
            
    elif args.type == 'improve_to_perfect':
        for m in metrics:
            perfect_val = 1
            if 'rmse' in m: 
                perfect_val = 0
    
            df[m] = (df[f'{m}_compared'] - df[f'{m}_baseline']) / ( perfect_val - df[f'{m}_baseline'] )
        
            # deal with cases where baseline or compared has perfect perf
            
            # baseline and compared have perfect perf -> delta = 0
            both_perfect = df.loc[(df[f'{m}_compared']==perfect_val)&(df[f'{m}_baseline']==perfect_val)].copy()
            if both_perfect.shape[0] > 0: 
                other = df.loc[~((df[f'{m}_compared']==perfect_val)&(df[f'{m}_baseline']==perfect_val))]
                both_perfect[m] = 0
                df = pd.concat([other, both_perfect], ignore_index=False).sort_index()
                               
            
            # baseline has perfect perf and compared is worst -> delta is the absolute delta
            base_perfect_compared_worst = df.loc[(df[f'{m}_compared']!=perfect_val)&(df[f'{m}_baseline']==perfect_val)].copy()
            if base_perfect_compared_worst.shape[0] > 0:
                other = df.loc[~((df[f'{m}_compared']!=perfect_val)&(df[f'{m}_baseline']==perfect_val))]
                base_perfect_compared_worst[m] = base_perfect_compared_worst[f'{m}_compared'] - base_perfect_compared_worst[f'{m}_baseline']
                
                df = pd.concat([other, base_perfect_compared_worst], ignore_index=False).sort_index()

            if m != 'efficiency_overall':
                assert df[m].notna().all(), f"detected NaN in improve_to_perfect delta of {m}"
            
    return df


def aggregate(task_deltas,metrics,suffix):
    # aggregate
    means = task_deltas[metrics].mean()
    medians = task_deltas[metrics].median()
    pc25 = task_deltas[metrics].quantile(.25)
    pc75 = task_deltas[metrics].quantile(.75)
    delta_global_means = pd.DataFrame([means.values], columns=means.index).add_suffix("_mean")
    delta_global_medians = pd.DataFrame([medians.values], columns=medians.index).add_suffix("_median")
    delta_global_25 = pd.DataFrame([pc25.values], columns=medians.index).add_suffix("_25pc")
    delta_global_75 = pd.DataFrame([pc75.values], columns=medians.index).add_suffix("_75pc")
    delta_global = delta_global_means.join([delta_global_medians,delta_global_25,delta_global_75])
    
    delta_assay_type_mean = task_deltas.groupby('assay_type').mean()[metrics].add_suffix("_mean")
    delta_assay_type_median = task_deltas.groupby('assay_type').median()[metrics].add_suffix("_median")
    delta_assay_type_25 = task_deltas.groupby('assay_type').quantile(.25)[metrics].add_suffix("_25pc")
    delta_assay_type_75 = task_deltas.groupby('assay_type').quantile(.75)[metrics].add_suffix("_75pc")
    delta_assay_type = delta_assay_type_mean.join([delta_assay_type_median,delta_assay_type_25,delta_assay_type_75]).reset_index()

    # save
    if not os.path.isdir(args.outdir): 
        os.makedirs(args.outdir)
        os.makedirs(args.outdir + '/cdf')
        os.makedirs(args.outdir + '/NOUPLOAD')
            
    task_deltas.to_csv(os.path.join(args.outdir, f'NOUPLOAD/deltas_per-task_performances_NOUPLOAD{suffix}.csv'), index=None)
    delta_assay_type.to_csv(os.path.join(args.outdir, f'deltas_per-assay_performances{suffix}.csv'), index=None)
    delta_global.to_csv(os.path.join(args.outdir, f'deltas_global_performances{suffix}.csv'), index=None)
    if args.verbose:
        print(f" > NOUPLOAD/deltas_per-task_performances_NOUPLOAD{suffix}.csv")
        print(f" > deltas_per-assay_performances{suffix}.csv")
        print(f" > deltas_global_performances{suffix}.csv\n")
    return

    
def interpolate_ecdf(distribution):
    from statsmodels.distributions.empirical_distribution import ECDF
    ecdf = ECDF(distribution.values)
    return_values = np.linspace(-1,1,100)
    return ecdf(return_values).round(4), return_values

def calculate_ecdf(full_df, metrics, comparison):
    ecdf_df=pd.DataFrame()
    ecdf_df_assay_type=pd.DataFrame()
    for metric_bin in metrics:
        ecdf=interpolate_ecdf(full_df[f'{metric_bin}_{comparison}'])
        ecdf_df=pd.concat((ecdf_df, \
            pd.DataFrame({'Density':ecdf[0], \
            'Metric Value':ecdf[1], \
            'Metric':metric_bin})))

        for assay_type, grouped_df_metric in full_df.groupby('assay_type'):
            ecdf_at=interpolate_ecdf(grouped_df_metric[f'{metric_bin}_{comparison}'])
            ecdf_df_assay_type=pd.concat((ecdf_df_assay_type, \
            pd.DataFrame({'Density':ecdf_at[0], \
                'Metric Value':ecdf_at[1], \
                'Metric':metric_bin, \
                'Assay_type':assay_type})))
    return ecdf_df, ecdf_df_assay_type

def run_ecdf(baseline_compared_df, metrics, fn):
    bl_comp_ecdf = [calculate_ecdf(baseline_compared_df, metrics, comparison_type) for comparison_type in ['baseline','compared']]
    ecdf_fns = [f"cdf/{fn}_cdfbaseline-cdfcompared.csv", f"cdf/{fn}_cdfbaseline-cdfcompared_assay_type.csv"]
    for ecdf_idx, ecdf_merge_cols in enumerate([['Metric Value','Metric'], ['Metric Value','Metric','Assay_type']]):
        ecdf_out = bl_comp_ecdf[0][ecdf_idx].merge(bl_comp_ecdf[1][ecdf_idx],left_on=ecdf_merge_cols,right_on=ecdf_merge_cols,how='left', suffixes=[f' Baseline',f' Compared'])
        ecdf_out[f'Baseline-Compared_CDF'] = ecdf_out[f'Density Baseline'] - ecdf_out[f'Density Compared']
        ecdf_out.to_csv(os.path.join(args.outdir, ecdf_fns[ecdf_idx]),index=False)
    return

def run_(task_perf, metrics, baseline_n):
    for subset_name in args.subset:
        subset = None
        suffix=''
        if subset_name is not None: 
            if args.verbose: 
                print(f"Subset {subset_name.split(os.sep)[-1]}")
            subset=pd.read_csv(subset_name)['input_assay_id'].unique()
            suffix='_'+subset_name.split(os.sep)[-1].replace('.csv','')
        elif args.verbose: 
            print(f"Full set")
            
        if (baseline_n is not None) and (subset_name is None): #perform the baseline topn comparison here
            nrows=int(len(task_perf)*baseline_n)
            if nrows <10: #sensible number minimum number of 10 tasks for the aggregate & cdf calculation
                if args.verbose: 
                    print(f"{nrows} {suffix} tasks required which is too few, skipping")                
            else:
                for metric in metrics:
                    if args.verbose: 
                        print(f"Baseline top {baseline_n} for metric {metric}")
                    baseline_suffix=suffix+f'_baseline-topn_{baseline_n}_{metric}'
                    bl_task_deltas = compute_task_deltas(task_perf.sort_values(f'{metric}_baseline',ascending=False).head(nrows), metrics, subset=subset)
                    aggregate(bl_task_deltas,metrics,baseline_suffix)
                    run_ecdf(task_perf.sort_values(f'{metric}_baseline',ascending=False).head(nrows), metrics, suffix+f'_baseline-topn_{baseline_n}')
        else:
            task_deltas = compute_task_deltas( task_perf, metrics, subset=subset )     
            aggregate(task_deltas,metrics,suffix)
            run_ecdf(task_deltas, metrics, suffix)
            for delta_topn in args.delta_topn:
                if (delta_topn is not None) and (subset_name is None):
                    for metric in metrics:
                        if args.verbose: 
                            print(f"Delta top {delta_topn} for metric {metric}")
                        topn_suffix=suffix+f'_delta-topn_{delta_topn}_{metric}'
                        if subset_name is not None: 
                            nrows=int(len(subset)*delta_topn)
                        else:
                            nrows=int(len(task_deltas)*delta_topn)
                        if nrows <10: #sensible number minimum number of 10 tasks for the aggregate & cdf calculation
                            if args.verbose: 
                                print(f"{nrows} {suffix} tasks required which is too few, skipping")
                        else:         
                            dtopn_task_deltas=task_deltas.sort_values(metric,ascending=False).head(nrows)
                            aggregate(dtopn_task_deltas,metrics,topn_suffix)
                            run_ecdf(dtopn_task_deltas, metrics, suffix+f'_delta-topn_{delta_topn}_{metric}')

def main():

    assert not os.path.isdir(args.outdir), "specified output directory already exists"
        
    # if subset(s) are provided, then add an entity so that we also perform a no (default) subset analysis
    args.subset.append(None)
    args.baseline_topn.append(None)
    args.delta_topn.append(None)
    
    if args.verbose: 
        print(f"Save relative deltas under : {args.outdir}")

    task_perf, metrics = load_task_perfs()

    for baseline_n in args.baseline_topn:
        run_(task_perf.copy(), metrics, baseline_n)


if __name__ == '__main__':
    main()


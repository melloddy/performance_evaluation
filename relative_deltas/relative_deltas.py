import os
import argparse
import pandas as pd

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

parser.add_argument("--outdir",
                    type=str,
                    help="output directory into which the resultig files will be saved.",
                    required=True)

parser.add_argument("-v", "--verbose", 
                    action="store_true", 
                    help="verbosity", 
                    default=False)

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
        
        metrics = cls_metrics
        cols2use = ['input_assay_id', 'threshold', f'cont_classification_task_id', 'assay_type'] + metrics
        
        # verif that tasks can match (i.e. cls vs clsaux situation)
        merged = baseline_task_perf[cols2use].merge(compared_task_perf[cols2use], on=['input_assay_id', 'threshold', 'assay_type'], suffixes=('_baseline', '_compared'))
        assert merged.shape[0] == baseline_task_perf.shape[0], "Not able to match tasks between baseline and compared... Are your input_assay_ids consistent between both?"
    
    return merged, metrics


def compute_task_deltas(df_, metrics):
    """ from the merged dataframe returned above, """
    
    df = df_.copy()
    
    if args.type == 'absolute':
        for m in metrics:
            df[m] = (df[f'{m}_compared'] - df[f'{m}_baseline'])
    
    elif args.type == 'relative_improve':
        for m in metrics:
            df[m] = (df[f'{m}_compared'] - df[f'{m}_baseline']) / df[f'{m}_baseline']
            
    elif args.type == 'improve_to_perfect':
        for m in metrics:
            perfect_val = 1
            if 'rmse' in m: 
                perfect_val = 0
                
            df[m] = (df[f'{m}_compared'] - df[f'{m}_baseline']) / ( perfect_val - df[f'{m}_baseline'] )
        
    return df


def main():

    assert not os.path.isdir(args.outdir), "specified output directory already exists"
    
    task_perf, metrics = load_task_perfs()
    task_deltas = compute_task_deltas( task_perf, metrics )
    
    # aggregate
    means = task_deltas[metrics].mean()
    delta_global = pd.DataFrame([means.values], columns=means.index)
    delta_assay_type = task_deltas.groupby('assay_type').mean()[metrics].reset_index()
    
    # save
    os.makedirs(args.outdir)
    
    task_deltas.to_csv(os.path.join(args.outdir, 'deltas_per-task_performances_NOUPLOAD.csv'), index=None)
    delta_assay_type.to_csv(os.path.join(args.outdir, 'deltas_per-assay_performances.csv'), index=None)
    delta_global.to_csv(os.path.join(args.outdir, 'deltas_global_performances.csv'), index=None)
    
    if args.verbose:
        print(f"Saved relative deltas under : {args.outdir}")
        print(f" > deltas_per-task_performances_NOUPLOAD.csv")
        print(f" > deltas_per-assay_performances.csv")
        print(f" > deltas_global_performances.csv")

if __name__ == '__main__':
	main()

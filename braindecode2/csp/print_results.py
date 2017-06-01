#!/usr/bin/env python
from braindecode.scripts.print_results import ResultPrinter
import argparse
import numpy as np

class CSPResultPrinter(ResultPrinter):
    def __init__(self, folder_name, only_last_fold=False):
        self.only_last_fold=only_last_fold
        self._folder_name = folder_name
    
    def _add_best_and_best_epoch(self, formatted_result, misclasses):
        return # doesnt make sense for csp 
    
    def _print_templates(self):
        return # TODELAY: reactivate, maybe also store templates?
    
    def _add_misclasses(self, formatted_result, misclasses):
        for key in misclasses:
            # get last epoch from all folds
            # need transform in case we just have one experiment(?)
            this_misclasses = misclasses[key]
            if this_misclasses.dtype == object:
                # in case we have some with 10 folds, some with three folds
                # it can happen that the inner array is wrapped...
                this_misclasses = this_misclasses[0]
            if self.only_last_fold:
                this_misclasses = this_misclasses[-1]
            this_mean = np.mean(this_misclasses)
            #this_mean = this_misclasses[-1]
            formatted_result[key] = "{:5.2f}%".format((1 - this_mean) * 100)
            if (isinstance(this_misclasses, list) and 
                    len(this_misclasses) > 1): # only for crossval
                this_std = np.std(this_misclasses)
                formatted_result[key + '_std'] = "{:4.2f}%".format(
                    this_std * 100)
     
    def _compute_final_misclasses(self, result_list, add_valid_test=False):
        """ Compute final fold-averaged misclasses for all experiments.
        Also works if there are no folds(train-test case)"""
        final_misclasses = {}
        misclasses = [r['misclasses'] for r in result_list]
        misclass_keys = misclasses[0].keys() # train,test etc
        for key in misclass_keys:
            this_misclasses = [m[key] for m in misclasses]
            if self.only_last_fold:
                fold_averaged_misclasses = np.array(this_misclasses)[:,-1]
            else:
                # avg over folds
                fold_averaged_misclasses = np.mean(this_misclasses, axis=1)
            
            final_misclasses[key] = fold_averaged_misclasses
            
        return final_misclasses
def parse_command_line_arguments():
    parser = argparse.ArgumentParser(
        description="""Print results stored in a folder.
        Example: ./print_results data/models/conv_nets/ """
    )
    parser.add_argument('results_folder', action='store',
                        choices=None,
                        help='A folder with results to print')
    parser.add_argument('--last_fold', action='store_true',
                        help='Only print results for last fold.')
    args = parser.parse_args()
    return args

def main():
    args = parse_command_line_arguments()
    result_printer = CSPResultPrinter(args.results_folder, 
        only_last_fold=args.last_fold)
    result_printer.print_results()

if __name__ == "__main__":
    main()

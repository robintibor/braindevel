#!/usr/bin/env python
import argparse
import re
import yaml
import sys
from copy import deepcopy
import numpy as np
import csv
import datetime
from collections import OrderedDict
# Result class needed for deserializing(!)
from braindecode.results.results import (ResultPool, 
    DatasetAveragedResults, Result)

class ResultPrinter:
    def __init__(self, folder_name):
        self._folder_name = folder_name
    
    def print_results(self, templates=False,
            constants=False,
            individual_datasets=True,
            start=None, stop=None,
            params=None):
        print ("Printing results in {:s}:".format(self._folder_name))
        self._collect_parameters_and_results(start, stop, params)
        self._format_results()
        self._print(templates, constants, individual_datasets)
        
    def _collect_parameters_and_results(self, start, stop, params):
        self._result_pool = ResultPool()
        self._result_pool.load_results(self._folder_name, start, stop, params)
        if (self._result_pool.have_varying_datasets() or
                self._result_pool.have_varying_leave_out()):
            self._dataset_averaged_results = DatasetAveragedResults()
            self._dataset_averaged_results.extract_results(self._result_pool)

    def _format_results(self):
        self._formatted_results = []
        misclasses_per_experiment = self._result_pool.get_misclasses()
        for misclasses in misclasses_per_experiment:
            formatted_result = self._format_result(misclasses)
            self._formatted_results.append(formatted_result)
        
    def _format_result(self, misclasses):
        """ Format result for one experiment. """
        formatted_result = {}
        self._add_misclasses(formatted_result, misclasses)
        self._add_best_and_best_epoch(formatted_result, misclasses)
        return formatted_result
    
    def _add_misclasses(self, formatted_result, misclasses):
        """ Add misclasses for one experiment"""
        for key in misclasses:
            # get last epoch from all folds
            # need transform in case we just have one experiment(?)
            this_misclasses = self._atleast_2d_or_1d_of_arr(misclasses[key])
            final_misclasses = []
            for misclass_fold in this_misclasses:
                final_misclasses.append(misclass_fold[-1])
            this_mean = np.mean(final_misclasses)
            formatted_result[key] = "{:5.2f}%".format((1 - this_mean) * 100)
            if (len(final_misclasses) > 1): # only for crossval
                this_std = np.std(final_misclasses)
                formatted_result[key + '_std'] = "{:4.2f}%".format(
                    this_std * 100)
    
    def _add_best_and_best_epoch(self, formatted_result, misclasses):
        if 'test' in misclasses:
            # Determine minimal number of epochs and 
            # only take misclass rates until that epoch
            test_misclasses = deepcopy(misclasses['test'])
            # transform to list of one list in case of only one experiment
            test_misclasses = self._atleast_2d_or_1d_of_arr(test_misclasses)
            min_epoch_num = np.min([len(a) for a in test_misclasses])
            same_epochs_misclasses = [a[0:min_epoch_num] for a in test_misclasses]
            same_epochs_misclasses = np.array(same_epochs_misclasses)
            average_misclass = np.mean(same_epochs_misclasses, axis=0)
            best_epoch = np.argmin(average_misclass)
            best_misclass = average_misclass[best_epoch]
            formatted_result['best'] = "{:5.2f}%".format((1 - best_misclass) * 100)
            # +1 for 1-based indexing
            formatted_result['best_epoch'] = "{:3d}".format(best_epoch + 1)
           
    def _atleast_2d_or_1d_of_arr(self, arr):
        if not isinstance(arr[0], list) and arr.ndim == 1:
            return np.atleast_2d(arr)
        else:
            return arr 
            
    def _print(self, templates, constants, individual_datasets):
        if (templates):
            self._print_templates()
        if (constants):
            self._print_constant_parameters()
        if (individual_datasets):
            self._print_experiments_result_table()
        if (self._result_pool.have_varying_datasets() or
                self._result_pool.have_varying_leave_out()):
            self._print_experiments_averaged_datasets_result_table()
        
    def _print_templates(self):
        # templates should all be same so just print first one
        print("Templates ...")
        for name, template in self._result_pool.template().iteritems():
            print(name + ":")
            # substitute object tags that cannot be loaded properly
            #value = re.sub(r"!obj:[a-zA-Z0-9_.]*\.([A-Za-z_]*)", r"\1:",
            #   template)
            # remove anchors...
            value = re.sub(r"\&([A-Za-z_0-9]*)", "",
               template)
            value = re.sub(r"!obj:[a-zA-Z0-9_.]*\.([A-Za-z_]*)", r"&\1",
               value)
            # substitute stars => references to other objects
            # which may be somehwere in the actual yaml-training file...
            value = value.replace("*", "")
            #yaml.dump(yaml.load(value), sys.stdout, default_flow_style=False)
            print value
    
    def _print_constant_parameters(self):
        print("Parameters...")
        yaml.dump(self._result_pool.constant_params(), sys.stdout, 
            default_flow_style=False)
        print('')
    
    def _print_experiments_result_table(self):
        table_headers, table_rows = self._create_experiments_headers_and_rows()
        table_headers, table_rows = self._prettify_headers_rows(table_headers,
            table_rows)
        self._print_markdown_table(table_headers, table_rows)
        #self._print_csv_table(table_headers, table_rows)
        self._print_table(table_headers, table_rows)
    
    def _create_experiments_headers_and_rows(self):
        ''' Create table rows and headers for result table'''
        param_headers, param_rows = self._create_experiment_param_headers_and_rows()
        result_headers, result_rows = self._create_result_headers_and_rows()
        # Check that results and params have same amount of rows
        # and that each row has same size
        assert(len(param_rows) == len(result_rows))
        for i in range(len(param_rows)):
            assert(len(param_rows[i]) == len(param_rows[0]))
            assert(len(result_rows[i]) == len(result_rows[0]))
        
        table_headers = param_headers + result_headers
        # merge rows together...
        table_rows = [param_rows[i] + result_rows[i] for i in range(len(param_rows))]
        # some hacky header_substitutions for nicer printing
        header_substitutions = {
            'frequency_start': 'freq_start',
            'frequency_stop': 'freq_stop',
            'updates_per_batch': 'batch_updates',
            'dataset_filename': 'filename',
            'max_increasing_epochs': '>epochs'}
        table_headers = [header if not header_substitutions.has_key(header) \
            else header_substitutions[header] \
            for header in table_headers]
        return table_headers, table_rows
    
    def _create_experiment_param_headers_and_rows(self):
        varying_param_keys = self._result_pool.varying_params()[0].keys()
        # Try to put dataset filename in second column after id...
        # TODELAY: remove testfilename from this
        if (self._result_pool.have_varying_datasets()):
            if 'dataset_filename' in varying_param_keys:
                filenamekey = 'dataset_filename'
            else:
                filenamekey = 'filename'
            varying_param_keys.remove(filenamekey)
            param_headers = ['id'] + [filenamekey] + varying_param_keys   
        else:
            param_headers = ['id'] + varying_param_keys       
        param_rows = []
        for result_obj_id in range(self._result_pool.num_experiments()):
            param_row = []
            # Put result_obj id nr first
            file_name = self._result_pool.result_file_names()[result_obj_id]
            result_obj_file_nr = int(file_name.split('.result.pkl')[0].split('/')[-1])
            param_row.append(result_obj_file_nr)
            varying_params_this_result_obj = self._result_pool.varying_params()[result_obj_id]
            # Put dataset filename second if exist
            if (self._result_pool.have_varying_datasets()):
                filename = varying_params_this_result_obj[filenamekey]
                # remove unnecessary details of filename
                filename = re.sub(r"(./)?data/[^/]*/", '', str(filename))
                filename = re.sub(r"MoSc[0-9]*S[0-9]*R[0-9]*_ds10_", '',
                    filename)
                filename = re.sub(r"_autoclean_.*", '', filename)
                filename = re.sub(".BBCI.mat", '', filename)
                param_row.append(filename)
            # Put rest of parameters
            for param_key in varying_param_keys:
                param_value = varying_params_this_result_obj[param_key]
                # TODELAY: remove again this if
                if param_key == 'test_filename':
                    param_value = re.sub(r"(./)?data/[^/]*/", '', str(param_value))
                    param_value = re.sub(r"MoSc[0-9]*S[0-9]*R[0-9]*_ds10_", '',
                        param_value)
                    param_value = re.sub(r"_autoclean_.*", '', param_value)
                    param_value = re.sub(".BBCI.mat", '', param_value)
                param_row.append(param_value)
            param_rows.append(param_row)
        return param_headers, param_rows
    
    def _create_result_headers_and_rows(self):
        result_headers = []
        result_headers.append('time')
        with_standard_deviation = self._formatted_results[0].has_key('train_std')
        
        # just put 'epoch' not 'best_epoch' so that header is not so wide
        result_type_headers = ['test', 'test_sample', 'best', 'epoch', 'train', 'valid']
        for result_type in result_type_headers:
            # check if result exists, if yes add it
            if (result_type in self._formatted_results[0] or 
                'best_' + result_type in self._formatted_results[0]):
                result_headers.append(result_type)
                if (with_standard_deviation and
                       result_type in ['test', 'train', 'valid']):
                    result_headers.append('std')
        result_rows = []
        for result_obj_id in range(self._result_pool.num_experiments()):
            result_row = []
            results = self._formatted_results[result_obj_id]
            #append training time
            training_time = self._result_pool.training_times()[result_obj_id]
            result_row.append(str(datetime.timedelta(
                seconds=round(training_time))))
            for result_type in ['test', 'test_sample', 'best', 'best_epoch', 'train', 'valid']:
                if result_type in results:
                    result_row.append(results[result_type])
                    if (with_standard_deviation and
                            result_type in ['test', 'train', 'valid']):
                        result_row.append(results[result_type + "_std"])
            result_rows.append(result_row)
        return result_headers, result_rows
    
        
    def _print_experiments_averaged_datasets_result_table(self):
        print ("\n\nDataset-Averaged Results:")
        table_headers, table_rows = \
            self._create_averaged_dataset_headers_and_rows()
        table_headers, table_rows = self._prettify_headers_rows(table_headers,
            table_rows)
        self._print_table(table_headers, table_rows)
        self._print_markdown_table(table_headers, table_rows)
            
    def _create_averaged_dataset_headers_and_rows(self):
        ''' Create table rows and headers for result table averaged over
        different datasets.
        So in the end have a table for each unique parameter combination.'''
        # TODELAY: don't use this weird hack to find out if there is
        # a cross validation ;)
        all_result_lists = self._dataset_averaged_results.results()
        headers = self._create_dataset_averaged_headers(all_result_lists)
        rows = []
        for i, result_list in enumerate(all_result_lists):
            row = [str(i), str(len(result_list))] # id in table, number of files
            parameter_row = self._create_parameter_row(result_list, headers)
            row += parameter_row
            misclasses = self._compute_final_misclasses(result_list)
            training_times = [r['training_time'] for r in result_list]
            result_row = self._create_result_row(headers, misclasses,
                training_times)
            row += result_row
            rows.append(row)
        return headers, rows

    @ staticmethod
    def _create_dataset_averaged_headers(all_result_list):
        params = deepcopy(all_result_list[0][0]['parameters'])
        misclass_keys = all_result_list[0][0]['misclasses'].keys()
        # sort to show test before train before valid
        misclass_keys = sorted(misclass_keys, 
            key=lambda x: ('test' in x) * 1 + ('train' in x ) * 2 + 
            ('valid' in x ) * 3)
        result_keys = ["time", "std"]
        for key in misclass_keys:
            result_keys.append(key)
            result_keys.append('std')
        params.pop('filename', None)
        params.pop('dataset_filename', None)
        params.pop('transfer_leave_out', None)
        params.pop('test_filename', None)
        params.pop('trainer_filename', None)
        return ["id", "files"] + params.keys() + result_keys
    @staticmethod
    def _create_result_row(headers, misclasses, training_times):
        row = []
        # Add training times
        row += [str(datetime.timedelta(
                 seconds=round(np.mean(training_times))))]
        row += [str(datetime.timedelta(
                 seconds=round(np.std(training_times))))]
        # Add misclass for valid test train etc.
        assert len(set(misclasses.keys()) - set(headers)) == 0, ("All misclass"
            "keys should be in headers")
        # Add results in order they are defined in headers(!)
        for key in headers:
            if key in misclasses.keys():
                row += ["{:4.2f}%".format(100 - np.mean(misclasses[key] * 100))]
                row += ["{:4.2f}%".format(np.std(misclasses[key]) * 100)]
        return row
    
    @staticmethod 
    def _create_parameter_row(result_list, headers):
        parameters = deepcopy(result_list[0]['parameters'])
        parameters.pop('dataset_filename', None)
        parameters.pop('filename', None)
        sorted_params = []
        for header in headers:
            if header in parameters:
                sorted_params.append(parameters[header])
        return sorted_params
        
        
    def _compute_final_misclasses(self, result_list):
        """ Compute final fold-averaged misclasses for all experiments.
        Also works if there are no folds(train-test case)"""
        final_misclasses = {}
        misclasses = [r['misclasses'] for r in result_list]
        misclass_keys = misclasses[0].keys() # train,test etc
        get_last_elem = np.vectorize(lambda a : a[-1])
        for key in misclass_keys:
            this_misclasses = [m[key] for m in misclasses]
            if  np.array(this_misclasses[0][0]).shape != ():
                this_final_misclasses = get_last_elem(this_misclasses)
            else:
                # can only happen in case all experiments have same number of
                # epochs
                this_final_misclasses = np.array(this_misclasses)[:,-1:]
            
            # avg over folds if necessary
            this_final_avg_misclasses = np.mean(this_final_misclasses, axis=1)
            final_misclasses[key] = this_final_avg_misclasses
            
        return final_misclasses
        
    def _prettify_headers_rows(self, table_headers, table_rows):
        """ Shorten names for nicer printing """
        pretty_headers = []
        substitutions = OrderedDict([
          ('frequency_start', 'freq_start'),
          ('frequency_stop', 'freq_stop'),
          ('updates_per_batch', 'batch_updates'),
          ('dataset_filename', 'filename'),
          ('max_increasing_epochs', '>epochs'),
          ('first_kernel_shape', 'k_shape1'),
          ('first_kernel_stride', 'k_stride1'),
          ('first_pool_shape', 'p_shape1'),
          ('first_pool_stride', 'p_stride1'),
          ('second_kernel_shape', 'k_shape2'),
          ('second_kernel_stride', 'k_stride2'),
          ('second_pool_shape', 'p_shape2'),
          ('second_pool_stride', 'p_stride2'),
          ('learning_', 'l'),
          ('momentum_rule', 'momentum'),
          ('sgd_callbacks', 'sgd_calls'),
          ('anneal', 'an'),
          ('max_kernel_norm', 'kernel_norm'),
          ('spatial', 'spat'),
          ('dropout', 'drop'),
          ('transform_function_and_args', 'transform_func'),
          ('square_amplitude', 'amplitude^2'),
          ('divide_win_length', 'div_win_len'),
          ('fraction_of_lowest_epoch' , 'frac_epochs'),
          ('kernel_shape', 'k_shape'),
          ('wanted_classes', 'classes'),
          ('!obj,pylearn2.models.mlp.IdentityConvNonlinearity { }', 'identity'),
          ('!obj,pylearn2.models.mlp.RectifierConvNonlinearity { }', 'rectified'),
          ('*two_file_splitter', 'two_file'),
          ('*train_test_splitter', 'train_test'),
          ('set_cleaner', 'set'),
          ('dataset_splitter', 'splitter'),
          ('train_cleaner', 'cleaner'),
          ('_func', ''),
          ('all_EEG_sensors', 'all_EEG'),
          ('stop_when_no_improvement', 'stop_improve'),
          ('num_selected_features', 'sel_feat'),
          ('num_selected_filterbands', 'sel_filt'),
          ('forward_steps', 'for'),
          ('backward_steps', 'back'),
          ('fraction_outside_trials', 'frac_out_trial'),
          ('updates_per_epoch', 'epoch_ups'),
          ('n_temporal_units', 't_units'),
          ('n_spat_units', 'spat_units'),
          ('use_test_as_valid', 'test=valid'),
          ('imbalance_factor', 'imba_factor'),
          ('n_sample_preds', 'preds'),
          ('pool_time_stride', 'pool_str'),
          ('pool_time_length', 'p_len'),
          ('oversample_targets', 'oversample'),
          ('first', '1st'),
          ('average_exc_pad', 'mean'),
          ('SumPool2dLayer', 'Sum'),
          ('Pool2DLayer', 'Pool'),
          ('final_dense_length', 'final_len'),
          ('batch_size', '#batch'),
          ('preprocessor', 'preproc'),
          ('standardize_series_wise', 'preproc_series'),
          ('remove_baseline_mean', 'rm_baseline'),
          ('input_time_length', 'input_len'),
          ('online_chan_freq_wise', 'online_c_1'),
          ('n_selected_features', 'n_features'),
          ('filter_length_', 'f_len'),
          ('filter_length', 'f_len'),
          ('filter_time_length', 'f_time_len'),
          ('num_filters_', 'n_f'),
          ('num_filters', 'n_f'),
          ])
        for header in table_headers:
            pretty_header = header
            for key in substitutions:
                pretty_header = pretty_header.replace(key, substitutions[key])
            pretty_headers.append(pretty_header)
        pretty_rows = []
        for row in table_rows:
            pretty_row = []
            # go through all parameter values and prettify
            for value in row:
                pretty_value = str(value)
                for key in substitutions:
                    pretty_value = pretty_value.replace(key, substitutions[key])
                pretty_row.append(pretty_value)
            pretty_rows.append(pretty_row)
            
        return pretty_headers, pretty_rows  
        
        
    def _print_table(self, headers, rows):
        print("\nTerminal Table\n")
        # first determine maximum length for any column
        # two spaces padding
        max_col_lengths = [0] * len(headers)
        for column in range(len(headers)):
            if (len(str(headers[column])) > max_col_lengths[column]):
                max_col_lengths[column] = len(str(headers[column]))
            for row in range(len(rows)):
                if(len(str(rows[row][column])) > max_col_lengths[column]):
                    max_col_lengths[column] = len(str(rows[row][column]))
        
        for index, header in enumerate(headers):
            length = max_col_lengths[index] + 2
            sys.stdout.write(("{:<" + str(length) + "}").format(header))
        print('')
        for row in rows:
            for column, value in enumerate(row):
                length = max_col_lengths[column] + 2
                sys.stdout.write(("{:<" + str(length) + "}").format(value))
            print('')
    
    def _print_csv_table(self, headers, rows):
        print("\nCSV Table\n")
        wr = csv.writer(sys.stdout)
        wr.writerow(headers)
        for row in rows:
            rowstrings = [str(value) for value in row]
            rowstrings = [rowstr.replace("$", "") for rowstr in rowstrings]
            wr.writerow(rowstrings)
            
    def _print_markdown_table(self, headers, rows):
        print("\nMarkdown Table\n")
        headerline = "|".join(headers)
        headerline = "|" + headerline + "|"
        print headerline
        
        # make seperatorline for table
        seperatorline = "|".join(["-" for _ in headers])
        seperatorline = "|" + seperatorline + "|"
        print seperatorline
        for row in rows:
            rowstrings = [str(value) for value in row]
            rowstrings = [rowstr.replace("$", "") for rowstr in rowstrings]
            rowline = "|".join(rowstrings)
            rowline = "|" + rowline + "|"
            print rowline
    
def parse_command_line_arguments():
    parser = argparse.ArgumentParser(
        description="""Print results stored in a folder.
        Example: ./print_results data/models/conv_nets/ """
    )
    parser.add_argument('results_folder', action='store',
                        choices=None,
                        help='A folder with results to print')
    args = parser.parse_args()
    return args

def main():
    args = parse_command_line_arguments()
    result_printer = ResultPrinter(args.results_folder)
    result_printer.print_results()

if __name__ == "__main__":
    main()

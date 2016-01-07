from braindecode.datahandling.splitters import PreprocessedSplitter

def get_full_and_merged_preprocessed_sets(train_dict):
    dataset = train_dict['dataset']
    full_set = train_dict['dataset']
    full_set.load()
    preproc = train_dict['exp_args']['preprocessor']
    splitter = train_dict['dataset_splitter']
    dataprovider = PreprocessedSplitter(splitter, preproc)

    preproc_sets = dataprovider.get_train_merged_valid_test(full_set)
    return full_set, preproc_sets
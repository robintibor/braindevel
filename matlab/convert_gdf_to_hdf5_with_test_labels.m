%% Produce combined files with train and test(evaluation files concatenated)
% ( from bci competition IV dataset 2a) .. assumes you created mat files
% before from the bci competition gdf files... and downloaded true label
% mat files

bci_competition_dir = 'data/bci-competition/BCICIV_2a/';
hdf5_dir = fullfile(bci_competition_dir, '/hdf5/');
true_label_dir = fullfile(bci_competition_dir, '/true_labels/');
combined_dir = fullfile(bci_competition_dir, '/hdf5-combined/');

train_files = dir(fullfile(hdf5_dir, '*T.mat'));

for i_name = 1:numel(train_files)
    train_file_name = train_files(i_name).name;
    fprintf('Processing %s...\n', train_file_name);
    % Load train test signal header
    % This part would have to be changed to load from gdf in case separated
    % mat files for train / test(evaluation) were not created
    [~,train_base_name,~] = fileparts(train_file_name);
    train_file_path = fullfile(hdf5_dir, strcat(train_base_name, '.mat'));
    train_combined = load(train_file_path);
    train_signal = train_combined.signal;
    train_header = train_combined.header;

    test_base_name = strcat(train_base_name(1:end-1), 'E');
    test_file_path= fullfile(hdf5_dir,  strcat(test_base_name, '.mat'));
    test_combined = load(test_file_path);
    test_signal = test_combined.signal;
    test_header = test_combined.header;
    % Finished loading of signal header...


    train_label_file_path = fullfile(true_label_dir, strcat(train_base_name, '.mat'));
    test_label_file_path = fullfile(true_label_dir, strcat(test_base_name, '.mat'));

    % load train labels just for check of correctness
    train_labels = load(train_label_file_path);
    train_labels = train_labels.classlabel;

    assert(isequal(train_header.Classlabel, train_labels));
    train_label_events = train_header.EVENT.TYP(train_header.EVENT.TYP >= 769 & ...
        train_header.EVENT.TYP <=772);
    assert(isequal(train_label_events, train_labels + 768));
    
    % Load test labels
    test_labels = load(test_label_file_path);
    test_labels = test_labels.classlabel;

    % check there are only unknown labels in test labels
    assert(all(test_header.EVENT.TYP(test_header.EVENT.TYP >= 769 & ...
        test_header.EVENT.TYP <= 783) == 783));
    test_header.Classlabel = test_labels;
    test_header.EVENT.TYP(test_header.EVENT.TYP == 783) = test_labels + 768;
    % Check this worked out fine.
    test_label_events = test_header.EVENT.TYP(test_header.EVENT.TYP >= 769 & ...
        test_header.EVENT.TYP <=772);
    assert(isequal(test_label_events, test_labels + 768));

    % concatenate both signals
    combined_signal = [train_signal; test_signal];
    combined_header = train_header;

    %concatenate both event structures (pos and typ) and classlabels also
    combined_header.Classlabel = [train_labels; test_labels];
    combined_header.EVENT.TYP = [train_header.EVENT.TYP; test_header.EVENT.TYP];
    combined_header.EVENT.TYP = [train_header.EVENT.TYP; test_header.EVENT.TYP];
    combined_header.EVENT.DUR =  [train_header.EVENT.DUR; test_header.EVENT.DUR];
    combined_header.EVENT.CHN =  [train_header.EVENT.CHN; test_header.EVENT.CHN];
    % add offset for test pos (number of samples in train signal) 
    combined_header.EVENT.POS = [train_header.EVENT.POS; ...
        test_header.EVENT.POS + size(train_signal, 1)];
    combined_file_name = strcat(train_base_name(1:3), 'TE.mat');
    combined_file_path = fullfile(combined_dir, combined_file_name);

    signal = combined_signal;
    header = combined_header;
    % now save...
    save(combined_file_path, 'signal', 'header', '-v7.3');
end

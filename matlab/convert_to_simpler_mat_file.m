%% Legacy script just commited in case it is needed... removable...
bci_competition_dir = 'data/bci-competition/BCICIV_2a/';
old_hdf5_dir = fullfile(bci_competition_dir, '/hdf5/');
new_hdf5_dir = fullfile(bci_competition_dir, '/hdf5new/');

old_mat_files = dir(fullfile(old_hdf5_dir, '*.mat'));


for i_name = 1:numel(old_mat_files)
   file_name = old_mat_files(i_name).name;
   file_path = fullfile(old_hdf5_dir, file_name);
   fprintf('Processing %s...\n', file_path);
   combined = load(file_path);
   signal = combined.combined.signal;
   header = combined.combined.header;
   [~,basename,~] = fileparts(file_name);
   new_mat_file_path = fullfile(new_hdf5_dir, strcat(basename, '.mat'));
   save(new_mat_file_path, 'signal', 'header', '-v7.3');
end
%% Convert bci competition gdf files to matlab/hdf5 files
% Assumes files to be converted are in those dirs, also assumes hdf5 subfolder already exists

bci_competition_dir = 'data/bci-competition/BCICIV_2a/';
gdf_dir = fullfile(bci_competition_dir, '/gdf/');
files = dir(fullfile(gdf_dir, '*.gdf'));
hdf5_dir = fullfile(bci_competition_dir, '/hdf5/');

for i_name = 1:numel(files)
   file_name = files(i_name).name;
   file_path = fullfile(gdf_dir, file_name);
   fprintf('Processing %s...\n', file_path);
   [signal, header] = sload(file_path);
   [~,basename,~] = fileparts(file_name);
   mat_file_path = fullfile(hdf5_dir, strcat(basename, '.mat'));
   save(mat_file_path, 'signal', 'header', '-v7.3');
end
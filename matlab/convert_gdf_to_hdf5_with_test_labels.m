%% Convert bci competition gdf files to matlab/hdf5 files
% Assumes files to be converted are in those dirs, also assumes hdf5 subfolder already exists

bci_competition_dir = 'data/bci-competition/BCICIV_2a/';
gdf_dir = fullfile(bci_competition_dir, '/gdf/');
files = dir(fullfile(gdf_dir, '*.gdf'));
hdf5_dir = fullfile(bci_competition_dir, '/hdf5/');
true_label_dir = fullfile(bci_competition_dir, 'true_labels');
for i_name = 1:numel(files)
   file_name = files(i_name).name;
   file_path = fullfile(gdf_dir, file_name);
   fprintf('Processing %s...\n', file_path);
   [s, h] = sload(file_path);
   [~,basename,~] = fileparts(file_name);
   combined = struct('signal', s, 'header', h);
   mat_file_path = fullfile(hdf5_dir, strcat(basename, '.mat'));
   save(mat_file_path, 'combined', '-v7.3');
end

i_name=1;
file_name = files(i_name).name;
file_path = fullfile(gdf_dir, file_name);

fprintf('Processing %s...\n', file_path);
[s, h] = sload(file_path);
[~,basename,~] = fileparts(file_name);

label_file_name = fullfile(true_label_dir, strcat(basename, '.mat'));

labels = load(label_file_name);
labels = labels.classlabel;

h.Classlabel = labels;
eventtypes = labels;
% change event types to match the 76... structure
% add offset to event pos (number pof samples in first signal
% concatenate both signals
%concatenate both event structures (pos and typ) and classlabels also
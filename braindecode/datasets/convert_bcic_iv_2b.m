folder = 'BCICIV_2b_gdf';
files = dir(fullfile(folder, '*.gdf'));
for f = files'
    full_name = fullfile(folder, f.name);
    fprintf('Processing %s...\n', full_name);
    [signal, header] = sload(full_name);
    new_name = strrep(full_name, '.gdf', '.hdf5');
    fprintf('Saving to %s...\n', new_name);
    save(new_name, 'signal', 'header', '-v7.3')
end

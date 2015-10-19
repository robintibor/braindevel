import numpy as np
cap_positions = \
    [['EOG_H',[],'Sync',[],[],[],[],[],'Fp1',[],'FPz',[],'Fp2',[],[],[],[],[],[],[],'EOG_V'],\
    [[],[],[],[],[],[],[],'AFp3h',[],[],[],[],[],'Afp4h',[],[],[],[],[],[],[]],\
    [[],[],'AF7',[],[],[],'AF3',[],[],[],'AFz',[],[],[],'AF4',[],[],[],'AF8',[],[]],\
    [[],[],[],[],[],'AFF5h',[],[],[],'AFF1',[],'AFF2',[],[],[],'AFF6h',[],[],[],[],[]],\
    [[],[],'F7',[],'F5',[],'F3',[],'F1',[],'Fz',[],'F2',[],'F4',[],'F6',[],'F8',[],[]],\
    [[],'FFT9h',[],'FFT7h',[],'FFC5h',[],'FFC3h',[],'FFC1h',[],'FFC2h',[],'FFC4h',[],'FFC6h',[],'FFT8h',[],'FFT10h',[]],\
    ['FT9',[],'FT7',[],'FC5',[],'FC3',[],'FC1',[],'FCz',[],'FC2',[],'FC4',[],'FC6',[],'FT8',[],'FT10'],\
    [[],'FTT9h',[],'FTT7h',[],'FCC5h',[],'FCC3h',[],'FCC1h',[],'FCC2h',[],'FCC4h',[],'FCC6h',[],'FTT8h',[],'FTT10h',[]],\
    ['M1',[],'T7',[],'C5',[],'C3',[],'C1',[],'Cz',[],'C2',[],'C4',[],'C6',[],'T8',[],'M2'],\
    [[],[],[],'TTP7h',[],'CCP5h',[],'CCP3h',[],'CCP1h',[],'CCP2h',[],'CCP4h',[],'CCP6h',[],'TTP8h',[],[],[]],\
    [[],[],'TP7',[],'CP5',[],'CP3',[],'CP1',[],'CPz',[],'CP2',[],'CP4',[],'CP6',[],'TP8',[],[]],\
    [[],'TPP9h',[],'TPP7h',[],'CPP5h',[],'CPP3h',[],'CPP1h',[],'CPP2h',[],'CPP4h',[],'CPP6h',[],'TPP8h',[],'TPP10h',[]],\
    ['P9',[],'P7',[],'P5',[],'P3',[],'P1',[],'Pz',[],'P2',[],'P4',[],'P6',[],'P8',[],'P10'],\
    [[],'PPO9h',[],[],[],'PPO5h',[],[],[],'PPO1',[],'PPO2',[],[],[],'PPO6h',[],[],[],'PPO10h',[]],\
    ['PO9',[],'PO7',[],'PO5',[],'PO3',[],[],[],'POz',[],[],[],'PO4',[],'PO6',[],'PO8',[],'PO10'],\
    [[],'POO9h',[],[],[],[],[],'POO3h',[],[],[],[],[],'POO4h',[],[],[],[],[],'POO10h',[]],\
    [[],[],[],[],[],[],[],[],'O1',[],'Oz',[],'O2',[],[],[],[],[],[],[],[]],\
    [[],[],[],[],[],[],[],[],[],'OI1h',[],'OI2h',[],[],[],[],[],[],[],[],[]],\
    ['EMG_LH',[],'EMG_LF',[],[],[],[],[],'I1',[],'Iz',[],'I2',[],[],[],[],[],'EMG_RF',[],'EMG_RH']]

tight_C_positions = [
    ['FFC5h','FFC3h','FFC1h',[],'FFC2h','FFC4h','FFC6h'],
    ['FC5','FC3','FC1','FCz','FC2','FC4','FC6'],
    ['FCC5h','FCC3h','FCC1h',[],'FCC2h','FCC4h','FCC6h'],
    ['C5','C3','C1','Cz','C2','C4','C6'],
    ['CCP5h','CCP3h','CCP1h',[],'CCP2h','CCP4h','CCP6h'],
    ['CP5','CP3','CP1','CPz','CP2','CP4','CP6'],
    ['CPP5h','CPP3h','CPP1h',[],'CPP2h','CPP4h','CPP6h']]

tight_Kaggle_positions =  [
    [[],[],'Fp1',[],'FP2',[],[]],
    [[],'F7','F3','Fz','F4','F8',[]],
    [[],'FC5', 'FC1',[],'FC2','FC6',[]],\
    [[],'T7','C3','Cz','C4','T8',[]],\
    ['TP9','CP5','CP1',[],'CP2','CP6','TP10'],\
    [[], 'P7', 'P3','Pz','P4','P8', []],\
    [[],'PO9','O1','Oz','O2','PO10',[]]]
    
def sort_topologically(sensor_names):
    """ Get sensors in a topologically sensible order.
    
    >>> sort_topologically(['O1', 'FP1'])
    ['FP1', 'O1']
    >>> sort_topologically(['FP2', 'FP1'])
    ['FP1', 'FP2']
    >>> sort_topologically(['O1', 'FP1', 'FP2'])
    ['FP1', 'FP2', 'O1']

    
    Check that sensors are sorted row-wise, i.e. first row, then second row,
    and inside rows sorted by column 
    >>> sort_topologically(['FP1', 'POO9h', 'FP2', 'POO3h'])
    ['FP1', 'FP2', 'POO9h', 'POO3h']
    
    # Check that all sensors exist
    >>> sort_topologically(['O5', 'POO9h', 'FP2', 'POO3h'])
    Traceback (most recent call last):
        ...
    AssertionError: Expect all sensors to exist in topo grid, not existing: set(['O5'])
    """
    flat_topo_all_sensors = np.array(cap_positions).flatten();
    sorted_sensor_names = []
    # Go through all sorted sensors and add those that are requested
    for sensor_name in flat_topo_all_sensors:
        for sensor_name_wanted in sensor_names:
            if sensor_name.lower() == sensor_name_wanted.lower():
                sorted_sensor_names.append(sensor_name_wanted)
    assert len(set(sensor_names) - set(sorted_sensor_names)) == 0, \
        "Expect all sensors to exist in topo grid, not existing: {:s}".format(
            str(set(sensor_names) - set(sorted_sensor_names)))
    return sorted_sensor_names
    
def get_sensor_pos(sensor_name, sensor_map=cap_positions):
    sensor_pos = np.where(np.char.lower(np.char.array(sensor_map)) == sensor_name.lower())
    # unpack them: they are 1-dimensional arrays before
    assert len(sensor_pos[0]) == 1, ("there should be a position for the sensor "
        "{:s}".format(sensor_name))
    return sensor_pos[0][0], sensor_pos[1][0]

def get_cap_shape():
    return np.array(cap_positions).shape

def get_C_sensors():
    C_sensors= ['FC5', 'FC1', 'FC2', 'FC6', 'C3', 'Cz', 'C4', 'CP5',
        'CP1', 'CP2', 'CP6', 'FC3', 'FCz', 'FC4', 'C5', 'C1', 'C2', 'C6',
        'CP3', 'CPz', 'CP4', 'FFC5h', 'FFC3h', 'FFC4h', 'FFC6h', 'FCC5h',
        'FCC3h', 'FCC4h', 'FCC6h', 'CCP5h', 'CCP3h', 'CCP4h', 'CCP6h', 'CPP5h',
        'CPP3h', 'CPP4h', 'CPP6h', 'FFC1h', 'FFC2h', 'FCC1h', 'FCC2h', 'CCP1h',
         'CCP2h', 'CPP1h', 'CPP2h']
    return C_sensors

def get_C_sensors_sorted():
    return sort_topologically(get_C_sensors())

def get_EEG_sensors():
    return ['Fp1', 'Fp2', 'Fpz', 'F7', 'F3', 'Fz', 'F4', 'F8',
            'FC5', 'FC1', 'FC2', 'FC6', 'M1', 'T7', 'C3', 'Cz', 'C4', 'T8', 'M2',
            'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8', 'POz', 'O1',
            'Oz', 'O2', 'AF7', 'AF3', 'AF4', 'AF8', 'F5', 'F1', 'F2', 'F6', 'FC3',
            'FCz', 'FC4', 'C5', 'C1', 'C2', 'C6', 'CP3', 'CPz', 'CP4', 'P5', 'P1',
            'P2', 'P6', 'PO5', 'PO3', 'PO4', 'PO6', 'FT7', 'FT8', 'TP7', 'TP8',
            'PO7', 'PO8', 'FT9', 'FT10', 'TPP9h', 'TPP10h', 'PO9', 'PO10', 'P9',
            'P10', 'AFF1', 'AFz', 'AFF2', 'FFC5h', 'FFC3h', 'FFC4h', 'FFC6h', 'FCC5h',
            'FCC3h', 'FCC4h', 'FCC6h', 'CCP5h', 'CCP3h', 'CCP4h', 'CCP6h', 'CPP5h',
            'CPP3h', 'CPP4h', 'CPP6h', 'PPO1', 'PPO2', 'I1', 'Iz', 'I2', 'AFp3h',
            'AFp4h', 'AFF5h', 'AFF6h', 'FFT7h', 'FFC1h', 'FFC2h', 'FFT8h', 'FTT9h',
            'FTT7h', 'FCC1h', 'FCC2h', 'FTT8h', 'FTT10h', 'TTP7h', 'CCP1h', 'CCP2h',
            'TTP8h', 'TPP7h', 'CPP1h', 'CPP2h', 'TPP8h', 'PPO9h', 'PPO5h', 'PPO6h',
            'PPO10h', 'POO9h', 'POO3h', 'POO4h', 'POO10h', 'OI1h', 'OI2h']

def get_EEG_sensors_sorted():
    return sort_topologically(get_EEG_sensors())

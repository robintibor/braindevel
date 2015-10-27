### Files to edit

# config.cfg
  * change path in 'function = ...' such that it finds datasets specified in runRobin.py
  * number_cv_folds has to match len(dataset_list) * 10 in runRobin.py

# runRobin.py
  * check dataset names (see above)

# HPOlib.cmd
  * example command of how to call HPOlib
  * change '/home/eggenspk/informatikhome/eggenspk/Projects/HPOlib/optimizers/smac/smac_2_08_00-master' to your location of SMAC [downloaded with HPOlib;typically in HPOlib/optimizers/smac/smac_2_08_master]

# startup.cmd
  * example of startup cmd, invoke before running somehting on cluster
  * change first line, such that virtualenv [or whatever] is loaded, if necessary
  * change last line, such that "import motor_imagery_learning.hyperopt.hyperopt as hyperopt" works
  * double-check THEANO_FLAGS

# smac_2_08_00-master/bandpower_params.pcs
  * used parameter space, adjust to your needs

#### How to run HPOlib
[1] Install HPOlib, see https://github.com/automl/HPOlib/tree/development ***Use development branch***
[2] Edit all files
[3] Qlogin to a GPU
[4] run startup script, be sure that THEANOFLAGS (especially compiledir) is set
[5] run command in HPOlib.cmd
[6] Watch output

This will run HPOlib for 100*18 configs, which is probably too long. You can reduced #datasets [and number_cv_folds] and number_of_jobs.


#### Debugging
Each HPOlib cmd creates a new folder, called smac_2_08_00-master_1000_date_and_time/
Output of currently running configs and crashes are stored as *instance.out
To manually run a config see first lines of corresponding *solver.out

### Create experiment
Call ./scripts/create_hyperopt_files.py with wanted experiments file name
Create a new folder for the experiment (below hyperopt folder?)
Copy contents of robin_rawnet there, exchange .pcs file and templates.yaml file 
with the files created with the script
Adapt as config.cfg written above 

### Robin Commands
(source your gpu-startup-file)
export PYTHONPATH=$PYTHONPATH:/home/schirrmr/motor-imagery/code/motor_imagery_learning/
cd robin_rawnet
HPOlib-run -o /home/schirrmr/motor-imagery/code/HPOlib/optimizers/smac/smac_2_08_00-master -s 1


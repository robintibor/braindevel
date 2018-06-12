DOCTEST_FILES =  experiments/parse.py  datahandling/preprocessing.py

# always run test.. necessary since we have a test directory 
.PHONY: unit-tests acceptance-tests

unit-tests:
	THEANO_FLAGS=optimizer=fast_compile,device=cpu,floatX=float32 py.test --doctest-modules test/unit_tests/ $(DOCTEST_FILES)

unit-tests-verbose:
	THEANO_FLAGS=optimizer=fast_compile,device=cpu,floatX=float32 py.test -vv --doctest-modules test/unit_tests/ $(DOCTEST_FILES)

acceptance-tests:
	THEANO_FLAGS=optimizer=fast_compile,device=cpu,floatX=float32 py.test -vv test/acceptance_tests

#. ../../../motor-imagery/ipython-notebooks-metagpu/venv/bin/activate && 

# set export XDG_RUNTIME_DIR="" for slurm 
run-ipython-notebooks-top-level:
	cd /home/schirrmr && XDG_RUNTIME_DIR="" /home/schirrmr/anaconda3/envs/default/bin/jupyter notebook --config=/home/schirrmr/.jupyter/profile_nbserver/ipython_config.py

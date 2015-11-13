DOCTEST_FILES =  experiments/parse.py  datahandling/preprocessing.py

# always run test.. necessary since we have a test directory 
.PHONY: unit-tests acceptance-tests

unit-tests:
	THEANO_FLAGS=mode=FAST_COMPILE,device=cpu,floatX=float32 py.test --doctest-modules test/unit_tests/ $(DOCTEST_FILES)

unit-tests-verbose:
	THEANO_FLAGS=mode=FAST_COMPILE,device=cpu,floatX=float32 py.test -vv --doctest-modules test/unit_tests/ $(DOCTEST_FILES)

acceptance-tests:
	THEANO_FLAGS=mode=FAST_COMPILE,device=cpu,floatX=float32 py.test -vv test/acceptance_tests

run-ipython-notebooks-top-level:
	. ../../../motor-imagery/ipython-notebooks-metagpu/venv/bin/activate && cd /home/schirrmr && ipython notebook --profile=nbserver

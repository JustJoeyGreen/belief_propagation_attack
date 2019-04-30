clean:
	rm -f -r build/
	rm -f belief_propagation_attack/*.so

clean-pyc:
	rm -f belief_propagation_attack/*.pyc

clean-c:
	rm -f belief_propagation_attack/*.c

clean-output:
	rm -f graphs/*
	rm -f output/*

clean-logs:
	rm -f -r logs/training
	rm -f -r logs/validation

clean-all: clean clean-pyc clean-c clean-output

dirs:
	mkdir -p graphs/ leakage/ logs/ models/ output/ templates/

.PHONY: build
build: clean dirs
	python setup.py build_ext --inplace
	mv *.so belief_propagation_attack/


.PHONY: cython-build dirs
cython-build: clean clean-pyc clean-c dirs
	python setup.py build_ext --inplace --use-cython
	mv *.so belief_propagation_attack/

.PHONY: cython-build
quick-build: dirs
	python setup.py build_ext --inplace --use-cython
	mv *.so belief_propagation_attack/

quick-build3: dirs
	python3 setup.py build_ext --inplace --use-cython
	mv *.so belief_propagation_attack/

install:
	pip install -r REQUIREMENTS.txt

test:
	python tests/test_utility.py

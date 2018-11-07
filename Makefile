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

clean-all: clean clean-pyc clean-c clean-output

.PHONY: build
build: clean
	python setup.py build_ext --inplace
	mv *.so belief_propagation_attack/
	mkdir -p graphs/

.PHONY: cython-build
cython-build: clean clean-pyc clean-c
	python setup.py build_ext --inplace --use-cython
	mv *.so belief_propagation_attack/
	mkdir -p graphs/

install:
	pip install -r REQUIREMENTS.txt

test:
	python tests/test_utility.py

#!/bin/bash

run_test() {
	./compile.sh train.cpp $1
	start=$SECONDS
	./train -i mnist_train.csv -c 100 -I 10 -l 0.01 -s network.bin
	duration=$(( SECONDS - start ))
	#return duration
	echo $duration
}


run_test 1
run_test 2
run_test 3
run_test 4
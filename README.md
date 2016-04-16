This is a deep learning network implementation based on [Karpathy's python implementation](http://karpathy.github.io/2015/05/21/rnn-effectiveness/). 

#Compilation
--------

You have the choice of compiling against different BLAS libraries relying either on the CPU (standard blas or openblas), CUDA, or Open CL). Please see the compile.sh script for options. 

Example compilation - compile the network for OpenCL: 
```
./compile.sh train.cpp 1
./compile.sh predict.cpp 1
```

#Execution
----------

Train with a data set in a csv file, 100 nodes, 10 iterations over the complete dataset, and a learning rate of 0.01. 

```
./train -i mnist_train.csv -c 100 -I 10 -l 0.01 -s network.bin
```


Predict from a data set in a csv file 

```
./predict -i mnist_train.csv -n network.bin -i mnist_test.csv
```

##Speed
Tests on an NVIDIA Gefore GTX 770 were quite disappointing. A script for testing is included as timings.sh. 


#TODO
-----

* refactor code: split classes into different files
* save and load network states: 
	save: Parameters class needs to be serialised or network state variables saved into binary file/files
	- load: Parameters class: initialise and set weight matrices from disk
* learn from streaming data: 
	- allow piping similar to linux/unix programs
	- more flexible input parameters (use param library?)
* split into learn and predict functions:
	- learn: only learns and saves the network state
	- predict: load a network state and - given an input - output the result?
* statistics (accuracy, false positives, ..., roc)
* cross-validation
* better learning
* different loss functions

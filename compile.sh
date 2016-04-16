#!/bin/bash

# Compile and link to a BLAS library (see platform)
# use ldd on linux and otool -L on macos to check that the correct library has been linked


filename=$1
platform=$2

filename=${filename:="backprop.cpp"}
prog="${filename%.*}"
platform=${platform:=3}

# TODO: CUDA
case "${platform}" in 
	1)
		echo "enabling clblas/gpu..."
		# g++ ${filename} -o ${prog} -O3 -DARMA_USE_LAPACK -DARMA_USE_BLAS -L/usr/local/Cellar/clblas/2.2/lib/ -lclblas -larmadillo
		g++ ${filename} -o ${prog} -O3 -DARMA_USE_BLAS -lclBLAS -larmadillo
		;;
	2)
		echo "enabling blas/cpu..."
		if [[ "$OSTYPE" == "darwin"* ]]; then
			g++ ${filename} -o ${prog} -O3 -DARMA_DONT_USE_WRAPPER -DARMA_USE_BLAS -framework accelerate 
		else
			g++ ${filename} -o ${prog} -O3 -DARMA_DONT_USE_WRAPPER -DARMA_USE_BLAS -lblas
		fi
		;;
	3)
		echo "enabling openblas/cpu..."
		# change the location of the openblas library below
		g++ ${filename} -o ${prog} -O3 -DARMA_DONT_USE_WRAPPER -DARMA_USE_BLAS -lopenblas 
		# -L/usr/local/Cellar/openblas/0.2.15/lib/ 
		;;
	4)
		echo "enabling cuda..."
		# change the location of the openblas library below
		nvcc ${filename} -o ${prog} -larmadillo -O3 -lnvblas -DARMA_USE_BLAS 
		;;

	*)
		echo "no matching option found!"
esac


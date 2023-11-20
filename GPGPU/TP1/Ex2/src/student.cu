/*
* TP 1 - Premiers pas en CUDA
* --------------------------
* Ex 2: Addition de vecteurs
*
* File: student.cu
* Author: Maxime MARIA
*/

#include "student.hpp"
#include "chronoGPU.hpp"

//#define NB_THREADS 4096			// question 4

namespace IMAC
{
	__global__ void sumArraysCUDA(const int n, const int *const dev_a, const int *const dev_b, int *const dev_res)
	{
		// question 4
		// int idx = threadIdx.x;
		// dev_res[idx] = dev_a[idx] + dev_b[idx];

		int idx = blockDim.x * blockIdx.x + threadIdx.x;		// i eme thread du block j
		int offset = gridDim.x * blockDim.x;					// nb de block * nb de thread dans un block
		
		while (idx < n) {
			dev_res[idx] = dev_a[idx] + dev_b[idx];
			idx += offset;
		}
	}

    void studentJob(const int size, const int *const a, const int *const b, int *const res)
	{
		ChronoGPU chrGPU;

		// 3 arrays for GPU
		int *dev_a = NULL;
		int *dev_b = NULL;
		int *dev_res = NULL;
		int nbThreads = 1;
		int nbBlocks = 1;

		// Allocate arrays on device (input and ouput)
		const size_t bytes = size * sizeof(int);
		std::cout 	<< "Allocating input (3 arrays): " 
					<< ( ( 3 * bytes ) >> 20 ) << " MB on Device" << std::endl;
		chrGPU.start();
		
		cudaMalloc((void **) &dev_a, bytes);
		cudaMalloc((void **) &dev_b, bytes);
		cudaMalloc((void **) &dev_res, bytes);

		chrGPU.stop();
		std::cout 	<< "-> Done : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;

		// Copy data from host to device (input arrays) 
		cudaMemcpy(dev_a, a, bytes, cudaMemcpyHostToDevice);
		cudaMemcpy(dev_b, b, bytes, cudaMemcpyHostToDevice);

		// Launch kernel

		// sumArraysCUDA<<<1, n>>>(size, dev_a, dev_b, dev_res); 							// question 3
		// sumArraysCUDA<<<(size/1024) + 1, 1024>>>(size, dev_a, dev_b, dev_res);			// question 4
		
		// question 5
		while(nbThreads < size && nbThreads < 1024) {
			nbThreads *= 2;
		}
		
		int littleOffset = size % nbThreads != 0 ? 1 : 0;
		nbBlocks = (size/nbThreads) > 1024 ? 1024 : (size/nbThreads) + littleOffset;

		sumArraysCUDA<<<nbBlocks, nbThreads>>>(size, dev_a, dev_b, dev_res);

		// Copy data from device to host (output array)  
		cudaMemcpy(res, dev_res, bytes, cudaMemcpyDeviceToHost);

		// Free arrays on device
		cudaFree(dev_a);
		cudaFree(dev_b);
		cudaFree(dev_res);
	}
}


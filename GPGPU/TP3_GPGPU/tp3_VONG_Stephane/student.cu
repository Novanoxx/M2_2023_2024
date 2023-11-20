/*
* TP 3 - Réduction CUDA
* --------------------------
* Mémoire paratagée, synchronisation, optimisation
*
* File: student.cu
* Author: Maxime MARIA
*/

#include "student.hpp"

namespace IMAC
{
	// ==================================================== EX 1
    __global__
    void maxReduce_ex1(const uint *const dev_array, const uint size, uint *const dev_partialMax)
	{
		int idX = blockDim.x * blockIdx.x + threadIdx.x;		// i eme thread du block j
		int id = threadIdx.x;
		int stride = 1;
		extern __shared__ uint sharedMemory[];

		if (idX < size) 
		{
			sharedMemory[id] = dev_array[idX]; 	
			__syncthreads();

			for(unsigned int step = 1; step < blockDim.x/2 + 1; step++)
			{
				if ((id + stride) < blockDim.x)
				{
					sharedMemory[id] = umax(sharedMemory[id], sharedMemory[id+stride]);
				}
				__syncthreads();
				stride *= 2;
			}
			if (id == 0)
			{
				dev_partialMax[blockIdx.x] = sharedMemory[0];
			}
		}
	}

	__global__
	void maxReduce_ex2(const uint *const dev_array, const uint size, uint *const dev_partialMax)
	{
		int idX = blockDim.x * blockIdx.x + threadIdx.x;		// i eme thread du block j
		int id = threadIdx.x;
		int stride = blockDim.x/2 + 1;
		extern __shared__ uint sharedMemory[];

		if (idX < size) 
		{
			sharedMemory[id] = dev_array[idX]; 	
			__syncthreads();

			for(unsigned int step = 1; step < blockDim.x/2 + 1; step++)
			{
				if ((id + stride) < blockDim.x)
					{
						sharedMemory[id] = umax(sharedMemory[id], sharedMemory[id+stride]);
					}
					__syncthreads();
					stride /= 2;
			}
			if (id == 0)
			{
				dev_partialMax[blockIdx.x] = sharedMemory[0];
			}
		}
	}

	__global__
	void maxReduce_ex3(const uint *const dev_array, const uint size, uint *const dev_partialMax)
	{
		int idX = (blockDim.x * 2) * blockIdx.x + threadIdx.x;		// i eme thread du block j
		int id = threadIdx.x;
		int stride = blockDim.x/2 + 1;
		extern __shared__ uint sharedMemory[];

		if (idX < size) 
		{
			sharedMemory[id] = umax(dev_array[idX], dev_array[idX + blockDim.x]);
			__syncthreads();

			for(unsigned int step = 1; step < blockDim.x/2 + 1; step++)
			{
				if ((id + stride) < blockDim.x)
				{
					sharedMemory[id] = umax(sharedMemory[id], sharedMemory[id+stride]);
				}
				__syncthreads();
				stride /= 2;
			}
			if (id == 0)
			{
				dev_partialMax[blockIdx.x] = sharedMemory[0];
			}
		}
	}

	__device__ void warpReduceEx4(volatile uint* sharedMem, int id) {
		for (int i = 32; i > 0; i /= 2)
		{
			sharedMem[id] = umax(sharedMem[id], sharedMem[id + i]);
		}
	}
	
	__global__
	void maxReduce_ex4(const uint *const dev_array, const uint size, uint *const dev_partialMax)
	{
		int idX = (blockDim.x * 2) * blockIdx.x + threadIdx.x;		// i eme thread du block j
		int id = threadIdx.x;
		int stride = blockDim.x/2 + 1;
		extern __shared__ uint sharedMemory[];

		if (idX < size) 
		{
			sharedMemory[id] = umax(dev_array[idX], dev_array[idX + blockDim.x]);
			__syncthreads();

			for(unsigned int step = 1; step < blockDim.x/2 + 1; step++)
			{
				if (step > 32)
				{
					if ((id + stride) < blockDim.x)
				{
					sharedMemory[id] = umax(sharedMemory[id], sharedMemory[id+stride]);
				}
				__syncthreads();
				stride /= 2;
				}
			}
			if (id < 32)
			{
				warpReduceEx4(sharedMemory, id);
			}
			if (id == 0)
			{
				dev_partialMax[blockIdx.x] = sharedMemory[0];
			}
		}
	}

	template<uint blockSize>
	__device__ void warpReduceEx5(volatile uint* sharedMem, int id) {
		for (int i = 32; i > 0; i /= 2)
		{
			if (blockSize >= i*2)
			{
				sharedMem[id] = umax(sharedMem[id], sharedMem[id + i]);
			}
		}
	}
	
	template<uint blockSize>
	__global__
	void maxReduce_ex5(const uint *const dev_array, const uint size, uint *const dev_partialMax)
	{
		int idX = (blockDim.x * 2) * blockIdx.x + threadIdx.x;		// i eme thread du block j
		int id = threadIdx.x;
		int stride = blockDim.x/2 + 1;
		extern __shared__ uint sharedMemory[];

		if (idX < size) 
		{
			sharedMemory[id] = umax(dev_array[idX], dev_array[idX + blockDim.x]);
			__syncthreads();

			for(unsigned int step = 1; step < blockDim.x/2 + 1; step++)
			{
				if (step > 32)
				{
					if ((id + stride) < blockDim.x)
				{
					sharedMemory[id] = umax(sharedMemory[id], sharedMemory[id+stride]);
				}
				__syncthreads();
				stride /= 2;
				}
			}
			for (int i = 1; i < 4; i++)
			{
				if (blockSize >= 512)
				{
					if (id < 512 / i)
					{
						sharedMemory[id] = umax(sharedMemory[id], sharedMemory[id+(512/i)]);
					}
				}
			}

			if (id < 32)
			{
				warpReduceEx4(sharedMemory, id);
			}
			
			if (id == 0)
			{
				dev_partialMax[blockIdx.x] = sharedMemory[0];
			}
		}
	}

	void studentJob(const std::vector<uint> &array, const uint resCPU /* Just for comparison */, const uint nbIterations)
    {
		uint *dev_array = NULL;
        const size_t bytes = array.size() * sizeof(uint);

		// Allocate array on GPU
		HANDLE_ERROR( cudaMalloc( (void**)&dev_array, bytes ) );
		// Copy data from host to device
		HANDLE_ERROR( cudaMemcpy( dev_array, array.data(), bytes, cudaMemcpyHostToDevice ) );

		std::cout << "Test with " << nbIterations << " iterations" << std::endl;

		std::cout << "========== Ex 1 " << std::endl;
		uint res1 = 0; // result
		// Launch reduction and get timing
		float2 timing1 = reduce<KERNEL_EX1>(nbIterations, dev_array, array.size(), res1);
		
        std::cout << " -> Done: ";
        printTiming(timing1);
		compare(res1, resCPU); // Compare results

		std::cout << "========== Ex 2 " << std::endl;
		uint res2 = 0; // result
		// Launch reduction and get timing
		float2 timing2 = reduce<KERNEL_EX2>(nbIterations, dev_array, array.size(), res2);
		
        std::cout << " -> Done: ";
        printTiming(timing2);
		compare(res2, resCPU);

		std::cout << "========== Ex 3 " << std::endl;
		uint res3 = 0; // result
		// Launch reduction and get timing
		float2 timing3 = reduce<KERNEL_EX3>(nbIterations, dev_array, array.size(), res3);
		
        std::cout << " -> Done: ";
        printTiming(timing3);
		compare(res3, resCPU);

		std::cout << "========== Ex 4 " << std::endl;
		uint res4 = 0; // result
		// Launch reduction and get timing
		float2 timing4 = reduce<KERNEL_EX4>(nbIterations, dev_array, array.size(), res4);
		
        std::cout << " -> Done: ";
        printTiming(timing4);
		compare(res4, resCPU);

		std::cout << "========== Ex 5 " << std::endl;
		uint res5 = 0; // result
		// Launch reduction and get timing
		float2 timing5 = reduce<KERNEL_EX5>(nbIterations, dev_array, array.size(), res5);
		
        std::cout << " -> Done: ";
        printTiming(timing5);
		compare(res5, resCPU);

		// Free array on GPU
		cudaFree( dev_array );
    }

	void printTiming(const float2 timing)
	{
		std::cout << ( timing.x < 1.f ? 1e3f * timing.x : timing.x ) << " us on device and ";
		std::cout << ( timing.y < 1.f ? 1e3f * timing.y : timing.y ) << " us on host." << std::endl;
	}

    void compare(const uint resGPU, const uint resCPU)
	{
		if (resGPU == resCPU)
		{
			std::cout << "Well done ! " << resGPU << " == " << resCPU << " !!!" << std::endl;
		}
		else
		{
			std::cout << "You failed ! " << resGPU << " != " << resCPU << " !!!" << std::endl;
		}
	}
}

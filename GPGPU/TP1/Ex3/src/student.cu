/*
* TP 1 - Premiers pas en CUDA
* --------------------------
* Ex 3: Filtre d'images sepia
*
* File: student.cu
* Author: Maxime MARIA
*/

#include "student.hpp"
#include "chronoGPU.hpp"

#include "math.h"

namespace IMAC
{
	// Exercice 3.4 fait en binome avec DIAGNE Ben
	__global__ void sepiaCUDA(const uint width, const uint height, const uchar *const dev_input, uchar *const dev_output)
	{
		int idX = blockDim.x * blockIdx.x + threadIdx.x;		// i eme thread du block j
		int idY = blockDim.y * blockIdx.y + threadIdx.y;		
		int id = 0;

		int offsetX = blockDim.x * gridDim.x;
		int offsetY = blockDim.y * gridDim.y;
		
		uchar r = 0;
		uchar g = 0;
		uchar b = 0;


		while (idY < height) {

			while (idX < width) {
				id = idY * width + idX;
				id *= 3;

				r = dev_input[id];
				g = dev_input[id + 1];
				b = dev_input[id + 2];
				dev_output[id] = (uchar) fminf( 255.f, ( r * .393f + g * .769f + b * .189f ) ) ;
				dev_output[id + 1] = (uchar) fminf( 255.f, ( r * .349f + g * .686f + b * .168f ) ) ;
				dev_output[id + 2] = (uchar) (fminf( 255.f, ( r * .272f + g * .534f + b * .131f ) ) );
				idX += offsetX;
			}
			idX = blockDim.x * blockIdx.x + threadIdx.x;		// Arriver au bout de l'image, on revient a la ligne
			idY += offsetY;										// Deplace l'id au prochain block
		}

/*
		while (idX < width) {
			idY = blockDim.y * blockIdx.y + threadIdx.y;

			while (idY < height) {
				id = idX * height + idY;
				id *= 3;

				r = dev_input[id];
				g = dev_input[id + 1];
				b = dev_input[id + 2];
				dev_output[id] = (uchar) fminf( 255.f, ( r * .393f + g * .769f + b * .189f ) ) ;
				dev_output[id + 1] = (uchar) fminf( 255.f, ( r * .349f + g * .686f + b * .168f ) ) ;
				dev_output[id + 2] = (uchar) (fminf( 255.f, ( r * .272f + g * .534f + b * .131f ) ) );
				idY += offsetY;
			}
			
			idX += offsetX;
		}
*/
	}

	void studentJob(const std::vector<uchar> &input, const uint width, const uint height, std::vector<uchar> &output)
	{
		ChronoGPU chrGPU;

		// 2 arrays for GPU
		uchar *dev_input = NULL;
		uchar *dev_output = NULL;
		int nbThreadsX = 1;
		int nbThreadsY = 1;
		int nbBlocksX = 1;
		int nbBlocksY = 1;

		const size_t bytes = width * height * 3 * sizeof(uchar);
		std::cout 	<< "Allocating input: " 
					<< ( ( 2 * bytes ) >> 20 ) << " MB on Device" << std::endl;
		chrGPU.start();
		
		cudaMalloc((void **) &dev_input, bytes);
		cudaMalloc((void **) &dev_output, bytes);

		chrGPU.stop();
		std::cout 	<< "-> Done : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;

		cudaMemcpy(dev_input, input.data(), bytes, cudaMemcpyHostToDevice);
		cudaMemcpy(dev_output, output.data(), bytes, cudaMemcpyHostToDevice);

		while(nbThreadsX * nbThreadsY < 1024) {
			if (nbThreadsX < width) {
				nbThreadsX *= 2;
			}
			if (nbThreadsY < height) {
				nbThreadsY *= 2;
			}
		}

		while(nbBlocksX * nbBlocksY < 64) {
			if (nbBlocksX < width/nbThreadsX) {
				nbBlocksX++;
			}
			if (nbBlocksY < height/nbThreadsY) {
				nbBlocksY++;
			}
		}
		//std::cout 	<< "-> ThreadsX : " << nbThreadsX << " -> ThreadsY : " << nbThreadsY << std::endl;
		//std::cout 	<< "-> BlockX : " << nbBlocksX << " -> BlockY : " << nbBlocksY << std::endl;
		sepiaCUDA<<<dim3(nbBlocksX, nbBlocksY), dim3(nbThreadsX, nbThreadsY)>>>(width, height, dev_input, dev_output);

		cudaMemcpy(output.data(), dev_output, bytes, cudaMemcpyDeviceToHost);

		cudaFree(dev_input);
		cudaFree(dev_output);
	}
}

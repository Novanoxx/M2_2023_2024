/*
* TP 2 - Convolution d'images
* --------------------------
* Mémoire constante et textures
*
* File: student.cu
* Author: Maxime MARIA
*/

#include "student.hpp"
#include "chronoGPU.hpp"

#include "math.h"

__constant__ float constDevMatConv[2048];	// Exercice 2
texture<uchar4, 1, cudaReadModeElementType> texture1D;
texture<uchar4, 2, cudaReadModeElementType> texture2D;

namespace IMAC
{

// ================================================== For image comparison
	std::ostream &operator <<(std::ostream &os, const uchar4 &c)
	{
		os << "[" << uint(c.x) << "," << uint(c.y) << "," << uint(c.z) << "," << uint(c.w) << "]";  
    	return os; 
	}

	void compareImages(const std::vector<uchar4> &a, const std::vector<uchar4> &b)
	{
		bool error = false;
		if (a.size() != b.size())
		{
			std::cout << "Size is different !" << std::endl;
			error = true;
		}
		else
		{
			for (uint i = 0; i < a.size(); ++i)
			{
				// Floating precision can cause small difference between host and device
				if (	std::abs(a[i].x - b[i].x) > 2 || std::abs(a[i].y - b[i].y) > 2 
					|| std::abs(a[i].z - b[i].z) > 2 || std::abs(a[i].w - b[i].w) > 2)
				{
					std::cout << "Error at index " << i << ": a = " << a[i] << " - b = " << b[i] << " - " << std::abs(a[i].x - b[i].x) << std::endl;
					error = true;
					break;
				}
			}
		}
		if (error)
		{
			std::cout << " -> You failed, retry!" << std::endl;
		}
		else
		{
			std::cout << " -> Well done!" << std::endl;
		}
	}
// ==================================================

	__global__ void convNaive(	const uchar4 *const dev_input, 
					const uint imgWidth, const uint imgHeight, 
					const float *const matConv, 
					const uint matSize, 
					uchar4 *const dev_output)
	{
		int idX = blockDim.x * blockIdx.x + threadIdx.x;		// i eme thread du block j
		int idY = blockDim.y * blockIdx.y + threadIdx.y;		

		int offsetX = blockDim.x * gridDim.x;
		int offsetY = blockDim.y * gridDim.y;

		while (idY < imgHeight) {

			while (idX < imgWidth) {
				float3 sum = make_float3(0.f,0.f,0.f);
				
				// Apply convolution
				for ( uint j = 0; j < matSize; ++j ) 
				{
					for ( uint i = 0; i < matSize; ++i ) 
					{
						int dX = idX + i - matSize / 2;
						int dY = idY + j - matSize / 2;

						// Handle borders
						if ( dX < 0 ) 
							dX = 0;

						if ( dX >= imgWidth ) 
							dX = imgWidth - 1;

						if ( dY < 0 ) 
							dY = 0;

						if ( dY >= imgHeight ) 
							dY = imgHeight - 1;

						const int idMat		= j * matSize + i;
						const int idPixel	= dY * imgWidth + dX;
						sum.x += (float)dev_input[idPixel].x * matConv[idMat];
						sum.y += (float)dev_input[idPixel].y * matConv[idMat];
						sum.z += (float)dev_input[idPixel].z * matConv[idMat];
					}
				}
				const int idOut = idY * imgWidth + idX;
				dev_output[idOut].x = (uchar)fminf(255.f, fmaxf(0.f, sum.x));
				dev_output[idOut].y = (uchar)fminf(255.f, fmaxf(0.f, sum.y));
				dev_output[idOut].z = (uchar)fminf(255.f, fmaxf(0.f, sum.z));
				dev_output[idOut].w = 255;
				idX += offsetX;
			}
			idX = blockDim.x * blockIdx.x + threadIdx.x;		// Arriver au bout de l'image, on revient a la ligne
			idY += offsetY;										// Deplace l'id au prochain block
		}
	}

    __global__ void convMemoireConstante(const uchar4 *const dev_input, 
					const uint imgWidth, const uint imgHeight,
					const uint matSize, 
					uchar4 *const dev_output) 
	{
		int idX = blockDim.x * blockIdx.x + threadIdx.x;		// i eme thread du block j
		int idY = blockDim.y * blockIdx.y + threadIdx.y;		

		int offsetX = blockDim.x * gridDim.x;
		int offsetY = blockDim.y * gridDim.y;

		while (idY < imgHeight) {

			while (idX < imgWidth) {
				float3 sum = make_float3(0.f,0.f,0.f);
				
				// Apply convolution
				for ( uint j = 0; j < matSize; ++j ) 
				{
					for ( uint i = 0; i < matSize; ++i ) 
					{
						int dX = idX + i - matSize / 2;
						int dY = idY + j - matSize / 2;

						// Handle borders
						if ( dX < 0 ) 
							dX = 0;

						if ( dX >= imgWidth ) 
							dX = imgWidth - 1;

						if ( dY < 0 ) 
							dY = 0;

						if ( dY >= imgHeight ) 
							dY = imgHeight - 1;

						const int idMat		= j * matSize + i;
						const int idPixel	= dY * imgWidth + dX;
						sum.x += (float)dev_input[idPixel].x * constDevMatConv[idMat];
						sum.y += (float)dev_input[idPixel].y * constDevMatConv[idMat];
						sum.z += (float)dev_input[idPixel].z * constDevMatConv[idMat];
					}
				}
				const int idOut = idY * imgWidth + idX;
				dev_output[idOut].x = (uchar)fminf(255.f, fmaxf(0.f, sum.x));
				dev_output[idOut].y = (uchar)fminf(255.f, fmaxf(0.f, sum.y));
				dev_output[idOut].z = (uchar)fminf(255.f, fmaxf(0.f, sum.z));
				dev_output[idOut].w = 255;
				idX += offsetX;
			}
			idX = blockDim.x * blockIdx.x + threadIdx.x;		// Arriver au bout de l'image, on revient a la ligne
			idY += offsetY;										// Deplace l'id au prochain block
		}
	}
	
	__global__ void convText1D(
					const uint imgWidth, const uint imgHeight,
					const uint matSize, 
					uchar4 *const dev_output) 
	{
		int idX = blockDim.x * blockIdx.x + threadIdx.x;		// i eme thread du block j
		int idY = blockDim.y * blockIdx.y + threadIdx.y;		

		int offsetX = blockDim.x * gridDim.x;
		int offsetY = blockDim.y * gridDim.y;

		while (idY < imgHeight) {

			while (idX < imgWidth) {
				float3 sum = make_float3(0.f,0.f,0.f);
				
				// Apply convolution
				for ( uint j = 0; j < matSize; ++j ) 
				{
					for ( uint i = 0; i < matSize; ++i ) 
					{
						int dX = idX + i - matSize / 2;
						int dY = idY + j - matSize / 2;

						// Handle borders
						if ( dX < 0 ) 
							dX = 0;

						if ( dX >= imgWidth ) 
							dX = imgWidth - 1;

						if ( dY < 0 ) 
							dY = 0;

						if ( dY >= imgHeight ) 
							dY = imgHeight - 1;

						const int idMat		= j * matSize + i;
						const int idPixel	= dY * imgWidth + dX;
						sum.x += (float)tex1Dfetch(texture1D, idPixel).x * constDevMatConv[idMat];
						sum.y += (float)tex1Dfetch(texture1D, idPixel).y * constDevMatConv[idMat];
						sum.z += (float)tex1Dfetch(texture1D, idPixel).z * constDevMatConv[idMat];
					}
				}
				const int idOut = idY * imgWidth + idX;
				dev_output[idOut].x = (uchar)fminf(255.f, fmaxf(0.f, sum.x));
				dev_output[idOut].y = (uchar)fminf(255.f, fmaxf(0.f, sum.y));
				dev_output[idOut].z = (uchar)fminf(255.f, fmaxf(0.f, sum.z));
				dev_output[idOut].w = 255;
				idX += offsetX;
			}
			idX = blockDim.x * blockIdx.x + threadIdx.x;		// Arriver au bout de l'image, on revient a la ligne
			idY += offsetY;										// Deplace l'id au prochain block
		}
	}

	__global__ void convText2D(
					const uint imgWidth, const uint imgHeight,
					const uint matSize, 
					uchar4 *const dev_output) 
	{
		int idX = blockDim.x * blockIdx.x + threadIdx.x;		// i eme thread du block j
		int idY = blockDim.y * blockIdx.y + threadIdx.y;		

		int offsetX = blockDim.x * gridDim.x;
		int offsetY = blockDim.y * gridDim.y;

		while (idY < imgHeight) {

			while (idX < imgWidth) {
				float3 sum = make_float3(0.f,0.f,0.f);
				
				// Apply convolution
				for ( uint j = 0; j < matSize; ++j ) 
				{
					for ( uint i = 0; i < matSize; ++i ) 
					{
						int dX = idX + i - matSize / 2;
						int dY = idY + j - matSize / 2;

						// Handle borders
						if ( dX < 0 ) 
							dX = 0;

						if ( dX >= imgWidth ) 
							dX = imgWidth - 1;

						if ( dY < 0 ) 
							dY = 0;

						if ( dY >= imgHeight ) 
							dY = imgHeight - 1;

						const int idMat		= j * matSize + i;
						const int idPixel	= dY * imgWidth + dX;
						sum.x += (float)tex2D(texture2D, dX, dY).x * constDevMatConv[idMat];
						sum.y += (float)tex2D(texture2D, dX, dY).y * constDevMatConv[idMat];
						sum.z += (float)tex2D(texture2D, dX, dY).z * constDevMatConv[idMat];
					}
				}
				const int idOut = idY * imgWidth + idX;
				dev_output[idOut].x = (uchar)fminf(255.f, fmaxf(0.f, sum.x));
				dev_output[idOut].y = (uchar)fminf(255.f, fmaxf(0.f, sum.y));
				dev_output[idOut].z = (uchar)fminf(255.f, fmaxf(0.f, sum.z));
				dev_output[idOut].w = 255;
				idX += offsetX;
			}
			idX = blockDim.x * blockIdx.x + threadIdx.x;		// Arriver au bout de l'image, on revient a la ligne
			idY += offsetY;										// Deplace l'id au prochain block
		}
	}

	void studentJob(const std::vector<uchar4> &inputImg, // Input image
					const uint imgWidth, const uint imgHeight, // Image size
                    const std::vector<float> &matConv, // Convolution matrix (square)
					const uint matSize, // Matrix size (width or height)
					const std::vector<uchar4> &resultCPU, // Just for comparison
                    std::vector<uchar4> &output // Output image
					)
	{
		uchar4 *dev_input = NULL;
		uchar4 *dev_output = NULL;
		// float *dev_matConv = NULL;		// Exercice 1
		int nbThreadsX = 1;
		int nbThreadsY = 1;
		size_t pitch;

		std::cout << "Process on GPU (sequential)"	<< std::endl;
		ChronoGPU chrGPU;
		chrGPU.start();

		const size_t bytes = imgWidth * imgHeight * sizeof(uchar4);
		std::cout 	<< "Allocating input: " 
					<< ( ( 2 * bytes ) >> 20 ) << " MB on Device" << std::endl;
		
		// cudaMalloc((void **) &dev_input, bytes);		// Exercice 1, 2 et 3
		cudaMallocPitch((void **) &dev_input, &pitch, imgWidth * sizeof(*dev_input), imgHeight);
		cudaMalloc((void **) &dev_output, bytes);
		// cudaMalloc((void **) &dev_matConv, bytes);	// Exercice 1

		chrGPU.stop();
		std::cout 	<< " -> Done : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;

		// cudaMemcpy(dev_input, inputImg.data(), bytes, cudaMemcpyHostToDevice);	// Exercice 1; 2 et 3
		cudaMemcpy2D(dev_input, pitch, inputImg.data(), imgWidth * sizeof(uchar4), imgWidth * sizeof(uchar4), imgHeight, cudaMemcpyHostToDevice);
		cudaMemcpy(dev_output, output.data(), bytes, cudaMemcpyHostToDevice);
		// Exercice 1
		// cudaMemcpy(dev_matConv, matConv.data(), bytes, cudaMemcpyHostToDevice);

		cudaMemcpyToSymbol(constDevMatConv, matConv.data(), matSize * matSize * sizeof(float));
		
		// Exercice 3
		// cudaBindTexture(0, texture1D, dev_input, bytes);

		// Exercice 4
		texture2D.normalized = false;
		cudaBindTexture2D(0, texture2D, dev_input, imgWidth, imgHeight, pitch);

		while(nbThreadsX * nbThreadsY < 1024 && (nbThreadsX < imgWidth || nbThreadsY < imgHeight)) {
			if (nbThreadsX < imgWidth) {
				nbThreadsX *= 2;
			}
			if (nbThreadsY < imgHeight) {
				nbThreadsY *= 2;
			}
		}

		chrGPU.start();
		// Exercice 1
		/*
		convNaive<<<imgWidth * imgHeight, dim3(nbThreadsX, nbThreadsY)>>>
		(	dev_input,
			imgWidth, 
			imgHeight, 
			dev_matConv, 
			matSize, 
			dev_output);
		*/
		
		// Exercice 2
		/*
		convMemoireConstante<<<imgWidth * imgHeight, dim3(nbThreadsX, nbThreadsY)>>>
		(	dev_input,
			imgWidth, 
			imgHeight,  
			matSize, 
			dev_output);
		*/

		// Exercice 3
		/*
		convText1D<<<imgWidth * imgHeight, dim3(nbThreadsX, nbThreadsY)>>>
		(	imgWidth, 
			imgHeight,  
			matSize, 
			dev_output);
		*/

		// Exercice 4
		/*
		convText2D<<<imgWidth * imgHeight, dim3(nbThreadsX, nbThreadsY)>>>
		(	imgWidth, 
			imgHeight,  
			matSize, 
			dev_output);
		*/	

		chrGPU.stop();
		std::cout 	<< " -> Done : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;

		cudaMemcpy(output.data(), dev_output, bytes, cudaMemcpyDeviceToHost);

		cudaFree(dev_input);
		cudaFree(dev_output);
		//cudaFree(dev_matConv);
	}
}

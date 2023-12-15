#include "student.hpp"
#include "chronoGPU.hpp"

#include "math.h"

namespace IMAC
{
	__device__
	void RGBtoXYZ(int R, int G, int B, double &X, double &Y, double &Z) {
		// Normalize RGB values
		double rgb[3] = {R / 255.0, G / 255.0, B / 255.0};
		
		// Apply gamma correction
		for (int i = 0; i < 3; i++)
		{
			if (rgb[i] > 0.04045)
			{
				rgb[i] = pow((rgb[i] + 0.055) / 1.055, 2.4);
			} else
			{
				rgb[i] /= 12.92;
			}
			// Convert sRGB to XYZ
			rgb[i] *= 100.0;
		}

		X = rgb[0] * 0.4124564 + rgb[1] * 0.3575761 + rgb[2] * 0.1804375;
		Y = rgb[0] * 0.2126729 + rgb[1] * 0.7151522 + rgb[2] * 0.0721750;
		Z = rgb[0] * 0.0193339 + rgb[1] * 0.1191920 + rgb[2] * 0.9503041;
	}

	__device__
	void XYZtoLab(double X, double Y, double Z, double &L, double &a, double &b) {
		// Normalize XYZ values
		double xyz[3] = {X/95.047, Y/100.000, Z/108.883};

		// Apply non-linear transformation
		for (int i = 0; i < 3; i++)
		{
			if (xyz[i] > 0.008856)
			{
				xyz[i] = pow(xyz[i], 1.0 / 3.0);
			} else
			{
				xyz[i] = (903.3 * xyz[i]+ 16.0) / 116.0;
			}
		}

		// Convert XYZ to Lab
		L = fmax(0.0, 116.0 * xyz[1] - 16.0);
		a = (xyz[0] - xyz[1]) * 500.0;
		b = (xyz[1] - xyz[2]) * 200.0;
	}

	__device__
	void RGBtoLab(int R, int G, int B, double &L, double &a, double &b) {
		double X, Y, Z;
		RGBtoXYZ(R, G, B, X, Y, Z);
		XYZtoLab(X, Y, Z, L, a, b);
	}

	__device__ 
	double LabDist(double *Lab1, double *Lab2) {
		return sqrt(pow(Lab2[0] - Lab1[0], 2) + pow(Lab2[1] - Lab1[1], 2) + pow(Lab2[2] - Lab1[2], 2));
	}

	__global__
	void CUDA(const uint width, const uint height, const uchar *const dev_input, uchar *const dev_output)
	{
		int idX = blockDim.x * blockIdx.x + threadIdx.x;		// i eme thread du block j
		int idY = blockDim.y * blockIdx.y + threadIdx.y;		
		int id = 0;

		int offsetX = blockDim.x * gridDim.x;
		int offsetY = blockDim.y * gridDim.y;

		double labPixel[3];
		double labPixelRight[3];
		double labPixelBottom[3];

		while (idY < height) {

			while (idX < width) {
				id = idY * width + idX;
				id *= 3;
				int idRight = idY * width + (idX + 1);
				int idBottom = (idY + 1) * width + idX;

				RGBtoLab(dev_input[id], dev_input[id + 1], dev_input[id + 2], labPixel[0], labPixel[1], labPixel[2]);
				RGBtoLab(dev_input[idRight], dev_input[idRight + 1], dev_input[idRight + 2], labPixelRight[0], labPixelRight[1], labPixelRight[2]);
				RGBtoLab(dev_input[idBottom], dev_input[idBottom + 1], dev_input[idBottom + 2], labPixelBottom[0], labPixelBottom[1], labPixelBottom[2]);

				auto rightDist = LabDist(labPixel, labPixelRight);
				auto bottomDist = LabDist(labPixel, labPixelBottom);

				if (rightDist > 120)
				{
					dev_output[id] = 255;
				} else
				{
					dev_output[id] = 0;
				}

				if (bottomDist > 120)
				{
					dev_output[id + 1] = 255;
				} else
				{
					dev_output[id + 1] = 0;
				}
				// Etape 1
				//dev_output[id] = L;
				//dev_output[id + 1] = a;
				//dev_output[id + 2] = b;
				dev_output[id + 2] = 0;
				idX += offsetX;
			}
			idX = blockDim.x * blockIdx.x + threadIdx.x;		// Arriver au bout de l'image, on revient a la ligne
			idY += offsetY;										// Deplace l'id au prochain block
		}
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
		
		CUDA<<<dim3(nbBlocksX, nbBlocksY), dim3(nbThreadsX, nbThreadsY)>>>(width, height, dev_input, dev_output);

		cudaMemcpy(output.data(), dev_output, bytes, cudaMemcpyDeviceToHost);

		cudaFree(dev_input);
		cudaFree(dev_output);
	}
}

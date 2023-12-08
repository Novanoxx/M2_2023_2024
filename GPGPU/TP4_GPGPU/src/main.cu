/*
* TP 1 - Premiers pas en CUDA
* --------------------------
* Ex 3: Filtre d'images sepia
*
* File: main.cpp
* Author: Maxime MARIA
*/

#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <iomanip>     
#include <cstring>
#include <exception>
#include <algorithm>

#include "student.hpp"
#include "chronoCPU.hpp"
#include "lodepng.h"

namespace IMAC
{
	// Print program usage
	void printUsageAndExit(const char *prg) 
	{
		std::cerr	<< "Usage: " << prg << std::endl
					<< " \t -f <F>: <F> image file name" << std::endl << std::endl;
		exit(EXIT_FAILURE);
	}

	void nothingCPU(const std::vector<uchar> &input, const uint width, const uint height, std::vector<uchar> &output)
	{
		std::cout << "Process on CPU (sequential)"	<< std::endl;
		ChronoCPU chrCPU;
		chrCPU.start();
		for (uint i = 0; i < height; ++i) 
		{
			for (uint j = 0; j < width; ++j) 
			{
				const uint id = (i * width + j) * 3;
				output[id] = input[id];
				output[id + 1] = input[id + 1];
				output[id + 2] = input[id + 2];
			}
		}
		chrCPU.stop();
		std::cout 	<< " -> Done : " << chrCPU.elapsedTime() << " ms" << std::endl << std::endl;
	}

	// Compare two vectors
	bool compare(const std::vector<uchar> &a, const std::vector<uchar> &b)
	{
		if (a.size() != b.size())
		{
			std::cout << "Size is different !" << std::endl;
			return false;
		}
		for (uint i = 0; i < a.size(); ++i)
		{
			// Floating precision can cause small difference between host and device
			if (std::abs(a[i] - b[i]) > 1)
			{
				std::cout << "Error at index " << i << ": a = " << uint(a[i]) << " - b = " << uint(b[i]) << std::endl;
				return false; 
			}
		}
		return true;
	}

	// Main function
	void main(int argc, char **argv) 
	{	
		char fileName[2048];

		// Parse command line
		if (argc == 1) 
		{
			std::cerr << "Please give a file..." << std::endl;
			printUsageAndExit(argv[0]);
		}

		for (int i = 1; i < argc; ++i) 
		{
			if (!strcmp(argv[i], "-f")) 
			{
				if (sscanf(argv[++i], "%s", fileName) != 1)
				{
					printUsageAndExit(argv[0]);
				}
			}
			else
			{
				printUsageAndExit(argv[0]);
			}
		}
		
		// Get input image
		std::vector<uchar> input;
		uint width;
		uint height;

		std::cout << "Loading " << fileName << std::endl;
		unsigned error = lodepng::decode(input, width, height, fileName, LCT_RGB);
		if (error)
		{
			throw std::runtime_error("Error loadpng::decode: " + std::string(lodepng_error_text(error)));
		}
		std::cout << "Image has " << width << " x " << height << " pixels (RGBA)" << std::endl;

		// Create 2 output images
		std::vector<uchar> outputCPU(3 * width * height);
		std::vector<uchar> outputGPU(3 * width * height);

		// Prepare output file name
		const std::string fileNameStr(fileName);
		std::size_t lastPoint = fileNameStr.find_last_of(".");
		std::string ext = fileNameStr.substr(lastPoint);
		std::string name = fileNameStr.substr(0,lastPoint);
		std::string outputGPUName = name + "_GPU" + ext;

		// Computation on CPU
		nothingCPU(input, width, height, outputCPU);
		
		std::cout 	<< "============================================"	<< std::endl
					<< "              STUDENT'S JOB !               "	<< std::endl
					<< "============================================"	<< std::endl;

		studentJob(input, width, height, outputGPU);

		std::cout << "Save image as: " << outputGPUName << std::endl;
		// error = lodepng::encode(outputGPUName, outputGPU, width, height, LCT_RGB);
		error = lodepng::encode(outputGPUName, outputGPU, width, height, LCT_RGB);
		if (error)
		{
			throw std::runtime_error("Error loadpng::decode: " + std::string(lodepng_error_text(error)));
		}
		
		std::cout << "============================================"	<< std::endl << std::endl;
	}
}

int main(int argc, char **argv) 
{
	try
	{
		IMAC::main(argc, argv);
	}
	catch (const std::exception &e)
	{
		std::cerr << e.what() << std::endl;
	}
}

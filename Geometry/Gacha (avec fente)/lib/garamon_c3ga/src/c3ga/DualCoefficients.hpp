// Copyright (c) 2018 by University Paris-Est Marne-la-Vallee
// Inner.hpp
// This file is part of the Garamon for c3ga.
// Authors: Stephane Breuils and Vincent Nozick
// Contact: vincent.nozick@u-pem.fr
//
// Licence MIT
// A a copy of the MIT License is given along with this program

// \file DualCoefficients.hpp
// \brief this files generates and load the elements of the fast dual into array of Eigen matrices 
// \author V. Nozick, S. Breuils

#ifndef C3GA_DUALCOEFFICIENTS_HPP__
#define C3GA_DUALCOEFFICIENTS_HPP__
#pragma once


#include <Eigen/Sparse>
#include <vector>
#include <array>
#include <algorithm>
#include <iterator>
#include <sstream>


/*!
 * @namespace c3ga
 */
namespace c3ga{

	/// Decode the string encodedPerGradeMatrixComponents to a vector of T. This vector will contain the coefficients used to compute the dual
	template<typename T>
	const std::vector<T> decodeStringOfDualCoefToVecOfT(std::string encodedPerGradeMatrixComponents) {
	    std::vector<T> resultDecodedGrade;
	    std::istringstream streamStringOverFloat(encodedPerGradeMatrixComponents);
	    std::copy(std::istream_iterator<float>(streamStringOverFloat),
		std::istream_iterator<float>(),
		std::back_inserter(resultDecodedGrade));
	    return resultDecodedGrade;
	}


	/// load the coefficients of the dual into array of Eigen matrices 
	template<typename T>
	const std::array<Eigen::Matrix<T, Eigen::Dynamic,1>,6> loadFastDualArray() {
		std::array<Eigen::Matrix<T, Eigen::Dynamic,1>,6> dualArrayCoefficients;
		std::string stringDualCoefficients="1.000000 -1.000000 -1.000000 1.000000 -1.000000 -1.000000 -1.000000 1.000000 -1.000000 -1.000000 1.000000 -1.000000 -1.000000 -1.000000 1.000000 -1.000000 1.000000 -1.000000 1.000000 1.000000 1.000000 -1.000000 1.000000 1.000000 -1.000000 1.000000 1.000000 1.000000 -1.000000 1.000000 1.000000 -1.000000 ";
		const std::vector<T> vectorDualComponents = decodeStringOfDualCoefToVecOfT<T>(stringDualCoefficients);
		dualArrayCoefficients[0]= Eigen::Matrix<T, Eigen::Dynamic,1>(1);
		for(unsigned int i=0;i<1;++i){
			dualArrayCoefficients[0].coeffRef(i-0) = (T)vectorDualComponents[i]; 
		}
		dualArrayCoefficients[1]= Eigen::Matrix<T, Eigen::Dynamic,1>(5);
		for(unsigned int i=1;i<6;++i){
			dualArrayCoefficients[1].coeffRef(i-1) = (T)vectorDualComponents[i]; 
		}
		dualArrayCoefficients[2]= Eigen::Matrix<T, Eigen::Dynamic,1>(10);
		for(unsigned int i=6;i<16;++i){
			dualArrayCoefficients[2].coeffRef(i-6) = (T)vectorDualComponents[i]; 
		}
		dualArrayCoefficients[3]= Eigen::Matrix<T, Eigen::Dynamic,1>(10);
		for(unsigned int i=16;i<26;++i){
			dualArrayCoefficients[3].coeffRef(i-16) = (T)vectorDualComponents[i]; 
		}
		dualArrayCoefficients[4]= Eigen::Matrix<T, Eigen::Dynamic,1>(5);
		for(unsigned int i=26;i<31;++i){
			dualArrayCoefficients[4].coeffRef(i-26) = (T)vectorDualComponents[i]; 
		}
		dualArrayCoefficients[5]= Eigen::Matrix<T, Eigen::Dynamic,1>(1);
		for(unsigned int i=31;i<32;++i){
			dualArrayCoefficients[5].coeffRef(0) = (T)vectorDualComponents[i]; 
		}
		return dualArrayCoefficients;
	}


}/// End of Namespace

#endif // C3GA_DUALCOEFFICIENTS_HPP__

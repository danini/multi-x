// Copyright(c) 2019 Daniel Barath
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// Please contact the author of this library if you have any questions.
// Author: Daniel Barath (majti89@gmail.com)
#pragma once

#include <cv.h>
#include <unordered_map>
#include "estimator.h"

#include <iostream>

namespace multix
{
	class Model
	{
	protected:
		std::unordered_map<std::string, float> parameters;

	public:
		std::vector<int> inliers;
		cv::Mat descriptor;
		cv::Mat descriptor_by_points;
		const Estimator<cv::Mat, Model> * estimator;

		void set_estimator(const Estimator<cv::Mat, Model> * _estimator)
		{
			estimator = _estimator;
		}

		virtual bool set_descriptor(cv::Mat _descriptor)
		{
			descriptor = _descriptor;
			return true;
		}

		cv::Mat get_descriptor()
		{
			return descriptor;
		}

		void set_parameter(const std::string _parameter_name, const float _value)
		{
			parameters[_parameter_name] = _value;
		}

		std::tuple<float, bool> get_parameter(std::string _parameter_name)
		{
			auto iterator = parameters.find(_parameter_name);
			if (iterator == parameters.end())
				return { 0, false };
			return { iterator->second, true };
		}

		Model()
		{

		}
	};

	class Line2D : public Model
	{
	public:
		int mss1, mss2;
		Line2D() {}
		Line2D(const Line2D& _other)
		{
			mss1 = _other.mss1;
			mss2 = _other.mss2;
			descriptor = _other.descriptor.clone();
			parameters = _other.parameters;
		}
	};

	class Motion : public Model
	{
	public:

		Motion() {}
		Motion(const Motion& _other)
		{
			parameters = _other.parameters;
			descriptor = _other.descriptor.clone();
		}
	};

	class Homography : public Model
	{
	public:
		Homography() {}
		Homography(const Homography& _other)
		{
			descriptor = _other.descriptor.clone();
			parameters = _other.parameters;
		}
	};

	class FundamentalMatrix : public Model
	{
	public:
		FundamentalMatrix() {}
		FundamentalMatrix(const FundamentalMatrix& _other)
		{
			descriptor = _other.descriptor.clone();
			parameters = _other.parameters;
		}
	};
}
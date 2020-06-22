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

#include <iostream>
#include <math.h>
#include <chrono>
#include <random>
#include <vector>
#include <cv.h>

#include "estimator.h"
#include "model.h"

// This is the estimator class for estimating a 2D line. A model estimation method and error calculation method are implemented
namespace multix
{
	class LineEstimator : public Estimator < cv::Mat, Model >
	{
	protected:
		cv::Size image_size;

	public:
		LineEstimator(cv::Size _image_size) : image_size(_image_size)
		{ }
		~LineEstimator() {}

		bool is_mode_seeking_applicable() const
		{
			return true;
		}

		int sample_size() const 
		{
			return 2;
		}

		int inlier_limit() const 
		{
			return 7 * sample_size();
		}

		float model_weight() const
		{
			return 1;
		}

		bool estimate_model_nonminimal(const cv::Mat * const _data,
			const int *_sample,
			int _sample_number,
			std::vector<Model>* _models) const
		{
			Model model;

			if (_sample_number < 2)
				return false;

			cv::Mat A(_sample_number, 3, CV_64F);
			int idx;
			cv::Mat mass_point = cv::Mat::zeros(1, 2, CV_32F);
			for (auto i = 0; i < _sample_number; ++i)
			{
				idx = _sample[i];
				mass_point.at<float>(0) += _data->at<float>(idx, 0);
				mass_point.at<float>(1) += _data->at<float>(idx, 1);

				A.at<double>(i, 0) = (double)_data->at<float>(idx, 0);
				A.at<double>(i, 1) = (double)_data->at<float>(idx, 1);
				A.at<double>(i, 2) = 1;
			}
			mass_point = mass_point * (1.0 / _sample_number);

			cv::Mat AtA = A.t() * A;
			cv::Mat eValues, eVectors;
			cv::eigen(AtA, eValues, eVectors);

			cv::Mat line = eVectors.row(2);
			line.convertTo(line, CV_32F);

			float length = sqrt(line.at<float>(0) * line.at<float>(0) + line.at<float>(1) * line.at<float>(1));
			line.at<float>(0) /= length;
			line.at<float>(1) /= length;

			line.at<float>(2) = -(line.at<float>(0) * mass_point.at<float>(0) + line.at<float>(1) * mass_point.at<float>(1));

			model.descriptor = line.t();

			if (represent_by_points) // Calculate the closest point to the origin
				calculate_point_representation(model.descriptor, model.descriptor_by_points);

			model.estimator = this;
			_models->push_back(model);

			if (model.descriptor.rows == 0 || model.descriptor.cols == 0)
				return false;
			return true;
		}

		bool estimate_model(const cv::Mat * const data,
			const int *sample,
			std::vector<Model>* models) const
		{
			return estimate_from_points(data, sample, models);
		}

		bool estimate_from_points(const cv::Mat * const _data,
			const int *_sample,
			std::vector<Model>* _models) const
		{
			if (_sample[0] == _sample[1])
				return false;

			// model calculation 
			Model model;

			cv::Mat pt1 = _data->row(_sample[0]);
			cv::Mat pt2 = _data->row(_sample[1]);

			cv::Mat v = pt2 - pt1;
			v = v / cv::norm(v);
			cv::Mat n = (cv::Mat_<float>(2, 1) << -v.at<float>(1), v.at<float>(0));
			float c = -(n.at<float>(0) * pt2.at<float>(0) + n.at<float>(1) * pt2.at<float>(1));

			model.descriptor = (cv::Mat_<float>(3, 1) << n.at<float>(0), n.at<float>(1), c);

			if (represent_by_points) // Calculate the closest point to the origin
				calculate_point_representation(model.descriptor, model.descriptor_by_points);

			model.estimator = this;
			_models->push_back(model);
			if (model.descriptor.rows == 0 || model.descriptor.cols == 0)
				return false;
			return true;
		}

		void calculate_implicit_form(const cv::Mat &_point_representation,
			cv::Mat &_descriptor) const
		{
			const cv::Mat pt1 = (cv::Mat_<float>(2, 1) << _point_representation.at<float>(0), _point_representation.at<float>(1));
			const cv::Mat pt2 = (cv::Mat_<float>(2, 1) << _point_representation.at<float>(2), _point_representation.at<float>(3));
			cv::Mat v = pt2 - pt1;
			v = v / cv::norm(v);

			float a = -v.at<float>(1);
			float b = v.at<float>(0);
			float c = -a * pt2.at<float>(0) - b * pt2.at<float>(1);

			_descriptor = (cv::Mat_<float>(3, 1) << a, b, c);
		}

		void calculate_point_representation(const cv::Mat &_descriptor,
			cv::Mat &_point_descriptor) const
		{
			const float a = _descriptor.at<float>(0);
			const float b = _descriptor.at<float>(1);
			const float c = _descriptor.at<float>(2);

			// Calculate the closest point to the origin
			cv::Mat A = (cv::Mat_<float>(3, 3) << 2, 0, 2 * a,
				0, 2, 2 * b,
				a, b, 0);
			cv::Mat inhom = (cv::Mat_<float>(3, 1) << 0, 0, -c);

			cv::Mat x = A.inv() * inhom;

			cv::Point2f pt1(x.at<float>(0), x.at<float>(1));

			// Step one unit in the normal direction
			cv::Mat inhom2 = (cv::Mat_<float>(3, 1) << 2 * image_size.width, 2 * image_size.height, -c);
			x = A.inv() * inhom2;

			cv::Point2f pt2(x.at<float>(0), x.at<float>(1));

			_point_descriptor = (cv::Mat_<float>(4, 1) << pt1.x, pt1.y, pt2.x, pt2.y);
		}

		void set_descriptor(Model &_model, 
			cv::Mat _descriptor) const
		{
			const float a = _descriptor.at<float>(0);
			const float b = _descriptor.at<float>(1);
			const float mag = sqrt(a * a + b * b);
			_descriptor.at<float>(0) = a / mag;
			_descriptor.at<float>(1) = b / mag;

			_model.descriptor = _descriptor;
		}

		float error(const cv::Mat& _point, 
			const Model& _model) const
		{
			return error(_point, _model.descriptor);
		}

		float error(const cv::Mat& _point, 
			const cv::Mat& _descriptor) const
		{
			float distance = abs(_point.at<float>(0) * _descriptor.at<float>(0) + 
				_point.at<float>(1) * _descriptor.at<float>(1) + 
				_descriptor.at<float>(2));
			return distance;
		}

		bool is_valid(const cv::Mat * const data, 
			const cv::Mat& descriptor, 
			std::vector<int> const * inliers) const
		{
			return true;
		}
	};
}
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


namespace multix
{
	// This is the estimator class for estimating a homography matrix between two images. A model estimation method and error calculation method are implemented
	class MotionEstimator : public Estimator < cv::Mat, Model >
	{
	protected:
		cv::Size image_size;

	public:
		MotionEstimator() 
		{
			represent_by_points = false;
		}

		~MotionEstimator() {}

		bool is_non_minimal_fitting_applicable() const override
		{
			return false;
		}

		bool is_mode_seeking_applicable() const
		{
			return true;
		}

		int sample_size() const 
		{
			return 4;
		}

		int inlier_limit() const 
		{
			return 7 * sample_size();
		}

		float model_weight() const
		{
			return 1;
		}

		void gram_smith_orthogonalization(cv::Mat const &_points, 
			cv::Mat &_result) const
		{
			int K = _points.rows;
			int D = _points.cols;

			static cv::Mat I = cv::Mat::eye(K, K, CV_32F);
			_result = cv::Mat::zeros(K, D, CV_32F);

			cv::Mat col = _points.col(0) / cv::norm(_points.col(0));
			col.copyTo(_result.col(0));

			for (auto i = 1; i < D; ++i)
			{
				cv::Mat new_column = (I - _result * _result.t()) * _points.col(i);
				new_column = new_column / norm(new_column);
				new_column.copyTo(_result.col(i));
			}
		}

		bool estimate_model(const cv::Mat * const _data,
			const int * const _sample,
			std::vector<Model>* _models) const
		{
			// model calculation 
			static const int M = sample_size();
			const int dimensions = _data->cols;

			cv::Mat tmp_points(dimensions, M, CV_32F);
			
			for (auto sample_idx = 0; sample_idx < M; ++sample_idx)
			{
				const float * data_ptr = reinterpret_cast<float *>(_data->data) + _sample[sample_idx] * dimensions;
				float * tmp_points_ptr = reinterpret_cast<float *>(tmp_points.data) + sample_idx;

				for (auto dim = 0; dim < _data->cols; ++dim)
				{
					*tmp_points_ptr = *data_ptr++;
					tmp_points_ptr += dimensions - 1;
				}
			}
			
			cv::Mat result;
			gram_smith_orthogonalization(tmp_points, result);

			static cv::Mat I = cv::Mat::eye(dimensions, dimensions, CV_32F);

			Motion model;
			model.descriptor = I - result * result.t();
			model.descriptor = model.descriptor;
			model.descriptor_by_points = model.descriptor;
			model.estimator = this;
			_models->push_back(model);
			return true;
		}

		bool estimate_model_nonminimal(const cv::Mat * const _data,
			const int * const _sample,
			int _sample_number,
			std::vector<Model>* _models) const
		{
			return false;
		}

		float error(const cv::Mat& _point, 
			const Model& _model) const
		{
			return error(_point, _model.descriptor);
		}

		float error(const cv::Mat& _point,
			const cv::Mat& _descriptor) const
		{
			const float* s = (float *)_point.data;
			const float* p = (float *)_descriptor.data;

			cv::Mat distance_matrix = _point * _descriptor;

			float dist = 0;
			for (int i = 0; i < _point.cols; ++i)
				dist += distance_matrix.at<float>(i) * distance_matrix.at<float>(i);
			return sqrt(dist); //abs(error[0]);
		}

		void calculate_implicit_form(const cv::Mat &_point_representation, 
			cv::Mat &_descriptor) const
		{

		}

		void calculate_point_representation(const cv::Mat &_descriptor, 
			cv::Mat &_point_descriptor) const
		{

		}

		void set_descriptor(Model &_model,
			cv::Mat _descriptor) const
		{
			float dimensions;

			if (_descriptor.rows == _descriptor.cols)
			{
				_model.descriptor = _descriptor;
				_model.descriptor_by_points = _model.descriptor;
			}
			else
			{
				if (_descriptor.cols == 1)
				{
					float sqrt_rows = sqrt(_descriptor.rows);
					if (sqrt_rows == (int)sqrt_rows)
					{
						_model.descriptor = _descriptor.reshape(0, sqrt_rows);
						_model.descriptor_by_points = _model.descriptor;
						return;
					}
				}
				else if (_descriptor.rows == 1)
				{
					float sqrt_cols = sqrt(_descriptor.cols);
					if (sqrt_cols == (int)sqrt_cols)
					{
						_model.descriptor = _descriptor.reshape(0, sqrt_cols);
						_model.descriptor_by_points = _model.descriptor;
						return;
					}
				}
			}
		}

		bool is_valid(const cv::Mat * const _data, 
			const cv::Mat& _descriptor, 
			std::vector<int> const * _inliers) const
		{
			return !_descriptor.empty();
		}
	};
}
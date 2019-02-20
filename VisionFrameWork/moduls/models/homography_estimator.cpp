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
	class HomographyEstimator : public Estimator < cv::Mat, Model >
	{
	protected:
		cv::Size image_size;
		const float stability_threshold = 1e-2f;

	public:
		HomographyEstimator(cv::Size _image_size) : image_size(_image_size)
		{
			represent_by_points = false;
		}
		~HomographyEstimator() {}

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

		bool estimate_model(const cv::Mat * const _data,
			const int * const _sample,
			std::vector<Model>* _models) const
		{
			// model calculation 
			static const int M = sample_size();

			static std::vector<cv::Point2d> pts1(M);
			static std::vector<cv::Point2d> pts2(M);

			for (auto i = 0; i < M; ++i)
			{
				pts1[i].x = static_cast<double>(_data->at<float>(_sample[i], 0));
				pts1[i].y = static_cast<double>(_data->at<float>(_sample[i], 1));
				pts2[i].x = static_cast<double>(_data->at<float>(_sample[i], 3));
				pts2[i].y = static_cast<double>(_data->at<float>(_sample[i], 4));
			}

			cv::Mat H = cv::findHomography(pts1, pts2);
			H.convertTo(H, CV_32F);
			
			if (H.empty())
				return false;

			float err;
			for (auto i = 0; i < M; ++i)
			{
				err = error(_data->row(_sample[i]), H);
				if (err > stability_threshold ||
					isnan(err))
					return false;
			}

			if (H.cols != 3 || H.rows != 3)
				return false;

			Homography model;
			set_descriptor(model, H);
			model.estimator = this;

			if (represent_by_points)
			{
				model.descriptor_by_points.create(8, 1, CV_32F);
				calculate_point_representation(model.descriptor, model.descriptor_by_points);
			}
			else
				model.descriptor_by_points = model.descriptor;

			_models->push_back(model);
			return true;
		}

		bool estimate_model_nonminimal(const cv::Mat * const _data,
			const int * const _sample,
			int _sample_number,
			std::vector<Model>* _models) const
		{
			if (_sample_number < sample_size())
				return false;

			// model calculation 
			int M = _sample_number;

			std::vector<cv::Point2d> pts1(M);
			std::vector<cv::Point2d> pts2(M);

			for (auto i = 0; i < M; ++i)
			{
				pts1[i].x = static_cast<double>(_data->at<float>(_sample[i], 0));
				pts1[i].y = static_cast<double>(_data->at<float>(_sample[i], 1));
				pts2[i].x = static_cast<double>(_data->at<float>(_sample[i], 3));
				pts2[i].y = static_cast<double>(_data->at<float>(_sample[i], 4));
			}

			cv::Mat H = findHomography(pts1, pts2);
			H.convertTo(H, CV_32F);

			if (H.cols != 3 || H.rows != 3)
				return false;

			Homography model;
			model.estimator = this;
			set_descriptor(model, H);
			_models->push_back(model);
			return true;
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

			const float x1 = *s;
			const float y1 = *(s + 1);
			const float x2 = *(s + 3);
			const float y2 = *(s + 4);

			const float t1 = *p * x1 + *(p + 1) * y1 + *(p + 2);
			const float t2 = *(p + 3) * x1 + *(p + 4) * y1 + *(p + 5);
			const float t3 = *(p + 6) * x1 + *(p + 7) * y1 + *(p + 8);

			const float d1 = x2 - (t1 / t3);
			const float d2 = y2 - (t2 / t3);

			return sqrt(d1 * d1 + d2 * d2);
		}

		void calculate_implicit_form(const cv::Mat &_point_representation, 
			cv::Mat &_descriptor) const
		{
			_descriptor = _point_representation.clone();
		}

		void calculate_point_representation(const cv::Mat &_descriptor, 
			cv::Mat &_point_descriptor) const
		{
			static cv::Mat pt1 = (cv::Mat_<float>(3, 1) << 0, 0, 1);
			static cv::Mat pt2 = (cv::Mat_<float>(3, 1) << image_size.width, 0, 1);
			static cv::Mat pt3 = (cv::Mat_<float>(3, 1) << 0, image_size.height, 1);
			static cv::Mat pt4 = (cv::Mat_<float>(3, 1) << image_size.width, image_size.height, 1);
			static cv::Mat *pts[4] = { &pt1, &pt2, &pt3, &pt4 };

			_point_descriptor.create(1, 8, CV_32F);
			float *point_descriptor_ptr = reinterpret_cast<float *>(_point_descriptor.data);

			cv::Mat transformed_point;
			for (auto point_idx = 0; point_idx < 4; ++point_idx)
			{
				transformed_point = _descriptor * *pts[point_idx];
				transformed_point = transformed_point / transformed_point.at<float>(2);
				*(point_descriptor_ptr++) = transformed_point.at<float>(0);
				*(point_descriptor_ptr++) = transformed_point.at<float>(1);
			}
			//std::cout << _point_descriptor << std::endl;
			//std::cout << std::endl;
		}

		void set_descriptor(Model &_model, 
			cv::Mat _descriptor) const
		{
			if (_descriptor.rows == _descriptor.cols && _descriptor.rows == 3)
			{
				_descriptor = _descriptor / _descriptor.at<float>(2,2);
				_model.descriptor = _descriptor;
				_model.descriptor_by_points = _model.descriptor;
			}
			else if (_descriptor.cols == 1 && _descriptor.rows == 9)
			{
				_descriptor = _descriptor / _descriptor.at<float>(2, 2);
				_model.descriptor = _descriptor.reshape(0, 3);
				_model.descriptor = _model.descriptor;
				_model.descriptor_by_points = _model.descriptor;
			}
			else if (_descriptor.rows == 1 && _descriptor.cols == 9)
			{
				_descriptor = _descriptor / _descriptor.at<float>(2, 2);
				_model.descriptor = _descriptor.reshape(0, 3);
				_model.descriptor = _model.descriptor;
				_model.descriptor_by_points = _model.descriptor;
			}
		}

		bool is_valid(const cv::Mat * const _data, 
			const cv::Mat& _descriptor, 
			std::vector<int> const * _inliers) const
		{
			return _descriptor.rows == 3 &&
				_descriptor.cols == 3;
		}
	};
}
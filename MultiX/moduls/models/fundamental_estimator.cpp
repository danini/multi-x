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

#include <opencv2\calib3d\calib3d.hpp>
#include "estimator.h"
#include "model.h"

namespace multix
{
	// This is the estimator class for estimating a homography matrix between two images. A model estimation method and error calculation method are implemented
	class FundamentalMatrixEstimator : public Estimator < cv::Mat, Model >
	{
	protected:

	public:
		FundamentalMatrixEstimator() 
		{
			represent_by_points = false;			
		}
		~FundamentalMatrixEstimator() {}

		bool is_mode_seeking_applicable() const
		{
			return true;
		}

		int sample_size() const {
			return 7;
		}

		int inlier_limit() const {
			return 7 * sample_size();
		}

		float model_weight() const
		{
			return 1.f;
		}

		bool estimate_model(const cv::Mat * const _data,
			const int * const _sample,
			std::vector<Model> *_models) const
		{
			// Model calculation 
			static const int M = sample_size();
			static std::vector<cv::Point2d> src_points(M), dst_points(M);

			for (auto point_idx = 0; point_idx < M; point_idx++)
			{
				float x0 = _data->at<float>(_sample[point_idx], 0), y0 = _data->at<float>(_sample[point_idx], 1);
				float x1 = _data->at<float>(_sample[point_idx], 3), y1 = _data->at<float>(_sample[point_idx], 4);
				src_points[point_idx].x = x0;
				src_points[point_idx].y = y0;
				dst_points[point_idx].x = x1;
				dst_points[point_idx].y = y1;
			}			

			std::vector<uchar> mask;
			cv::Mat Fs = cv::findFundamentalMat(src_points, dst_points, mask, CV_FM_7POINT);
			Fs.convertTo(Fs, CV_32F);

			for (auto F_idx = 0; F_idx < Fs.rows / 3; ++F_idx)
			{
				cv::Mat F(3, 3, CV_32F);
				memcpy(F.data, Fs.data + F_idx * 9 * sizeof(float), 9 * sizeof(float));

				/*std::cout << "---\nError = " << error(_data->row(_sample[0]), F) << std::endl;
				std::cout << "Error = " << error(_data->row(_sample[1]), F) << std::endl;
				std::cout << "Error = " << error(_data->row(_sample[2]), F) << std::endl;
				std::cout << "Error = " << error(_data->row(_sample[3]), F) << std::endl << std::endl << std::endl;*/
				
				Model model;
				model.estimator = this;
				set_descriptor(model, F);

				if (is_represented_by_points())
					calculate_point_representation(model.descriptor, model.descriptor_by_points);
				_models->push_back(model);
			}

			if (_models->size() == 0)
				return false;
			return true;
		}

		bool estimate_model_nonminimal(const cv::Mat * const _data,
			const int * const _sample,
			int _sample_number,
			std::vector<Model>* _models) const
		{
			// Model calculation 
			const int M = _sample_number;
			if (M < 8)
				return false;
			
			std::vector<cv::Point2d> src_points(M), dst_points(M);

			for (auto point_idx = 0; point_idx < M; point_idx++)
			{
				float x0 = _data->at<float>(_sample[point_idx], 0), y0 = _data->at<float>(_sample[point_idx], 1);
				float x1 = _data->at<float>(_sample[point_idx], 3), y1 = _data->at<float>(_sample[point_idx], 4);
				src_points[point_idx].x = x0;
				src_points[point_idx].y = y0;
				dst_points[point_idx].x = x1;
				dst_points[point_idx].y = y1;
			}

			std::vector<uchar> mask;
			cv::Mat F = cv::findFundamentalMat(src_points, dst_points, mask, CV_FM_8POINT);
			F.convertTo(F, CV_32F);

			Model model;
			model.estimator = this;
			set_descriptor(model, F);

			if (is_represented_by_points())
				calculate_point_representation(model.descriptor, model.descriptor_by_points);
			_models->push_back(model);

			if (_models->size() == 0)
				return false;
			return true;
		}

		float error(const cv::Mat& _point, 
			const Model& _model) const
		{
			const float* s = (float *)_point.data;
			const float x1 = *s;
			const float y1 = *(s + 1);
			const float x2 = *(s + 3);
			const float y2 = *(s + 4);

			const float* p = (float *)_model.descriptor.data;

			const float l1 = *(p)* x2 + *(p + 3) * y2 + *(p + 6);
			const float l2 = *(p + 1) * x2 + *(p + 4) * y2 + *(p + 7);
			const float l3 = *(p + 2) * x2 + *(p + 5) * y2 + *(p + 8);

			const float t1 = *(p)* x1 + *(p + 1) * y1 + *(p + 2);
			const float t2 = *(p + 3) * x1 + *(p + 4) * y1 + *(p + 5);
			const float t3 = *(p + 6) * x1 + *(p + 7) * y1 + *(p + 8);

			const float a1 = l1 * x1 + l2 * y1 + l3;
			const float a2 = sqrt(l1 * l1 + l2 * l2);

			const float b1 = t1 * x2 + t2 * y2 + t3;
			const float b2 = sqrt(t1 * t1 + t2 * t2);

			const float d = l1 * x2 + l2 * y2 + l3;
			const float d1 = a1 / a2;
			const float d2 = b1 / b2;

			return abs(0.5f * (d1 + d2));
		}

		float error(const cv::Mat& _point, 
			const cv::Mat& _descriptor) const
		{
			const float* s = (float *)_point.data;
			const float x1 = *s;
			const float y1 = *(s + 1);
			const float x2 = *(s + 3);
			const float y2 = *(s + 4);

			const float* p = (float *)_descriptor.data;

			const float l1 = *(p)* x2 + *(p + 3) * y2 + *(p + 6);
			const float l2 = *(p + 1) * x2 + *(p + 4) * y2 + *(p + 7);
			const float l3 = *(p + 2) * x2 + *(p + 5) * y2 + *(p + 8);

			const float t1 = *(p)* x1 + *(p + 1) * y1 + *(p + 2);
			const float t2 = *(p + 3) * x1 + *(p + 4) * y1 + *(p + 5);
			const float t3 = *(p + 6) * x1 + *(p + 7) * y1 + *(p + 8);

			const float a1 = l1 * x1 + l2 * y1 + l3;
			const float a2 = sqrt(l1 * l1 + l2 * l2);

			const float b1 = t1 * x2 + t2 * y2 + t3;
			const float b2 = sqrt(t1 * t1 + t2 * t2);

			const float d = l1 * x2 + l2 * y2 + l3;
			const float d1 = a1 / a2;
			const float d2 = b1 / b2;

			return abs(0.5f * (d1 + d2));
		}

		void set_descriptor(Model &_model, 
			cv::Mat _descriptor) const
		{
			if (_descriptor.rows == _descriptor.cols && _descriptor.rows == 3)
			{
				_model.descriptor = _descriptor;
				_model.descriptor = _model.descriptor / _model.descriptor.at<float>(2, 2);
				_model.descriptor_by_points = _model.descriptor;
			}
			else if (_descriptor.cols == 1 && _descriptor.rows == 9)
			{
				_model.descriptor = _descriptor.reshape(0, 3);
				_model.descriptor = _model.descriptor;
				_model.descriptor = _model.descriptor / _model.descriptor.at<float>(2, 2);
				_model.descriptor_by_points = _model.descriptor;
			}
			else if (_descriptor.rows == 1 && _descriptor.cols == 9)
			{
				_model.descriptor = _descriptor.reshape(0, 3);
				_model.descriptor = _model.descriptor;
				_model.descriptor = _model.descriptor / _model.descriptor.at<float>(2, 2);
				_model.descriptor_by_points = _model.descriptor;
			}
		}

		void calculate_implicit_form(const cv::Mat &_point_representation, 
			cv::Mat &_descriptor) const
		{
			_descriptor = _point_representation.clone();
		}

		void calculate_point_representation(const cv::Mat &_descriptor, 
			cv::Mat &_point_descriptor) const
		{
			_point_descriptor = _descriptor.clone();
		}

		bool is_valid(const cv::Mat * const _data, 
			const cv::Mat& _descriptor, 
			std::vector<int> const *_inliers) const
		{
			return true;
		}

		/************** oriented constraints ******************/
		void epipole(cv::Mat &_ec,
			const cv::Mat *_F) const
		{
			_ec = _F->row(0).cross(_F->row(2));

			for (auto i = 0; i < 3; i++)
				if ((_ec.at<float>(i) > 1.9984e-15) || (_ec.at<float>(i) < -1.9984e-15)) return;
			_ec = _F->row(1).cross(_F->row(2));
		}

		float getorisig(const cv::Mat *_F, 
			const cv::Mat *_ec, 
			const cv::Mat &_u) const
		{
			float s1, s2;

			s1 = _F->at<float>(0) * _u.at<float>(3) + _F->at<float>(3) * _u.at<float>(4) + _F->at<float>(6) * _u.at<float>(5);
			s2 = _ec->at<float>(1) * _u.at<float>(2) - _ec->at<float>(2) * _u.at<float>(1);
			return(s1 * s2);
		}

		int all_ori_valid(const cv::Mat *_F, 
			const cv::Mat * const _data, 
			const int * const _sample, 
			int _N) const
		{
			cv::Mat ec;
			float sig, sig1;
			int i;
			epipole(ec, _F);

			sig1 = getorisig(_F, &ec, _data->row(_sample[0]));
			for (i = 1; i < _N; i++)
			{
				sig = getorisig(_F, &ec, _data->row(_sample[i]));
				if (sig1 * sig < 0) return 0;
			}
			return 1;

		}
	};
}
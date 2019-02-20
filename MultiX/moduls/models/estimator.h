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
#include <opencv2\calib3d\calib3d.hpp>

namespace multix
{
	// Templated class for estimating a model for RANSAC. This class is purely a
	// virtual class and should be implemented for the specific task that RANSAC is
	// being used for. 
	template <typename DatumType, typename ModelType> class Estimator
	{
	protected:
		bool represent_by_points;

	public:
		typedef DatumType Datum;
		typedef ModelType Model;

		Estimator() : represent_by_points(true) {}
		virtual ~Estimator() {}

		bool is_represented_by_points() const { return represent_by_points; }
		virtual bool is_non_minimal_fitting_applicable() const { return true; }

		// Get the minimum number of samples needed to generate a model.
		virtual int sample_size() const = 0;
		virtual int inlier_limit() const = 0;
		virtual float model_weight() const = 0;
		virtual bool is_mode_seeking_applicable() const = 0;

		// Given a set of data points, estimate the model. Users should implement this
		// function appropriately for the task being solved. Returns true for
		// successful model estimation (and outputs model), false for failed
		// estimation. Typically, this is a minimal set, but it is not required to be.
		virtual bool estimate_model(const cv::Mat * const _data,
			const int *_sample, 
			std::vector<Model>* _model) const = 0;

		// Estimate a model from a non-minimal sampling of the data. E.g. for a line,
		// use SVD on a set of points instead of constructing a line from two points.
		// By default, this simply implements the minimal case.
		virtual bool estimate_model_nonminimal(const cv::Mat * const _data,
			const int *_sample,
			int _sample_number,
			std::vector<Model>* _model) const = 0;

		// Refine the model based on an updated subset of data, and a pre-computed
		// model. Can be optionally implemented.
		virtual bool refine_model(const std::vector<Datum>& _data, 
			Model* _model) const 
		{
			return true;
		}

		// Given a model and a data point, calculate the error. Users should implement
		// this function appropriately for the task being solved.
		virtual float error(const Datum& data, 
			const Model& model) const = 0;

		virtual float error(const Datum& data, 
			const cv::Mat& model) const = 0;

		// Enable a quick check to see if the model is valid. This can be a geometric
		// check or some other verification of the model structure.
		virtual bool is_valid(const cv::Mat * const _data, 
			const cv::Mat& _descriptor, 
			std::vector<int> const *_inliers) const = 0;
		
		virtual void set_descriptor(Model &_model, 
			Datum _descriptor) const = 0;

		virtual void calculate_point_representation(const cv::Mat &_descriptor,
			cv::Mat &_point_descriptor) const = 0;

		virtual void calculate_implicit_form(const cv::Mat &_point_representation,
			cv::Mat &_descriptor) const = 0;
	};

}  // namespace multix
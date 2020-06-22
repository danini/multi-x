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

#define M_PI 3.1415926535897932384626433832795028841971

#include <string.h>
#include <opencv\cv.hpp>
#include <fstream>
#include <cv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <functional>
#include <algorithm>
#include <opencv2\xfeatures2d\nonfree.hpp>
#include <math.h>
#include "model.h"

/*
Generate random 2D lines and points
*/
template<typename T>
void generate_2d_lines(int _line_number,
	int _points_per_lines,
	int _outlier_number,
	T _noise,
	cv::Mat &_points,
	std::vector<multix::Model> &_lines,
	cv::Size _size,
	std::vector<int> * const _labeling = NULL,
	T _margin = 10.0f,
	const T * const _data = NULL)
{
	static cv::RNG rng(123456);
	const int NI = _line_number * _points_per_lines;
	const int N = NI + _outlier_number;

	_points = cv::Mat_<T>(N, 2);
	T *points_ptr = reinterpret_cast<T*>(_points.data);
	_lines.resize(_line_number);
	if (_labeling != NULL)
		_labeling->resize(N);

	// Generate lines
	if (_data == NULL)
	{
		T angle, center_x, center_y;
		for (auto i = 0; i < _line_number; ++i)
		{
			angle = static_cast<T>(rng.uniform(0.0, M_PI));
			center_x = rng.uniform(_margin, _size.width - 2 * _margin);
			center_y = rng.uniform(_margin, _size.height - 2 * _margin);

			T a = cos(angle);
			T b = sin(angle);
			T c = -a * center_x - b * center_y;

			_lines[i].descriptor = (cv::Mat_<T>(3, 1) << a, b, c);
		}

		for (auto i = 0; i < N; ++i)
		{
			T x, y;

			if (i < NI) // Inliers
			{
				const int line_idx = i / _points_per_lines;
				const T a = _lines[line_idx].descriptor.at<T>(0);
				const T b = _lines[line_idx].descriptor.at<T>(1);
				const T c = _lines[line_idx].descriptor.at<T>(2);

				while (1)
				{
					if (a > b)
					{
						y = static_cast<T>(rng.uniform(0, _size.height));
						x = (-y * b - c) / a;
					}
					else
					{
						x = static_cast<T>(rng.uniform(0, _size.width));
						y = (-x * a - c) / b;
					}

					if (x >= 0 && x < _size.width &&
						y >= 0 && y < _size.height)
						break;
				}

				*(points_ptr++) = x + static_cast<T>(rng.gaussian(_noise));
				*(points_ptr++) = y + static_cast<T>(rng.gaussian(_noise));

				if (_labeling != NULL)
					_labeling->at(i) = line_idx;

			}
			else // Outliers
			{
				*(points_ptr++) = static_cast<T>(rng.uniform(0, _size.width));
				*(points_ptr++) = static_cast<T>(rng.uniform(0, _size.height));

				if (_labeling != NULL)
					_labeling->at(i) = 0;
			}
		}
	}
	else
	{
		for (auto i = 0; i < _line_number; ++i)
		{
			int offset = i * 7;
			_lines[i].descriptor = (cv::Mat_<T>(3, 1) << _data[offset], _data[offset + 1], _data[offset + 2]);
		}

		for (auto i = 0; i < N; ++i)
		{
			T x, y;

			if (i < NI) // Inliers
			{
				const int line_idx = i / _points_per_lines;
				int offset = line_idx * 7;
				const T a = _lines[line_idx].descriptor.at<T>(0);
				const T b = _lines[line_idx].descriptor.at<T>(1);
				const T c = _lines[line_idx].descriptor.at<T>(2);

				while (1)
				{
					if (a > b)
					{
						y = static_cast<T>(rng.uniform(0, _size.height));
						x = (-y * b - c) / a;
					}
					else
					{
						x = static_cast<T>(rng.uniform(0, _size.width));
						y = (-x * a - c) / b;
					}

					if (x > MAX(0, _data[offset + 3]) &&
						x < MIN(_data[offset + 4], _size.width) &&
						y >= MAX(0, _data[offset + 5]) &&
						y < MIN(_data[offset + 6], _size.height))
						break;
				}

				*(points_ptr++) = x + static_cast<T>(rng.gaussian(_noise));
				*(points_ptr++) = y + static_cast<T>(rng.gaussian(_noise));

				if (_labeling != NULL)
					_labeling->at(i) = line_idx;

			}
			else // Outliers
			{
				*(points_ptr++) = static_cast<T>(rng.uniform(0, _size.width));
				*(points_ptr++) = static_cast<T>(rng.uniform(0, _size.height));

				if (_labeling != NULL)
					_labeling->at(i) = 0;
			}
		}
	}
}

/*
Draw 2D points to image
*/
template<typename T>
void draw_2d_points(const cv::Mat const *_points, 
	cv::Mat &_image, 
	cv::Scalar _color, 
	int _size)
{
	for (auto i = 0; i < _points->rows; ++i)
		cv::circle(_image, static_cast<cv::Point_<T>>(_points->row(i)), _size, _color, -1);
}

/*
Draw 2D line in implicit form to image
*/
template<typename T>
void draw_2d_line(const cv::Mat const *_line, 
	cv::Mat &_image, 
	cv::Scalar _color, 
	int _size)
{
	const T a = _line->at<T>(0);
	const T b = _line->at<T>(1);
	const T c = _line->at<T>(2);
	
	if (abs(a) > abs(b))
	{
		const T y1 = 0;
		const T x1 = -c / a;

		const T y2 = static_cast<T>(_image.rows);
		const T x2 = -(c + b * y2) / a;

		cv::line(_image, 
			cv::Point_<T>(x1, y1), 
			cv::Point_<T>(x2, y2), 
			_color, 
			_size);
	}
	else
	{
		const T x1 = 0;
		const T y1 = -c / b;

		const T x2 = static_cast<T>(_image.cols);
		const T y2 = -(c + a * x2) / b;

		cv::line(_image, 
			cv::Point_<T>(x1, y1), 
			cv::Point_<T>(x2, y2), 
			_color, 
			_size);
	}
}

/*
Detect and match SURF features
*/
inline void feature_detection(cv::Mat _image1, 
	cv::Mat _image2, 
	std::vector<cv::Point2f> &_src_points,
	std::vector<cv::Point2f> &_dst_points)
{
	printf("Detect SURF features\n");
	cv::Mat descriptors1, descriptors2;
	std::vector<cv::KeyPoint> keypoints1, keypoints2;
	
	cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create();
	detector->detect(_image1, keypoints1);
	detector->compute(_image1, keypoints1, descriptors1);
	printf("Features found in the first image: %d\n", static_cast<int>(keypoints1.size()));

	detector->detect(_image2, keypoints2);
	detector->compute(_image2, keypoints2, descriptors2);
	printf("Features found in the second image: %d\n", static_cast<int>(keypoints2.size()));

	std::vector<std::vector< cv::DMatch >> matches_vector;
	cv::FlannBasedMatcher matcher(new cv::flann::KDTreeIndexParams(5), new cv::flann::SearchParams(32));
	matcher.knnMatch(descriptors1, descriptors2, matches_vector, 2);

	for (auto m : matches_vector)
	{
		if (m.size() == 2 && m[0].distance < m[1].distance * 0.75)
		{
			auto& kp1 = keypoints1[m[0].queryIdx];
			auto& kp2 = keypoints2[m[0].trainIdx];
			_src_points.push_back(kp1.pt);
			_dst_points.push_back(kp2.pt);
		}
	}
	printf("Match number: %d\n", static_cast<int>(_src_points.size()));
}

/*
Load file to N x M matrix with or without labels
*/
inline void load_n_times_m_matrix(std::string _source_file,
	cv::Mat &_data,
	std::vector<int> *_labels,
	int _rows = -1,
	int _columns = -1,
	bool _has_header = true)
{
	std::ifstream file(_source_file);

	if (!file.is_open())
	{
		printf("Error while opening '%s'.\n", _source_file.c_str());
		return;
	}

	if (_has_header)
	{
		if (_rows == -1 || _columns == -1)
		{
			file >> _columns >> _rows;
			_columns = 2 * _columns;
		}

		if (_labels != NULL)
			_labels->resize(_rows);
		_data.create(_rows, _columns, CV_32F);
		auto data_ptr = reinterpret_cast<float *>(_data.data);

		for (auto row = 0; row < _rows; ++row)
		{
			for (auto column = 0; column < _columns; ++column)
				file >> *data_ptr++;

			if (_labels != NULL)
				file >> _labels->at(row);
		}
	}
	else
	{
		if (_rows > 0 && _columns > 0)
		{
			_data.create(_rows, _columns, CV_32F);
			auto data_ptr = reinterpret_cast<float *>(_data.data);
			while (file >> *data_ptr++);
		}
		else if (_columns > 0)
		{
			std::vector<float> data;
			float value;
			size_t counter = 0;
			while (file >> value) 
			{
				++counter;
				if (!_labels == NULL && counter % _columns == 0)
					_labels->push_back(value);
				else
					data.push_back(value);
			}
			_rows = data.size() / (_columns - (_labels == NULL ? 0 : 1));			
			_data.create(_rows, _columns - (_labels == NULL ? 0 : 1), CV_32F);
			memcpy(_data.data, &data[0], data.size() * sizeof(float));
		}
		else
		{
			printf("Error in loading '%s'\n", _source_file.c_str());
		}
	}

	file.close();
}

inline void project_points_to_r_dimensional_space(cv::Mat _points, 
	cv::Mat &_projected_points, 
	int _r)
{
	cv::Mat S, U, Vt;
	cv::SVD::compute(_points, S, U, Vt, cv::SVD::FULL_UV);

	_projected_points = cv::Mat(_points.rows, _r, CV_32F);

	for (int i = 0; i < _r; ++i)
		U.col(i).copyTo(_projected_points.col(i));
}
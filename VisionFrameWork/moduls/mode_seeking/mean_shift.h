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

#include <thread>
#include <cv.h>

template<typename T>
class mean_shift_clustering
{
public:
	enum distance_metric { eucledian, pointwise_eucledian_2d };
	enum pointwise_distance { max, mean };

	mean_shift_clustering() :
		max_iterations(100000),
		max_inner_iterations(10000),
		current_distance_metric(distance_metric::eucledian),
		current_pointwise_distance(pointwise_distance::mean)
	{ 

	}
	~mean_shift_clustering() {}

	void set_distance_metric(distance_metric _metric) { current_distance_metric = _metric; }

	void cluster(cv::InputArray _data, 
		T _band_width, 
		cv::OutputArray _clusters, 
		std::vector<std::vector<int>> &_cluster_points);
	
protected:
	int max_iterations;
	int max_inner_iterations;
	pointwise_distance current_pointwise_distance;
	distance_metric current_distance_metric;
	
	void get_min_max_position(cv::Mat const &_data, 
		cv::Mat &_min_position, 
		cv::Mat &_max_position);

	void distance_eucledian(const cv::Mat const &_data,
		const cv::Mat &_old_mean,
		const int _point_number,
		const int _dimension_number,
		const float _sqr_band_width,
		cv::Mat &_current_cluster_votes,
		cv::Mat &_new_mean,
		int * const _been_visited_flag,
		std::vector<int> &_current_members,
		std::vector<int> &_inlier_indices);

	void distance_pointwise_eucledian_2d(const cv::Mat const &_data,
		const cv::Mat &_old_mean,
		const int _point_number,
		const int _dimension_number,
		const float _sqr_band_width,
		cv::Mat &_current_cluster_votes,
		cv::Mat &_new_mean,
		int * const _been_visited_flag,
		std::vector<int> &_current_members,
		std::vector<int> &_inlier_indices);
};

template<typename T>
void mean_shift_clustering<T>::cluster(cv::InputArray _data, 
	T _band_width, 
	cv::OutputArray _clusters, 
	std::vector<std::vector<int>> &_cluster_points)
{ 
	// Initialization
	static cv::RNG rng(123456);
	const cv::Mat data = _data.getMat();

	const int dimension_number = data.cols;
	const int point_number = data.rows;
	const T sqr_band_width = _band_width * _band_width;
	const T half_band_width = _band_width / 2;

	int cluster_number = 0;

	std::vector<int> init_point_indices(point_number);
	auto been_visited_flag = std::make_unique<int[]>(point_number);
	std::vector<cv::Mat> cluster_votes;

	concurrency::parallel_for(0, point_number, [&](int i)
	{
		init_point_indices[i] = i;
		been_visited_flag[i] = 0;
	});

	cv::Mat min_position, max_position;
	get_min_max_position(data, min_position, max_position);

	cv::Mat boundBox = max_position - min_position;
	T bounding_box_size = static_cast<T>(norm(boundBox));
	T stop_threshold = static_cast<T>(1e-3 * _band_width);

	std::vector<cv::Mat> cluster_centers;
	std::vector<int> current_members;
	current_members.reserve(point_number);

	int last_idx_size = static_cast<int>(init_point_indices.size());
	int iterations = 0, temp_idx, point_idx;
	cv::Mat current_mean, current_cluster_votes;
	while (init_point_indices.size() && iterations++ < max_iterations)
	{
		temp_idx = rng.uniform(0, static_cast<int>(init_point_indices.size()));
		point_idx = init_point_indices[temp_idx];

		current_mean = data.row(point_idx).clone();
		current_cluster_votes = cv::Mat_<int>(1, point_number, 0);
		current_members.resize(0);

		int inner_iterations = 0;
		while (inner_iterations++ < max_inner_iterations)
		{
			// dist squared from mean to all points still active
			cv::Mat old_mean = current_mean.clone();
			std::vector<int> inlier_indices;
			inlier_indices.reserve(point_number);

			current_mean = cv::Mat::zeros(1, data.cols, data.type());
			switch (current_distance_metric)
			{
			case distance_metric::pointwise_eucledian_2d:
				distance_pointwise_eucledian_2d(data,
					old_mean,
					point_number,
					dimension_number,
					sqr_band_width,
					current_cluster_votes,
					current_mean,
					been_visited_flag.get(),
					current_members,
					inlier_indices);
				break;
			case distance_metric::eucledian: default:
				distance_eucledian(data,
					old_mean,
					point_number,
					dimension_number,
					sqr_band_width,
					current_cluster_votes,
					current_mean,
					been_visited_flag.get(),
					current_members,
					inlier_indices);
				break;
			}

			current_mean = current_mean / static_cast<T>(inlier_indices.size());

			if (inlier_indices.size() == 0)
				current_mean = old_mean;

			if (norm(current_mean - old_mean) < stop_threshold)
			{
				int merge_with = -1;
				for (auto cluster_idx = 0; cluster_idx < cluster_centers.size(); ++cluster_idx)
				{
					T distance_to_other = static_cast<T>(norm(current_mean - cluster_centers[cluster_idx]));
					if (distance_to_other < half_band_width)
					{
						merge_with = cluster_idx;
						break;
					}
				}

				if (merge_with > -1)
				{
					cluster_centers[merge_with] = 0.5 * (cluster_centers[merge_with] + current_mean);
					cluster_votes[merge_with] = cluster_votes[merge_with] + current_cluster_votes;
				}
				else
				{
					cluster_centers.push_back(current_mean);
					cluster_votes.push_back(current_cluster_votes);
				}
				break;
			}
		}

		init_point_indices.resize(0);
		for (auto point_idx = 0; point_idx < point_number; ++point_idx)
			if (been_visited_flag[point_idx] == 0)
				init_point_indices.push_back(point_idx);

		if (last_idx_size == init_point_indices.size())
			break;
		last_idx_size = static_cast<int>(init_point_indices.size());
	}

	std::vector<int> final_cluster_votes(point_number, 0);
	std::vector<int> final_cluster_idx(point_number, -1);

	for (auto r = 0; r < cluster_votes.size(); ++r)
	{
		for (auto point_idx = 0; point_idx < point_number; ++point_idx)
		{
			if (final_cluster_votes[point_idx] < cluster_votes[r].at<int>(point_idx))
			{
				final_cluster_votes[point_idx] = cluster_votes[r].at<int>(point_idx);
				final_cluster_idx[point_idx] = r;
			}
		}
	}

	if (cluster_votes.size() == 0)
	{
		return;
	}

	_clusters.create(static_cast<int>(cluster_votes.size()), dimension_number, data.type());
	cv::Mat const &clusters_ref = _clusters.getMatRef();

	_cluster_points.resize(cluster_votes.size());
	for (auto i = 0; i < final_cluster_idx.size(); ++i)
	{
		if (final_cluster_idx[i] == -1)
			continue;
		_cluster_points[final_cluster_idx[i]].push_back(i);
	}
	for (auto i = 0; i < cluster_centers.size(); ++i)
		cluster_centers[i].copyTo(clusters_ref.row(i));
}

template<typename T>
void mean_shift_clustering<T>::distance_pointwise_eucledian_2d(const cv::Mat const &_data,
	const cv::Mat &_old_mean,
	const int _point_number,
	const int _dimension_number,
	const float _sqr_band_width,
	cv::Mat &_current_cluster_votes,
	cv::Mat &_new_mean,
	int * const _been_visited_flag,
	std::vector<int> &_current_members,
	std::vector<int> &_inlier_indices)
{
	const T * data_ptr = reinterpret_cast<T *>(_data.data);
	const T * mean_ptr = reinterpret_cast<T *>(_old_mean.data);
	const int number_of_points_in_set = _dimension_number / 2;

	for (auto point_idx = 0; point_idx < _point_number; ++point_idx)
	{
		T x, y, dx, dy, distance = 0;
		for (auto dim = 0; dim < _dimension_number; dim += 2)
		{
			x = *(data_ptr++);
			y = *(data_ptr++);
			dx = x - *(mean_ptr + dim);
			dy = y - *(mean_ptr + dim + 1);
			if (current_pointwise_distance == pointwise_distance::max)
				distance = MAX(distance, dx * dx + dy * dy);
			else
				distance += sqrt(dx * dx + dy * dy);
		}

		if (current_pointwise_distance == pointwise_distance::mean)
		{
			distance = distance / number_of_points_in_set;
			distance = distance * distance;
		}

		if (distance < _sqr_band_width)
		{
			++_current_cluster_votes.at<int>(point_idx);
			_inlier_indices.push_back(point_idx);

			_new_mean = _new_mean + _data.row(point_idx);
			_been_visited_flag[point_idx] = 1;
			_current_members.push_back(point_idx);
		}
	}
}

template<typename T>
void mean_shift_clustering<T>::distance_eucledian(const cv::Mat const &_data,
	const cv::Mat &_old_mean,
	const int _point_number,
	const int _dimension_number,
	const float _sqr_band_width,
	cv::Mat &_current_cluster_votes,
	cv::Mat &_new_mean,
	int * const _been_visited_flag,
	std::vector<int> &_current_members,
	std::vector<int> &_inlier_indices)
{
	const T * data_ptr = reinterpret_cast<T *>(_data.data);
	const T * mean_ptr = reinterpret_cast<T *>(_old_mean.data);

	for (auto point_idx = 0; point_idx < _point_number; ++point_idx)
	{
		T value, difference, distance = 0;
		for (auto dim = 0; dim < _dimension_number; ++dim)
		{
			value = *(data_ptr++);
			difference = value - *(mean_ptr + dim);
			distance += difference * difference;
		}
		distance = sqrt(distance);

		if (distance < _sqr_band_width)
		{
			++_current_cluster_votes.at<int>(point_idx);
			_inlier_indices.push_back(point_idx);

			_new_mean = _new_mean + _data.row(point_idx);
			_been_visited_flag[point_idx] = 1;
			_current_members.push_back(point_idx);
		}
	}
}

template<typename T>
void mean_shift_clustering<T>::get_min_max_position(cv::Mat const &_data, 
	cv::Mat &_min_position, 
	cv::Mat &_max_position)
{
	_min_position = cv::Mat_<T>(1, _data.cols);
	_max_position = cv::Mat_<T>(1, _data.cols);
	const T * data_ptr = reinterpret_cast<T*>(_data.data);
	T * const min_position_ptr = reinterpret_cast<T*>(_min_position.data);
	T * const max_position_ptr = reinterpret_cast<T*>(_max_position.data);
	
	T val;
	for (auto i = 0; i < _data.rows; ++i)
	{
		for (auto dim = 0; dim < _data.cols; ++dim)
		{
			val = *(data_ptr++);

			if (i == 0 || *(min_position_ptr + dim) > val)
				*(min_position_ptr + dim) = _data.at<T>(i, dim);
			if (i == 0 || *(max_position_ptr + dim) < val)
				*(max_position_ptr + dim) = _data.at<T>(i, dim);
		}
	}
}


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
#include <map>
#include <set>
#include <algorithm>
#include <cmath>

#include "helpers.h"
#include "model.h"
#include "line_estimator.cpp"
#include "fundamental_estimator.cpp"
#include "homography_estimator.cpp"
#include "motion_estimator.cpp"

#include "uniform_sampler.h"
#include "napsac_sampler.h"

#include <opencv2/features2d/features2d.hpp>
#include <opencv2\xfeatures2d\nonfree.hpp>
#include "opencv2/opencv.hpp"

#include "MultiX.h"

// Apply homography fitting
void homography_fitting(std::string _data_source,
	std::string _source_img_1,
	std::string _source_img_2,
	std::string _scene_name,
	bool _use_annotation,
	int _problem_idx,
	float _inlier_outlier_threshold,
	float _spatial_coherence_weight,
	float _mode_seeking_threshold,
	float _sphere_radius,
	float _label_cost,
	int _sampler_idx,
	bool _drawing_flag);

// Apply fundamental matrix fitting
void fundamental_matrix_fitting(std::string _data_source,
	std::string _source_img_1,
	std::string _source_img_2,
	std::string _scene_name,
	bool _use_annotation,
	int _problem_idx,
	float _inlier_outlier_threshold,
	float _spatial_coherence_weight,
	float _mode_seeking_threshold,
	float _sphere_radius,
	float _label_cost,
	int _sampler_idx,
	bool _drawing_flag);

// Apply motion fitting
void motion_fitting(std::string _data_source,
	std::string _source_video_location,
	std::string _scene_name,
	bool _use_annotation,
	int _problem_idx,
	float _inlier_outlier_threshold,
	float _spatial_coherence_weight,
	float _mode_seeking_threshold,
	float _sphere_radius,
	float _label_cost,
	int _sampler_idx,
	bool _drawing_flag);

// Compute the misclassification error
float compute_misclassification_error(std::vector<int> const &labeling, int K1, std::vector<int> const &annotations, int K2, int pointNumber);

// Return the names of the built-in scenes
std::vector<std::string> get_built_in_scenes(const int _problem_idx);

// Returns the default parameterization
void get_default_parameterization(size_t _problem_idx,
	float &_inlier_outlier_threshold,
	float &_spatial_coherence_weight,
	float &_mode_seeking_threshold,
	float &_sphere_radius,
	float &_label_cost,
	int &_sampler_idx);

/*
	TODO list:
	- implement 2D line fitting
	- implement median-shift
	- implement adaptive parameter setting
	- replace homography and fundamental matrix estimator function by something faster than OpenCV's findFundamentalMat and findHomography functions
*/

int main(int argc, const char* argv[])
{
	srand(static_cast<unsigned int>(time(NULL)));

	bool use_built_in_test;
	size_t problem_idx;
	float inlier_outlier_threshold;
	float spatial_coherence_weight;
	float mode_seeking_threshold;
	float sphere_radius;
	float label_cost;
	int sampler_idx;
	bool use_annotations;
	std::string source_file;
	std::string scene_name;
	std::string video_location;
	std::string image_location_1;
	std::string image_location_2;
	bool drawing_flag;

	use_built_in_test = true; 
	use_annotations = true;
	drawing_flag = true;
	problem_idx = 2; // Motion (0), two-view motion (1), homography (2), 2d line (3; not implemented yet)
	std::vector<std::string> built_in_scenes = get_built_in_scenes(problem_idx);

	for each (std::string scene_name in built_in_scenes)
	{
		std::string problem_name;
		switch (problem_idx)
		{
		case 0:
			problem_name = "motion";
			printf("Fitting multiple motions to scene \"%s\".\n", scene_name.c_str());
			break;
		case 1:
			problem_name = "fundamental_matrix";
			printf("Fitting multiple fundamental matrices to scene \"%s\".\n", scene_name.c_str());
			break;
		case 2:
			problem_name = "homography";
			printf("Fitting multiple homographies to scene \"%s\".\n", scene_name.c_str());
			break;
		default:
			break;
		}

		get_default_parameterization(problem_idx,
			inlier_outlier_threshold,
			spatial_coherence_weight,
			mode_seeking_threshold,
			sphere_radius,
			label_cost,
			sampler_idx);

		printf("The used parameters are:\n");
		printf("\tInlier-outlier threshold = %.3f\n", inlier_outlier_threshold);
		printf("\tSpatial coherence weight = %.3f\n", spatial_coherence_weight);
		printf("\tHypersphere radius for the neighborhood calculation = %.3f\n", sphere_radius);
		printf("\tLabel cost = %.3f\n", label_cost);
		printf("\tMode-seeking threshold = %.3f\n", mode_seeking_threshold);
		printf("\tSampler = %s\n", sampler_idx == 1 ? "uniform sampler" : "NAPSAC sampler");

		source_file = "data/" + problem_name + "/" + scene_name + "/" + scene_name + ".txt";
		video_location = "data/" + problem_name + "/" + scene_name + "/" + scene_name + ".avi";
		image_location_1 = "data/" + problem_name + "/" + scene_name + "/" + scene_name + "1.png";
		image_location_2 = "data/" + problem_name + "/" + scene_name + "/" + scene_name + "2.png";

		switch (problem_idx)
		{
		case 0:
			motion_fitting(source_file,
				video_location,
				scene_name,
				use_annotations,
				problem_idx,
				inlier_outlier_threshold,
				spatial_coherence_weight,
				mode_seeking_threshold,
				sphere_radius,
				label_cost,
				sampler_idx,
				drawing_flag);
			break;
		case 1:
			fundamental_matrix_fitting(source_file,
				image_location_1,
				image_location_2,
				scene_name,
				use_annotations,
				problem_idx,
				inlier_outlier_threshold,
				spatial_coherence_weight,
				mode_seeking_threshold,
				sphere_radius,
				label_cost,
				sampler_idx,
				drawing_flag);
			break;
		case 2:
			homography_fitting(source_file,
				image_location_1,
				image_location_2,
				scene_name,
				use_annotations,
				problem_idx,
				inlier_outlier_threshold,
				spatial_coherence_weight,
				mode_seeking_threshold,
				sphere_radius,
				label_cost,
				sampler_idx,
				drawing_flag);
			break;
		case 3:
			printf("2D line fitting is not implemented yet.\n");
			break;
		default:
			break;
		}
	}

 	return 0;
}

void get_default_parameterization(size_t _problem_idx,
	float &_inlier_outlier_threshold,
	float &_spatial_coherence_weight,
	float &_mode_seeking_threshold,
	float &_sphere_radius,
	float &_label_cost,
	int &_sampler_idx)
{
	switch (_problem_idx)
	{
	case 0: // For motion fitting, the parameters are tuned minimizing the average misclassification error on the Hopkins155 dataset.
		_inlier_outlier_threshold = 0.014f;
		_spatial_coherence_weight = 0.2f; // TODO: try around 0.2
		_mode_seeking_threshold = 0.4f; // TODO: try around 0.4
		_sphere_radius = 0.05f;
		_label_cost = 10; // TODO: try around 10
		_sampler_idx = 2;
		break;
	case 1: // For two-view motion fitting, the parameters are tuned minimizing the average misclassification error on the AdelaideRMF motion dataset.
		_inlier_outlier_threshold = 2.2f;
		_spatial_coherence_weight = 0.1f;
		_mode_seeking_threshold = 0.011f;
		_sphere_radius = 50.0f;
		_label_cost = 16.0f;
		_sampler_idx = 2;
		break;
	case 2: // For homography fitting, the parameters are tuned minimizing the average misclassification error on the AdelaideRMF homography and on Multi-H datasets.
		_inlier_outlier_threshold = 3.8;
		_spatial_coherence_weight = 0.3;
		_mode_seeking_threshold = 0.5f;
		_sphere_radius = 50.0f;
		_label_cost = 8.0f;
		_sampler_idx = 2;
		break;
	case 3: // For line fitting TODO
		_inlier_outlier_threshold = 0;
		_spatial_coherence_weight = 0;
		_mode_seeking_threshold = 0;
		_sphere_radius = 0;
		_label_cost = 0;
		_sampler_idx = 1;
		break;
	default:
		break;
	}
}

std::vector<std::string> get_built_in_scenes(const int _problem_idx)
{
	switch (_problem_idx)
	{
	case 0: // Motion fitting
		return { "articulated", "cars10", "cars2" };
	case 1: // Two-view motion fitting
		return { "biscuitbookbox", "cubechips", "gamebiscuit" };
	case 2: // Homography fitting
		return { "johnssona", "sene", "oldclassicswing" };
	case 3: // 2D line fitting
		return {  };
	}
	return {};
}

void homography_fitting(std::string _data_source,
	std::string _source_img_1,
	std::string _source_img_2,
	std::string _scene_name,
	bool _use_annotation,
	int _problem_idx,
	float _inlier_outlier_threshold,
	float _spatial_coherence_weight,
	float _mode_seeking_threshold,
	float _sphere_radius,
	float _label_cost,
	int _sampler_idx,
	bool _drawing_flag)
{
	/*
		Load data
	*/
	cv::Mat img1 = cv::imread(_source_img_1);
	cv::Mat img2 = cv::imread(_source_img_2);

	if (img1.empty() || img2.empty())
	{
		printf("Error while loading the images.\n");
		return;
	}

	cv::Mat points;
	std::vector<int> reference_labeling;
	int reference_instance_number = 0;

	if (_use_annotation)
	{
		load_n_times_m_matrix(_data_source, points, &reference_labeling, -1, 7, false);
		reference_instance_number = *std::max_element(std::begin(reference_labeling), std::end(reference_labeling));
	}
	else
		load_n_times_m_matrix(_data_source, points, &reference_labeling, -1, 6, false);

	if (points.empty())
	{
		printf("Error while loading '%s'.\n", _data_source.c_str());
		return;
	}
	
	const int point_number = static_cast<int>(points.rows);
	const int initial_instance_number = 5 * point_number;

	/*
		Initialize the required estimators
	*/
	std::vector<multix::Estimator<cv::Mat, multix::Model> * > estimators;
	multix::HomographyEstimator estimator(img1.size());
	estimators.push_back(&estimator);

	/*
		Initialize the required samples
	*/
	std::vector<multix::Sampler<cv::Mat> *> samplers;

	std::vector<std::vector<cv::DMatch>> neighbors(points.rows);
	cv::FlannBasedMatcher flann;
	flann.knnMatch(points, points, neighbors, 2 * estimator.sample_size()); // Getting the neighborhood by FLANN

	multix::Sampler<cv::Mat> *sampler;
	if (_sampler_idx == 1)
	{
		sampler = new multix::UniformSampler<cv::Mat>();
		dynamic_cast<multix::UniformSampler<cv::Mat>*>(sampler)->initialize(point_number);
		samplers.push_back(sampler);
	}
	else
	{
		sampler = new multix::NapsacSampler<cv::Mat>();
		dynamic_cast<multix::NapsacSampler<cv::Mat>*>(sampler)->initialize(point_number, &neighbors);
		samplers.push_back(sampler);
	}

	/*
		Run Multi-X
	*/
	multix::MultiX * method = new multix::MultiX;
	method->settings.neighbor_sphere_radius = _sphere_radius;
	method->settings.spatial_coherence_weight = _spatial_coherence_weight;
	method->settings.mode_seeking_band_width = _mode_seeking_threshold;
	method->settings.label_complexity_weight = _label_cost;
	method->settings.inlier_outlier_threshold = _inlier_outlier_threshold;
	method->settings.instance_number = initial_instance_number;
	method->settings.log = true;
	method->run(estimators, samplers, &points);

	/*
		Compute the missclassification error if needed
	*/
	if (_use_annotation)
	{
		std::vector<int> labeling(point_number);
		for (auto j = 0; j < point_number; ++j)
			labeling[j] = method->get_label(j);
		float misclassification_error = compute_misclassification_error(labeling,
			method->get_instance_number(),
			reference_labeling,
			reference_instance_number,
			point_number);

		printf("Misclassification error = %.2f%%\n", misclassification_error);
	}

	/*
		Draw the results
	*/
	if (_drawing_flag)
	{
		cv::RNG rng(12345);
		for (auto i = 0; i < method->get_instance_number(); ++i)
		{
			cv::Scalar color(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
			for (auto j = 0; j < point_number; ++j)
			{
				if (method->get_label(j) == 0)
				{
					cv::circle(img1, cv::Point2f(points.at<float>(j, 0), points.at<float>(j, 1)), 4, cv::Scalar(0, 0, 0), -1);
					cv::circle(img2, cv::Point2f(points.at<float>(j, 3), points.at<float>(j, 4)), 4, cv::Scalar(0, 0, 0), -1);
				}
				else if (method->get_label(j) == i + 1)
				{
					cv::circle(img1, cv::Point2f(points.at<float>(j, 0), points.at<float>(j, 1)), 4, color, -1);
					cv::circle(img2, cv::Point2f(points.at<float>(j, 3), points.at<float>(j, 4)), 4, color, -1);
				}
			}
		}

		cv::imshow("Image 1", img1);
		cv::imshow("Image 2", img2);
		printf("Press a key to continue...\n");
		cv::waitKey(0);
	}

	delete method;
}

void fundamental_matrix_fitting(std::string _data_source,
	std::string _source_img_1,
	std::string _source_img_2,
	std::string _scene_name,
	bool _use_annotation,
	int _problem_idx,
	float _inlier_outlier_threshold,
	float _spatial_coherence_weight,
	float _mode_seeking_threshold,
	float _sphere_radius,
	float _label_cost,
	int _sampler_idx,
	bool _drawing_flag)
{
	/*
		Load data
	*/
	// Load images
	cv::Mat img1 = cv::imread(_source_img_1);
	cv::Mat img2 = cv::imread(_source_img_2);

	if (img1.empty() || img2.empty())
	{
		printf("Error while loading the images.\n");
		return;
	}

	cv::Mat points;
	std::vector<int> reference_labeling;
	int reference_instance_number = 0;

	if (_use_annotation)
	{
		load_n_times_m_matrix(_data_source, points, &reference_labeling, -1, 7, false);
		reference_instance_number = *std::max_element(std::begin(reference_labeling), std::end(reference_labeling));
	}
	else
		load_n_times_m_matrix(_data_source, points, &reference_labeling, -1, 6, false);

	if (points.empty())
	{
		printf("Error while loading '%s'.\n", _data_source.c_str());
		return;
	}

	const int point_number = static_cast<int>(points.rows);
	const int initial_instance_number = 5 * point_number;

	/*
		Initialize the required estimators
	*/
	std::vector<multix::Estimator<cv::Mat, multix::Model> * > estimators;
	multix::FundamentalMatrixEstimator estimator;
	estimators.push_back(&estimator);

	/*
		Initialize the required samples
	*/
	std::vector<multix::Sampler<cv::Mat> *> samplers;

	std::vector<std::vector<cv::DMatch>> neighbors(points.rows);
	cv::FlannBasedMatcher flann;
	flann.knnMatch(points, points, neighbors, 2 * estimator.sample_size()); // Getting the neighborhood by FLANN

	// Initializing NAPSAC sampler
	multix::Sampler<cv::Mat> *sampler;
	if (_sampler_idx == 1)
	{
		sampler = new multix::UniformSampler<cv::Mat>();
		dynamic_cast< multix::UniformSampler<cv::Mat>* >(sampler)->initialize(point_number);
		samplers.push_back(sampler);
	}
	else
	{
		sampler = new multix::NapsacSampler<cv::Mat>();
		dynamic_cast< multix::NapsacSampler<cv::Mat>* >(sampler)->initialize(point_number, &neighbors);
		samplers.push_back(sampler);
	}

	/*
		Run Multi-X
	*/
	multix::MultiX * method = new multix::MultiX;
	method->settings.neighbor_sphere_radius = _sphere_radius;
	method->settings.spatial_coherence_weight = _spatial_coherence_weight;
	method->settings.mode_seeking_band_width = _mode_seeking_threshold;
	method->settings.label_complexity_weight = _label_cost;
	method->settings.inlier_outlier_threshold = _inlier_outlier_threshold;
	method->settings.instance_number = initial_instance_number;
	method->settings.log = true;
	method->run(estimators, samplers, &points);

	/*
		Compute the missclassification error if needed
	*/
	if (_use_annotation)  
	{
		std::vector<int> labeling(point_number);
		for (auto j = 0; j < point_number; ++j)
			labeling[j] = method->get_label(j);
		float misclassification_error = compute_misclassification_error(labeling,
			method->get_instance_number(),
			reference_labeling,
			reference_instance_number,
			point_number);

		printf("Misclassification error = %.2f%%\n", misclassification_error);
	}

	/*
		Draw the results
	*/
	if (_drawing_flag)
	{
		cv::RNG rng(12345);
		for (auto i = 0; i < method->get_instance_number(); ++i)
		{
			cv::Scalar color(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
			for (auto j = 0; j < point_number; ++j)
			{
				if (method->get_label(j) == 0)
				{
					cv::circle(img1, cv::Point2f(points.at<float>(j, 0), points.at<float>(j, 1)), 4, cv::Scalar(0, 0, 0), -1);
					cv::circle(img2, cv::Point2f(points.at<float>(j, 3), points.at<float>(j, 4)), 4, cv::Scalar(0, 0, 0), -1);
				}
				else if (method->get_label(j) == i + 1)
				{
					cv::circle(img1, cv::Point2f(points.at<float>(j, 0), points.at<float>(j, 1)), 4, color, -1);
					cv::circle(img2, cv::Point2f(points.at<float>(j, 3), points.at<float>(j, 4)), 4, color, -1);
				}
			}
		}

		cv::imshow("Image 1", img1);
		cv::imshow("Image 2", img2);
		printf("Press a key to continue...\n");
		cv::waitKey(0);
	}

	delete method;
}

void motion_fitting(std::string _data_source,
	std::string _source_video_location,
	std::string _scene_name,
	bool _use_annotation,
	int _problem_idx,
	float _inlier_outlier_threshold,
	float _spatial_coherence_weight,
	float _mode_seeking_threshold,
	float _sphere_radius,
	float _label_cost,
	int _sampler_idx,
	bool _drawing_flag)
{
	/*
		Load data
	*/
	cv::Mat points, projected_points;
	std::vector<int> reference_labeling;
	int reference_instance_number = 0;

	if (_use_annotation)
	{
		load_n_times_m_matrix(_data_source, points, &reference_labeling);
		reference_instance_number = *std::max_element(std::begin(reference_labeling), std::end(reference_labeling));
	}
	else
		load_n_times_m_matrix(_data_source, points, nullptr);

	const int point_number = points.rows;
	const int initial_instance_number = 5 * point_number;

	project_points_to_r_dimensional_space(points, projected_points, 5);

	/*
		Initialize the required estimators
	*/
	std::vector<multix::Estimator<cv::Mat, multix::Model> * > estimators;
	multix::MotionEstimator estimator;
	estimators.push_back(&estimator);

	/*
		Initialize the required samples
	*/
	std::vector<multix::Sampler<cv::Mat> *> samplers;

	std::vector<std::vector<cv::DMatch>> neighbors(points.rows);
	cv::FlannBasedMatcher flann;
	flann.knnMatch(points, points, neighbors, 2 * estimator.sample_size()); // Getting the neighborhood by FLANN

	multix::Sampler<cv::Mat> *sampler;
	if (_sampler_idx == 1)
	{
		sampler = new multix::UniformSampler<cv::Mat>();
		dynamic_cast<multix::UniformSampler<cv::Mat>*>(sampler)->initialize(point_number);
		samplers.push_back(sampler);
	}
	else
	{
		sampler = new multix::NapsacSampler<cv::Mat>();
		dynamic_cast<multix::NapsacSampler<cv::Mat>*>(sampler)->initialize(point_number, &neighbors);
		samplers.push_back(sampler);
	}

	/*
		Run Multi-X
	*/
	multix::MultiX * method = new multix::MultiX;
	method->settings.neighbor_sphere_radius = _sphere_radius;
	method->settings.spatial_coherence_weight = _spatial_coherence_weight;
	method->settings.mode_seeking_band_width = _mode_seeking_threshold;
	method->settings.label_complexity_weight = _label_cost;
	method->settings.inlier_outlier_threshold = _inlier_outlier_threshold;
	method->settings.instance_number = initial_instance_number;
	method->settings.log = true;
	method->run(estimators, samplers, &projected_points);

	if (_use_annotation) // Compute the missclassification error if needed
	{
		std::vector<int> labeling(point_number);
		for (auto j = 0; j < point_number; ++j)
			labeling[j] = method->get_label(j);
		float misclassification_error = compute_misclassification_error(labeling,
			method->get_instance_number(),
			reference_labeling,
			reference_instance_number,
			point_number);
		printf("Misclassification error = %.2f%%\n", misclassification_error);
	}

	if (_drawing_flag)
	{
		cv::VideoCapture video(_source_video_location);
		
		if (video.isOpened())
		{
			const auto instance_number = method->get_instance_number() + 1;
			std::unique_ptr<cv::Scalar[]> colors = std::make_unique<cv::Scalar[]>(instance_number);
			cv::RNG rng(12345);
			colors[0] = cv::Scalar(0, 0, 0);
			for (auto color_idx = 0; color_idx < instance_number; ++color_idx)
				colors[color_idx + 1] = cv::Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));

			cv::Mat frame;
			int frame_idx = 0;
			while (1) {
				// Capture frame-by-frame
				video >> frame;
				
				// If the frame is empty, break immediately
				if (frame.empty())
					break;
				
				for (auto point_idx = 0; point_idx < points.rows; ++point_idx)
				{
					const int label = method->get_label(point_idx);

					if (_scene_name == "cars2" || _scene_name == "cars10")
						cv::circle(frame,
							cv::Point2f(points.at<float>(point_idx, 2 * frame_idx), frame.rows - points.at<float>(point_idx, 2 * frame_idx + 1)),
							4, colors[label], -1);
					else
						cv::circle(frame,
							cv::Point2f(points.at<float>(point_idx, 2 * frame_idx), points.at<float>(point_idx, 2 * frame_idx + 1)),
							4, colors[label], -1);
				}
				++frame_idx;

				// Display the resulting frame
				imshow("Frame", frame);

				// Press  ESC on keyboard to exit
				char c = (char)cv::waitKey(100);
				if (c == 27)
					break;
			}

			printf("Press a key to continue...\n");
			cv::waitKey(0);
			frame.release();
		}
		else
			printf("Video file \"%s\" cannot be openned.\n", _source_video_location.c_str());
	}

	delete method;
}

float compute_misclassification_error(std::vector<int> const &_labeling, 
	int _instance_number_1, 
	std::vector<int> const &_annotations, 
	int _instance_number_2, 
	int _point_number)
{
	if (_instance_number_1 == 0)
		return 100;

	std::vector<std::set<int>> obtained_clusters(_instance_number_1 + 1);
	std::vector<std::set<int>> ground_truth_clusters(_instance_number_2 + 1);

	std::vector<int> ground_truth_pairs(_instance_number_2 + 1, 0);

	std::vector<int> all_labels_1(_point_number, -1);
	std::vector<int> all_labels_2(_point_number, -1);

	for (auto i = 0; i < _point_number; ++i)
	{
		if (_labeling[i] + 1 >= obtained_clusters.size())
		{
			obtained_clusters.resize(_labeling[i] + 2);
			_instance_number_1 = static_cast<int>(obtained_clusters.size()) - 1;
		}

		all_labels_1[i] = _labeling[i] + 1;
		obtained_clusters[_labeling[i] + 1].insert(i);
	}
	
	for (auto i = 0; i < _point_number; ++i)
	{
		if (_annotations[i] + 1 >= ground_truth_clusters.size())
		{
			ground_truth_clusters.resize(_annotations[i] + 2);
			_instance_number_2 = static_cast<int>(ground_truth_clusters.size()) - 1;
		}

		all_labels_2[i] = _annotations[i] + 1;
		ground_truth_clusters[_annotations[i] + 1].insert(i);
	}

	//return 0;

	// Find pairs
	std::vector<bool> usability_mask(obtained_clusters.size(), false);
	std::vector<int> cluster_to_cluster(ground_truth_clusters.size(), -1);
	for (auto i = 1; i < ground_truth_clusters.size(); ++i)
	{
		int best_cluster = -1;
		int best_size = -1;

		for (auto j = 1; j < obtained_clusters.size(); ++j)
		{
			if (usability_mask[j]) 
				continue;

			std::set<int> intersect;
			std::set_intersection(obtained_clusters[j].begin(), obtained_clusters[j].end(), ground_truth_clusters[i].begin(), ground_truth_clusters[i].end(),
				std::inserter(intersect, intersect.begin()));

			if (best_cluster == -1 || best_size < intersect.size())
			{
				best_size = static_cast<int>(intersect.size());
				best_cluster = j;
			}
		}

		if (best_cluster == -1)
			continue;

		cluster_to_cluster[i] = best_cluster;
		usability_mask[best_cluster] = true;
		ground_truth_pairs[i] = best_cluster;
	}

	for (auto j = 0; j < _labeling.size(); ++j)
	{
		if (cluster_to_cluster[all_labels_2[j]] == -1)
			continue;

		all_labels_2[j] = cluster_to_cluster[all_labels_2[j]];
	}

	int error = 0;
	for (auto i = 0; i < all_labels_1.size(); ++i)
	{
		//cout << labelsAll1[i] << " " << labelsAll2[i] << endl;
		if (all_labels_1[i] != all_labels_2[i])
		{
			++error;
		}
	}

	return 100.0f * (error / (float)all_labels_1.size());
}
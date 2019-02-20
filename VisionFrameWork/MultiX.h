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

#include <opencv\cv.hpp>
#include <cv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <functional>
#include <algorithm>
#include <unordered_map>

#include "moduls/models/model.h"
#include "moduls/models/line_estimator.cpp"
#include "moduls/models/fundamental_estimator.cpp"
#include "moduls/models/homography_estimator.cpp"

#include "moduls/sampler/uniform_sampler.h"
#include "moduls/mode_seeking/mean_shift.h"

#include "moduls/gcoptimization/GCoptimization.h"

namespace multix
{
	#define ENERGY_PRECISION 1

	struct Setting
	{
		int max_thread_number;
		float inlier_outlier_threshold;
		float neighbor_sphere_radius;
		int instance_number;
		float spatial_coherence_weight;
		float mode_seeking_band_width;
		float label_complexity_weight;
		int minimum_point_number;
		int max_iterations;
		int max_iterations_for_mode_seeking;
		int max_iterations_without_change;
		bool apply_mode_seeking;
		bool remove_one_element_clusters;
		bool log;
	};

	struct EnergyDataStructure
	{
		const cv::Mat * const points;
		const std::vector<multix::Model> * const instances;
		const std::vector<multix::Estimator<cv::Mat, multix::Model> * > * const estimators;
		const float energy_lambda;
		const float inlier_outlier_threshold;

		EnergyDataStructure(const cv::Mat * const _points,
			const std::vector<multix::Model> * const _instances,
			const std::vector<multix::Estimator<cv::Mat, multix::Model> * > * const _estimators,
			const float _lambda,
			const float _threshold) :
			points(_points),
			instances(_instances),
			estimators(_estimators),
			energy_lambda(_lambda),
			inlier_outlier_threshold(_threshold)
		{
		}
	};

	inline float spatial_coherence_energy(int _p1, int _p2, int _l1, int _l2, void *_data)
	{
		EnergyDataStructure *myData = (EnergyDataStructure *)_data;

		const float spatial_coherence_weight = myData->energy_lambda;
		return _l1 != _l2 ? ENERGY_PRECISION * spatial_coherence_weight : 0;
	}

	inline float data_energy(int _p, int _l, void *_data)
	{
		EnergyDataStructure *myData = (EnergyDataStructure *)_data;

		const float spatial_coherence_weight = myData->energy_lambda == 0 ? 1 : myData->energy_lambda;
		const float inlier_outlier_threshold = myData->inlier_outlier_threshold;
		const float sqr_threshold = inlier_outlier_threshold * inlier_outlier_threshold;
		const float truncated_threshold = sqr_threshold * 9 / 4;

		if (_l == 0)
			return ENERGY_PRECISION; // The outlier class's cost is 1.

		const multix::Model instance = myData->instances->at(_l - 1);
		const cv::Mat point = myData->points->row(_p);
		float distance = instance.estimator->error(point, instance);

		distance = distance * distance;

		if (distance > truncated_threshold)
			return ENERGY_PRECISION * 2.0f; // The cost of assigning a point to an instance if the distance is higher than the threshold is higher than 1, instance_idx.e. the cost of assigning to the outlier class 
		return ENERGY_PRECISION * distance / truncated_threshold; // The cost of assigning a point to an instance is in-between 0 and 1 determined by the truncated squared error.
	}

	class MultiX
	{
	protected:
		std::vector<multix::Model> instances; // Vector of model instances
		std::vector<std::vector<int>> instances_per_estimator; // The indices of the model instances enlisted for each estimator, instance_idx.e. class, separately

		std::unique_ptr<GCoptimizationGeneralGraph> gc_optimization; // Alpha-expansion framework
		std::vector<std::vector<cv::DMatch>> neighbors; // The neighborhood structure
		std::vector<std::vector<int>> points_per_instance; // The indices of the points assigned to a model
		std::vector<int> outliers; // The indices of points not assigned to any instances

		float last_energy, prev_last_energy; // Previous energies 

	public:
		Setting settings;

		// The constructor of the Multi-X class
		MultiX() : prev_last_energy(-2),
			last_energy(-1)
		{
			settings.max_thread_number = INT_MAX;
			settings.inlier_outlier_threshold = 3.1f;
			settings.max_iterations = 10;
			settings.neighbor_sphere_radius = 10.0;
			settings.label_complexity_weight = 20.0f;
			settings.instance_number = 5000;
			settings.spatial_coherence_weight = 0.31f;
			settings.minimum_point_number = 1;
			settings.max_iterations_for_mode_seeking = 1000;
			settings.mode_seeking_band_width = 2.0f;
			settings.apply_mode_seeking = true;
			settings.remove_one_element_clusters = false;
			settings.max_iterations_without_change = 5;
			settings.log = false;
		}

		~MultiX()
		{
		}

		int get_instance_number() { return static_cast<int>(instances.size()); } // Returns the number of instances
		const multix::Model const * get_instance(const int _idx) { return &instances[_idx]; } // Return the pointer of an instance
		int get_label(const int _point_idx) { return gc_optimization == NULL ? 0 : gc_optimization->whatLabel(_point_idx); } // Return the label of a point

		void log(std::string _message, 
			bool _plain_text = false)
		{
			if (settings.log)
			{
				if (_plain_text)
					printf("%s", _message.c_str());
				else
					printf("[Multi-X] %s.\n", _message.c_str());
			}
		}

		// Start the Multi-X algorithm
		void run(std::vector<multix::Estimator<cv::Mat, multix::Model> * > _estimators,
			std::vector<multix::Sampler<cv::Mat> *> _samplers,
			const cv::Mat * _points)
		{
			std::chrono::time_point<std::chrono::system_clock> start, end;

			// Initializing variables
			if (settings.log)
				start = std::chrono::system_clock::now();
			initialization(_points, &_estimators);
			if (settings.log)
			{
				end = std::chrono::system_clock::now();
				std::chrono::duration<double> elapsed_seconds = end - start;
				log("Neighborhood calculation took " + std::to_string(elapsed_seconds.count()) + " secs");
			}

			// Generating initial instances
			if (settings.log)
				start = std::chrono::system_clock::now();
			instance_generation(&_estimators, &_samplers, _points);
			if (settings.log)
			{
				end = std::chrono::system_clock::now();
				std::chrono::duration<double> elapsed_seconds = end - start;
				log("Instance generation took " + std::to_string(elapsed_seconds.count()) + " secs");
			}

			// Alternating optimization
			if (settings.log)
			{
				start = std::chrono::system_clock::now();

				log("Alternating Optimization started..");
				log("\n", true);
			}
			alternating_optimization(&_estimators, &_samplers, _points);
			if (settings.log)
			{
				end = std::chrono::system_clock::now();
				std::chrono::duration<double> elapsed_seconds = end - start;

				log("Alternating Optimization finished");
				log("Optimization took " + std::to_string(elapsed_seconds.count()) + " secs");
				log("\n", true);
			}
		}

		void initialization(const cv::Mat * _points,
			std::vector<multix::Estimator<cv::Mat, multix::Model> * > *_estimators)
		{
			// Determine neighborhoods if needed
			if (settings.spatial_coherence_weight > 0)
			{
				neighbors.resize(_points->rows);
				cv::FlannBasedMatcher flann;
				flann.radiusMatch(*_points,
					*_points,
					neighbors,
					settings.neighbor_sphere_radius);
			}
		}

		void alternating_optimization(std::vector<multix::Estimator<cv::Mat, multix::Model> * > *_estimators,
			std::vector<multix::Sampler<cv::Mat> *> *_samplers,
			const cv::Mat * _points)
		{
			float energy; // The energy
			int iteration_number = 0; // The number of current iterations
			bool convergenve = false; // A flag for the convergence
			bool changed = false; // Is something changed in two iterations
			int non_changed_number = 0; // The number of not-changing in a row of iterations

			while (!convergenve && iteration_number <= settings.max_iterations) // The main iteration of alternating optimization
			{
				log("Iteration number #" + std::to_string(iteration_number + 1));
				log("Instance # before mode-seeking = " + std::to_string(instances.size()));

				// Mode-seeking
				if (settings.apply_mode_seeking)
					apply_mode_seeking(_estimators, _points, iteration_number, changed);
				log("Instance # after mode-seeking = " + std::to_string(instances.size()));

				if (changed == false)
					++non_changed_number;
				else
					non_changed_number = 0;

				// Labeling
				apply_labeling(_estimators, _samplers, _points, changed, energy);
				log("Energy of alpha-expansion = " + std::to_string(energy));

				// Remove outliers
				assign_points_to_instance(_estimators, _points);

				if (instances.size() == 0)
					break;

				// Model Re-Estimation
				parameter_reestimation(_estimators, _points);

				// Model Validation
				changed = false;
				instance_validation(_estimators, _points, changed);

				++iteration_number;

				convergenve = !changed && (instances.size() == 0 || 
					fabs(energy - last_energy) < FLT_EPSILON || 
					fabs(energy - prev_last_energy) < FLT_EPSILON || 
					non_changed_number > settings.max_iterations_without_change);
				prev_last_energy = last_energy;
				last_energy = energy;

				log("\n", true);

				if (instances.size() == 0)
					break;
			}
		}

		// Put each point's index to the appropriate container depending on the label which it got
		void assign_points_to_instance(std::vector<multix::Estimator<cv::Mat, multix::Model> * > *_estimators,
			const cv::Mat * _points)
		{
			if (points_per_instance.size() > 0) // If not empty then make it empty
			{
				outliers.clear();
				outliers.resize(0);
				points_per_instance.clear();
				points_per_instance.resize(0);
			}
			points_per_instance.resize(instances.size(), std::vector<int>());
			
			if (instances.size() == 0)
			{
				for (auto point_idx = 0; point_idx < _points->rows; ++point_idx)
					outliers.push_back(point_idx);
				return;
			}

			for (auto instance_idx = 0; instance_idx < instances.size(); ++instance_idx)
				instances[instance_idx].inliers.resize(0);

			// Put each point's index to the appropriate container depending on the label which it got
			GCoptimization::LabelID label;
			for (auto point_idx = 0; point_idx < _points->rows; ++point_idx)
			{
				label = gc_optimization->whatLabel(point_idx);
				if (label == 0)
					outliers.push_back(point_idx);
				else
				{
					points_per_instance[label - 1].push_back(point_idx);
					instances[label - 1].inliers.push_back(point_idx);
				}
			}
		}

		void instance_validation(std::vector<multix::Estimator<cv::Mat, multix::Model> * > *_estimators,
			const cv::Mat * _points,
			bool &_changed)
		{
			// Remove instances having not enough inlier_number
			for (auto i = static_cast<int>(instances.size()) - 1; i >= 0; --i)
			{
				if (points_per_instance[i].size() < settings.minimum_point_number ||
					!instances[i].estimator->is_valid(_points, instances[i].descriptor, &points_per_instance[i]))
				{
					instances.erase(instances.begin() + i);
					_changed = true;
				}
			}

			// Assign the instances to their estimators
			for (auto i = 0; i < _estimators->size(); ++i)
				instances_per_estimator[i].resize(0);

			for (auto i = 0; i < instances.size(); ++i)
			{
				for (auto j = 0; j < _estimators->size(); ++j)
				{
					if (instances[i].estimator == _estimators->at(j))
					{
						instances_per_estimator[j].push_back(i);
						break;
					}
				}
			}
		}

		void parameter_reestimation(std::vector<multix::Estimator<cv::Mat, multix::Model> * > *_estimators,
			const cv::Mat * _points)
		{
			const int workers = MIN(settings.max_thread_number, MIN(static_cast<int>(instances.size()), std::thread::hardware_concurrency()));
			std::vector<std::thread> threads;

			static auto parallel_code = [](const int _process_idx,
				const int _core_number,
				const cv::Mat *_points,
				std::vector<multix::Model> * const _instances,
				const std::vector<std::vector<int>> * const _points_per_instance)
			{
				const int instance_number = static_cast<int>(_instances->size());
				const int jobs_per_thread = instance_number / _core_number; // Number of jobs for the current thread
				const int start_idx = _process_idx * jobs_per_thread; // First job index
				const int end_idx = _process_idx == _core_number - 1 ? instance_number : (_process_idx + 1) * jobs_per_thread; // Last job index

				for (auto instance_idx = start_idx; instance_idx < end_idx; ++instance_idx)
				{
					const std::vector<int> * const current_inliers = &_points_per_instance->at(instance_idx);
					multix::Model * const current_instance = &_instances->at(instance_idx);
					const Estimator<cv::Mat, Model> * estimator = current_instance->estimator;

					if (current_inliers->size() < estimator->sample_size()) // Model having fewer inlier_number than the minimum sample, can't be re-estimated
						continue;

					const int N = static_cast<int>(current_inliers->size());
					const int * const sample = &(*current_inliers)[0];

					std::vector<multix::Model> models;
					if (!estimator->estimate_model_nonminimal(_points,
						sample,
						N,
						&models))
						continue;

					// TODO: handle cases when more models are generated
					if (models.size() > 1)
						continue;

					estimator->set_descriptor(*current_instance,
						models[0].descriptor);
				}
			};

			if (workers == 1)
			{
				parallel_code(0,
					workers,
					_points,
					&instances,
					&points_per_instance);
			}
			else
			{
				for (int i = 0; i < workers; ++i)
				{
					threads.push_back(std::thread(parallel_code,
						i,
						workers,
						_points,
						&instances,
						&points_per_instance));
				}

				for (std::thread& t : threads)
					t.join();
			}
		}

		void apply_labeling(std::vector<multix::Estimator<cv::Mat, multix::Model> * > * const _estimators,
			std::vector<multix::Sampler<cv::Mat> *> *_samplers,
			const cv::Mat * const _points,
			bool _changed,
			float &_energy)
		{
			// Set previous labeling if not changed
			std::vector<int> previous_labels;
			if (!_changed && gc_optimization != NULL)
			{
				previous_labels.resize(instances.size());
				for (auto i = 0; i < instances.size(); ++i)
					previous_labels[i] = gc_optimization->whatLabel(i);
			}

			if (instances.size() == 0)
				return;

			if (gc_optimization != NULL)
				gc_optimization.reset();
			
			gc_optimization = std::make_unique<GCoptimizationGeneralGraph>(_points->rows, 
				static_cast<GCoptimization::LabelID>(instances.size() + 1));
			
			EnergyDataStructure energy_data_container(_points, &instances, _estimators, settings.spatial_coherence_weight, settings.inlier_outlier_threshold);
			gc_optimization->setDataCost(&data_energy, &energy_data_container);
			if (settings.spatial_coherence_weight > 0.0)
				gc_optimization->setSmoothCost(&spatial_coherence_energy, &energy_data_container);
			
			if (settings.label_complexity_weight > 0.0)
			{
				if (_estimators->size() == 1)
					gc_optimization->setLabelCost(static_cast<float>(settings.label_complexity_weight * ENERGY_PRECISION));
				else
				{
					float *costs = new float[instances.size() + 1];
					costs[0] = 0;
					for (auto i = 0; i < instances.size(); ++i)
						costs[i + 1] = static_cast<float>(round(settings.label_complexity_weight * instances[i].estimator->model_weight()));
					gc_optimization->setLabelCost(costs);
					delete[] costs;
				}
			}

			// Set neighbourhood
			if (settings.spatial_coherence_weight > 0.0)
				for (auto i = 0; i < neighbors.size(); ++i)
					for (auto j = 0; j < neighbors[i].size(); ++j)
					{
						int idx = neighbors[i][j].trainIdx;
						if (idx != i)
							gc_optimization->setNeighbors(i, idx);
					}

			if (!_changed && previous_labels.size() > 0)
			{
				for (auto i = 0; i < instances.size(); ++i)
					gc_optimization->setLabel(i, previous_labels[i]);
				previous_labels.clear();
			}

			int iteration_number;
			_energy = static_cast<float>(gc_optimization->expansion(iteration_number, 1000));
		}

		void apply_mode_seeking(std::vector<multix::Estimator<cv::Mat, multix::Model> * > * const _estimators,
			const cv::Mat * _points,
			const int _iteration_number,
			bool &_changed)
		{
			if (instances.size() <= 1)
				return;

			std::vector<multix::Model> temp_instances; // Temporary instances
			temp_instances.reserve(instances.size());
			std::vector<std::vector<int>> temp_inst_per_estimator(_estimators->size());

			// Apply mode-seeking to all classes independently
			for (auto est_idx = 0; est_idx < _estimators->size(); ++est_idx)
			{
				const std::vector<int> * const current_instances = &instances_per_estimator[est_idx];
				const int N = static_cast<int>(current_instances->size());
				const multix::Estimator<cv::Mat, multix::Model> * const estimator = _estimators->at(est_idx);

				if (!estimator->is_mode_seeking_applicable()) // Just copy instances for classes for which mode-seeking is not applicable
				{
					temp_inst_per_estimator[est_idx].reserve(N);
					for (auto i = 0; i < N; ++i)
					{
						temp_inst_per_estimator[est_idx].push_back(static_cast<int>(temp_instances.size()));
						temp_instances.push_back(instances[current_instances->at(i)]);
					}
					continue;
				}

				if (N <= 1) // In case of having less than two instances
				{

					if (N == 1) // Having one instance
					{
						int idx = current_instances->at(0);
											   
						multix::Model model;
						model.estimator = instances[idx].estimator;
						model.estimator->set_descriptor(model, instances[idx].descriptor);

						if (!estimator->is_valid(NULL, model.get_descriptor(), NULL))
							continue;

						temp_inst_per_estimator[est_idx].push_back(static_cast<int>(temp_instances.size()));
						temp_instances.push_back(model);
					}
					continue;
				}

				// Build concatenated instance matrix (each row represents a model instance)
				const int first_idx = current_instances->at(0); // Index of the first instance
				const int elements = instances[first_idx].descriptor_by_points.size().area(); // Number of elements in the descriptor
				const int rows = instances[first_idx].descriptor_by_points.rows; // Number of rows in the descriptor
				const int cols = instances[first_idx].descriptor_by_points.cols; // Number of columns in the descriptor

				cv::Mat instance_descriptors(N, elements, CV_32F); // Instance matrix
				float *descriptors_ptr = reinterpret_cast<float *>(instance_descriptors.data);

				// Copying data to the instance matrix
				int idx;
				const float *tmp_descriptor_ptr;
				for (auto i = 0; i < N; ++i)
				{
					idx = current_instances->at(i);
					tmp_descriptor_ptr = reinterpret_cast<float *>(instances[idx].descriptor_by_points.data);
					for (auto element = 0; element < elements; ++element)
						*(descriptors_ptr++) = *(tmp_descriptor_ptr++);
				}

				// Apply mode-seeking
				const bool first_iteration = _iteration_number == 0;

				{
					mean_shift_clustering<float> mean_shift;
					if (estimator->is_represented_by_points())
						mean_shift.set_distance_metric(mean_shift_clustering<float>::distance_metric::pointwise_eucledian_2d);
					else
						mean_shift.set_distance_metric(mean_shift_clustering<float>::distance_metric::eucledian);
					cv::Mat clusters; // Clusters
					std::vector<std::vector<int>> cluster_points; // Indices of the corresponding instances for each cluster 
					mean_shift.cluster(instance_descriptors, settings.mode_seeking_band_width, clusters, cluster_points); // Mean-shift to the instances

					if (clusters.empty()) // There was some error
					{
						for (auto i = 0; i < N; ++i)
						{
							temp_inst_per_estimator[est_idx].push_back(static_cast<int>(temp_instances.size()));
							temp_instances.push_back(instances[current_instances->at(i)]);
						}
						continue;
					}

					// Replace instances by the corresponding modes
					for (auto cluster_idx = 0; cluster_idx < clusters.rows; ++cluster_idx)
					{
						const auto N = static_cast<int>(cluster_points.at(cluster_idx).size()); // Number of instances in the cluster
						if (N < 2) // Continue if there are fewer than 2 instances in the cluster
						{
							if (((!settings.remove_one_element_clusters && first_iteration) || 
								!first_iteration) && 
								N == 1) // In all but the first iteration, do not remove clusters with only one instance in them
							{
								const int tmp_instance_idx = cluster_points.at(cluster_idx)[0];
								temp_inst_per_estimator[est_idx].push_back(static_cast<int>(temp_instances.size()));
								temp_instances.push_back(instances[tmp_instance_idx]);
							}
							continue;
						}

						if (estimator->is_non_minimal_fitting_applicable()) // Apply non-minimal fitting to the clusters' points' if applicable
						{
							/*
								Least squares fitting to the inlier_number of all instances in the cluster
							*/
							// Count all the inlier_number corresponding to an instance in the cluster
							const std::vector<int> *cluster_point_indices = &cluster_points.at(cluster_idx);
							std::unordered_map<int, bool> unique_indices_map;
							int idx, point_idx;
							for (auto instance_idx = 0; instance_idx < N; ++instance_idx) // Put the indices into a hashmap to have only unique indices
							{
								idx = current_instances->at(cluster_point_indices->at(instance_idx));
								for (point_idx = 0; point_idx < instances[idx].inliers.size(); ++point_idx)
									unique_indices_map[instances[idx].inliers[point_idx]] = true;
							}
							const int Ni = static_cast<int>(unique_indices_map.size()); // The number of unique indices
							
							if (Ni < estimator->sample_size())
								continue;

							int *temp_sample = new int[Ni]; // Temporary sample
							int sample_idx = 0;
							for (auto key : unique_indices_map) // Construct the sample
								temp_sample[sample_idx++] = key.first;

							// Least squares fitting
							std::vector<multix::Model> models;
							_estimators->at(est_idx)->estimate_model_nonminimal(_points, temp_sample, Ni, &models);
							delete temp_sample;

							// Copy the estimated model instances to the output instance container
							for (auto i = 0; i < models.size(); ++i)
							{
								if (models[i].descriptor.rows != 0)
								{
									temp_inst_per_estimator[est_idx].push_back(static_cast<int>(temp_instances.size()));
									temp_instances.push_back(models[i]);
								}
							}
						}
						else
						{
							// Add the modes as now instances
							multix::Model model;
							estimator->set_descriptor(model, clusters.row(cluster_idx));
							model.estimator = estimator;
							
							if (!estimator->is_valid(NULL, model.get_descriptor(), NULL))
								continue; 

							temp_inst_per_estimator[est_idx].push_back(static_cast<int>(temp_instances.size()));
							temp_instances.push_back(model);
						}
					}
				}
				instance_descriptors.release();
			}

			if (instances.size() != temp_instances.size())
				_changed = true;

			instances = temp_instances;
			instances_per_estimator = temp_inst_per_estimator;
		}

		void instance_generation(std::vector<multix::Estimator<cv::Mat, multix::Model> * > *_estimators,
			std::vector<multix::Sampler<cv::Mat> *> *_samplers,
			const cv::Mat * _points)
		{
			const int N = static_cast<int>(_estimators->size()) * settings.instance_number; // Calculate the number of instances
			instances.resize(N); // Occupy memory for all the instances
			instances_per_estimator.resize(_estimators->size());

			static auto parallel_code = [](const int _process_idx,
				const int _core_number,
				const int _instance_number,
				const int _instance_number_per_estimator,
				std::vector<multix::Estimator<cv::Mat, multix::Model> * > *_estimators,
				std::vector<multix::Sampler<cv::Mat> *> * const _samplers,
				const cv::Mat *_points,
				const float threshold,
				std::vector<multix::Model> * const _instances,
				std::vector<std::vector<int>> * const _instances_per_estimator,
				int _time_out = 100)
			{
				const int jobs_per_thread = _instance_number / _core_number; // Number of jobs for the current thread
				const int start_idx = _process_idx * jobs_per_thread; // First job index
				const int end_idx = _process_idx == _core_number - 1 ? _instance_number : (_process_idx + 1) * jobs_per_thread; // Last job index
				
				const size_t estimator_number = _estimators->size(); // Number of estimators

				// Get a global or a per-estimator sampler
				std::unique_ptr<multix::Sampler<cv::Mat>> sampler = NULL;

				if (_samplers->size() == 1) // Use the same sampler for all model classes if only one is given
					sampler = _samplers->at(0)->clone();

				int * sample = NULL;
				const multix::Estimator<cv::Mat, multix::Model> const *estimator = NULL;
				int sample_size = 0, counter;
				std::vector<int> * local_instance_index_container = NULL;

				for (int job_idx = start_idx; job_idx < end_idx; ++job_idx)
				{
					const int class_idx = job_idx / _instance_number_per_estimator; // Current class's index
					const int instance_idx = job_idx - class_idx * _instance_number_per_estimator; // Current instance's index

					if (instance_idx == 0 || job_idx == start_idx) // Before the first instance is generated of a class, update every needed parameters
					{
						estimator = _estimators->at(class_idx); // Update the estimator
						sample_size = estimator->sample_size(); // Update the sample size

						if (sample != NULL) // Update the sample container
							delete sample;
						sample = new int[sample_size];
						
						local_instance_index_container = &_instances_per_estimator->at(class_idx);

						if (_samplers->size() > 1) // Update the sampler is more than one is used
							sampler = _samplers->at(class_idx)->clone();
					}

					std::vector<multix::Model> models;
					counter = 0;
					int best_instance_idx = 0;

					std::vector<int> best_inliers;
					do // Do the sampling until a good instance is generated
					{
						models.resize(0);

						sampler->sample(NULL, sample, sample_size); // Sample
						
						// Estimate model parameters from the selected samples
						if (!estimator->estimate_model(_points, sample, &models)) // Estimate models exploiting the minimal sample
							continue;

						// If multiple instances are generated from a single sample choose the best
						//if (models.size() > 1)
						int best_inlier_number = 0;

						for (auto estimated_instance_idx = 0; estimated_instance_idx < models.size(); ++estimated_instance_idx)
						{
							cv::Mat descriptor = models[estimated_instance_idx].descriptor;
							int inlier_number = 0;
							std::vector<int> current_inliers;
							current_inliers.reserve(_points->rows);

							for (auto point_idx = 0; point_idx < _points->rows; ++point_idx)
							{
								if (estimator->error(_points->row(point_idx), descriptor) < threshold)
								{
									++inlier_number;
									current_inliers.push_back(point_idx);
								}

								if (models.size() > 1 && 
									inlier_number + (_points->rows - point_idx) < best_inlier_number) // Break if there is no chance of being better than the best
									break;
							}

							if (inlier_number > best_inlier_number)
							{
								best_inliers = current_inliers;
								best_inlier_number = inlier_number;
								best_instance_idx = estimated_instance_idx;
							}
						}

						if (best_inlier_number > sample_size)
							break;
					} while (counter++ < _time_out);
					
					if (counter == _time_out || models.size() == 0)
					{
						printf("Error during the instance generation step. No sample lead to a good instance.\n");
						continue;
					}

					models[best_instance_idx].inliers.resize(best_inliers.size()); // Put the minimal sample into the inlier container of the instance
					for (auto sample_idx = 0; sample_idx < best_inliers.size(); ++sample_idx)
						models[best_instance_idx].inliers[sample_idx] = best_inliers[sample_idx];

					// Put all estimated model instances into the container
					local_instance_index_container->at(instance_idx) = job_idx; // Store the instance indices for each estimator
					_instances->at(job_idx) = models[best_instance_idx];
				}

				delete sample;
			};

			// Occupy the memory
			for (auto class_idx = 0; class_idx < _estimators->size(); ++class_idx)
				instances_per_estimator[class_idx].resize(settings.instance_number);
			
			std::vector<std::thread>  threads;
			const int workers = MIN(settings.max_thread_number, MIN(N, std::thread::hardware_concurrency()));

			if (workers == 1)
			{
				parallel_code(0, workers, N, settings.instance_number,
					_estimators,
					_samplers,
					_points,
					settings.inlier_outlier_threshold,
					&instances,
					&instances_per_estimator);
			}
			else
			{
				for (int instance_idx = 0; instance_idx < workers; ++instance_idx)
				{
					threads.push_back(std::thread(parallel_code,
						instance_idx, workers, N, settings.instance_number,
						_estimators,
						_samplers,
						_points,
						settings.inlier_outlier_threshold,
						&instances,
						&instances_per_estimator));
				}

				for (std::thread& t : threads)
					t.join();
			}
		}
	};
}
#pragma once

#include <iostream>
#include <math.h>
#include <chrono>
#include <random>
#include <vector>
#include <cv.h>

#include "estimator.h"
#include "model.h"

using namespace std;
using namespace multix;
using namespace cv;

// This is the estimator class for estimating a 2D line. A model estimation method and error calculation method are implemented
class LineSegmentEstimator : public Estimator < Mat, Model >
{
protected:
	float min_length;
	float max_length;

public:
	enum DataType { Points, PointsAndTangents };

	LineSegmentEstimator() : represent_by_points(true), 
		data_type(DataType::Points),
		min_length(0),
		max_length(FLT_MAX)
	{ }
	~LineSegmentEstimator() {}

	void SetDataType(DataType _data_type)
	{
		data_type = _data_type;
	}

	int sample_size() const {
		return 2;
	}

	int inlier_limit() const {
		return  7 * sample_size();
	}

	float model_weight() const
	{
		return 1;
	}

	bool is_mode_seeking_applicable() const
	{
		return false;
	}

	void SetMinLength(float length)
	{
		min_length = length;
	}

	void SetMaxLength(float length)
	{
		max_length = length;
	}

	bool estimate_model_nonminimal(const Mat * const data,
		const int *sample,
		int sample_number,
		vector<Model>* models) const
	{
		Model model;

		if (sample_number < 2)
			return false;
		
		Mat A(sample_number, 3, CV_64F);
		int idx;
		Mat mass_point = Mat::zeros(1, 2, CV_32F);
		for (int i = 0; i < sample_number; ++i)
		{
			idx = sample[i];
			mass_point.at<float>(0) += data->at<float>(idx, 0);
			mass_point.at<float>(1) += data->at<float>(idx, 1);

			A.at<double>(i, 0) = (double)data->at<float>(idx, 0);
			A.at<double>(i, 1) = (double)data->at<float>(idx, 1);
			A.at<double>(i, 2) = 1;
		}
		mass_point = mass_point * (1.0 / sample_number);

		Mat AtA = A.t() * A;
		Mat eValues, eVectors;
		eigen(AtA, eValues, eVectors);

		Mat line = eVectors.row(2);
		line.convertTo(line, CV_32F);
		
		float length = sqrt(line.at<float>(0) * line.at<float>(0) + line.at<float>(1) * line.at<float>(1));
		line.at<float>(0) /= length;
		line.at<float>(1) /= length;
		line.at<float>(2) = -(line.at<float>(0) * mass_point.at<float>(0) + line.at<float>(1) * mass_point.at<float>(1));
		line = line.t();

		if (represent_by_points) // Calculate the closest point to the origin
		{
			const float steepness = line.at<float>(1) / line.at<float>(0);
			float min_x = FLT_MAX, min_y = FLT_MAX;
			float max_x = -FLT_MAX, max_y = -FLT_MAX;
			float min_dist = FLT_MAX, max_dist = -FLT_MAX;

			idx = sample[0];
			float x0 = data->at<float>(idx, 0), y0 = data->at<float>(idx, 1);
			float x1;
			GetClosestPointOnTheLine((Mat_<float>(2, 1) << x0, y0), line, x0, y0);

			for (int i = 0; i < sample_number; ++i)
			{
				idx = sample[i];
				float x1 = data->at<float>(idx, 0), y1 = data->at<float>(idx, 1);
				GetClosestPointOnTheLine((Mat_<float>(2, 1) << x1, y1), line, x1, y1);

				float dx = x1 - x0;
				float dy = y1 - y0;

				float sqr_dist = dx*dx + dy*dy;
				if (x1 < x0)
					sqr_dist = -sqr_dist;
				if (sqr_dist < min_dist)
				{
					min_dist = sqr_dist;
					min_x = x1;
					min_y = y1;
				} 

				if (sqr_dist > max_dist)
				{
					max_dist = sqr_dist;
					max_x = x1;
					max_y = y1;
				}

				/*if (steepness < 0.5)
				{
					if (data->at<float>(idx, 1) < min_y)
					{
						min_x = data->at<float>(idx, 0);
						min_y = data->at<float>(idx, 1);
					} else if (data->at<float>(idx, 1) > max_y)
					{
						max_x = data->at<float>(idx, 0);
						max_y = data->at<float>(idx, 1);
					}
				}
				else
				{
					if (data->at<float>(idx, 0) < min_x)
					{
						min_x = data->at<float>(idx, 0);
						min_y = data->at<float>(idx, 1);
					}
					else if (data->at<float>(idx, 0) > max_x)
					{
						max_x = data->at<float>(idx, 0);
						max_y = data->at<float>(idx, 1);
					}
				}*/
			}
			
			/*GetClosestPointOnTheLine((Mat_<float>(2, 1) << min_x, min_y), line, x0, y0);
			GetClosestPointOnTheLine((Mat_<float>(2, 1) << max_x, max_y), line, x1, y1);*/

			const float dx = max_x - min_x;
			const float dy = max_y - min_y;
			const float dist = dx*dx + dy*dy;
			if (dist < min_length * min_length ||
				dist > max_length * max_length)
				return false;

			model.descriptor_by_points = (Mat_<float>(4, 1) << min_x, min_y, max_x, max_y);
		}
		
		model.descriptor = line;
		model.estimator = this;
		models->push_back(model);

		if (model.descriptor.rows == 0 || model.descriptor.cols == 0)
			return false;
		return true;
	}

	bool estimate_model(const Mat * const data,
		const int *sample, 
		vector<Model>* models) const
	{
		return estimate_from_points(data, sample, models);
	}
	
	bool estimate_from_points(const Mat * const data,
		const int *sample,
		vector<Model>* models) const
	{
		if (sample[0] == sample[1])
			return false;

		// model calculation 
		Model model;

		Mat pt1 = data->row(sample[0]);
		Mat pt2 = data->row(sample[1]);
		
		Mat v = pt2 - pt1;
		float l = norm(v);
		if (l < min_length || l > max_length)
			return false;

		v = v / l;
		Mat n = (Mat_<float>(2, 1) << -v.at<float>(1), v.at<float>(0));
		float c = -(n.at<float>(0) * pt2.at<float>(0) + n.at<float>(1) * pt2.at<float>(1));

		if (represent_by_points) // Calculate the closest point to the origin
			calculate_point_representation(pt1, pt2, model.descriptor_by_points);
		
		model.descriptor = (Mat_<float>(3, 1) << n.at<float>(0), n.at<float>(1), c);
		if (model.descriptor.rows == 0 || model.descriptor.cols == 0)
			return false;

		model.estimator = this;
		models->push_back(model);
		return true;
	}

	void calculate_implicit_form(const Mat &point_representation, Mat &descriptor) const
	{
		const Mat pt1 = (Mat_<float>(2, 1) << point_representation.at<float>(0), point_representation.at<float>(1));
		const Mat pt2 = (Mat_<float>(2, 1) << point_representation.at<float>(2), point_representation.at<float>(3));
		Mat v = pt2 - pt1;
		v = v / norm(v);
		
		float a = -v.at<float>(1);
		float b = v.at<float>(0);
		float c = -a * pt2.at<float>(0) - b * pt2.at<float>(1);

		descriptor = (Mat_<float>(3, 1) << a, b, c);
	}

	void calculate_point_representation(Mat point1, Mat point2, Mat &descriptor) const
	{
		descriptor = (Mat_<float>(4, 1) << point1.at<float>(0), point1.at<float>(1), point2.at<float>(0), point2.at<float>(1));
	}

	void set_descriptor(Model &model, Mat descriptor) const
	{
		const float a = descriptor.at<float>(0);
		const float b = descriptor.at<float>(1);
		const float mag = sqrt(a*a + b*b);
		descriptor.at<float>(0) = a / mag;
		descriptor.at<float>(1) = b / mag;

		model.descriptor = descriptor;
	}

	float error(const Mat& point, const Model& model) const
	{
		float x, y;
		GetClosestPointOnTheLine(point, model.descriptor, x, y);
		
		const float *desc_points_ptr = (float *)model.descriptor_by_points.data;
		const float x0 = *(desc_points_ptr);
		const float y0 = *(desc_points_ptr + 1);
		const float x1 = *(desc_points_ptr + 2);
		const float y1 = *(desc_points_ptr + 3);

		const float vx = x1 - x0;
		const float vy = y1 - y0;

		const float spatial_coherence_weight = (x - x0) / vx;

		float distance;
		if (spatial_coherence_weight < 0)
		{
			const float dx = point.at<float>(0) - x0;
			const float dy = point.at<float>(1) - y0;
			distance = sqrt(dx*dx + dy*dy);
		}
		else if (spatial_coherence_weight >= 1)
		{
			const float dx = point.at<float>(0) - x1;
			const float dy = point.at<float>(1) - y1;
			distance = sqrt(dx*dx + dy*dy);
		} else
			distance = abs(point.at<float>(0) * model.descriptor.at<float>(0) + point.at<float>(1) * model.descriptor.at<float>(1) + model.descriptor.at<float>(2));
		return (double)distance;
	}

	float error(const Mat& point, const Mat& descriptor) const
	{
		printf("[Line Segment Error] Not implemented properly!\n");
		float distance = abs(point.at<float>(0) * descriptor.at<float>(0) + point.at<float>(1) * descriptor.at<float>(1) + descriptor.at<float>(2));
		return distance;
	}

	bool is_valid(const Mat * const data, const Mat& descriptor, vector<int> const * inliers) const
	{
		return true;
	}

	void GetClosestPointOnTheLine(const Mat& point_0, const Mat &descriptor, float &x, float &y) const
	{
		const float *point_0_ptr = (float *)point_0.data;
		const float x0 = *(point_0_ptr);
		const float y0 = *(point_0_ptr + 1);

		const float *desc_ptr = (float *)descriptor.data;
		const float a = *(desc_ptr);
		const float b = *(desc_ptr + 1);
		const float c = *(desc_ptr + 2);
		
		const float a_b = b / a;
		const float b2 = b * b;
		
		x = (b2 * x0 - a * (b * y0 + c)) / (a * a + b2);
		y = y0 - a_b * x0 + a_b * x;
	}

protected:
	bool represent_by_points;
	DataType data_type;
};
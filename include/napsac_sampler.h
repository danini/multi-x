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

#include <stdlib.h>
#include <algorithm>
#include <chrono>
#include <random>
#include <vector>
#include <unordered_map>
#include <cv.h>

#include "sampler.h"

namespace multix 
{
	// Purely virtual class used for the sampling consensus methods
	template <class Datum> class NapsacSampler : public Sampler<Datum> 
	{
	public:		
		explicit NapsacSampler()
			: Sampler<Datum>() {}
		~NapsacSampler() {}
		
		explicit NapsacSampler(const NapsacSampler &sampler)
		{
			neighbors = sampler.neighbors;
		}

		std::unique_ptr<Sampler> clone() const override
		{
			return std::make_unique<NapsacSampler>(*this);
		}

		// Initializes any non-trivial variables and sets up sampler if necessary. Must be called before Sample is called.
		bool initialize() { return true;  }

		bool initialize(const int _point_number,
			std::vector<std::vector<cv::DMatch>> * _neighbors)
		{
			neighbors = _neighbors;
			return true;
		}
		
		// Samples the input variable data and fills the std::vector subset with the samples.
		bool sample(const std::vector<Datum>* _data,
			int * const _subset,
			int _min_num_samples)
		{
			const int point_number = static_cast<int>(neighbors->size());
			int point_idx;
			
			while (1)
			{
				point_idx = random_integer(0, point_number - 1);
				const std::vector<cv::DMatch> * const current_neighbors = &neighbors->at(point_idx);
				const int current_size = static_cast<int>(current_neighbors->size());

				if (current_size < _min_num_samples + 1)
					continue;

				_subset[_min_num_samples - 1] = point_idx;

				// Generate a unique index set assuming that the first neighbor is the chosen point itself
				random_unique_integer_set(1, current_size - 1, _subset, _min_num_samples - 1);
				for (auto i = 0; i < _min_num_samples - 1; ++i)
					_subset[i] = current_neighbors->at(_subset[i]).trainIdx;
				break;
			}

			return true;
		}

	protected:
		std::vector<std::vector<cv::DMatch>> * neighbors;
	};

}  // namespace multix

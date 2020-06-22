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

#include <vector>

namespace multix 
{
	// Purely virtual class used for the sampling of consensus methods
	template <class Datum> class Sampler 
	{
	protected:
		// Get a random int between lower and upper (inclusive).
		int random_integer(const int _lower,
			const int _upper)
		{
			static std::default_random_engine util_generator;
			std::uniform_int_distribution<int> distribution(_lower, _upper);
			return distribution(util_generator);
		}

		void random_unique_integer_set(const int _lower,
			const int _upper,
			int * const _sample,
			const size_t _number)
		{
			static std::default_random_engine util_generator;
			std::uniform_int_distribution<int> distribution(_lower, _upper);

			for (auto i = 0; i < _number; i++) {
				_sample[i] = distribution(util_generator);
				for (auto j = i - 1; j >= 0; j--) {
					if (_sample[i] == _sample[j]) {
						i--;
						break;
					}
				}
			}
		}

	public:
		explicit Sampler() {}

		virtual std::unique_ptr<Sampler> clone() const = 0;

		// Initializes any non-trivial variables and sets up sampler if
		// necessary. Must be called before Sample is called.
		virtual bool initialize() = 0;

		virtual ~Sampler() {}
		// Samples the input variable data and fills the std::vector subset with the
		// samples.
		virtual bool sample(const std::vector<Datum>* _data,
			int * const _subset,
			int _min_num_samples) = 0;
	};

}  // namespace multix


// Copyright (C) 2013 The Regents of the University of California (Regents).
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above
//       copyright notice, this list of conditions and the following
//       disclaimer in the documentation and/or other materials provided
//       with the distribution.
//
//     * Neither the name of The Regents or University of California nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Please contact the author of this library if you have any questions.
// Author: Chris Sweeney (cmsweeney@cs.ucsb.edu)

#ifndef THEIA_SOLVERS_QUALITY_MEASUREMENT_H_
#define THEIA_SOLVERS_QUALITY_MEASUREMENT_H_

#include <vector>

namespace theia 
{
	// Purely virtual class to be used with sampling consensus estimators
	// (e.g. Ransac, Prosac, MLESac, etc.). This class is implemented to assess the
	// quality of the data. A trivial example is the inlier quality measurement
	// (i.e. if the error for a measurement is less than a threshold, then it is an
	// inlier).
	class QualityMeasurement 
	{
	public:
		enum 
		{
			INVALID = -99999
		};

		explicit QualityMeasurement(const double error_thresh)
			: error_thresh_(error_thresh) {}
		virtual ~QualityMeasurement() {}

		// Initializes any non-trivial variables and sets up sampler if
		// necessary. Must be called before Compare is called.
		virtual bool Initialize() 
		{
			return true; 
		}

		// Given the residuals, assess a quality metric for the data.
		virtual double Calculate(const std::vector<double>& residuals) = 0;

		// Given two quality measurements, determine which is betters. Note that
		// larger is not always better! Returns true if quality1 is of higher quality
		// than quality2.
		virtual bool Compare(const double quality1, const double quality2) const = 0;

		// Returns the maximum inlier ratio found thus far through the Compare
		// call. This is used in SampleConsensusEstimator to recompute the necessary
		// number of iterations.
		virtual double GetInlierRatio() const = 0;

	protected:
		double error_thresh_;
	};

}  // namespace theia

#endif  // THEIA_SOLVERS_QUALITY_MEASUREMENT_H_

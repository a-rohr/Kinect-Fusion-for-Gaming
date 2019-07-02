#pragma once
#include "SimpleMesh.h"
#include <algorithm>

class ProcrustesAligner {
public:
	Matrix4f estimatePose(const std::vector<Vector3f>& sourcePoints, const std::vector<Vector3f>& targetPoints) {
		ASSERT(sourcePoints.size() == targetPoints.size() && "The number of source and target points should be the same, since every source point is matched with corresponding target point.");

		// We estimate the pose between source and target points using Procrustes algorithm.
		// Our shapes have the same scale, therefore we don't estimate scale. We estimated rotation and translation
		// from source points to target points.

		auto sourceMean = computeMean(sourcePoints);
		auto targetMean = computeMean(targetPoints);
		
		Matrix3f rotation = estimateRotation(sourcePoints, sourceMean, targetPoints, targetMean);
		Vector3f translation = computeTranslation(sourceMean, targetMean);

		// To apply the pose to point x on shape X in the case of Procrustes, we execute:
		// 1. Translation of a point to the shape Y: x' = x + t
		// 2. Rotation of the point around the mean of shape Y: 
		//    y = R (x' - yMean) + yMean = R (x + t - yMean) + yMean = R x + (R t - R yMean + yMean)

		std::cout << "Rotation:\n" << rotation << std::endl;
		std::cout << "Translation:\n" << translation << std::endl;
		
		Matrix4f estimatedPose = Matrix4f::Identity();
		estimatedPose.block(0, 0, 3, 3) = rotation;
		estimatedPose.block(0, 3, 3, 1) = rotation * translation - rotation * targetMean + targetMean;

		return estimatedPose;
	}

private:
	Vector3f computeMean(const std::vector<Vector3f>& points) {
		// TODO: Compute the mean of input points.

		Vector3f mean = Vector3f::Zero();
		for (Vector3f point : points)
		{
			mean += point;
		}

		mean = mean / points.size();
		return mean;
	}

	Matrix3f estimateRotation(const std::vector<Vector3f>& sourcePoints, const Vector3f& sourceMean, const std::vector<Vector3f>& targetPoints, const Vector3f& targetMean) {
		// TODO: Estimate the rotation from source to target points, following the Procrustes algorithm.
		// To compute the singular value decomposition you can use JacobiSVD() from Eigen.
		// Important: The covariance matrices should contain mean-centered source/target points. 
		std::vector<Vector3f> meanCenteredSourcePoints;
		for (Vector3f sourcePoint : sourcePoints)
		{
			meanCenteredSourcePoints.push_back(sourcePoint - sourceMean);
		}

		std::vector<Vector3f> meanCenteredTargetPoints;
		for (Vector3f targetPoint : targetPoints)
		{
			meanCenteredTargetPoints.push_back(targetPoint - targetMean);
		}

		int numPoints = meanCenteredTargetPoints.size();
		Matrix3f crossCov;

		for (int r = 0; r < 3; r++)
		{
			for (int c = 0; c < 3; c++)
			{
				float total = 0;
				for (int n = 0; n < numPoints; n++)
				{
					total += meanCenteredTargetPoints[n](r) * meanCenteredSourcePoints[n](c);
				}

				crossCov(r, c) = total;
			}
		}

		JacobiSVD<MatrixXf> svd(crossCov, ComputeThinU | ComputeThinV);

		Matrix3f R = svd.matrixU() * svd.matrixV().transpose();
		return R;
	}

	Vector3f computeTranslation(const Vector3f& sourceMean, const Vector3f& targetMean) {
		// TODO: Compute the translation vector from source to target points.
		
		return targetMean - sourceMean;
		//return Vector3f::Zero();
	}
};
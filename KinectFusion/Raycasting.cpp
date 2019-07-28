#include <iostream>

#include <Eigen/Core>

#include "Raycasting.hpp"

bool search_zero_crossing(Eigen::Vector3d origin, Eigen::Vector3d ray, double& length,
	const double stepSizeVoxel, const double epsilon, TsdfUtils::TsdfData tsdfData)
{
	Eigen::Vector3d point = origin + ray * length;
	float pointValue = TsdfUtils::getInterpolatedTsdfValue(point, tsdfData);
	float previousPointValue = pointValue;

	double stepSize = tsdfData.voxelSize * stepSizeVoxel;
	double previousLength = length;

	while (pointValue > 0.98f)
	{
		previousPointValue = pointValue;
		previousLength = length;
		length += 0.014;
		point = origin + ray * length;
		if (!TsdfUtils::isInTsdfGrid(point, tsdfData.size)) { return false; }
		pointValue = TsdfUtils::getInterpolatedTsdfValue(point, tsdfData);

	}

	while (true)
	{
		if ((previousPointValue > 0) && (pointValue <= 0))
		{
			float zerolength;
			float zeroValue;
			do //Iterpolation to find zero crossing
			{
				zerolength = previousLength - ((length - previousLength)*previousPointValue) / (pointValue - previousPointValue);
				Eigen::Vector3d zero_point = origin + ray * zerolength;
				zeroValue = TsdfUtils::getInterpolatedTsdfValue(zero_point, tsdfData);
				if (zeroValue < 0)
				{
					length = zerolength;
					pointValue = zeroValue;
				}
				else
				{
					previousLength = zerolength;
					previousPointValue = zeroValue;
				}

			} while (std::abs(zeroValue) > (1e-3));
			length = zerolength;
			return true;
		}

		previousPointValue = pointValue;
		previousLength = length;
		length += stepSize;
		point = origin + ray * length;
		if (!TsdfUtils::isInTsdfGrid(point, tsdfData.size)) { return false; }
		pointValue = TsdfUtils::getInterpolatedTsdfValue(point, tsdfData);

	}
	//


	return true;
}


void castRays(TsdfUtils::TsdfData tsdfData, Eigen::Matrix4f cameraPose, Eigen::Matrix3d cameraIntrisic,
	const unsigned int resolutionWidth, const unsigned int resolutionHeight,
	const double stepSizeVoxel, const double epsilon,
	cv::Mat& depthImage, cv::Mat& normalImage)
{
	depthImage.setTo(cv::Scalar(-std::numeric_limits<float>::infinity()));
	normalImage.setTo(cv::Vec3f(-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()));

	double fx = cameraIntrisic(0, 0)*resolutionWidth;
	double fy = cameraIntrisic(1, 1)*resolutionHeight;
	double cx = cameraIntrisic(0, 2)*resolutionWidth - 0.5;
	double cy = cameraIntrisic(1, 2)*resolutionHeight - 0.5;

	Eigen::Matrix3d cam_R = cameraPose.cast<double>().block(0, 0, 3, 3).transpose();
	Eigen::Vector3d cam_t = -cam_R * cameraPose.cast<double>().col(3).head(3);
	Eigen::Vector3d origin = cam_t;

#pragma omp parallel for
	for (int v = 0; v < resolutionHeight; ++v)
	{
		for (int u = 0; u < resolutionWidth; ++u)
		{
			double rayX = ((double)u - cx) / fx;
			double rayY = ((double)v - cy) / fy;
			Eigen::Vector3d ray(rayX, rayY, 1);

			ray = cam_R * ray;
			cv::Vec3f normal;
			double length;

			if (TsdfUtils::projectToTsdf(origin, ray, length, tsdfData.size) && // Does the ray hit the voxel grid
				search_zero_crossing(origin, ray, length, stepSizeVoxel, epsilon, tsdfData)) // Does the ray hit a zero crossing
			{
				depthImage.at<float>(v, u) = (float)length;

				Eigen::Vector3d point = origin + ray * length;

				const double voxelSize = voxelSize;

				float valueXForward = TsdfUtils::getInterpolatedTsdfValue(point + Eigen::Vector3d(voxelSize, 0, 0), tsdfData);
				float valueXBackward = TsdfUtils::getInterpolatedTsdfValue(point + Eigen::Vector3d(-voxelSize, 0, 0), tsdfData);


				float valueYForward = TsdfUtils::getInterpolatedTsdfValue(point + Eigen::Vector3d(0, voxelSize, 0), tsdfData);
				float valueYBackward = TsdfUtils::getInterpolatedTsdfValue(point + Eigen::Vector3d(0, -voxelSize, 0), tsdfData);


				float valueZForward = TsdfUtils::getInterpolatedTsdfValue(point + Eigen::Vector3d(0, 0, voxelSize), tsdfData);
				float valueZBackward = TsdfUtils::getInterpolatedTsdfValue(point + Eigen::Vector3d(0, 0, -voxelSize), tsdfData);

				Eigen::Vector3d normalVec(
					(valueXForward - valueXBackward) / 2,
					(valueYForward - valueYBackward) / 2,
					(valueZForward - valueZBackward) / 2
				);
				normalVec = cam_R.transpose() * normalVec;
				normalVec.normalize();

				normal(0) = (float)normalVec.x();
				normal(1) = (float)normalVec.y();
				normal(2) = (float)normalVec.z();

				normalImage.at<cv::Vec3f>(v, u) = normal;
			}
			else
			{
				depthImage.at<float>(v, u) = -std::numeric_limits<float>::infinity();
				normalImage.at<cv::Vec3f>(v, u) = cv::Vec3f(-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity());
			}
		}
	}

}
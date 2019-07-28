#include <Eigen/Core>
#include <opencv2/core/core.hpp>

#ifndef KINECT_FUSION_RAYTRACER_HPP
#define KINECT_FUSION_RAYTRACER_HPP

#include "TsdfUtils.h"

void raytraceImage(TsdfUtils::TsdfData tsdfData, Eigen::Matrix4f cameraPose, Eigen::Matrix3d cameraIntrisic,
                      unsigned int resolutionWidth, unsigned int resolutionHeight,
                      const double stepSizeVoxel, const double epsilon,
                      cv::Mat& depthImage, cv::Mat& normalImage);

inline bool zero_cross_finder(TsdfUtils::TsdfData tsdfData, Eigen::Vector3d origin, Eigen::Vector3d rayDir, double &length, float miu,
	Eigen::Vector3d &vertex, Eigen::Vector3f &normal, const double sizeVoxel)
{

	length = 0.4; //minimum possible distance for kinect at 0.4
	Eigen::Vector3f normal_aux;


	//INITIALIZATION VALUES
	Eigen::Vector3d currentPos = origin + length * rayDir;
	if (length > 8) { return false; }//Sensor limit

	if (!TsdfUtils::isInTsdfGrid(currentPos, tsdfData.size)) { return false; };

	float current_Val = TsdfUtils::getInterpolatedTsdfValue(currentPos, tsdfData);
	float prev_Val = current_Val;
	float prev_Length = length;

	if (current_Val < 0) { return false; }

	//While in absolute free space

	while (current_Val > 0.98f) //(ideally ==1.0f)
	{
		prev_Val = current_Val;
		prev_Length = length;
		length += miu;
		currentPos = origin + length * rayDir;
		if (length > 8) { return false; }
		if (!TsdfUtils::isInTsdfGrid(currentPos, tsdfData.size)) { return false; };


		current_Val = TsdfUtils::getTsdfValueFromWorldCoord(currentPos, tsdfData); //This is the rough voxel Value

		//Explanation: If voxel value is 1, it would need to advance at least miu before crossing
		//zero-level surface
	}

	//Replace rough value with trilinear interp
	prev_Val = TsdfUtils::getInterpolatedTsdfValue(origin + prev_Length * rayDir, tsdfData);
	current_Val = TsdfUtils::getInterpolatedTsdfValue(currentPos, tsdfData);

	while (true)
	{
		if (current_Val < 0 && prev_Val>0)
		{
			float zerolength;
			float zeroValue;
			do //Iterpolation to find zero crossing
			{
				zerolength = prev_Length - ((length - prev_Length)*prev_Val) / (current_Val - prev_Val);
				zeroValue = TsdfUtils::getInterpolatedTsdfValue(origin + zerolength * rayDir, tsdfData);
				if (zeroValue < 0)
				{
					length = zerolength;
					current_Val = zeroValue;
				}
				else
				{
					prev_Length = zerolength;
					prev_Val = zeroValue;
				}

			} while (std::abs(zeroValue) > (1e-3));
			//((length - prev_Length) > sizeVoxel); //Small distance to get good interpolation

			//FINALLY: WE GOT THE RIGHT INTERSECTION. Now lets change "vertex" and "normal"

			length = zerolength;
			//Calculate position and normal
			vertex = origin + length * rayDir;

			//We approximate the gradient to get the normal with finite differences

			float diff_X = TsdfUtils::getInterpolatedTsdfValue(vertex + Eigen::Vector3d(sizeVoxel, 0, 0), tsdfData) -
				TsdfUtils::getInterpolatedTsdfValue(vertex - Eigen::Vector3d(sizeVoxel, 0, 0), tsdfData);
			float diff_Y = TsdfUtils::getInterpolatedTsdfValue(vertex + Eigen::Vector3d(0, sizeVoxel, 0), tsdfData) -
				TsdfUtils::getInterpolatedTsdfValue(vertex - Eigen::Vector3d(0, sizeVoxel, 0), tsdfData);
			float diff_Z = TsdfUtils::getInterpolatedTsdfValue(vertex + Eigen::Vector3d(0, 0, sizeVoxel), tsdfData) -
				TsdfUtils::getInterpolatedTsdfValue(vertex - Eigen::Vector3d(0, 0, sizeVoxel), tsdfData);

			normal = Eigen::Vector3f(diff_X / 2, diff_Y / 2, diff_Z / 2);
			normal.normalize();

			//normal(0) = normal_aux.x();
			//normal(1) = normal_aux.y();
			//normal(2) = normal_aux.z();

			return true;
		}
		prev_Val = current_Val;
		prev_Length = length;
		length += 1.5*sizeVoxel;
		currentPos = origin + length * rayDir;
		if (length > 8) { return false; }//The vision limit is 8m. V E R I F Y
		if (!TsdfUtils::isInTsdfGrid(currentPos, tsdfData.size)) { return false; };
		current_Val = TsdfUtils::getInterpolatedTsdfValue(currentPos, tsdfData);

	}

	return false;
}


inline void castRays3(TsdfUtils::TsdfData tsdfData, Eigen::Matrix4f cameraPose, Eigen::Matrix3d intrinsics, unsigned int resolutionWidth, const unsigned int resolutionHeight, cv::Mat& depthImage, cv::Mat& normalImage)
{
	Eigen::Matrix3d cam_R = cameraPose.cast<double>().block(0, 0, 3, 3).transpose();
	Eigen::Vector3d cam_t = -cam_R * cameraPose.cast<double>().col(3).head(3);
	Eigen::Vector3d origin = cam_t;

	depthImage.setTo(cv::Scalar(-std::numeric_limits<float>::infinity()));
	normalImage.setTo(cv::Vec3f(-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()));

	double _fovX = intrinsics(0, 0) * resolutionWidth;
	double _fovY = intrinsics(1, 1) * resolutionHeight;
	double _cX = intrinsics(0, 2) * resolutionWidth;
	double _cY = intrinsics(1, 2) * resolutionHeight;

//#pragma omp parallel for 
	for (int y = 0; y < resolutionHeight; y++)
	{

		for (int x = 0; x < resolutionWidth; x++)
		{
			Eigen::Vector3d rayDir = Eigen::Vector3d((x - _cX) / _fovX, (y - _cY) / _fovY, 1);
			rayDir.normalize();

			rayDir = cam_R * rayDir;

			double length = 0;

			cv::Vec3f normal_im;

			Eigen::Vector3d vertex;
			Eigen::Vector3f normal;

			if (zero_cross_finder(tsdfData, origin, rayDir, length, tsdfData.truncationDistance, vertex, normal, tsdfData.voxelSize)) // Does the ray hit a zero crossing
			{

				Eigen::Vector3d point = vertex;
				Eigen::Vector3f normalVec = normal;
				normalVec.normalize();

				normal_im(0) = (float)normalVec.x();
				normal_im(1) = (float)normalVec.y();
				normal_im(2) = (float)normalVec.z();
				#pragma omp critical(pushhit)
				{
					depthImage.at<float>(y, x) = length;
					normalImage.at<cv::Vec3f>(y, x) = normal_im;
				}
			}
		}
	}
}

#endif //KINECT_FUSION_RAYTRACER_HPP

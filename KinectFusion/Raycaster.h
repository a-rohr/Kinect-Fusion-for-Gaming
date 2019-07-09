#pragma once
#include "Volume.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

class Raycaster {
public:
	Raycaster(Matrix4f &camPose, Matrix3f &intrinsics, int depthImgHeight, int depthImgWidth, BYTE * colorRGBX) {
		

		// Get depth intrinsics.
		_fovX = intrinsics(0, 0);
		_fovY = intrinsics(1, 1);
		_cX = intrinsics(0, 2);
		_cY = intrinsics(1, 2);

		// Compute inverse depth extrinsics.
		_rotation = camPose.block(0, 0, 3, 3).transpose();
		_translation = camPose.block<3, 1>(0, 3);
		_translation = -_rotation * _translation;

		_colorRGBX = colorRGBX;
		_depthImgHeight = depthImgHeight;
		_depthImgWidth = depthImgWidth;
	}


	bool zero_cross_finder(Volume &vol, Vector3d origin, Vector3d rayDir, double &length, float miu,
		Vector3d &vertex, Vector3f &normal, const double sizeVoxel)
	{
		
		length = 0.4; //minimum possible distance for kinect at 0.4
		Vector3f normal_aux;


		//INITIALIZATION VALUES
		Vector3d currentPos = origin + length * rayDir;
		if (length > 8) { return false; }//Sensor limit
		if (!vol.withinGrid(currentPos)) { return false; };

		float current_Val = vol.get_InterpVal(currentPos);
		float prev_Val = current_Val;
		float prev_Length = length;

		if (current_Val < 0) { return false; }

		//While in absolute free space

		while (current_Val > 0.99f) //(ideally ==1.0f)
		{
			prev_Val = current_Val;
			prev_Length = length;
			length += miu;
			currentPos = origin + length * rayDir;
			if (length > 8) { return false; }
			if (!vol.withinGrid(currentPos)) { return false; };
			
			current_Val = vol.getNodeValAtWorld(currentPos); //This is the rough voxel Value

			//Explanation: If voxel value is 1, it would need to advance at least miu before crossing
			//zero-level surface
		}

		//Replace rough value with trilinear interp
		prev_Val = vol.get_InterpVal(origin + prev_Length * rayDir);
		current_Val = vol.get_InterpVal(currentPos);

		while (true)
		{
			if (current_Val < 0 && prev_Val>0)
			{
				float zerolength;
				float zeroValue;
				do //Iterpolation to find zero crossing
				{
					zerolength = prev_Length - ((length - prev_Length)*prev_Val) / (current_Val - prev_Val);
					zeroValue = vol.get_InterpVal(origin + zerolength * rayDir);
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

				} while ((prev_Length - length) > sizeVoxel); //Small distance to get good interpolation

				//FINALLY: WE GOT THE RIGHT INTERSECTION. Now lets change "vertex" and "normal"

				length = zerolength;
				//Calculate position and normal
				vertex = origin + length * rayDir;

				//We approximate the gradient to get the normal with finite differences
				float diff_X = vol.get_InterpVal(vertex + Vector3d(sizeVoxel, 0, 0)) -
					vol.get_InterpVal(vertex - Vector3d(sizeVoxel, 0, 0));
				float diff_Y = vol.get_InterpVal(vertex + Vector3d(0, sizeVoxel, 0)) -
					vol.get_InterpVal(vertex - Vector3d(0, sizeVoxel, 0));
				float diff_Z = vol.get_InterpVal(vertex + Vector3d(0, 0, sizeVoxel)) -
					vol.get_InterpVal(vertex - Vector3d(0, 0, sizeVoxel));

				normal = Vector3f(diff_X / 2, diff_Y / 2, diff_Z / 2);
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
			if (!vol.withinGrid(currentPos)) { return false; };
			current_Val = vol.get_InterpVal(currentPos);

		}

		return false;
	}

	void castRays3(Volume &vol, cv::Mat& depthImage, cv::Mat& normalImage, std::vector<Vector3f> &hitPoints,
		std::vector<Vector3f> &hitNormals, float miu)
	{

		miu = 0.04;
		std::cout << "Executing Raycasting" << std::endl;
		clock_t begin = clock();

		double sizeVoxel = vol.cellSizeX_g;
		//float sizeVoxel = (vol.upperTopPoint_g.x() - vol.lowerLeftPoint_g.x()) / vol.getDimX();

		Eigen::Vector3d origin = _translation.cast<double>();

		hitPoints.reserve(_depthImgHeight * _depthImgWidth);
		hitNormals.reserve(_depthImgHeight * _depthImgWidth);

		#pragma omp parallel for  //size of hitPoints & hitNormals are not equal when enabled
		for (int y = 0; y < _depthImgHeight; y++)
		{
			
			for (int x = 0; x < _depthImgWidth; x++)
			{
				Vector3d rayDir = Vector3d((x - _cX) / _fovX, (y - _cY) / _fovY, 1);
				rayDir.normalize();

				rayDir = _rotation.cast<double>() * rayDir;

				double length = 0;

				cv::Vec3f normal_im;

				Vector3d vertex;
				Vector3f normal;

				if (zero_cross_finder(vol, origin, rayDir, length, miu, vertex, normal, sizeVoxel)) // Does the ray hit a zero crossing
				{

					Vector3d point = vertex;
					Vector3f normalVec = normal;
					normalVec.normalize();

					//const double voxelSize = vol.cellSizeX_g;


					normal_im(0) = (float)normalVec.x();
					normal_im(1) = (float)normalVec.y();
					normal_im(2) = (float)normalVec.z();

					//normalImage.at<cv::Vec3f>(y, x) = normal;
					#pragma omp critical(pushhit)
					{
						hitPoints.push_back(point.cast<float>());
						hitNormals.push_back(normalVec);
					}
				}
				else
				{
					depthImage.at<float>(y, x) = -std::numeric_limits<float>::infinity();
					//depthImage.at<float>(v, u) = 0.0f;
					//normalImage.at<cv::Vec3f>(v, u) = normal;
					normalImage.at<cv::Vec3f>(y, x) = cv::Vec3f(-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity());

				}
			}
		}

		clock_t end = clock();
		double elapsedSecs = double(end - begin) / 1000;
		std::cout << "Raycasting finished in " << elapsedSecs << " seconds." << std::endl;
	}

	bool searchRay(Volume& vol, Eigen::Vector3d origin, Eigen::Vector3d ray, double& length, const double stepSizeVoxel, const double epsilon)
	{
		double voxelSize = vol.cellSizeX_g;

		Eigen::Vector3d point = origin + ray * length;
		float pointValue = vol.get_InterpVal(point);
		float previousPointValue = pointValue;


		double stepSize = voxelSize * stepSizeVoxel;
		double previousLength = length;

		while (true)
		{
			previousLength = length;
			length += stepSize;
			point = origin + ray * length;

			if (!vol.withinGrid(point)) { 
				return false; 
			}

			previousPointValue = pointValue;
			pointValue = vol.get_InterpVal(point);

			if (previousPointValue > 0.0 && pointValue < 0.0) { break; }
		}

		while (true)
		{
			double middleLength = (previousLength + length) / 2;
			float middleValue = vol.get_InterpVal(origin + ray * middleLength);

			if (middleValue > epsilon)
			{
				previousLength = middleLength;
			}
			else if (middleValue < -epsilon)
			{
				length = middleLength;
			}
			else
			{
				break;
			}

			if (std::abs(length - previousLength) < 1e-2)
			{
				break;
			}
		}

		return true;
	}


	void castRays2(Volume &vol, cv::Mat& depthImage, cv::Mat& normalImage, std::vector<Vector3f> &hitPoints, std::vector<Vector3f> &hitNormals)
	{

		std::cout << "Executing Raycasting" << std::endl;
		clock_t begin = clock();

		//Eigen::Vector3f origin = _translation;
		double stepSizeVoxel = 1.5;

		double epsilon = 1e-3;
		Eigen::Vector3d origin = _translation.cast<double>();

		hitPoints.reserve(_depthImgHeight * _depthImgWidth);
		hitNormals.reserve(_depthImgHeight * _depthImgWidth);



		#pragma omp parallel for  //: size of hitPoints & hitNormals are not equal when enabled
		for (int y = 0; y < _depthImgHeight; y++)
		{
			//if (y % 40 == 0)
			//std::cout << "Processing row #" << y << std::endl;
			for (int x = 0; x < _depthImgWidth; x++)
			{
				//double rayX = ((double)x - _cX) / _fovX;
				//double rayY = ((double)y - _cY) / _fovY;
				//Vector3d ray(rayX, rayY, 1.0);
				double rayX = ((double)x - _cX) / _fovX;
				double rayY = ((double)y - _cY) / _fovY;
				Eigen::Vector3d ray(rayX, rayY, 1);

				ray = _rotation.cast<double>() * ray;
				cv::Vec3f normal;
				double length = 0;


				if (searchRay(vol, origin, ray, length, stepSizeVoxel, epsilon)) // Does the ray hit a zero crossing
				{
					/*depthImage.at<float>(y, x) = (float)length;

					if (depthImage.at<float>(y, x) == 0.0f)
						std::cout << "Invalid depth value at: (" << y << " , " << x << ") will exit now!\n";
					assert(depthImage.at<float>(v, u) != 0.0f);*/


					Eigen::Vector3d point = origin + ray * length;

					const double voxelSize = vol.cellSizeX_g;

					float valueXForward = vol.get_InterpVal(point + Eigen::Vector3d(voxelSize, 0, 0));
					float valueXBackward = vol.get_InterpVal(point + Eigen::Vector3d(-voxelSize, 0, 0));

					float valueYForward = vol.get_InterpVal(point + Eigen::Vector3d(0, voxelSize, 0));
					float valueYBackward = vol.get_InterpVal(point + Eigen::Vector3d(0, -voxelSize, 0));

					float valueZForward = vol.get_InterpVal(point + Eigen::Vector3d(0, 0, voxelSize));
					float valueZBackward = vol.get_InterpVal(point + Eigen::Vector3d(0, 0, -voxelSize));

					Vector3f normalVec(
						(valueXForward - valueXBackward) / 2,
						(valueYForward - valueYBackward) / 2,
						(valueZForward - valueZBackward) / 2
					);
					normalVec = _rotation * normalVec;
					normalVec.normalize();

					normal(0) = (float)normalVec.x();
					normal(1) = (float)normalVec.y();
					normal(2) = (float)normalVec.z();

					//normalImage.at<cv::Vec3f>(y, x) = normal;
					#pragma omp critical(pushhit)
					{
						hitPoints.push_back(point.cast<float>());
						hitNormals.push_back(normalVec);
					}
				}
				else
				{
					depthImage.at<float>(y, x) = -std::numeric_limits<float>::infinity();
					//depthImage.at<float>(v, u) = 0.0f;
					//normalImage.at<cv::Vec3f>(v, u) = normal;
					normalImage.at<cv::Vec3f>(y, x) = cv::Vec3f(-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity());
					
				}
			}
		}

		clock_t end = clock();
		double elapsedSecs = double(end - begin) / 1000;
		std::cout << "Raycasting finished in " << elapsedSecs << " seconds." << std::endl;
	}

private:
	float _fovX, _fovY, _cX, _cY;
	int _depthImgHeight, _depthImgWidth;
	
	Matrix3f _rotation;
	Vector3f _translation;

	BYTE * _colorRGBX;

};
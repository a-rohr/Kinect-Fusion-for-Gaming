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


	void castRays(Volume &vol,  bool showImg = true, int imgIndex = -1)
	{
		std::cout << "Executing Raycasting" << std::endl;
		clock_t begin = clock();

		cv::Mat outputImg(_depthImgHeight, _depthImgWidth, CV_8UC3, cv::Scalar(0, 0, 0));

		#pragma omp parallel for
		for (int y = 0; y < _depthImgHeight; y++)
		{
			for (int x = 0; x < _depthImgWidth; x++)
			{
				float rayX = ((float)x - _cX) / _fovX;
				float rayY = ((float)y - _cY) / _fovY;
				Vector3f ray(rayX, rayY, 1.0);

				ray = _rotation.transpose() * ray;
				cv::Vec3f normal;
				double length;

				Eigen::Vector3f origin = _translation;

				/*int depthAtCamera = 0;
				Vector3f pixel_0 = _rotationInv * Vector3f((x - _cX) / _fovX * depthAtCamera, (y - _cY) / _fovY * depthAtCamera, depthAtCamera) + _translationInv;
				int depthFrontOfCamera = 1;
				Vector3f pixel_1 = _rotationInv * Vector3f((x - _cX) / _fovX * depthFrontOfCamera, (y - _cY) / _fovY * depthFrontOfCamera, depthFrontOfCamera) + _translationInv;

				Vector3f rayDir = pixel_1 - pixel_0;
				rayDir.normalize();*/

				Vector3f rayDirReverse = Vector3f(ray.x(), ray.y(), -ray.z());

				float rayLength = 0;
				Vector3f g = (origin + rayLength * ray);
				Vector3i g_i = vol.getNodeFromWorld(g);
				if (vol.isVoxelIndexValid(g_i))
				{
					// Fast Raycasting
					/*float temporalRayLength = 0;
					Vector3f temp_g = g;
					Vector3i temp_g_i = g_i;
					double fastIncrement = 0.3;
					while (vol.isVoxelIndexValid(temp_g_i))
					{
						temporalRayLength += fastIncrement;
						temp_g = (origin + temporalRayLength * ray);
						temp_g_i = vol.getNodeFromWorld(temp_g);
						if (!vol.isVoxelIndexValid(temp_g_i) || vol.get(temp_g_i) < 0)
						{
							rayLength = temporalRayLength - fastIncrement;
							break;
						}
					}*/

					// Slow Raycasting
					double val_curr = vol.get(g_i);
					Vector3f g_prev;
					double val_prev;
					while (vol.isVoxelIndexValid(vol.getNodeFromWorld(g)))
					{
						rayLength += 0.01;
						g_prev = g;
						val_prev = val_curr;
						g = (origin + rayLength * ray);

						g_i = vol.getNodeFromWorld(g);
						if (!vol.isVoxelIndexValid(g_i))
							break;

						double val_curr = vol.get(g_i);
						if (val_prev > 0 && val_curr < 0)
						{
							
							//outputImg.at<uchar>(y, x) = (int)(((g - translation).squaredNorm() / 2.0) * 255)

							double n_x = vol.get(g_i + Vector3i(1, 0, 0)) - vol.get(g_i + Vector3i(-1, 0, 0));
							double n_y = vol.get(g_i + Vector3i(0, 1, 0)) - vol.get(g_i + Vector3i(0, -1, 0));
							double n_z = vol.get(g_i + Vector3i(0, 0, 1)) - vol.get(g_i + Vector3i(0, 0, -1));


							//double n_x = vol.get(g_i + Vector3i(1, 0, 0)) - vol.get(g_i + Vector3i(-1, 0, 0));
							//double n_y = vol.get(g_i + Vector3i(0, 1, 0)) - vol.get(g_i + Vector3i(0, -1, 0));
							//double n_z = vol.get(g_i + Vector3i(0, 0, 1)) - vol.get(g_i + Vector3i(0, 0, -1));
							Vector3f normal;
							//normal = _rotationInv * Vector3f(n_x, n_y, n_z);
							normal.normalize();

							//pointCloud.m_points.push_back(g);
							//pointCloud.m_normals.push_back(normal);
							
							int idxColor = 4 * (y * _depthImgWidth + x);

							//outputImg.at<uchar>(y, x) = (int)(std::max(normal.dot(rayDirReverse), 0.0f) * 255);
							int intensity = (int)(((g - _translation).squaredNorm() / 2.0) * 256) % 256;

							outputImg.at<cv::Vec3b>(cv::Point(x, y)) = cv::Vec3b(
								intensity,
								intensity,
								intensity
							);

							/*outputImg.at<cv::Vec3b>(cv::Point(x, y)) = cv::Vec3b(
								_colorRGBX[idxColor + 2],
								_colorRGBX[idxColor + 1],
								_colorRGBX[idxColor + 0]
							);*/

							break;
						}
					}
				}
			}
		}

		clock_t end = clock();
		double elapsedSecs = double(end - begin) / 1000;
		std::cout << "Raycasting finished in " << elapsedSecs << " seconds." << std::endl;

		if (showImg)
		{
			cv::Mat flippedOutImg = cv::Mat::zeros(_depthImgHeight, _depthImgWidth, 0);
			cv::flip(outputImg, flippedOutImg, 0); // flip around x-axis
			cv::imshow("Frame" + std::to_string(imgIndex), outputImg);


			/*cv::Mat colorImg = cv::Mat::zeros(depthImgHeight, depthImgWidth, CV_8UC3);
			for (int y = 0; y < depthImgHeight; y++)
			{
				for (int x = 0; x < depthImgWidth; x++)
				{
					cv::Vec3b color = colorImg.at<cv::Vec3b>(cv::Point(x, y));
					int idxColor = 4 * (y * depthImgWidth + x);
					color[0] = colorRGBX[idxColor + 2];
					color[1] = colorRGBX[idxColor + 1];
					color[2] = colorRGBX[idxColor + 0];
					colorImg.at<cv::Vec3b>(cv::Point(x, y)) = color;
				}
			}
			cv::imshow("Frame colored", colorImg);*/

		}
	}

	bool searchRay(Volume& vol, Eigen::Vector3d origin, Eigen::Vector3d ray, double& length, const double stepSizeVoxel, const double epsilon)
	{
		double voxelSize = vol.cellSizeX_g;

		Eigen::Vector3d point = origin + ray * length;
		float pointValue = vol.getValueAtPoint(point);
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
			pointValue = vol.getValueAtPoint(point);

			if (previousPointValue > 0.0 && pointValue < 0.0) { break; }
		}

		while (true)
		{
			double middleLength = (previousLength + length) / 2;
			float middleValue = vol.getValueAtPoint(origin + ray * middleLength);

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
		double stepSizeVoxel = .3;

		double epsilon = 1e-3;
		Eigen::Vector3d origin = _translation.cast<double>();

		hitPoints.reserve(_depthImgHeight * _depthImgWidth);
		hitNormals.reserve(_depthImgHeight * _depthImgWidth);



		//#pragma omp parallel for  : size of hitPoints & hitNormals are not equal when enabled
		for (int y = 0; y < _depthImgHeight; y++)
		{
			if (y % 40 == 0)
			std::cout << "Processing row #" << y << std::endl;
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

					float valueXForward = vol.getValueAtPoint(point + Eigen::Vector3d(voxelSize, 0, 0));
					float valueXBackward = vol.getValueAtPoint(point + Eigen::Vector3d(-voxelSize, 0, 0));

					float valueYForward = vol.getValueAtPoint(point + Eigen::Vector3d(0, voxelSize, 0));
					float valueYBackward = vol.getValueAtPoint(point + Eigen::Vector3d(0, -voxelSize, 0));

					float valueZForward = vol.getValueAtPoint(point + Eigen::Vector3d(0, 0, voxelSize));
					float valueZBackward = vol.getValueAtPoint(point + Eigen::Vector3d(0, 0, -voxelSize));

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
					hitPoints.push_back(point.cast<float>());
					hitNormals.push_back(normalVec);
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
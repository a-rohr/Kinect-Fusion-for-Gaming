#include <iostream>
#include <fstream>

#include "Eigen.h"
#include "VirtualSensor.h"
#include "PointCloud.h"
#include "ICPOptimizer.h"

#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "Volume.h"
#include <limits>
#include <time.h>
#include <fstream>

using namespace std;
using namespace cv;

/*bool searchRay(Volume& volume, Eigen::Vector3d origin, Eigen::Vector3d ray, double& length,
	const double stepSizeVoxel, const double epsilon)
{
	Eigen::Vector3d point = origin + ray * length;
	float pointValue = volume.get(point);
	float previousPointValue = pointValue;

	double stepSize = volume.dddx * stepSizeVoxel;
	double previousLength = length;

	while (true)
	{
		previousLength = length;
		length += stepSize;
		point = origin + ray * length;

		if (! volume.isVoxelIndexValid(point)) 
		{ 
			return false; 
		}

		previousPointValue = pointValue;
		pointValue = volume.get(point);

		if (previousPointValue > 0.0 && pointValue < 0.0) { break; }
	}

	while (true)
	{
		double middleLength = (previousLength + length) / 2;
		float middleValue = volume.get(origin + ray * middleLength);

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
}*/

void writeVolToFile(Volume &vol)
{
	ofstream outfile;
	outfile.open("tsdf.txt");
	for (int z = 0; z < (int)vol.getDimZ(); z++)
	{
		for (int y = 0; y < (int)vol.getDimY(); y++)
		{
			for (int x = 0; x < (int)vol.getDimX(); x++)
			{
				double v = vol.get(x, y, z);
				v = round(v * 100) / 100;
				outfile << v;
				if (x + 1 != vol.getDimX())
				{
					outfile << " ";
				}
			}
			outfile << endl;
		}
		outfile << endl;
	}

}

void pointCloudToImg(PointCloud &points, VirtualSensor &sensor, Matrix4f &cameraPose, int i = -1)
{
	int depthImgHeight = sensor.getDepthImageHeight();
	int depthImgWidth = sensor.getDepthImageWidth();
	cout << depthImgHeight << " x " << depthImgWidth << endl;
	Mat outputImg = Mat::zeros(depthImgHeight, depthImgWidth, CV_8U);

	auto intrinsics = sensor.getDepthIntrinsics();

	cout << "Executing TSDF" << endl;
	clock_t begin = clock();

	//auto nearestNeighborSearch = make_unique<NearestNeighborSearchFlann>();
	//nearestNeighborSearch->buildIndex(points.getPoints());

	unsigned int mc_res = 20;
	cout << "TSDF resolution: " << mc_res << endl;
	Volume vol(Vector3d(0, 0, 0), Vector3d(1, 1, 1), mc_res, mc_res, mc_res, 1);

	//auto nearestQueryResults = nearestNeighborSearch->queryMatches(vol.getAllVoxelPos());

	float truncation_coeff = 1;

	#pragma omp parallel for
	for (int x = 0; x < (int)vol.getDimX(); x++)
	{
		for (int y = 0; y < (int)vol.getDimY(); y++)
		{
			for (int z = 0; z < (int)vol.getDimZ(); z++)
			{

				Vector3f v_g = vol.pos(x, y, z).cast<float>();

				Vector3f v = cameraPose.block<3, 3>(0, 0) * v_g + cameraPose.block<3, 1>(0, 3);

				Vector3f p_homo = intrinsics * v;
				int pixel_u = (int)(p_homo.x() / p_homo.z());
				int pixel_v = (int)(p_homo.y() / p_homo.z());

				if (pixel_u > 0 && pixel_u < depthImgWidth && pixel_v > 0 && pixel_v < depthImgHeight && v_g.z() < 2.1  && v_g.z() > -0.1)
				{
					unsigned int depthIdx = pixel_v * depthImgWidth + pixel_u; // linearized index

					float sdf;
					float measuredDepth = sensor.getDepth()[depthIdx];
					if (measuredDepth > 0 && measuredDepth < 10)
					{
						sdf = sensor.getDepth()[depthIdx] - (cameraPose.block<3, 1>(0, 3) - v_g).squaredNorm();
					}
					else
					{
						sdf = 1;
					}
					
					float tsdf;
					if (sdf > 0)
					{
						tsdf = MIN(1, sdf / truncation_coeff);
					}
					else
					{
						tsdf = MAX(-1, sdf / truncation_coeff);
					}

					double w_prev = vol.getWeight(x, y, z);
					double w = MIN(5, w_prev + 1);

					double tsdf_prev = vol.get(x, y, z);

					double tsdf_avg = (tsdf_prev * w_prev + tsdf * w) / (w_prev + w);

					if (tsdf_avg > 1 || tsdf_avg < -1)
					{
						cout << "lala" << endl;
					}
					
					//cout << x << " " << y << z << tsdf_avg << endl;
					vol.set(x, y, z, tsdf_avg);
					vol.setWeight(x, y, z, w);
				}
				else
				{
					// Front of the object + zero weight
					vol.set(x, y, z, 1);
					vol.setWeight(x, y, z, 0);
				}

				/*Match nearestQueryResult = nearestQueryResults[x * vol.getDimX() + y * vol.getDimY() + z];

				if (nearestQueryResult.weight > 0.9)	// can be 0 or 1 but float
				{
					float distance = (points.getPoints()[nearestQueryResult.idx] - pos).squaredNorm();
					vol.set(x, y, z, distance);
				}
				else
				{
					vol.set(x, y, z, 99000);
				}*/
			}
		}
	}

	clock_t end = clock();
	double elapsedSecs = double(end - begin) / CLOCKS_PER_SEC;
	cout << "TSDF finished in " << elapsedSecs << " seconds." << endl;

	//writeVolToFile(vol);


	if (false)
	{
		// Get depth intrinsics.
		float fovX = intrinsics(0, 0);
		float fovY = intrinsics(1, 1);
		float cX = intrinsics(0, 2);
		float cY = intrinsics(1, 2);

		double fx = intrinsics(0, 0)*depthImgWidth;
		double fy = intrinsics(1, 1)*depthImgHeight;
		double cx = intrinsics(0, 2)*depthImgWidth - 0.5;
		double cy = intrinsics(1, 2)*depthImgHeight - 0.5;


		// Compute inverse depth extrinsics.
		Matrix4f depthExtrinsicsInv = cameraPose.inverse();
		Matrix3f rotationInv = depthExtrinsicsInv.block(0, 0, 3, 3);
		Vector3f translationInv = depthExtrinsicsInv.block(0, 3, 3, 1);
		Vector3f translation = cameraPose.block<3, 1>(0, 3);


		for (int y = 0; y < depthImgHeight; y++)
		{
			for (int x = 0; x < depthImgWidth; x++)
			{
				/*double rayX = ((double)x - cx) / fx;
				double rayY = ((double)y - cy) / fy;
				Eigen::Vector3f ray(rayX, rayY, 1);
				//ray.normalize();

				ray = cameraPose.block<3, 3>(0, 0) * ray + cameraPose.block<3, 1>(0, 3);
				cv::Vec3f normal;
				double length;*/


				int depthAtCamera = 0;
				Vector3f pixel_0 = rotationInv * Vector3f((x - cX) / fovX * depthAtCamera, (y - cY) / fovY * depthAtCamera, depthAtCamera) + translationInv;
				int depthFrontOfCamera = 1;
				Vector3f pixel_1 = rotationInv * Vector3f((x - cX) / fovX * depthFrontOfCamera, (y - cY) / fovY * depthFrontOfCamera, depthFrontOfCamera) + translationInv;

				Vector3i rayStart = vol.getNodeFromWorld(pixel_0);
				Vector3i rayNext = vol.getNodeFromWorld(pixel_1);
				Vector3f rayDir = (rayNext - rayStart).cast<float>();
				rayDir.normalize();

				float rayLength = 0;
				Vector3i g = (rayStart.cast<float>() + rayLength * rayDir).cast<int>();
				Vector3i g_prev;
				while (vol.isVoxelIndexValid(g))
				{
					rayLength++;
					g_prev = g;
					g = (rayStart.cast<float>() + rayLength * rayDir).cast<int>();

					double val_curr = vol.get(g);
					double val_prev = vol.get(g_prev);
					if (val_prev > 0 && val_curr < 0)
					{
						Vector3i p = g;
						outputImg.at<uchar>(y, x) = 255;
						break;
					}

				}
			}
		}

		/*for (auto point : points.getPoints())
		//for (int i = 0; i < 100; i++)
		{
			Vector3f uvw = intrinsics * (cameraPose.block<3, 3>(0, 0) * point + cameraPose.block<3, 1>(3, 0));
			float u = uvw.x() / uvw.z();
			float v = uvw.y() / uvw.z();

			//cout << u << "   " << v << endl;

			mat.at<uchar>(v, u) = 255;
		}*/

		imshow("Frame" + to_string(i), outputImg);
		waitKey(0);
	}

	
}


int buildSensor(VirtualSensor &sensor)
{
	string filenameIn = PROJECT_DIR + std::string("/data/rgbd_dataset_freiburg1_xyz/");
	string filenameBaseOut = PROJECT_DIR + std::string("/results/mesh_");

	// Load video
	std::cout << "Initialize virtual sensor..." << std::endl;
	if (!sensor.init(filenameIn)) {
		std::cout << "Failed to initialize the sensor!\nCheck file path!" << std::endl;
		return false;
	}
	return true;
}

/*
int main()
{
	VirtualSensor sensor;
	if (!buildSensor(sensor))
	{
		return -1;
	}

	sensor.processNextFrame();
	PointCloud target{ sensor.getDepth(), sensor.getDepthIntrinsics(), sensor.getDepthExtrinsics(), sensor.getDepthImageWidth(), sensor.getDepthImageHeight() };
	
	// Setup the optimizer.
	ICPOptimizer optimizer;
	optimizer.setMatchingMaxDistance(0.1f);
	optimizer.usePointToPlaneConstraints(true);
	optimizer.setNbOfIterations(10);

	// We store the estimated camera poses.
	std::vector<Matrix4f> estimatedPoses;
	Matrix4f currentCameraToWorld = Matrix4f::Identity();
	estimatedPoses.push_back(currentCameraToWorld.inverse());

	vector<PointCloud> listPointClouds;
	listPointClouds.push_back(target);
	//PointCloud allPointClouds(listPointClouds);

	Matrix4f currCamPose;
	currCamPose << 1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1;

	pointCloudToImg(target, sensor, currCamPose);
	
	int i = 1;
	const int iMax = 1;
	std::string filenameBaseOut = PROJECT_DIR + std::string("/results/mesh_");
	while (sensor.processNextFrame() && i <= iMax) {
		cout << "Target has " << target.getPoints().size() << " points!" << endl;
		// Estimate the current camera pose from source to target mesh with ICP optimization.
		// We downsample the source image to speed up the correspondence matching.
		PointCloud source{ sensor.getDepth(), sensor.getDepthIntrinsics(), sensor.getDepthExtrinsics(), sensor.getDepthImageWidth(), sensor.getDepthImageHeight(), 8 };
		currentCameraToWorld = optimizer.estimatePose(source, target, currentCameraToWorld);

		// Invert the transformation matrix to get the current camera pose.
		Matrix4f currentCameraPose = currentCameraToWorld.inverse();
		std::cout << "Current camera pose: " << std::endl << currentCameraPose << std::endl;
		estimatedPoses.push_back(currentCameraPose);

		listPointClouds.push_back(source);
		target = PointCloud(listPointClouds);
		//allPointClouds = PointCloud(listPointClouds);

		pointCloudToImg(target, sensor, currentCameraPose, i);

		if (i % 5 == 0) {
			// We write out the mesh to file for debugging.
			SimpleMesh currentDepthMesh{ sensor, currentCameraPose, 0.1f };
			SimpleMesh currentCameraMesh = SimpleMesh::camera(currentCameraPose, 0.0015f);
			SimpleMesh resultingMesh = SimpleMesh::joinMeshes(currentDepthMesh, currentCameraMesh, Matrix4f::Identity());

			std::stringstream ss;
			ss << filenameBaseOut << sensor.getCurrentFrameCnt() << ".off";
			if (!resultingMesh.writeMesh(ss.str())) {
				std::cout << "Failed to write mesh!\nCheck file path!" << std::endl;
				return -1;
			}
		}

		i++;
	}	

	waitKey(0);
	cout << "Hello" << endl;
}*/
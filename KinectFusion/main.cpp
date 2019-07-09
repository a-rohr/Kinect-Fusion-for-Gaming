#include <iostream>
#include <fstream>

#include "Eigen.h"
#include "VirtualSensor.h"
#include "SimpleMesh.h"
#include "ICPOptimizer.h"
#include "ProcrustesAligner.h"
#include "PointCloud.h"
#include "Volume.h"
#include <fstream>
#include <time.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "MarchingCubes.h"

#include "Raycaster.h"


void saveVolume(Volume &vol, std::string filenameOut)
{
	clock_t begin = clock();
	SimpleMesh mesh;
	#pragma omp parallel for
	for (int x = 0; x < vol.getDimX() - 1; x++)
	{
		//std::cerr << "Marching Cubes on slice " << x << " of " << vol.getDimX() << std::endl;

		for (unsigned int y = 0; y < vol.getDimY() - 1; y++)
		{
			for (unsigned int z = 0; z < vol.getDimZ() - 1; z++)
			{
				ProcessVolumeCell(&vol, (unsigned int)x, y, z, 0.00f, &mesh);
			}
		}
	}

	// write mesh to file
	if (!mesh.writeMesh(filenameOut))
	{
		std::cout << "ERROR: unable to write output file!" << std::endl;
	}

	clock_t end = clock();
	double elapsedSecs = double(end - begin) / 1000;
	std::cout << "saveVolume finished in " << elapsedSecs << " seconds." << std::endl;

}

void saveVolume(Volume &vol, std::string filenameOut, Matrix4f currentCameraPose, Matrix3f intrinsics, unsigned int depthImgHeight, unsigned int depthImgWidth)
{
	clock_t begin = clock();
	SimpleMesh mesh;

	#pragma omp parallel for
	for (int x = 0; x < vol.getDimX() - 1; x++)
	{
		//std::cerr << "Marching Cubes on slice " << x << " of " << vol.getDimX() << std::endl;

		for (unsigned int y = 0; y < vol.getDimY() - 1; y++)
		{
			for (unsigned int z = 0; z < vol.getDimZ() - 1; z++)
			{
				ProcessVolumeCell(&vol, (unsigned int)x, y, z, 0.00f, &mesh);
			}
		}
	}

	SimpleMesh currentCameraMesh = SimpleMesh::camera(currentCameraPose, 0.0015f);
	SimpleMesh resultingMesh = SimpleMesh::joinMeshes(mesh, currentCameraMesh, Matrix4f::Identity());



	// RAY SECTION

	Matrix3f _rotation = currentCameraPose.block(0, 0, 3, 3).transpose();
	Vector3f _translation = currentCameraPose.block<3, 1>(0, 3);
	_translation = -_rotation * _translation;


	float _fovx = intrinsics(0, 0);
	float _fovy = intrinsics(1, 1);
	float _cx = intrinsics(0, 2);
	float _cy = intrinsics(1, 2);


	std::vector<Vector2i> rayPixels;
	rayPixels.push_back(Vector2i(0, 0));
	rayPixels.push_back(Vector2i(depthImgWidth - 1, 0));
	rayPixels.push_back(Vector2i(0, depthImgHeight - 1));
	rayPixels.push_back(Vector2i(depthImgWidth - 1, depthImgHeight - 1));
	rayPixels.push_back(Vector2i(400, 400));


	Eigen::Vector3f origin = _translation;

	for (int i = 0; i < rayPixels.size(); i++)
	{
		Vector2i pixel = rayPixels[i];
		double rayX = ((double)pixel[0] - _cx) / _fovx;
		double rayY = ((double)pixel[1] - _cy) / _fovy;
		Eigen::Vector3d ray(rayX, rayY, 1);

		ray = _rotation.cast<double>() * ray;
		cv::Vec3f normal;
		double length = 0;

		float sign;
		if (i % 2 == 0)
		{
			sign = 1.0;
		}
		else
		{
			sign = 1.0;
		}
		Vector3f end = origin + sign * ray.cast<float>();
		SimpleMesh lineMesh(origin, end, 100, 0.1);

		resultingMesh = SimpleMesh::joinMeshes(resultingMesh, lineMesh, Matrix4f::Identity());
	}
	
	// write mesh to file
	if (!resultingMesh.writeMesh(filenameOut))
	{
		std::cout << "ERROR: unable to write output file!" << std::endl;
	}

	clock_t end = clock();
	double elapsedSecs = double(end - begin) / 1000;
	std::cout << "saveVolume finished in " << elapsedSecs << " seconds." << std::endl;
}


void localTSDF(Volume &vol, VirtualSensor &sensor, Matrix4f &cameraPose, float maxZ)
{
	std::cout << "Executing TSDF" << std::endl;
	clock_t begin = clock();

	vol.clean();

	int depthImgHeight = sensor.getDepthImageHeight();
	int depthImgWidth = sensor.getDepthImageWidth();

	auto intrinsics = sensor.getDepthIntrinsics();
	float truncation_coeff = 0.04;
	//float truncation_coeff = 0.025;	

	#pragma omp parallel for
	for (int x = 0; x < (int)vol.getDimX(); x++)
	{
		for (int y = 0; y < (int)vol.getDimY(); y++)
		{
			for (int z = 0; z < (int)vol.getDimZ(); z++)
			{
				Eigen::Vector4f worldPointHomo;
				worldPointHomo << vol.pos(x, y, z).cast<float>(), 1;

				//transform world point to cam coords
				Eigen::Vector3f cameraPoint = (cameraPose * worldPointHomo).head(3);

				//point behind camera or too far away
				if (cameraPoint.z() <= 0 || cameraPoint.z() > maxZ) {
					continue;
				}

				//Project point to pixel in depthmap
				Eigen::Vector3f pixPointHomo = intrinsics * cameraPoint;
				Eigen::Vector2i pixPoint;
				pixPoint << pixPointHomo.x() / pixPointHomo.z(), pixPointHomo.y() / pixPointHomo.z();

				if (pixPoint[0] < 0 || pixPoint[0] >= depthImgWidth || pixPoint[1] < 0 || pixPoint[1] >= depthImgHeight)
				{
					continue;
				}

				int depthRow = pixPoint.y();//_camResolution.y()-1 - pixPoint.y(); //row0 is at top. y0 is at bottom.
				int depthCol = pixPoint.x();
				double pointDepth = cameraPoint.z();
				unsigned int depthIdx = pixPoint[1] * depthImgWidth + pixPoint[0]; // linearized index
				float TSDF_val = (float)sensor.getDepth()[depthIdx] - (float)pointDepth;

				//truncate SDF value
				if (TSDF_val >= -truncation_coeff) {
					TSDF_val = std::min(1.0f, fabsf(TSDF_val) / truncation_coeff)*copysignf(1.0f, TSDF_val);
				}
				else { //too far behind obstacle
					continue;
				}

				vol.set(x, y, z, TSDF_val);
				vol.setWeight(x, y, z, 1.0);
			}
		}
	}

	clock_t end = clock();
	double elapsedSecs = double(end - begin) / 1000;
	std::cout << "TSDF finished in " << elapsedSecs << " seconds." << std::endl;
}


void fuseFrames(Volume &global, Volume &current)
{
	for (int x = 0; x < (int)global.getDimX(); x++)
	{
		for (int y = 0; y < (int)global.getDimY(); y++)
		{
			for (int z = 0; z < (int)global.getDimZ(); z++)
			{
				double globalVal = global.get(x, y, z);
				double globalWeight = global.getWeight(x, y, z);

				double currentVal = current.get(x, y, z);
				double currentWeight = current.getWeight(x, y, z);

				double newWeight = globalWeight + currentWeight;
				double newVal;
				if (newWeight != 0)
				{
					newVal = (globalVal * globalWeight + currentVal * currentWeight) / newWeight;
				}
				else
				{
					newVal = 1;
				}

				global.set(x, y, z, newVal);
				if (newWeight > 5)
				{
					newWeight = 5;
				}
				global.setWeight(x, y, z, newWeight);
			}
		}
	}
}


int buildSensor(VirtualSensor &sensor)
{
	std::string filenameIn = PROJECT_DIR + std::string("/data/rgbd_dataset_freiburg1_xyz/");

	// Load video
	std::cout << "Initialize virtual sensor..." << std::endl;
	if (!sensor.init(filenameIn)) {
		std::cout << "Failed to initialize the sensor!\nCheck file path!" << std::endl;
		return false;
	}
	return true;
}


int executeKinect(float minx, float miny, float minz, float maxx, float maxy, float maxz)
{
	VirtualSensor sensor;
	if (!buildSensor(sensor))
	{
		return -1;
	}

	unsigned int mc_res = 256;

	Vector3d worldStart(minx, miny, minz);
	Vector3d worldEnd(maxx, maxy, maxz);
	Volume globalVolume(worldStart, worldEnd, mc_res, mc_res, mc_res);
	Volume localVolume(worldStart, worldEnd, mc_res, mc_res, mc_res);

	sensor.processNextFrame();
	PointCloud target{ sensor.getDepth(), sensor.getDepthIntrinsics(), sensor.getDepthExtrinsics(), sensor.getDepthImageWidth(), sensor.getDepthImageHeight() };

	// Setup the optimizer.
	ICPOptimizer optimizer;
	optimizer.setMatchingMaxDistance(0.025f);
	optimizer.usePointToPlaneConstraints(true);
	optimizer.setNbOfIterations(10);


	Matrix4f currentCameraPose;// = Matrix4f::Identity();
	
	// 1.3434 0.6271 1.6606
	//currentCameraPose(0, 3) = 1.5;
	//currentCameraPose(1, 3) = 1.5;
	//currentCameraPose(2, 3) = 1.5;
	currentCameraPose = sensor.getTrajectory();
	Matrix4f currentCameraToWorld = currentCameraPose.inverse(); 
	std::cout << "Initial camera pose: " << std::endl << currentCameraPose << std::endl;

	localTSDF(globalVolume, sensor, currentCameraPose, worldEnd.z());

	
	std::string filenameOut = PROJECT_DIR + std::string("/results/result_global_0.off");
	//saveVolume(globalVolume, filenameOut, currentCameraPose, sensor.getDepthIntrinsics(), sensor.getDepthImageHeight(), sensor.getDepthImageWidth());

	
	cv::Mat depthImage = cv::Mat::zeros(sensor.getDepthImageHeight(), sensor.getDepthImageWidth(), CV_32F);
	cv::Mat normalMap = cv::Mat::zeros(sensor.getDepthImageHeight(), sensor.getDepthImageWidth(), CV_32FC3);


	int i = 1;
	const int iMax = 10;
	while (sensor.processNextFrame() && i <= iMax) {
		Raycaster raycaster(currentCameraPose, sensor.getDepthIntrinsics(), sensor.getDepthImageHeight(), sensor.getDepthImageWidth(), sensor.getColorRGBX());

		std::vector<Vector3f> raycastHitPoints;
		std::vector<Vector3f> raycastHitNormals;
		//raycaster.castRays2(globalVolume, depthImage, normalMap, raycastHitPoints, raycastHitNormals);
		raycaster.castRays3(globalVolume, depthImage, normalMap, raycastHitPoints, raycastHitNormals, 0.04);
		std::cout << "Hit Points size: " << raycastHitPoints.size() << std::endl;
		std::cout << "Hit Normals size: " << raycastHitNormals.size() << std::endl;

		target.m_points = raycastHitPoints;
		target.m_normals = raycastHitNormals;


		std::cout << "Target has " << target.getPoints().size() << " points!" << std::endl;
		PointCloud source{ sensor.getDepth(), sensor.getDepthIntrinsics(), sensor.getDepthExtrinsics(), sensor.getDepthImageWidth(), sensor.getDepthImageHeight(), 8 };
		//currentCameraPose = sensor.getTrajectory();
		//currentCameraToWorld = currentCameraPose.inverse();
		currentCameraToWorld = optimizer.estimatePose(source, target, currentCameraToWorld);

		// Invert the transformation matrix to get the current camera pose.
		currentCameraPose = currentCameraToWorld.inverse();
		std::cout << "Current camera pose: " << std::endl << currentCameraPose << std::endl;

		localTSDF(localVolume, sensor, currentCameraPose, worldEnd.z());
		fuseFrames(globalVolume, localVolume);

		
		if (i % 10 == 0)
		{
			filenameOut = PROJECT_DIR + std::string("/results/result_local_" + std::to_string(i) + ".off");
			saveVolume(localVolume, filenameOut, currentCameraPose, sensor.getDepthIntrinsics(), sensor.getDepthImageHeight(), sensor.getDepthImageWidth());

			filenameOut = PROJECT_DIR + std::string("/results/result_global_" + std::to_string(i) + ".off");
			saveVolume(globalVolume, filenameOut, currentCameraPose, sensor.getDepthIntrinsics(), sensor.getDepthImageHeight(), sensor.getDepthImageWidth());
		}
		i++;
	}

	std::cout << "Ended" << std::endl;
	return 0;

}


int main()
{
	
	float minx, miny, minz;
	float maxx, maxy, maxz;
	while (false)
	{
		
		std::cout << "Min X: ";
		std::cin >> minx;

		std::cout << "Min Y: ";
		std::cin >> miny;

		std::cout << "Min Z: ";
		std::cin >> minz;


		std::cout << "Max X: ";
		std::cin >> maxx;

		std::cout << "Max Y: ";
		std::cin >> maxy;

		std::cout << "Max Z: ";
		std::cin >> maxz;

		executeKinect(minx, miny, minz, maxx, maxy, maxz);
	}
	
	executeKinect(-2, -2, -2, 2, 2, 2);
	
	
	
	
	
}
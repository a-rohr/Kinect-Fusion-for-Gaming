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

void writeVolToFile(Volume &vol, bool onlyPrint)
{
	std::ofstream outfile;
	if (!onlyPrint)
		outfile.open("tsdf.txt");
	int maxX = -999;
	int maxY = -999;
	int maxZ = -999;
	int minX = 999;
	int minY = 999;
	int minZ = 999;


	for (int z = 0; z < (int)vol.getDimZ(); z++)
	{
		for (int y = 0; y < (int)vol.getDimY(); y++)
		{
			for (int x = 0; x < (int)vol.getDimX(); x++)
			{
				double v = vol.get(x, y, z);
				Vector3d pos = vol.pos(x, y, z);
				if (v >= -0.1 && v <= 0.1)
				{
					//std::cout << "HOHOHO" << std::endl;
					//if (vol.getValueAtPoint(pos) < 1)
					//{
						//uint posIdx = vol.getPosFromTuple(x, y, z);

						v = round(v * 100) / 100;
						if (!onlyPrint)
						{
							outfile << x << " " << y << " " << z << " ";
							outfile << vol.posX(x) << " " << vol.posY(y) << " " << vol.posZ(z) << " " << v << " " << vol.getValueAtPoint(vol.pos(x, y, z)) << std::endl;
						}


						if (x > maxX)
							maxX = x;

						if (y > maxY)
							maxY = y;

						if (z > maxZ)
							maxZ = z;

						if (x < minX)
							minX = x;

						if (y < minY)
							minY = y;

						if (z < minZ)
							minZ = z;
					//}

				}
				
				/*outfile << v;
				if (x + 1 != vol.getDimX())
				{
					outfile << " ";
				}*/

				
			}
		}
	}

	std::cout << "x min: " << minX << " max: " << maxX << std::endl;
	std::cout << "y min: " << minY << " max: " << maxY << std::endl;
	std::cout << "z min: " << minZ << " max: " << maxZ << std::endl;
}

void writeDepthToFile(VirtualSensor &sensor)
{
	std::ofstream outfile;
	outfile.open("sensor_depth.txt");
	for (int y = 0; y < (int)sensor.getDepthImageHeight(); y++)
	{
		for (int x = 0; x < (int)sensor.getDepthImageWidth(); x++)
		{
			outfile << sensor.getDepth()[y * (int)sensor.getDepthImageHeight() + x];
			outfile << " ";
		}
		outfile << std::endl;
	}
}

void writeDepthImgToFile(cv::Mat &img)
{
	std::ofstream outfile;
	outfile.open("depth_img.txt");
	for (int y = 0; y < img.rows; y++)
	{
		for (int x = 0; x < img.cols; x++)
		{
			outfile << img.at<float>(cv::Point(x, y));
			outfile << " ";
		}
		outfile << std::endl;
	}
}

void saveVolume(Volume &vol, std::string filenameOut)
{
	SimpleMesh mesh;
	//#pragma omp parallel for
	for (int x = 0; x < vol.getDimX() - 1; x++)
	{
		if (x % 16 == 0)
		{
			std::cerr << "Marching Cubes on slice " << x << " of " << vol.getDimX() << std::endl;
		}

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

}

void saveVolume(Volume &vol, std::string filenameOut, Matrix4f currentCameraPose, Matrix3f intrinsics, unsigned int depthImgHeight, unsigned int depthImgWidth)
{
	SimpleMesh mesh;

	for (unsigned int x = 0; x < vol.getDimX() - 1; x++)
	{
		if (x % 16 == 0)
		{
			std::cerr << "Marching Cubes on slice " << x << " of " << vol.getDimX() << std::endl;
		}

		for (unsigned int y = 0; y < vol.getDimY() - 1; y++)
		{
			for (unsigned int z = 0; z < vol.getDimZ() - 1; z++)
			{
				ProcessVolumeCell(&vol, x, y, z, 0.00f, &mesh);
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

}



void localTSDF(Volume &vol, VirtualSensor &sensor, Matrix4f &cameraPose, float maxZ)
{
	std::cout << "Executing TSDF" << std::endl;
	clock_t begin = clock();

	vol.clean();

	int depthImgHeight = sensor.getDepthImageHeight();
	int depthImgWidth = sensor.getDepthImageWidth();

	auto intrinsics = sensor.getDepthIntrinsics();
	float truncation_coeff = 0.025;

	Eigen::Vector3f cameraPoint;
	Eigen::Vector3f pixPointHomo;

	
	for (int x = 0; x < (int)vol.getDimX(); x++)
	{
		for (int y = 0; y < (int)vol.getDimY(); y++)
		{
			for (int z = 0; z < (int)vol.getDimZ(); z++)
			{
				Eigen::Vector4f worldPointHomo;
				worldPointHomo << vol.pos(x, y, z).cast<float>(), 1;

				//transform world point to cam coords
				cameraPoint = (cameraPose * worldPointHomo).head(3);

				//point behind camera or too far away
				if (cameraPoint.z() <= 0 || cameraPoint.z() > maxZ) {
					continue;
				}

				//Project point to pixel in depthmap
				pixPointHomo = intrinsics * cameraPoint;
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

void processDepth(Volume &vol, PointCloud &points, PointCloud &nextPoints, VirtualSensor &sensor, Matrix4f &cameraPose, int i = -1)
{
	int depthImgHeight = sensor.getDepthImageHeight();
	int depthImgWidth = sensor.getDepthImageWidth();
	cv::Mat outputImg = cv::Mat::zeros(depthImgHeight, depthImgWidth, 0);

	auto intrinsics = sensor.getDepthIntrinsics();

	std::cout << "Executing TSDF" << std::endl;
	clock_t begin = clock();

	//localTSDF(vol, sensor, cameraPose);

	/*float truncation_coeff = 0.1;

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
						sdf = 100;
					}

					float tsdf;
					if (sdf > 0)
					{
						tsdf = std::min(1.0f, sdf / truncation_coeff);
					}
					else
					{
						tsdf = std::max(-1.0f, sdf / truncation_coeff);
					}

					double w_prev = vol.getWeight(x, y, z);
					double w = std::min(5.0, w_prev + 1);

					double tsdf_prev = vol.get(x, y, z);
					double tsdf_avg = (tsdf_prev * w_prev + tsdf * w) / (w_prev + w);

					assert(tsdf_avg <= 1);
					assert(tsdf_avg >= -1);

					vol.set(x, y, z, tsdf_avg);
					vol.setWeight(x, y, z, w);
					//vol.setColor(x, y, z, sensor.getColorRGBX()[])
				}
				else
				{
					// Front of the object + zero weight
					vol.set(x, y, z, 1);
					vol.setWeight(x, y, z, 0);
				}
			}
		}
	}*/

	//writeVolToFile(vol);
	/*Eigen::AngleAxisd rollAngle(0, Eigen::Vector3d::UnitZ());
	Eigen::AngleAxisd yawAngle(-3, Eigen::Vector3d::UnitY());
	Eigen::AngleAxisd pitchAngle(0, Eigen::Vector3d::UnitX());

	Eigen::Quaternion<double> q = rollAngle * yawAngle * pitchAngle;
	Eigen::Matrix3d rotationMatrix = q.matrix();

	Matrix4f viewCamPose;
	viewCamPose << rotationMatrix(0, 0), rotationMatrix(0, 1), rotationMatrix(0, 2), -0.8,
		rotationMatrix(1, 0), rotationMatrix(1, 1), rotationMatrix(1, 2), 0.5,
		rotationMatrix(2, 0), rotationMatrix(2, 1), rotationMatrix(2, 2), 0,
		0, 0, 0, 1;*/

	Raycaster raycaster(cameraPose, intrinsics, depthImgHeight, depthImgWidth, sensor.getColorRGBX());
	//raycaster.castRays(vol, true, i);
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

void lala()
{
	unsigned int mc_res = 128;
	Volume vol(Vector3d(0, 0, 0), Vector3d(0.5, 0.5, 0.5), mc_res, mc_res, mc_res);

	//auto nearestQueryResults = nearestNeighborSearch->queryMatches(vol.getAllVoxelPos());

	float truncation_coeff = 1;

	#pragma omp parallel for
	for (int x = 0; x < (int)vol.getDimX(); x++)
	{
		for (int y = 0; y < (int)vol.getDimY(); y++)
		{
			for (int z = 0; z < (int)vol.getDimZ(); z++)
			{
				if (80 <= z && z < 110) {
					if (60 <= x && x < 70) {
						if (60 <= y && y < 70) {
							vol.set(x, y, z, -0.1);
						}
						else {
							vol.set(x, y, z, 0.1);
						}
					}
					else {
						vol.set(x, y, z, 0.1);
					}
				}
				else {
					vol.set(x, y, z, 0.1);
				}
			}
		}
	}

	Volume vol2(Vector3d(-2, -2, -2), Vector3d(2, 2, 2), mc_res, mc_res, mc_res);
	#pragma omp parallel for
	for (int x = 0; x < (int)vol2.getDimX(); x++)
	{
		for (int y = 0; y < (int)vol2.getDimY(); y++)
		{
			for (int z = 0; z < (int)vol2.getDimZ(); z++)
			{
				bool assigned = false;
				if (50 <= z && z < 60) {
					if (50 <= x && x < 80) {
						if (60 <= y && y < 70) {
							vol2.set(x, y, z, -0.1);
							assigned = true;
						}
					}
				}
				else if (80 <= z && z < 110) {
					if (60 <= x && x < 70) {
						if (60 <= y && y < 70) {
							vol2.set(x, y, z, -0.1);
							assigned = true;
						}
					}
				}
				else {
					
				}
				if (!assigned)
				{
					vol2.set(x, y, z, 0.1);
				}
			}
		}
	}

	Matrix4f currCamPose;
	currCamPose << 1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1;

	VirtualSensor sensor;
	if (!buildSensor(sensor))
	{
		std::cout << "Failed!" << std::endl;
	}

	//localTSDF(vol, target, sensor, initCamPose);
	//processDepth(volume, target, nextPoints, sensor, currCamPose);
	//Raycaster raycasterFirst(initCamPose, sensor.getDepthIntrinsics(), sensor.getDepthImageHeight(), sensor.getDepthImageWidth(), sensor.getColorRGBX());
	//raycasterFirst.castRays(globalVolume, true, 100);

	std::string filenameOut = PROJECT_DIR + std::string("/results/result_first.off");
	saveVolume(vol, filenameOut);

	std::string filenameOut2 = PROJECT_DIR + std::string("/results/result_second.off");
	saveVolume(vol2, filenameOut2);


	fuseFrames(vol, vol2);
	std::string filenameOut3 = PROJECT_DIR + std::string("/results/result_combined.off");
	saveVolume(vol, filenameOut3);

	//Raycaster raycaster(currCamPose, sensor.getDepthIntrinsics(), sensor.getDepthImageHeight(), sensor.getDepthImageWidth(), sensor.getColorRGBX());
	//PointCloud nextPts;
	//raycaster.castRays(vol, true);
}

void putBBrect(float minx, float miny, float minz, float maxx, float maxy, float maxz)
{
	std::string filenameOut = PROJECT_DIR + std::string("/results/bb_rect.off");

	SimpleMesh mesh0(Vector3f(minx, miny, minz), Vector3f(maxx, miny, minz), 1000, 0.1, Vector4uc(255, 0, 0, 255));
	SimpleMesh mesh1(Vector3f(minx, miny, minz), Vector3f(minx, maxy, minz), 1000, 0.1, Vector4uc(0, 255, 0, 255));
	SimpleMesh mesh2(Vector3f(minx, miny, minz), Vector3f(minx, miny, maxz), 1000, 0.1, Vector4uc(0, 0, 255, 255));
	//SimpleMesh mesh3(Vector3f(maxx, miny, minz), Vector3f(maxx, maxy, minz), 1000, 0.1);
	
	/*SimpleMesh mesh3(Vector3f(minx, maxy, minz), Vector3f(maxx, maxy, minz), 1000, 0.1);
	SimpleMesh mesh5(Vector3f(maxx, miny, minz), Vector3f(maxx, miny, maxz), 1000, 0.1);
	SimpleMesh mesh6(Vector3f(maxx, maxy, minz), Vector3f(maxx, maxy, maxz), 1000, 0.1);
	SimpleMesh mesh7(Vector3f(maxx, miny, maxz), Vector3f(maxx, maxy, maxz), 1000, 0.1);
	SimpleMesh mesh8(Vector3f(minx, maxy, minz), Vector3f(minx, maxy, maxz), 1000, 0.1);
	SimpleMesh mesh9(Vector3f(minx, maxy, maxz), Vector3f(minx, miny, maxz), 1000, 0.1);
	SimpleMesh mesh10(Vector3f(minx, miny, maxz), Vector3f(maxx, miny, maxz), 1000, 0.1);
	SimpleMesh mesh11(Vector3f(minx, maxy, maxz), Vector3f(maxx, maxy, maxz), 1000, 0.1);*/


	SimpleMesh joinedMesh = SimpleMesh::joinMeshes(mesh0, mesh1, Matrix4f::Identity());
	SimpleMesh joinedMesh2 = SimpleMesh::joinMeshes(joinedMesh, mesh2, Matrix4f::Identity());
	//SimpleMesh joinedMesh3 = SimpleMesh::joinMeshes(joinedMesh2, mesh3, Matrix4f::Identity());
	/*joinedMesh = SimpleMesh::joinMeshes(joinedMesh, mesh4, Matrix4f::Identity());
	joinedMesh = SimpleMesh::joinMeshes(joinedMesh, mesh5, Matrix4f::Identity());
	joinedMesh = SimpleMesh::joinMeshes(joinedMesh, mesh6, Matrix4f::Identity());
	joinedMesh = SimpleMesh::joinMeshes(joinedMesh, mesh7, Matrix4f::Identity());
	joinedMesh = SimpleMesh::joinMeshes(joinedMesh, mesh8, Matrix4f::Identity());
	joinedMesh = SimpleMesh::joinMeshes(joinedMesh, mesh9, Matrix4f::Identity());
	joinedMesh = SimpleMesh::joinMeshes(joinedMesh, mesh10, Matrix4f::Identity());
	joinedMesh = SimpleMesh::joinMeshes(joinedMesh, mesh11, Matrix4f::Identity());*/

	//SimpleMesh mesh0(Vector3f(minx, miny, minz), Vector3f(maxx, maxy, maxz), 1000, 0.1);

	// write mesh to file
	if (!joinedMesh2.writeMesh(filenameOut))
	{
		std::cout << "ERROR: unable to write output file!" << std::endl;
	}
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
	/*currentCameraPose << 1, 0, 0, 0,
		0, -1, 0, 0,
		0, 0, -1, 0,
		0, 0, 0, 1;*/

	// 1.3434 0.6271 1.6606
	//currentCameraPose(0, 3) = 1.5;
	//currentCameraPose(1, 3) = 1.5;
	//currentCameraPose(2, 3) = 1.5;
	currentCameraPose = sensor.getTrajectory();
	Matrix4f currentCameraToWorld = currentCameraPose.inverse(); //Matrix4f::Identity();
	std::cout << "Initial camera pose: " << std::endl << currentCameraPose << std::endl;


	//Matrix4f currentCameraPose = Matrix4f::Identity();
	//currentCameraPose(0, 3) = 2;
	//currentCameraPose(1, 3) = 2;
	//currentCameraPose(2, 3) = 2;
	//Matrix4f currentCameraToWorld = currentCameraPose.inverse();

	/*int x = 35;
	int y = 51;
	int z = 80;

	Vector3d point_g = globalVolume.pos(35, 51, 80);
	float val = globalVolume.getValueAtPoint(point_g);*/


	localTSDF(globalVolume, sensor, currentCameraPose, worldEnd.z());

	//writeDepthToFile(sensor);
	std::string filenameOut = PROJECT_DIR + std::string("/results/result_global_0.off");
	saveVolume(globalVolume, filenameOut, currentCameraPose, sensor.getDepthIntrinsics(), sensor.getDepthImageHeight(), sensor.getDepthImageWidth());
	//writeVolToFile(globalVolume, false);

	//std::vector<Vector3f> raycastPts;

	cv::Mat depthImage = cv::Mat::zeros(sensor.getDepthImageHeight(), sensor.getDepthImageWidth(), CV_32F);
	cv::Mat normalMap = cv::Mat::zeros(sensor.getDepthImageHeight(), sensor.getDepthImageWidth(), CV_32FC3);


	int i = 1;
	const int iMax = 30;
	while (sensor.processNextFrame() && i <= iMax) {
		Raycaster raycaster(currentCameraPose, sensor.getDepthIntrinsics(), sensor.getDepthImageHeight(), sensor.getDepthImageWidth(), sensor.getColorRGBX());

		std::vector<Vector3f> raycastHitPoints;
		std::vector<Vector3f> raycastHitNormals;
		raycaster.castRays2(globalVolume, depthImage, normalMap, raycastHitPoints, raycastHitNormals);
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

void testMeshLine()
{
	std::string filenameOut = PROJECT_DIR + std::string("/results/line_test.off");
	SimpleMesh mesh (Vector3f(0, 0, 0), Vector3f(1, 1, 1), 100, 0.1);

	SimpleMesh currentCameraMesh = SimpleMesh::camera(Matrix4f::Identity(), 0.0015f);
	SimpleMesh resultingMesh = SimpleMesh::joinMeshes(mesh, currentCameraMesh, Matrix4f::Identity());

	// write mesh to file
	if (!resultingMesh.writeMesh(filenameOut))
	{
		std::cout << "ERROR: unable to write output file!" << std::endl;
	}
}

int main()
{
	//lala();
	//testMeshLine();

	//float camX, camY, camZ;
	float minx, miny, minz;
	float maxx, maxy, maxz;
	while (false)
	{
		/*std::cout << "Cam X: ";
		std::cin >> camX;

		std::cout << "Cam Y: ";
		std::cin >> camY;

		std::cout << "Cam Z: ";
		std::cin >> camZ;*/



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
	//executeKinect(-1, -1.5, 0, 1, 0.5, 2);
	//putBBrect(-1, -1.5, 0, 1, 0.5, 2);
	executeKinect(-2, -2, -2, 2, 2, 2);
	
	
	
	
	
}
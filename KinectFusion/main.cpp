#pragma once

#include <iostream>
#include <fstream>
#include <windows.h>
#include <sys/stat.h>
#include <stdio.h>
#include <algorithm>
#include <cctype>
#include <string>

#include <Eigen/Core>
#include "Eigen.h"

#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>

#include "Raycasting.hpp"

#include "KinectOptimizer.h"
#include "VirtualSensor.h"
#include "SimpleMesh.h"
#include "PointCloud.h"
#include "MarchingCubes.h"
#include "DepthCamera.h"
#include "TsdfUtils.h"

#define PROJECT 1


void saveVolume(TsdfUtils::TsdfData tsdfData, std::string filenameOut, Matrix4f currentCameraPose)
{
	clock_t begin = clock();
	SimpleMesh mesh;

#pragma omp parallel for
	for (int x = 0; x < tsdfData.resolution - 1; x++)
	{
		for (unsigned int y = 0; y < tsdfData.resolution - 1; y++)
		{
			for (unsigned int z = 0; z < tsdfData.resolution - 1; z++)
			{
				ProcessVolumeCell(tsdfData, (unsigned int)x, y, z, 0.00f, &mesh);
			}
		}
	}

	SimpleMesh currentCameraMesh = SimpleMesh::camera(currentCameraPose, 0.0015f);
	SimpleMesh resultingMesh = SimpleMesh::joinMeshes(mesh, currentCameraMesh, Matrix4f::Identity());

	// write mesh to file
	if (!resultingMesh.writeMeshObj(filenameOut))
	{
		std::cout << "ERROR: unable to write output file!" << std::endl;
	}

	clock_t end = clock();
	double elapsedSecs = double(end - begin) / 1000;
	std::cout << ">>>Volume is saved in " << elapsedSecs << " seconds." << std::endl;
}

cv::Mat color_depth(const cv::Mat& depth_map)
{
	cv::Mat output = depth_map.clone();
	double min, max;
	cv::minMaxIdx(depth_map, &min, &max);
	output -= min;
	cv::convertScaleAbs(output, output, 255 / (max - min));
	cv::applyColorMap(output, output, cv::COLORMAP_JET);
	return output;
}

int main() {

	std::string recording_name, data_location;
	int dataset_used, dataset_TUM, dataset_direct, dataset_freiburg, dataset_pause, ITER_Print;
	bool recording = false, processing = false;
	bool KinectDataset, KinectRecord, KinectDirect, KinectPause;
	int i = 1;
	unsigned int resolution;
	double size;
	float truncationDistance;
	std::string filenameIn, filenameBaseOut;
	std::vector<int> iterNumbers;

	////////////////////Data Input///////////////////////
	std::cout << "Is the dataset used? 1 / 0: ";
	std::cin >> dataset_used;

	if (dataset_used) {
		std::cout << "Is the dataset from TUM? 1 / 0: ";
		std::cin >> dataset_TUM;
		if (dataset_TUM) {
			KinectDataset = 1;
			KinectRecord = 0;
			std::cout << "Freiburg 1? 1 / 0: ";
			std::cin >> dataset_freiburg;
			if (dataset_freiburg) {
				filenameIn = PROJECT_DIR + std::string("data\\rgbd_dataset_freiburg1_xyz\\");
			}
			else {
				filenameIn = PROJECT_DIR + std::string("data\\rgbd_dataset_freiburg2_xyz\\");
			}
			std::cout << "Enter the name of the recording save folder: ";
			std::cin >> recording_name;
			filenameBaseOut = PROJECT_DIR + std::string("results\\mesh\\") + recording_name + std::string("\\");
		}
		else {
			KinectDataset = 0;
			KinectRecord =0;
			std::cout << "Enter the name of the folder where the data is: ";
			std::cin >> data_location;
			filenameIn = PROJECT_DIR + std::string("data\\") + data_location + std::string("\\");
			std::cout << "Enter the name of the recording save folder: ";
			std::cin >> recording_name;
			filenameBaseOut = PROJECT_DIR + std::string("results\\mesh\\") + recording_name + std::string("\\");
		}
	}
	else {
		std::cout << "Do you want to record the whole dataset straight away? 1 / 0: ";
		std::cin >> dataset_direct;

		std::cout << "Enter the name of the recording save folder: ";
		std::cin >> recording_name;
		filenameBaseOut = PROJECT_DIR + std::string("results\\mesh\\") + recording_name + std::string("\\");
		filenameIn = filenameBaseOut;
		if (dataset_direct) {
			KinectDataset = 0;
			KinectRecord = 1;
			KinectDirect = 1;
		}
		else {
			KinectDataset =0;
			KinectRecord =1;
			KinectDirect =0;
			std::cout << "Do you want to pause between frames? 1 / 0: ";
			std::cin >> dataset_pause;
			if (dataset_pause) {
				KinectPause= 1;
			}
			else {
				KinectPause =0;
			}
			
		}
	}
	////////////////////Data Input End///////////////////////

	cv::utils::fs::createDirectory(filenameBaseOut);
	std::cout << "Data:" << filenameIn << std::endl;
	std::cout << "Mesh:" << filenameBaseOut << std::endl;
	std::cout << "Initialize virtual sensor..." << std::endl;
	VirtualSensor sensor;

	/////////////////////Parameter setting////////////////////////
	if (KinectDataset) {
		std::cout << "How often to print out the iterations: ";
		std::cin >> ITER_Print;
		printf("KinectDataset parameters are set.\n");
		resolution = 300;
		size = 3.0;
		truncationDistance = 0.015;
		iterNumbers = std::vector<int>{ 30, 15, 12};
		if (!sensor.init(filenameIn)) {
			std::cout << "Failed to initialize the sensor!\nCheck file path!" << std::endl;
			return -1;
		}
	}
	else if (!KinectDirect){
		std::cout << "How often to print out the iterations: ";
		std::cin >> ITER_Print;
		printf("Provide data for your dataset.\n");
		std::cout << "Resolution (300): ";
		std::cin >> resolution;
		std::cout << "Size (1.0): ";
		std::cin >> size;
		std::cout << "Truncation Distance (0.015): ";
		std::cin >> truncationDistance;
		iterNumbers = std::vector<int>{ 10, 5, 3 };
		if (!sensor.initNoTxt()) {
			std::cout << "Failed to initialize the sensor!\nCheck file path!" << std::endl;
			return -1;
		}
	}
	else {
		resolution = 300;
		size = 1.0;
		truncationDistance = 0.015;
		iterNumbers = std::vector<int>{ 10, 5, 3 };
		if (!sensor.initNoTxt()) {
			std::cout << "Failed to initialize the sensor!\nCheck file path!" << std::endl;
			return -1;
		}
	}
	/////////////////////Parameter setting End////////////////////////

	Eigen::Matrix3d cameraIntrinsic = sensor.getDepthIntrinsics().cast<double>();
	int depthCols = sensor.getDepthImageWidth();
	int depthRows = sensor.getDepthImageHeight();
	Eigen::Vector2i camResolution(depthCols, depthRows);
	
	// TSDF Setup
	TsdfUtils::TsdfData tsdfData;
	tsdfData.numVoxels = resolution * resolution * resolution;
	tsdfData.resolution = resolution;
	tsdfData.size = size;
	tsdfData.voxelSize = size / resolution;
	tsdfData.tsdf = new float[tsdfData.numVoxels];
	tsdfData.weights = new float[tsdfData.numVoxels];
	tsdfData.truncationDistance = truncationDistance;
	std::fill_n(tsdfData.tsdf, tsdfData.numVoxels, 1.0f);
	std::fill_n(tsdfData.weights, tsdfData.numVoxels, 0.0f);

	if (KinectDataset) {
		sensor.processNextFrame();
		KinectFusionOptimizer optimizer;
		optimizer.setNbOfIterationsInPyramid(iterNumbers);
		optimizer.setMatchingMaxDistance(0.15);
		unsigned int depthFrameSize = sensor.getDepthImageWidth()* sensor.getDepthImageHeight();
		float* prev_depthMap;
		cv::Mat depthImage = cv::Mat::zeros(sensor.getDepthImageHeight(), sensor.getDepthImageWidth(), CV_32F);
		cv::Mat normalMap = cv::Mat::zeros(sensor.getDepthImageHeight(), sensor.getDepthImageWidth(), CV_32FC3);

		Matrix4f currentCameraPose;
		currentCameraPose << 1, 0, 0, -1.5,
			0, 1, 0, -0.5,
			0, 0, 1, 0.5,
			0, 0, 0, 1;
		Matrix4f currentCameraToWorld = currentCameraPose.inverse();
		
		float* depthMapArr = sensor.getDepth();
		Eigen::MatrixXd depthMap = Eigen::Map< Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> >(depthMapArr, depthRows, depthCols).cast<double>();

		TsdfUtils::fuseTsdf(tsdfData, cameraIntrinsic, camResolution,
			depthMap, currentCameraPose.cast<double>());

		Eigen::Matrix3d normalized_Intrinsics = sensor.getDepthIntrinsics().cast<double>();
		normalized_Intrinsics.row(0) = normalized_Intrinsics.row(0) / sensor.getDepthImageWidth();
		normalized_Intrinsics.row(1) = normalized_Intrinsics.row(1) / sensor.getDepthImageHeight();
		raytraceImage(tsdfData, currentCameraPose, normalized_Intrinsics, sensor.getDepthImageWidth(), sensor.getDepthImageHeight(), 1.5, 1e-3, depthImage, normalMap);
		prev_depthMap = depthImage.ptr<float>(0);
		while (sensor.processNextFrame()) {
			Matrix4f currentCameraPose;
			currentCameraToWorld = optimizer.estimatePoseInPyramid(sensor, prev_depthMap, currentCameraToWorld);
			currentCameraPose = currentCameraToWorld.inverse();
			std::cout << "Current Frame #: " << i + 1 << std::endl;
			std::cout << "Current calculated camera pose: " << std::endl << currentCameraPose << std::endl;

			Eigen::MatrixXd depthMap = Eigen::Map< Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> >(sensor.getDepth(), depthRows, depthCols).cast<double>();
			TsdfUtils::fuseTsdf(tsdfData, cameraIntrinsic, camResolution,
				depthMap, currentCameraPose.cast<double>());

			if (i % ITER_Print == 0) {
				std::stringstream ss;
				ss << filenameBaseOut << sensor.getCurrentFrameCnt() << ".obj";
				saveVolume(tsdfData, ss.str(), currentCameraPose);
			}
			raytraceImage(tsdfData, currentCameraPose, normalized_Intrinsics,
				sensor.getDepthImageWidth(), sensor.getDepthImageHeight(),
				1.5, 1e-3, depthImage, normalMap);

			prev_depthMap = depthImage.ptr<float>(0);
			i++;
		}
		return 0;
	}
	else if (!KinectDataset && !KinectRecord) {
		sensor.processNextFrameNoTxt(filenameIn);
		KinectFusionOptimizer optimizer;
		optimizer.setNbOfIterationsInPyramid(iterNumbers);
		optimizer.setMatchingMaxDistance(0.15);
		unsigned int depthFrameSize = sensor.getDepthImageWidth()* sensor.getDepthImageHeight();
		float* prev_depthMap;
		cv::Mat depthImage = cv::Mat::zeros(sensor.getDepthImageHeight(), sensor.getDepthImageWidth(), CV_32F);
		cv::Mat normalMap = cv::Mat::zeros(sensor.getDepthImageHeight(), sensor.getDepthImageWidth(), CV_32FC3);
		Matrix4f currentCameraPose;
		currentCameraPose << 1, 0, 0, -0.5,
			0, 1, 0, -0.8,
			0, 0, 1, 0.5,
			0, 0, 0, 1;
		Matrix4f currentCameraToWorld = currentCameraPose.inverse();
		float* depthMapArr = sensor.getDepth();
		Eigen::MatrixXd depthMap = Eigen::Map< Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> >(depthMapArr, depthRows, depthCols).cast<double>();
		TsdfUtils::fuseTsdf(tsdfData, cameraIntrinsic, camResolution,
			depthMap, currentCameraPose.cast<double>());
		
		Eigen::Matrix3d normalized_Intrinsics = sensor.getDepthIntrinsics().cast<double>();
		normalized_Intrinsics.row(0) = normalized_Intrinsics.row(0) / sensor.getDepthImageWidth();
		normalized_Intrinsics.row(1) = normalized_Intrinsics.row(1) / sensor.getDepthImageHeight();
		raytraceImage(tsdfData, currentCameraPose, normalized_Intrinsics, sensor.getDepthImageWidth(), sensor.getDepthImageHeight(), 1.5, 1e-3, depthImage, normalMap);
		prev_depthMap = depthImage.ptr<float>(0);
		while (sensor.processNextFrameNoTxt(filenameIn)) {
			Matrix4f currentCameraPose;
			currentCameraToWorld = optimizer.estimatePoseInPyramid(sensor, prev_depthMap, currentCameraToWorld);
			currentCameraPose = currentCameraToWorld.inverse();
			std::cout << "Current Frame #: " << i << std::endl;
			std::cout << "Current calculated camera pose: " << std::endl << currentCameraPose << std::endl;

			Eigen::MatrixXd depthMap = Eigen::Map< Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> >(sensor.getDepth(), depthRows, depthCols).cast<double>();
			TsdfUtils::fuseTsdf(tsdfData, cameraIntrinsic, camResolution,
				depthMap, currentCameraPose.cast<double>());

			if (i % ITER_Print == 0) {
				std::stringstream ss;
				ss << filenameBaseOut << sensor.getCurrentFrameCnt() << ".obj";
				saveVolume(tsdfData, ss.str(), currentCameraPose);
			}

			raytraceImage(tsdfData, currentCameraPose, normalized_Intrinsics,
				sensor.getDepthImageWidth(), sensor.getDepthImageHeight(),
				1.5, 1e-3, depthImage, normalMap);
			prev_depthMap = depthImage.ptr<float>(0);
			i++;
		}
		return 0;
	}
	else if(!KinectDirect){

		std::unique_ptr<DepthCamera> camera = std::make_unique<KinectCamera>();
		int current_index = 0;
		cv::namedWindow("KinectFusion RGB Image.");
		cv::namedWindow("KinectFusion Depth Image.");
		InputFrame frame = camera->grab_frame();
		cv::imshow("KinectFusion RGB Image.", frame.color_map);
		cv::imshow("KinectFusion Depth Image.", frame.depth_map);
		auto depth_file = std::stringstream();
		auto color_file = std::stringstream();
		std::cout << filenameIn << std::endl;
		std::cout << filenameBaseOut << std::endl;
		depth_file << filenameIn << "seq_depth" << std::setfill('0') << std::setw(5) << current_index << ".png";
		color_file << filenameIn << "seq_color" << std::setfill('0') << std::setw(5) << current_index << ".png";
		cv::Mat depth_image;
		frame.depth_map.convertTo(depth_image, CV_16UC1);
		cv::imwrite(depth_file.str(), depth_image);
		cv::imwrite(color_file.str(), frame.color_map);

		++current_index;
		++i;
		sensor.processNextFrameNoTxt(filenameIn);
		KinectFusionOptimizer optimizer;
		optimizer.setNbOfIterationsInPyramid(iterNumbers);
		unsigned int depthFrameSize = sensor.getDepthImageWidth()* sensor.getDepthImageHeight();
		float* prev_depthMap;
		cv::Mat depthImage = cv::Mat::zeros(sensor.getDepthImageHeight(), sensor.getDepthImageWidth(), CV_32F);
		cv::Mat normalMap = cv::Mat::zeros(sensor.getDepthImageHeight(), sensor.getDepthImageWidth(), CV_32FC3);

		const Matrix4f zeroPose = Matrix4f::Identity();
		Matrix4f currentCameraToWorld = zeroPose.inverse();
		Matrix4d currentCameraPose = currentCameraToWorld.inverse().cast<double>();
		float* depthMapArr = sensor.getDepth();
		Eigen::MatrixXd depthMap = Eigen::Map< Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> >(depthMapArr, depthRows, depthCols).cast<double>();
		
		TsdfUtils::fuseTsdf(tsdfData, cameraIntrinsic, camResolution,
			depthMap, currentCameraPose.cast<double>());

		Eigen::Matrix3d normalized_Intrinsics = sensor.getDepthIntrinsics().cast<double>();
		normalized_Intrinsics.row(0) = normalized_Intrinsics.row(0) / sensor.getDepthImageWidth();
		normalized_Intrinsics.row(1) = normalized_Intrinsics.row(1) / sensor.getDepthImageHeight();
		raytraceImage(tsdfData, currentCameraPose.cast<float>(), normalized_Intrinsics, sensor.getDepthImageWidth(), sensor.getDepthImageHeight(), 1.5, 1e-3, depthImage, normalMap);
		prev_depthMap = depthImage.ptr<float>(0);

		printf("=========Done Iteration #%i, press 'c' to continue. Press 'q' to finish.\n", current_index);
		while (true) {
			InputFrame frame = camera->grab_frame();
			cv::imshow("KinectFusion RGB Image.", frame.color_map);
			cv::imshow("KinectFusion Depth Image.", frame.depth_map);
			int key = cv::waitKey(1);
			if (key == 'c') {
				processing = true;
			}
			if (key == 'q') {
				if (recording) {
					CameraParameters cam_params = camera->get_parameters();
					std::ofstream cam_params_stream;
					cam_params_stream.open(filenameIn + "seq_cparam.txt");
					cam_params_stream << cam_params.image_width << " " << cam_params.image_height << std::endl;
					cam_params_stream << cam_params.focal_x << " " << cam_params.focal_y << std::endl;
					cam_params_stream << cam_params.principal_x << " " << cam_params.principal_y << std::endl;
					cam_params_stream.close();
				}
				break;
			}
			while (processing) {
				auto depth_file = std::stringstream();
				auto color_file = std::stringstream();
				depth_file << filenameIn << "seq_depth" << std::setfill('0') << std::setw(5) << current_index << ".png";
				color_file << filenameIn << "seq_color" << std::setfill('0') << std::setw(5) << current_index << ".png";
				cv::Mat depth_image;
				frame.depth_map.convertTo(depth_image, CV_16UC1);
				cv::imwrite(depth_file.str(), depth_image);
				cv::imwrite(color_file.str(), frame.color_map);
				++current_index;
				sensor.processNextFrameNoTxt(filenameIn);
				currentCameraToWorld = optimizer.estimatePoseInPyramid(sensor, prev_depthMap, currentCameraToWorld);
				Matrix4f currentCameraPose = currentCameraToWorld.inverse();
				std::cout << "Current calculated camera pose: " << std::endl << currentCameraPose << std::endl;
				Eigen::MatrixXd depthMap = Eigen::Map< Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> >(sensor.getDepth(), depthRows, depthCols).cast<double>();
				
				TsdfUtils::fuseTsdf(tsdfData, cameraIntrinsic, camResolution, depthMap, currentCameraPose.cast<double>());
				if (i % ITER_Print == 0) {
					std::stringstream ss;
					ss << filenameBaseOut << sensor.getCurrentFrameCnt() << ".obj";
					saveVolume(tsdfData, ss.str(), currentCameraPose);
				}
				raytraceImage(tsdfData, currentCameraPose, normalized_Intrinsics,
					sensor.getDepthImageWidth(), sensor.getDepthImageHeight(),
					1.5, 1e-3, depthImage, normalMap);

				prev_depthMap = depthImage.ptr<float>(0);
				if (KinectPause) {
					processing = false;
					printf("=========Done Iteration #%i, press 'c' to continue.\n", current_index);
				}
				i++;
			}
		}
	}
	else {

		std::unique_ptr<DepthCamera> camera = std::make_unique<KinectCamera>();
		int current_index = 0;
		cv::namedWindow("KinectFusion RGB Image.");
		cv::namedWindow("KinectFusion Depth Image.");
		printf("Press 's' to start the recording. Press 'q' to stop the recording.\n");
		while (true) {
			InputFrame frame = camera->grab_frame();
			cv::imshow("KinectFusion RGB Image.", frame.color_map);
			cv::imshow("KinectFusion Depth Image.", color_depth(frame.depth_map));

			if (recording) {
				auto depth_file = std::stringstream();
				auto color_file = std::stringstream();
				depth_file << filenameIn << "seq_depth" << std::setfill('0') << std::setw(5) << current_index << ".png";
				color_file << filenameIn << "seq_color" << std::setfill('0') << std::setw(5) << current_index << ".png";
				cv::Mat depth_image;
				frame.depth_map.convertTo(depth_image, CV_16UC1);
				cv::imwrite(depth_file.str(), depth_image);
				cv::imwrite(color_file.str(), frame.color_map);

				++current_index;
			}

			int key = cv::waitKey(1);
			if (key == 's') {
				recording = true;
				printf("Recording is taking place.\n");
			}

			if (key == 'q') {
				if (recording) {
					CameraParameters cam_params = camera->get_parameters();
					std::ofstream cam_params_stream;
					cam_params_stream.open(filenameIn + "seq_cparam.txt");
					cam_params_stream << cam_params.image_width << " " << cam_params.image_height << std::endl;
					cam_params_stream << cam_params.focal_x << " " << cam_params.focal_y << std::endl;
					cam_params_stream << cam_params.principal_x << " " << cam_params.principal_y << std::endl;
					cam_params_stream.close();
				}
				break;
			}
		}
	}
	return EXIT_SUCCESS;
}
#include "DepthCamera.h"

#include <iostream>
#include <fstream>
#include <iomanip>

#include <opencv2/imgproc.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv/cv.hpp>


KinectCamera::KinectCamera() :
	device{}, depthStream{}, colorStream{}, depthFrame{},
	colorFrame{}, cam_params{}
{
	openni::OpenNI::initialize();

	openni::Array<openni::DeviceInfo> deviceInfoList;
	openni::OpenNI::enumerateDevices(&deviceInfoList);

	std::cout << deviceInfoList.getSize() << std::endl;
	for (int i = 0; i < deviceInfoList.getSize(); ++i) {
		std::cout << deviceInfoList[i].getName() << ", "
			<< deviceInfoList[i].getVendor() << ", "
			<< deviceInfoList[i].getUri() << ", "
			<< std::endl;
	}

	auto ret = device.open(openni::ANY_DEVICE);

	openni::VideoMode depthMode;
	depthMode.setResolution(640, 480);
	depthMode.setFps(30);
	depthMode.setPixelFormat(openni::PIXEL_FORMAT_DEPTH_1_MM);

	openni::VideoMode colorMode;
	colorMode.setResolution(640, 480);
	colorMode.setFps(30);
	colorMode.setPixelFormat(openni::PIXEL_FORMAT_RGB888);

	depthStream.create(device, openni::SENSOR_DEPTH);
	depthStream.setVideoMode(depthMode);
	depthStream.start();

	colorStream.create(device, openni::SENSOR_COLOR);
	colorStream.setVideoMode(colorMode);

	openni::CameraSettings* cameraSettings = colorStream.getCameraSettings();

	cameraSettings->setAutoExposureEnabled(true);
	cameraSettings->setAutoWhiteBalanceEnabled(true);
	//cameraSettings->setExposure(0);
	//cameraSettings->setGain(200);

	cameraSettings = colorStream.getCameraSettings();
	if (cameraSettings != nullptr) {
		std::cout << "Camera Settings" << std::endl;
		std::cout << " Auto Exposure Enabled      : " << cameraSettings->getAutoExposureEnabled() << std::endl;
		std::cout << " Auto WhiteBalance Enabled  : " << cameraSettings->getAutoWhiteBalanceEnabled() << std::endl;
		std::cout << " Exposure                   : " << cameraSettings->getExposure() << std::endl;
		std::cout << " Gain                       : " << cameraSettings->getGain() << std::endl;
	}

	colorStream.start();

	if (openni::STATUS_OK != device.setDepthColorSyncEnabled(true)) {
		std::cout << "setDepthColorSyncEnabled is disable." << std::endl;
	}
	if (openni::STATUS_OK != device.setImageRegistrationMode(openni::IMAGE_REGISTRATION_DEPTH_TO_COLOR)) {
		std::cout << "setImageRegistrationMode is disable." << std::endl;
	}

	openni::VideoMode depthVideoMode = depthStream.getVideoMode();

	CameraParameters cp;
	cp.image_width = depthStream.getVideoMode().getResolutionX();
	cp.image_height = depthStream.getVideoMode().getResolutionY();
	cp.focal_x = cp.focal_y = 525.0;
	cp.principal_x = cp.image_width / 2 - 0.5f;
	cp.principal_y = cp.image_height / 2 - 0.5f;

	cam_params = cp;
}

InputFrame KinectCamera::grab_frame() const
{
	depthStream.readFrame(&depthFrame);
	colorStream.readFrame(&colorFrame);

	if (!depthFrame.isValid() || depthFrame.getData() == nullptr ||
		!colorFrame.isValid() || colorFrame.getData() == nullptr) {
		throw std::runtime_error{ "Frame data retrieval error" };
	}
	else {
		cv::Mat depthImg16U{ depthStream.getVideoMode().getResolutionY(),
			depthStream.getVideoMode().getResolutionX(),
			CV_16U,
			static_cast<char*>(const_cast<void*>(depthFrame.getData())) };
		cv::Mat depth_image;
		depthImg16U.convertTo(depth_image, CV_32FC1);
		cv::flip(depth_image, depth_image, 1);

		cv::Mat color_image{ colorStream.getVideoMode().getResolutionY(),
			colorStream.getVideoMode().getResolutionX(),
			CV_8UC3,
			static_cast<char*>(const_cast<void*>(colorFrame.getData())) };
		cv::cvtColor(color_image, color_image, cv::COLOR_BGR2RGB);
		cv::flip(color_image, color_image, 1);

		return InputFrame{ depth_image, color_image };
	}
}

CameraParameters KinectCamera::get_parameters() const
{
	return cam_params;
}
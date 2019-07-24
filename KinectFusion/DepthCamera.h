#include "DataTypes.h"
#include <cmath>
#include <cstdio>
#include <Windows.h>
#include <vector>
#include <OpenNI.h>

using kinectfusion::CameraParameters;

struct InputFrame {
	cv::Mat_<float> depth_map;
	cv::Mat_<cv::Vec3b> color_map;
};

class DepthCamera {
public:
	virtual ~DepthCamera() = default;

	virtual InputFrame grab_frame() const = 0;
	virtual CameraParameters get_parameters() const = 0;
};

class KinectCamera : public DepthCamera {
public:
	KinectCamera();
	~KinectCamera() override = default;

	InputFrame grab_frame() const override;
	CameraParameters get_parameters() const override;

private:
	openni::Device device;
	mutable openni::VideoStream depthStream;
	mutable openni::VideoStream colorStream;
	mutable openni::VideoFrameRef depthFrame;
	mutable openni::VideoFrameRef colorFrame;

	CameraParameters cam_params;
};

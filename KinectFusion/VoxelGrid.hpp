#include <Eigen/Core>
#include <iostream>

#ifndef KINECT_FUSION_VOXELGRID_H
#define KINECT_FUSION_VOXELGRID_H


class VoxelGrid
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VoxelGrid(unsigned int resolution, double size);
    ~VoxelGrid ();

    float getValue(unsigned int x, unsigned int y, unsigned int z);
	void setValue(unsigned int x, unsigned int y, unsigned int z, float value);

    bool projectRayToVoxelPoint (Eigen::Vector3d origin, Eigen::Vector3d direction, double& length);
    bool  withinGrid(Eigen::Vector3d point);
    Eigen::Vector3d getPointAtIndex(Eigen::Vector3i index);

    float getValueAtPoint(Eigen::Vector3d point);
    void setAllValues(float);

    void operator= (const VoxelGrid&);
    VoxelGrid operator+ (const VoxelGrid&);
    VoxelGrid operator* (const VoxelGrid&);
    VoxelGrid operator/ (const VoxelGrid&);

	inline Eigen::Vector3d pos(int i, int j, int k) const
	{
		Eigen::Vector3d coord(0, 0, 0);

		float dd = 1.0f / (resolution - 1);

		coord[0] = size * (double(i)*dd);
		coord[1] = size * (double(j)*dd);
		coord[2] = size * (double(k)*dd);

		return coord;
	}

    const unsigned int resolution;
    const double size;
    const double voxelSize;
    int numElements;
	float *voxelData;
};

#endif //KINECT_FUSION_VOXELGRID_H

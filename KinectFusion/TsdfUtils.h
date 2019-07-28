#pragma once
#ifndef TSDFUTILS_H
#define TSDFUTILS_H 


namespace TsdfUtils
{
	struct TsdfData
	{
		float *tsdf;
		float *weights;
		unsigned int resolution;
		double size;
		double voxelSize;
		unsigned int numVoxels;
		float truncationDistance;
	};



	inline Eigen::Vector3d pos(int i, int j, int k, unsigned int tsdfResolution, double tsdfSize)
	{
		Eigen::Vector3d coord(0, 0, 0);

		float dd = 1.0f / (tsdfResolution - 1);

		coord[0] = tsdfSize * (double(i)*dd);
		coord[1] = tsdfSize * (double(j)*dd);
		coord[2] = tsdfSize * (double(k)*dd);

		return coord;
	}


	inline Eigen::Vector3d tsdfGetPointAtIndex(Eigen::Vector3i index, unsigned int tsdfResolution, double tsdfSize) {
		Eigen::Vector3d point = index.cast<double>() / (double(tsdfResolution - 1));
		point = point * tsdfSize;
		return point;
	}

	inline float getTsdfValue(unsigned int x, unsigned int y, unsigned int z, TsdfData tsdfData)
	{
		if (x >= tsdfData.resolution || y >= tsdfData.resolution || z >= tsdfData.resolution) { return 0.0f; }
		return tsdfData.tsdf[x + y * tsdfData.resolution + z * tsdfData.resolution*tsdfData.resolution];
	}

	inline float getWeightValue(unsigned int x, unsigned int y, unsigned int z, TsdfData tsdfData)
	{
		if (x >= tsdfData.resolution || y >= tsdfData.resolution || z >= tsdfData.resolution) { return 0.0f; }
		return tsdfData.weights[x + y * tsdfData.resolution + z * tsdfData.resolution*tsdfData.resolution];
	}

	inline void setValue(unsigned int x, unsigned int y, unsigned int z, float value,
		unsigned int tsdfResolution, float *tsdf)
	{
		if (x >= tsdfResolution || y >= tsdfResolution || z >= tsdfResolution) { return; }
		tsdf[x + y * tsdfResolution + z * tsdfResolution*tsdfResolution] = value;
	}

	inline bool tsdfWithinGrid(Eigen::Vector3d point, double tsdfSize)
	{
		double x = point.x();
		double y = point.y();
		double z = point.z();

		return x >= 0 && x <= tsdfSize &&
			y >= 0 && y <= tsdfSize &&
			z >= 0 && z <= tsdfSize;
	}


	inline void fuseTsdf(TsdfUtils::TsdfData tsdfData,
		Eigen::Matrix3d &_cameraIntrinsic,
		Eigen::Vector2i &_camResolution,
		Eigen::MatrixXd &depthMap, 
		Eigen::Matrix4d cameraExtrinsic
		)
	{
#pragma omp parallel for
		for (int xi = 0; xi < tsdfData.resolution; ++xi) {
			for (int yi = 0; yi < tsdfData.resolution; ++yi) {
				for (int zi = 0; zi < tsdfData.resolution; ++zi) {
					Eigen::Vector3i voxelIndex(xi, yi, zi);

					//convert voxel indexes to world coordinates
					Eigen::Vector4d worldPointHomo;
					worldPointHomo << TsdfUtils::tsdfGetPointAtIndex(voxelIndex, tsdfData.resolution, tsdfData.size), 1;

					//transform world point to cam coords
					Eigen::Vector3d cameraPoint = (cameraExtrinsic * worldPointHomo).head(3);

					//point behind camera or too far away
					if (cameraPoint.z() <= 0 || cameraPoint.z() > 3.0) {
						continue;
					}

					//Project point to pixel in depthmap
					Eigen::Vector3d pixPointHomo = _cameraIntrinsic * cameraPoint;
					Eigen::Vector2i pixPoint;
					pixPoint << pixPointHomo.x() / pixPointHomo.z(), pixPointHomo.y() / pixPointHomo.z();

					//if pix outisde depthmap
					if (bool(((pixPoint - _camResolution).array() >= 0).any()) || bool(((pixPoint).array() < 0).any())) {
						continue;
					}

					//calc SDF value.
					int depthRow = pixPoint.y();//_camResolution.y()-1 - pixPoint.y(); //row0 is at top. y0 is at bottom.
					int depthCol = pixPoint.x();
					double pointDepth = cameraPoint.z();
					float TSDF_val = (float)depthMap.coeff(depthRow, depthCol) - (float)pointDepth;

					//truncate SDF value
					if (TSDF_val >= -tsdfData.truncationDistance) {
						TSDF_val = std::min(1.0f, fabsf(TSDF_val) / tsdfData.truncationDistance)*copysignf(1.0f, TSDF_val);
					}
					else { //too far behind obstacle
						continue;
					}


					
					float newGlobalVal = TsdfUtils::getWeightValue(xi, yi, zi, tsdfData) * TsdfUtils::getTsdfValue(xi, yi, zi, tsdfData)
						+ 1 * TSDF_val;

					float newGlobalWeight = TsdfUtils::getWeightValue(xi, yi, zi, tsdfData) + 1;

					newGlobalVal /= newGlobalWeight;

					TsdfUtils::setValue(xi, yi, zi, newGlobalVal, tsdfData.resolution, tsdfData.tsdf);
					TsdfUtils::setValue(xi, yi, zi, newGlobalWeight, tsdfData.resolution, tsdfData.weights);
				}
			}
		}
	}


	inline bool projectRayToVoxelPoint(
		Eigen::Vector3d origin, Eigen::Vector3d direction, double& length,
		double volumeSize)
	{
		double dirfrac_x = 1.0 / direction.x();
		double dirfrac_y = 1.0 / direction.y();
		double dirfrac_z = 1.0 / direction.z();

		Eigen::Vector3d lb = Eigen::Vector3d::Zero();
		Eigen::Vector3d rt = Eigen::Vector3d::Ones()*(volumeSize);

		double t1 = (lb.x() - origin.x())*dirfrac_x;
		double t2 = (rt.x() - origin.x())*dirfrac_x;
		double t3 = (lb.y() - origin.y())*dirfrac_y;
		double t4 = (rt.y() - origin.y())*dirfrac_y;
		double t5 = (lb.z() - origin.z())*dirfrac_z;
		double t6 = (rt.z() - origin.z())*dirfrac_z;

		double tmin = std::fmax(std::fmax(std::fmin(t1, t2), std::fmin(t3, t4)), std::fmin(t5, t6));
		double tmax = std::fmin(std::fmin(std::fmax(t1, t2), std::fmax(t3, t4)), std::fmax(t5, t6));

		// if tmax < 0, ray (line) is intersecting AABB, but the whole AABB is behind us
		if (tmax < 0)
		{
			length = tmax;
			return false;
		}

		// if tmin > tmax, ray doesn't intersect AABB
		if (tmin > tmax)
		{
			length = tmax;
			return false;
		}


		length = tmin;
		return true;
	}


	inline float getValueAtPoint(Eigen::Vector3d point, TsdfData tsdfData)
	{
		// Clamp point to within the voxel volume

		auto x_local = float(point.x() / tsdfData.size)*tsdfData.resolution;
		auto y_local = float(point.y() / tsdfData.size)*tsdfData.resolution;
		auto z_local = float(point.z() / tsdfData.size)*tsdfData.resolution;

		if (x_local < 0.0) { x_local = 0.0; }
		if (y_local < 0.0) { y_local = 0.0; }
		if (z_local < 0.0) { z_local = 0.0; }

		if (x_local > tsdfData.resolution - 2) { x_local = float(tsdfData.resolution - 2); }
		if (y_local > tsdfData.resolution - 2) { y_local = float(tsdfData.resolution - 2); }
		if (z_local > tsdfData.resolution - 2) { z_local = float(tsdfData.resolution - 2); }

		auto x1 = (unsigned int)x_local;
		auto y1 = (unsigned int)y_local;
		auto z1 = (unsigned int)z_local;

		if (x1 > tsdfData.resolution - 2) { x1 = tsdfData.resolution - 2; }
		if (y1 > tsdfData.resolution - 2) { y1 = tsdfData.resolution - 2; }
		if (z1 > tsdfData.resolution - 2) { z1 = tsdfData.resolution - 2; }

		float xd = x_local - x1;
		float yd = y_local - y1;
		float zd = z_local - z1;

		float c000 = TsdfUtils::getTsdfValue(x1, y1, z1, tsdfData);
		float c001 = TsdfUtils::getTsdfValue(x1, y1, z1 + 1, tsdfData);
		float c010 = TsdfUtils::getTsdfValue(x1, y1 + 1, z1, tsdfData);
		float c011 = TsdfUtils::getTsdfValue(x1, y1 + 1, z1 + 1, tsdfData);
		float c100 = TsdfUtils::getTsdfValue(x1 + 1, y1, z1, tsdfData);
		float c101 = TsdfUtils::getTsdfValue(x1 + 1, y1, z1 + 1, tsdfData);
		float c110 = TsdfUtils::getTsdfValue(x1 + 1, y1 + 1, z1, tsdfData);
		float c111 = TsdfUtils::getTsdfValue(x1 + 1, y1 + 1, z1 + 1, tsdfData);

		float c00 = c000 * (1 - xd) + c100 * xd;
		float c01 = c001 * (1 - xd) + c101 * xd;
		float c10 = c010 * (1 - xd) + c110 * xd;
		float c11 = c011 * (1 - xd) + c111 * xd;

		float c0 = c00 * (1 - yd) + c10 * yd;
		float c1 = c01 * (1 - yd) + c11 * yd;

		return c0 * (1 - zd) + c1 * zd;
	}
}


#endif // !TSDFUTILS_H
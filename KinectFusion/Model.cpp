#include "Model.hpp"

ModelReconstructor::ModelReconstructor(float truncationDistance,
	unsigned int resolution,
	double size,
	Eigen::Matrix3d cameraIntrinsic,
	Eigen::Vector2i camResolution)
	: _TSDF_global(new VoxelGrid(resolution, size)),
	_weights_global(new VoxelGrid(resolution, size))
{
	_truncationDistance = truncationDistance;
	_resolution = resolution;
	_size = size;
	_cameraIntrinsic = cameraIntrinsic;
	_camResolution = camResolution;

	_weights_global->setAllValues(0.0);
	_TSDF_global->setAllValues(1.0);
}

VoxelGrid *ModelReconstructor::getModel()
{
	return _TSDF_global;
}

void ModelReconstructor::fuseFrame(Eigen::MatrixXd depthMap, Eigen::Matrix4d cameraExtrinsic)
{
	std::cout << "Fusing Frame Parallel... " << std::endl;

	#pragma omp parallel for
	for (int xi = 0; xi < _resolution; ++xi) {
		for (int yi = 0; yi < _resolution; ++yi) {
			for (int zi = 0; zi < _resolution; ++zi) {
				Eigen::Vector3i voxelIndex(xi, yi, zi);

				//convert voxel indexes to world coordinates
				Eigen::Vector4d worldPointHomo;
				worldPointHomo << _TSDF_global->getPointAtIndex(voxelIndex), 1;

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
				if (TSDF_val >= -_truncationDistance) {
					TSDF_val = std::min(1.0f, fabsf(TSDF_val) / _truncationDistance)*copysignf(1.0f, TSDF_val);
				}
				else { //too far behind obstacle
					continue;
				}


				float newGlobalVal = _weights_global->getValue(xi, yi, zi) * _TSDF_global->getValue(xi, yi, zi)
					+ 1 * TSDF_val;

				float newGlobalWeight = _weights_global->getValue(xi, yi, zi) + 1;

				newGlobalVal /= newGlobalWeight;

				_TSDF_global->setValue(xi, yi, zi, newGlobalVal);
				_weights_global->setValue(xi, yi, zi, newGlobalWeight);
			}
		}
	}
}

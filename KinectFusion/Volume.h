#pragma once

#ifndef VOLUME_H
#define VOLUME_H

#include <limits>
#include "Eigen.h"
#include <FreeImage.h>
typedef unsigned int uint;

//! A regular volume dataset
class Volume
{
public:

	//! Initializes an empty volume dataset.
	Volume(Vector3d lowerLeftPoint_g_, Vector3d upperTopPoint_g_, uint numCellsX_ = 10, uint numCellsY_ = 10, uint numCellsZ_ = 10);

	~Volume();

	inline void computeMinMaxValues(double& minVal, double& maxVal) const
	{
		minVal = std::numeric_limits<double>::max();
		maxVal = -minVal;
		for (uint i1 = 0; i1 < numCellsX*numCellsY*numCellsZ; i1++)
		{
			if (minVal > vol[i1]) minVal = vol[i1];
			if (maxVal < vol[i1]) maxVal = vol[i1];
		}
	}

	//! Computes spacing in x,y,z-directions.
	void compute_ddx_dddx();

	//! Zeros out the memory
	void zeroOutMemory();

	//! Set the value at i.
	inline void set(uint i, double val)
	{
		if (val > maxValue)
			maxValue = val;

		if (val < minValue)
			minValue = val;

		vol[i] = val;
	}

	//! Set the value at i.
	inline void setWeight(uint i, double w)
	{
		weight[i] = w;
	}

	inline void setColor(uint x_, uint y_, uint z_, BYTE* col)
	{
		uint loc = 4 * getPosFromTuple(x_, y_, z_);
		color[loc] = col[0];
		color[loc + 1] = col[1];
		color[loc + 2] = col[2];
		color[loc + 3] = col[3];
	}

	//! Set the value at (x_, y_, z_).
	inline void set(uint x_, uint y_, uint z_, double val)
	{
		vol[getPosFromTuple(x_, y_, z_)] = val;
	};

	//! Set the value at (x_, y_, z_).
	inline void setWeight(uint x_, uint y_, uint z_, double w)
	{
		weight[getPosFromTuple(x_, y_, z_)] = w;
	};

	//! Get the value at (x_, y_, z_).
	inline double get(uint i) const
	{
		return vol[i];
	};

	inline double getWeight(uint i) const
	{
		return weight[i];
	};

	inline BYTE* getColor(uint x_, uint y_, uint z_, BYTE* col)
	{
		uint loc = 4 * getPosFromTuple(x_, y_, z_);
		return &color[loc];
	}

	float get_InterpVal(Eigen::Vector3d point)
	{
		//See https://www.wikiwand.com/en/Trilinear_interpolation
		
		// Clamp point to within the voxel volume

		auto x_local = float((point.x() - lowerLeftPoint_g.x()) / totalLength.x())*numCellsX;
		auto y_local = float((point.y() - lowerLeftPoint_g.y()) / totalLength.y())*numCellsY;
		auto z_local = float((point.z() - lowerLeftPoint_g.z()) / totalLength.z())*numCellsZ;

		if (x_local < 0.0) { x_local = 0.0; }
		if (y_local < 0.0) { y_local = 0.0; }
		if (z_local < 0.0) { z_local = 0.0; }

		if (x_local > numCellsX - 2) { x_local = float(numCellsX - 2); }
		if (y_local > numCellsY - 2) { y_local = float(numCellsY - 2); }
		if (z_local > numCellsZ - 2) { z_local = float(numCellsZ - 2); }

		auto x1 = (unsigned int)x_local;
		auto y1 = (unsigned int)y_local;
		auto z1 = (unsigned int)z_local;

		if (x1 > numCellsX - 2) { x1 = numCellsX - 2; }
		if (y1 > numCellsY - 2) { y1 = numCellsY - 2; }
		if (z1 > numCellsZ - 2) { z1 = numCellsZ - 2; }

		float xd = x_local - x1;
		float yd = y_local - y1;
		float zd = z_local - z1;

		float c000 = get(x1, y1, z1);
		float c001 = get(x1, y1, z1 + 1);
		float c010 = get(x1, y1 + 1, z1);
		float c011 = get(x1, y1 + 1, z1 + 1);
		float c100 = get(x1 + 1, y1, z1);
		float c101 = get(x1 + 1, y1, z1 + 1);
		float c110 = get(x1 + 1, y1 + 1, z1);
		float c111 = get(x1 + 1, y1 + 1, z1 + 1);

		float c00 = c000 * (1 - xd) + c100 * xd;
		float c01 = c001 * (1 - xd) + c101 * xd;
		float c10 = c010 * (1 - xd) + c110 * xd;
		float c11 = c011 * (1 - xd) + c111 * xd;

		float c0 = c00 * (1 - yd) + c10 * yd;
		float c1 = c01 * (1 - yd) + c11 * yd;

		return c0 * (1 - zd) + c1 * zd;
	}

	//! Get the value at (x_, y_, z_).
	inline double get(uint x_, uint y_, uint z_) const
	{
		if (x_ >= numCellsX || y_ >= numCellsY || z_ >= numCellsZ)
		{
			return 0;
		}
		return vol[getPosFromTuple(x_, y_, z_)];
	};

	//! Get the value at (x_, y_, z_).
	inline double getWeight(uint x_, uint y_, uint z_) const
	{
		return weight[getPosFromTuple(x_, y_, z_)];
	};

	//! Get the value at (pos.x, pos.y, pos.z).
	inline double get(const Vector3i& pos_) const
	{
		return(get(pos_[0], pos_[1], pos_[2]));
	}

	//! Returns the cartesian x-coordinates of node (i,..).
	inline double posX(int i) const
	{
		return lowerLeftPoint_g[0] + totalLength[0] * (double(i)*ddx);
	}

	//! Returns the cartesian y-coordinates of node (..,i,..).
	inline double posY(int i) const
	{
		return lowerLeftPoint_g[1] + totalLength[1] * (double(i)*ddy);
	}

	//! Returns the cartesian z-coordinates of node (..,i).
	inline double posZ(int i) const
	{
		return lowerLeftPoint_g[2] + totalLength[2] * (double(i)*ddz);
	}

	//! Returns the cartesian coordinates of node (i,j,k).
	inline Vector3d pos(int i, int j, int k) const
	{
		Vector3d coord(0, 0, 0);

		coord[0] = lowerLeftPoint_g[0] + totalLength[0]*(double(i)*ddx);
		coord[1] = lowerLeftPoint_g[1] + totalLength[1]*(double(j)*ddy);
		coord[2] = lowerLeftPoint_g[2] + totalLength[2]*(double(k)*ddz);

		return coord;
	}

	//! Returns the cartesian coordinates of node (i,j,k).
	inline std::vector<Vector3f> getAllVoxelPos() const
	{
		std::vector<Vector3f> allVoxelPos;

		int dimX = (int)getDimX();
		int dimY = (int)getDimY();
		int dimZ = (int)getDimZ();

		int numVoxels = (dimX * dimY * dimZ);
		allVoxelPos.reserve(numVoxels);

		#pragma omp parallel for
		for (int x = 0; x < dimX; x++)
		{
			for (int y = 0; y < dimY; y++)
			{
				for (int z = 0; z < dimZ; z++)
				{
					allVoxelPos.push_back(pos(x, y, z).cast<float>());
				}
			}
		}

		return allVoxelPos;
	}

	//! Returns the Data.
	double* getData();

	//! Sets all entries in the volume to '0'
	void clean();

	//! Returns number of cells in x-dir.
	inline uint getDimX() const { return numCellsX; }

	//! Returns number of cells in y-dir.
	inline uint getDimY() const { return numCellsY; }

	//! Returns number of cells in z-dir.
	inline uint getDimZ() const { return numCellsZ; }

	inline Vector3d getMin() { return lowerLeftPoint_g; }
	inline Vector3d getMax() { return upperTopPoint_g; }

	//! Sets minimum extension
	void SetMin(Vector3d min_);

	//! Sets maximum extension
	void SetMax(Vector3d max_);

	inline uint getPosFromTuple(int x, int y, int z) const
	{
		return x + y * numCellsX + z * numCellsX * numCellsY;
		//return z + y * numCellsZ + x * numCellsY * numCellsZ;
	}

	inline Vector3i getNodeFromWorld(Vector3f worldCoord)	// Cartesian to node coordinates
	{
		Vector3i idx;
		idx[0] = (int)((worldCoord[0] - lowerLeftPoint_g[0]) / (totalLength[0] * ddx));
		idx[1] = (int)((worldCoord[1] - lowerLeftPoint_g[1]) / (totalLength[1] * ddx));
		idx[2] = (int)((worldCoord[2] - lowerLeftPoint_g[2]) / (totalLength[2] * ddx));
		return idx;
	}

	inline bool isVoxelIndexValid(Vector3i point)
	{
		return point.x() >= 0 && point.y() >= 0 && point.z() >= 0
			&& point.x() < numCellsX
			&& point.y() < numCellsY
			&& point.z() < numCellsZ;
	}

	inline bool withinGrid(Vector3d point)
	{
		return point.x() >= lowerLeftPoint_g.x() && point.y() >= lowerLeftPoint_g.y() && point.z() >= lowerLeftPoint_g.z()
			&& point.x() < upperTopPoint_g.x()
			&& point.y() < upperTopPoint_g.y()
			&& point.z() < upperTopPoint_g.z();
	}

	inline double getNodeValAtWorld(Vector3d worldCoord)
	{
		return get(getNodeFromWorld(worldCoord.cast<float>()));
	}

	//! Lower left and Upper right corner.
	Vector3d lowerLeftPoint_g, upperTopPoint_g;

	//! max-min
	Vector3d totalLength;

	double ddx, ddy, ddz;
	double cellSizeX_g, cellSizeY_g, cellSizeZ_g;

	//! Number of cells in x, y and z-direction.
	uint numCellsX, numCellsY, numCellsZ;

	double* vol;
	double* weight;
	BYTE* color;

	double maxValue, minValue;

private:

	//! x,y,z access to vol*
	inline double vol_access(int x, int y, int z) const
	{
		return vol[getPosFromTuple(x, y, z)];
	}

	//! x,y,z access to weight*
	inline double weight_access(int x, int y, int z) const
	{
		return weight[getPosFromTuple(x, y, z)];
	}
};

#endif // VOLUME_H

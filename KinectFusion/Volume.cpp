#include "Volume.h"

//! Initializes an empty volume dataset.
Volume::Volume(Vector3d lowerLeftPoint_g_, Vector3d upperTopPoint_g_, uint numCellsX_, uint numCellsY_, uint numCellsZ_)
{
	lowerLeftPoint_g = lowerLeftPoint_g_;
	upperTopPoint_g = upperTopPoint_g_;
	totalLength = upperTopPoint_g - lowerLeftPoint_g;
	numCellsX = numCellsX_;
	numCellsY = numCellsY_;
	numCellsZ = numCellsZ_;
	vol = NULL;
	weight = NULL;
	color = NULL;

	vol = new double[numCellsX*numCellsY*numCellsZ];
	weight = new double[numCellsX*numCellsY*numCellsZ];
	color = new BYTE[4 * numCellsX * numCellsY * numCellsZ];

	compute_ddx_dddx();
}

Volume::~Volume()
{
	delete[] vol;
	delete[] weight;
	delete[] color;
};


//! Computes spacing in x,y,z-directions.
void Volume::compute_ddx_dddx()
{
	ddx = 1.0f / (numCellsX - 1);
	ddy = 1.0f / (numCellsY - 1);
	ddz = 1.0f / (numCellsZ - 1);

	cellSizeX_g = (upperTopPoint_g[0] - lowerLeftPoint_g[0]) / (numCellsX - 1);
	cellSizeY_g = (upperTopPoint_g[1] - lowerLeftPoint_g[1]) / (numCellsY - 1);
	cellSizeZ_g = (upperTopPoint_g[2] - lowerLeftPoint_g[2]) / (numCellsZ - 1);

	if (numCellsZ == 1)
	{
		ddz = 0;
		cellSizeZ_g = 0;
	}

	totalLength = upperTopPoint_g - lowerLeftPoint_g;
}

//! Zeros out the memory
void Volume::zeroOutMemory()
{
	for (uint i1 = 0; i1 < numCellsX*numCellsY*numCellsZ; i1++)
	{
		vol[i1] = double(0);
		weight[i1] = double(0);
	}
		
}

//! Returns the Data.
double* Volume::getData()
{
	return vol;
};

//! Sets all entries in the volume to '0'
void Volume::clean()
{
	for (uint i1 = 0; i1 < numCellsX*numCellsY*numCellsZ; i1++)
	{
		vol[i1] = double(1.0);
		weight[i1] = double(0.0);
	}
}

//! Sets minimum extension
void Volume::SetMin(Vector3d min_)
{
	lowerLeftPoint_g = min_;
	totalLength = upperTopPoint_g - lowerLeftPoint_g;
}

//! Sets maximum extension
void Volume::SetMax(Vector3d max_)
{
	upperTopPoint_g = max_;
	totalLength = upperTopPoint_g - lowerLeftPoint_g;
}

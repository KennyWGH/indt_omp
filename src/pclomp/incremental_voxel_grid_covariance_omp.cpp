/**
 * Copyright 2022 WANG_Guanhua(guanhuamail@163.com)
 * 
 * Software License Agreement (BSD License)
 * 
 * All rights reserved.
*/

#include "pclomp/incremental_voxel_grid_covariance_omp.h"
#include "pclomp/incremental_voxel_grid_covariance_omp_impl.hpp"

template class pclomp::IncrementalVoxelGridCovariance<pcl::PointXYZ>;
template class pclomp::IncrementalVoxelGridCovariance<pcl::PointXYZI>;

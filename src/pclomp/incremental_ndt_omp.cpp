/**
 * Copyright 2022 WANG_Guanhua(guanhuamail@163.com)
 * 
 * Software License Agreement (BSD License)
 * 
 * All rights reserved.
*/

#include "pclomp/incremental_ndt_omp.h"
#include "pclomp/incremental_ndt_omp_impl.hpp"

template class pclomp::IncrementalNDT<pcl::PointXYZ, pcl::PointXYZ>;
template class pclomp::IncrementalNDT<pcl::PointXYZI, pcl::PointXYZI>;

/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2010-2011, Willow Garage, Inc.
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the copyright holder(s) nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

/**
 * Copyright 2022 WANG_Guanhua(guanhuamail@163.com)
 * 
 * Software License Agreement (BSD License)
 * 
 * All rights reserved.
*/

#ifndef PCL_INCREMENTAL_VOXEL_GRID_COVARIANCE_OMP_IMPL_H_
#define PCL_INCREMENTAL_VOXEL_GRID_COVARIANCE_OMP_IMPL_H_

#include <pcl/common/common.h>
#include <pcl/filters/boost.h>
#include "incremental_voxel_grid_covariance_omp.h"
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <boost/make_unique.hpp>

/** 在incremental版本中，我们顺便把重要的函数注释一下，主打一个开发者友好. */

namespace {
  bool IsEqualIndex(const Eigen::Array3i& a, const Eigen::Array3i& b) {
      return a[0] == b[0] && a[1] == b[1] && a[2] == b[2];
  }
}


// 这个函数是初始化过程的核心函数，负责把target点云栅格化和高斯化，
// 维护栅格点云到哈希表leaf的映射关系，并且把栅格几何中心作为点云输出(也即形参中的output).
// 因此，这个函数也是我们增量式改造的重点部分.
template<typename PointT> void
pclomp::IncrementalVoxelGridCovariance<PointT>::applyFilter (PointCloud &output)
{
  // **************************** incremental mode **************************** //

  if (enable_incremetal_mode_) {
    applyFilterUnderIncrementalMode(output);
    return;
  }

  // **************************** traditional mode **************************** //

  voxel_centroids_leaf_indices_.clear ();

  // input_ 指的是输入的target点云，这个指针本体位于基类pcl::Filter<PointT>中
  // Has the input dataset been set already?
  if (!input_)
  {
    PCL_WARN ("[pcl::%s::applyFilter] No input dataset given!\n", getClassName ().c_str ());
    output.width = output.height = 0;
    output.points.clear ();
    return;
  }

  // Copy the header (and thus the frame_id) + allocate enough space for points
  output.height = 1;                          // downsampling breaks the organized structure
  output.is_dense = true;                     // we filter out invalid points
  output.points.clear ();

  // 获取target点云的空间范围
  // Get the minimum and maximum dimensions
  Eigen::Vector4f min_p, max_p;
  if (!filter_field_name_.empty ()) { 
    // If we don't want to process the entire cloud...
    pcl::getMinMax3D<PointT> (input_, filter_field_name_, 
      static_cast<float> (filter_limit_min_), 
      static_cast<float> (filter_limit_max_), 
      min_p, max_p, 
      filter_limit_negative_);
  } else {
    pcl::getMinMax3D<PointT> (*input_, min_p, max_p);
  }

  // 根据空间范围，估算leaf/voxel的数量；如果leaf数量太多以至于发生数值溢出，报错，建议调大voxel边长
  // Check that the leaf size is not too small, given the size of the data
  int64_t dx = static_cast<int64_t>((max_p[0] - min_p[0]) * inverse_leaf_size_[0])+1;
  int64_t dy = static_cast<int64_t>((max_p[1] - min_p[1]) * inverse_leaf_size_[1])+1;
  int64_t dz = static_cast<int64_t>((max_p[2] - min_p[2]) * inverse_leaf_size_[2])+1;
  if((dx*dy*dz) > std::numeric_limits<int32_t>::max()) {
    PCL_WARN("[pcl::%s::applyFilter] Leaf size is too small for the input dataset. Integer indices would overflow.", getClassName().c_str());
    output.clear();
    return;
  }

  // 计算包络盒范围
  // Compute the minimum and maximum bounding box values
  min_b_[0] = static_cast<int> (floor (min_p[0] * inverse_leaf_size_[0]));
  max_b_[0] = static_cast<int> (floor (max_p[0] * inverse_leaf_size_[0]));
  min_b_[1] = static_cast<int> (floor (min_p[1] * inverse_leaf_size_[1]));
  max_b_[1] = static_cast<int> (floor (max_p[1] * inverse_leaf_size_[1]));
  min_b_[2] = static_cast<int> (floor (min_p[2] * inverse_leaf_size_[2]));
  max_b_[2] = static_cast<int> (floor (max_p[2] * inverse_leaf_size_[2]));

  // 计算三个维度上各需要分割出多少个voxel，三者相乘即为所需voxel总数。
  // Compute the number of divisions needed along all axis
  div_b_ = max_b_ - min_b_ + Eigen::Vector4i::Ones ();
  div_b_[3] = 0;

  // 清空哈希表
  // Clear the leaves
  leaves_.clear ();
  // leaves_.reserve(8192);

  // 这是干嘛的？回头再来看
  // Set up the division multiplier
  divb_mul_ = Eigen::Vector4i (1, div_b_[0], div_b_[0] * div_b_[1], 0);

  // 这个指的是模板点类型含有多少个field，默认4.
  int centroid_size = 4;

  // 这个布尔量位于基类pcl::VoxelGrid中，为true则对所有fields降采样，否则只对XYZ降采样.
  if (downsample_all_data_) {
    centroid_size = boost::mpl::size<FieldList>::value;
  }

  // 如果模板类型PointT含有RGB这样的field，降采样时也要考虑这些field.
  // ---[ RGB special case
  std::vector<pcl::PCLPointField> fields;
  int rgba_index = -1;
  rgba_index = pcl::getFieldIndex<PointT> ("rgb", fields);
  if (rgba_index == -1) /*返回-1代表没有这个field*/ {
    rgba_index = pcl::getFieldIndex<PointT> ("rgba", fields);
  } 
  if (rgba_index >= 0) {
    rgba_index = fields[rgba_index].offset;
    centroid_size += 3;
  }

  // *****************************************************************
  // 第一轮: 遍历所有点，插入相应的voxel中,同时放入哈希表；对voxel以半成品的方式计算mean和covariance。

  // 如果启用了针对特定field的过滤，执行之。
  // If we don't want to process the entire cloud, but rather filter points far away from the viewpoint first...
  if (!filter_field_name_.empty ()) {
    // Get the distance field index，也即获得field的offset
    std::vector<pcl::PCLPointField> fields;
    int distance_idx = pcl::getFieldIndex<PointT> (filter_field_name_, fields);
    if (distance_idx == -1) /*意味着这个field不存在，警告*/ { 
      PCL_WARN ("[pcl::%s::applyFilter] Invalid filter field name. Index is %d.\n", getClassName ().c_str (), distance_idx);
    }

    // First pass: go over all points and insert them into the right leaf
    for (size_t cp = 0; cp < input_->points.size (); ++cp) {
      // Check if the point is invalid
      if (!input_->is_dense) {
        if (!std::isfinite (input_->points[cp].x) ||
            !std::isfinite (input_->points[cp].y) ||
            !std::isfinite (input_->points[cp].z)) {
          continue;
        }
      }

      // Get the distance value，也即获得指定field的值。
      const uint8_t* pt_data = reinterpret_cast<const uint8_t*> (&input_->points[cp]);
      float distance_value = 0;
      memcpy (&distance_value, pt_data + fields[distance_idx].offset, sizeof (float));

      // 对field执行过滤。
      if (filter_limit_negative_) {
        // Use a threshold for cutting out points which inside the interval
        if ((distance_value < filter_limit_max_) && (distance_value > filter_limit_min_)) {
          continue;
        }
      } else {
        // Use a threshold for cutting out points which are too close/far away
        if ((distance_value > filter_limit_max_) || (distance_value < filter_limit_min_)) {
          continue;
        }
      }

      // 计算点在三个维度上分别的索引。
      int ijk0 = static_cast<int> (floor (input_->points[cp].x * inverse_leaf_size_[0]) - static_cast<float> (min_b_[0]));
      int ijk1 = static_cast<int> (floor (input_->points[cp].y * inverse_leaf_size_[1]) - static_cast<float> (min_b_[1]));
      int ijk2 = static_cast<int> (floor (input_->points[cp].z * inverse_leaf_size_[2]) - static_cast<float> (min_b_[2]));

      // 三维索引转一维索引,一维索引同时可以哈希表中的key。
      // Compute the centroid leaf index
      int idx = ijk0 * divb_mul_[0] + ijk1 * divb_mul_[1] + ijk2 * divb_mul_[2];

      Leaf& leaf = leaves_[idx];
      if (leaf.nr_points == 0)
      {
        leaf.centroid.resize (centroid_size); //centroid的field要和PointT保持一致
        leaf.centroid.setZero ();
      }

      Eigen::Vector3d pt3d (input_->points[cp].x, input_->points[cp].y, input_->points[cp].z);
      // Accumulate point sum for centroid calculation
      leaf.mean_ += pt3d;
      // Accumulate x*xT for single pass covariance calculation
      leaf.cov_ += pt3d * pt3d.transpose ();

      // Do we need to process all the fields?
      if (!downsample_all_data_) {
        Eigen::Vector4f pt (input_->points[cp].x, input_->points[cp].y, input_->points[cp].z, 0);
        leaf.centroid.template head<4> () += pt;
      } else {
        // Copy all the fields
        Eigen::VectorXf centroid = Eigen::VectorXf::Zero (centroid_size);
        // ---[ RGB special case
        if (rgba_index >= 0)
        {
          // fill r/g/b data
          int rgb;
          memcpy (&rgb, reinterpret_cast<const char*> (&input_->points[cp]) + rgba_index, sizeof (int));
          centroid[centroid_size - 3] = static_cast<float> ((rgb >> 16) & 0x0000ff);
          centroid[centroid_size - 2] = static_cast<float> ((rgb >> 8) & 0x0000ff);
          centroid[centroid_size - 1] = static_cast<float> ((rgb) & 0x0000ff);
        }
        pcl::for_each_type<FieldList> (pcl::NdCopyPointEigenFunctor<PointT> (input_->points[cp], centroid));
        leaf.centroid += centroid;
      }
      // 别忘了更新leaf中点的数量，后边计算mean和cov都依赖这个信息。
      ++leaf.nr_points;
    }
  }
  // No distance filtering, process all data
  else {
    // First pass: go over all points and insert them into the right leaf
    for (size_t cp = 0; cp < input_->points.size (); ++cp) {
      // Check if the point is invalid
      if (!input_->is_dense) {
        if (!std::isfinite (input_->points[cp].x) ||
            !std::isfinite (input_->points[cp].y) ||
            !std::isfinite (input_->points[cp].z)) {
          continue;
        }
      }

      int ijk0 = static_cast<int> (floor (input_->points[cp].x * inverse_leaf_size_[0]) - static_cast<float> (min_b_[0]));
      int ijk1 = static_cast<int> (floor (input_->points[cp].y * inverse_leaf_size_[1]) - static_cast<float> (min_b_[1]));
      int ijk2 = static_cast<int> (floor (input_->points[cp].z * inverse_leaf_size_[2]) - static_cast<float> (min_b_[2]));

      // Compute the centroid leaf index
      int idx = ijk0 * divb_mul_[0] + ijk1 * divb_mul_[1] + ijk2 * divb_mul_[2];

      //int idx = (((input_->points[cp].getArray4fmap () * inverse_leaf_size_).template cast<int> ()).matrix () - min_b_).dot (divb_mul_);
      Leaf& leaf = leaves_[idx];
      if (leaf.nr_points == 0) {
        leaf.centroid.resize (centroid_size);
        leaf.centroid.setZero ();
      }

      Eigen::Vector3d pt3d (input_->points[cp].x, input_->points[cp].y, input_->points[cp].z);
      // Accumulate point sum for centroid calculation
      leaf.mean_ += pt3d;
      // Accumulate x*xT for single pass covariance calculation
      leaf.cov_ += pt3d * pt3d.transpose ();

      // Do we need to process all the fields?
      if (!downsample_all_data_) {
        Eigen::Vector4f pt (input_->points[cp].x, input_->points[cp].y, input_->points[cp].z, 0);
        leaf.centroid.template head<4> () += pt;
      }
      else {
        // Copy all the fields
        Eigen::VectorXf centroid = Eigen::VectorXf::Zero (centroid_size);
        // ---[ RGB special case
        if (rgba_index >= 0)
        {
          // Fill r/g/b data, assuming that the order is BGRA
          int rgb;
          memcpy (&rgb, reinterpret_cast<const char*> (&input_->points[cp]) + rgba_index, sizeof (int));
          centroid[centroid_size - 3] = static_cast<float> ((rgb >> 16) & 0x0000ff);
          centroid[centroid_size - 2] = static_cast<float> ((rgb >> 8) & 0x0000ff);
          centroid[centroid_size - 1] = static_cast<float> ((rgb) & 0x0000ff);
        }
        pcl::for_each_type<FieldList> (pcl::NdCopyPointEigenFunctor<PointT> (input_->points[cp], centroid));
        leaf.centroid += centroid;
      }
      ++leaf.nr_points;
    }
  }

  // *****************************************************************
  // 第二轮: 遍历所有leaf/voxel，计算得到真正的mean和covariance。
  // Second pass: go over all leaves and compute centroids and covariance matrices
  output.points.reserve (leaves_.size ());
  if (searchable_) { // if true, will gather voxel centroids as point cloud for building kdtree.
    voxel_centroids_leaf_indices_.reserve (leaves_.size ());
  }

  // 注意哈希表中的leaf是无序的，我们在 leaf layout 中按key的顺序保存一个索引
  int cp = 0;
  if (save_leaf_layout_) {
    leaf_layout_.resize (div_b_[0] * div_b_[1] * div_b_[2], -1);
  }

  // Eigen values and vectors calculated to prevent near singular matrices
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigensolver;
  Eigen::Matrix3d eigen_val;
  Eigen::Vector3d pt_sum;

  // Eigen values less than a threshold of max eigen value are inflated to a set fraction of the max eigen value.
  double min_covar_eigvalue;

  for (auto it = leaves_.begin (); it != leaves_.end (); ++it) {
    // Normalize the centroid
    Leaf& leaf = it->second;

    // centroid和mean的区别: 前者按需要考虑所有field的均值，后者仅考虑xyz的均值
    // Normalize the centroid
    leaf.centroid /= static_cast<float> (leaf.nr_points);
    // Point sum used for single pass covariance calculation
    pt_sum = leaf.mean_;
    // Normalize mean
    leaf.mean_ /= leaf.nr_points;

    // If the voxel contains sufficient points, its covariance is calculated and is added to the voxel centroids and output clouds.
    // Points with less than the minimum points will have a can not be accurately approximated using a normal distribution.
    if (leaf.nr_points >= min_points_per_voxel_) {
      if (save_leaf_layout_) {
        leaf_layout_[it->first] = cp++;
      }

      output.push_back (PointT ());

      // Do we need to process all the fields?
      if (!downsample_all_data_) {
        output.points.back ().x = leaf.centroid[0];
        output.points.back ().y = leaf.centroid[1];
        output.points.back ().z = leaf.centroid[2];
      } else {
        pcl::for_each_type<FieldList> (pcl::NdCopyEigenPointFunctor<PointT> (leaf.centroid, output.back ()));
        // ---[ RGB special case
        if (rgba_index >= 0) {
          // pack r/g/b into rgb
          float r = leaf.centroid[centroid_size - 3], g = leaf.centroid[centroid_size - 2], b = leaf.centroid[centroid_size - 1];
          int rgb = (static_cast<int> (r)) << 16 | (static_cast<int> (g)) << 8 | (static_cast<int> (b));
          memcpy (reinterpret_cast<char*> (&output.points.back ()) + rgba_index, &rgb, sizeof (float));
        }
      }

      // 这个索引映射表是执行近邻搜索的必要工具: 也即通过kdtree找centroid,再通过这个映射表找leaf.
      // Stores the voxel indices for fast access searching
      if (searchable_)
        voxel_centroids_leaf_indices_.push_back (static_cast<int> (it->first));

      // 这里真正计算leaf中的协方差
      // Single pass covariance calculation
      leaf.cov_ = (leaf.cov_ - 2 * (pt_sum * leaf.mean_.transpose ())) / leaf.nr_points + leaf.mean_ * leaf.mean_.transpose ();
      leaf.cov_ *= (leaf.nr_points - 1.0) / leaf.nr_points;

      // 计算和平滑特征值
      //Normalize Eigen Val such that max no more than 100x min.
      eigensolver.compute (leaf.cov_);
      eigen_val = eigensolver.eigenvalues ().asDiagonal ();
      leaf.evecs_ = eigensolver.eigenvectors ();
      // 屏蔽异常特征值issue
      if (eigen_val (0, 0) < 0 || eigen_val (1, 1) < 0 || eigen_val (2, 2) <= 0) {
        leaf.nr_points = -1;
        continue;
      }

      // 避免协方差矩阵接近奇异(引入数值计算bug)
      // Avoids matrices near singularities (eq 6.11)[Magnusson 2009]
      min_covar_eigvalue = min_covar_eigvalue_mult_ * eigen_val (2, 2);
      if (eigen_val (0, 0) < min_covar_eigvalue) {
        eigen_val (0, 0) = min_covar_eigvalue;

        if (eigen_val (1, 1) < min_covar_eigvalue) {
          eigen_val (1, 1) = min_covar_eigvalue;
        }
        // 根据平滑后的特征值重新设置协方差矩阵
        leaf.cov_ = leaf.evecs_ * eigen_val * leaf.evecs_.inverse ();
      }
      leaf.evals_ = eigen_val.diagonal ();
      leaf.icov_ = leaf.cov_.inverse ();
      // 屏蔽数值异常issue
      if (leaf.icov_.maxCoeff () == std::numeric_limits<float>::infinity ( )
          || leaf.icov_.minCoeff () == -std::numeric_limits<float>::infinity ( ) )
      {
        leaf.nr_points = -1;
      }

    }
  }

  output.width = static_cast<uint32_t> (output.points.size ());
  // ************** done. **************
}


// [WANG_Guanhua] 增量式版本核心代码：增量式地更新内部的voxels。
template<typename PointT> void
pclomp::IncrementalVoxelGridCovariance<PointT>::applyFilterUnderIncrementalMode (PointCloud &output)
{
  // // init containers with empty voxels inside spatial range limit.
  // bool enable_init_voxels = false;
  // if (voxels_map_.empty() && enable_init_voxels) {
  //   const int alongX_num_voxels = bounding_box_size_[0] * inverse_leaf_size_[0] + 1;
  //   const int alongY_num_voxels = bounding_box_size_[1] * inverse_leaf_size_[1] + 1;
  //   const int alongZ_num_voxels = bounding_box_size_[2] * inverse_leaf_size_[2] + 1;
  //   const Eigen::Array3i offset = {alongX_num_voxels / 2, 
  //                                   alongY_num_voxels / 2, 
  //                                   alongZ_num_voxels / 2};
  //   size_t num_inited_voxels = 0;
  //   for (int i = 0; i < alongX_num_voxels; ++i) {
  //     for (int j = 0; j < alongY_num_voxels; ++j) {
  //       for (int k = 0; k < alongZ_num_voxels; ++k) {
  //         Eigen::Array3i voxel_3dkey = Eigen::Array3i(i,j,k) - offset;
  //         voxels_list_.push_front(
  //           std::make_pair(voxel_3dkey, 
  //             boost::make_unique<Leaf>()));
  //         voxels_map_.insert(std::make_pair(voxel_3dkey, voxels_list_.begin()));
  //         ++num_inited_voxels;
  //       }
  //     }
  //   }
  // }

  // reset these variables unconditionally.
  voxel_centroids_leaf_3d_indices_.clear ();
  min_b_ = Eigen::Vector4i::Zero();
  max_b_ = Eigen::Vector4i::Zero();
  div_b_ = Eigen::Vector4i::Zero();
  divb_mul_ = Eigen::Vector4i::Zero();

  if (!input_) {
    PCL_WARN ("[pcl::%s::applyFilterUnderIncrementalMode] No input dataset given!\n", getClassName ().c_str ());
    output.width = output.height = 0;
    output.points.clear ();
    return;
  }

  if (verbose_info_level_ >= 2) {
    std::cout << "[debug] processing pointcloud with points num: " << input_->points.size() << std::endl;
  }

  output.height = 1;
  output.is_dense = true;
  output.points.clear ();

  // how many fields we need to consider, by default is 4.
  int centroid_size = 4;

  // this variable is in pcl::VoxelGrid; if true, process all field, otherwise only xyz.
  if (downsample_all_data_) {
    centroid_size = boost::mpl::size<FieldList>::value;
  }

  // in case the PointT type has rgb field.
  // ---[ RGB special case
  std::vector<pcl::PCLPointField> fields;
  int rgba_index = -1;
  rgba_index = pcl::getFieldIndex<PointT> ("rgb", fields);
  if (rgba_index == -1) /*this field doesn't exist*/ {
    rgba_index = pcl::getFieldIndex<PointT> ("rgba", fields);
  } 
  if (rgba_index >= 0) {
    rgba_index = fields[rgba_index].offset;
    centroid_size += 3;
  }

  // ****************************************************************************************************
  // During first pass, collect all leaves that were inserted with new points for further update.
  SpaceHashMap leafves_need_update;

  // If we don't want to process the entire cloud, but rather filter points far away from the viewpoint first...
  if (!filter_field_name_.empty ()) {
    // Get the distance field index
    std::vector<pcl::PCLPointField> fields;
    int distance_idx = pcl::getFieldIndex<PointT> (filter_field_name_, fields);
    if (distance_idx == -1) /*this field doesn't exist*/ { 
      PCL_WARN ("[pcl::%s::applyFilterUnderIncrementalMode] Invalid filter field name. Index is %d.\n", getClassName ().c_str (), distance_idx);
    }

    // First pass: go over all points and insert them into the right leaf
    for (size_t cp = 0; cp < input_->points.size (); ++cp) {
      // Check if the point is invalid
      if (!input_->is_dense) {
        if (!std::isfinite (input_->points[cp].x) ||
            !std::isfinite (input_->points[cp].y) ||
            !std::isfinite (input_->points[cp].z)) {
          continue;
        }
      }

      // Get the distance value，也即获得指定field的值。
      const uint8_t* pt_data = reinterpret_cast<const uint8_t*> (&input_->points[cp]);
      float distance_value = 0;
      memcpy (&distance_value, pt_data + fields[distance_idx].offset, sizeof (float));

      // 对field执行过滤。
      if (filter_limit_negative_) {
        // Use a threshold for cutting out points which inside the interval
        if ((distance_value < filter_limit_max_) && (distance_value > filter_limit_min_)) {
          continue;
        }
      } else {
        // Use a threshold for cutting out points which are too close/far away
        if ((distance_value > filter_limit_max_) || (distance_value < filter_limit_min_)) {
          continue;
        }
      }

      // compute voxel 3d-index.
      int ijk0 = static_cast<int> (floor (input_->points[cp].x * inverse_leaf_size_[0]));
      int ijk1 = static_cast<int> (floor (input_->points[cp].y * inverse_leaf_size_[1]));
      int ijk2 = static_cast<int> (floor (input_->points[cp].z * inverse_leaf_size_[2]));

      // find corresponding leaf!
      Eigen::Array3i voxel_3dkey(ijk0, ijk1, ijk2);
      auto voxel_ite = voxels_map_.find(voxel_3dkey);
      if (voxel_ite != voxels_map_.end()) {
        // Handle exception.
        if (!IsEqualIndex(voxel_3dkey, voxel_ite->second->first)) {
          PCL_WARN ("[pcl::%s::applyFilterUnderIncrementalMode] voxel index conflicts: [%d, %d, %d] != [%d, %d, %d], skip it.\n", 
            getClassName ().c_str (), voxel_3dkey[0], voxel_3dkey[1], voxel_3dkey[2], 
            voxel_ite->second->first[0], voxel_ite->second->first[1], voxel_ite->second->first[2]);
          continue;
        }
      }
      else {
        // add new leaf.
        voxels_list_.push_front(std::make_pair(voxel_3dkey, boost::make_unique<Leaf>()));
        voxels_map_.insert(std::make_pair(voxel_3dkey, voxels_list_.begin()));
        voxel_ite = voxels_map_.find(voxel_3dkey);
      }

      // record leaf for further mean & covariance update.
      leafves_need_update.insert((*voxel_ite));

      // insert point to leaf.
      Leaf& leaf = *(voxel_ite->second->second);

      if (leaf.nr_points == 0)
      {
        leaf.centroid.resize (centroid_size); //centroid的field要和PointT保持一致
        leaf.centroid.setZero ();
      }

      Eigen::Vector3d pt3d (input_->points[cp].x, input_->points[cp].y, input_->points[cp].z);
      // Accumulate point sum for centroid calculation
      // leaf.mean_ += pt3d;
      leaf.pt_sum_ += pt3d;
      // Accumulate x*xT for single pass covariance calculation
      // leaf.cov_ += pt3d * pt3d.transpose ();
      leaf.pt_sq_sum_ += pt3d * pt3d.transpose ();

      // Do we need to process all the fields?
      if (!downsample_all_data_) {
        Eigen::Vector4f pt (input_->points[cp].x, input_->points[cp].y, input_->points[cp].z, 0);
        // leaf.centroid.template head<4> () += pt;
        leaf.centroid_sum_.template head<4> () += pt;
      } else {
        // Copy all the fields
        Eigen::VectorXf centroid = Eigen::VectorXf::Zero (centroid_size);
        // ---[ RGB special case
        if (rgba_index >= 0)
        {
          // fill r/g/b data
          int rgb;
          memcpy (&rgb, reinterpret_cast<const char*> (&input_->points[cp]) + rgba_index, sizeof (int));
          centroid[centroid_size - 3] = static_cast<float> ((rgb >> 16) & 0x0000ff);
          centroid[centroid_size - 2] = static_cast<float> ((rgb >> 8) & 0x0000ff);
          centroid[centroid_size - 1] = static_cast<float> ((rgb) & 0x0000ff);
        }
        pcl::for_each_type<FieldList> (pcl::NdCopyPointEigenFunctor<PointT> (input_->points[cp], centroid));
        // leaf.centroid += centroid;
        leaf.centroid_sum_ += centroid;
      }
      // you must not forget to update this variable!
      ++leaf.nr_points;
    }
  }
  // No distance filtering, process all data
  else {
    
    // std::cout << "[debug] voxel size: " << leaf_size_.transpose() << std::endl;
    // std::cout << "[debug] voxel size inverse: " << inverse_leaf_size_.transpose() << std::endl;
    // First pass: go over all points and insert them into the right leaf
    for (size_t cp = 0; cp < input_->points.size (); ++cp) {
      // Check if the point is invalid
      if (!input_->is_dense) {
        if (!std::isfinite (input_->points[cp].x) ||
            !std::isfinite (input_->points[cp].y) ||
            !std::isfinite (input_->points[cp].z)) {
          continue;
        }
      }

      // compute voxel 3d-index.
      int ijk0 = static_cast<int> (floor (input_->points[cp].x * inverse_leaf_size_[0]));
      int ijk1 = static_cast<int> (floor (input_->points[cp].y * inverse_leaf_size_[1]));
      int ijk2 = static_cast<int> (floor (input_->points[cp].z * inverse_leaf_size_[2]));

      // find corresponding leaf!
      Eigen::Array3i voxel_3dkey(ijk0, ijk1, ijk2);
      // std::cout << "[debug] point projected to voxel-key: " << voxel_3dkey.transpose() << std::endl;
      auto voxel_ite = voxels_map_.find(voxel_3dkey);
      if (voxel_ite != voxels_map_.end()) {
        // Handle exception.
        if (!IsEqualIndex(voxel_3dkey, voxel_ite->second->first)) {
          PCL_WARN ("[pcl::%s::applyFilterUnderIncrementalMode] voxel index conflicts: [%d, %d, %d] != [%d, %d, %d], skip it.\n", 
            getClassName ().c_str (), voxel_3dkey[0], voxel_3dkey[1], voxel_3dkey[2], 
            voxel_ite->second->first[0], voxel_ite->second->first[1], voxel_ite->second->first[2]);
          continue;
        }
        // std::cout << "[debug] voxel already existed. " << std::endl;
      }
      else {
        // add new leaf.
        voxels_list_.push_front(std::make_pair(voxel_3dkey, boost::make_unique<Leaf>()));
        voxels_map_.insert(std::make_pair(voxel_3dkey, voxels_list_.begin()));
        voxel_ite = voxels_map_.find(voxel_3dkey);
        if (verbose_info_level_ >= 1) {
          std::cout << "[debug] creating new voxel, key=[" << voxel_3dkey.transpose() 
            << "], LeafSize=[" << leaf_size_.transpose() << "]" << std::endl;
        }
        
      }

      // record leaf for further mean & covariance update.
      leafves_need_update.insert((*voxel_ite));

      // insert point to leaf.
      Leaf& leaf = *(voxel_ite->second->second);

      if (leaf.nr_points == 0) {
        leaf.centroid.resize (centroid_size);
        leaf.centroid.setZero ();
        leaf.centroid_sum_.resize (centroid_size);
        leaf.centroid_sum_.setZero ();
      }

      Eigen::Vector3d pt3d (input_->points[cp].x, input_->points[cp].y, input_->points[cp].z);
      // Accumulate point sum for centroid calculation
      // leaf.mean_ += pt3d;
      leaf.pt_sum_ += pt3d;
      // Accumulate x*xT for single pass covariance calculation
      // leaf.cov_ += pt3d * pt3d.transpose ();
      leaf.pt_sq_sum_ += pt3d * pt3d.transpose ();

      // Do we need to process all the fields?
      if (!downsample_all_data_) {
        Eigen::Vector4f pt (input_->points[cp].x, input_->points[cp].y, input_->points[cp].z, 0);
        // leaf.centroid.template head<4> () += pt;
        leaf.centroid_sum_.template head<4> () += pt;
      }
      else {
        // Copy all the fields
        Eigen::VectorXf centroid = Eigen::VectorXf::Zero (centroid_size);
        // ---[ RGB special case
        if (rgba_index >= 0)
        {
          // Fill r/g/b data, assuming that the order is BGRA
          int rgb;
          memcpy (&rgb, reinterpret_cast<const char*> (&input_->points[cp]) + rgba_index, sizeof (int));
          centroid[centroid_size - 3] = static_cast<float> ((rgb >> 16) & 0x0000ff);
          centroid[centroid_size - 2] = static_cast<float> ((rgb >> 8) & 0x0000ff);
          centroid[centroid_size - 1] = static_cast<float> ((rgb) & 0x0000ff);
        }
        pcl::for_each_type<FieldList> (pcl::NdCopyPointEigenFunctor<PointT> (input_->points[cp], centroid));
        // leaf.centroid += centroid;
        leaf.centroid_sum_ += centroid;
      }
      ++leaf.nr_points;
    }
  }

  // ****************************************************************************************************
  // Second pass: go over all newly-inserted leaves and compute centroids and covariance matrices.

  // for incremental version, voxel are placed discretely, layout collapse！
  save_leaf_layout_ = false;
  leaf_layout_.clear();

  // Eigen values and vectors calculated to prevent near singular matrices
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigensolver;
  Eigen::Matrix3d eigen_val;
  Eigen::Vector3d pt_sum;

  // Eigen values less than a threshold of max eigen value are inflated to a set fraction of the max eigen value.
  double min_covar_eigvalue;

  for (auto& it : leafves_need_update) {
    Leaf& leaf = *(it.second->second);

    // that's pretty cool, really love this, the code here looks much more clean and elegant.
    leaf.UpdateState(min_points_per_voxel_);

    if (verbose_info_level_ >= 2) {
      std::cout << "[debug] updated voxel [" << it.first.transpose() 
        << "], pts=" << leaf.nr_points << ", mean=[" << leaf.mean_.transpose() 
        << "], EigenValue=[" << leaf.evals_.transpose() << "]" << std::endl;
    }
  }

  // ****************************************************************************************************
  // Third pass: go over all leaves and generate output, update data or information if demanded.
  // 可用于更新: output点云，包络盒范围信息，centroid点云索引到哈希表key的映射表。
  if (searchable_ || generate_voxel_centroid_cloud_) {
    output.points.reserve (voxels_list_.size ());
  }
  if (searchable_) { //in incremental mode, this should be false, since we avoid building kdtree.
    voxel_centroids_leaf_3d_indices_.reserve (voxels_list_.size ());
  }
  if (enable_obb_update_) {
    min_b_ = Eigen::Vector4i::Ones() * std::numeric_limits<int>::max();
    max_b_ = Eigen::Vector4i::Ones() * std::numeric_limits<int>::min();
    min_b_[3] = 0;
    max_b_[3] = 0;
    div_b_ = Eigen::Vector4i::Zero();
    divb_mul_ = Eigen::Vector4i::Zero();
  }
  if (searchable_ || generate_voxel_centroid_cloud_ || enable_obb_update_) {
    for (auto it = voxels_list_.begin(); it != voxels_list_.end(); ++it) {
      const Eigen::Array3i key = it->first;
      Leaf& leaf = *(it->second);

      if (searchable_ || generate_voxel_centroid_cloud_) {
        output.push_back (PointT ());
        // Do we need to process all the fields?
        if (!downsample_all_data_) {
          output.points.back ().x = leaf.centroid[0];
          output.points.back ().y = leaf.centroid[1];
          output.points.back ().z = leaf.centroid[2];
        } else {
          pcl::for_each_type<FieldList> (pcl::NdCopyEigenPointFunctor<PointT> (leaf.centroid, output.back ()));
          // ---[ RGB special case
          if (rgba_index >= 0) {
            // pack r/g/b into rgb
            float r = leaf.centroid[centroid_size - 3], g = leaf.centroid[centroid_size - 2], b = leaf.centroid[centroid_size - 1];
            int rgb = (static_cast<int> (r)) << 16 | (static_cast<int> (g)) << 8 | (static_cast<int> (b));
            memcpy (reinterpret_cast<char*> (&output.points.back ()) + rgba_index, &rgb, sizeof (float));
          }
        }
      }

      // Stores the voxel indices for fast access searching (kdtree for 1d-index, then to 3d-key, then query hash map)
      if (searchable_) {
        voxel_centroids_leaf_3d_indices_.push_back ((it->first));
      }

      if (enable_obb_update_) {
        // Compute the minimum and maximum bounding box values
        min_b_[0] = key[0] < min_b_[0] ? key[0] : min_b_[0];
        min_b_[1] = key[1] < min_b_[1] ? key[1] : min_b_[1];
        min_b_[2] = key[2] < min_b_[2] ? key[2] : min_b_[2];
        max_b_[0] = key[0] > max_b_[0] ? key[0] : max_b_[0];
        max_b_[1] = key[1] > max_b_[1] ? key[1] : max_b_[1];
        max_b_[2] = key[2] > max_b_[2] ? key[2] : max_b_[2];

        // Compute the number of divisions along all axis
        div_b_ = max_b_ - min_b_ + Eigen::Vector4i::Ones ();
        div_b_[3] = 0;

        // Set up the division multiplier
        divb_mul_ = Eigen::Vector4i (1, div_b_[0], div_b_[0] * div_b_[1], 0);
      }
    }
  }

  output.width = static_cast<uint32_t> (output.points.size ());
  // ************** done. **************
}


// 根据当前位置和BoundingBox大小，动态卸载Box范围之外的voxel。
template<typename PointT> void
pclomp::IncrementalVoxelGridCovariance<PointT>::updateVoxelMapCenterAndTrim(
  const Eigen::Vector3f &position) {

  // Important action: Trim voxels around current position when necessary!
  current_position_ = position;
  if ((current_position_ - last_trimmed_position_).norm() > trim_every_n_meters_) {
    last_trimmed_position_ = current_position_;
    Eigen::Array3f current_border_max = (current_position_ + 0.5 * bounding_box_size_).array();
    Eigen::Array3f current_border_min = (current_position_ - 0.5 * bounding_box_size_).array();
    Eigen::Array3f inv_leaf_size = Eigen::Array3f(
      inverse_leaf_size_[0], inverse_leaf_size_[1], inverse_leaf_size_[2]);
    Eigen::Array3i border_max_index = (current_border_max * inv_leaf_size).cast<int>();
    Eigen::Array3i border_min_index = (current_border_min * inv_leaf_size).cast<int>();
    float removed_voxels_ave_ditance = 0;
    size_t removed_voxels_counts = 0;
    size_t all_voxels_counts = voxels_list_.size();
    if (verbose_info_level_ >= 0) {
      std::cout << "[debug] try to remove extra voxels beyond bounding-box. \n";
      std::cout << "[debug] current position [" << current_position_.transpose() << "]. \n";
      std::cout << "[debug] bounding-box size [" << bounding_box_size_.transpose() << "]. \n";
      std::cout << "[debug] current bounding-box: \n"
        << "        X[" << current_border_min[0] << "," << current_border_max[0] << "]\n"
        << "        Y[" << current_border_min[1] << "," << current_border_max[1] << "]\n"
        << "        Z[" << current_border_min[2] << "," << current_border_max[2] << "]" << std::endl;
    }
    for (auto it = voxels_list_.begin(); it != voxels_list_.end(); ) {
      const auto diff_to_max = it->first - border_max_index;
      const auto diff_to_min = it->first - border_min_index;
      if ((diff_to_max > 0).any() || (diff_to_min < 0).any()) {
        if (verbose_info_level_ >= 1) {
          Eigen::Vector3f voxel_position(
            (it->first[0] + 0.5) * leaf_size_[0],
            (it->first[1] + 0.5) * leaf_size_[1],
            (it->first[2] + 0.5) * leaf_size_[2] );
          removed_voxels_ave_ditance += (voxel_position - current_position_).norm();
          std::cout << "[debug] removing voxel [" << it->first.transpose() 
            << "], voxel position [" << voxel_position.transpose() 
            << "], distance to POI [" << (voxel_position - current_position_).norm() 
            << "]" << std::endl;
        }
        // unload voxels beyond spatial limit.
        voxels_map_.erase(it->first);
        it = voxels_list_.erase(it);
        removed_voxels_counts++;
      } else {
        it++;
      }
    }
    if (verbose_info_level_ >= 0) {
      if (removed_voxels_counts > 0) {
        removed_voxels_ave_ditance /= removed_voxels_counts;
      }
      std::cout << "[debug] Removed & unloaded [" << removed_voxels_counts 
        << " / " << all_voxels_counts << "] voxels beyond bounding-box, ave-distance = " 
        << removed_voxels_ave_ditance << std::endl;
    }
    PCL_INFO ("[pcl::%s::applyFilterUnderIncrementalMode] Unloaded %d / %d voxels.\n", 
      getClassName ().c_str (), removed_voxels_counts, all_voxels_counts);
  }


}


// 自定义捕获哪些邻居栅格
template<typename PointT> int
pclomp::IncrementalVoxelGridCovariance<PointT>::getNeighborhoodAtPoint(
  const Eigen::MatrixXi& relative_coordinates, 
  const PointT& reference_point, 
  std::vector<LeafConstPtr> &neighbors) const
{
	neighbors.clear();

	// Find displacement coordinates
	Eigen::Vector4i ijk(static_cast<int> (floor(reference_point.x / leaf_size_[0])),
		static_cast<int> (floor(reference_point.y / leaf_size_[1])),
		static_cast<int> (floor(reference_point.z / leaf_size_[2])), 0);
	Eigen::Array4i diff2min = min_b_ - ijk;
	Eigen::Array4i diff2max = max_b_ - ijk;
	neighbors.reserve(relative_coordinates.cols());

  // ************** incremental mode **************
  // [WANG_Guanhua] 增量式版本核心代码：查询N近邻voxels
  if (enable_incremetal_mode_) {
    for (int ni = 0; ni < relative_coordinates.cols(); ni++)
    {
      Eigen::Vector4i displacement = (Eigen::Vector4i() << relative_coordinates.col(ni), 0).finished();
      Eigen::Vector4i voxel_ijk = ijk + displacement;
      const Eigen::Array3i voxel_key(voxel_ijk[0], voxel_ijk[1], voxel_ijk[2]);
      auto it = voxels_map_.find(voxel_key);
      if (it != voxels_map_.end()) {
        LeafConstPtr leaf = it->second->second.get();
        neighbors.push_back(leaf);
      }
    }
    // done.
    return (static_cast<int> (neighbors.size()));
  }

  // ************** traditional mode **************
	// Check each neighbor to see if it is occupied and contains sufficient points
	// Slower than radius search because needs to check 26 indices
	for (int ni = 0; ni < relative_coordinates.cols(); ni++)
	{
		Eigen::Vector4i displacement = (Eigen::Vector4i() << relative_coordinates.col(ni), 0).finished();
		// Checking if the specified cell is in the grid
		if ((diff2min <= displacement.array()).all() && (diff2max >= displacement.array()).all())
		{
			auto leaf_iter = leaves_.find(((ijk + displacement - min_b_).dot(divb_mul_)));
			if (leaf_iter != leaves_.end() && leaf_iter->second.nr_points >= min_points_per_voxel_)
			{
				LeafConstPtr leaf = &(leaf_iter->second);
				neighbors.push_back(leaf);
			}
		}
	}

	return (static_cast<int> (neighbors.size()));
}


// 捕获26邻居栅格
template<typename PointT> int
pclomp::IncrementalVoxelGridCovariance<PointT>::getNeighborhoodAtPoint(
  const PointT& reference_point, 
  std::vector<LeafConstPtr> &neighbors) const
{
	neighbors.clear();

	// Find displacement coordinates
	Eigen::MatrixXi relative_coordinates = pcl::getAllNeighborCellIndices();
	return getNeighborhoodAtPoint(relative_coordinates, reference_point, neighbors);
}


// 捕获7邻居栅格
template<typename PointT> int
pclomp::IncrementalVoxelGridCovariance<PointT>::getNeighborhoodAtPoint7(
  const PointT& reference_point, 
  std::vector<LeafConstPtr> &neighbors) const
{
	neighbors.clear();

	Eigen::MatrixXi relative_coordinates(3, 7);
	relative_coordinates.setZero();
	relative_coordinates(0, 1) = 1;
	relative_coordinates(0, 2) = -1;
	relative_coordinates(1, 3) = 1;
	relative_coordinates(1, 4) = -1;
	relative_coordinates(2, 5) = 1;
	relative_coordinates(2, 6) = -1;

	return getNeighborhoodAtPoint(relative_coordinates, reference_point, neighbors);
}


// 捕获1邻居栅格
template<typename PointT> int
pclomp::IncrementalVoxelGridCovariance<PointT>::getNeighborhoodAtPoint1(
  const PointT& reference_point, 
  std::vector<LeafConstPtr> &neighbors) const
{
	neighbors.clear();
	return getNeighborhoodAtPoint(Eigen::MatrixXi::Zero(3,1), reference_point, neighbors);
}


// 从高斯分布中采样点云，用于可视化
template<typename PointT> void
pclomp::IncrementalVoxelGridCovariance<PointT>::getDisplayCloud (
  pcl::PointCloud<pcl::PointXYZI>& cell_cloud, const int& num_points_per_voxel)
{
  cell_cloud.clear ();

  const int pnt_per_cell = num_points_per_voxel > 0 ? num_points_per_voxel : 1000; // official version is 1000.
  boost::mt19937 rng;
  boost::normal_distribution<> nd (0.0, leaf_size_.head (3).norm ());
  boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > var_nor (rng, nd);

  Eigen::LLT<Eigen::Matrix3d> llt_of_cov;
  Eigen::Matrix3d cholesky_decomp;
  Eigen::Vector3d cell_mean;
  Eigen::Vector3d rand_point;
  Eigen::Vector3d dist_point;

  // ************** incremental mode **************
  if (enable_incremetal_mode_) {
    for (auto it = voxels_list_.begin(); it != voxels_list_.end(); ++it) {
      const Eigen::Array3i key = it->first;
      Leaf& leaf = *(it->second);
      if (leaf.nr_points >= min_points_per_voxel_) {
        cell_mean = leaf.mean_;
        llt_of_cov.compute (leaf.cov_);
        cholesky_decomp = llt_of_cov.matrixL ();

        // Random points generated by sampling the normal distribution given by voxel mean and covariance matrix
        for (int i = 0; i < pnt_per_cell; i++)
        {
          rand_point = Eigen::Vector3d (var_nor (), var_nor (), var_nor ());
          dist_point = cell_mean + cholesky_decomp * rand_point;
          pcl::PointXYZI point_to_save;
          point_to_save.x = static_cast<float> (dist_point (0));
          point_to_save.y = static_cast<float> (dist_point (1));
          point_to_save.z = static_cast<float> (dist_point (2));
          point_to_save.intensity = static_cast<float> (leaf.nr_points);
          cell_cloud.push_back (point_to_save);
        }
      }
    }
    return;
  }

  // ************** traditional mode **************
  // Generate points for each occupied voxel with sufficient points.
  for (auto it = leaves_.begin (); it != leaves_.end (); ++it)
  {
    Leaf& leaf = it->second;

    if (leaf.nr_points >= min_points_per_voxel_)
    {
      cell_mean = leaf.mean_;
      llt_of_cov.compute (leaf.cov_);
      cholesky_decomp = llt_of_cov.matrixL ();

      // Random points generated by sampling the normal distribution given by voxel mean and covariance matrix
      for (int i = 0; i < pnt_per_cell; i++)
      {
        rand_point = Eigen::Vector3d (var_nor (), var_nor (), var_nor ());
        dist_point = cell_mean + cholesky_decomp * rand_point;
        pcl::PointXYZI point_to_save;
        point_to_save.x = static_cast<float> (dist_point (0));
        point_to_save.y = static_cast<float> (dist_point (1));
        point_to_save.z = static_cast<float> (dist_point (2));
        point_to_save.intensity = static_cast<float> (leaf.nr_points);
        cell_cloud.push_back (point_to_save);
      }
    }
  }
}


#define PCL_INSTANTIATE_IncrementalVoxelGridCovariance(T) template class PCL_EXPORTS pcl::IncrementalVoxelGridCovariance<T>;

#endif    // PCL_INCREMENTAL_VOXEL_GRID_COVARIANCE_OMP_IMPL_H_

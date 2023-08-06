/**
 * Copyright 2022 WANG_Guanhua(guanhuamail@163.com)
 * 
 * Software License Agreement (BSD License)
 * 
 * All rights reserved.
*/

#include <iostream>
// #include <ros/ros.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
// #include <pcl/registration/ndt.h>
// #include <pcl/registration/gicp.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/pcl_visualizer.h>

#include "pclomp/incremental_ndt_omp.h"

/**
 * \brief In this demo, we focus on the incremental-updating of gaussian distribution inside one single voxel.
 * try to give a 'L'-like point cloud first, then add more points to make it '+'-like, then add more points along height.
 * [WANG_Guanhua 2023/06] 
 * 
 * run_update_voxel_map_distribution_demo
 * run_update_voxel_distribution_demo
 * 
 * run_update_voxel_map_gaussian_demo
 * run_update_voxel_gaussian_demo
*/


int main(int argc, char** argv) {
  // if(argc > 2) {
  //   std::cout << "usage: align target.pcd source.pcd" << std::endl;
  //   return 0;
  // }

  // std::string target_pcd = argv[1];
  // std::string source_pcd = argv[2];

  float ui_box_orig = 0;
  float ui_box_len = 1.0;
  Eigen::Vector3d ui_color(0.3, 0.3, 0.3);
  double ui_point_size = 5;

  const float origin_x = 0.5, origin_y = 0.5, origin_z = 0.5;
  const int counts = 5;
  const float step = 0.0999;

  pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud1(new pcl::PointCloud<pcl::PointXYZ>());
  pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud2(new pcl::PointCloud<pcl::PointXYZ>());
  pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud3(new pcl::PointCloud<pcl::PointXYZ>());

  // Draw x-axis, makes a line
  input_cloud1->points.push_back(pcl::PointXYZ(origin_x, origin_y, origin_z));
  for (int i = 1; i <= counts; ++i) {
    input_cloud1->points.push_back(pcl::PointXYZ(origin_x + step * i, origin_y, origin_z));
    input_cloud1->points.push_back(pcl::PointXYZ(origin_x - step * i, origin_y, origin_z));
  }

  // Draw y-axis, makes a plane
  for (int i = 1; i <= counts; ++i) {
    input_cloud2->points.push_back(pcl::PointXYZ(origin_x, origin_y + step * i, origin_z));
    input_cloud2->points.push_back(pcl::PointXYZ(origin_x, origin_y - step * i, origin_z));
  }

  // Draw z-axis, makes a sphere
  for (int i = 1; i <= counts; ++i) {
    input_cloud3->points.push_back(pcl::PointXYZ(origin_x, origin_y, origin_z + step * i));
    input_cloud3->points.push_back(pcl::PointXYZ(origin_x, origin_y, origin_z - step * i));
  }

  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> input1_handler(input_cloud1, 255.0, 69.0, 0.0);
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> input2_handler(input_cloud2, 127.0, 255.0, 0.0);
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> input3_handler(input_cloud3, 205.0, 0.0, 205.0);

  pclomp::IncrementalVoxelGridCovariance<pcl::PointXYZ> incre_voxel_grid;
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_out1(new pcl::PointCloud<pcl::PointXYZI>());
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_out2(new pcl::PointCloud<pcl::PointXYZI>());
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_out3(new pcl::PointCloud<pcl::PointXYZI>());
  incre_voxel_grid.setLeafSize(1.0, 1.0, 1.0);
  incre_voxel_grid.showVerboseInfo(true);

  std::cout << "set input input_cloud1" << std::endl;
  incre_voxel_grid.setInputCloud(input_cloud1);
  incre_voxel_grid.filter();
  incre_voxel_grid.getDisplayCloud(*cloud_out1, 1000);

  std::cout << "set input input_cloud2" << std::endl;
  incre_voxel_grid.setInputCloud(input_cloud2);
  incre_voxel_grid.filter();
  incre_voxel_grid.getDisplayCloud(*cloud_out2, 1000);

  std::cout << "set input input_cloud3" << std::endl;
  incre_voxel_grid.setInputCloud(input_cloud3);
  incre_voxel_grid.filter();
  incre_voxel_grid.getDisplayCloud(*cloud_out3, 1000);

  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> output1_handler(cloud_out1, 186.0, 0.0, 0.0); // 255.0, 99.0, 71.0
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> output2_handler(cloud_out2, 127.0, 255.0, 0.0);
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> output3_handler(cloud_out3, 205.0, 0.0, 205.0);

  // visualization
  bool visualize_input = false;
  if (visualize_input) {
    pcl::visualization::PCLVisualizer vis_input("point cloud inserted to voxel in 3 times");
    vis_input.addPointCloud(input_cloud1, input1_handler, "input1");
    vis_input.addPointCloud(input_cloud2, input2_handler, "input2");
    vis_input.addPointCloud(input_cloud3, input3_handler, "input3");
    vis_input.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, ui_point_size, "input1");
    vis_input.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, ui_point_size, "input2");
    vis_input.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, ui_point_size, "input3");
    vis_input.addCube(ui_box_orig, ui_box_len, ui_box_orig, ui_box_len, ui_box_orig, ui_box_len, ui_color[0], ui_color[1], ui_color[2], "cube");
    vis_input.setShapeRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_REPRESENTATION, 
      pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, "cube");
    vis_input.addCoordinateSystem(0.5);
  }

  pcl::visualization::PCLVisualizer vis_output1("point cloud sampled from gaussian distribution (1st)");
  vis_output1.addPointCloud(cloud_out1, output1_handler, "output1");
  vis_output1.addPointCloud(input_cloud1, input1_handler, "input1");
  vis_output1.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, ui_point_size, "input1");
  vis_output1.addCube(ui_box_orig, ui_box_len, ui_box_orig, ui_box_len, ui_box_orig, ui_box_len, ui_color[0], ui_color[1], ui_color[2], "cube");
  vis_output1.setShapeRenderingProperties(
    pcl::visualization::PCL_VISUALIZER_REPRESENTATION, 
    pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, "cube");
  vis_output1.addCoordinateSystem(0.5);

  pcl::visualization::PCLVisualizer vis_output2("point cloud sampled from gaussian distribution (2nd)");
  vis_output2.addPointCloud(cloud_out2, output2_handler, "output2");
  vis_output2.addPointCloud(input_cloud1, input1_handler, "input1");
  vis_output2.addPointCloud(input_cloud2, input2_handler, "input2");
  vis_output2.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, ui_point_size, "input1");
  vis_output2.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, ui_point_size, "input2");
  vis_output2.addCube(ui_box_orig, ui_box_len, ui_box_orig, ui_box_len, ui_box_orig, ui_box_len, ui_color[0], ui_color[1], ui_color[2], "cube");
  vis_output2.setShapeRenderingProperties(
    pcl::visualization::PCL_VISUALIZER_REPRESENTATION, 
    pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, "cube");
  vis_output2.addCoordinateSystem(0.5);
  
  pcl::visualization::PCLVisualizer vis_output3("point cloud sampled from gaussian distribution (3rd)");
  vis_output3.addPointCloud(cloud_out3, output3_handler, "output3");
  vis_output3.addPointCloud(input_cloud1, input1_handler, "input1");
  vis_output3.addPointCloud(input_cloud2, input2_handler, "input2");
  vis_output3.addPointCloud(input_cloud3, input3_handler, "input3");
  vis_output3.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, ui_point_size, "input1");
  vis_output3.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, ui_point_size, "input2");
  vis_output3.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, ui_point_size, "input3");
  vis_output3.addCube(ui_box_orig, ui_box_len, ui_box_orig, ui_box_len, ui_box_orig, ui_box_len, ui_color[0], ui_color[1], ui_color[2], "cube");
  vis_output3.setShapeRenderingProperties(
    pcl::visualization::PCL_VISUALIZER_REPRESENTATION, 
    pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, "cube");
  vis_output3.addCoordinateSystem(0.5);
  vis_output3.spin();

  // return 0;



  // ################################## test eigen ##################################

  Eigen::Vector3f current_position_(150, 150, 0);
  Eigen::Vector3f bounding_box_size_(100, 100, 60);
  Eigen::Array3f current_limit_max = (current_position_ + 0.5 * bounding_box_size_).array();
  Eigen::Array3f current_limit_min = (current_position_ - 0.5 * bounding_box_size_).array();
  Eigen::Array3f inv_leaf_size = Eigen::Array3f(2.0, 1.0, 0.5);
  Eigen::Array3i indexed_limit_max = (current_limit_max * inv_leaf_size).cast<int>();
  Eigen::Array3i indexed_limit_min = (current_limit_min * inv_leaf_size).cast<int>();
  std::cout << "current_limit_max: " << current_limit_max.transpose() << std::endl;
  std::cout << "current_limit_min: " << current_limit_min.transpose() << std::endl;
  std::cout << "indexed_limit_max: " << indexed_limit_max.transpose() << std::endl;
  std::cout << "indexed_limit_min: " << indexed_limit_min.transpose() << std::endl;

  Eigen::Array3i index1(400, 201, 14);
  Eigen::Array3i index2(199, 101, 0);

  std::cout << "index1: " << index1.transpose() << std::endl;
  if (((index1 - indexed_limit_max) > 0).any()) {
    std::cout << "index1 beyond max point: " << (index1 - indexed_limit_max).transpose() << std::endl;
  }
  if (((index1 - indexed_limit_min) < 0).any()) {
    std::cout << "index1 beyond min point: " << (index1 - indexed_limit_min).transpose() << std::endl;
  }

  std::cout << "index2: " << index2.transpose() << std::endl;
  if (((index2 - indexed_limit_max) > 0).any()) {
    std::cout << "index2 beyond max point: " << (index2 - indexed_limit_max).transpose() << std::endl;
  }
  if (((index2 - indexed_limit_min) < 0).any()) {
    std::cout << "index2 beyond min point: " << (index2 - indexed_limit_min).transpose() << std::endl;
  }

  std::cout << "index1: " << index1.transpose() << std::endl;
  std::cout << "index1 + 0.5: " << (index1.cast<float>() + 0.5).transpose() << std::endl;
  Eigen::Array3f vsdd(1, 2, 0.5);
  std::cout << "(index1 + 0.5) * (1, 2, 0.5): " << ((index1.cast<float>() + 0.5) * vsdd).transpose() << std::endl;


  // ################################## test eigen docomposition ##################################

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigensolver;
  
  Eigen::Vector3d pt_sum;

  Eigen::Matrix3d cov_;
  cov_ <<10.2 , 2.3 , 1.0 , 3.4 , 19.8 , 1.9 , 3.2 , 6.2 , 10.;
  std::cout << "covariance matrix: \n" << cov_ << std::endl;

  Eigen::Matrix3d eigen_val;
  Eigen::Matrix3d evecs_;

  eigensolver.compute (cov_);
  eigen_val = eigensolver.eigenvalues ().asDiagonal ();
  evecs_ = eigensolver.eigenvectors ();

  std::cout << "eigen_val: \n" << eigen_val << std::endl;
  std::cout << "evecs_   : \n" << evecs_ << std::endl;


  // ################################## test smart ptr & ptr ##################################

  std::shared_ptr<std::vector<int>> int_vec_sh_ptr(new std::vector<int>);
  int_vec_sh_ptr->push_back(1);
  int_vec_sh_ptr->push_back(2);

  std::unique_ptr<std::vector<int>> int_vec_uq_ptr(new std::vector<int>);
  int_vec_uq_ptr->push_back(1);
  int_vec_uq_ptr->push_back(2);
  int_vec_uq_ptr->push_back(3);

  auto ptr1 = int_vec_sh_ptr.get();
  auto ptr2 = int_vec_uq_ptr.get();

  std::cout << " shared ptr info: size=" << int_vec_sh_ptr->size() << ", back=" << int_vec_sh_ptr->back() << std::endl;
  std::cout << " shared ptr info: size=" << ptr1->size() << ", back=" << ptr1->back() << std::endl;
  std::cout << " unique ptr info: size=" << int_vec_uq_ptr->size() << ", back=" << int_vec_uq_ptr->back() << std::endl;
  std::cout << " unique ptr info: size=" << ptr2->size() << ", back=" << ptr2->back() << std::endl;



  return 0;
}
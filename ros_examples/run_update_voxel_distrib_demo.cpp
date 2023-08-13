/**
 * Copyright 2022 WANG_Guanhua(guanhuamail@163.com)
 * 
 * Software License Agreement (BSD License)
 * 
 * All rights reserved.
*/

#include <iostream>
#include <thread>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include "pclomp/incremental_ndt_omp.h"
#include "utils/utility.h"

/**
 * \brief In this demo, we focus on the incremental-updating of gaussian distribution inside one single voxel.
 * try to give a 'L'-like point cloud first, then add more points to make it '+'-like, then add more points along height.
 * [WANG_Guanhua 2023/06] 
 * 
 * run_update_voxel_map_distrib_ros_demo
 * run_update_voxel_distrib_ros_demo
 * 
 * run_update_voxel_map_gaussian_demo
 * run_update_voxel_gaussian_demo
*/


int main(int argc, char** argv) {

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

  // Draw x-axis points, makes a line
  input_cloud1->points.push_back(pcl::PointXYZ(origin_x, origin_y, origin_z));
  for (int i = 1; i <= counts - 2; ++i) {
    input_cloud1->points.push_back(pcl::PointXYZ(origin_x + step * i, origin_y, origin_z));
    input_cloud1->points.push_back(pcl::PointXYZ(origin_x - step * i, origin_y, origin_z));
  }

  // Draw y-axis points, together makes a plane
  for (int i = 1; i <= counts + 0; ++i) {
    input_cloud2->points.push_back(pcl::PointXYZ(origin_x, origin_y + step * i, origin_z));
    input_cloud2->points.push_back(pcl::PointXYZ(origin_x, origin_y - step * i, origin_z));
  }

  // Draw z-axis points, together makes a ellipsoid
  for (int i = 1; i <= counts - 1; ++i) {
    input_cloud3->points.push_back(pcl::PointXYZ(origin_x, origin_y, origin_z + step * i));
    input_cloud3->points.push_back(pcl::PointXYZ(origin_x, origin_y, origin_z - step * i));
  }

  // core.
  pclomp::IncrementalVoxelGridCovariance<pcl::PointXYZ> ivoxel_map;
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_out1(new pcl::PointCloud<pcl::PointXYZI>());
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_out2(new pcl::PointCloud<pcl::PointXYZI>());
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_out3(new pcl::PointCloud<pcl::PointXYZI>());
  
  ivoxel_map.setLeafSize(1.0, 1.0, 1.0);
  ivoxel_map.enableInLeafDownsample(false);
  ivoxel_map.logVerboseInfo(true);

  std::cout << "set input: input_cloud1" << std::endl;
  ivoxel_map.setInputCloud(input_cloud1);
  ivoxel_map.filter();
  auto voxel_data1 = ivoxel_map.getVoxelsData();
  ivoxel_map.getDisplayCloud(*cloud_out1, 1000);

  std::cout << "set input: input_cloud2" << std::endl;
  ivoxel_map.setInputCloud(input_cloud2);
  ivoxel_map.filter();
  auto voxel_data2 = ivoxel_map.getVoxelsData();
  ivoxel_map.getDisplayCloud(*cloud_out2, 1000);

  std::cout << "set input: input_cloud3" << std::endl;
  ivoxel_map.setInputCloud(input_cloud3);
  ivoxel_map.filter();
  auto voxel_data3 = ivoxel_map.getVoxelsData();
  ivoxel_map.getDisplayCloud(*cloud_out3, 1000);

  // visualization through ros rviz
  bool visualize_through_ros = true;
  if (visualize_through_ros) {
    ros::init(argc, argv, "run_update_voxel_distrib_demo");
    ros::NodeHandle node_handle_;
    auto pub_voxel_points = node_handle_.advertise<sensor_msgs::PointCloud2>("/voxel_points", 1);
    auto pub_voxel_ellipsoid = node_handle_.advertise<visualization_msgs::Marker>("/voxel_ellipsoid", 1);

    sensor_msgs::PointCloud2 ros_cloud1, ros_cloud2, ros_cloud3;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_all_(new pcl::PointCloud<pcl::PointXYZ>());
    (*cloud_all_) += (*input_cloud1);
    pcl::toROSMsg(*cloud_all_, ros_cloud1);
    (*cloud_all_) += (*input_cloud2);
    pcl::toROSMsg(*cloud_all_, ros_cloud2);
    (*cloud_all_) += (*input_cloud3);
    pcl::toROSMsg(*cloud_all_, ros_cloud3);
    ros_cloud1.header.frame_id = ros_cloud2.header.frame_id = ros_cloud3.header.frame_id = "map";

    auto ellipsoid1 = ToRosMarkerSphere(voxel_data1->front(), "map", ros::Time::now(), ros::Duration(0), "ellipsoid", 0.8, 3.0);
    auto ellipsoid2 = ToRosMarkerSphere(voxel_data2->front(), "map", ros::Time::now(), ros::Duration(0), "ellipsoid", 0.8, 3.0);
    auto ellipsoid3 = ToRosMarkerSphere(voxel_data3->front(), "map", ros::Time::now(), ros::Duration(0), "ellipsoid", 0.8, 3.0);
    auto cube_lines = ToRosMarkerCubeLines(voxel_data1->front(), "map", ros::Time::now(), ros::Duration(0), "cube", 0.8, 0.005);
    ellipsoid1.color.r = 1.0; ellipsoid1.color.g = 0.0; ellipsoid1.color.b = 0.0;
    ellipsoid2.color.r = 0.0; ellipsoid2.color.g = 1.0; ellipsoid2.color.b = 0.0;
    ellipsoid3.color.r = 0.0; ellipsoid3.color.g = 0.0; ellipsoid3.color.b = 1.0;

    while (ros::ok()) {
      pub_voxel_points.publish(ros_cloud1);
      pub_voxel_ellipsoid.publish(ellipsoid1);
      pub_voxel_ellipsoid.publish(cube_lines);
      std::cout << "published 1st round result: eigen values = [" << voxel_data1->front().evals_.transpose() << "]" << std::endl;
      std::this_thread::sleep_for(std::chrono::duration<double>(3.0));

      pub_voxel_points.publish(ros_cloud2);
      pub_voxel_ellipsoid.publish(ellipsoid2);
      pub_voxel_ellipsoid.publish(cube_lines);
      std::cout << "published 2nd round result: eigen values = [" << voxel_data2->front().evals_.transpose() << "]" << std::endl;
      std::this_thread::sleep_for(std::chrono::duration<double>(3.0));

      pub_voxel_points.publish(ros_cloud3);
      pub_voxel_ellipsoid.publish(ellipsoid3);
      pub_voxel_ellipsoid.publish(cube_lines);
      std::cout << "published 3rd round result: eigen values = [" << voxel_data3->front().evals_.transpose() << "]" << std::endl;
      std::this_thread::sleep_for(std::chrono::duration<double>(5.0));
    }
  }

  // visualization through pcl window
  bool visualize_through_pcl = false;
  if (visualize_through_pcl) {
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> input1_handler(input_cloud1, 255.0, 69.0, 0.0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> input2_handler(input_cloud2, 127.0, 255.0, 0.0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> input3_handler(input_cloud3, 205.0, 0.0, 205.0);

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> output1_handler(cloud_out1, 186.0, 0.0, 0.0); // 255.0, 99.0, 71.0
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> output2_handler(cloud_out2, 127.0, 255.0, 0.0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> output3_handler(cloud_out3, 205.0, 0.0, 205.0);

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
  }


  return 0;
}
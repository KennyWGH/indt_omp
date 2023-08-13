/**
 * Copyright 2022 WANG_Guanhua(guanhuamail@163.com)
 * 
 * Software License Agreement (BSD License)
 * 
 * All rights reserved.
*/

#include "utility.h"

#include "geometry_msgs/Point.h"

namespace {

geometry_msgs::Point toGeometryPoint(const Eigen::Array3f& position) 
{
  geometry_msgs::Point point;
  point.x = position[0];
  point.y = position[1];
  point.z = position[2];
  return point;
}

}

namespace param {
  //
  double xxx = 0;
}


Eigen::Vector3f GenerateRGB(const float value) 
{
  float factor = value;
  factor = value > 1.f ? 1.f : value;
  factor = value < 0.f ? 0.f : value;
  return Eigen::Vector3f(factor, 1 - factor, 0);
}


visualization_msgs::Marker ToRosMarkerSphere(
  const pclomp::VoxelData& voxel_data,
  const std::string& frame_id, 
  const ros::Time& stamp,
  const ros::Duration& duration,
  const std::string& namesapce,
  const float& transparency,
  const float& scale)
{
  // assign info
  visualization_msgs::Marker ui_ellipsoid;
  ui_ellipsoid.header.frame_id = frame_id;
  ui_ellipsoid.header.stamp = stamp;
  ui_ellipsoid.lifetime = duration;
  ui_ellipsoid.ns = namesapce;
  // ui_ellipsoid.id = int(voxel_data.id_); // numeric overflow.
  ui_ellipsoid.id = 0;
  ui_ellipsoid.type = visualization_msgs::Marker::SPHERE;
  ui_ellipsoid.action = visualization_msgs::Marker::ADD;

  ui_ellipsoid.pose.position.x = voxel_data.mean_(0);
  ui_ellipsoid.pose.position.y = voxel_data.mean_(1);
  ui_ellipsoid.pose.position.z = voxel_data.mean_(2);

  // Flip Z axis of eigen vector matrix if determinant is -1
  auto eigenVectors = voxel_data.evecs_;
  Eigen::Quaterniond q;
  if (eigenVectors.determinant() < 0) {
    Eigen::Matrix3d flipZ180 = Eigen::Matrix3d::Identity();
    flipZ180(2,2) = -1;
    q = eigenVectors * flipZ180;
  } else {
    q = eigenVectors;
  }
  q.normalize();
  ui_ellipsoid.pose.orientation.x = q.x();
  ui_ellipsoid.pose.orientation.y = q.y();
  ui_ellipsoid.pose.orientation.z = q.z();
  ui_ellipsoid.pose.orientation.w = q.w();
  ui_ellipsoid.scale.x = sqrt(voxel_data.evals_(0)) * scale; // 2 sigma ~95,4%
  ui_ellipsoid.scale.y = sqrt(voxel_data.evals_(1)) * scale; // 2 sigma ~95,4%
  ui_ellipsoid.scale.z = sqrt(voxel_data.evals_(2)) * scale; // 2 sigma ~95,4%

  // cast normal direction to rgb value.
  // 最小特征值对应特征向量(相当于面特征法向量)的z向分量,决定颜色
  double min_eval_evec_z_projection = std::fabs(eigenVectors(2,0));
  auto rgb = GenerateRGB(min_eval_evec_z_projection);
  ui_ellipsoid.color.a = transparency;
  ui_ellipsoid.color.r = rgb[0];
  ui_ellipsoid.color.g = rgb[1];
  ui_ellipsoid.color.b = rgb[2];

  return ui_ellipsoid;
}


visualization_msgs::Marker ToRosMarkerCubeLines(
  const pclomp::VoxelData& voxel_data,
  const std::string& frame_id, 
  const ros::Time& stamp,
  const ros::Duration& duration,
  const std::string& namesapce,
  const float& transparency,
  const float& line_width) 
{
  // assign info
  visualization_msgs::Marker ui_lines;
  ui_lines.header.frame_id = frame_id;
  ui_lines.header.stamp = stamp;
  ui_lines.lifetime = duration;
  ui_lines.ns = namesapce;
  // ui_lines.id = int(voxel_data.id_); // numeric overflow.
  ui_lines.type = visualization_msgs::Marker::LINE_STRIP;
  ui_lines.action = visualization_msgs::Marker::ADD;

  ui_lines.pose.orientation.w = 1.0;
  ui_lines.scale.x = line_width;
  // ui_lines.LINE_STRIP;

  auto& cube_size = voxel_data.voxel_size_;
  Eigen::Array3f p0 = voxel_data.corner_origin_;
  Eigen::Array3f p1 = p0 + Eigen::Array3f(cube_size[0], 0, 0);
  Eigen::Array3f p2 = p0 + Eigen::Array3f(cube_size[0], cube_size[1], 0);
  Eigen::Array3f p3 = p0 + Eigen::Array3f(0, cube_size[1], 0);
  Eigen::Array3f p4 = p0 + Eigen::Array3f(0, 0, cube_size[2]);
  Eigen::Array3f p5 = p0 + Eigen::Array3f(cube_size[0], 0, cube_size[2]);
  Eigen::Array3f p6 = p0 + Eigen::Array3f(cube_size[0], cube_size[1], cube_size[2]);
  Eigen::Array3f p7 = p0 + Eigen::Array3f(0, cube_size[1], cube_size[2]);

  ui_lines.points.push_back(toGeometryPoint(p0));
  ui_lines.points.push_back(toGeometryPoint(p1));
  ui_lines.points.push_back(toGeometryPoint(p2));
  ui_lines.points.push_back(toGeometryPoint(p3));
  ui_lines.points.push_back(toGeometryPoint(p0));
  ui_lines.points.push_back(toGeometryPoint(p4));
  ui_lines.points.push_back(toGeometryPoint(p5));
  ui_lines.points.push_back(toGeometryPoint(p6));
  ui_lines.points.push_back(toGeometryPoint(p7));
  ui_lines.points.push_back(toGeometryPoint(p4));
  ui_lines.points.push_back(toGeometryPoint(p5));
  ui_lines.points.push_back(toGeometryPoint(p1));
  ui_lines.points.push_back(toGeometryPoint(p2));
  ui_lines.points.push_back(toGeometryPoint(p6));
  ui_lines.points.push_back(toGeometryPoint(p7));
  ui_lines.points.push_back(toGeometryPoint(p3));
  // ui_lines.points.push_back(toGeometryPoint(p2));
  // ui_lines.points.push_back(toGeometryPoint(p6));
  // ui_lines.points.push_back(toGeometryPoint(p5));
  // ui_lines.points.push_back(toGeometryPoint(p1));

  // Flip Z axis of eigen vector matrix if determinant is -1
  auto eigenVectors = voxel_data.evecs_;
  Eigen::Quaterniond q;
  if (eigenVectors.determinant() < 0) {
    Eigen::Matrix3d flipZ180 = Eigen::Matrix3d::Identity();
    flipZ180(2,2) = -1;
    q = eigenVectors * flipZ180;
  } else {
    q = eigenVectors;
  }

  // cast normal direction to rgb value.
  // 最小特征值对应特征向量(相当于面特征法向量)的z向分量
  double min_eval_evec_z_projection = std::fabs(eigenVectors(2,0));
  auto rgb = GenerateRGB(min_eval_evec_z_projection);

  ui_lines.color.a = transparency;
  // ui_lines.color.r = rgb[0];
  // ui_lines.color.g = rgb[1];
  // ui_lines.color.b = rgb[2];
  ui_lines.color.r = 1.0;
  ui_lines.color.g = 1.0;
  ui_lines.color.b = 1.0;

  return ui_lines;
}


bool LoadKeyframePoses(const std::string& file_path, std::map<double, TimedPose<double>>& poses)
{
  //
  return true;
}





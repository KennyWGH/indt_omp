/**
 * Copyright 2022 WANG_Guanhua(guanhuamail@163.com)
 * 
 * Software License Agreement (BSD License)
 * 
 * All rights reserved.
*/


#include <vector>
#include <string>
#include <cstddef>
#include <chrono>
#include <numeric>
#include <cstdlib>
#include <map>
#include <unordered_map>
#include <boost/filesystem.hpp>

#include "ros/ros.h"
#include "nav_msgs/Path.h"
#include "nav_msgs/Odometry.h"
#include "sensor_msgs/PointCloud2.h"
#include "visualization_msgs/Marker.h"
#include "visualization_msgs/MarkerArray.h"
#include "tf2_ros/transform_broadcaster.h"
#include "pcl_conversions/pcl_conversions.h"

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include "pclomp/incremental_voxel_grid_covariance_omp.h"


namespace param {
  //
  extern double xxx;
}


/// @brief cast a float value [0,1] to rgb values.
/// @param value 
/// @return 
Eigen::Vector3f GenerateRGB(const float value);


/// @brief Draw ellipsoid of normal distribution.
/// @param voxel_data 
/// @param frame_id 
/// @param stamp 
/// @param duration 
/// @param namesapce 
/// @return 
visualization_msgs::Marker ToRosMarkerSphere(
  const pclomp::VoxelData& voxel_data,
  const std::string& frame_id, 
  const ros::Time& stamp,
  const ros::Duration& duration,
  const std::string& namesapce,
  const float& transparency = 0.8,
  const float& scale = 3.0);


/// @brief Draw cube edge-lines for a voxel.
/// @param voxel_data 
/// @param frame_id 
/// @param stamp 
/// @param duration 
/// @param namesapce 
/// @return 
visualization_msgs::Marker ToRosMarkerCubeLines(
  const pclomp::VoxelData& voxel_data,
  const std::string& frame_id, 
  const ros::Time& stamp,
  const ros::Duration& duration,
  const std::string& namesapce,
  const float& transparency = 0.8,
  const float& line_width = 0.01);


/// @brief A simple time consumption collector.
class DurationCollector {
 public:
  using SteadyClock = std::chrono::steady_clock;
  DurationCollector() {}
  ~DurationCollector() {}
  void Start() {
    time_point_start = SteadyClock::now();
  }
  void Count() {
    single_shot_duration_ = 
      std::chrono::duration_cast<std::chrono::duration<double>>(
        SteadyClock::now() - time_point_start).count();
    durations_.push_back(single_shot_duration_);
  }
  double GetDuration() {
    return single_shot_duration_;
  }
  std::vector<double> GetData() {
    return durations_;
  }
  double GetAveDuration() {
    if (!durations_.empty()) {
      auto sum = 
        std::accumulate(durations_.begin(), durations_.end(), 0.0);
      auto ave = sum / durations_.size();
      return ave;
    }
    return 0;
  }
  size_t counts() {
    return durations_.size();
  }

 private:
  double single_shot_duration_ = 0;
  SteadyClock::time_point time_point_start;
  std::vector<double> durations_;
};


template<typename T>
struct TimedPose {
  // constructors.
  using Quaternion = Eigen::Quaternion<T>;
  using Vector3 = Eigen::Matrix<T, 3, 1>;

  TimedPose()
    : timestamp_(0), 
      orientation_(Quaternion::Identity()), 
      translation_(Vector3::Zero()) {}
  TimedPose(const double& timestamp, 
            const Quaternion& q,
            const Vector3& t = Vector3::Zero())
    : timestamp_(timestamp), 
      orientation_(q), 
      translation_(t) {}
  TimedPose(const double& t, 
            const Eigen::Matrix<T, 4, 4>& pose)
    : timestamp_(t), 
      orientation_(pose.template block<3,3>(0,0)), 
      translation_(pose.template block<3,1>(0,3)) {}
  ~TimedPose() {}

  Eigen::Matrix<T, 4, 4> ToMatrix4() {
    Eigen::Matrix<T, 4, 4> pose = Eigen::Matrix<T, 4, 4>::template Identity();
    // pose.template block<3,3>(0,0) = Eigen::Matrix<T, 3, 3>()
    // pose.template block<3,1>(0,3)
    return pose;
  }

  double timestamp_ = 0;
  Quaternion orientation_;
  Vector3 translation_;
};


bool LoadKeyframePoses(const std::string& file_path, std::map<double, TimedPose<double>>& poses);


struct Keyframe {
  Keyframe(const long& id, const Eigen::Affine3f& pose, 
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud)
    : id_(id), pose_(pose), pointcloud_(cloud) {}
  long id_ = 0;
  double timestamp_ = 0;
  Eigen::Affine3f pose_ = Eigen::Affine3f::Identity();
  pcl::PointCloud<pcl::PointXYZI>::Ptr pointcloud_ = nullptr;
};


template <typename T>
Eigen::Quaternion<T> AngleAxisVectorToRotationQuaternion(
    const Eigen::Matrix<T, 3, 1>& angle_axis) {
  T scale = T(0.5);
  T w = T(1.);
  constexpr double kCutoffAngle = 1e-8;  // We linearize below this angle.
  if (angle_axis.squaredNorm() > kCutoffAngle) {
    const T norm = angle_axis.norm();
    scale = sin(norm / 2.) / norm;
    w = cos(norm / 2.);
  }
  const Eigen::Matrix<T, 3, 1> quaternion_xyz = scale * angle_axis;
  return Eigen::Quaternion<T>(w, quaternion_xyz.x(), quaternion_xyz.y(),
                              quaternion_xyz.z());
}

struct TileInfo {
  long id_;
  Eigen::Array2i index_;
  Eigen::Vector3f center_; // avoid possible eigen-stl conflicts.
  std::string file_path_;
};


inline Eigen::Vector3f getTileCenterCoordinate(const float& tile_size, const Eigen::Array2i& index2d) {
  return Eigen::Vector3f((index2d[0] + 0.5f) * tile_size, (index2d[1] + 0.5f) * tile_size, 0);
}

inline long IndexToKey(const int i, const int j) {
    return i * long(1000000) + j;
}

inline long IndexToKey(const Eigen::Array2i index) {
    return index[0] * long(1000000) + index[1];
}

inline Eigen::Array2i KeyToIndex(const long key) {
    return Eigen::Array2i(key / 1000000, key % 1000000);
}

inline Eigen::Array2i PointToIndex(const Eigen::Vector3f& point, const float& tile_size) {
    return Eigen::Array2i(int(point.x() / tile_size), int(point.y() / tile_size));
}

inline long PointToKey(const Eigen::Vector3f& point, const float& tile_size) {
    return IndexToKey(int(point.x() / tile_size), int(point.y() / tile_size));
}

//

// std::un


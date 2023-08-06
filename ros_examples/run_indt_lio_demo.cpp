
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

#include <pcl/filters/voxel_grid.h>

#include "ros/ros.h"
#include "nav_msgs/Path.h"
#include "nav_msgs/Odometry.h"
#include "sensor_msgs/PointCloud2.h"
#include "tf2_ros/transform_broadcaster.h"
#include "pcl_conversions/pcl_conversions.h"

#include "pclomp/incremental_ndt_omp.h"

/**
 * \brief Run iNDT-based Lidar-Odom Demo.
 * [WANG_Guanhua 2023/06]
*/

std::string pointcloud2_topic = "/points_raw";
std::string indt_scan_topic = "/indt_scan";
std::string indt_map_topic = "/indt_map";
ros::Subscriber sub_pointcloud;
ros::Publisher pub_indt_scan;
ros::Publisher pub_indt_map;
std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

const std::string map_frame_id = "map";
const std::string lidar_frame_id = "lidar";
const double kRadToDeg = 180. / M_PI;

struct Keyframe {
  Keyframe(const long& id, const Eigen::Matrix4f& pose, 
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud)
    : id_(id), pose_(pose), pointcloud_(cloud) {}
  long id_ = 0;
  double timestamp_ = 0;
  Eigen::Matrix4f pose_ = Eigen::Matrix4f::Identity();
  pcl::PointCloud<pcl::PointXYZI>::Ptr pointcloud_ = nullptr;
};

bool inited_ = false;
size_t received_cloud_counts_ = 0;
Eigen::Matrix4f last_pose_ = Eigen::Matrix4f::Identity();
size_t kf_counts_ = 0;
long current_kf_id_ = 0;
Eigen::Matrix4f last_kf_pose_ = Eigen::Matrix4f::Identity();
std::deque<Keyframe> kf_cache_;

pclomp::IncrementalNDT<pcl::PointXYZI, pcl::PointXYZI>::Ptr iNDT_;

bool debug_mode_ = true;
int visual_points_per_voxel = 10;

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

DurationCollector alignment_duration_collector_;
DurationCollector update_map_duration_collector_;
DurationCollector retrive_points_duration_collector_;

void HandlePointCloud2Message(const sensor_msgs::PointCloud2::ConstPtr& msg);

void PublishTfMessage(const Eigen::Quaterniond orient, const Eigen::Vector3d position);


int main(int argc, char** argv) {
  std::cout << "usage format : run_indt_lo_demo [point cloud topic] [debug mode] [num points per ndt voxel]" << std::endl;
  std::cout << "usage example: run_indt_lo_demo /points_raw true 10" << std::endl;
  if (argc >= 2) {
    if (std::string(argv[1]).find("help") != std::string::npos) {
      return 0;
    }
    std::cout << "parsed point-cloud topic from command-line arguments: " << argv[1] << std::endl;
    pointcloud2_topic = argv[1];
  }
  if (argc >= 3) {
    if (argv[2] == "true") {
      debug_mode_ = true;
    } else if (argv[2] == "false") {
      debug_mode_ = false;
    } else {
      std::cout << "invalid command-line arguments: " << argv[2] << std::endl;
    }
  }
  if (argc >= 4) {
    int value = std::stoi(argv[3]);
    if (value < 0 || value > 1000) {
      std::cout << "invalid command-line arguments: " << argv[3] 
        << ", supposed to be integer within [0, 1000]" << std::endl;
    } else {
      visual_points_per_voxel = value;
    }
  }
  std::cout << "[report param] pointcloud2_topic = " << pointcloud2_topic << std::endl;
  std::cout << "[report param] debug_mode_ = " << debug_mode_ << std::endl;
  std::cout << "[report param] visual_points_per_voxel = " << visual_points_per_voxel << std::endl;

  iNDT_.reset(new pclomp::IncrementalNDT<pcl::PointXYZI, pcl::PointXYZI>());
  iNDT_->setSpatialBoundingBoxForVoxels(Eigen::Vector3f(240, 240, 100));
  iNDT_->setAutoTrimEveryNMeters(10.f);
  iNDT_->setNeighborhoodSearchMethod(pclomp::DIRECT7);
  iNDT_->setMaximumIterations(30);        // by default is 10
  iNDT_->setResolution(1.0);              // by default is 1.0
  iNDT_->setStepSize(0.2);                // by default is 0.1
  iNDT_->setTransformationEpsilon(1e-4);  // by default 0.1
  iNDT_->setNumThreads(8);
  iNDT_->setVerboseMode(debug_mode_);
  // // iNDT_->resetMask();

  ros::init(argc, argv, "run_indt_lo_demo");
  ros::NodeHandle node_handle_;
  sub_pointcloud = node_handle_.subscribe<sensor_msgs::PointCloud2>(
    pointcloud2_topic, 1, HandlePointCloud2Message);
  pub_indt_scan = node_handle_.advertise<sensor_msgs::PointCloud2>(indt_scan_topic, 1);
  pub_indt_map = node_handle_.advertise<sensor_msgs::PointCloud2>(indt_map_topic, 1);
  tf_broadcaster_.reset(new tf2_ros::TransformBroadcaster);

  ::ros::spin();

  std::cout << "[duration report] alignment counts: " << alignment_duration_collector_.counts()
    << ", ave duration: " << alignment_duration_collector_.GetAveDuration() << std::endl;
  std::cout << "[duration report] update map counts: " << update_map_duration_collector_.counts()
    << ", ave duration: " << update_map_duration_collector_.GetAveDuration() << std::endl;
  std::cout << "[duration report] retrive map cloud counts: " << retrive_points_duration_collector_.counts()
    << ", ave duration: " << retrive_points_duration_collector_.GetAveDuration() << std::endl;

  return 0;
}

// **************************************************

void PublishTfMessage(const Eigen::Quaterniond orient, 
  const Eigen::Vector3d position, const ros::Time& ros_time) {
  geometry_msgs::TransformStamped rostf_sensor_to_map;
  rostf_sensor_to_map.header.stamp = ros_time;
  rostf_sensor_to_map.header.frame_id = map_frame_id;
  rostf_sensor_to_map.child_frame_id = lidar_frame_id;
  rostf_sensor_to_map.transform.rotation.w = orient.w();
  rostf_sensor_to_map.transform.rotation.x = orient.x();
  rostf_sensor_to_map.transform.rotation.y = orient.y();
  rostf_sensor_to_map.transform.rotation.z = orient.z();
  rostf_sensor_to_map.transform.translation.x = position.x();
  rostf_sensor_to_map.transform.translation.y = position.y();
  rostf_sensor_to_map.transform.translation.z = position.z();
  tf_broadcaster_->sendTransform(rostf_sensor_to_map);
}

void HandlePointCloud2Message(const sensor_msgs::PointCloud2::ConstPtr& msg) {
  // convert to pcl format.
  pcl::PointCloud<pcl::PointXYZI>::Ptr raw_cloud(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::fromROSMsg(*msg, *raw_cloud);
  double raw_timestamp = msg->header.stamp.toSec();
  received_cloud_counts_++;
  if (received_cloud_counts_ % 20 == 0) {
    std::cout << "received point cloud number since started: " << received_cloud_counts_ << std::endl;
  }

  // preprocess point cloud.
  pcl::PointCloud<pcl::PointXYZI>::Ptr ds_cloud(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::VoxelGrid<pcl::PointXYZI> voxel_filter;
  voxel_filter.setLeafSize(0.2f, 0.2f, 0.2f);
  voxel_filter.setInputCloud(raw_cloud);
  voxel_filter.filter(*ds_cloud);

  // set global variables.
  pcl::PointCloud<pcl::PointXYZI>::Ptr aligned_cloud(new pcl::PointCloud<pcl::PointXYZI>);
  Eigen::Matrix4f aligned_pose = Eigen::Matrix4f::Identity();
  Eigen::Matrix3d aligned_orient_m = Eigen::Matrix3d::Identity();
  Eigen::Vector3d aligned_position = Eigen::Vector3d::Identity();
  Eigen::Quaterniond aligned_orient_q = Eigen::Quaterniond::Identity();
  Eigen::Matrix4f last_incre = Eigen::Matrix4f::Identity();
  int iteration_times = 0;
  double avg_prob_score = 0;
  bool found_new_keyframe = false;

  // ndt alignment.
  if (!inited_) {
    (*aligned_cloud) = (*ds_cloud);
    last_pose_ = aligned_pose;
  } else {
    alignment_duration_collector_.Start();
    iNDT_->setInputSource(ds_cloud);
    iNDT_->align(*aligned_cloud, last_pose_); // use last aligned pose as prediction.
    iteration_times = iNDT_->getFinalNumIteration();
    avg_prob_score = iNDT_->getTransformationProbability();
    aligned_pose = iNDT_->getFinalTransformation();
    last_incre = iNDT_->getLastIncrementalTransformation();
    aligned_orient_m = aligned_pose.block<3,3>(0,0).cast<double>();
    aligned_position = aligned_pose.block<3,1>(0,3).cast<double>();
    aligned_orient_q = Eigen::Quaterniond(aligned_orient_m);
    last_pose_ = aligned_pose;
    alignment_duration_collector_.Count();
  }
  if (debug_mode_) {
    std::cout << "[NDT Report]: iteration_times=" << iteration_times 
      << ", point_ave_score=" << avg_prob_score 
      << ", duration=" << alignment_duration_collector_.GetDuration() << std::endl;
  }

  // add keyframe.
  if (!inited_) /*first keyframe*/ {
    inited_ = true;
    found_new_keyframe = true;
    update_map_duration_collector_.Start();
    iNDT_->setInputTarget(aligned_cloud);
    update_map_duration_collector_.Count();
    kf_cache_.emplace_back(current_kf_id_, aligned_pose, ds_cloud); // note that keyframe cloud is represented in lidar frame.
    last_kf_pose_ = aligned_pose;
    current_kf_id_++;
    std::cout << "************************************************** " << std::endl;
    std::cout << "initialized iNDT with first keyframe, pts = " << ds_cloud->size() << std::endl;
  } else /*motion filter*/ {
    const auto delta_pose = last_kf_pose_.inverse() *  aligned_pose;
    Eigen::Vector3f delta_trans = delta_pose.block<3,1>(0,3);
    Eigen::Matrix3f delta_rotat = delta_pose.block<3,3>(0,0);
    float delta_angle = Eigen::AngleAxisf(delta_rotat).angle() * kRadToDeg;
    float delta_distance = delta_trans.norm();
    if (delta_distance > 0.2 || delta_angle > 5.0) {
      found_new_keyframe = true;
      update_map_duration_collector_.Start();
      iNDT_->setInputTarget(aligned_cloud);
      update_map_duration_collector_.Count();
      kf_cache_.emplace_back(current_kf_id_, aligned_pose, ds_cloud);
      last_kf_pose_ = aligned_pose;
      current_kf_id_++;
      std::cout << "delta_p = " << delta_distance << ", delta_angle = " << delta_angle 
        << "), map_update_duration = " << update_map_duration_collector_.GetDuration() 
        << ", add new keyframe (id=" << current_kf_id_ 
        << ", pts=" << ds_cloud->size() << ")" << std::endl;
    }
  }

  // publish aligned pose & cloud to ros.
  PublishTfMessage(aligned_orient_q, aligned_position, msg->header.stamp);
  if (pub_indt_scan.getNumSubscribers() > 0) {
    sensor_msgs::PointCloud2 aligned_ros_cloud;
    pcl::toROSMsg(*aligned_cloud, aligned_ros_cloud);
    aligned_ros_cloud.header.stamp = msg->header.stamp;
    aligned_ros_cloud.header.frame_id = map_frame_id;
    pub_indt_scan.publish(aligned_ros_cloud);
  }
  if (pub_indt_map.getNumSubscribers() > 0 && debug_mode_ && found_new_keyframe) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr ndt_map_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    retrive_points_duration_collector_.Start();
    iNDT_->getDisplayCloud(*ndt_map_cloud, visual_points_per_voxel);
    retrive_points_duration_collector_.Count();
    sensor_msgs::PointCloud2 ndt_map_ros_cloud;
    pcl::toROSMsg(*ndt_map_cloud, ndt_map_ros_cloud);
    ndt_map_ros_cloud.header.stamp = msg->header.stamp;
    ndt_map_ros_cloud.header.frame_id = map_frame_id;
    pub_indt_map.publish(ndt_map_ros_cloud);
  }

  // done.
  return;
}
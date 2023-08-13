
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
#include <unordered_map>
#include <boost/filesystem.hpp>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>

#include <ros/ros.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <tf2_ros/transform_broadcaster.h>
#include <pcl_conversions/pcl_conversions.h>

#include "pclomp/incremental_ndt_omp.h"
#include "utils/utility.h"

/**
 * \brief [WANG_Guanhua 2023/06] Run iNDT-based Lidar-Localization Demo. 
*/

std::string map_path = "";
std::string pointcloud2_topic = "/points_raw";
std::string imu_topic = "/imu_correct";
std::string indt_scan_topic = "/indt_scan";
std::string indt_map_tiles_topic = "/indt_loaded_map";
std::string indt_map_points_topic = "/indt_map_points";
std::string indt_map_ellipsoids_topic = "/indt_map_ellipsoids";
std::string indt_map_boxes_topic = "/indt_map_boxes";
std::string indt_path_topic = "/indt_path";
ros::Subscriber sub_pointcloud;
ros::Subscriber sub_imu;
ros::Publisher pub_indt_scan;
ros::Publisher pub_loaded_tiles;
ros::Publisher pub_indt_map_points;
ros::Publisher pub_indt_map_ellipsoids;
ros::Publisher pub_indt_map_boxes;
ros::Publisher pub_indt_path;
std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
nav_msgs::Path ros_path;

const std::string map_frame_id = "map";
const std::string lidar_frame_id = "lidar";
const double kRadToDeg = 180. / M_PI;

bool debug_mode_ = true;
bool visualize_inserted_points = true;
bool use_imu_prediction = true;

bool inited_ = false;
size_t received_cloud_counts_ = 0;
Eigen::Affine3f last_pose_ = Eigen::Affine3f::Identity();
bool init_imu_interp_success = false;
bool last_imu_interp_success = false;
Eigen::Quaterniond init_imu_orient = Eigen::Quaterniond::Identity();
Eigen::Quaterniond last_imu_orient = Eigen::Quaterniond::Identity();

float map_tile_size_ = 50.0;
float map_preload_range = 130.0;
float map_unload_range = 180.0;
float check_every_n_meters = 10.0;
Eigen::Vector3f last_checked_position_;
bool map_updated_ = false;
std::vector<TileInfo> registered_tiles_list_;
std::unordered_map<long, pcl::PointCloud<pcl::PointXYZI>::Ptr> loaded_tiles_;

size_t kf_counts_ = 0;
long current_kf_id_ = 0;
Eigen::Affine3f last_kf_pose_ = Eigen::Affine3f::Identity();
std::deque<Keyframe> kf_cache_;
std::deque<TimedPose<double>> imu_pose_cache_;

pclomp::IncrementalNDT<pcl::PointXYZI, pcl::PointXYZI>::Ptr iNDT_;

DurationCollector alignment_duration_collector_;
DurationCollector update_map_duration_collector_;
DurationCollector retrive_points_duration_collector_;

void HandleImuMessage(const sensor_msgs::Imu::ConstPtr& msg);

void HandlePointCloud2Message(const sensor_msgs::PointCloud2::ConstPtr& msg);

void PublishTfMessage(
  const Eigen::Quaternionf orient, const Eigen::Vector3f position, 
  const ros::Time& ros_time, const std::string& child_frameid);

// **********************************************************************************

int main(int argc, char** argv) {
  std::cout << "usage format : run_indt_loc_demo [map path] [pointcloud topic] [imu topic] [debug mode]" << std::endl;
  std::cout << "usage example: run_indt_loc_demo /home/xxx/tiles/ /points_raw /imu_correct true" << std::endl;
  if (argc >= 2) {
    if (std::string(argv[1]).find("help") != std::string::npos) {
      return 0;
    }
    std::cout << "parsed map path from command-line argument: " << argv[1] << std::endl;
    map_path = argv[1];
  }
  if (argc >= 3) {
    std::cout << "parsed pointcloud topic from command-line argument: " << argv[2] << std::endl;
    pointcloud2_topic = argv[2];
  }
  if (argc >= 4) {
    std::cout << "parsed imu topic from command-line argument: " << argv[3] << std::endl;
    imu_topic = argv[3];
  }
  if (argc >= 5) {
    if (argv[4] == "true") {
      debug_mode_ = true;
    } else if (argv[4] == "false") {
      debug_mode_ = false;
    } else {
      std::cout << "invalid command-line argument: " << argv[4] << std::endl;
    }
  }
  
  std::cout << "[report param]  map_path = " << map_path << std::endl;
  std::cout << "[report param]  pointcloud2_topic = " << pointcloud2_topic << std::endl;
  std::cout << "[report param]          imu_topic = " << imu_topic << std::endl;
  std::cout << "[report param] use_imu_prediction = " << use_imu_prediction << std::endl;
  std::cout << "[report param]        debug_mode_ = " << debug_mode_ << std::endl; 
  
  // search map tiles.
  boost::filesystem::path path(map_path);
  if (boost::filesystem::exists(path)) {
    boost::filesystem::directory_iterator dir(path);
    long count = 0;
    for (const auto &iter : dir) {
      auto file_name = iter.path().filename().string();
      if (file_name.find("tile_") != std::string::npos && 
        file_name.find(".pcd") != std::string::npos) {
        count++;
        std::stringstream ss;
        ss << file_name;
        int x, y;
        char prefix[5], delim, suffix[4];
        ss >> prefix[0] >> prefix[1] >> prefix[2] >> prefix[3] >> prefix[4] 
          >> x >> delim >> y >> suffix[0] >> suffix[1] >> suffix[2] >> suffix[3];
        // std::cout << file_name << ": " << prefix << " " << x << " " << delim << " " << y << " " 
        //   << suffix[0] << suffix[1] << suffix[2] << suffix[3] << " | count=" << count << std::endl;

        Eigen::Array2i index(x, y);
        registered_tiles_list_.push_back({IndexToKey(index), index, 
          getTileCenterCoordinate(map_tile_size_, index), 
          map_path + file_name});
      }
    }
    std::cout << "parsed tiles count: " << registered_tiles_list_.size() << std::endl;
    if (registered_tiles_list_.empty()) {
      std::cout << "found no map data." << std::endl;
      return -1;
    }
  } else {
    std::cout << "map path doesn't exist: " << map_path << std::endl;
    return -1;
  }

  // initialize iNDT.
  iNDT_.reset(new pclomp::IncrementalNDT<pcl::PointXYZI, pcl::PointXYZI>());
  iNDT_->enableIncrementalMode(true);     // by default is true
  iNDT_->setSpatialBoundingBoxForVoxels(Eigen::Vector3f(240, 240, 100));
  iNDT_->setAutoTrimEveryNMeters(false, 10.f);  // we use manual-trim.
  iNDT_->setNeighborhoodSearchMethod(pclomp::DIRECT7);
  iNDT_->enableInLeafDownsample(true, 10);    // by default is false
  iNDT_->setResolution(2.0);              // by default is 1.0
  iNDT_->setStepSize(0.2);                // by default is 0.1
  iNDT_->setMaximumIterations(30);        // by default is 10
  iNDT_->setTransformationEpsilon(1e-4);  // by default 0.1
  iNDT_->setNumThreads(8);
  iNDT_->setVerboseMode(debug_mode_);
  // // iNDT_->resetMask();

  ros::init(argc, argv, "run_indt_loc_demo");
  ros::NodeHandle node_handle_;
  sub_pointcloud = node_handle_.subscribe<sensor_msgs::PointCloud2>(
    pointcloud2_topic, 1, HandlePointCloud2Message);
  sub_imu = node_handle_.subscribe<sensor_msgs::Imu>(
    imu_topic, 1, HandleImuMessage);
  pub_indt_scan = node_handle_.advertise<sensor_msgs::PointCloud2>(indt_scan_topic, 1);
  pub_loaded_tiles = node_handle_.advertise<sensor_msgs::PointCloud2>(indt_map_tiles_topic, 1);
  pub_indt_map_points = node_handle_.advertise<sensor_msgs::PointCloud2>(indt_map_points_topic, 1);
  pub_indt_map_ellipsoids = node_handle_.advertise<visualization_msgs::MarkerArray>(indt_map_ellipsoids_topic, 1);
  pub_indt_map_boxes = node_handle_.advertise<visualization_msgs::MarkerArray>(indt_map_boxes_topic, 1);
  pub_indt_path = node_handle_.advertise<nav_msgs::Path>(indt_path_topic, 1);
  tf_broadcaster_.reset(new tf2_ros::TransformBroadcaster);

  std::cout << "******************** System Started. ******************** " << std::endl;

  // give a correction for initial pose.
  last_pose_;
  Eigen::Affine3f correction(Eigen::AngleAxisf(-0.5 * M_PI, Eigen::Vector3f::UnitZ()));
  last_pose_ = last_pose_ * correction;

  ::ros::spin();

  std::cout << "[duration report] alignment counts: " << alignment_duration_collector_.counts()
    << ", ave duration: " << alignment_duration_collector_.GetAveDuration() << std::endl;
  std::cout << "[duration report] update map counts: " << update_map_duration_collector_.counts()
    << ", ave duration: " << update_map_duration_collector_.GetAveDuration() << std::endl;
  std::cout << "[duration report] retrive map cloud counts: " << retrive_points_duration_collector_.counts()
    << ", ave duration: " << retrive_points_duration_collector_.GetAveDuration() << std::endl;

  return 0;
}

// **********************************************************************************

void HandleImuMessage(const sensor_msgs::Imu::ConstPtr& msg) {
  if (imu_pose_cache_.empty()) {
    imu_pose_cache_.emplace_back(msg->header.stamp.toSec(), Eigen::Matrix4d::Identity());
    return;
  }

  double fresh_imu_ts = msg->header.stamp.toSec();
  const double delta_t = fresh_imu_ts - imu_pose_cache_.back().timestamp_;
  if (delta_t <= 0) { return; }

  // rotation integration.
  Eigen::Matrix<double, 3, 1> delta_angle(
    msg->angular_velocity.x * delta_t, 
    msg->angular_velocity.y * delta_t, 
    msg->angular_velocity.z * delta_t);
  Eigen::Quaterniond delta_q = AngleAxisVectorToRotationQuaternion(delta_angle);
  Eigen::Quaterniond orientation = imu_pose_cache_.back().orientation_ * delta_q;
  imu_pose_cache_.emplace_back(fresh_imu_ts, orientation);

  while (imu_pose_cache_.back().timestamp_ - imu_pose_cache_.front().timestamp_ > 5.0) {
    imu_pose_cache_.pop_front();
  }

  return;
}

void HandlePointCloud2Message(const sensor_msgs::PointCloud2::ConstPtr& msg) {
  // convert to pcl format.
  pcl::PointCloud<pcl::PointXYZI>::Ptr raw_cloud(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::fromROSMsg(*msg, *raw_cloud);
  const double fresh_cloud_ts = msg->header.stamp.toSec();
  received_cloud_counts_++;
  if (received_cloud_counts_ % 20 == 0) {
    const double imu_cache_duration = imu_pose_cache_.empty() ? 0 
      : imu_pose_cache_.back().timestamp_ - imu_pose_cache_.front().timestamp_;
    std::cout << "##### received point cloud num since started: " << received_cloud_counts_ << std::endl;
    std::cout << "##### current imu pose queue, counts=" << imu_pose_cache_.size() 
      << ", duration=" << imu_cache_duration << std::endl;
  }

  // preprocess point cloud.
  pcl::PointCloud<pcl::PointXYZI>::Ptr ds_cloud(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::VoxelGrid<pcl::PointXYZI> voxel_filter;
  voxel_filter.setLeafSize(0.2f, 0.2f, 0.2f);
  voxel_filter.setInputCloud(raw_cloud);
  voxel_filter.filter(*ds_cloud);

  // set global variables for alignment.
  pcl::PointCloud<pcl::PointXYZI>::Ptr aligned_cloud(new pcl::PointCloud<pcl::PointXYZI>);
  Eigen::Affine3f aligned_pose = Eigen::Affine3f::Identity();
  Eigen::Matrix3f aligned_orient_m = Eigen::Matrix3f::Identity();
  Eigen::Vector3f aligned_position = Eigen::Vector3f::Identity();
  Eigen::Quaternionf aligned_orient_q = Eigen::Quaternionf::Identity();
  int iteration_times = 0;
  double avg_prob_score = 0;

  // pose prediction by IMU integration.
  bool current_imu_interp_success = false;
  Eigen::Quaterniond curr_imu_orient = Eigen::Quaterniond::Identity();
  if (imu_pose_cache_.size() > 10) {
    for (auto it = imu_pose_cache_.rbegin(); it != imu_pose_cache_.rend(); ++it) {
      if (std::next(it) == imu_pose_cache_.rend()) {
        break;
      }
      if (std::next(it)->timestamp_ < fresh_cloud_ts && it->timestamp_ > fresh_cloud_ts) {
        curr_imu_orient = std::next(it)->orientation_;
        current_imu_interp_success = true;
      }
    }
  }
  bool delta_orient_success = false;
  Eigen::Quaterniond delta_orient = Eigen::Quaterniond::Identity();
  if (current_imu_interp_success && last_imu_interp_success) {
    delta_orient = last_imu_orient.inverse() * curr_imu_orient;
    delta_orient_success = true;
  }
  if (current_imu_interp_success) {
    last_imu_orient = curr_imu_orient;
    last_imu_interp_success = true;
  }
  Eigen::Affine3f pose_prediction(last_pose_);
  if (delta_orient_success) {
    Eigen::Affine3f incre(delta_orient.matrix().cast<float>());
    if (use_imu_prediction) {
      pose_prediction = last_pose_ * incre; // imu prediction sometimes sucks.
    } else {
      std::cout << "[--- Warning --] align without imu prediction [manually disabled]" << std::endl;
    }
    
  } else {
    std::cout << "[--- Warning --] align without imu prediction" << std::endl;
  }


  // preload map tiles.
  Eigen::Vector3f reference_position = pose_prediction.translation();
  float distance_since_last_check = (reference_position - last_checked_position_).norm();
  bool need_check = distance_since_last_check > check_every_n_meters;
  if (need_check) { last_checked_position_ = reference_position; }
  if (need_check || loaded_tiles_.empty() ) {
    for (const auto& tile : registered_tiles_list_) {
      if ((reference_position - tile.center_).head<2>().norm() < map_preload_range) {
        if (loaded_tiles_.find(tile.id_) == loaded_tiles_.end()) {
          pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_(new pcl::PointCloud<pcl::PointXYZI>);
          auto res = pcl::io::loadPCDFile(tile.file_path_, *cloud_);
          if (res == 0) {
            std::cout << "loaded map tile [" << tile.file_path_ << "]." << std::endl;
            loaded_tiles_.insert({tile.id_, cloud_});
            map_updated_ = true;
            // here.
            update_map_duration_collector_.Start();
            iNDT_->setInputTarget(cloud_);
            inited_ = true;
            update_map_duration_collector_.Count();
            std::cout << "map_update_duration = " << update_map_duration_collector_.GetDuration() 
              << ", pts=" << cloud_->size() << ")" << std::endl;
          } else {
            std::cout << "load map tile [" << tile.file_path_ << "] failed." << std::endl;
          }
        }
      }
    }
  }

  // unload map tiles.
  if (need_check && !loaded_tiles_.empty()) {
    for (auto it = loaded_tiles_.begin(); it != loaded_tiles_.end(); ) {
      auto center = getTileCenterCoordinate(map_tile_size_, KeyToIndex(it->first));
      if ((reference_position - center).head<2>().norm() > map_unload_range) {
        // std::cout << "unloaded map tile [" <<  << "]." << std::endl;
        // 
        Eigen::Vector3f min_p(center[0] - 0.5 * map_tile_size_, 
          center[1] - 0.5 * map_tile_size_, std::numeric_limits<float>::min());
        Eigen::Vector3f max_p(center[0] + 0.5 * map_tile_size_, 
          center[1] + 0.5 * map_tile_size_, std::numeric_limits<float>::max());
        iNDT_->trimBySpecificAreaOnce(min_p, max_p);
        it = loaded_tiles_.erase(it);
        map_updated_ = true;
      } else {
        it++;
      }
    }
  }


  // ndt localization.
  if (!inited_) {
    (*aligned_cloud) = (*ds_cloud);
    last_pose_ = aligned_pose;
  } else {
    alignment_duration_collector_.Start();
    iNDT_->setInputSource(ds_cloud);
    iNDT_->align(*aligned_cloud, pose_prediction.matrix());
    iteration_times = iNDT_->getFinalNumIteration();
    avg_prob_score = iNDT_->getTransformationProbability();
    aligned_pose = Eigen::Affine3f(iNDT_->getFinalTransformation());
    aligned_orient_m = aligned_pose.rotation();
    aligned_position = aligned_pose.translation();
    aligned_orient_q = Eigen::Quaternionf(aligned_orient_m);
    last_pose_ = aligned_pose;
    alignment_duration_collector_.Count();
  }
  if (debug_mode_) {
    std::cout << "[iNDT Alignment]: iteration_times=" << iteration_times 
      << ", point_ave_score=" << avg_prob_score 
      << ", duration=" << alignment_duration_collector_.GetDuration() << std::endl;
  }

  // // The system should initialize with imu.
  // if (!inited_) {
  //   static int try_imu_init_count = 0;
  //   if (!current_imu_interp_success) {
  //     if (try_imu_init_count < 5) {
  //       std::cout << "[--------------] Waiting for imu (" 
  //         << try_imu_init_count << ") ..." << std::endl;
  //       try_imu_init_count++;
  //       return; 
  //     }
  //   } else {
  //     init_imu_orient = curr_imu_orient;
  //     init_imu_interp_success = true;
  //     std::cout << "[--------------] Initialized with imu." << std::endl;
  //   }
  // }

  // publish aligned pose & cloud to ros.
  PublishTfMessage(aligned_orient_q, aligned_position, msg->header.stamp, lidar_frame_id);
  // if (init_imu_interp_success) {
  //   PublishTfMessage((init_imu_orient.inverse() * curr_imu_orient).cast<float>(), 
  //     aligned_position, msg->header.stamp, "imu");
  // }
  if (pub_indt_scan.getNumSubscribers() > 0) {
    sensor_msgs::PointCloud2 aligned_ros_cloud;
    pcl::toROSMsg(*aligned_cloud, aligned_ros_cloud);
    aligned_ros_cloud.header.stamp = msg->header.stamp;
    aligned_ros_cloud.header.frame_id = map_frame_id;
    pub_indt_scan.publish(aligned_ros_cloud);
  }
  if ((pub_indt_map_ellipsoids.getNumSubscribers() > 0 || 
      pub_indt_map_boxes.getNumSubscribers() > 0) &&
      debug_mode_ && map_updated_) {
    auto voxels_data = *(iNDT_->getVoxelsData());
    // publish normal distribution
    if (pub_indt_map_ellipsoids.getNumSubscribers()) {
      visualization_msgs::MarkerArray ellipsoids_array;
      for (size_t i = 0; i < voxels_data.size(); ++i) {
        auto ui_ellipsoid = ToRosMarkerSphere(voxels_data[i], 
          map_frame_id, ros::Time::now(), ros::Duration(10), "ndt_map_ellipsoid");
        ui_ellipsoid.id = i;
        ellipsoids_array.markers.push_back(ui_ellipsoid);
      }
      pub_indt_map_ellipsoids.publish(ellipsoids_array);
    }
    // publish voxel cube
    if (pub_indt_map_boxes.getNumSubscribers()) {
      visualization_msgs::MarkerArray boxes_array;
      for (size_t i = 0; i < voxels_data.size(); ++i) {
        auto ui_lines = ToRosMarkerCubeLines(voxels_data[i],
          map_frame_id, ros::Time::now(), ros::Duration(0), "ndt_map_lines", 0.5);
        ui_lines.id = i;
        boxes_array.markers.push_back(ui_lines);
      }
      pub_indt_map_boxes.publish(boxes_array);
    }
  }
  if (pub_indt_map_points.getNumSubscribers() > 0 && map_updated_) {
    sensor_msgs::PointCloud2 ndt_map_ros_cloud;
    if (!visualize_inserted_points) {
      pcl::PointCloud<pcl::PointXYZI>::Ptr ndt_map_cloud(new pcl::PointCloud<pcl::PointXYZI>);
      retrive_points_duration_collector_.Start();
      iNDT_->getDisplayCloud(*ndt_map_cloud, 10);
      retrive_points_duration_collector_.Count();
      pcl::toROSMsg(*ndt_map_cloud, ndt_map_ros_cloud);
    } else {
      retrive_points_duration_collector_.Start();
      auto inserted_point_cloud = iNDT_->getInLeafPointCloud();
      retrive_points_duration_collector_.Count();
      pcl::toROSMsg(*inserted_point_cloud, ndt_map_ros_cloud);
    }
    ndt_map_ros_cloud.header.stamp = msg->header.stamp;
    ndt_map_ros_cloud.header.frame_id = map_frame_id;
    pub_indt_map_points.publish(ndt_map_ros_cloud);
  }
  if (pub_indt_path.getNumSubscribers() > 0) {
    ros_path.header.stamp = msg->header.stamp;
    ros_path.header.frame_id = map_frame_id;
    geometry_msgs::PoseStamped stamped_pose;
    stamped_pose.pose.position.x = aligned_position.x();
    stamped_pose.pose.position.y = aligned_position.y();
    stamped_pose.pose.position.z = aligned_position.z();
    stamped_pose.pose.orientation.w = aligned_orient_q.w();
    stamped_pose.pose.orientation.x = aligned_orient_q.x();
    stamped_pose.pose.orientation.y = aligned_orient_q.y();
    stamped_pose.pose.orientation.z = aligned_orient_q.z();
    stamped_pose.header.stamp = msg->header.stamp;
    stamped_pose.header.frame_id = map_frame_id;
    ros_path.poses.push_back(stamped_pose);
    ros_path.header.stamp = stamped_pose.header.stamp;
    pub_indt_path.publish(ros_path);
  }

  // done.
  return;
}

void PublishTfMessage(const Eigen::Quaternionf orient, const Eigen::Vector3f position, 
  const ros::Time& ros_time, const std::string& child_frameid) {
  geometry_msgs::TransformStamped rostf_sensor_to_map;
  rostf_sensor_to_map.header.stamp = ros_time;
  rostf_sensor_to_map.header.frame_id = map_frame_id;
  rostf_sensor_to_map.child_frame_id = child_frameid;
  rostf_sensor_to_map.transform.rotation.w = orient.w();
  rostf_sensor_to_map.transform.rotation.x = orient.x();
  rostf_sensor_to_map.transform.rotation.y = orient.y();
  rostf_sensor_to_map.transform.rotation.z = orient.z();
  rostf_sensor_to_map.transform.translation.x = position.x();
  rostf_sensor_to_map.transform.translation.y = position.y();
  rostf_sensor_to_map.transform.translation.z = position.z();
  tf_broadcaster_->sendTransform(rostf_sensor_to_map);
}


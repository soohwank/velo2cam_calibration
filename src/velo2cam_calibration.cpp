/*
  velo2cam_calibration - Automatic calibration algorithm for extrinsic
  parameters of a stereo camera and a velodyne Copyright (C) 2017-2021 Jorge
  Beltran, Carlos Guindel

  This file is part of velo2cam_calibration.

  velo2cam_calibration is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 2 of the License, or
  (at your option) any later version.

  velo2cam_calibration is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with velo2cam_calibration.  If not, see <http://www.gnu.org/licenses/>.
*/

/*
  velo2cam_calibration: Perform the registration step
*/

#define PCL_NO_PRECOMPILE

#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/transforms.h>
#include <ros/package.h>
#include <ros/ros.h>
#include <std_msgs/Empty.h>
#include <std_msgs/Int32.h>
#include <tf/tf.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <tinyxml.h>
#include <velo2cam_calibration/ClusterCentroids.h>
#include <velo2cam_utils.h>

// SWAN: opencv for reprojection errors
#include <opencv2/opencv.hpp>
using namespace cv;

// SWAN: camera matrix and distortion coefficients from the launch file
Mat cameraMatrix(3, 3, CV_32F);
Mat distCoeffs(1, 5, CV_32F);

// SWAN: copy a utility function from mono_qr_pattern.cpp
Point2f projectPointDist(cv::Point3f pt_cv, const Mat intrinsics,
                         const Mat distCoeffs) {
  // Project a 3D point taking into account distortion
  vector<Point3f> input{pt_cv};
  vector<Point2f> projectedPoints;
  projectedPoints.resize(
      1);  // TODO: Do it batched? (cv::circle is not batched anyway)
  projectPoints(input, Mat::zeros(3, 1, CV_64FC1), Mat::zeros(3, 1, CV_64FC1),
                intrinsics, distCoeffs, projectedPoints);
  return projectedPoints[0];
}

Eigen::MatrixXf reviseTransformation(Eigen::Matrix4f &L_T_C_ROS)
{
  // R
  Eigen::MatrixXf L_R_C_ROS(3, 3);
  L_R_C_ROS(0, 0) = L_T_C_ROS(0, 0);
  L_R_C_ROS(0, 1) = L_T_C_ROS(0, 1);
  L_R_C_ROS(0, 2) = L_T_C_ROS(0, 2);
  L_R_C_ROS(1, 0) = L_T_C_ROS(1, 0);
  L_R_C_ROS(1, 1) = L_T_C_ROS(1, 1);
  L_R_C_ROS(1, 2) = L_T_C_ROS(1, 2);
  L_R_C_ROS(2, 0) = L_T_C_ROS(2, 0);
  L_R_C_ROS(2, 1) = L_T_C_ROS(2, 1);
  L_R_C_ROS(2, 2) = L_T_C_ROS(2, 2);

  // t
  Eigen::MatrixXf L_t_C_ROS(3, 1);
  L_t_C_ROS(0, 0) = L_T_C_ROS(0, 3);
  L_t_C_ROS(1, 0) = L_T_C_ROS(1, 3);
  L_t_C_ROS(2, 0) = L_T_C_ROS(2, 3);

  // C_T_L: transform points from lidar coords to camera coords
  Eigen::MatrixXf C_ROS_R_L(3,3);
  Eigen::MatrixXf C_ROS_t_L(3,1);
  C_ROS_R_L = L_R_C_ROS.transpose();
  C_ROS_t_L = - C_ROS_R_L * L_t_C_ROS;  

  // translation from ROS camera to conventional camera
  //
  //      <ROS camera>               <camera>
  //
  //       (forward)    X                  z (forward)
  //           (up) Z   ^                  ^
  //                ^  /                  /
  //                | /                  /
  //                |/                  /
  // (left) Y <------                  --------> x (right)
  //                                   |
  //                                   |
  //                                   |
  //                                   v
  //                                   y (down)

  // SWAN: Why is the rotation matrix transposed???
  Eigen::MatrixXf C_ROS_T_L(4,4); // translation matrix lidar-camera
  C_ROS_T_L << C_ROS_R_L(0), C_ROS_R_L(3), C_ROS_R_L(6), C_ROS_t_L(0),
               C_ROS_R_L(1), C_ROS_R_L(4), C_ROS_R_L(7), C_ROS_t_L(1),
               C_ROS_R_L(2), C_ROS_R_L(5), C_ROS_R_L(8), C_ROS_t_L(2),
               0,        0,        0,        1;

  Eigen::MatrixXf C_T_C_ROS(4,4);
  C_T_C_ROS <<  0, -1,  0,  0,   // x = -Y
                0,  0, -1,  0,   // y = -Z
                1,  0,  0,  0,   // z = X
                0,  0,  0,  1;
  
  Eigen::MatrixXf C_T_L = C_T_C_ROS * C_ROS_T_L;

  return C_T_L;
}

using namespace std;
using namespace sensor_msgs;



ros::Publisher clusters_sensor2_pub, clusters_sensor1_pub;
ros::Publisher colour_sensor2_pub, colour_sensor1_pub;
ros::Publisher sensor_switch_pub;
ros::Publisher iterations_pub;
int nFrames;
bool sensor1Received, sensor2Received;

// SWAN: Cp
std::vector<pcl::PointXYZ> sensor1_vector(4);         // SWAN: Cp for the current target pose
pcl::PointCloud<pcl::PointXYZ>::Ptr sensor1_cloud;    // SWAN: Cp for all target poses
pcl::PointCloud<pcl::PointXYZI>::Ptr isensor1_cloud;  // SWAN: Cp for all target poses with intensity increased by 0.3
std::vector<std::vector<std::tuple<int, int, pcl::PointCloud<pcl::PointXYZ>,
                                   std::vector<pcl::PointXYZ>>>>
    sensor1_buffer;                                   // SWAN: Cp for all target poses

std::vector<pcl::PointXYZ> sensor2_vector(4);         // SWAN: Lp for the current target pose
pcl::PointCloud<pcl::PointXYZ>::Ptr sensor2_cloud;    // SWAN: Lp for all target poses
pcl::PointCloud<pcl::PointXYZI>::Ptr isensor2_cloud;  // SWAN: Lp for all target poses with intensity increased by 0.3
std::vector<std::vector<std::tuple<int, int, pcl::PointCloud<pcl::PointXYZ>,
                                   std::vector<pcl::PointXYZ>>>>
    sensor2_buffer;                                   // SWAN: Lp for all target poses

tf::StampedTransform tf_sensor1_sensor2;

bool is_sensor1_cam, is_sensor2_cam;
bool skip_warmup, single_pose_mode;
string sensor1_frame_id = "";
string sensor1_rotated_frame_id = "";
string sensor2_frame_id = "";
string sensor2_rotated_frame_id = "";

typedef Eigen::Matrix<double, 12, 12> Matrix12d;
typedef Eigen::Matrix<double, 12, 1> Vector12d;

tf::Transform transf;

int S1_WARMUP_COUNT = 0, S2_WARMUP_COUNT = 0;
bool S1_WARMUP_DONE = false, S2_WARMUP_DONE = false;
int TARGET_POSITIONS_COUNT = 0; // SWAN: Number of Target Poses
int TARGET_ITERATIONS = 30;

bool sync_iterations;
bool save_to_file_;
bool publish_tf_;
bool calibration_ended;
bool results_every_pose;

long int sensor1_count, sensor2_count;

std::ofstream savefile;

void sensor1_callback(const velo2cam_calibration::ClusterCentroids::ConstPtr sensor1_centroids);
void sensor2_callback(      velo2cam_calibration::ClusterCentroids::ConstPtr sensor2_centroids);

ros::NodeHandle *nh_;

ros::Subscriber sensor1_sub, sensor2_sub;

void calibrateExtrinsics(int seek_iter = -1) {
  std::vector<pcl::PointXYZ> local_sensor1_vector, local_sensor2_vector;
  pcl::PointCloud<pcl::PointXYZ>::Ptr local_sensor1_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr local_sensor2_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ> local_l_cloud, local_c_cloud;

  int used_sensor2, used_sensor1;

  // Get final frame names for TF broadcaster
  string sensor1_final_transformation_frame = sensor1_frame_id;
  if (is_sensor1_cam) {
    sensor1_final_transformation_frame = sensor1_rotated_frame_id;
  }
  string sensor2_final_transformation_frame = sensor2_frame_id;
  if (is_sensor2_cam) {
    sensor2_final_transformation_frame = sensor2_rotated_frame_id;
  }

  int total_sensor1, total_sensor2;

  if (seek_iter > 0) {  // Add clouds (per sensor) from every position using
                        // last 'seek_iter' detection
    if (DEBUG) ROS_INFO("Seeking %d iterations", seek_iter);

    for (int i = 0; i < TARGET_POSITIONS_COUNT + 1; ++i) {
      if (DEBUG)
        ROS_INFO("Target position: %d, Last sensor2: %d, last sensor1: %d",
                 i + 1, std::get<0>(sensor2_buffer[i].back()),
                 std::get<0>(sensor1_buffer[i].back()));
      // Sensor 1
      auto it1 = std::find_if(
          sensor1_buffer[i].begin(), sensor1_buffer[i].end(),
          [&seek_iter](
              const std::tuple<int, int, pcl::PointCloud<pcl::PointXYZ>,
                               std::vector<pcl::PointXYZ>> &e) {
            return std::get<0>(e) == seek_iter;
          });
      if (it1 == sensor1_buffer[i].end()) {
        ROS_WARN("Could not sync sensor1");
        return;
      }

      local_sensor1_vector.insert(
          local_sensor1_vector.end(), std::get<3>(*it1).begin(),
          std::get<3>(*it1).end());  // Add sorted centers (for equations)
      *local_sensor1_cloud +=
          std::get<2>(*it1);  // Add centers cloud (for registration)
      used_sensor1 = std::get<1>(*it1);
      total_sensor1 = std::get<0>(*it1);

      // Sensor 2
      auto it2 = std::find_if(
          sensor2_buffer[i].begin(), sensor2_buffer[i].end(),
          [&seek_iter](
              const std::tuple<int, int, pcl::PointCloud<pcl::PointXYZ>,
                               std::vector<pcl::PointXYZ>> &e) {
            return std::get<0>(e) == seek_iter;
          });
      if (it2 == sensor2_buffer[i].end()) {
        ROS_WARN("Could not sync sensor2");
        return;
      }

      local_sensor2_vector.insert(
          local_sensor2_vector.end(), std::get<3>(*it2).begin(),
          std::get<3>(*it2).end());  // Add sorted centers (for equations)
      *local_sensor2_cloud +=
          std::get<2>(*it2);  // Add centers cloud (for registration)
      used_sensor2 = std::get<1>(*it2);
      total_sensor2 = std::get<0>(*it2);
    }
    ROS_INFO("Synchronizing cluster centroids");
  }
  else
  { // Add clouds (per sensor) from every position using last available
    // detection
    for (int i = 0; i < TARGET_POSITIONS_COUNT + 1; ++i) {
      // Sensor 1
      local_sensor1_vector.insert(
          local_sensor1_vector.end(),
          std::get<3>(sensor1_buffer[i].back()).begin(),
          std::get<3>(sensor1_buffer[i].back())
              .end());  // Add sorted centers (for equations)
      *local_sensor1_cloud += std::get<2>(
          sensor1_buffer[i].back());  // Add centers cloud (for registration)
      used_sensor1 = std::get<1>(sensor2_buffer[i].back());

      // Sensor 2
      local_sensor2_vector.insert(
          local_sensor2_vector.end(),
          std::get<3>(sensor2_buffer[i].back()).begin(),
          std::get<3>(sensor2_buffer[i].back())
              .end());  // Add sorted centers (for equations)
      *local_sensor2_cloud += std::get<2>(
          sensor2_buffer[i].back());  // Add centers cloud (for registration)
    }
  }

  if (DEBUG) {
    sensor_msgs::PointCloud2 ros_cloud;
    pcl::toROSMsg(*local_sensor2_cloud, ros_cloud);
    ros_cloud.header.frame_id = sensor2_rotated_frame_id;
    clusters_sensor2_pub.publish(ros_cloud);

    pcl::toROSMsg(*local_sensor1_cloud, ros_cloud);
    ros_cloud.header.frame_id = sensor1_frame_id;
    clusters_sensor1_pub.publish(ros_cloud);
  }

  // SVD code
  pcl::PointCloud<pcl::PointXYZ>::Ptr sorted_centers1(
      new pcl::PointCloud<pcl::PointXYZ>());
  pcl::PointCloud<pcl::PointXYZ>::Ptr sorted_centers2(
      new pcl::PointCloud<pcl::PointXYZ>());

  for (int i = 0; i < local_sensor1_vector.size(); ++i) {
    sorted_centers1->push_back(local_sensor1_vector[i]);
    sorted_centers2->push_back(local_sensor2_vector[i]);
  }

  // SWAN: final_transformation = L_T_C_ROS
  //       sorted_centers1: C_ROS_p1 in Camera ROS coords (camera-extracted points in Camera ROS coords)
  //       sorted_centers2: L_p2 in LiDAR coords          ( lidar-extracted points in LiDAR      coords)
  Eigen::Matrix4f final_transformation;
  const pcl::registration::TransformationEstimationSVD<pcl::PointXYZ,
                                                       pcl::PointXYZ>
      trans_est_svd(true);
  trans_est_svd.estimateRigidTransformation(*sorted_centers1, // SWAN: source = mono camera
                                            *sorted_centers2, // SWAN: target = lidar
                                            final_transformation);

  tf::Matrix3x3 tf3d;
  tf3d.setValue(final_transformation(0, 0), final_transformation(0, 1), final_transformation(0, 2),
                final_transformation(1, 0), final_transformation(1, 1), final_transformation(1, 2),
                final_transformation(2, 0), final_transformation(2, 1), final_transformation(2, 2));

  tf::Quaternion tfqt;
  tf3d.getRotation(tfqt);

  tf::Vector3 origin;
  origin.setValue(final_transformation(0, 3), final_transformation(1, 3),
                  final_transformation(2, 3));

  transf.setOrigin(origin);
  transf.setRotation(tfqt);

  static tf::TransformBroadcaster br;
  tf_sensor1_sensor2 = tf::StampedTransform(transf.inverse(), ros::Time::now(),
                                            sensor1_final_transformation_frame,
                                            sensor2_final_transformation_frame);
  if (publish_tf_) br.sendTransform(tf_sensor1_sensor2);

  tf::Transform inverse = tf_sensor1_sensor2.inverse();
  double roll, pitch, yaw;
  double xt = inverse.getOrigin().getX(), yt = inverse.getOrigin().getY(),
         zt = inverse.getOrigin().getZ();
  inverse.getBasis().getRPY(roll, pitch, yaw);

  if (save_to_file_) {
    savefile << seek_iter << ", " << xt << ", " << yt << ", " << zt << ", "
             << roll << ", " << pitch << ", " << yaw << ", " << used_sensor1
             << ", " << used_sensor2 << ", " << total_sensor1 << ", "
             << total_sensor2 << endl;
  }

  cout << setprecision(4) << std::fixed;
  cout << "Calibration finished succesfully." << endl;
  cout << "Extrinsic parameters:" << endl;
  cout << "x = " << xt << "\ty = " << yt << "\tz = " << zt << endl;
  cout << "roll = " << roll << "\tpitch = " << pitch << "\tyaw = " << yaw << endl;

  // SWAN: Print Results
  {
    int pt_count;

    ROS_INFO("[SWAN] *** Sensor 1 (Camera): camera-extracted points in Camera ROS coords  ***");
    if (save_to_file_) savefile << endl << "[SWAN] *** Sensor 1 (Camera): camera-extracted points in Camera ROS coords ***" << endl;
    pt_count = 0;
    for (const auto& pt: sorted_centers1->points)
    {
      ROS_INFO("C_ROS_p1_%d = (%f, %f, %f)", pt_count, pt.x, pt.y, pt.z);
      if (save_to_file_) savefile << "C_ROS_p1_" << pt_count << " = (" << pt.x << ", " << pt.y << ", " << pt.z << ")" << endl;
      pt_count++;
    }

    ROS_INFO("[SWAN] *** Sensor 2 (LiDAR): lidar-extracted points in LiDAR coords ***");
    if (save_to_file_) savefile << endl << "[SWAN] *** Sensor 2 (LiDAR): lidar-extracted points in LiDAR coords ***" << endl;
    pt_count = 0;
    for (const auto& pt: sorted_centers2->points)
    {
      ROS_INFO("L_p2_%d = (%f, %f, %f)", pt_count, pt.x, pt.y, pt.z);
      if (save_to_file_) savefile << "L_p2_" << pt_count << " = (" << pt.x << ", " << pt.y << ", " << pt.z << ")" << endl;
      pt_count++;
    }

    ROS_INFO("[SWAN] *** Transformation: L_T_C_ROS ***");
    if (save_to_file_) savefile << endl << "[SWAN] *** Transformation: L_T_C_ROS ***" << endl;
    for (int row = 0; row < 4; row++)
    {
      ROS_INFO("%f, %f, %f, %f", final_transformation(row, 0),
                                 final_transformation(row, 1),
                                 final_transformation(row, 2),
                                 final_transformation(row, 3));
      if (save_to_file_) savefile << final_transformation(row, 0) << ", " 
                                  << final_transformation(row, 1) << ", " 
                                  << final_transformation(row, 2) << ", " 
                                  << final_transformation(row, 3) << endl;
    }

    ROS_INFO("[SWAN] *** 3D-3D Error ***");
    if (save_to_file_) savefile << endl << "[SWAN] *** 3D-3D Error ***" << endl;
    pt_count = 0;
    float total_error = 0.f;
    pcl::PointCloud<pcl::PointXYZ>::iterator it1 = sorted_centers1->begin();
    pcl::PointCloud<pcl::PointXYZ>::iterator it2 = sorted_centers2->begin();
    for(; it1 != sorted_centers1->end(); it1++, it2++)
    {
      // Points
      pcl::PointXYZ &S_p1 = *it1; // source points in Source coords
      pcl::PointXYZ &T_p2 = *it2; // target points in Target coords

      // T_p1 = T_T_S * S_p1
      pcl::PointXYZ T_p1(final_transformation(0, 0)*(S_p1.x) + final_transformation(0, 1)*(S_p1.y) + final_transformation(0, 2)*(S_p1.z) + final_transformation(0, 3),
                         final_transformation(1, 0)*(S_p1.x) + final_transformation(1, 1)*(S_p1.y) + final_transformation(1, 2)*(S_p1.z) + final_transformation(1, 3),
                         final_transformation(2, 0)*(S_p1.x) + final_transformation(2, 1)*(S_p1.y) + final_transformation(2, 2)*(S_p1.z) + final_transformation(2, 3));
      
      // Error in Camera coords
      float error = sqrt(powf(T_p2.x - T_p1.x, 2) + powf(T_p2.y - T_p1.y, 2) + powf(T_p2.z - T_p1.z, 2));
      ROS_INFO("Error_%d = %f", pt_count, error);
      if (save_to_file_) savefile << "Error_" << pt_count << " = " << error << endl;
      pt_count++;
      total_error += error;
    }
    float average_error = total_error / (float) pt_count;
    ROS_INFO("Average 3D-3D Error = %f", average_error);
    if (save_to_file_) savefile << "Average 3D-3D Error = " << average_error << endl;


    ROS_INFO("[SWAN] *** 3D-2D Error ***");
    if (save_to_file_) savefile << endl << "[SWAN] *** 3D-2D Error ***" << endl;
   
    pt_count = 0;
    total_error = 0.f;
    it1 = sorted_centers1->begin();
    it2 = sorted_centers2->begin();

    // L_T_C_ROS
    Eigen::Matrix4f L_T_C_ROS = final_transformation;

    // C_T_L
    Eigen::MatrixXf C_T_L = reviseTransformation(L_T_C_ROS);

    // C_T_C_ROS
    Eigen::MatrixXf C_T_C_ROS = C_T_L * L_T_C_ROS;

    for(; it1 != sorted_centers1->end(); it1++, it2++)
    {
      // Points in OpenCV
      cv::Point3f C_ROS_p1(it1->x, it1->y, it1->z);
      cv::Point3f L_p2(it2->x, it2->y, it2->z);
      ROS_INFO("3D: C_ROS_p1_%d = (%f, %f, %f)", pt_count, C_ROS_p1.x, C_ROS_p1.y, C_ROS_p1.z);
      ROS_INFO("3D: L_p2_%d = (%f, %f, %f)", pt_count, L_p2.x, L_p2.y, L_p2.z);

      // C_p1 = C_T_C_ROS * C_ROS_p1
      cv::Point3f C_p1(C_T_C_ROS(0, 0)*(C_ROS_p1.x) + C_T_C_ROS(0, 1)*(C_ROS_p1.y) + C_T_C_ROS(0, 2)*(C_ROS_p1.z) + C_T_C_ROS(0, 3),
                       C_T_C_ROS(1, 0)*(C_ROS_p1.x) + C_T_C_ROS(1, 1)*(C_ROS_p1.y) + C_T_C_ROS(1, 2)*(C_ROS_p1.z) + C_T_C_ROS(1, 3),
                       C_T_C_ROS(2, 0)*(C_ROS_p1.x) + C_T_C_ROS(2, 1)*(C_ROS_p1.y) + C_T_C_ROS(2, 2)*(C_ROS_p1.z) + C_T_C_ROS(2, 3));

      ROS_INFO("3D: C_p1_%d = (%f, %f, %f)", pt_count, C_p1.x, C_p1.y, C_p1.z);

      // C_p2 = C_T_L * L_p2
      cv::Point3f C_p2(C_T_L(0, 0)*(L_p2.x) + C_T_L(0, 1)*(L_p2.y) + C_T_L(0, 2)*(L_p2.z) + C_T_L(0, 3),
                       C_T_L(1, 0)*(L_p2.x) + C_T_L(1, 1)*(L_p2.y) + C_T_L(1, 2)*(L_p2.z) + C_T_L(1, 3),
                       C_T_L(2, 0)*(L_p2.x) + C_T_L(2, 1)*(L_p2.y) + C_T_L(2, 2)*(L_p2.z) + C_T_L(2, 3));

      cv::Point2f uv_C = projectPointDist(C_p1, cameraMatrix, distCoeffs);
      cv::Point2f uv_L = projectPointDist(C_p2, cameraMatrix, distCoeffs);
      ROS_INFO("2D: I_p1_%d = (%f, %f)", pt_count, uv_C.x, uv_C.y);
      ROS_INFO("2D: I_p2_%d = (%f, %f)", pt_count, uv_L.x, uv_L.y);
      
      // Error in image coords
      float error = sqrt(powf(uv_C.x - uv_L.x, 2) + powf(uv_C.y - uv_L.y, 2));
      ROS_INFO("Error_%d = %f", pt_count, error);
      if (save_to_file_) savefile << "Error_" << pt_count << " = " << error << endl;
      pt_count++;
      total_error += error;
    }
    average_error = total_error / (float) pt_count;
    ROS_INFO("Average 3D-2D Reprojection Error = %f", average_error);
    if (save_to_file_) savefile << "Average 3D-2D Reprojection Error = " << average_error << endl;

    ROS_INFO("[SWAN] *** PC on Image ***");
    if (save_to_file_) savefile << endl << "[SWAN] *** PC on Image ***" << endl;

    // intrinsics: K
    ROS_INFO("camera_matrix: [%f, %f, %f, 0.0,", cameraMatrix.at<float>(0, 0), cameraMatrix.at<float>(0, 1), cameraMatrix.at<float>(0, 2));
    ROS_INFO("                %f, %f, %f, 0.0,", cameraMatrix.at<float>(1, 0), cameraMatrix.at<float>(1, 1), cameraMatrix.at<float>(1, 2));
    ROS_INFO("                %f, %f, %f, 0.0]", cameraMatrix.at<float>(2, 0), cameraMatrix.at<float>(2, 1), cameraMatrix.at<float>(2, 2));
    if (save_to_file_) savefile << "camera_matrix: [" << cameraMatrix.at<float>(0, 0) << ", " << cameraMatrix.at<float>(0, 1) << ", " << cameraMatrix.at<float>(0, 2) << ", 0.0," << endl;
    if (save_to_file_) savefile << "                " << cameraMatrix.at<float>(1, 0) << ", " << cameraMatrix.at<float>(1, 1) << ", " << cameraMatrix.at<float>(1, 2) << ", 0.0," << endl;
    if (save_to_file_) savefile << "                " << cameraMatrix.at<float>(2, 0) << ", " << cameraMatrix.at<float>(2, 1) << ", " << cameraMatrix.at<float>(2, 2) << ", 0.0]" << endl;

    // distortion coeffients: k1, k2, p1, p2, k3
    ROS_INFO("distortion: [%f, %f, %f, %f, %f]", distCoeffs.at<float>(0, 0), 
                                                 distCoeffs.at<float>(0, 1), 
                                                 distCoeffs.at<float>(0, 2), 
                                                 distCoeffs.at<float>(0, 3), 
                                                 distCoeffs.at<float>(0, 4));
    if (save_to_file_) savefile << "distortion: [" << distCoeffs.at<float>(0, 0) << ", "
                                                   << distCoeffs.at<float>(0, 1) << ", "
                                                   << distCoeffs.at<float>(0, 2) << ", "
                                                   << distCoeffs.at<float>(0, 3) << ", "
                                                   << distCoeffs.at<float>(0, 4) << "]";

    // R
    ROS_INFO("rlc: [%f, %f, %f,", final_transformation(0, 0), final_transformation(0, 1), final_transformation(0, 2));
    ROS_INFO("      %f, %f, %f,", final_transformation(1, 0), final_transformation(1, 1), final_transformation(1, 2));
    ROS_INFO("      %f, %f, %f]", final_transformation(2, 0), final_transformation(2, 1), final_transformation(2, 2));
    if (save_to_file_) savefile << "rlc: [" << final_transformation(0, 0) << ", " << final_transformation(0, 1) << ", " << final_transformation(0, 2) << endl;
    if (save_to_file_) savefile << "      " << final_transformation(1, 0) << ", " << final_transformation(1, 1) << ", " << final_transformation(1, 2) << endl;
    if (save_to_file_) savefile << "      " << final_transformation(2, 0) << ", " << final_transformation(2, 1) << ", " << final_transformation(2, 2) << endl;

    // t
    ROS_INFO("tlc: [%f, %f, %f]", final_transformation(0, 3), final_transformation(1, 3), final_transformation(2, 3));
    if (save_to_file_) savefile << "tlc: [" << final_transformation(0, 3) << ", " << final_transformation(1, 3) << ", " << final_transformation(2, 3) << endl;
  }

  sensor1Received = false;
  sensor2Received = false;
}

void sensor1_callback(const velo2cam_calibration::ClusterCentroids::ConstPtr sensor1_centroids)
{
  sensor1_frame_id = sensor1_centroids->header.frame_id;
  if (!S1_WARMUP_DONE) {
    S1_WARMUP_COUNT++;
    cout << "Clusters from " << sensor1_frame_id << ": " << S1_WARMUP_COUNT
         << "/10" << '\r' << flush;
    if (S1_WARMUP_COUNT >= 10)  // TODO: Change to param?
    {
      cout << endl;
      sensor1_sub.shutdown();
      sensor2_sub.shutdown();

      cout << "Clusters from " << sensor1_frame_id
           << " received. Is the warmup done? [Y/n]" << endl;
      string answer;
      getline(cin, answer);
      if (answer == "y" || answer == "Y" || answer == "") {
        S1_WARMUP_DONE = !S1_WARMUP_DONE;

        if (!S2_WARMUP_DONE) {
          cout << "Filters for sensor 1 are adjusted now. Please, proceed with "
                  "the other sensor."
               << endl;
        } else {  // Both sensors adjusted
          cout << "Warmup phase completed. Starting calibration phase." << endl;
          std_msgs::Empty myMsg;
          sensor_switch_pub.publish(myMsg);  //
        }
      } else {  // Reset counter to allow further warmup
        S1_WARMUP_COUNT = 0;
      }

      sensor1_sub = nh_->subscribe<velo2cam_calibration::ClusterCentroids>("cloud1", 100, sensor1_callback);
      sensor2_sub = nh_->subscribe<velo2cam_calibration::ClusterCentroids>("cloud2", 100, sensor2_callback);
    }
    return;
  }

  if (!S2_WARMUP_DONE) {
    return;
  }

  if (DEBUG) ROS_INFO("sensor1 (%s) pattern ready!", sensor1_frame_id.c_str());

  if (sensor1_buffer.size() == TARGET_POSITIONS_COUNT) {
    sensor1_buffer.resize(TARGET_POSITIONS_COUNT + 1);
  }

  // SWAN: if it is a camera
  if (is_sensor1_cam) {
    std::ostringstream sstream;
    sstream << "rotated_" << sensor1_frame_id;
    sensor1_rotated_frame_id = sstream.str();
    cout << "[SWAN] sensor1_rotated_frame_id = " << sensor1_rotated_frame_id << endl;
    cout << "[SWAN] sensor1_frame_id = " << sensor1_frame_id << endl;

    pcl::PointCloud<pcl::PointXYZ>::Ptr xy_sensor1_cloud(
        new pcl::PointCloud<pcl::PointXYZ>());

    fromROSMsg(sensor1_centroids->cloud, *xy_sensor1_cloud);

    tf::TransformListener listener;
    tf::StampedTransform transform;
    try
    {
      listener.waitForTransform(sensor1_rotated_frame_id, sensor1_frame_id,
                                ros::Time(0), ros::Duration(20.0));
      listener.lookupTransform(sensor1_rotated_frame_id, sensor1_frame_id,
                               ros::Time(0), transform);
      
      ROS_INFO("*******************************");
      ROS_INFO("[SWAN] sensor1_rotated_frame_id = %s", sensor1_rotated_frame_id.c_str());
      ROS_INFO("[SWAN] sensor1_frame_id = %s", sensor1_frame_id.c_str());
      ROS_INFO("*******************************");
    } 
    catch (tf::TransformException &ex)
    {
      ROS_WARN("TF exception:\n%s", ex.what());
      return;
    }

    {
      ROS_INFO("********************************");
      ROS_INFO("[SWAN] (camera) sensor1_rotated_frame_id = %s", sensor1_rotated_frame_id.c_str());
      ROS_INFO("[SWAN] (camera) transform");
      tf::Matrix3x3 R = transform.getBasis();
      tf::Vector3   t = transform.getOrigin();
      ROS_INFO("\t\t%f\t%f\t%f\t%f", R[0].getX(), R[0].getY(), R[0].getZ(), t.getX());
      ROS_INFO("\t\t%f\t%f\t%f\t%f", R[1].getX(), R[1].getY(), R[1].getZ(), t.getY());
      ROS_INFO("\t\t%f\t%f\t%f\t%f", R[2].getX(), R[2].getY(), R[2].getZ(), t.getZ());
      ROS_INFO("********************************");

      // [SWAN] transform: rotate points from Camera coords to ROS coords
      // Translation vector: [0, 0, 0]^\top
      // Rotation Matrix: ROS_R_Camera
      // [ 0,  0,  1]
      // [-1,  0,  0]
      // [ 0, -1,  0]

      //
      //      from (ROS)       to (Camera)
      //
      //          z'  x'               z
      //          ^   ^               ^
      //          |  /               /
      //          | /               /
      //          |/               /
      // y' <-----R               C-------->x
      //                          |
      //                          |
      //                          |
      //                          v
      //                          y
    }

    tf::Transform inverse = transform.inverse();
    double roll, pitch, yaw;
    inverse.getBasis().getRPY(roll, pitch, yaw);

    // SWAN: for camera, transform the centroids
    pcl_ros::transformPointCloud(*xy_sensor1_cloud, *sensor1_cloud, transform);
  } 
  else
  {
    // SWAN: for lidar, use it without transformation
    fromROSMsg(sensor1_centroids->cloud, *sensor1_cloud);
  }

  sensor1Received = true;

  sortPatternCenters(sensor1_cloud, sensor1_vector);
  if (DEBUG)
  {
    colourCenters(sensor1_vector, isensor1_cloud);

    sensor_msgs::PointCloud2 colour_cloud;
    pcl::toROSMsg(*isensor1_cloud, colour_cloud);
    colour_cloud.header.frame_id =
        is_sensor1_cam ? sensor1_rotated_frame_id : sensor1_frame_id;
    colour_sensor1_pub.publish(colour_cloud);
  }

  // SWAN: accumulate Cp for all target poses
  sensor1_buffer[TARGET_POSITIONS_COUNT].push_back(
      std::tuple<int, int, pcl::PointCloud<pcl::PointXYZ>,
                 std::vector<pcl::PointXYZ>>(
          sensor1_centroids->total_iterations,
          sensor1_centroids->cluster_iterations, *sensor1_cloud,
          sensor1_vector));
  sensor1_count = sensor1_centroids->total_iterations;

  if (DEBUG) ROS_INFO("[V2C] sensor1: %d", TARGET_POSITIONS_COUNT);

  for (vector<pcl::PointXYZ>::iterator it = sensor1_vector.begin();
       it < sensor1_vector.end(); ++it) {
    if (DEBUG)
      cout << "c" << it - sensor1_vector.begin() << "="
           << "[" << (*it).x << " " << (*it).y << " " << (*it).z << "]" << endl;
  }

  // sync_iterations is designed to extract a calibration result every single
  // frame, so we cannot wait until TARGET_ITERATIONS
  if (sync_iterations) {
    if (sensor2_count >= sensor1_count) {
      calibrateExtrinsics(sensor1_count);
    } else {
      if (tf_sensor1_sensor2.frame_id_ != "" &&
          tf_sensor1_sensor2.child_frame_id_ != "") {
        static tf::TransformBroadcaster br;
        tf_sensor1_sensor2.stamp_ = ros::Time::now();
        if (publish_tf_) br.sendTransform(tf_sensor1_sensor2);
      }
    }
    return;
  }

  // Normal operation (sync_iterations=false)
  if (sensor1Received && sensor2Received) {
    cout << min(sensor1_count, sensor2_count) << "/30 iterations" << '\r' << flush;

    std_msgs::Int32 it;
    it.data = min(sensor1_count, sensor2_count);
    iterations_pub.publish(it);

    if (sensor1_count >= TARGET_ITERATIONS &&
        sensor2_count >= TARGET_ITERATIONS) {
      cout << endl;
      sensor1_sub.shutdown();
      sensor2_sub.shutdown();

      string answer;
      if (single_pose_mode) {
        answer = "n";
      } else {
        cout << "Target iterations reached. Do you need another target "
                "location? [y/N]"
             << endl;
        getline(cin, answer);
      }
      if (answer == "n" || answer == "N" || answer == "") {
        calibrateExtrinsics(-1);
        calibration_ended = true;
      } else {  // Move the target and start over
        if (results_every_pose) calibrateExtrinsics(-1);
        TARGET_POSITIONS_COUNT++;
        cout << "Please, move the target to its new position and adjust the "
                "filters for each sensor before the calibration starts."
             << endl;
        // Start over if other position of the target is required
        std_msgs::Empty myMsg;
        sensor_switch_pub.publish(myMsg);  // Set sensor nodes to warmup phase
        S1_WARMUP_DONE = false;
        S1_WARMUP_COUNT = 0;
        S2_WARMUP_DONE = false;
        S2_WARMUP_COUNT = 0;
        sensor1Received = false;
        sensor2Received = false;
        sensor1_count = 0;
        sensor2_count = 0;
      }
      sensor1_sub = nh_->subscribe<velo2cam_calibration::ClusterCentroids>(
          "cloud1", 100, sensor1_callback);
      sensor2_sub = nh_->subscribe<velo2cam_calibration::ClusterCentroids>(
          "cloud2", 100, sensor2_callback);
      return;
    }
  } else {
    if (tf_sensor1_sensor2.frame_id_ != "" &&
        tf_sensor1_sensor2.child_frame_id_ != "") {
      static tf::TransformBroadcaster br;
      tf_sensor1_sensor2.stamp_ = ros::Time::now();
      if (publish_tf_) br.sendTransform(tf_sensor1_sensor2);
    }
  }
}

void sensor2_callback(velo2cam_calibration::ClusterCentroids::ConstPtr sensor2_centroids) {
  sensor2_frame_id = sensor2_centroids->header.frame_id;
  if (!S2_WARMUP_DONE && S1_WARMUP_DONE) {
    S2_WARMUP_COUNT++;
    cout << "Clusters from " << sensor2_frame_id << ": " << S2_WARMUP_COUNT
         << "/10" << '\r' << flush;
    if (S2_WARMUP_COUNT >= 10)  // TODO: Change to param?
    {
      cout << endl;
      sensor1_sub.shutdown();
      sensor2_sub.shutdown();

      cout << "Clusters from " << sensor2_frame_id
           << " received. Is the warmup done? (you can also reset this "
              "position) [Y/n/r]"
           << endl;
      string answer;
      getline(cin, answer);
      if (answer == "y" || answer == "Y" || answer == "") {
        S2_WARMUP_DONE = !S2_WARMUP_DONE;

        if (!S1_WARMUP_DONE) {
          cout << "Filters for sensor 2 are adjusted now. Please, proceed with "
                  "the other sensor."
               << endl;
        } else {  // Both sensors adjusted
          cout << "Warmup phase completed. Starting calibration phase." << endl;
          std_msgs::Empty myMsg;
          sensor_switch_pub.publish(myMsg);  //
        }
      } else if (answer == "r" ||
                 answer == "R") {  // Reset this position and
                                   // go back to Sensor 1 warmup
        S1_WARMUP_DONE = false;
        S1_WARMUP_COUNT = 0;
        S2_WARMUP_DONE = false;
        S2_WARMUP_COUNT = 0;
        sensor1Received = false;
        sensor2Received = false;
        sensor1_count = 0;
        sensor2_count = 0;
        cout << "Please, adjust the filters for each sensor before the "
                "calibration starts."
             << endl;
      } else {  // Reset counter to allow further warmup
        S2_WARMUP_COUNT = 0;
      }
      sensor1_sub = nh_->subscribe<velo2cam_calibration::ClusterCentroids>(
          "cloud1", 100, sensor1_callback);
      sensor2_sub = nh_->subscribe<velo2cam_calibration::ClusterCentroids>(
          "cloud2", 100, sensor2_callback);
    }
    return;
  } else if (!S2_WARMUP_DONE) {
    return;
  }
  if (DEBUG) ROS_INFO("sensor2 (%s) pattern ready!", sensor2_frame_id.c_str());

  if (sensor2_buffer.size() == TARGET_POSITIONS_COUNT) {
    sensor2_buffer.resize(TARGET_POSITIONS_COUNT + 1);
  }

  if (is_sensor2_cam) {
    std::ostringstream sstream;
    sstream << "rotated_" << sensor2_frame_id;
    sensor2_rotated_frame_id = sstream.str();

    pcl::PointCloud<pcl::PointXYZ>::Ptr xy_sensor2_cloud(
        new pcl::PointCloud<pcl::PointXYZ>());

    fromROSMsg(sensor2_centroids->cloud, *xy_sensor2_cloud);

    tf::TransformListener listener;
    tf::StampedTransform transform;
    try {
      listener.waitForTransform(sensor2_rotated_frame_id, sensor2_frame_id,
                                ros::Time(0), ros::Duration(20.0));
      listener.lookupTransform(sensor2_rotated_frame_id, sensor2_frame_id,
                               ros::Time(0), transform);
    } catch (tf::TransformException &ex) {
      ROS_WARN("TF exception:\n%s", ex.what());
      return;
    }

    tf::Transform inverse = transform.inverse();
    double roll, pitch, yaw;
    inverse.getBasis().getRPY(roll, pitch, yaw);

    pcl_ros::transformPointCloud(*xy_sensor2_cloud, *sensor2_cloud, transform);
  } else {
    fromROSMsg(sensor2_centroids->cloud, *sensor2_cloud);
  }

  sensor2Received = true;

  sortPatternCenters(sensor2_cloud, sensor2_vector);

  if (DEBUG) {
    colourCenters(sensor2_vector, isensor2_cloud);

    sensor_msgs::PointCloud2 colour_cloud;
    pcl::toROSMsg(*isensor2_cloud, colour_cloud);
    colour_cloud.header.frame_id =
        is_sensor2_cam ? sensor2_rotated_frame_id : sensor2_frame_id;
    colour_sensor2_pub.publish(colour_cloud);
  }

  sensor2_buffer[TARGET_POSITIONS_COUNT].push_back(
      std::tuple<int, int, pcl::PointCloud<pcl::PointXYZ>,
                 std::vector<pcl::PointXYZ>>(
          sensor2_centroids->total_iterations,
          sensor2_centroids->cluster_iterations, *sensor2_cloud,
          sensor2_vector));
  sensor2_count = sensor2_centroids->total_iterations;

  if (DEBUG) ROS_INFO("[V2C] sensor2: %d", TARGET_POSITIONS_COUNT);

  for (vector<pcl::PointXYZ>::iterator it = sensor2_vector.begin();
       it < sensor2_vector.end(); ++it) {
    if (DEBUG)
      cout << "l" << it - sensor2_vector.begin() << "="
           << "[" << (*it).x << " " << (*it).y << " " << (*it).z << "]" << endl;
  }

  // sync_iterations is designed to extract a calibration result every single
  // frame, so we cannot wait until TARGET_ITERATIONS
  if (sync_iterations) {
    if (sensor1_count >= sensor2_count) {
      calibrateExtrinsics(sensor2_count);
    } else {
      if (tf_sensor1_sensor2.frame_id_ != "" &&
          tf_sensor1_sensor2.child_frame_id_ != "") {
        static tf::TransformBroadcaster br;
        tf_sensor1_sensor2.stamp_ = ros::Time::now();
        if (publish_tf_) br.sendTransform(tf_sensor1_sensor2);
      }
    }
    return;
  }

  // Normal operation (sync_iterations=false)
  if (sensor1Received && sensor2Received) {
    cout << min(sensor1_count, sensor2_count) << "/30 iterations" << '\r' << flush;

    std_msgs::Int32 it;
    it.data = min(sensor1_count, sensor2_count);
    iterations_pub.publish(it);

    if (sensor1_count >= TARGET_ITERATIONS &&
        sensor2_count >= TARGET_ITERATIONS) {
      cout << endl;
      sensor1_sub.shutdown();
      sensor2_sub.shutdown();

      string answer;
      if (single_pose_mode) {
        answer = "n";
      } else {
        cout << "Target iterations reached. Do you need another target "
                "location? [y/N]"
             << endl;
        getline(cin, answer);
      }
      if (answer == "n" || answer == "N" || answer == "") {
        calibrateExtrinsics(-1);
        calibration_ended = true;
      } else {  // Move the target and start over
        if (results_every_pose) calibrateExtrinsics(-1);
        TARGET_POSITIONS_COUNT++;
        cout << "Please, move the target to its new position and adjust the "
                "filters for each sensor before the calibration starts."
             << endl;
        // Start over if other position of the target is required
        std_msgs::Empty myMsg;
        sensor_switch_pub.publish(myMsg);  // Set sensor nodes to warmup phase
        S1_WARMUP_DONE = false;
        S1_WARMUP_COUNT = 0;
        S2_WARMUP_DONE = false;
        S2_WARMUP_COUNT = 0;
        sensor1Received = false;
        sensor2Received = false;
        sensor1_count = 0;
        sensor2_count = 0;
      }
      sensor1_sub = nh_->subscribe<velo2cam_calibration::ClusterCentroids>(
          "cloud1", 100, sensor1_callback);
      sensor2_sub = nh_->subscribe<velo2cam_calibration::ClusterCentroids>(
          "cloud2", 100, sensor2_callback);
      return;
    }
  } else {
    if (tf_sensor1_sensor2.frame_id_ != "" &&
        tf_sensor1_sensor2.child_frame_id_ != "") {
      static tf::TransformBroadcaster br;
      tf_sensor1_sensor2.stamp_ = ros::Time::now();
      if (publish_tf_) br.sendTransform(tf_sensor1_sensor2);
    }
  }
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "velo2cam_calibration");
  ros::NodeHandle nh;              // GLOBAL
  nh_ = new ros::NodeHandle("~");  // LOCAL

  string csv_name;

  nh_->param<bool>("sync_iterations", sync_iterations, false);
  nh_->param<bool>("save_to_file", save_to_file_, false);
  nh_->param<bool>("publish_tf", publish_tf_, true);
  nh_->param<bool>("is_sensor2_cam", is_sensor2_cam, false);
  nh_->param<bool>("is_sensor1_cam", is_sensor1_cam, false);
  nh_->param<bool>("skip_warmup", skip_warmup, false);
  nh_->param<bool>("single_pose_mode", single_pose_mode, false);
  nh_->param<bool>("results_every_pose", results_every_pose, false);
  nh_->param<string>("csv_name", csv_name,
                     "registration_" + currentDateTime() + ".csv");

  {
    // SWAN: set K and D
    float dummy_default_value = -1.f;

    // camera matrix
    cameraMatrix.at<float>(0, 0) = nh_->param<float>("fx", dummy_default_value);
    cameraMatrix.at<float>(0, 1) = 0.f;
    cameraMatrix.at<float>(0, 2) = nh_->param<float>("cx", dummy_default_value);
    cameraMatrix.at<float>(1, 0) = 0.f;
    cameraMatrix.at<float>(1, 1) = nh_->param<float>("fy", dummy_default_value);
    cameraMatrix.at<float>(1, 2) = nh_->param<float>("cy", dummy_default_value);
    cameraMatrix.at<float>(2, 0) = 0.f;
    cameraMatrix.at<float>(2, 1) = 0.f;
    cameraMatrix.at<float>(2, 2) = 1.f;
    ROS_INFO("************ [SWAN] K ***************");
    ROS_INFO("[%f,  %f,  %f]", cameraMatrix.at<float>(0, 0), cameraMatrix.at<float>(0, 1), cameraMatrix.at<float>(0, 2));
    ROS_INFO("[%f,  %f,  %f]", cameraMatrix.at<float>(1, 0), cameraMatrix.at<float>(1, 1), cameraMatrix.at<float>(1, 2));
    ROS_INFO("[%f,  %f,  %f]", cameraMatrix.at<float>(2, 0), cameraMatrix.at<float>(2, 1), cameraMatrix.at<float>(2, 2));
    ROS_INFO("*************************************");


    // distortion coeffients: k1, k2, p1, p2, k3
    distCoeffs.at<float>(0, 0) = nh_->param<float>("k1", dummy_default_value);
    distCoeffs.at<float>(0, 1) = nh_->param<float>("k2", dummy_default_value);
    distCoeffs.at<float>(0, 2) = nh_->param<float>("p1", dummy_default_value);
    distCoeffs.at<float>(0, 3) = nh_->param<float>("p2", dummy_default_value);
    distCoeffs.at<float>(0, 4) = nh_->param<float>("k3", dummy_default_value);
    ROS_INFO("************ [SWAN] D ***************");
    ROS_INFO("[%f,  %f,  %f,  %f,  %f]", distCoeffs.at<float>(0, 0), 
                                         distCoeffs.at<float>(0, 1), 
                                         distCoeffs.at<float>(0, 2), 
                                         distCoeffs.at<float>(0, 3), 
                                         distCoeffs.at<float>(0, 4));
    ROS_INFO("*************************************");
  }

  sensor1Received = false;
  sensor1_cloud   = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
  isensor1_cloud  = pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);

  sensor2Received = false;
  sensor2_cloud   = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
  isensor2_cloud  = pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);

  // SWAN: mono camera: centers_cloud = not rotated (in camera coords)
  //            lidar : centers_cloud = rotated (in camera coords)
  sensor1_sub = nh_->subscribe<velo2cam_calibration::ClusterCentroids>("cloud1", 100, sensor1_callback); // SWAN: /mono_pattern_0/centers_cloud
  sensor2_sub = nh_->subscribe<velo2cam_calibration::ClusterCentroids>("cloud2", 100, sensor2_callback); // SWAN: /lidar_pattern_0/centers_cloud

  if (DEBUG)
  {
    clusters_sensor2_pub = nh_->advertise<sensor_msgs::PointCloud2>("clusters_sensor2", 1);
    clusters_sensor1_pub = nh_->advertise<sensor_msgs::PointCloud2>("clusters_sensor1", 1);

    colour_sensor2_pub = nh_->advertise<sensor_msgs::PointCloud2>("colour_sensor2", 1);
    colour_sensor1_pub = nh_->advertise<sensor_msgs::PointCloud2>("colour_sensor1", 1);
  }

  sensor_switch_pub = nh.advertise<std_msgs::Empty>("warmup_switch", 1);
  iterations_pub    = nh_->advertise<std_msgs::Int32>("iterations", 1);

  calibration_ended = false;

  if (save_to_file_) {
    ostringstream os;
    os << getenv("HOME") << "/v2c_experiments/" << csv_name;
    if (save_to_file_) {
      if (DEBUG) ROS_INFO("Opening %s", os.str().c_str());
      savefile.open(os.str().c_str());
      savefile << "it, x, y, z, r, p, y, used_sen1, used_sen2, total_sen1, "
                  "total_sen2"
               << endl;
    }
  }

  if (skip_warmup) {
    S1_WARMUP_DONE = true;
    S2_WARMUP_DONE = true;
    ROS_WARN("Skipping warmup");
  } else {
    cout << "Please, adjust the filters for each sensor before the calibration "
            "starts."
         << endl;
  }

  ros::Rate loop_rate(30);
  while (ros::ok() && !calibration_ended) {
    ros::spinOnce();
  }

  sensor1_sub.shutdown();
  sensor2_sub.shutdown();

  if (save_to_file_) savefile.close();

  // Save calibration params to launch file for testing

  // Get time
  time_t rawtime;
  struct tm *timeinfo;
  char buffer[80];

  time(&rawtime);
  timeinfo = localtime(&rawtime);

  strftime(buffer, 80, "%Y-%m-%d-%H-%M-%S", timeinfo);
  std::string str(buffer);

  // Get tf data
  tf::Transform inverse = tf_sensor1_sensor2.inverse();
  double roll, pitch, yaw;
  double xt = inverse.getOrigin().getX(), yt = inverse.getOrigin().getY(),
         zt = inverse.getOrigin().getZ();
  inverse.getBasis().getRPY(roll, pitch, yaw);

  std::string path = ros::package::getPath("velo2cam_calibration");
  string backuppath = path + "/launch/calibrated_tf_" + str + ".launch";
  path = path + "/launch/calibrated_tf.launch";

  cout << endl
       << "Creating .launch file with calibrated TF in: " << endl
       << path.c_str() << endl;
  // Create .launch file with calibrated TF
  TiXmlDocument doc;
  TiXmlDeclaration *decl = new TiXmlDeclaration("1.0", "utf-8", "");
  doc.LinkEndChild(decl);
  TiXmlElement *root = new TiXmlElement("launch");
  doc.LinkEndChild(root);

  TiXmlElement *arg = new TiXmlElement("arg");
  arg->SetAttribute("name", "stdout");
  arg->SetAttribute("default", "screen");
  root->LinkEndChild(arg);

  string sensor2_final_transformation_frame = sensor2_frame_id;
  if (is_sensor2_cam) {
    sensor2_final_transformation_frame = sensor2_rotated_frame_id;
    std::ostringstream sensor2_rot_stream_pub;
    sensor2_rot_stream_pub << "0 0 0 -1.57079632679 0 -1.57079632679 "
                           << sensor2_rotated_frame_id << " "
                           << sensor2_frame_id << " 10";
    string sensor2_rotation = sensor2_rot_stream_pub.str();

    TiXmlElement *sensor2_rotation_node = new TiXmlElement("node");
    sensor2_rotation_node->SetAttribute("pkg", "tf");
    sensor2_rotation_node->SetAttribute("type", "static_transform_publisher");
    sensor2_rotation_node->SetAttribute("name", "sensor2_rot_tf");
    sensor2_rotation_node->SetAttribute("args", sensor2_rotation);
    root->LinkEndChild(sensor2_rotation_node);
  }

  string sensor1_final_transformation_frame = sensor1_frame_id;
  if (is_sensor1_cam) {
    sensor1_final_transformation_frame = sensor1_rotated_frame_id;
    std::ostringstream sensor1_rot_stream_pub;
    sensor1_rot_stream_pub << "0 0 0 -1.57079632679 0 -1.57079632679 "
                           << sensor1_rotated_frame_id << " "
                           << sensor1_frame_id << " 10";
    string sensor1_rotation = sensor1_rot_stream_pub.str();

    TiXmlElement *sensor1_rotation_node = new TiXmlElement("node");
    sensor1_rotation_node->SetAttribute("pkg", "tf");
    sensor1_rotation_node->SetAttribute("type", "static_transform_publisher");
    sensor1_rotation_node->SetAttribute("name", "sensor1_rot_tf");
    sensor1_rotation_node->SetAttribute("args", sensor1_rotation);
    root->LinkEndChild(sensor1_rotation_node);
  }

  std::ostringstream sstream;
  sstream << xt << " " << yt << " " << zt << " " << yaw << " " << pitch << " "
          << roll << " " << sensor2_final_transformation_frame << " "
          << sensor1_final_transformation_frame << " 100";
  string tf_args = sstream.str();

  TiXmlElement *node = new TiXmlElement("node");
  node->SetAttribute("pkg", "tf");
  node->SetAttribute("type", "static_transform_publisher");
  node->SetAttribute("name", "velo2cam_tf");
  node->SetAttribute("args", tf_args);
  root->LinkEndChild(node);

  // Save XML file and copy
  doc.SaveFile(path);
  doc.SaveFile(backuppath);

  if (DEBUG) cout << "Calibration process finished." << endl;

  return 0;
}

/*
 * COPYRIGHT AND PERMISSION NOTICE
 * Penn Software MSCKF_VIO
 * Copyright (C) 2017 The Trustees of the University of Pennsylvania
 * All rights reserved.
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <iterator>
#include <algorithm>
#include <fstream>
#include <Eigen/SVD>
#include <Eigen/QR>
#include <Eigen/SparseCore>
#include <Eigen/SPQRSupport>
#include <boost/math/distributions/chi_squared.hpp>
#include <eigen_conversions/eigen_msg.h>
#include <tf_conversions/tf_eigen.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>

#include <ackermann_msckf/ackermann_msckf.h>
#include <ackermann_msckf/math_utils.hpp>
#include <ackermann_msckf/utils.h>
#include <nav_msgs/Odometry.h>

#include <sophus/se3.hpp>

using namespace std;
using namespace Eigen;

namespace ackermann_msckf{
// Static member variables in IMUState class.
StateIDType IMUState::next_id = 0;
double IMUState::gyro_noise = 0.001;
double IMUState::acc_noise = 0.01;
double IMUState::gyro_bias_noise = 0.001;
double IMUState::acc_bias_noise = 0.01;
Vector3d IMUState::gravity = Vector3d(0, 0, -GRAVITY_ACCELERATION);

// Static member variables in CAMState class.
Isometry3d CAMState::T_cam0_cam1 = Isometry3d::Identity();

// Static member variables in Feature class.
FeatureIDType Feature::next_id = 0;
double Feature::observation_noise = 0.01;
Feature::OptimizationConfig Feature::optimization_config;

map<int, double> AckermannMsckf::chi_squared_test_table;

/**
 * @brief AckermannMsckf
 *
 */
AckermannMsckf::AckermannMsckf(ros::NodeHandle& pnh):
    is_gravity_set(false),
    is_first_img(true),
    use_ackermann(false),
    nh(pnh) {
    return;
}

/**
 * @brief initialize
 *
 */
bool AckermannMsckf::initialize() {
    if (!loadParameters()) return false;
    ROS_INFO("Finish loading ROS parameters...");

    // Initialize state server
    state_server.continuous_noise_cov =
            Matrix<double, 12, 12>::Zero();
    state_server.continuous_noise_cov.block<3, 3>(0, 0) =
            Matrix3d::Identity()*IMUState::gyro_noise;
    state_server.continuous_noise_cov.block<3, 3>(3, 3) =
            Matrix3d::Identity()*IMUState::gyro_bias_noise;
    state_server.continuous_noise_cov.block<3, 3>(6, 6) =
            Matrix3d::Identity()*IMUState::acc_noise;
    state_server.continuous_noise_cov.block<3, 3>(9, 9) =
            Matrix3d::Identity()*IMUState::acc_bias_noise;

    // Initialize the chi squared test table with confidence
    // level 0.95.
    for (int i = 1; i < 100; ++i) {
        boost::math::chi_squared chi_squared_dist(i);
        chi_squared_test_table[i] =
                boost::math::quantile(chi_squared_dist, 0.05);
    }

    if (!createRosIO()) return false;
    ROS_INFO("Finish creating ROS IO...");

    return true;
}

/**
 * @brief loadParameters
 *
 */
bool AckermannMsckf::loadParameters() {

    nh.param<bool>("use_a27_platform", use_a27_platform, true);
    nh.param<bool>("use_svd_ex", use_svd_ex, true);
    nh.param<bool>("use_debug", use_debug, true);
    nh.param<bool>("use_ackermann", use_ackermann, true);
    nh.param<bool>("ackermann_update_v", ackermann_update_v, true);
    nh.param<bool>("ackermann_update_q", ackermann_update_q, true);
    nh.param<bool>("ackermann_update_v_ci", ackermann_update_v_ci, false);
    nh.param<bool>("ackermann_update_q_ci", ackermann_update_q_ci, false);

    // ackermann参数
    nh.param<double>("ackermann_wheel_base",
        ackermann_wheel_base, 1.9);
    nh.param<double>("ackermann_tire_base",
        ackermann_tire_base, 0.98);
    nh.param<double>("noise/ackermann_velocity_x_std",
        ackermann_speed_x_noise, 0.5);
    nh.param<double>("noise/ackermann_velocity_y_std",
        ackermann_speed_y_noise, 0.5);
    nh.param<double>("noise/ackermann_velocity_z_std",
        ackermann_speed_z_noise, 0.5);
        nh.param<double>("noise/ackermann_steerAngle_std",
        ackermann_steering_noise, 30);
    nh.param<double>("ackermann_steer_ratio",
        ackermann_steer_ratio, 360/21);
    nh.param<double>("noise/ackermann_heading_white_std",
        ackermann_heading_white_noise, 1);
    nh.param<double>("noise/ackermann_x_white_std",
        ackermann_x_white_noise, 0.1); 
    nh.param<double>("noise/ackermann_y_white_std",
        ackermann_y_white_noise, 0.1);  
    nh.param<int>("ackermann_rate",ackermann_rate, 150);

    nh.param<bool>("use_offline_bias", use_offline_bias, false);
    double gyro_bias_a, gyro_bias_b, gyro_bias_c;
    nh.param<double>("initial_state/gyro_bias_a",
        gyro_bias_a, 0.001);
    nh.param<double>("initial_state/gyro_bias_b",
        gyro_bias_b, 0.001);
    nh.param<double>("initial_state/gyro_bias_c",
        gyro_bias_c, 0.001);
    state_server.imu_state.initial_bias = {gyro_bias_a, gyro_bias_b, gyro_bias_c};
    
    nh.param<string>("output_path",output_path,"");
    ROS_INFO_STREAM("Loaded ex_output_file: " << output_path);

    string file_time;
		[&]() { 
			 time_t timep;
		     time (&timep);
		     char tmp[64];
		     strftime(tmp, sizeof(tmp), "%Y-%m-%d-%H-%M-%S",localtime(&timep) );
		     file_time = tmp;
			 }();

    std::string delim = ",";

    // pose
    pose_file.open((output_path + "pose_ack_msckf_comp" + "_" + file_time + ".txt"), std::ios::out);
    if(!pose_file.is_open())
    {
        cerr << "pose_ack_msckf_comp is not open" << endl;
    }

    // odom
    odom_file.open((output_path + "odom_ack_msckf_comp" + "_" + file_time + ".csv"), std::ios::out);
    if(!odom_file.is_open())
    {
        cerr << "odom_ack_msckf_comp is not open" << endl;
    }
    
    // std
    std_file.open((output_path + "std_ack_msckf_comp" + "_" + file_time + ".csv"), std::ios::out);
    if(!std_file.is_open())
    {
        cerr << "std_ack_msckf_comp is not open" << endl;
    }
    
    // rmse nees
    rmse_file.open((output_path + "rmse_ack_msckf_comp" + "_" + file_time + ".csv"), std::ios::out);
    if(!rmse_file.is_open())
    {
        cerr << "rmse_ack_msckf_comp  is not open" << endl;
    }

    // time
    time_file.open((output_path + "time_ack_msckf_comp" + "_" + file_time + ".csv"), std::ios::out);
    if(!time_file .is_open())
    {
        cerr << "time_ack_msckf_comp  is not open" << endl;
    }

    // odom
    odom_file << "#Time(sec),";
    odom_file << "Dtime(sec),";
    odom_file  << delim;
    odom_file << "x(m),y(m),z(m),";
    odom_file << "qx,qy,qz,qw,";
    odom_file << "roll_x_G(deg),pitch_y_G(deg),yaw_z_G(deg),";
    odom_file << "vx(m/s),vy(m/s),vz(m/s),";
    odom_file << "wx(rad/s),wy(rad/s),wz(rad/s),";
    odom_file  << delim;
    odom_file << "gt_x(m),gt_y(m),gt_z(m),";
    odom_file << "gt_qx,gt_qy,gt_qz,gt_qw,";
    odom_file << "gt_roll_x_G(deg),gt_pitch_y_G(deg),gt_yaw_z_G(deg),";
    odom_file << "gt_vx(m/s),gt_vy(m/s),gt_vz(m/s),";
    odom_file << "gt_wx(rad/s),gt_wy(rad/s),gt_wz(rad/s),";
    odom_file  << delim;
    odom_file << "Sr,Sr_avg,";
    odom_file << std::endl;

    // std
    std_file << "#Time(sec),";
    std_file << "Dtime(sec),";
    std_file  << delim;
    std_file  << "err_rx(deg),err_ry(deg),err_rz(deg),";
    std_file  << "std3_rx(deg),std3_ry(deg),std3_rz(deg),";
    std_file  << "-std3_rx(deg),-std3_ry(deg),-std3_rz(deg),";
    std_file << delim;
    std_file  << "err_px(m),err_py(m),err_pz(m),";
    std_file  << "std3_px(m),std3_py(m),std3_pz(m),";
    std_file  << "-std3_px(m),-std3_py(m),-std3_pz(m),";
    std_file  << delim;
    std_file << "err_vx(m/s),err_vy(m/s),err_vz(m/s),";
    std_file  << "std3_vx(m/s),std3_vy(m/s),std3_vz(m/s),";
    std_file  << "-std3_vx(m/s),-std3_vy(m/s),-std3_vz(m/s),";
    std_file << delim;
    std_file  << "err_bgx(deg/h),err_bgy(deg/h),err_bgz(deg/h),";
    std_file  << "std3_bgx(deg/h),std3_bgy(deg/h),std3_bgz(deg/h),";
    std_file  << "-std3_bgx(deg/h),-std3_bgy(deg/h),-std3_bgz(deg/h),";
    std_file << delim;
    std_file  << "err_bax(m/s^2),err_bay(m/s^2),err_baz(m/s^2),";
    std_file  << "std3_bax(m/s^2),std3_bay(m/s^2),std3_baz(m/s^2),";
    std_file  << "-std3_bax(m/s^2),-std3_bay(m/s^2),-std3_baz(m/s^2),";
    std_file << std::endl;

    // rmse nees
    rmse_file  << "#Time(sec),";
    rmse_file << "Dtime(sec),";
    rmse_file  << delim;
    rmse_file  << "serr_rx(deg^2),serr_ry(deg^2),serr_rz(deg^2),serr_rxyz(deg^2),";
    rmse_file  << "nees_rx,nees_ry,nees_rz,";
    rmse_file  << delim;
    rmse_file  << "serr_px(m^2),serr_py(m^2),serr_pz(m^2),serr_pxyz(m^2),";
    rmse_file  << "nees_px,nees_py,nees_pz,";
    rmse_file  << delim;
    rmse_file  << "serr_ribx(deg^2),serr_riby(deg^2),serr_ribz(deg^2),";
    rmse_file  << "nees_ribx,nees_riby,nees_ribz,";
    rmse_file  << delim;
    rmse_file  << "serr_pbix(m^2),serr_pbiy(m^2),serr_pbiz(m^2),";
    rmse_file  << "nees_pbix,nees_pbiy,nees_pbiz,";
    rmse_file  << delim;
    rmse_file  << "serr_bgx,serr_bgy,serr_bgz,";
    rmse_file  << "nees_bgx,nees_bgy,nees_bgz,";
    rmse_file  << delim;
    rmse_file  << "serr_bax,serr_bay,serr_baz,";
    rmse_file  << "nees_bax,nees_bay,nees_baz,";
    rmse_file  << delim;
    rmse_file  << "serr_vx,serr_vy,serr_vz,";
    rmse_file  << "nees_vx,nees_vy,nees_vz,";
    rmse_file  << std::endl;

    // time
    time_file  << "#Time(sec),";
    time_file  << "Dtime(sec),";
    time_file  << delim;
    time_file  << "process_time(ms),total_time(ms),avg_time(ms),";
    time_file  << std::endl;

    // Frame id
    nh.param<string>("fixed_frame_id", fixed_frame_id, "odom");
    nh.param<string>("child_frame_id", child_frame_id, "robot");
    nh.param<bool>("publish_tf", publish_tf, true);
    nh.param<double>("frame_rate", frame_rate, 30.0);
    nh.param<double>("position_std_threshold", position_std_threshold, 8.0);

    nh.param<double>("rotation_threshold", rotation_threshold, 0.2618);
    nh.param<double>("translation_threshold", translation_threshold, 0.4);
    nh.param<double>("tracking_rate_threshold", tracking_rate_threshold, 0.5);

    // Feature optimization parameters
    nh.param<double>("feature/config/translation_threshold",
                     Feature::optimization_config.translation_threshold, 0.2);

    // Noise related parameters
    nh.param<double>("noise/gyro", IMUState::gyro_noise, 0.001);
    nh.param<double>("noise/acc", IMUState::acc_noise, 0.01);
    nh.param<double>("noise/gyro_bias", IMUState::gyro_bias_noise, 0.001);
    nh.param<double>("noise/acc_bias", IMUState::acc_bias_noise, 0.01);
    nh.param<double>("noise/feature", Feature::observation_noise, 0.01);

    // Use variance instead of standard deviation.
    IMUState::gyro_noise *= IMUState::gyro_noise;
    IMUState::acc_noise *= IMUState::acc_noise;
    IMUState::gyro_bias_noise *= IMUState::gyro_bias_noise;
    IMUState::acc_bias_noise *= IMUState::acc_bias_noise;
    Feature::observation_noise *= Feature::observation_noise;

    // Set the initial IMU state.
    // The intial orientation and position will be set to the origin
    // implicitly. But the initial velocity and bias can be
    // set by parameters.
    nh.param<double>("initial_state/velocity/x",
                     state_server.imu_state.velocity(0), 0.0);
    nh.param<double>("initial_state/velocity/y",
                     state_server.imu_state.velocity(1), 0.0);
    nh.param<double>("initial_state/velocity/z",
                     state_server.imu_state.velocity(2), 0.0);

    // The initial covariance of orientation and position can be
    // set to 0. But for velocity, bias and extrinsic parameters,
    // there should be nontrivial uncertainty.
    double gyro_bias_cov, acc_bias_cov, velocity_cov;
    nh.param<double>("initial_covariance/velocity",
                     velocity_cov, 0.25);
                    
    nh.param<double>("initial_covariance/gyro_bias",
                     gyro_bias_cov, 1e-4);
    nh.param<double>("initial_covariance/acc_bias",
                     acc_bias_cov, 1e-2);

    double extrinsic_rotation_cov, extrinsic_translation_cov;
    nh.param<double>("initial_covariance/extrinsic_rotation_cov",
                     extrinsic_rotation_cov, 3.0462e-4);
    nh.param<double>("initial_covariance/extrinsic_translation_cov",
                     extrinsic_translation_cov, 1e-4);

    state_server.state_cov = MatrixXd::Zero(21, 21);
    for (int i = 3; i < 6; ++i)
        state_server.state_cov(i, i) = gyro_bias_cov;
    for (int i = 6; i < 9; ++i)
        state_server.state_cov(i, i) = velocity_cov;
    for (int i = 9; i < 12; ++i)
        state_server.state_cov(i, i) = acc_bias_cov;
    for (int i = 15; i < 18; ++i)
        state_server.state_cov(i, i) = extrinsic_rotation_cov;
    for (int i = 18; i < 21; ++i)
        state_server.state_cov(i, i) = extrinsic_translation_cov;

    // Hamilton -> JPL
    CAMState::T_cam0_cam1 =
            utils::getTransformEigen(nh, "cam1/T_cn_cnm1");
    Isometry3d T_imu_cam0 = utils::getTransformEigen(nh, "cam0/T_cam_imu");
    state_server.imu_state.T_imu_cam0 = Eigen::Isometry3d::Identity();
    state_server.imu_state.R_imu_cam0 = T_imu_cam0.linear();
    state_server.imu_state.t_cam0_imu = T_imu_cam0.inverse().translation();
    state_server.imu_state.T_imu_cam0.linear() = state_server.imu_state.R_imu_cam0;
    state_server.imu_state.T_imu_cam0.translation() = T_imu_cam0.translation();

    // platform
    if(use_a27_platform){
        if(use_svd_ex){
            // is_svd
            // Hamilton -> JPL
            Eigen::Isometry3d T_imu_gps = utils::getTransformEigen(nh, "T_gps_imu");
            Eigen::Isometry3d T_body_gps = utils::getTransformEigen(nh, "T_gps_body");
            
            state_server.imu_state.T_imu_body = T_body_gps.inverse() * T_imu_gps;
        }else{
            // ib_hand
            // Hamilton -> JPL
            state_server.imu_state.T_imu_body =
                utils::getTransformEigen(nh, "T_body_imu");
        }   

    }else{
        // c11_platform 
        if(use_svd_ex){
            // is_svd
            // Hamilton -> JPL
            Eigen::Isometry3d T_imu_gps = utils::getTransformEigen(nh, "T_gps_imu");
            Eigen::Isometry3d T_body_gps = utils::getTransformEigen(nh, "T_gps_body");
            
            state_server.imu_state.T_imu_body = T_body_gps.inverse() * T_imu_gps;
        }else{
            // kalibr
            // Hamilton -> JPL
            CAMState::T_cam0_cam1 =
                    utils::getTransformEigen(nh, "cam1/T_cn_cnm1");
            Eigen::Isometry3d T_cam1_calibcam2 = utils::getTransformEigen(nh, "T_calibcam2_cam1");
            Eigen::Isometry3d T_calibcam2_body = utils::getTransformEigen(nh, "T_calibcam2_body").inverse();
            Eigen::Isometry3d T_imu_cam1 =  CAMState::T_cam0_cam1 * T_imu_cam0;

            state_server.imu_state.T_imu_body = T_calibcam2_body * T_cam1_calibcam2 * T_imu_cam1;
        }
    }

    state_server.imu_state.R_imu_body = state_server.imu_state.T_imu_body.linear();
    state_server.imu_state.t_imu_body = state_server.imu_state.T_imu_body.translation();
    state_server.imu_state.T_body_imu = state_server.imu_state.T_imu_body.inverse();
    state_server.imu_state.t_body_imu = state_server.imu_state.T_body_imu.translation();

    // body <-> camera
    T_body_cam0 = state_server.imu_state.T_imu_cam0 * state_server.imu_state.T_imu_body.inverse();
    R_body_cam0 = T_body_cam0.linear();
    t_body_cam0 = T_body_cam0.translation();

    // Maximum number of camera states to be stored
    nh.param<int>("max_cam_state_size", max_cam_state_size, 30);
    
    ROS_INFO("===========================================");
    ROS_INFO("ackermann_speed_x_noise: %.10f", ackermann_speed_x_noise);
    ROS_INFO("ackermann_speed_y_noise: %.10f", ackermann_speed_y_noise);
    ROS_INFO("ackermann_speed_z_noise: %.10f", ackermann_speed_z_noise);
    ROS_INFO("ackermann_steering_noise: %.10f", ackermann_steering_noise);
    ROS_INFO("fixed frame id: %s", fixed_frame_id.c_str());
    ROS_INFO("child frame id: %s", child_frame_id.c_str());
    ROS_INFO("publish tf: %d", publish_tf);
    ROS_INFO("frame rate: %f", frame_rate);
    ROS_INFO("position std threshold: %f", position_std_threshold);
    ROS_INFO("Keyframe rotation threshold: %f", rotation_threshold);
    ROS_INFO("Keyframe translation threshold: %f", translation_threshold);
    ROS_INFO("Keyframe tracking rate threshold: %f", tracking_rate_threshold);
    ROS_INFO("gyro noise: %.10f", IMUState::gyro_noise);
    ROS_INFO("gyro bias noise: %.10f", IMUState::gyro_bias_noise);
    ROS_INFO("acc noise: %.10f", IMUState::acc_noise);
    ROS_INFO("acc bias noise: %.10f", IMUState::acc_bias_noise);
    ROS_INFO("observation noise: %.10f", Feature::observation_noise);
    ROS_INFO("initial velocity: %f, %f, %f",
             state_server.imu_state.velocity(0),
             state_server.imu_state.velocity(1),
             state_server.imu_state.velocity(2));
    ROS_INFO("initial gyro bias cov: %f", gyro_bias_cov);
    ROS_INFO("initial acc bias cov: %f", acc_bias_cov);
    ROS_INFO("initial velocity cov: %f", velocity_cov);
    ROS_INFO("initial extrinsic rotation cov: %f",
             extrinsic_rotation_cov);
    ROS_INFO("initial extrinsic translation cov: %f",
             extrinsic_translation_cov);
    ROS_INFO("max camera state #: %d", max_cam_state_size);
    ROS_INFO("===========================================");
    return true;
}

/**
 * @brief createRosIO
 *
 */
bool AckermannMsckf::createRosIO() {
    odom_pub = nh.advertise<nav_msgs::Odometry>("ackermann_msckf_odom", 30);
    feature_pub = nh.advertise<sensor_msgs::PointCloud2>(
                "feature_point_cloud", 10);

    // imu
    imu_sub = nh.subscribe("imu", 200,
                           &AckermannMsckf::imuCallback, this);

    // ackermann
    ackermann_sub = nh.subscribe("ackermann", 200,
      &AckermannMsckf::ackermannCallback, this);

    // camera features
    feature_sub = nh.subscribe("features", 30,
                               &AckermannMsckf::featureCallback, this);

    // gt
    gt_init_sub = nh.subscribe("gt", 50,
      &AckermannMsckf::gtInitCallback, this);

    // save csv
    csv_timer = nh.createTimer(ros::Duration(1),&AckermannMsckf::csv_timer_callBack,this);

    return true;
}

/**
 * @brief gtInitCallback 
 *  
 */
void AckermannMsckf::gtInitCallback(
    const nav_msgs::OdometryConstPtr& msg) {

  gt_msg_buffer.push_back(*msg);
  
  // If this is the first_gt_init_msg, set
  // the initial frame.
  if (!is_gt_init_set) {
    Quaterniond orientation;
    Vector3d translation;
    tf::pointMsgToEigen(
        msg->pose.pose.position, translation);
    tf::quaternionMsgToEigen(
        msg->pose.pose.orientation, orientation);
    T_B0_GS_gt.linear() = orientation.toRotationMatrix();
    T_B0_GS_gt.translation() = translation;
    is_gt_init_set = true;
    double roll,pitch,yaw;
    tf::Matrix3x3(tf::Quaternion(orientation.x(),orientation.y(),orientation.z(),orientation.w())).getRPY(roll,pitch,yaw,1);    
    std::cout<< "ackermann_msckf gt_q_B0_GB : roll="<< roll * 180 / M_PI <<",　pitch="<< pitch * 180 / M_PI <<",　yaw="<< yaw * 180 / M_PI << std::endl;
    std::cout<< "ackermann_msckf gt_t_B0_GB : x="<< translation(0) <<", y="<< translation(1) <<", z="<< translation(2) << std::endl;
    
    gt_odom_last = *msg;
    gt_odom_curr = gt_odom_last; 
    t_GBi_GB_last =  T_B0_GS_gt.translation();

  }
  return;
}

/**
 * @brief initializeGravityAndBias 
 *  
 */
void AckermannMsckf::initializeGravityAndBias() {

    // Initialize gravity and gyro bias.
    Vector3d sum_angular_vel = Vector3d::Zero();
    Vector3d sum_linear_acc = Vector3d::Zero();

    for (const auto& imu_msg : imu_msg_buffer) {
        Vector3d angular_vel = Vector3d::Zero();
        Vector3d linear_acc = Vector3d::Zero();

        tf::vectorMsgToEigen(imu_msg.angular_velocity, angular_vel);
        tf::vectorMsgToEigen(imu_msg.linear_acceleration, linear_acc);

        sum_angular_vel += angular_vel;
        sum_linear_acc += linear_acc;
    }

    // if use_offline_bias
    if(use_offline_bias){
        state_server.imu_state.gyro_bias = state_server.imu_state.initial_bias;
    }else{
        state_server.imu_state.gyro_bias =
        sum_angular_vel / imu_msg_buffer.size();
    }
    cout << "state_server.imu_state.gyro_bias: " << state_server.imu_state.gyro_bias.transpose() << endl;

    // This is the gravity in the IMU frame.
    Vector3d gravity_imu =
            sum_linear_acc / imu_msg_buffer.size();

    // Initialize the initial orientation, so that the estimation
    // is consistent with the inertial frame.
    double gravity_norm = gravity_imu.norm();
    IMUState::gravity = Vector3d(0.0, 0.0, -gravity_norm);
    ROS_INFO("Gravity: %f",gravity_norm);

    Quaterniond q0_I_G = Quaterniond::FromTwoVectors(
                gravity_imu, -IMUState::gravity);
    state_server.imu_state.orientation =
            rotationToQuaternion(q0_I_G.toRotationMatrix().transpose());
    cout<<state_server.imu_state.orientation<<endl;

    q_I0_G_ = q0_I_G;
    q_G_I0_ = q_I0_G_.inverse();
    return;
}

/**
 * @brief preIntegrationAck 
 *  
 */ 
void AckermannMsckf::preIntegrationAck(double dt, const double speed_x, const double steering_angle)
{
    if (first_ack)
    {
        first_ack= false;
        tmp_pre_integration = new IntegrationBase{
                speed_x_0, 
                steering_angle_0 ,
                ackermann_wheel_base,
                ackermann_tire_base,
                ackermann_speed_x_noise,
                ackermann_steering_noise,
                ackermann_steer_ratio,
                ackermann_heading_white_noise,
                ackermann_x_white_noise,
                ackermann_y_white_noise,
                ackermann_rate};
    }
    if(odom_frame_count != 0){
        tmp_pre_integration->push_back_ack(dt, speed_x, steering_angle);
    }
    speed_x_0 = speed_x;
    steering_angle_0 = steering_angle;
}

/**
 * @brief ackermannProcess 
 *  Ackermann error state measurement model
 */ 
void AckermannMsckf::ackermannProcess(){
    if(state_server.cam_states.size() >= 5){    
        if(ackermann_msg_buffer.size()!=0) {
            
            ackermann_update_count++;

            // Move the iterator to the key position.
            auto current_cam_state_iter = state_server.cam_states.end();
            --current_cam_state_iter;
            auto current_cam_id = current_cam_state_iter->first;
            
            auto last_cam_state_iter = current_cam_state_iter;
            for(int i = 0; i<1; i++){
                --last_cam_state_iter;
            }
            auto last_cam_id = last_cam_state_iter->first;

            delta_cam_state_time_ack = current_cam_state_iter->second.time - last_cam_state_iter->second.time;

            // Find the start and the end limit within the imu msg buffer.
            auto ackermann_begin_iter = ackermann_msg_buffer.begin();

            // ack_time <= last_cam_time
            if(first_init){
                if(last_cam_state_iter->second.time < ackermann_begin_iter->header.stamp.toSec()){
                    cout.precision(11);
                    cout << fixed << "ackermann_msg_buffer.front().header.stamp.toSec(): " << ackermann_msg_buffer.front().header.stamp.toSec() << endl;
                    cout << fixed << "ackermann_begin_iter->header.stamp.toSec(): " << ackermann_begin_iter->header.stamp.toSec() << endl;
                    cout << fixed << "last_cam_state_iter->second.time: " << last_cam_state_iter->second.time << endl;
                    cout << "waiting..." << endl;
                    cout << endl;
                    return;
                }
                first_init = false;
            }
            first_init = false;
           
            while (ackermann_begin_iter != ackermann_msg_buffer.end()) {
                if ((ackermann_begin_iter->header.stamp.toSec() <
                    last_cam_state_iter->second.time) )
                ++ackermann_begin_iter;
                else
                break;
            }
            auto ackermann_end_iter = ackermann_begin_iter;
            while (ackermann_end_iter != ackermann_msg_buffer.end()) {
                if ((ackermann_end_iter->header.stamp.toSec() <
                    current_cam_state_iter->second.time))
                ++ackermann_end_iter;
                else
                break;
            }
            auto ackermann_end_save_iter = ackermann_end_iter;
            if((ackermann_msg_buffer.end()-1)->header.stamp.toSec() < current_cam_state_iter->second.time){
                ackermann_end_save_iter = ackermann_msg_buffer.end()-1;
            }else if(ackermann_end_iter->header.stamp.toSec()>current_cam_state_iter->second.time){
                --ackermann_end_save_iter;
            }
            
            std::vector<ackermann_msckf::AckermannDriveStamped> ACKs;
            ACKs.assign(ackermann_begin_iter,ackermann_end_save_iter+1);

            if(fabs(ackermann_end_save_iter->header.stamp.toSec() - current_cam_state_iter->second.time) > 1 || 
                fabs(ackermann_begin_iter->header.stamp.toSec() - last_cam_state_iter->second.time) > 1){
                ROS_WARN("!!!Bad time!!!");
                first_init = true;
                first_ack = true;
                odom_frame_count = 0;
                current_integrate_time = -1;
                last_dt_1 = 0;
                last_dt_1_ack = 0;
                delta_p_.setZero();
                delta_q_ = Eigen::Quaterniond::Identity();
                if (tmp_pre_integration != nullptr){
                    delete tmp_pre_integration;
                    tmp_pre_integration = nullptr;
                }  
                return; 
            }

            std::pair<std::vector<ackermann_msckf::AckermannDriveStamped>, CAMState> ack_measurement;
            ack_measurement = make_pair(ACKs,current_cam_state_iter->second);

            if (ack_measurement.first.empty()){
                ROS_WARN("No ackermann between two image, restart!");
                first_init = true;
                first_ack = true;
                odom_frame_count = 0;
                current_integrate_time = -1;
                last_dt_1 = 0;
                last_dt_1_ack = 0;
                delta_p_.setZero();
                delta_q_ = Eigen::Quaterniond::Identity();
                if (tmp_pre_integration != nullptr){
                    delete tmp_pre_integration;
                    tmp_pre_integration = nullptr;
                }  
                return;
            }
            ackermann_msg_buffer.erase(ackermann_msg_buffer.begin(), ackermann_begin_iter);
            if(!ack_measurement.first.empty()){
                auto camera_msg = ack_measurement.second;
                
                speed_avg = 0;
                speed_cntr = 0;

                double speed_x = 0;
                double steering_angle = 0; 

                auto ack_msg_iter = ack_measurement.first.begin();
                while (ack_msg_iter != ack_measurement.first.end()) {
                    double t = ack_msg_iter->header.stamp.toSec();
                    double img_t = camera_msg.time;

                    if(ack_msg_iter != (ack_measurement.first.end()-1)){
                        if (current_integrate_time < 0)
                            current_integrate_time = t;
                        if(first_ack){
                            if(odom_frame_count == 0){
                                speed_x_0 = 0;
                                steering_angle_0 = 0;
                            }else{
                                double dt_2 = t - current_integrate_time;

                                assert(last_dt_1 >= 0);
                                assert(dt_2 >= 0);
                                assert(last_dt_1 + dt_2 >= 0);
                                
                                double w1,w2;
                                if((last_dt_1 + dt_2) == 0){
                                    w1 = 0.5;
                                    w2 = 0.5;
                                }else{
                                    w1 = dt_2 / (last_dt_1 + dt_2);
                                    w2 = last_dt_1 / (last_dt_1 + dt_2);
                                }
                              
                                speed_x_0 = w1 * speed_x_0 + w2 * ack_msg_iter->drive.speed;        
                                steering_angle_0 = w1 * steering_angle_0 + w2 * ack_msg_iter->drive.steering_angle;
                            
                            
                            }
                        }
                        double dt = t - current_integrate_time;
                        assert(dt >= 0);
                        speed_x =  ack_msg_iter->drive.speed;
                        steering_angle = ack_msg_iter->drive.steering_angle;  
                        current_integrate_time = t;
                        preIntegrationAck(dt, speed_x, steering_angle);

                        speed_avg += speed_x;
                        speed_cntr++;

                    }else{
                        double dt_end = t - current_integrate_time;
                        assert(dt_end >= 0);
                        current_integrate_time = t;
                        speed_x = ack_msg_iter->drive.speed;
                        steering_angle = ack_msg_iter->drive.steering_angle;
                        preIntegrationAck(dt_end, speed_x, steering_angle);

                        double dt_1 = img_t - current_integrate_time;
                    
                        current_integrate_time = img_t;
                        assert(dt_1 >= 0);

                        last_dt_1 = dt_1;
                        last_dt_1_ack = last_dt_1;

                        speed_x = ack_msg_iter->drive.speed;
                        steering_angle = ack_msg_iter->drive.steering_angle;
                        preIntegrationAck(dt_1, speed_x, steering_angle);

                        speed_avg += speed_x;
                        speed_cntr++;
                    }     

                     ++ack_msg_iter;
                }

                if(odom_frame_count != 0){
                    // update
                    auto cam_state_iter_curr = state_server.cam_states.find(current_cam_id);
                    int cam_state_cntr_curr = std::distance(state_server.cam_states.begin(), cam_state_iter_curr);
                    auto cam_state_iter_last = state_server.cam_states.find(last_cam_id);
                    int cam_state_cntr_last = std::distance(state_server.cam_states.begin(), cam_state_iter_last);

                    MatrixXd H_x = MatrixXd::Zero(6,21+6*state_server.cam_states.size());
                    VectorXd r = VectorXd::Zero(6);

                    //-------------------------------------------------
                    Matrix3d R_G_Ci = quaternionToRotation(last_cam_state_iter->second.orientation);
                    Vector3d t_Ci_G = last_cam_state_iter->second.position;
                    Matrix3d R_G_Cj = quaternionToRotation(current_cam_state_iter->second.orientation);
                    Vector3d t_Cj_G = current_cam_state_iter->second.position;
                    Matrix3d C_R_j_i =  R_G_Ci * R_G_Cj.transpose();
                    Vector3d C_t_j_i_G =  t_Cj_G - t_Ci_G;
                    Vector3d C_t_j_i_C =  R_G_Ci * C_t_j_i_G;
                    Matrix3d R_i_c = state_server.imu_state.R_imu_cam0;
                    Vector3d t_c_i = state_server.imu_state.t_cam0_imu;
                    Vector3d t_body_imu = state_server.imu_state.t_body_imu;
                    Vector3d w_G_i = state_server.imu_state.angular_velocity;
                    //-------------------------------------------------

                    // tunable parameter configurations 
                    //-------------------------------------------------
                    if(ackermann_update_q){
                        Eigen::Matrix3d B_R_j_i_ = R_body_cam0.transpose() * C_R_j_i * R_body_cam0;
                        Eigen::Quaterniond B_Q_j_i = tmp_pre_integration->delta_q;
                        Sophus::SO3d SO3_B_R_j_i( B_Q_j_i.toRotationMatrix() );
                        Sophus::SO3d SO3_deta_B_R_j_i( SO3_B_R_j_i.inverse() * Sophus::SO3d(B_R_j_i_));
                        Eigen::Vector3d so3_deta_B_R_j_i = SO3_deta_B_R_j_i.log();
                        
                        if(use_debug){
                            cout.precision(11);  
                            cout<<"Before so3_deta_B_R_j_i (deg)= "<< so3_deta_B_R_j_i.transpose() * 180/M_PI << endl;
                        }

                        r.head<3>(3) = so3_deta_B_R_j_i;

                        // H17
                        H_x.block<3,3>(0,21+6*cam_state_cntr_curr) = - R_body_cam0.transpose() * C_R_j_i;

                        // H12
                        H_x.block<3,3>(0,15) =  R_body_cam0.transpose() * C_R_j_i - R_body_cam0.transpose();
                        
                        if(ackermann_update_q_ci){
                            // H15
                            H_x.block<3,3>(0,21+6*cam_state_cntr_last) =  R_body_cam0.transpose();
                        }

                    }

                    //-------------------------------------------------
                    if(ackermann_update_v){
                        Eigen::Vector3d B_v = Vector3d::Zero();
                        Eigen::Vector3d B_v__ = Vector3d::Zero();

                        speed_avg = speed_avg / speed_cntr;
                        B_v = Vector3d(speed_avg,0,0);
                        B_v__ = R_body_cam0.transpose() * 
                            (( C_R_j_i * t_body_cam0 +  C_t_j_i_C ) - t_body_cam0) / delta_cam_state_time_ack;
    
                        r.tail<3>() = B_v - B_v__;

                        // H27
                        H_x.block<3,3>(3,21+6*cam_state_cntr_curr) =  (-R_body_cam0.transpose() * 
                            C_R_j_i * skewSymmetric(t_body_cam0)) / delta_cam_state_time_ack;
                        
                        // H28
                        H_x.block<3,3>(3,21+6*cam_state_cntr_curr+3) =  (R_body_cam0.transpose() 
                        * R_G_Ci) / delta_cam_state_time_ack;

                        // H22
                        H_x.block<3,3>(3,15) =  R_body_cam0.transpose() * (
                            C_R_j_i * skewSymmetric(t_body_cam0)
                            - skewSymmetric( C_R_j_i * t_body_cam0 + C_t_j_i_C) 
                        ) / delta_cam_state_time_ack;
                        
                        // H23
                        H_x.block<3,3>(3,18) =  (R_body_cam0.transpose() * 
                            R_i_c - R_body_cam0.transpose() * C_R_j_i * R_i_c) / delta_cam_state_time_ack;

                        if(ackermann_update_v_ci){
                            // H25
                            H_x.block<3,3>(3,21+6*cam_state_cntr_last) =  R_body_cam0.transpose() * 
                            skewSymmetric( C_R_j_i * t_body_cam0 + C_t_j_i_C ) / delta_cam_state_time_ack;
                            
                            // H26
                            H_x.block<3,3>(3,21+6*cam_state_cntr_last+3) =  -(R_body_cam0.transpose() 
                            * R_G_Ci) / delta_cam_state_time_ack;
                        }
                        
                    }

                    Eigen::MatrixXd estimateErrorCovariance = tmp_pre_integration->estimateErrorCovariance_;
                    double estimateErrorCovariance_q = estimateErrorCovariance(2,2); 
                                     
                    Matrix3d B_R_cov;
                    B_R_cov <<  estimateErrorCovariance_q *  10, 0, 0,
                                0, estimateErrorCovariance_q * 10, 0,
                                0, 0, estimateErrorCovariance_q;
                    Matrix3d B_v_cov;
                    B_v_cov <<  pow(ackermann_speed_x_noise , 2) ,0,0,
                                0, pow(ackermann_speed_y_noise , 2),0,
                                0, 0,pow(ackermann_speed_z_noise , 2);
              
                    MatrixXd noise=MatrixXd::Identity(6,6);
                    noise.block<3,3>(0,0)= B_R_cov;
                    noise.block<3,3>(3,3)= B_v_cov;
      
                   if(use_debug){
                        cout.precision(11);
                        cout<< fixed << "Before Ackermann Update r: " << r.head<3>().transpose() *  180/M_PI << " " << r.tail<3>().transpose() <<endl;
                    }

                    // AckermannmeasurementUpdate
                    AckermannmeasurementUpdate(H_x,r,noise);
        
                    // debug
                    //-------------------------------------------------
                    R_G_Ci = quaternionToRotation(last_cam_state_iter->second.orientation);
                    t_Ci_G = last_cam_state_iter->second.position;
                    R_G_Cj = quaternionToRotation(current_cam_state_iter->second.orientation);
                    t_Cj_G = current_cam_state_iter->second.position;
                    C_R_j_i =  R_G_Ci * R_G_Cj.transpose();
                    C_t_j_i_G =  t_Cj_G - t_Ci_G;
                    C_t_j_i_C =  R_G_Ci * C_t_j_i_G;
                    //-------------------------------------------------
                    
                    if(ackermann_update_q){
                        Eigen::Matrix3d B_R_j_i_ = R_body_cam0.transpose() * C_R_j_i * R_body_cam0;
                        Eigen::Quaterniond B_Q_j_i = tmp_pre_integration->delta_q;
                        Sophus::SO3d SO3_B_R_j_i( B_Q_j_i.toRotationMatrix() );
                        Sophus::SO3d SO3_deta_B_R_j_i( SO3_B_R_j_i.inverse() * Sophus::SO3d(B_R_j_i_));
                        Eigen::Vector3d so3_deta_B_R_j_i = SO3_deta_B_R_j_i.log();

                        r.head<3>(3) = so3_deta_B_R_j_i;
                    }
                    
                    if(ackermann_update_v){
                        Eigen::Vector3d B_v = Vector3d::Zero();
                        Eigen::Vector3d B_v__ = Vector3d::Zero();

                        speed_avg = speed_avg / speed_cntr;
                        B_v = Vector3d(speed_avg,0,0);
                        B_v__ = R_body_cam0.transpose() * 
                            (( C_R_j_i * t_body_cam0 +  C_t_j_i_C ) - t_body_cam0) / delta_cam_state_time_ack;
                        
                        r.tail<3>() = B_v - B_v__;
                    }

                    if(use_debug){
                        cout.precision(11);
                        cout<< fixed << "After Ackermann Update r: " << r.head<3>().transpose() *  180/M_PI << " " << r.tail<3>().transpose() <<endl;
                        cout << "---------------------------" << endl;
                    }
                }

                odom_frame_count++;
                first_ack = true;
                if (tmp_pre_integration != nullptr){
                    delete tmp_pre_integration;
                    tmp_pre_integration = nullptr;
                }  
            }

        }       
    }
}

/**
 * @brief featureCallback 
 *  
 */ 
void AckermannMsckf::featureCallback(
        const CameraMeasurementConstPtr& msg) {

    if (!is_gravity_set) return;
    if (!is_gt_init_set) return;

    {
        // csv_curr_time
        csv_curr_time = ros::Time::now().toSec();
        is_csv_curr_time_init = true;
    }

    // is_first_img
    if (is_first_img) {
        is_first_img = false;
        state_server.imu_state.time = msg->header.stamp.toSec();
        state_server.imu_state.gt_time = msg->header.stamp.toSec();

        // save time
        DfirstTime = msg->header.stamp.toSec();
    }
    // total Dtime
    Dtime = msg->header.stamp.toSec() - DfirstTime;

    static double max_processing_time = 0.0;
    static int critical_time_cntr = 0;
    double processing_start_time = ros::Time::now().toSec();

    // Publish the odometry.
    ros::Time start_time = ros::Time::now();
    publish(msg->header.stamp);
    double publish_time = (
                ros::Time::now()-start_time).toSec();

    // Propogate the IMU state.
    start_time = ros::Time::now();
    batchImuProcessing(msg->header.stamp.toSec());
    batchGtProcessing(msg->header.stamp.toSec());
    double imu_processing_time = (
                ros::Time::now()-start_time).toSec();
   
    // Augment the state vector.
    start_time = ros::Time::now();
    stateAugmentation(msg->header.stamp.toSec());
    double state_augmentation_time = (
                ros::Time::now()-start_time).toSec();

    // update Ackermann measurement
    if(use_ackermann){
        ackermannProcess();
    }

    // Add new observations 
    start_time = ros::Time::now();
    addFeatureObservations(msg);
    double add_observations_time = (
                ros::Time::now()-start_time).toSec();

    // Perform measurement update if necessary.
    start_time = ros::Time::now();
    removeLostFeatures();
    double remove_lost_features_time = (
                ros::Time::now()-start_time).toSec();
    start_time = ros::Time::now();
    pruneCamStateBuffer();
    double prune_cam_states_time = (
                ros::Time::now()-start_time).toSec();

    // Reset the system if necessary.
    onlineReset();

    double processing_end_time = ros::Time::now().toSec();
    double processing_time =
            processing_end_time - processing_start_time;
    if (processing_time > 1.0/frame_rate) {
        ++critical_time_cntr;
        ROS_INFO("\033[1;31mTotal processing time %f/%d...\033[0m",
                 processing_time, critical_time_cntr);
        printf("Remove lost features time: %f/%f\n",
               remove_lost_features_time, remove_lost_features_time/processing_time);
        printf("Remove camera states time: %f/%f\n",
               prune_cam_states_time, prune_cam_states_time/processing_time);
    }

    CSVDATA_TIME csvdata_time;
    csvdata_time.time = msg->header.stamp.toSec();
    csvdata_time.Dtime = Dtime;
    csvdata_time.process_time = processing_time * 1000;
    total_time += csvdata_time.process_time;
    csvdata_time.total_time = total_time;
    csvdata_time.avg_time = total_time / global_count;
    csvData_time.push_back(csvdata_time);

    return;
}

/**
 * @brief batchGtProcessing 
 *  
 */
void AckermannMsckf::batchGtProcessing(const double& time_bound) {

  // Counter how many IMU msgs in the buffer are used.
  int used_gt_msg_cntr = 0;

  for (const auto& gt_msg : gt_msg_buffer) {
    double gt_time = gt_msg.header.stamp.toSec();
    if (gt_time < state_server.imu_state.gt_time) {
      ++used_gt_msg_cntr;
      continue;
    }
    if (gt_time > time_bound) break;

    // Save the newest gt msg
    gt_odom_curr = gt_msg;
    state_server.imu_state.gt_time = gt_time;

    ++used_gt_msg_cntr;
  }

  // Remove all used IMU msgs.
  gt_msg_buffer.erase(gt_msg_buffer.begin(),
      gt_msg_buffer.begin() + used_gt_msg_cntr);

  return;
}

/**
 * @brief AckermannmeasurementUpdate 
 *  
 */
void AckermannMsckf::AckermannmeasurementUpdate(const Eigen::MatrixXd& H,const Eigen::VectorXd&r,const Eigen::MatrixXd &noise)
{
    if (H.rows() == 0 || r.rows() == 0) return;

    MatrixXd H_thin;
    VectorXd r_thin;

    if (H.rows() > H.cols()) {
        // Convert H to a sparse matrix.
        SparseMatrix<double> H_sparse = H.sparseView();

        // Perform QR decomposition on H_sparse.
        SPQR<SparseMatrix<double> > spqr_helper;
        spqr_helper.setSPQROrdering(SPQR_ORDERING_NATURAL);
        spqr_helper.compute(H_sparse);

        MatrixXd H_temp;
        VectorXd r_temp;
        (spqr_helper.matrixQ().transpose() * H).evalTo(H_temp);
        (spqr_helper.matrixQ().transpose() * r).evalTo(r_temp);

        H_thin = H_temp.topRows(21+state_server.cam_states.size()*6);
        r_thin = r_temp.head(21+state_server.cam_states.size()*6);

    } else {
        H_thin = H;
        r_thin = r;
    }

    // Compute the Kalman gain.
    const MatrixXd& P = state_server.state_cov;
    MatrixXd S = H_thin*P*H_thin.transpose() +noise;
    MatrixXd K_transpose = S.ldlt().solve(H_thin*P);
    MatrixXd K = K_transpose.transpose();

    // Compute the error of the state.
    VectorXd delta_x = K * r_thin;

    // delta_x_imu
    const VectorXd& delta_x_imu = delta_x.head<21>();

    if (
        delta_x_imu.segment<3>(6).norm() > 0.5 ||
        delta_x_imu.segment<3>(12).norm() > 1.0) {
        printf("delta velocity: %f\n", delta_x_imu.segment<3>(6).norm());
        printf("delta position: %f\n", delta_x_imu.segment<3>(12).norm());
        ROS_WARN("Update change is too large.");
    }

    // update imu_state
    const Vector4d dq_imu =
            smallAngleQuaternion(delta_x_imu.head<3>());
    state_server.imu_state.orientation = quaternionMultiplication(
                dq_imu, state_server.imu_state.orientation);
    state_server.imu_state.gyro_bias += delta_x_imu.segment<3>(3);
    state_server.imu_state.velocity += delta_x_imu.segment<3>(6);
    state_server.imu_state.acc_bias += delta_x_imu.segment<3>(9);
    state_server.imu_state.position += delta_x_imu.segment<3>(12);

    // update extrinsic
    const Vector4d dq_extrinsic =
            smallAngleQuaternion(delta_x_imu.segment<3>(15));
    state_server.imu_state.R_imu_cam0 = quaternionToRotation(
                dq_extrinsic) * state_server.imu_state.R_imu_cam0;
    state_server.imu_state.t_cam0_imu += delta_x_imu.segment<3>(18);

    Eigen::Isometry3d T_cam0_imu = Eigen::Isometry3d::Identity();
    T_cam0_imu.linear() = state_server.imu_state.R_imu_cam0 .inverse();
    T_cam0_imu.translation() =  state_server.imu_state.t_cam0_imu;
    state_server.imu_state.T_imu_cam0 = T_cam0_imu.inverse();
    T_body_cam0 = state_server.imu_state.T_imu_cam0 * state_server.imu_state.T_imu_body.inverse();
    R_body_cam0 = T_body_cam0.linear();
    t_body_cam0 = T_body_cam0.translation();

    // Update the camera states.
    auto cam_state_iter = state_server.cam_states.begin();
    for (int i = 0; i < state_server.cam_states.size();
         ++i, ++cam_state_iter) {
        const VectorXd& delta_x_cam = delta_x.segment<6>(21+i*6);
        const Vector4d dq_cam = smallAngleQuaternion(delta_x_cam.head<3>());
        cam_state_iter->second.orientation = quaternionMultiplication(
                    dq_cam, cam_state_iter->second.orientation);
        cam_state_iter->second.position += delta_x_cam.segment<3>(3);
    }

    // Update state covariance.
    MatrixXd I_KH = MatrixXd::Identity(K.rows(), H_thin.cols()) - K*H_thin;
    state_server.state_cov = I_KH*state_server.state_cov;

    // Fix the covariance to be symmetric
    MatrixXd state_cov_fixed = (state_server.state_cov +
                                state_server.state_cov.transpose()) / 2.0;
    state_server.state_cov = state_cov_fixed;

    return;
}

/**
 * @brief measurementUpdate 
 *  camera features 
 */ 
void AckermannMsckf::measurementUpdate(
        const MatrixXd& H, const VectorXd& r) {

    if (H.rows() == 0 || r.rows() == 0) return;

    MatrixXd H_thin;
    VectorXd r_thin;

    if (H.rows() > H.cols()) {
        // Convert H to a sparse matrix.
        SparseMatrix<double> H_sparse = H.sparseView();

        // Perform QR decomposition on H_sparse.
        SPQR<SparseMatrix<double> > spqr_helper;
        spqr_helper.setSPQROrdering(SPQR_ORDERING_NATURAL);
        spqr_helper.compute(H_sparse);

        MatrixXd H_temp;
        VectorXd r_temp;
        (spqr_helper.matrixQ().transpose() * H).evalTo(H_temp);
        (spqr_helper.matrixQ().transpose() * r).evalTo(r_temp);

        H_thin = H_temp.topRows(21+state_server.cam_states.size()*6);
        r_thin = r_temp.head(21+state_server.cam_states.size()*6);


    } else {
        H_thin = H;
        r_thin = r;
    }

    // Compute the Kalman gain.
    const MatrixXd& P = state_server.state_cov;
    MatrixXd S = H_thin*P*H_thin.transpose() +
            Feature::observation_noise*MatrixXd::Identity(
                H_thin.rows(), H_thin.rows());
    //MatrixXd K_transpose = S.fullPivHouseholderQr().solve(H_thin*P);
    MatrixXd K_transpose = S.ldlt().solve(H_thin*P);
    MatrixXd K = K_transpose.transpose();

    // Compute the error of the state.
    VectorXd delta_x = K * r_thin;

    // delta_x_imu
    const VectorXd& delta_x_imu = delta_x.head<21>();

    if (
        delta_x_imu.segment<3>(6).norm() > 0.5 ||
        delta_x_imu.segment<3>(12).norm() > 1.0) {
        printf("delta velocity: %f\n", delta_x_imu.segment<3>(6).norm());
        printf("delta position: %f\n", delta_x_imu.segment<3>(12).norm());
        ROS_WARN("Update change is too large.");
    }

    // Update imu_state
    const Vector4d dq_imu =
            smallAngleQuaternion(delta_x_imu.head<3>());
    state_server.imu_state.orientation = quaternionMultiplication(
                dq_imu, state_server.imu_state.orientation);
    state_server.imu_state.gyro_bias += delta_x_imu.segment<3>(3);
    state_server.imu_state.velocity += delta_x_imu.segment<3>(6);
    state_server.imu_state.acc_bias += delta_x_imu.segment<3>(9);
    state_server.imu_state.position += delta_x_imu.segment<3>(12);

    // update extrinsic
    const Vector4d dq_extrinsic =
            smallAngleQuaternion(delta_x_imu.segment<3>(15));
    state_server.imu_state.R_imu_cam0 = quaternionToRotation(
                dq_extrinsic) * state_server.imu_state.R_imu_cam0;
    state_server.imu_state.t_cam0_imu += delta_x_imu.segment<3>(18);

    Eigen::Isometry3d T_cam0_imu = Eigen::Isometry3d::Identity();
    T_cam0_imu.linear() = state_server.imu_state.R_imu_cam0 .inverse();
    T_cam0_imu.translation() =  state_server.imu_state.t_cam0_imu;
    state_server.imu_state.T_imu_cam0 = T_cam0_imu.inverse();
    T_body_cam0 = state_server.imu_state.T_imu_cam0 * state_server.imu_state.T_imu_body.inverse();
    R_body_cam0 = T_body_cam0.linear();
    t_body_cam0 = T_body_cam0.translation();

    // Update the camera states.
   auto cam_state_iter = state_server.cam_states.begin();
    for (int i = 0; i < state_server.cam_states.size();
         ++i, ++cam_state_iter) {
        const VectorXd& delta_x_cam = delta_x.segment<6>(21+i*6);
        const Vector4d dq_cam = smallAngleQuaternion(delta_x_cam.head<3>());
        cam_state_iter->second.orientation = quaternionMultiplication(
                    dq_cam, cam_state_iter->second.orientation);
        cam_state_iter->second.position += delta_x_cam.segment<3>(3);
    }

    // Update state covariance.
    MatrixXd I_KH = MatrixXd::Identity(K.rows(), H_thin.cols()) - K*H_thin;
    //state_server.state_cov = I_KH*state_server.state_cov*I_KH.transpose() +
    //  K*K.transpose()*Feature::observation_noise;
    state_server.state_cov = I_KH*state_server.state_cov;

    // Fix the covariance to be symmetric
    MatrixXd state_cov_fixed = (state_server.state_cov +
                                state_server.state_cov.transpose()) / 2.0;
    state_server.state_cov = state_cov_fixed;

    return;
}

/**
 * @brief imuCallback 
 *  
 */
void AckermannMsckf::imuCallback(
        const sensor_msgs::ImuConstPtr& msg) {

    imu_msg_buffer.push_back(*msg);

    if (!is_gravity_set) {
        if (imu_msg_buffer.size() < 200) return;
        initializeGravityAndBias();

        cout << "^^^^^^^^^^^^^^^^^^^^^^^^^^^" << endl;
        cout << "gyro_bias:\n" << state_server.imu_state.gyro_bias.transpose() << endl;
        cout << "^^^^^^^^^^^^^^^^^^^^^^^^^^^" << endl;

        is_gravity_set = true;
    }

    return;
}

/**
 * @brief ackCallback 
 *  
 */
void AckermannMsckf::ackermannCallback(const ackermann_msckf::AckermannDriveStamped::ConstPtr& msg){
  ackermann_msg_buffer.push_back(*msg);
  return;
}

/**
 * @brief batchImuProcessing 
 *  
 */
void AckermannMsckf::batchImuProcessing(const double& time_bound) {
    // Counter how many IMU msgs in the buffer are used.
    int used_imu_msg_cntr = 0;
    for (const auto& imu_msg : imu_msg_buffer) {
        double imu_time = imu_msg.header.stamp.toSec();
        if (imu_time < state_server.imu_state.time) {
            ++used_imu_msg_cntr;
            continue;
        }
        if (imu_time > time_bound) break;

        // Convert the msgs.
        Vector3d m_gyro, m_acc;
        tf::vectorMsgToEigen(imu_msg.angular_velocity, m_gyro);
        tf::vectorMsgToEigen(imu_msg.linear_acceleration, m_acc);

        // Execute process model.
        processModel(imu_time, m_gyro, m_acc);
        ++used_imu_msg_cntr;
    }

    // Set the state ID for the new IMU state.
    state_server.imu_state.id = IMUState::next_id++;

    // Remove all used IMU msgs.
    imu_msg_buffer.erase(imu_msg_buffer.begin(),
                         imu_msg_buffer.begin()+used_imu_msg_cntr);

    return;
}

/**
 * @brief processModel 
 *  
 */
void AckermannMsckf::processModel(const double& time,
                            const Vector3d& m_gyro,
                            const Vector3d& m_acc) {

    // Remove the bias from the measured gyro and acceleration
    IMUState& imu_state = state_server.imu_state;
    Vector3d gyro = m_gyro - imu_state.gyro_bias;
    Vector3d acc = m_acc - imu_state.acc_bias;
    double dtime = time - imu_state.time;

    // Compute discrete transition and noise covariance matrix
    Matrix<double, 21, 21> F = Matrix<double, 21, 21>::Zero();
    Matrix<double, 21, 12> G = Matrix<double, 21, 12>::Zero();

    F.block<3, 3>(0, 0) = -skewSymmetric(gyro);
    F.block<3, 3>(0, 3) = -Matrix3d::Identity();
    F.block<3, 3>(6, 0) = -quaternionToRotation(
                imu_state.orientation).transpose()*skewSymmetric(acc);
    F.block<3, 3>(6, 9) = -quaternionToRotation(
                imu_state.orientation).transpose();
    F.block<3, 3>(12, 6) = Matrix3d::Identity();

    G.block<3, 3>(0, 0) = -Matrix3d::Identity();
    G.block<3, 3>(3, 3) = Matrix3d::Identity();
    G.block<3, 3>(6, 6) = -quaternionToRotation(
                imu_state.orientation).transpose();
    G.block<3, 3>(9, 9) = Matrix3d::Identity();

    // Approximate matrix exponential to the 3rd order
    Matrix<double, 21, 21> Fdt = F * dtime;
    Matrix<double, 21, 21> Fdt_square = Fdt * Fdt;
    Matrix<double, 21, 21> Fdt_cube = Fdt_square * Fdt;
    Matrix<double, 21, 21> Phi = Matrix<double, 21, 21>::Identity() +
            Fdt + 0.5*Fdt_square + (1.0/6.0)*Fdt_cube;

    // Propogate the state using 4th order Runge-Kutta
    predictNewState(dtime, gyro, acc);

    // Modify the transition matrix
    Matrix3d R_kk_1 = quaternionToRotation(imu_state.orientation_null);
    Phi.block<3, 3>(0, 0) =
            quaternionToRotation(imu_state.orientation) * R_kk_1.transpose();

    Vector3d u = R_kk_1 * IMUState::gravity;
    RowVector3d s = (u.transpose()*u).inverse() * u.transpose();

    Matrix3d A1 = Phi.block<3, 3>(6, 0);
    Vector3d w1 = skewSymmetric(
                imu_state.velocity_null-imu_state.velocity) * IMUState::gravity;
    Phi.block<3, 3>(6, 0) = A1 - (A1*u-w1)*s;

    Matrix3d A2 = Phi.block<3, 3>(12, 0);
    Vector3d w2 = skewSymmetric(
                dtime*imu_state.velocity_null+imu_state.position_null-
                imu_state.position) * IMUState::gravity;
    Phi.block<3, 3>(12, 0) = A2 - (A2*u-w2)*s;

    // Propogate the state covariance matrix.
    Matrix<double, 21, 21> Q = Phi*G*state_server.continuous_noise_cov*
            G.transpose()*Phi.transpose()*dtime;
    state_server.state_cov.block<21, 21>(0, 0) =
            Phi*state_server.state_cov.block<21, 21>(0, 0)*Phi.transpose() + Q;

    //             [ P_I_I(k|k)      P_I_C(k|k)]
    // P_k_k(-)  = [                           ]
    //             [ P_I_C(k|k).T    P_C_C(k|k)]
    //          [ P_I_I(k+1|k)    Φ * P_I_C(k|k)]
    // P_k_k  = [                           ]
    //          [ P_I_C(k|k).T * Φ.T  P_C_C(k|k)]
    if (state_server.cam_states.size() > 0) {
        state_server.state_cov.block(
                    0, 21, 21, state_server.state_cov.cols()-21) =
                Phi * state_server.state_cov.block(
                    0, 21, 21, state_server.state_cov.cols()-21);
        state_server.state_cov.block(
                    21, 0, state_server.state_cov.rows()-21, 21) =
                state_server.state_cov.block(
                    21, 0, state_server.state_cov.rows()-21, 21) * Phi.transpose();
    }

    MatrixXd state_cov_fixed = (state_server.state_cov +
                                state_server.state_cov.transpose()) / 2.0;
    state_server.state_cov = state_cov_fixed;

    // Update the state correspondes to null space.
    imu_state.orientation_null = imu_state.orientation;
    imu_state.position_null = imu_state.position;
    imu_state.velocity_null = imu_state.velocity;

    // Update the state info
    state_server.imu_state.time = time;
    return;
}

/**
 * @brief predictNewState 
 *  
 */
void AckermannMsckf::predictNewState(const double& dt,
                               const Vector3d& gyro,
                               const Vector3d& acc) {

    double gyro_norm = gyro.norm();
    Matrix4d Omega = Matrix4d::Zero();
    Omega.block<3, 3>(0, 0) = -skewSymmetric(gyro);
    Omega.block<3, 1>(0, 3) = gyro;
    Omega.block<1, 3>(3, 0) = -gyro;

    Vector4d& q = state_server.imu_state.orientation;
    Vector3d& v = state_server.imu_state.velocity;
    Vector3d& p = state_server.imu_state.position;

    Vector4d dq_dt, dq_dt2;
    if (gyro_norm > 1e-5) {
        dq_dt = (cos(gyro_norm*dt*0.5)*Matrix4d::Identity() +
                 1/gyro_norm*sin(gyro_norm*dt*0.5)*Omega) * q;
        dq_dt2 = (cos(gyro_norm*dt*0.25)*Matrix4d::Identity() +
                  1/gyro_norm*sin(gyro_norm*dt*0.25)*Omega) * q;
    }
    else {
        dq_dt = (Matrix4d::Identity()+0.5*dt*Omega) *
                cos(gyro_norm*dt*0.5) * q;
        dq_dt2 = (Matrix4d::Identity()+0.25*dt*Omega) *
                cos(gyro_norm*dt*0.25) * q;
    }
    Matrix3d dR_dt_transpose = quaternionToRotation(dq_dt).transpose();
    Matrix3d dR_dt2_transpose = quaternionToRotation(dq_dt2).transpose();

    // k1 = f(tn, yn)
    Vector3d k1_v_dot = quaternionToRotation(q).transpose()*acc +
            IMUState::gravity;
    Vector3d k1_p_dot = v;

    // k2 = f(tn+dt/2, yn+k1*dt/2)
    Vector3d k1_v = v + k1_v_dot*dt/2;
    Vector3d k2_v_dot = dR_dt2_transpose*acc +
            IMUState::gravity;
    Vector3d k2_p_dot = k1_v;

    // k3 = f(tn+dt/2, yn+k2*dt/2)
    Vector3d k2_v = v + k2_v_dot*dt/2;
    Vector3d k3_v_dot = dR_dt2_transpose*acc +
            IMUState::gravity;
    Vector3d k3_p_dot = k2_v;

    // k4 = f(tn+dt, yn+k3*dt)
    Vector3d k3_v = v + k3_v_dot*dt;
    Vector3d k4_v_dot = dR_dt_transpose*acc +
            IMUState::gravity;
    Vector3d k4_p_dot = k3_v;

    // yn+1 = yn + dt/6*(k1+2*k2+2*k3+k4)
    q = dq_dt;
    quaternionNormalize(q);
    v = v + dt/6*(k1_v_dot+2*k2_v_dot+2*k3_v_dot+k4_v_dot);
    p = p + dt/6*(k1_p_dot+2*k2_p_dot+2*k3_p_dot+k4_p_dot);

    state_server.imu_state.angular_velocity = gyro;
    return;
}

/**
 * @brief stateAugmentation 
 *  
 */
void AckermannMsckf::stateAugmentation(const double& time) {

    const Matrix3d& R_i_c = state_server.imu_state.R_imu_cam0;
    const Vector3d& t_c_i = state_server.imu_state.t_cam0_imu;

    // Add a new camera state to the state server.
    Matrix3d R_w_i = quaternionToRotation(
                state_server.imu_state.orientation);
    Matrix3d R_w_c = R_i_c * R_w_i;
    Vector3d t_c_w = state_server.imu_state.position +
            R_w_i.transpose()*t_c_i;

    state_server.cam_states[state_server.imu_state.id] =
            CAMState(state_server.imu_state.id);
    CAMState& cam_state = state_server.cam_states[
            state_server.imu_state.id];

    cam_state.time = time;
    cam_state.orientation = rotationToQuaternion(R_w_c);
    cam_state.position = t_c_w;

    cam_state.orientation_null = cam_state.orientation;
    cam_state.position_null = cam_state.position;

    // Update the covariance matrix of the state.
    // To simplify computation, the matrix J below is the nontrivial block
    // in Equation (16) in "A Multi-State Constraint Kalman Filter for Vision
    // -aided Inertial Navigation".
    Matrix<double, 6, 21> J = Matrix<double, 6, 21>::Zero();
    J.block<3, 3>(0, 0) = R_i_c;
    J.block<3, 3>(0, 15) = Matrix3d::Identity();
    J.block<3, 3>(3, 0) = -R_w_i.transpose()*skewSymmetric(t_c_i);
    J.block<3, 3>(3, 12) = Matrix3d::Identity();
    J.block<3, 3>(3, 18) = R_w_i.transpose();

    // Resize the state covariance matrix.
    size_t old_rows = state_server.state_cov.rows();
    size_t old_cols = state_server.state_cov.cols();
    state_server.state_cov.conservativeResize(old_rows+6, old_cols+6);

    // Rename some matrix blocks for convenience.
    const Matrix<double, 21, 21>& P11 =
            state_server.state_cov.block<21, 21>(0, 0);
    const MatrixXd& P12 =
            state_server.state_cov.block(0, 21, 21, old_cols-21);

    // Fill in the augmented state covariance.
    state_server.state_cov.block(old_rows, 0, 6, old_cols) << J*P11, J*P12;
    state_server.state_cov.block(0, old_cols, old_rows, 6) =
            state_server.state_cov.block(old_rows, 0, 6, old_cols).transpose();
    state_server.state_cov.block<6, 6>(old_rows, old_cols) =
            J * P11 * J.transpose();

    // Fix the covariance to be symmetric
    MatrixXd state_cov_fixed = (state_server.state_cov +
                                state_server.state_cov.transpose()) / 2.0;
    state_server.state_cov = state_cov_fixed;

    return;
}

/**
 * @brief addFeatureObservations 
 *  
 */
void AckermannMsckf::addFeatureObservations(
        const CameraMeasurementConstPtr& msg) {

    StateIDType state_id = state_server.imu_state.id;
    int curr_feature_num = map_server.size();
    int tracked_feature_num = 0;

    // Add new observations for existing features or new
    // features in the map server.
    for (const auto& feature : msg->features) {
        if (map_server.find(feature.id) == map_server.end()) {
            // This is a new feature.
            map_server[feature.id] = Feature(feature.id);
            map_server[feature.id].observations[state_id] =
                    Vector4d(feature.u0, feature.v0,
                             feature.u1, feature.v1);
        } else {
            // This is an old feature.
            map_server[feature.id].observations[state_id] =
                    Vector4d(feature.u0, feature.v0,
                             feature.u1, feature.v1);
            ++tracked_feature_num;
        }
    }

    tracking_rate =
            static_cast<double>(tracked_feature_num) /
            static_cast<double>(curr_feature_num);

    return;
}

/**
 * @brief measurementJacobian 
 *  camera features 
 */ 
void AckermannMsckf::measurementJacobian(
        const StateIDType& cam_state_id,
        const FeatureIDType& feature_id,
        Matrix<double, 4, 6>& H_x, Matrix<double, 4, 3>& H_f, Vector4d& r) {

    // Prepare all the required data.
    const CAMState& cam_state = state_server.cam_states[cam_state_id];
    const Feature& feature = map_server[feature_id];

    // Cam0 pose.
    Matrix3d R_w_c0 = quaternionToRotation(cam_state.orientation);
    const Vector3d& t_c0_w = cam_state.position;

    // Cam1 pose.
    Matrix3d R_c0_c1 = CAMState::T_cam0_cam1.linear();
    Matrix3d R_w_c1 = CAMState::T_cam0_cam1.linear() * R_w_c0;
    Vector3d t_c1_w = t_c0_w - R_w_c1.transpose()*CAMState::T_cam0_cam1.translation();

    // 3d feature position in the world frame.
    // And its observation with the stereo cameras.
    const Vector3d& p_w = feature.position;
    const Vector4d& z = feature.observations.find(cam_state_id)->second;

    // Convert the feature position from the world frame to
    // the cam0 and cam1 frame.
    Vector3d p_c0 = R_w_c0 * (p_w-t_c0_w);
    Vector3d p_c1 = R_w_c1 * (p_w-t_c1_w);

    // Compute the Jacobians.
    Matrix<double, 4, 3> dz_dpc0 = Matrix<double, 4, 3>::Zero();
    dz_dpc0(0, 0) = 1 / p_c0(2);
    dz_dpc0(1, 1) = 1 / p_c0(2);
    dz_dpc0(0, 2) = -p_c0(0) / (p_c0(2)*p_c0(2));
    dz_dpc0(1, 2) = -p_c0(1) / (p_c0(2)*p_c0(2));

    Matrix<double, 4, 3> dz_dpc1 = Matrix<double, 4, 3>::Zero();
    dz_dpc1(2, 0) = 1 / p_c1(2);
    dz_dpc1(3, 1) = 1 / p_c1(2);
    dz_dpc1(2, 2) = -p_c1(0) / (p_c1(2)*p_c1(2));
    dz_dpc1(3, 2) = -p_c1(1) / (p_c1(2)*p_c1(2));

    Matrix<double, 3, 6> dpc0_dxc = Matrix<double, 3, 6>::Zero();
    dpc0_dxc.leftCols(3) = skewSymmetric(p_c0);
    dpc0_dxc.rightCols(3) = -R_w_c0;

    Matrix<double, 3, 6> dpc1_dxc = Matrix<double, 3, 6>::Zero();
    dpc1_dxc.leftCols(3) = R_c0_c1 * skewSymmetric(p_c0);
    dpc1_dxc.rightCols(3) = -R_w_c1;

    Matrix3d dpc0_dpg = R_w_c0;
    Matrix3d dpc1_dpg = R_w_c1;

    H_x = dz_dpc0*dpc0_dxc + dz_dpc1*dpc1_dxc;
    H_f = dz_dpc0*dpc0_dpg + dz_dpc1*dpc1_dpg;

    // Modifty the measurement Jacobian to ensure
    // observability constrain.
    Matrix<double, 4, 6> A = H_x;
    Matrix<double, 6, 1> u = Matrix<double, 6, 1>::Zero();
    u.block<3, 1>(0, 0) = quaternionToRotation(
                cam_state.orientation_null) * IMUState::gravity;
    u.block<3, 1>(3, 0) = skewSymmetric(
                p_w-cam_state.position_null) * IMUState::gravity;
    H_x = A - A*u*(u.transpose()*u).inverse()*u.transpose();
    H_f = -H_x.block<4, 3>(0, 3);

    // Compute the residual.
    r = z - Vector4d(p_c0(0)/p_c0(2), p_c0(1)/p_c0(2),
                     p_c1(0)/p_c1(2), p_c1(1)/p_c1(2));

    return;
}

/**
 * @brief featureJacobian 
 *  
 */ 
void AckermannMsckf::featureJacobian(
        const FeatureIDType& feature_id,
        const std::vector<StateIDType>& cam_state_ids,
        MatrixXd& H_x, VectorXd& r) {

    const auto& feature = map_server[feature_id];

    // Check how many camera states in the provided camera
    // id camera has actually seen this feature.
    vector<StateIDType> valid_cam_state_ids(0);
    for (const auto& cam_id : cam_state_ids) {
        if (feature.observations.find(cam_id) ==
                feature.observations.end()) continue;

        valid_cam_state_ids.push_back(cam_id);
    }

    int jacobian_row_size = 0;
    jacobian_row_size = 4 * valid_cam_state_ids.size();

    MatrixXd H_xj = MatrixXd::Zero(jacobian_row_size,
                                   21+state_server.cam_states.size()*6);
    MatrixXd H_fj = MatrixXd::Zero(jacobian_row_size, 3);
    VectorXd r_j = VectorXd::Zero(jacobian_row_size);
    int stack_cntr = 0;

    for (const auto& cam_id : valid_cam_state_ids) {

        Matrix<double, 4, 6> H_xi = Matrix<double, 4, 6>::Zero();
        Matrix<double, 4, 3> H_fi = Matrix<double, 4, 3>::Zero();
        Vector4d r_i = Vector4d::Zero();
        measurementJacobian(cam_id, feature.id, H_xi, H_fi, r_i);

        auto cam_state_iter = state_server.cam_states.find(cam_id);
        int cam_state_cntr = std::distance(
                    state_server.cam_states.begin(), cam_state_iter);

        // Stack the Jacobians.
        H_xj.block<4, 6>(stack_cntr, 21+6*cam_state_cntr) = H_xi;
        H_fj.block<4, 3>(stack_cntr, 0) = H_fi;
        r_j.segment<4>(stack_cntr) = r_i;
        stack_cntr += 4;
    }

    // Project the residual and Jacobians onto the nullspace
    // of H_fj.
    JacobiSVD<MatrixXd> svd_helper(H_fj, ComputeFullU | ComputeThinV);
    MatrixXd A = svd_helper.matrixU().rightCols(
                jacobian_row_size - 3);

    H_x = A.transpose() * H_xj;
    r = A.transpose() * r_j;

    return;
}

/**
 * @brief gatingTest 
 *  
 */
bool AckermannMsckf::gatingTest(
        const MatrixXd& H, const VectorXd& r, const int& dof) {

    MatrixXd P1 = H * state_server.state_cov * H.transpose();
    MatrixXd P2 = Feature::observation_noise *
            MatrixXd::Identity(H.rows(), H.rows());
    double gamma = r.transpose() * (P1+P2).ldlt().solve(r);

    if (gamma < chi_squared_test_table[dof]) {
        return true;
    } else {
        return false;
    }

}

/**
 * @brief removeLostFeatures 
 *  
 */ 
void AckermannMsckf::removeLostFeatures() {

    // Remove the features that lost track.
    // BTW, find the size the final Jacobian matrix and residual vector.
    int jacobian_row_size = 0;
    vector<FeatureIDType> invalid_feature_ids(0);
    vector<FeatureIDType> processed_feature_ids(0);

    for (auto iter = map_server.begin();
         iter != map_server.end(); ++iter) {
        auto& feature = iter->second;

        // Pass the features that are still being tracked.
        if (feature.observations.find(state_server.imu_state.id) !=
                feature.observations.end()) continue;
        if (feature.observations.size() < 3) {
            invalid_feature_ids.push_back(feature.id);
            continue;
        }

        // Check if the feature can be initialized if it
        // has not been.
        if (!feature.is_initialized) {
            if (!feature.checkMotion(state_server.cam_states)) {
                invalid_feature_ids.push_back(feature.id);
                continue;
            } else {
                if(!feature.initializePosition(state_server.cam_states)) {
                    invalid_feature_ids.push_back(feature.id);
                    continue;
                }
            }
        }

        jacobian_row_size += 4*feature.observations.size() - 3;
        processed_feature_ids.push_back(feature.id);
    }

    // Remove the features that do not have enough measurements.
    for (const auto& feature_id : invalid_feature_ids)
        map_server.erase(feature_id);

    // Return if there is no lost feature to be processed.
    if (processed_feature_ids.size() == 0) return;

    MatrixXd H_x = MatrixXd::Zero(jacobian_row_size,
                                  21+6*state_server.cam_states.size());
    VectorXd r = VectorXd::Zero(jacobian_row_size);
    int stack_cntr = 0;

    // Process the features which lose track.
    for (const auto& feature_id : processed_feature_ids) {
        auto& feature = map_server[feature_id];

        vector<StateIDType> cam_state_ids(0);
        for (const auto& measurement : feature.observations)
            cam_state_ids.push_back(measurement.first);

        MatrixXd H_xj;
        VectorXd r_j;
        featureJacobian(feature.id, cam_state_ids, H_xj, r_j);

        if (gatingTest(H_xj, r_j, cam_state_ids.size()-1)) {
            H_x.block(stack_cntr, 0, H_xj.rows(), H_xj.cols()) = H_xj;
            r.segment(stack_cntr, r_j.rows()) = r_j;
            stack_cntr += H_xj.rows();
        }

        // Put an upper bound on the row size of measurement Jacobian,
        // which helps guarantee the executation time.
        if (stack_cntr > 1500) break;
    }

    H_x.conservativeResize(stack_cntr, H_x.cols());
    r.conservativeResize(stack_cntr);

    // Perform the measurement update step.
    measurementUpdate(H_x, r);

    // Remove all processed features from the map.
    for (const auto& feature_id : processed_feature_ids)
        map_server.erase(feature_id);

    return;
}

/**
 * @brief findRedundantCamStates 
 *  
 */
void AckermannMsckf::findRedundantCamStates(
        vector<StateIDType>& rm_cam_state_ids) {

    // Move the iterator to the key position.
    auto key_cam_state_iter = state_server.cam_states.end();
    for (int i = 0; i < 4; ++i)
        --key_cam_state_iter;
    auto cam_state_iter = key_cam_state_iter;
    ++cam_state_iter;
    auto first_cam_state_iter = state_server.cam_states.begin();

    // Pose of the key camera state.
    const Vector3d key_position =
            key_cam_state_iter->second.position;
    const Matrix3d key_rotation = quaternionToRotation(
                key_cam_state_iter->second.orientation);

    // Mark the camera states to be removed based on the
    // motion between states.
    for (int i = 0; i < 2; ++i) {
        const Vector3d position =
                cam_state_iter->second.position;
        const Matrix3d rotation = quaternionToRotation(
                    cam_state_iter->second.orientation);

        double distance = (position-key_position).norm();
        double angle = AngleAxisd(
                    rotation*key_rotation.transpose()).angle();

        if (angle < rotation_threshold &&
            distance < translation_threshold &&
            tracking_rate > tracking_rate_threshold) {
            rm_cam_state_ids.push_back(cam_state_iter->first);
            ++cam_state_iter;
        } else {
            rm_cam_state_ids.push_back(first_cam_state_iter->first);
            ++first_cam_state_iter;
        }
    }

    // Sort the elements in the output vector.
    sort(rm_cam_state_ids.begin(), rm_cam_state_ids.end());

    return;
}

/**
 * @brief pruneCamStateBuffer 
 *  
 */ 
void AckermannMsckf::pruneCamStateBuffer() {

    if (state_server.cam_states.size() < max_cam_state_size)
        return;

    // Find two camera states to be removed.
    vector<StateIDType> rm_cam_state_ids(0);
    findRedundantCamStates(rm_cam_state_ids);

    // Find the size of the Jacobian matrix.
    int jacobian_row_size = 0;
    for (auto& item : map_server) {
        auto& feature = item.second;
        // Check how many camera states to be removed are associated
        // with this feature.
        vector<StateIDType> involved_cam_state_ids(0);
        for (const auto& cam_id : rm_cam_state_ids) {
            if (feature.observations.find(cam_id) !=
                    feature.observations.end())
                involved_cam_state_ids.push_back(cam_id);
        }

        if (involved_cam_state_ids.size() == 0) continue;
        if (involved_cam_state_ids.size() == 1) {
            feature.observations.erase(involved_cam_state_ids[0]);
            continue;
        }

        if (!feature.is_initialized) {
            // Check if the feature can be initialize.
            if (!feature.checkMotion(state_server.cam_states)) {
                // If the feature cannot be initialized, just remove
                // the observations associated with the camera states
                // to be removed.
                for (const auto& cam_id : involved_cam_state_ids)
                    feature.observations.erase(cam_id);
                continue;
            } else {
                if(!feature.initializePosition(state_server.cam_states)) {
                    for (const auto& cam_id : involved_cam_state_ids)
                        feature.observations.erase(cam_id);
                    continue;
                }
            }
        }

        jacobian_row_size += 4*involved_cam_state_ids.size() - 3;
    }

    // Compute the Jacobian and residual.
    MatrixXd H_x = MatrixXd::Zero(jacobian_row_size,
                                  21+6*state_server.cam_states.size());
    VectorXd r = VectorXd::Zero(jacobian_row_size);
    int stack_cntr = 0;

    for (auto& item : map_server) {
        auto& feature = item.second;
        // Check how many camera states to be removed are associated
        // with this feature.
        vector<StateIDType> involved_cam_state_ids(0);
        for (const auto& cam_id : rm_cam_state_ids) {
            if (feature.observations.find(cam_id) !=
                    feature.observations.end())
                involved_cam_state_ids.push_back(cam_id);
        }

        if (involved_cam_state_ids.size() == 0) continue;

        MatrixXd H_xj;
        VectorXd r_j;
        featureJacobian(feature.id, involved_cam_state_ids, H_xj, r_j);

        if (gatingTest(H_xj, r_j, involved_cam_state_ids.size())) {
            H_x.block(stack_cntr, 0, H_xj.rows(), H_xj.cols()) = H_xj;
            r.segment(stack_cntr, r_j.rows()) = r_j;
            stack_cntr += H_xj.rows();
        }

        for (const auto& cam_id : involved_cam_state_ids)
            feature.observations.erase(cam_id);
    }

    H_x.conservativeResize(stack_cntr, H_x.cols());
    r.conservativeResize(stack_cntr);

    // Perform measurement update.
    measurementUpdate(H_x, r);

    for (const auto& cam_id : rm_cam_state_ids) {
        int cam_sequence = std::distance(state_server.cam_states.begin(),
                                         state_server.cam_states.find(cam_id));
        int cam_state_start = 21 + 6*cam_sequence;
        int cam_state_end = cam_state_start + 6;

        // Remove the corresponding rows and columns in the state
        // covariance matrix.
        if (cam_state_end < state_server.state_cov.rows()) {
            state_server.state_cov.block(cam_state_start, 0,
                                         state_server.state_cov.rows()-cam_state_end,
                                         state_server.state_cov.cols()) =
                    state_server.state_cov.block(cam_state_end, 0,
                                                 state_server.state_cov.rows()-cam_state_end,
                                                 state_server.state_cov.cols());

            state_server.state_cov.block(0, cam_state_start,
                                         state_server.state_cov.rows(),
                                         state_server.state_cov.cols()-cam_state_end) =
                    state_server.state_cov.block(0, cam_state_end,
                                                 state_server.state_cov.rows(),
                                                 state_server.state_cov.cols()-cam_state_end);

            state_server.state_cov.conservativeResize(
                        state_server.state_cov.rows()-6, state_server.state_cov.cols()-6);
        } else {
            state_server.state_cov.conservativeResize(
                        state_server.state_cov.rows()-6, state_server.state_cov.cols()-6);
        }

        // Remove this camera state in the state vector.
        state_server.cam_states.erase(cam_id);
    }

    return;
}

/**
 * @brief onlineReset 
 *  
 */ 
void AckermannMsckf::onlineReset() {

    // Never perform online reset if position std threshold
    // is non-positive.
    if (position_std_threshold <= 0) return;
    static long long int online_reset_counter = 0;

    // Check the uncertainty of positions to determine if
    // the system can be reset.
    double position_x_std = std::sqrt(state_server.state_cov(12, 12));
    double position_y_std = std::sqrt(state_server.state_cov(13, 13));
    double position_z_std = std::sqrt(state_server.state_cov(14, 14));

    if (position_x_std < position_std_threshold &&
            position_y_std < position_std_threshold &&
            position_z_std < position_std_threshold) return;

    ROS_WARN("Start %lld online reset procedure...",
             ++online_reset_counter);
    ROS_INFO("Stardard deviation in xyz: %f, %f, %f",
             position_x_std, position_y_std, position_z_std);

    // Remove all existing camera states.
    state_server.cam_states.clear();

    // Clear all exsiting features in the map.
    map_server.clear();

    // Reset the state covariance.
    double gyro_bias_cov, acc_bias_cov, velocity_cov;
    nh.param<double>("initial_covariance/velocity",
                     velocity_cov, 0.25);
    nh.param<double>("initial_covariance/gyro_bias",
                     gyro_bias_cov, 1e-4);
    nh.param<double>("initial_covariance/acc_bias",
                     acc_bias_cov, 1e-2);

    double extrinsic_rotation_cov, extrinsic_translation_cov;
    nh.param<double>("initial_covariance/extrinsic_rotation_cov",
                     extrinsic_rotation_cov, 3.0462e-4);
    nh.param<double>("initial_covariance/extrinsic_translation_cov",
                     extrinsic_translation_cov, 1e-4);

    state_server.state_cov = MatrixXd::Zero(21, 21);
    for (int i = 3; i < 6; ++i)
        state_server.state_cov(i, i) = gyro_bias_cov;
    for (int i = 6; i < 9; ++i)
        state_server.state_cov(i, i) = velocity_cov;
    for (int i = 9; i < 12; ++i)
        state_server.state_cov(i, i) = acc_bias_cov;
    for (int i = 15; i < 18; ++i)
        state_server.state_cov(i, i) = extrinsic_rotation_cov;
    for (int i = 18; i < 21; ++i)
        state_server.state_cov(i, i) = extrinsic_translation_cov;

    ROS_WARN("%lld online reset complete...", online_reset_counter);
    return;
}

/**
 * @brief publish 
 *  
 */ 
void AckermannMsckf::publish(const ros::Time& time) {

  const IMUState& imu_state = state_server.imu_state;

  // T_Ii_GI
  Eigen::Isometry3d T_Ii_GI = Eigen::Isometry3d::Identity();
  T_Ii_GI.linear() = quaternionToRotation(
      imu_state.orientation).transpose();
  T_Ii_GI.translation() = imu_state.position;

  Eigen::Isometry3d T_GI_I0 = Eigen::Isometry3d::Identity();
  T_GI_I0.linear() = q_G_I0_.toRotationMatrix();
  T_GI_I0.translation() = Eigen::Vector3d (0,0,0);
  
  Eigen::Isometry3d T_GI_B0 = imu_state.T_imu_body * T_GI_I0;
  Eigen::Isometry3d T_GI_W = Eigen::Isometry3d::Identity();
  Eigen::Isometry3d T_GI_GB = Eigen::Isometry3d::Identity();
  
  T_GI_W = T_GI_B0;
  T_GI_GB = T_B0_GS_gt * T_GI_B0;
  
////////////////////////////////////////////////////////////////////////////////
  // {EST and GT}
  Eigen::Quaterniond q_Bi_W;
  Vector3d t_Bi_W;
  Eigen::Quaterniond q_GBi_GB;
  Vector3d t_GBi_GB;
  Eigen::Quaterniond q_Bi_W_gt;
  Vector3d t_Bi_W_gt;
  Eigen::Quaterniond q_GBi_GB_gt;
  Vector3d t_GBi_GB_gt;

  // {EST}
  // {W}
  Eigen::Isometry3d T_Bi_W = T_GI_W * T_Ii_GI * imu_state.T_imu_body.inverse();
  q_Bi_W = T_Bi_W.linear();
  t_Bi_W = T_Bi_W.translation();
  // {G}
  Eigen::Isometry3d T_GBi_GB = T_GI_GB * T_Ii_GI * imu_state.T_imu_body.inverse();
  q_GBi_GB = T_GBi_GB.linear();
  t_GBi_GB = T_GBi_GB.translation();
  
  // {GT} 
  Eigen::Quaterniond gt_orientation;
  Eigen::Vector3d gt_translation;
  Eigen::Vector3d gt_translation_last;
  Eigen::Vector3d gt_bv;
  Eigen::Vector3d gt_bw;
  tf::pointMsgToEigen(
      gt_odom_last.pose.pose.position, gt_translation_last);
  tf::pointMsgToEigen(
      gt_odom_curr.pose.pose.position, gt_translation);
  tf::quaternionMsgToEigen(
      gt_odom_curr.pose.pose.orientation, gt_orientation);
  tf::vectorMsgToEigen(gt_odom_curr.twist.twist.linear,gt_bv);
  tf::vectorMsgToEigen(gt_odom_curr.twist.twist.angular,gt_bw);
  // {G}
  Eigen::Isometry3d gt_T_Si_GS = Eigen::Isometry3d::Identity();
  gt_T_Si_GS.linear() = gt_orientation.toRotationMatrix();
  gt_T_Si_GS.translation() = gt_translation;
  q_GBi_GB_gt = gt_T_Si_GS.linear();
  t_GBi_GB_gt = gt_T_Si_GS.translation();
  // {W}
  Eigen::Isometry3d gt_T_Si_W = Eigen::Isometry3d::Identity();
  gt_T_Si_W = T_B0_GS_gt.inverse() * gt_T_Si_GS;
  q_Bi_W_gt = gt_T_Si_W.linear();
  t_Bi_W_gt = gt_T_Si_W.translation();

  // save pose {W}
  Eigen::Vector3d p_wi = t_Bi_W;
  Eigen::Quaterniond q_wi = q_Bi_W;
  Eigen::Vector3d gt_p_wi = t_Bi_W_gt;
  Eigen::Quaterniond gt_q_wi =  q_Bi_W_gt;

  // EST roll pitch yaw
  double roll_GBi_GB, pitch_GBi_GB, yaw_GBi_GB;
  tf::Matrix3x3(tf::Quaternion(q_GBi_GB.x(),q_GBi_GB.y(),q_GBi_GB.z(),q_GBi_GB.w())).getRPY(roll_GBi_GB, pitch_GBi_GB, yaw_GBi_GB, 1); 
  roll_GBi_GB = roll_GBi_GB * 180 / M_PI;
  pitch_GBi_GB = pitch_GBi_GB * 180 / M_PI;
  yaw_GBi_GB = yaw_GBi_GB * 180 / M_PI;

  // GT roll pitch yaw
  double roll_GBi_GB_gt, pitch_GBi_GB_gt, yaw_GBi_GB_gt;
  tf::Matrix3x3(tf::Quaternion(q_GBi_GB_gt.x(),q_GBi_GB_gt.y(),q_GBi_GB_gt.z(),q_GBi_GB_gt.w())).getRPY(roll_GBi_GB_gt, pitch_GBi_GB_gt, yaw_GBi_GB_gt, 1);
  roll_GBi_GB_gt = roll_GBi_GB_gt * 180 / M_PI;
  pitch_GBi_GB_gt = pitch_GBi_GB_gt * 180 / M_PI;
  yaw_GBi_GB_gt = yaw_GBi_GB_gt * 180 / M_PI;

////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////
  // Publish tf {W}
  if (publish_tf) {
    tf::Transform T_Bi_W_tf;
    tf::transformEigenToTF(T_Bi_W, T_Bi_W_tf);
    tf_pub.sendTransform(tf::StampedTransform(
          T_Bi_W_tf, time, fixed_frame_id, child_frame_id));
  }
  // Publish the odometry
  nav_msgs::Odometry odom_msg;
  odom_msg.header.stamp = time;
  odom_msg.header.frame_id = fixed_frame_id;
  odom_msg.child_frame_id = child_frame_id;

  Eigen::Matrix3d R_Bi_GI = T_Ii_GI.linear() *  imu_state.T_imu_body.inverse().linear();
  Eigen::Matrix3d R_GI_Bi = R_Bi_GI.inverse();
  // bv
  const Vector3d& w_G_i = state_server.imu_state.angular_velocity;
  Eigen::Vector3d b_v = R_GI_Bi * imu_state.velocity + imu_state.T_imu_body.linear() * skewSymmetric(w_G_i) * (imu_state.T_imu_body.inverse().translation());
  // bw
  Eigen::Vector3d b_w = imu_state.T_imu_body.linear() * w_G_i;
  tf::poseEigenToMsg(T_Bi_W, odom_msg.pose.pose);
  tf::vectorEigenToMsg(b_v, odom_msg.twist.twist.linear);
  tf::vectorEigenToMsg(b_w, odom_msg.twist.twist.angular);

  // Convert the covariance.
  Matrix3d P_oo = state_server.state_cov.block<3, 3>(0, 0);
  Matrix3d P_op = state_server.state_cov.block<3, 3>(0, 12);
  Matrix3d P_po = state_server.state_cov.block<3, 3>(12, 0);
  Matrix3d P_pp = state_server.state_cov.block<3, 3>(12, 12);
  Matrix<double, 6, 6> P_imu_pose = Matrix<double, 6, 6>::Zero();
  P_imu_pose << P_pp, P_po, P_op, P_oo;
  Matrix<double, 6, 6> H_pose = Matrix<double, 6, 6>::Zero();
  H_pose.block<3, 3>(0, 0) =  T_GI_W.linear();
  H_pose.block<3, 3>(3, 3) =  T_GI_W.linear();
  Matrix<double, 6, 6> P_body_pose = H_pose *
    P_imu_pose * H_pose.transpose();
  for (int i = 0; i < 6; ++i)
    for (int j = 0; j < 6; ++j)
      odom_msg.pose.covariance[6*i+j] = P_body_pose(i, j);
  // Construct the covariance for the velocity.
  Matrix3d P_imu_vel = state_server.state_cov.block<3, 3>(6, 6);
  Matrix3d H_vel = R_GI_Bi;
  Matrix3d P_body_vel = H_vel * P_imu_vel * H_vel.transpose();
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      odom_msg.twist.covariance[i*6+j] = P_body_vel(i, j);
  odom_pub.publish(odom_msg);

  // Publish the 3D positions of the features {W}
  pcl::PointCloud<pcl::PointXYZ>::Ptr feature_msg_ptr(
      new pcl::PointCloud<pcl::PointXYZ>());
  feature_msg_ptr->header.frame_id = fixed_frame_id;
  feature_msg_ptr->height = 1;
  for (const auto& item : map_server) {
    const auto& feature = item.second;
    if (feature.is_initialized) {
      
      Vector3d feature_position =
        T_GI_W.linear() * feature.position + T_GI_W.translation();

      feature_msg_ptr->points.push_back(pcl::PointXYZ(
            feature_position(0), feature_position(1), feature_position(2)));
    }
  }
  feature_msg_ptr->width = feature_msg_ptr->points.size();
  feature_pub.publish(feature_msg_ptr);
//////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////
    // save
    double dStamp = time.toSec();
                
    // GT
    Eigen::Isometry3d gt_T_Ii_GI = Eigen::Isometry3d::Identity();
    gt_T_Ii_GI = T_GI_GB.inverse() * gt_T_Si_GS * imu_state.T_imu_body;
    Eigen::Quaterniond gt_q_Ii_GI;
    Eigen::Vector3d gt_t_Ii_GI;
    Eigen::Vector3d gt_GIv;
    gt_q_Ii_GI = gt_T_Ii_GI.linear();
    gt_t_Ii_GI = gt_T_Ii_GI.translation();
    gt_GIv = R_GI_Bi.transpose() * (gt_bv - imu_state.T_imu_body.linear() * skewSymmetric(w_G_i) * (imu_state.T_imu_body.inverse().translation()));

    // save odom 
    double scale_ratio;
    double delta_t_est = (t_GBi_GB - t_GBi_GB_last).norm();
    double delta_t_gt = (gt_translation - gt_translation_last).norm();
    double delta_t_est_gt = delta_t_est - delta_t_gt;
    t_GBi_GB_last = t_GBi_GB;
    gt_odom_last = gt_odom_curr;
    Eigen::Vector3d t_Ii_GI = T_Ii_GI.translation();
    Eigen::Quaterniond q_Ii_GI = Eigen::Quaterniond(T_Ii_GI.linear());
    if(is_first_sr){
        is_first_sr = false;
        scale_ratio = 0;
    }else{
        if(delta_t_est <= 0.01 && abs(delta_t_est_gt) <= 0.03){
            scale_ratio = 0;
        }else if(delta_t_gt <= 0.01 && abs(delta_t_est_gt) <= 0.03){
        scale_ratio = 0;
        }else if(delta_t_est <= 0.01 && abs(delta_t_est_gt) > 0.03){
            scale_ratio = -(delta_t_gt / (delta_t_est + 0.03) - 1);
        }else if(delta_t_gt <= 0.01 && abs(delta_t_est_gt) > 0.03){
            scale_ratio = delta_t_est / (delta_t_gt + 0.03) - 1;
        }else{
            if(delta_t_est_gt >=0){
                scale_ratio = delta_t_est / delta_t_gt - 1;
            }else{
                scale_ratio = -(delta_t_gt / delta_t_est - 1);
            }
        }
    }  
    global_count++;
    global_scale_ratio += scale_ratio;

    CSVDATA_ODOM csvdata_odom;
    csvdata_odom.time = dStamp;
    csvdata_odom.Dtime = Dtime;
    csvdata_odom.pB = p_wi;
    csvdata_odom.qB = q_wi;
    csvdata_odom.roll = roll_GBi_GB;
    csvdata_odom.pitch = pitch_GBi_GB;
    csvdata_odom.yaw = yaw_GBi_GB;
    csvdata_odom.vB = b_v;
    csvdata_odom.wB = b_w;
    csvdata_odom.gt_pB = gt_p_wi;
    csvdata_odom.gt_qB = gt_q_wi;
    csvdata_odom.gt_roll = roll_GBi_GB_gt;
    csvdata_odom.gt_pitch = pitch_GBi_GB_gt;
    csvdata_odom.gt_yaw = yaw_GBi_GB_gt;
    csvdata_odom.gt_vB = gt_bv;
    csvdata_odom.gt_wB = gt_bw;
    csvdata_odom.Sr = scale_ratio;
    csvdata_odom.Sr_avg = global_scale_ratio / global_count;
    csvData_odom.push_back(csvdata_odom);

    // save std {GI}
    Eigen::Vector3d err_p_Ii_GI = gt_t_Ii_GI - t_Ii_GI; 
    Eigen::Vector3d err_v_GIv = gt_GIv - imu_state.velocity;
    // r err in {GI}
    Eigen::Quaterniond q_err_r_Ii_GI = (q_Ii_GI.inverse()) * gt_q_Ii_GI;
    double roll_err_r_GI, pitch_err_r_GI, yaw_err_r_GI;
    tf::Matrix3x3(tf::Quaternion(q_err_r_Ii_GI.x(),q_err_r_Ii_GI.y(),q_err_r_Ii_GI.z(),q_err_r_Ii_GI.w())).getRPY(roll_err_r_GI, pitch_err_r_GI, yaw_err_r_GI, 1);
    Vector3d err_r_GI = Vector3d(roll_err_r_GI,pitch_err_r_GI,yaw_err_r_GI);
    double err_rx, err_ry, err_rz;
    // test
    err_rx = err_r_GI[2] * 180 / M_PI;    
    err_ry = err_r_GI[1] * 180 / M_PI;
    err_rz = err_r_GI[0] * 180 / M_PI;
    double err_bgx = 0;
    double err_bgy = 0;
    double err_bgz = 0; 
    double err_vx = err_v_GIv(0);
    double err_vy = err_v_GIv(1);
    double err_vz = err_v_GIv(2);
    double err_bax = 0;
    double err_bay = 0;
    double err_baz = 0; 
    double err_px = err_p_Ii_GI(0);
    double err_py = err_p_Ii_GI(1);
    double err_pz = err_p_Ii_GI(2);
    double std_rx = std::sqrt(state_server.state_cov(0, 0));
    double std_ry = std::sqrt(state_server.state_cov(1, 1));
    double std_rz = std::sqrt(state_server.state_cov(2, 2));
    // r std in {GI}
    Vector3d std_r_GI_vector = Vector3d(std_rx,std_ry,std_rz);
    Vector4d std_q_GI =
        smallAngleQuaternion(std_r_GI_vector);
    quaternionNormalize(std_q_GI);
    double roll_std_r_GI, pitch_std_r_GI, yaw_std_r_GI;
    tf::Matrix3x3(tf::Quaternion(std_q_GI.x(),std_q_GI.y(),std_q_GI.z(),std_q_GI.w())).getRPY(roll_std_r_GI, pitch_std_r_GI, yaw_std_r_GI, 1);
    Vector3d std_r_GI = Vector3d(roll_std_r_GI,pitch_std_r_GI,yaw_std_r_GI);
    // test
    std_rx = std_r_GI[2] * 180 / M_PI;
    std_ry = std_r_GI[1] * 180 / M_PI;
    std_rz = std_r_GI[0] * 180 / M_PI;
    double std_bgx = std::sqrt(state_server.state_cov(3, 3)) * 3600 * 180 / M_PI;;
    double std_bgy = std::sqrt(state_server.state_cov(4, 4)) * 3600 * 180 / M_PI;;
    double std_bgz = std::sqrt(state_server.state_cov(5, 5)) * 3600 * 180 / M_PI;;
    double std_vx = std::sqrt(state_server.state_cov(6, 6));
    double std_vy = std::sqrt(state_server.state_cov(7, 7));
    double std_vz = std::sqrt(state_server.state_cov(8, 8));
    double std_bax = std::sqrt(state_server.state_cov(9, 9));
    double std_bay = std::sqrt(state_server.state_cov(10, 10));
    double std_baz = std::sqrt(state_server.state_cov(11, 11));
    double std_px = std::sqrt(state_server.state_cov(12, 12));
    double std_py = std::sqrt(state_server.state_cov(13, 13));
    double std_pz = std::sqrt(state_server.state_cov(14, 14));
    std_rx = abs(std_rx);
    std_ry = abs(std_ry);
    std_rz = abs(std_rz);
    std_bgx = abs(std_bgx);
    std_bgy = abs(std_bgy);
    std_bgz = abs(std_bgz);
    std_vx = abs(std_vx);
    std_vy = abs(std_vy);
    std_vz = abs(std_vz);
    std_bax = abs(std_bax);
    std_bay = abs(std_bay);
    std_baz = abs(std_baz);
    std_px = abs(std_px);
    std_py = abs(std_py);
    std_pz = abs(std_pz);

    CSVDATA_RMSE csvdata_rmse;
    csvdata_rmse.time = dStamp;
    csvdata_rmse.Dtime = Dtime;
    csvdata_rmse.err_rx = err_rx;
    csvdata_rmse.err_ry = err_ry;
    csvdata_rmse.err_rz = err_rz;
    csvdata_rmse.err_px = err_px;
    csvdata_rmse.err_py = err_py;
    csvdata_rmse.err_pz = err_pz;
    csvdata_rmse.err_vx = err_vx;
    csvdata_rmse.err_vy = err_vy;
    csvdata_rmse.err_vz = err_vz;
    csvdata_rmse.err_bgx = err_bgx;
    csvdata_rmse.err_bgy = err_bgy;
    csvdata_rmse.err_bgz = err_bgz;
    csvdata_rmse.err_bax = err_bax;
    csvdata_rmse.err_bay = err_bay;
    csvdata_rmse.err_baz = err_baz;

    csvdata_rmse.std_rx = std_rx;
    csvdata_rmse.std_ry = std_ry;
    csvdata_rmse.std_rz = std_rz;
    csvdata_rmse.std_px = std_px;
    csvdata_rmse.std_py = std_py;
    csvdata_rmse.std_pz = std_pz;
    csvdata_rmse.std_vx = std_vx;
    csvdata_rmse.std_vy = std_vy;
    csvdata_rmse.std_vz = std_vz;
    csvdata_rmse.std_bgx = std_bgx;
    csvdata_rmse.std_bgy = std_bgy;
    csvdata_rmse.std_bgz = std_bgz;
    csvdata_rmse.std_bax = std_bax;
    csvdata_rmse.std_bay = std_bay;
    csvdata_rmse.std_baz = std_baz;
    csvData_rmse.push_back(csvdata_rmse);

///////////////////////////////////////////////////////////////

  return;
}

/**
 * @brief csv_timer_callBack 
 *  
 */ 
void AckermannMsckf::csv_timer_callBack(const ros::TimerEvent& event){
  
  if (!is_csv_curr_time_init) return;

  if(ros::Time::now().toSec() - csv_curr_time > 120){
    pose_file.close();
    odom_file.close();
    std_file.close();
    rmse_file.close();
    time_file.close();
    ros::shutdown();
  }

  if(csvData_odom.empty() && csvData_rmse.empty() && csvData_time.empty()){
    return;
  }

  std::string delim = ",";

  if(ros::Time::now().toSec() - csv_curr_time > 1){

    // save pose odom
    for (const auto& csvdata_odom : csvData_odom) {

      // save pose 
      // TUM #timestamp(sec) x y z q_x q_y q_z q_w
      pose_file.precision(16);
      pose_file << fixed << csvdata_odom.time << " " << csvdata_odom.pB(0) << " " << csvdata_odom.pB(1) << " " << csvdata_odom.pB(2) << " "
              << csvdata_odom.qB.x() << " " << csvdata_odom.qB.y() << " " << csvdata_odom.qB.z() << " " << csvdata_odom.qB.w() << endl;
      
      odom_file.precision(16);
      odom_file << fixed << csvdata_odom.time << delim << csvdata_odom.Dtime << delim;
      odom_file  << delim;
      odom_file << csvdata_odom.pB(0) << delim << csvdata_odom.pB(1) << delim << csvdata_odom.pB(2) << delim;
      odom_file << csvdata_odom.qB.x() << delim <<  csvdata_odom.qB.y() << delim << csvdata_odom.qB.z() << delim << csvdata_odom.qB.w() << delim;
      odom_file << csvdata_odom.roll << delim << csvdata_odom.pitch << delim << csvdata_odom.yaw << delim;
      odom_file << csvdata_odom.vB(0) << delim << csvdata_odom.vB(1) << delim << csvdata_odom.vB(2) << delim;
      odom_file << csvdata_odom.wB(0) << delim << csvdata_odom.wB(1) << delim << csvdata_odom.wB(2) << delim;
      odom_file  << delim;
      // gt
      odom_file << csvdata_odom.gt_pB(0) << delim << csvdata_odom.gt_pB(1) << delim << csvdata_odom.gt_pB(2) << delim;
      odom_file << csvdata_odom.gt_qB.x() << delim <<  csvdata_odom.gt_qB.y() << delim << csvdata_odom.gt_qB.z() << delim << csvdata_odom.gt_qB.w() << delim;
      odom_file << csvdata_odom.gt_roll << delim << csvdata_odom.gt_pitch << delim << csvdata_odom.gt_yaw << delim;
      odom_file << csvdata_odom.gt_vB(0) << delim << csvdata_odom.gt_vB(1) << delim << csvdata_odom.gt_vB(2) << delim;
      odom_file << csvdata_odom.gt_wB(0) << delim << csvdata_odom.gt_wB(1) << delim << csvdata_odom.gt_wB(2) << delim;
      odom_file  << delim;
      odom_file << csvdata_odom.Sr << delim << csvdata_odom.Sr_avg << delim;
      odom_file << std::endl;

    } 
    std::cout << "-------pose_file odom_file save done!!!---------" << std::endl;

    // save rmse
    for (const auto& csvdata_rmse : csvData_rmse) {

        double dStamp = csvdata_rmse.time;;
        double Dtime = csvdata_rmse.Dtime;

        double err_rx = csvdata_rmse.err_rx;
        double err_ry = csvdata_rmse.err_ry;
        double err_rz = csvdata_rmse.err_rz;
        double err_bgx = csvdata_rmse.err_bgx;
        double err_bgy = csvdata_rmse.err_bgy;
        double err_bgz = csvdata_rmse.err_bgz;
        double err_vx = csvdata_rmse.err_vx;
        double err_vy = csvdata_rmse.err_vy;
        double err_vz = csvdata_rmse.err_vz;
        double err_bax = csvdata_rmse.err_bax;
        double err_bay = csvdata_rmse.err_bay;
        double err_baz = csvdata_rmse.err_baz;
        double err_px = csvdata_rmse.err_px;
        double err_py = csvdata_rmse.err_py;
        double err_pz = csvdata_rmse.err_pz;
   
        double std_rx = csvdata_rmse.std_rx;
        double std_ry = csvdata_rmse.std_ry; 
        double std_rz = csvdata_rmse.std_rz;;
        double std_bgx = csvdata_rmse.std_bgx;
        double std_bgy = csvdata_rmse.std_bgy;
        double std_bgz = csvdata_rmse.std_bgz;
        double std_vx = csvdata_rmse.std_vx;
        double std_vy = csvdata_rmse.std_vy;
        double std_vz = csvdata_rmse.std_vz;
        double std_bax = csvdata_rmse.std_bax;
        double std_bay = csvdata_rmse.std_bay;
        double std_baz = csvdata_rmse.std_baz;
        double std_px = csvdata_rmse.std_px;
        double std_py = csvdata_rmse.std_py;
        double std_pz = csvdata_rmse.std_pz;

        std_file.precision(16);
        std_file << fixed << dStamp << delim << Dtime << delim;
        std_file  << delim;
        std_file  << err_rx << delim << err_ry << delim << err_rz << delim;
        std_file  << 3 * std_rx << delim << 3 * std_ry << delim << 3 * std_rz << delim;
        std_file  << -3 * std_rx << delim << -3 * std_ry << delim << -3 * std_rz << delim;
        std_file << delim;
        std_file  << err_px << delim << err_py << delim << err_pz << delim;
        std_file  << 3 * std_px << delim << 3 * std_py << delim << 3 * std_pz << delim;
        std_file  << -3 * std_px << delim << -3 * std_py << delim << -3 * std_pz << delim;
        std_file << delim;
        std_file << err_vx << delim << err_vy << delim << err_vz << delim;
        std_file << 3 * std_vx << delim << 3 * std_vy << delim << 3 * std_vz << delim;
        std_file << -3 * std_vx << delim << -3 * std_vy << delim << -3 * std_vz << delim;
        std_file  << delim;
        std_file  << err_bgx << delim << err_bgy << delim << err_bgz << delim;
        std_file  << 3 * std_bgx << delim << 3 * std_bgy << delim << 3 * std_bgz << delim;
        std_file  << -3 * std_bgx << delim << -3 * std_bgy << delim << -3 * std_bgz << delim;
        std_file << delim;
        std_file  << err_bax << delim << err_bay << delim << err_baz << delim;
        std_file  << 3 * std_bax << delim << 3 * std_bay << delim << 3 * std_baz << delim;
        std_file  << -3 * std_bax << delim << -3 * std_bay << delim << -3 * std_baz << delim;
        std_file << std::endl;

        // save rmse
        double nees_rx;
        double nees_ry;
        double nees_rz;
        double nees_px;
        double nees_py;
        double nees_pz;
        double nees_ribx;
        double nees_riby;
        double nees_ribz;
        double nees_pbix;
        double nees_pbiy;
        double nees_pbiz;
        double nees_bgx;
        double nees_bgy;
        double nees_bgz;
        double nees_bax;
        double nees_bay;
        double nees_baz;
        double nees_vx;
        double nees_vy;
        double nees_vz;
        if(is_first_nees){
          is_first_nees = false;
          nees_rx = 0;
          nees_ry = 0;
          nees_rz = 0;
          nees_px = 0;
          nees_py = 0;
          nees_pz = 0;
          nees_ribx = 0;
          nees_riby = 0;
          nees_ribz = 0;
          nees_pbix = 0;
          nees_pbiy = 0;
          nees_pbiz = 0;
          nees_bgx = 0;
          nees_bgy = 0;
          nees_bgz = 0;
          nees_bax = 0;
          nees_bay = 0;
          nees_baz = 0;
          nees_vx = 0;
          nees_vy = 0;
          nees_vz = 0;
        }else{
          nees_rx = err_rx * err_rx / (std_rx * std_rx);
          nees_ry = err_ry * err_ry / (std_ry * std_ry);
          nees_rz = err_rz * err_rz / (std_rz * std_rz);
          nees_px = err_px * err_px / (std_px * std_px);
          nees_py = err_py * err_py / (std_py * std_py);
          nees_pz = err_pz * err_pz / (std_pz * std_pz);
          nees_ribx = 0;
          nees_riby = 0;
          nees_ribz = 0;
          nees_pbix = 0;
          nees_pbiy = 0;
          nees_pbiz = 0;
          nees_bgx = err_bgx * err_bgx / (std_bgx * std_bgx);
          nees_bgy = err_bgy * err_bgy / (std_bgy * std_bgy);
          nees_bgz = err_bgz * err_bgz / (std_bgz * std_bgz);
          nees_bax = err_bax * err_bax / (std_bax * std_bax);
          nees_bay = err_bay * err_bay / (std_bay * std_bay);
          nees_baz = err_baz * err_baz / (std_baz * std_baz);
          nees_vx = err_vx * err_vx / (std_vx * std_vx);
          nees_vy = err_vy * err_vy / (std_vy * std_vy);
          nees_vz = err_vz * err_vz / (std_vz * std_vz);
        }
        rmse_file.precision(16);
        rmse_file << fixed << dStamp << delim << Dtime << delim;
        rmse_file  << delim;
        rmse_file << err_rx * err_rx << delim << err_ry * err_ry << delim << err_rz * err_rz << delim;
        rmse_file << err_rx * err_rx + err_ry * err_ry + err_rz * err_rz << delim;
        rmse_file << nees_rx << delim << nees_ry << delim << nees_rz << delim;
        rmse_file << delim;
        rmse_file << err_px * err_px << delim << err_py * err_py << delim << err_pz * err_pz << delim;
        rmse_file << err_px * err_px + err_py * err_py + err_pz * err_pz << delim;
        rmse_file << nees_px << delim << nees_py << delim << nees_pz << delim;
        rmse_file  << delim;
        rmse_file << 0 << delim << 0 << delim << 0 << delim;
        rmse_file << nees_ribx << delim << nees_riby << delim << nees_ribz << delim;
        rmse_file  << delim;
        rmse_file << 0 << delim << 0 << delim << 0 << delim;
        rmse_file << nees_pbix << delim << nees_pbiy << delim << nees_pbiz << delim;
        rmse_file  << delim;
        rmse_file << err_bgx * err_bgx << delim << err_bgy * err_bgy << delim << err_bgz * err_bgz << delim;
        rmse_file << nees_bgx << delim << nees_bgy << delim << nees_bgz << delim;
        rmse_file  << delim;
        rmse_file << err_bax * err_bax << delim << err_bay * err_bay << delim << err_baz * err_baz << delim;
        rmse_file << nees_bax << delim << nees_bay << delim << nees_baz << delim;
        rmse_file << delim;
        rmse_file << err_vx * err_vx << delim << err_vy * err_vy << delim << err_vz * err_vz << delim;
        rmse_file << nees_vx << delim << nees_vy << delim << nees_vz << delim;
        rmse_file  << std::endl;
    }
    std::cout << "-------rmse_file save done!!!---------" << std::endl;

    // save time
    for (const auto& csvdata_time : csvData_time) {

        double dStamp = csvdata_time.time;;
        double Dtime = csvdata_time.Dtime;
        double process_time = csvdata_time.process_time;
        double total_time = csvdata_time.total_time;
        double avg_time = csvdata_time.avg_time;

        time_file.precision(16);
        time_file << fixed << dStamp << delim << Dtime << delim;
        time_file << delim;
        time_file << process_time << delim << total_time << delim << avg_time << delim;
        time_file << std::endl;

    }
    std::cout << "-------time_file save done!!!---------" << std::endl;

    std::cout << std::endl;
    csvData_odom.clear();
    csvData_rmse.clear();
    csvData_time.clear();

  }

  return;

}

} // namespace ackermann_msckf


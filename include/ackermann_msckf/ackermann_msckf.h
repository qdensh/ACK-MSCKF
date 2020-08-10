/*
 * COPYRIGHT AND PERMISSION NOTICE
 * Penn Software MSCKF_VIO
 * Copyright (C) 2017 The Trustees of the University of Pennsylvania
 * All rights reserved.
 */

#ifndef MSCKF_VIO_H
#define MSCKF_VIO_H

#include <map>
#include <set>
#include <vector>
#include <queue>
#include <string>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <boost/shared_ptr.hpp>

#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/NavSatFix.h>
#include <geometry_msgs/TwistWithCovarianceStamped.h>
#include <geometry_msgs/Vector3Stamped.h>
#include <tf/transform_broadcaster.h>
#include <std_srvs/Trigger.h>

#include "imu_state.h"
#include "cam_state.h"
#include "feature.hpp"
#include <ackermann_msckf/CameraMeasurement.h>
#include <ackermann_msckf/AckermannDriveStamped.h>
#include "ackermann_msckf/integration_base.h"

namespace ackermann_msckf {
/*
 * @brief AckermannMsckf Implements the algorithm in
 *    Anatasios I. Mourikis, and Stergios I. Roumeliotis,
 *    "A Multi-State Constraint Kalman Filter for Vision-aided
 *    Inertial Navigation",
 *    http://www.ee.ucr.edu/~mourikis/tech_reports/TR_MSCKF.pdf
 */

class AckermannMsckf {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Constructor
    AckermannMsckf(ros::NodeHandle& pnh);
    // Disable copy and assign constructor
    AckermannMsckf(const AckermannMsckf&) = delete;
    AckermannMsckf operator=(const AckermannMsckf&) = delete;

    // Destructor
    ~AckermannMsckf() {}

    /*
     * @brief initialize Initialize the VIO.
     */
    bool initialize();

    /*
     * @brief reset Resets the VIO to initial status.
     */
    void reset();

    typedef boost::shared_ptr<AckermannMsckf> Ptr;
    typedef boost::shared_ptr<const AckermannMsckf> ConstPtr;

  private:
    /*
     * @brief StateServer Store one IMU states and several
     *    camera states for constructing measurement
     *    model.
     */
    struct StateServer {
      IMUState imu_state;
      CamStateServer cam_states;

      // State covariance matrix
      Eigen::MatrixXd state_cov;
      Eigen::Matrix<double, 12, 12> continuous_noise_cov;
    };

    /*
     * @brief loadParameters
     *    Load parameters from the parameter server.
     */
    bool loadParameters();

    /*
     * @brief createRosIO
     *    Create ros publisher and subscirbers.
     */
    bool createRosIO();

    /*
     * @brief imuCallback
     *    Callback function for the imu message.
     * @param msg IMU msg.
     */
    void imuCallback(const sensor_msgs::ImuConstPtr& msg);

   /*
     * @brief ackermannCallback
     *    Callback function for the ackermann message.
     * @param msg Ackermann msg.
     */
    void ackermannCallback(const ackermann_msckf::AckermannDriveStamped::ConstPtr& msg);

    /*
     * @brief featureCallback
     *    Callback function for feature measurements.
     * @param msg Stereo feature measurements.
     */
    void featureCallback(const CameraMeasurementConstPtr& msg);

    /*
     * @brief publish Publish the results of VIO.
     * @param time The time stamp of output msgs.
     */
    void publish(const ros::Time& time);

    /*
     * @brief initializegravityAndBias
     *    Initialize the IMU bias and initial orientation
     *    based on the first few IMU readings.
     */
    void initializeGravityAndBias();

    // Filter related functions
    // Propogate the state
    void batchImuProcessing(
        const double& time_bound);
    void processModel(const double& time,
        const Eigen::Vector3d& m_gyro,
        const Eigen::Vector3d& m_acc);
    void predictNewState(const double& dt,
        const Eigen::Vector3d& gyro,
        const Eigen::Vector3d& acc);

    // Measurement update
    void stateAugmentation(const double& time);
    void addFeatureObservations(const CameraMeasurementConstPtr& msg);
    // This function is used to compute the measurement Jacobian
    // for a single feature observed at a single camera frame.
    void measurementJacobian(const StateIDType& cam_state_id,
        const FeatureIDType& feature_id,
        Eigen::Matrix<double, 4, 6>& H_x,
        Eigen::Matrix<double, 4, 3>& H_f,
        Eigen::Vector4d& r);
    // This function computes the Jacobian of all measurements viewed
    // in the given camera states of this feature.
    void featureJacobian(const FeatureIDType& feature_id,
        const std::vector<StateIDType>& cam_state_ids,
        Eigen::MatrixXd& H_x, Eigen::VectorXd& r);
    void measurementUpdate(const Eigen::MatrixXd& H,
        const Eigen::VectorXd& r);
    bool gatingTest(const Eigen::MatrixXd& H,
        const Eigen::VectorXd&r, const int& dof);
    void removeLostFeatures();
    void findRedundantCamStates(
        std::vector<StateIDType>& rm_cam_state_ids);
    void pruneCamStateBuffer();
    // Reset the system online if the uncertainty is too large.
    void onlineReset();

    // Chi squared test table.
    static std::map<int, double> chi_squared_test_table;

    // State vector
    StateServer state_server;
    // Maximum number of camera states
    int max_cam_state_size;

    // Features used
    MapServer map_server;

    // IMU data buffer
    // This is buffer is used to handle the unsynchronization or
    // transfer delay between IMU and Image messages.
    std::vector<sensor_msgs::Imu> imu_msg_buffer;

    // Indicate if the gravity vector is set.
    bool is_gravity_set;

    // Indicate if the received image is the first one. The
    // system will start after receiving the first image.
    bool is_first_img;

    // The position uncertainty threshold is used to determine
    // when to reset the system online. Otherwise, the ever-
    // increaseing uncertainty will make the estimation unstable.
    // Note this online reset will be some dead-reckoning.
    // Set this threshold to nonpositive to disable online reset.
    double position_std_threshold;

    // Tracking rate
    double tracking_rate;

    // Threshold for determine keyframes
    double translation_threshold;
    double rotation_threshold;
    double tracking_rate_threshold;

    // Ros node handle
    ros::NodeHandle nh;

    // Subscribers and publishers
    ros::Subscriber imu_sub;
    ros::Subscriber feature_sub;
    ros::Publisher odom_pub;
    ros::Publisher feature_pub;
    tf::TransformBroadcaster tf_pub;
    ros::ServiceServer reset_srv;
    ros::Subscriber ackermann_sub;

    // Frame id
    std::string fixed_frame_id;
    std::string child_frame_id;

    // Whether to publish tf or not.
    bool publish_tf;

    // Framte rate of the stereo images. This variable is
    // only used to determine the timing threshold of
    // each iteration of the filter.
    double frame_rate;

    ros::Subscriber mocap_odom_sub;
    ros::Publisher mocap_odom_pub;
    geometry_msgs::TransformStamped raw_mocap_odom_msg;
    Eigen::Isometry3d mocap_initial_frame;

    void AckermannmeasurementUpdate(const Eigen::MatrixXd& H,const Eigen::VectorXd&r,const Eigen::MatrixXd &noise);
     // Ackermann data buffer
    // This is buffer is used to handle the unsynchronization or
    // transfer delay between Ackermann and Image messages.
    std::vector<ackermann_msckf::AckermannDriveStamped> ackermann_msg_buffer;

    nav_msgs::Odometry odom_fix;

    Eigen::Quaterniond q_G_I0_;
    Eigen::Quaterniond q_I0_G_;

    Eigen::Matrix3d   R_body_cam0;
    Eigen::Vector3d   t_body_cam0;
    Eigen::Isometry3d T_body_cam0;

    void ackermannProcess();
    void ackermannIntegrate(auto& begin_iter, auto& end_iter);

    Eigen::MatrixXd estimateErrorCovariance_;
    Eigen::MatrixXd transferFunctionJacobian_;
    Eigen::MatrixXd processFunctionJacobian_;
    Eigen::MatrixXd processNoiseCovariance_ackerman_;
    int ackermann_update_count = 0;

    bool first_ack = true;
    double current_integrate_time = -1;
    void preIntegrationAck(double dt, const double speed_x, const double steering_angle);
    long long int odom_frame_count = 0;
    IntegrationBase *tmp_pre_integration;
    double speed_x_0;
    double steering_angle_0;
    double ackermann_wheel_base;
    double ackermann_tire_base;
    double ackermann_speed_x_noise;
    double ackermann_speed_y_noise;
    double ackermann_speed_z_noise;
    double ackermann_steering_noise;
    double ackermann_steer_ratio;
    bool first_init = true;
    bool debug_flag = false;

    double delta_cam_state_time_ack;

    double last_dt_1 = 0;
    double last_dt_1_ack = 0;

    Eigen::Vector3d delta_p_ = Eigen::Vector3d::Zero();
    Eigen::Quaterniond delta_q_ = Eigen::Quaterniond::Identity();

    double ackermann_heading_white_noise;
    double ackermann_x_white_noise;
    double ackermann_y_white_noise;
    int ackermann_rate;

    double speed_avg = 0;
    double speed_cntr = 0;

    // debug
    bool use_a27_platform;
    bool use_svd_ex;
    bool use_debug;
    bool use_ackermann;
    bool ackermann_update_v = true;
    bool ackermann_update_q = true;
    bool ackermann_update_v_ci = false;
    bool ackermann_update_q_ci = false;
    bool use_offline_bias = false;

    // gt
    ros::Subscriber gt_init_sub;
    void gtInitCallback(const nav_msgs::OdometryConstPtr& msg);
    Eigen::Isometry3d T_B0_GS_gt = Eigen::Isometry3d::Identity();
    bool is_gt_init_set = false;
    std::vector<nav_msgs::Odometry> gt_msg_buffer;
    nav_msgs::Odometry gt_odom_curr;
    nav_msgs::Odometry gt_odom_last;
    Eigen::Vector3d t_GBi_GB_last;
    void batchGtProcessing(
        const double& time_bound);
    bool is_first_sr = true;
    bool is_first_nees = true;
    unsigned long long int global_count = 0;
    double global_scale_ratio = 0;
    bool use_oc_vins;

    // save file
    std::string output_path;
    std::ofstream pose_file;
    std::ofstream odom_file;
    std::ofstream std_file;
    std::ofstream rmse_file;

    double DfirstTime;
    double Dtime;

    // ODOM
    struct CSVDATA_ODOM {
        double time;
        double Dtime;
        Eigen::Vector3d pB;
        Eigen::Quaterniond qB;
        double roll, pitch, yaw;
        Eigen::Vector3d vB;
        Eigen::Vector3d wB;

        Eigen::Vector3d gt_pB;
        Eigen::Quaterniond gt_qB;
        double gt_roll, gt_pitch, gt_yaw;
        Eigen::Vector3d gt_vB;
        Eigen::Vector3d gt_wB;

        double Sr;
        double Sr_avg;

        Eigen::Vector3d pIBs;
        Eigen::Quaterniond qIBs;
        double roll_ibx, pitch_iby, yaw_ibz;
        Eigen::Vector3d b_v_la;

        Eigen::Matrix<double, 5, 1> k_b;
    };
    std::vector<struct CSVDATA_ODOM> csvData_odom;

    ros::Timer csv_timer;
    void csv_timer_callBack(const ros::TimerEvent& event);
    double csv_curr_time = 0;

    // RMSE
    struct CSVDATA_RMSE {
        double time;
        double Dtime;

        double err_rx, err_ry, err_rz;
        double err_px;
        double err_py;
        double err_pz;
        double err_vx;
        double err_vy;
        double err_vz;
        double err_ribx;
        double err_riby;
        double err_ribz;
        double err_pbix;
        double err_pbiy;
        double err_pbiz;
        double err_bgx;
        double err_bgy;
        double err_bgz;
        double err_bax;
        double err_bay;
        double err_baz;
        
        double std_rx, std_ry, std_rz;
        double std_px;
        double std_py;
        double std_pz;
        double std_vx;
        double std_vy;
        double std_vz;
        double std_ribx;
        double std_riby;
        double std_ribz;
        double std_pbix;
        double std_pbiy;
        double std_pbiz;
        double std_bgx;
        double std_bgy;
        double std_bgz;
        double std_bax;
        double std_bay;
        double std_baz;
        
    };
    std::vector<struct CSVDATA_RMSE> csvData_rmse;
    
    std::ofstream time_file;
    // TIME
    struct CSVDATA_TIME {
        double time;
        double Dtime;

        double process_time, avg_time, total_time;
    };
    std::vector<struct CSVDATA_TIME> csvData_time;
    double total_time = 0;
    bool is_csv_curr_time_init = false;

};

typedef AckermannMsckf::Ptr AckermannMsckfPtr;
typedef AckermannMsckf::ConstPtr AckermannMsckfConstPtr;

} // namespace ackermann_msckf

#endif

#pragma once

#include<iostream>

#include <ceres/ceres.h>
using namespace Eigen;
using namespace std;

class IntegrationBase
{
  public:
    IntegrationBase() = delete;

    //ack
    IntegrationBase(const double speed_x, const double steering_angle,
        const double ackermann_wheel_base, const double ackermann_tire_base,
        const double ackermann_speed_noise, const double ackermann_steering_noise, 
        const double ackermann_steer_ratio, double ackermann_heading_white_noise, 
        double ackermann_x_white_noise, double ackermann_y_white_noise, int ackermann_rate):
        estimateErrorCovariance_(3,3),
        transferFunctionJacobian_(3,3),
        processFunctionJacobian_(3,2)
    {
        // cout << "IntegrationBase begin: " << endl;
        speed_x_0 = speed_x;
        steering_angle_0 = steering_angle;

        ackermann_wheel_base_ = ackermann_wheel_base;
        ackermann_tire_base_ = ackermann_tire_base;
        ackermann_speed_noise_ = ackermann_speed_noise;
        ackermann_steering_noise_ = ackermann_steering_noise;
        ackermann_steer_ratio_ = ackermann_steer_ratio;
        ackermann_heading_white_noise_ = ackermann_heading_white_noise;
        ackermann_x_white_noise_ = ackermann_x_white_noise;
        ackermann_y_white_noise_ = ackermann_y_white_noise;
        ackermann_rate_ = ackermann_rate;

        delta_q = Eigen::Quaterniond::Identity();
        delta_p = Eigen::Vector3d::Zero();

        ackermann_x_ = 0;
        ackermann_y_ = 0;
        ackermann_heading_ = 0;
        // Clear the Ackermann Jacobian and noise
        transferFunctionJacobian_.setZero();
        processFunctionJacobian_.setZero();
        estimateErrorCovariance_.setIdentity();
        estimateErrorCovariance_ *= 1e-9;
        processNoiseCovariance_ackerman_ = Eigen::Matrix<double, 2, 2>::Zero();
        processNoiseCovariance_ackerman_(0,0) =  pow(ackermann_speed_noise_ , 2); 
        processNoiseCovariance_ackerman_(1,1) =  pow(ackermann_steering_noise_ * M_PI /180 , 2); 
        processWhiteNoiseCovariance_ = Eigen::Matrix<double, 3, 3>::Zero();
        processWhiteNoiseCovariance_(0,0) = pow(ackermann_x_white_noise_ , 2); 
        processWhiteNoiseCovariance_(1,1) = pow(ackermann_y_white_noise_ , 2); 
        processWhiteNoiseCovariance_(2,2) = pow(ackermann_heading_white_noise_ * M_PI /180 , 2);
       
        
        if(steering_angle_0 > 0){
            steer_radius_0 = (ackermann_wheel_base_/tan(steering_angle_0) - ackermann_tire_base_/2);
        }
        else{
            steer_radius_0 = (ackermann_wheel_base_/tan(steering_angle_0) + ackermann_tire_base_/2);
        }
        speed_x_0 = speed_x_0;
        ackmann_gyro_0 = speed_x_0 / steer_radius_0;
        steering_angle_0 = steering_angle_0;
    }




    void push_back_ack(double dt, const double speed_x, const double steering_angle)
    {
         if(steering_angle > 0){
            steer_radius_1 = (ackermann_wheel_base_/tan(steering_angle) - ackermann_tire_base_/2);
        }
        else{
            steer_radius_1 = (ackermann_wheel_base_/tan(steering_angle) + ackermann_tire_base_/2);
        }
        speed_x_1 = speed_x;
        ackmann_gyro_1 = speed_x_1 / steer_radius_1;
        steering_angle_1 = steering_angle;
        propagate_ack(dt);
    }


    // propagate_ack
    void propagate_ack(double _dt)
    {
        dt = _dt;
        Quaterniond result_delta_q;
        Eigen::Vector3d result_delta_p;

        midPointIntegration_ack(_dt, delta_q, result_delta_q, delta_p, result_delta_p);

        delta_q = result_delta_q;
        delta_q.normalize();
        delta_p = result_delta_p;

        speed_x_0 = speed_x_1;
        steering_angle_0 = steering_angle_1;
        ackmann_gyro_0 = ackmann_gyro_1;
        steer_radius_0 = steer_radius_1;  
    }



    void midPointIntegration_ack(double _dt, 
                            const Eigen::Quaterniond &delta_q,
                            Eigen::Quaterniond &result_delta_q,
                            const Eigen::Vector3d &delta_p,
                            Eigen::Vector3d &result_delta_p)
    {
        //ROS_INFO("midpoint integration");
        double un_gyr = 0.5 * (ackmann_gyro_0 + ackmann_gyro_1);
        double un_v =  0.5 * (speed_x_0 + speed_x_1);
        Eigen::Vector3d un_v_ = {speed_x_1,0,0};

        result_delta_q = delta_q * Quaterniond(1, 0, 0 , un_gyr * _dt / 2);
        result_delta_q.normalize();

        result_delta_p = delta_p + result_delta_q * un_v_ * _dt;

        double delta = _dt;
        const double linear = 0.5 * (speed_x_0 + speed_x_1) * delta;
        double r_mul_tan = 0;
        if(steering_angle_0 > 0){
            r_mul_tan = ackermann_wheel_base_ - tan(steering_angle_0) * ackermann_wheel_base_/2;
        }
        else{
            r_mul_tan = ackermann_wheel_base_ + tan(steering_angle_0) * ackermann_wheel_base_/2;
        }

         // Handle wrapping / Normalize
        if(ackermann_heading_ >= M_PI) ackermann_heading_ -= 2.0 * M_PI;
        if(ackermann_heading_ <= (-M_PI)) ackermann_heading_ += 2.0 * M_PI;
        if (delta < 0.0001)
            return; // Interval too small to integrate with    
            
        
        double pow_r_mul_tan = pow(r_mul_tan,2);
        // dr = ds/R
        const double deta_yaw = linear / steer_radius_0;

        double cos_y = cos(ackermann_heading_ + deta_yaw/2 );
        double sin_y = sin(ackermann_heading_ + deta_yaw/2 );                
        double pow_d = pow(linear,2);
        double pow_r = pow(steer_radius_0,2);
        double pow_tan = pow(tan(steering_angle_0),2);
        double dFx_dY = -linear * sin(ackermann_heading_ + deta_yaw/2 );
        double dFy_dY = linear * cos(ackermann_heading_ + deta_yaw/2 );

        // Much of the transfer function Jacobian is identical to the transfer function
        // transferFunctionJacobian_ = transferFunction_;
        transferFunctionJacobian_(0, 0) = 1;
        transferFunctionJacobian_(0, 2) = dFx_dY;
        transferFunctionJacobian_(1, 1) = 1;
        transferFunctionJacobian_(1, 2) = dFy_dY;
        transferFunctionJacobian_(2, 2) = 1;

        // // std::cout << "predict:use_ackermann_:transferFunctionJacobian_ " << std::endl;
        processFunctionJacobian_(0,0) = -deta_yaw * delta  * sin_y / 2 + delta * cos_y;
        processFunctionJacobian_(0,1) = pow_d * ackermann_wheel_base_ * (pow_tan + 1) * sin_y / ( ( 2 * ackermann_steer_ratio_ * pow_r_mul_tan ) );
        processFunctionJacobian_(1,0) = deta_yaw * delta * cos_y / 2 + delta * sin_y;
        processFunctionJacobian_(1,1) = -pow_d * ackermann_wheel_base_ * (pow_tan + 1) * sin_y / ( ( 2 * ackermann_steer_ratio_ * pow_r_mul_tan ) );
        processFunctionJacobian_(2,0) = delta / steer_radius_0;
        processFunctionJacobian_(2,1) = linear * ackermann_wheel_base_ * (pow_tan + 1)/( ( ackermann_steer_ratio_ * pow_r_mul_tan ) );

        // (3) Project the error forward: P = J * P * J' + V * Q * V'
        estimateErrorCovariance_ = (transferFunctionJacobian_ *
                                    estimateErrorCovariance_ *
                                    transferFunctionJacobian_.transpose());
        estimateErrorCovariance_.noalias() += (processFunctionJacobian_ * processNoiseCovariance_ackerman_ * processFunctionJacobian_.transpose()) + processWhiteNoiseCovariance_;

       
        /// Integrate odometry: RungeKutta2
        const double direction = ackermann_heading_ + deta_yaw * 0.5;
        /// Runge-Kutta 2nd order integration:
        ackermann_x_       += linear * cos(direction);
        ackermann_y_       += linear * sin(direction);
        ackermann_heading_ += deta_yaw;
        // Handle wrapping / Normalize
        if(ackermann_heading_ >= M_PI)ackermann_heading_ -= 2.0 * M_PI;
        if(ackermann_heading_ <= (-M_PI))ackermann_heading_ += 2.0 * M_PI;

    
    }


    // imu
    IntegrationBase(const Eigen::Vector3d &_acc_0, const Eigen::Vector3d &_gyr_0,
                    const Eigen::Vector3d &_linearized_ba, const Eigen::Vector3d &_linearized_bg)
        : acc_0{_acc_0}, gyr_0{_gyr_0}, linearized_acc{_acc_0}, linearized_gyr{_gyr_0},
          linearized_ba{_linearized_ba}, linearized_bg{_linearized_bg},
            jacobian{Eigen::Matrix<double, 15, 15>::Identity()}, covariance{Eigen::Matrix<double, 15, 15>::Zero()},
          sum_dt{0.0}, delta_p{Eigen::Vector3d::Zero()}, delta_q{Eigen::Quaterniond::Identity()}, delta_v{Eigen::Vector3d::Zero()}

    {
        noise = Eigen::Matrix<double, 18, 18>::Zero();
        // noise.block<3, 3>(0, 0) =  (ACC_N * ACC_N) * Eigen::Matrix3d::Identity();
        // noise.block<3, 3>(3, 3) =  (GYR_N * GYR_N) * Eigen::Matrix3d::Identity();
        // noise.block<3, 3>(6, 6) =  (ACC_N * ACC_N) * Eigen::Matrix3d::Identity();
        // noise.block<3, 3>(9, 9) =  (GYR_N * GYR_N) * Eigen::Matrix3d::Identity();
        // noise.block<3, 3>(12, 12) =  (ACC_W * ACC_W) * Eigen::Matrix3d::Identity();
        // noise.block<3, 3>(15, 15) =  (GYR_W * GYR_W) * Eigen::Matrix3d::Identity();
    }





    void push_back(double dt, const Eigen::Vector3d &acc, const Eigen::Vector3d &gyr)
    {
        propagate(dt, acc, gyr);
    }



    void midPointIntegration(double _dt, 
                            const Eigen::Vector3d &_acc_0, const Eigen::Vector3d &_gyr_0,
                            const Eigen::Vector3d &_acc_1, const Eigen::Vector3d &_gyr_1,
                            const Eigen::Vector3d &delta_p, const Eigen::Quaterniond &delta_q, const Eigen::Vector3d &delta_v,
                            const Eigen::Vector3d &linearized_ba, const Eigen::Vector3d &linearized_bg,
                            Eigen::Vector3d &result_delta_p, Eigen::Quaterniond &result_delta_q, Eigen::Vector3d &result_delta_v,
                            Eigen::Vector3d &result_linearized_ba, Eigen::Vector3d &result_linearized_bg, bool update_jacobian)
    {
        //ROS_INFO("midpoint integration");
        Vector3d un_acc_0 = delta_q * (_acc_0 - linearized_ba);
        Vector3d un_gyr = 0.5 * (_gyr_0 + _gyr_1) - linearized_bg;
        result_delta_q = delta_q * Quaterniond(1, un_gyr(0) * _dt / 2, un_gyr(1) * _dt / 2, un_gyr(2) * _dt / 2);
        result_delta_q.normalize();
        Vector3d un_acc_1 = result_delta_q * (_acc_1 - linearized_ba);
        Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        result_delta_p = delta_p + delta_v * _dt + 0.5 * un_acc * _dt * _dt;
        result_delta_v = delta_v + un_acc * _dt;
        result_linearized_ba = linearized_ba;
        result_linearized_bg = linearized_bg;         
    }





    void propagate(double _dt, const Eigen::Vector3d &_acc_1, const Eigen::Vector3d &_gyr_1)
    {
        dt = _dt;
        acc_1 = _acc_1;
        gyr_1 = _gyr_1;
        Vector3d result_delta_p;
        Quaterniond result_delta_q;
        Vector3d result_delta_v;
        Vector3d result_linearized_ba;
        Vector3d result_linearized_bg;

        midPointIntegration(_dt, acc_0, gyr_0, _acc_1, _gyr_1, delta_p, delta_q, delta_v,
                            linearized_ba, linearized_bg,
                            result_delta_p, result_delta_q, result_delta_v,
                            result_linearized_ba, result_linearized_bg, 1);

        //checkJacobian(_dt, acc_0, gyr_0, acc_1, gyr_1, delta_p, delta_q, delta_v,
        //                    linearized_ba, linearized_bg);
        delta_p = result_delta_p;
        delta_q = result_delta_q;
        delta_v = result_delta_v;
        linearized_ba = result_linearized_ba;
        linearized_bg = result_linearized_bg;
        delta_q.normalize();
        sum_dt += dt;
        acc_0 = acc_1;
        gyr_0 = gyr_1;  
     
    }


    double dt;
    Eigen::Vector3d acc_0, gyr_0;
    Eigen::Vector3d acc_1, gyr_1;

    const Eigen::Vector3d linearized_acc, linearized_gyr;
    Eigen::Vector3d linearized_ba, linearized_bg;

    Eigen::Matrix<double, 15, 15> jacobian, covariance;
    Eigen::Matrix<double, 15, 15> step_jacobian;
    Eigen::Matrix<double, 15, 18> step_V;
    Eigen::Matrix<double, 18, 18> noise;

    double sum_dt;
    Eigen::Vector3d delta_p;
    Eigen::Quaterniond delta_q;
    Eigen::Vector3d delta_v;

    std::vector<double> dt_buf;
    std::vector<Eigen::Vector3d> acc_buf;
    std::vector<Eigen::Vector3d> gyr_buf;



    // ack
    double speed_x_0;
    double steering_angle_0;
    double ackmann_gyro_0;
    double steer_radius_0;
    double speed_x_1;
    double steering_angle_1;
    double ackmann_gyro_1;
    double steer_radius_1;

    double ackermann_wheel_base_;
    double ackermann_tire_base_;
    double ackermann_speed_noise_;
    double ackermann_steering_noise_;
    double ackermann_steer_ratio_;

    double ackermann_heading_white_noise_;
    double ackermann_x_white_noise_;
    double ackermann_y_white_noise_;
    int ackermann_rate_;


    Eigen::Matrix<double, 1, 1> estimateErrorCovariance_q_;
    double estimateErrorCovariance_p_;

    Eigen::Matrix<double, 2, 2> processNoiseCovariance_ackerman_;

    Eigen::Matrix<double, 1, 1> processWhiteNoiseCovariance_ackerman_q;
    double processWhiteNoiseCovariance_ackerman_p;


    Eigen::MatrixXd estimateErrorCovariance_;
    Eigen::MatrixXd transferFunctionJacobian_;

    Eigen::MatrixXd processFunctionJacobian_;

    Eigen::MatrixXd processWhiteNoiseCovariance_;

    double ackermann_heading_;
    double ackermann_x_;
    double ackermann_y_; 


};

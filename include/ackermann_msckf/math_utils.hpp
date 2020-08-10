/*
 * COPYRIGHT AND PERMISSION NOTICE
 * Penn Software MSCKF_VIO
 * Copyright (C) 2017 The Trustees of the University of Pennsylvania
 * All rights reserved.
 */

#ifndef MSCKF_VIO_MATH_UTILS_HPP
#define MSCKF_VIO_MATH_UTILS_HPP

#include <cmath>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

namespace ackermann_msckf {

/*
 *  @brief Create a skew-symmetric matrix from a 3-element vector.
 *  @note Performs the operation:
 *  w   ->  [  0 -w3  w2]
 *          [ w3   0 -w1]
 *          [-w2  w1   0]
 */
inline Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d& w) {
  Eigen::Matrix3d w_hat;
  w_hat(0, 0) = 0;
  w_hat(0, 1) = -w(2);
  w_hat(0, 2) = w(1);
  w_hat(1, 0) = w(2);
  w_hat(1, 1) = 0;
  w_hat(1, 2) = -w(0);
  w_hat(2, 0) = -w(1);
  w_hat(2, 1) = w(0);
  w_hat(2, 2) = 0;
  return w_hat;
}

/*
 * @brief Normalize the given quaternion to unit quaternion.
 */
inline void quaternionNormalize(Eigen::Vector4d& q) {
  double norm = q.norm();
  q = q / norm;
  return;
}

/*
 * @brief Perform q1 * q2
 */
inline Eigen::Vector4d quaternionMultiplication(
    const Eigen::Vector4d& q1,
    const Eigen::Vector4d& q2) {
  Eigen::Matrix4d L;
  L(0, 0) =  q1(3); L(0, 1) =  q1(2); L(0, 2) = -q1(1); L(0, 3) =  q1(0);
  L(1, 0) = -q1(2); L(1, 1) =  q1(3); L(1, 2) =  q1(0); L(1, 3) =  q1(1);
  L(2, 0) =  q1(1); L(2, 1) = -q1(0); L(2, 2) =  q1(3); L(2, 3) =  q1(2);
  L(3, 0) = -q1(0); L(3, 1) = -q1(1); L(3, 2) = -q1(2); L(3, 3) =  q1(3);

  Eigen::Vector4d q = L * q2;
  quaternionNormalize(q);
  return q;
}

/*
 * @brief Convert the vector part of a quaternion to a
 *    full quaternion.
 * @note This function is useful to convert delta quaternion
 *    which is usually a 3x1 vector to a full quaternion.
 *    For more details, check Section 3.2 "Kalman Filter Update" in
 *    "Indirect Kalman Filter for 3D Attitude Estimation:
 *    A Tutorial for quaternion Algebra".
 */
inline Eigen::Vector4d smallAngleQuaternion(
    const Eigen::Vector3d& dtheta) {

  Eigen::Vector3d dq = dtheta / 2.0;
  Eigen::Vector4d q;
  double dq_square_norm = dq.squaredNorm();

  if (dq_square_norm <= 1) {
    q.head<3>() = dq;
    q(3) = std::sqrt(1-dq_square_norm);
  } else {
    q.head<3>() = dq;
    q(3) = 1;
    q = q / std::sqrt(1+dq_square_norm);
  }

  return q;
}

/*
 * @brief Convert a quaternion to the corresponding rotation matrix
 * @note Pay attention to the convention used. The function follows the
 *    conversion in "Indirect Kalman Filter for 3D Attitude Estimation:
 *    A Tutorial for Quaternion Algebra", Equation (78).
 *
 *    The input quaternion should be in the form
 *      [q1, q2, q3, q4(scalar)]^T
 */
inline Eigen::Matrix3d quaternionToRotation(
    const Eigen::Vector4d& q) {
  const Eigen::Vector3d& q_vec = q.block(0, 0, 3, 1);
  const double& q4 = q(3);
  Eigen::Matrix3d R =
    (2*q4*q4-1)*Eigen::Matrix3d::Identity() -
    2*q4*skewSymmetric(q_vec) +
    2*q_vec*q_vec.transpose();
  //TODO: Is it necessary to use the approximation equation
  //    (Equation (87)) when the rotation angle is small?
  return R;
}

/*
 * @brief Convert a rotation matrix to a quaternion.
 * @note Pay attention to the convention used. The function follows the
 *    conversion in "Indirect Kalman Filter for 3D Attitude Estimation:
 *    A Tutorial for Quaternion Algebra", Equation (78).
 *
 *    The input quaternion should be in the form
 *      [q1, q2, q3, q4(scalar)]^T
 */
inline Eigen::Vector4d rotationToQuaternion(
    const Eigen::Matrix3d& R) {
  Eigen::Vector4d score;
  score(0) = R(0, 0);
  score(1) = R(1, 1);
  score(2) = R(2, 2);
  score(3) = R.trace();

  int max_row = 0, max_col = 0;
  score.maxCoeff(&max_row, &max_col);

  Eigen::Vector4d q = Eigen::Vector4d::Zero();
  if (max_row == 0) {
    q(0) = std::sqrt(1+2*R(0, 0)-R.trace()) / 2.0;
    q(1) = (R(0, 1)+R(1, 0)) / (4*q(0));
    q(2) = (R(0, 2)+R(2, 0)) / (4*q(0));
    q(3) = (R(1, 2)-R(2, 1)) / (4*q(0));
  } else if (max_row == 1) {
    q(1) = std::sqrt(1+2*R(1, 1)-R.trace()) / 2.0;
    q(0) = (R(0, 1)+R(1, 0)) / (4*q(1));
    q(2) = (R(1, 2)+R(2, 1)) / (4*q(1));
    q(3) = (R(2, 0)-R(0, 2)) / (4*q(1));
  } else if (max_row == 2) {
    q(2) = std::sqrt(1+2*R(2, 2)-R.trace()) / 2.0;
    q(0) = (R(0, 2)+R(2, 0)) / (4*q(2));
    q(1) = (R(1, 2)+R(2, 1)) / (4*q(2));
    q(3) = (R(0, 1)-R(1, 0)) / (4*q(2));
  } else {
    q(3) = std::sqrt(1+R.trace()) / 2.0;
    q(0) = (R(1, 2)-R(2, 1)) / (4*q(3));
    q(1) = (R(2, 0)-R(0, 2)) / (4*q(3));
    q(2) = (R(0, 1)-R(1, 0)) / (4*q(3));
  }

  if (q(3) < 0) q = -q;
  quaternionNormalize(q);
  return q;
}

///////////////////////////////////////////////////////////////////////////
// CVMat <-> cVMatx
inline cv::Mat CvMat2CvMatInverse(const cv::Mat &Tcw)
{
    cv::Mat Rcw = Tcw.rowRange(0,3).colRange(0,3);
    cv::Mat tcw = Tcw.rowRange(0,3).col(3);
    cv::Mat Rwc = Rcw.t();
    cv::Mat twc = -Rwc*tcw;

    cv::Mat Twc = cv::Mat::eye(4,4,Tcw.type());
    Rwc.copyTo(Twc.rowRange(0,3).colRange(0,3));
    twc.copyTo(Twc.rowRange(0,3).col(3));

    return Twc.clone();
}

inline cv::Matx44d cvMatx44d2MatInverse(const cv::Matx44d& M)
{
    cv::Matx33d R = M.get_minor<3, 3>(0, 0);
    R = R.t();
    cv::Vec3d t(M(0, 3), M(1, 3), M(2, 3));
    t = -R * t;
    cv::Matx44d out(
        R(0, 0), R(0, 1), R(0, 2), t(0),
        R(1, 0), R(1, 1), R(1, 2), t(1),
        R(2, 0), R(2, 1), R(2, 2), t(2),
        0.0, 0.0, 0.0, 1.0);

    return out;
}

inline cv::Mat CvMatx44d2CvMat(const cv::Matx44d& matx44d)
{
    cv::Mat out = cv::Mat::zeros(4, 4, CV_64FC1);
    for (int c = 0; c < 4; ++c)
        for (int r = 0; r < 4; ++r)
            out.ptr<double>(r)[c] = matx44d(r, c);
    return out;
}

inline cv::Mat CvMatx33d2CvMat(const cv::Matx33d& matx33d)
{
    cv::Mat out = cv::Mat::zeros(3, 3, CV_64FC1);
    for (int c = 0; c < 3; ++c)
        for (int r = 0; r < 3; ++r)
            out.ptr<double>(r)[c] = matx33d(r, c);
    return out;
}

///////////////////////////////////////////////////////////////////////////
// Eigen -> CV 
// Eigen -> CvMat
inline cv::Mat Vector3d2CvMat(const Eigen::Matrix<double,3,1> &m)
{
      cv::Mat cvMat(3,1,CV_32F);
      for(int i=0;i<3;i++)
        cvMat.at<float>(i)=m(i);
      return cvMat.clone();
}

inline cv::Mat Matrix3d2CvMat(const Eigen::Matrix3d &m)
{
    cv::Mat cvMat(3,3,CV_32F);
    for(int i=0;i<3;i++)
    for(int j=0; j<3; j++)
    cvMat.at<float>(i,j)=m(i,j);

    return cvMat.clone();
}

inline cv::Mat Matrix4d2CvMat(const Eigen::Matrix<double,4,4> &m)
{
    cv::Mat cvMat(4,4,CV_32F);
    for(int i=0;i<4;i++)
    for(int j=0; j<4; j++)
        cvMat.at<float>(i,j)=m(i,j);

    return cvMat.clone();
}

// Eigen -> cvMatx
inline cv::Matx<double, 4, 4> ogv2ocv(const Eigen::Matrix<double, 3, 4>& ogv_mat)
{
    cv::Matx34d ocv_mat;
    cv::eigen2cv(ogv_mat, ocv_mat);

    return cv::Matx<double, 4, 4>(
        ocv_mat(0, 0), ocv_mat(0, 1), ocv_mat(0, 2), ocv_mat(0, 3),
        ocv_mat(1, 0), ocv_mat(1, 1), ocv_mat(1, 2), ocv_mat(1, 3),
        ocv_mat(2, 0), ocv_mat(2, 1), ocv_mat(2, 2), ocv_mat(2, 3),
        0.0, 0.0, 0.0, 1.0);
}

inline cv::Matx44d Matrix4d2cvMat44d(const Eigen::Matrix<double, 4, 4>& m)
{
    cv::Matx44d cvMat = cv::Matx44d::eye();
    cv::eigen2cv(m, cvMat);
    return cvMat;
}

inline cv::Matx33d Matrix3d2cvMat33d(const Eigen::Matrix3d& m)
{
    cv::Matx33d cvMat = cv::Matx33d::eye();
    cv::eigen2cv(m, cvMat);
    return cvMat;
}

inline cv::Matx44d MatrixSE32CvSE3(const Eigen::Matrix<double, 3, 3>& R,
    const Eigen::Matrix<double, 3, 1> &t)
{
    cv::Matx44d cvMat = cv::Matx44d::eye();
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            cvMat(i, j) = R(i, j);

    for (int i = 0; i < 3; ++i)
        cvMat(i, 3) = t(i);

    return cvMat;
}

///////////////////////////////////////////////////////////////////////////
// CV -> Eigen
// cvMat2Eigen T
inline Eigen::Isometry3d cvMat2Eigen( cv::Mat& R, cv::Mat& tvec )
{
    // cv::Mat R;
    // cv::Rodrigues( rvec, R );
    Eigen::Matrix3d r;
    cv::cv2eigen(R, r);
  
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();

    Eigen::AngleAxisd angle(r);
    Eigen::Translation<double,3> trans(tvec.at<double>(0,0), tvec.at<double>(0,1), tvec.at<double>(0,2));
    T = angle;
    T(0,3) = tvec.at<double>(0,0); 
    T(1,3) = tvec.at<double>(0,1); 
    T(2,3) = tvec.at<double>(0,2);
    return T;
}

inline Eigen::Matrix<double,3,1> CvMat2Vector3d(const cv::Mat &cvVector)
{
    Eigen::Matrix<double,3,1> v;
    v << cvVector.at<float>(0), cvVector.at<float>(1), cvVector.at<float>(2);

    return v;
}

inline Eigen::Matrix<double,3,3> CvMat2Matrix3d(const cv::Mat &cvMat3)
{
    Eigen::Matrix<double,3,3> M;

    M << cvMat3.at<float>(0,0), cvMat3.at<float>(0,1), cvMat3.at<float>(0,2),
    cvMat3.at<float>(1,0), cvMat3.at<float>(1,1), cvMat3.at<float>(1,2),
    cvMat3.at<float>(2,0), cvMat3.at<float>(2,1), cvMat3.at<float>(2,2);

    return M;
}

// cvMatx -> Eigen
inline Eigen::Matrix<double, 3, 3> cvMatx33d2Matrix3d(const cv::Matx33d& cvMat3)
{
    Eigen::Matrix<double, 3, 3> M;

    M << cvMat3(0, 0), cvMat3(0, 1), cvMat3(0, 2),
        cvMat3(1, 0), cvMat3(1, 1), cvMat3(1, 2),
        cvMat3(2, 0), cvMat3(2, 1), cvMat3(2, 2);

    return M;
}

inline Eigen::Matrix<double, 3, 1> cvVec4d2Vector3d(const cv::Vec4d& cvVector)
{
    Eigen::Matrix<double, 3, 1> v;
    v << cvVector(0), cvVector(1), cvVector(2);
    return v;
}

inline Eigen::Matrix<double, 3, 1> cvVec3d2Vector3d(const cv::Vec3d& cvVector)
{
    Eigen::Matrix<double, 3, 1> v;
    v << cvVector(0), cvVector(1), cvVector(2);

    return v;
}
//////////////////////////////////////////////////////////////////////////////////

} // end namespace ackermann_msckf

#endif // MSCKF_VIO_MATH_UTILS_HPP

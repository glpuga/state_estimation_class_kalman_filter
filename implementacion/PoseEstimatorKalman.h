#pragma once

#include <Eigen/Core>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <opencv2/core.hpp>

#include "sophus/se3.hpp"

#define MAX_LEVELS 6
#define NUM_SAMPLES 800

class PoseEstimatorKalman {
 public:
  PoseEstimatorKalman(double fx, double fy, double cx, double cy, int32_t width, int32_t height);

  void setReferenceFrame(const cv::Mat &keyFrame);

  void setReferenceDepth(const cv::Mat &image_depth);

  Sophus::SE3f updateEstimator(const cv::Mat &frame);

  void reset();

  Sophus::SE3f estimated_reference_to_current_transform_;

 private:
  static constexpr double initial_R_standard_dev_ = 0.01;
  static constexpr double initial_Q_standard_dev_ = 4.0;

  const Eigen::Matrix<float, 6, 6> autocov_Q_;
  const Eigen::MatrixXf autocov_R_;

  Eigen::Matrix3f camera_matrix_K_[MAX_LEVELS];
  Eigen::Matrix3f inverse_camera_matrix_K_[MAX_LEVELS];

  int width_per_level_[MAX_LEVELS];
  int height_per_level_[MAX_LEVELS];

  cv::Mat reference_frame_[MAX_LEVELS];
  cv::Mat reference_depth_[MAX_LEVELS];

  Eigen::Matrix<float, 6, 1> state_x_;
  Eigen::Matrix<float, 6, 6> autocov_P_;

  cv::Mat calculateFrameDerivative(const cv::Mat &frame, const int lvl);

  float calculateResidual(const cv::Mat &frame, const Sophus::SE3f &pose, const int lvl);

  void calculateKalmanUpdate(const cv::Mat &frame, const cv::Mat &frameDer, const int lvl);
};

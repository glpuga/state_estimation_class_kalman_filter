#pragma once

#include <opencv2/core.hpp>

#include <fstream>
#include <iostream>

//#include "Common/se3.h"
#include "sophus/se3.hpp"
#include <Eigen/Core>

//#include "Utils/tictoc.h"

#define MAX_LEVELS 6
#define NUM_SAMPLES 800

class PoseEstimatorKalman {
public:
  PoseEstimatorKalman(float fx, float fy, float cx, float cy, int width,
                      int height);

  void setKeyFrame(cv::Mat keyFrame);
  void setIdepth(cv::Mat image_depth);

  Sophus::SE3f updatePose(cv::Mat frame);
  void reset();

  Sophus::SE3f framePose;

private:
  Eigen::Matrix3f K_[MAX_LEVELS];
  Eigen::Matrix3f K_inverse_[MAX_LEVELS];

  int width_per_level_[MAX_LEVELS];
  int height_per_level_[MAX_LEVELS];

  cv::Mat base_frame_[MAX_LEVELS];
  cv::Mat base_depth_[MAX_LEVELS];

  Eigen::Matrix<float, 6, 1> transform_innovation_;
  Eigen::Matrix<float, 6, 6> sigma_;

  Eigen::Matrix<float, 6, 6> R_;
  Eigen::MatrixXf Q_;

  cv::Mat frameDerivative(cv::Mat frame, int lvl);

  float calcResidual(cv::Mat frame, Sophus::SE3f framePose, int lvl);

  void calcKalmanInc(cv::Mat frame, cv::Mat frameDer, int lvl);
};

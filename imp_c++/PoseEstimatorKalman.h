#pragma once

#include <opencv2/core.hpp>

#include <iostream>
#include <fstream>

//#include "Common/se3.h"
#include <Eigen/Core>
#include "sophus/se3.hpp"

//#include "Utils/tictoc.h"

#define MAX_LEVELS 6
#define NUM_SAMPLES 800

class PoseEstimatorKalman
{
public:
    PoseEstimatorKalman(float fx, float fy, float cx, float cy, int _width, int _height);

    void setKeyFrame(cv::Mat _keyFrame);
    void setIdepth(cv::Mat _idepth);

    Sophus::SE3f updatePose(cv::Mat frame);
    void reset();

    Sophus::SE3f framePose;

private:

    Eigen::Matrix3f K[MAX_LEVELS];
    Eigen::Matrix3f KInv[MAX_LEVELS];

    int width[MAX_LEVELS], height[MAX_LEVELS];

    cv::Mat keyFrame[MAX_LEVELS];
    cv::Mat idepth[MAX_LEVELS];

    Eigen::Matrix<float, 6, 1> mu;
    Eigen::Matrix<float, 6, 6> sigma;

    Eigen::Matrix<float, 6, 6> R;
    Eigen::MatrixXf Q;

    cv::Mat frameDerivative(cv::Mat frame, int lvl);

    float calcResidual(cv::Mat frame, Sophus::SE3f framePose, int lvl);

    void calcKalmanInc(cv::Mat frame, cv::Mat frameDer, int lvl);
};

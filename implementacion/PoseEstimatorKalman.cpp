#include "PoseEstimatorKalman.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

PoseEstimatorKalman::PoseEstimatorKalman(float fx, float fy, float cx, float cy,
                                         int width, int height) {
  for (int lvl = 0; lvl < MAX_LEVELS; lvl++) {
    float scale = std::pow(2.0f, float(lvl));

    width_per_level_[lvl] = int(width / scale);
    height_per_level_[lvl] = int(height / scale);

    K_[lvl] = Eigen::Matrix3f::Zero();
    K_[lvl](0, 0) = fx / scale;
    K_[lvl](1, 1) = fy / scale;
    K_[lvl](2, 2) = 1.0f;
    K_[lvl](0, 2) = cx / scale;
    K_[lvl](1, 2) = cy / scale;

    K_inverse_[lvl] = K_[lvl].inverse();
  }

  transform_innovation_ = Sophus::SE3f::log(framePose);
  sigma_ = Eigen::MatrixXf::Identity(6, 6) * pow(0.01, 2);

  R_ = Eigen::MatrixXf::Identity(6, 6) * pow(0.01, 2);
  /*
  R_ <<   0.0168461,  0.000671446,  -0.00099905, -0.000323362,   0.00701273,
  -0.00436534, 0.000671446,   0.00689487,  0.000451642,   -0.0017632,
  0.000341444, -0.000483745, -0.00099905,  0.000451642,   0.00496891,
  0.000918066, -0.000556971, -0.000559598, -0.000323362,   -0.0017632,
  0.000918066,   0.00129314, -0.000236871, -8.30372e-05, 0.00701273,
  0.000341444, -0.000556971, -0.000236871,   0.00402693,  -0.00226152,
       -0.00436534, -0.000483745, -0.000559598, -8.30372e-05,  -0.00226152,
  0.00245291;
       */
  Q_ = Eigen::MatrixXf::Identity(NUM_SAMPLES, NUM_SAMPLES) * pow(4.0, 2.0);
}

void PoseEstimatorKalman::reset() {
  framePose = Sophus::SE3f();
  transform_innovation_ = Sophus::SE3f::log(framePose);
  sigma_ = Eigen::MatrixXf::Identity(6, 6) * pow(0.01, 2);
}

void PoseEstimatorKalman::setKeyFrame(cv::Mat keyFrame) {
  for (int lvl = 0; lvl < MAX_LEVELS; lvl++) {
    cv::resize(keyFrame, base_frame_[lvl],
               cv::Size(width_per_level_[lvl], height_per_level_[lvl]), 0, 0,
               cv::INTER_AREA);
  }
}

void PoseEstimatorKalman::setIdepth(cv::Mat image_depth) {
  for (int lvl = 0; lvl < MAX_LEVELS; lvl++) {
    cv::resize(image_depth, base_depth_[lvl],
               cv::Size(width_per_level_[lvl], height_per_level_[lvl]), 0, 0,
               cv::INTER_AREA);
  }
}

Sophus::SE3f PoseEstimatorKalman::updatePose(cv::Mat frame) {
  cv::Mat level_frame[MAX_LEVELS];
  cv::Mat frameDer[MAX_LEVELS];

  for (int lvl = 0; lvl < MAX_LEVELS; lvl++) {
    cv::resize(frame, level_frame[lvl],
               cv::Size(width_per_level_[lvl], height_per_level_[lvl]), 0, 0,
               cv::INTER_AREA);
    frameDer[lvl] = frameDerivative(level_frame[lvl], lvl);
  }

  for (int lvl = MAX_LEVELS - 1; lvl >= 0; lvl--) {
    float last_error = calcResidual(level_frame[lvl], framePose, lvl);

    std::cout << "lvl " << lvl << " init error " << last_error << std::endl;
    calcKalmanInc(level_frame[lvl], frameDer[lvl], lvl);

    framePose = Sophus::SE3f::exp(transform_innovation_) * framePose;

    float error = calcResidual(level_frame[lvl], framePose, lvl);

    if (error < last_error)
      std::cout << "good update, error " << error << std::endl;
    else
      std::cout << "bad update, error " << error << std::endl;
  }

  return framePose;
}

void PoseEstimatorKalman::calcKalmanInc(cv::Mat frame, cv::Mat frameDer,
                                        int lvl) {}

float PoseEstimatorKalman::calcResidual(cv::Mat frame, Sophus::SE3f framePose,
                                        int lvl) {
  // std::cout << "entrando calcResidual" << std::endl;

  float residual = 0;
  int num = 0;

  cv::Mat debug(height_per_level_[lvl], width_per_level_[lvl], CV_32FC1, 0.0);

  for (int y = 0; y < height_per_level_[lvl]; y++)
    for (int x = 0; x < width_per_level_[lvl]; x++) {

      uchar vkf = base_frame_[lvl].at<uchar>(y, x);
      float keyframeId = base_depth_[lvl].at<float>(y, x);

      Eigen::Vector3f poinKeyframe =
          (K_inverse_[lvl] * Eigen::Vector3f(x, y, 1.0)) / keyframeId;
      Eigen::Vector3f pointFrame = framePose * poinKeyframe;

      if (pointFrame(2) <= 0.0)
        continue;

      Eigen::Vector3f pixelFrame = (K_[lvl] * pointFrame) / pointFrame(2);

      if (pixelFrame(0) < 0.0 || pixelFrame(0) > width_per_level_[lvl] ||
          pixelFrame(1) < 0 || pixelFrame(1) > height_per_level_[lvl])
        continue;

      uchar vf = frame.at<uchar>(pixelFrame(1), pixelFrame(0));

      float res = (vkf - vf);

      residual += res * res;
      num++;

      debug.at<float>(y, x) = abs(res) * 0.01;
    }

  cv::namedWindow("calcResidual debug", cv::WINDOW_NORMAL);
  cv::imshow("calcResidual debug", debug);
  cv::waitKey(30);

  return residual / num;
}

cv::Mat PoseEstimatorKalman::frameDerivative(cv::Mat frame, int lvl) {
  cv::Mat der(height_per_level_[lvl], width_per_level_[lvl], CV_32FC2,
              cv::Vec2f(0.0, 0.0));

  for (int y = 1; y < height_per_level_[lvl] - 1; y++)
    for (int x = 1; x < width_per_level_[lvl] - 1; x++) {
      cv::Vec2f d;
      d.val[0] =
          float((frame.at<uchar>(y, x + 1)) - (frame.at<uchar>(y, x - 1))) /
          2.0;
      d.val[1] =
          float((frame.at<uchar>(y + 1, x)) - (frame.at<uchar>(y - 1, x))) /
          2.0;

      der.at<cv::Vec2f>(y, x) = d;
    }

  return der;
}

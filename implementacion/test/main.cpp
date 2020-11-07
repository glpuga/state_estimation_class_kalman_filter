#include <stdio.h>

#include <Eigen/Core>
#include <cstdint>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <string>

#include "PoseEstimatorKalman.h"
#include "sophus/se3.hpp"
#include "utils/convertAhandaPovRayToStandard.h"

const std::string dataset_path_{"../../dataset/"};

inline bool fileExist(const std::string &name) {
  std::ifstream f(name.c_str());
  return f.good();
}

int32_t main(void) {
  int32_t frame_count_value = 0;
  int32_t frame_counter_direction = 1;

  constexpr int32_t width_per_level_ = 640;
  constexpr int32_t height_per_level_ = 480;

  constexpr double fx = 481.20;
  constexpr double fy = 480.0;
  constexpr double cx = 319.5;
  constexpr double cy = 239.5;

  auto itodigits = [](const int32_t count) {
    constexpr int32_t MAXLEN = 4;
    char buffer[MAXLEN];
    snprintf(buffer, MAXLEN, "%03d", count);
    return std::string(buffer);
  };

  cv::Mat reference_frame_ = cv::imread(dataset_path_ + "scene_000.png", cv::IMREAD_GRAYSCALE);

  Sophus::SE3f reference_camera_world_pose = readPose((dataset_path_ + "scene_000.txt").c_str());

  cv::Mat reference_frame_depth;

  cv::FileStorage fs(dataset_path_ + "scene_depth_000.yml", cv::FileStorage::READ);
  fs["idepth"] >> reference_frame_depth;

  PoseEstimatorKalman kalman_pose_estimator(fx, fy, cx, cy, width_per_level_, height_per_level_);

  kalman_pose_estimator.setReferenceFrame(reference_frame_.clone());
  kalman_pose_estimator.setReferenceDepth(reference_frame_depth);
  kalman_pose_estimator.reset();

  while (1) {
    frame_count_value += frame_counter_direction;
    if (frame_count_value > 598) frame_counter_direction = -1;
    if (frame_count_value < 2) frame_counter_direction = 1;

    const std::string current_frame_path = dataset_path_ + "scene_" + itodigits(frame_count_value) + ".png";
    const std::string scene_pose_path = dataset_path_ + "scene_" + itodigits(frame_count_value) + ".txt";
    const std::string scene_depth_path = dataset_path_ + "scene_depth_" + itodigits(frame_count_value) + ".yml";

    // Read the current current_frame
    const cv::Mat current_frame = cv::imread(current_frame_path, cv::IMREAD_GRAYSCALE);
    // Read the current_frame's pose
    const Sophus::SE3f scene_camera_world_pose = readPose(scene_pose_path.c_str());

    // Determine the pose of the reference camera in the current frame's
    // coordinate frame
    Sophus::SE3f transform_reference_to_current_frame = scene_camera_world_pose * reference_camera_world_pose.inverse();

    kalman_pose_estimator.updateEstimator(current_frame);

    std::cout << std::endl;
    std::cout << "---" << std::endl;

    std::cout << std::endl;
    const auto &reference_tranform = transform_reference_to_current_frame.matrix();
    std::cout << "Real pose relative to reference" << std::endl;
    std::cout << reference_tranform << std::endl;

    std::cout << std::endl;
    const auto &kalman_estimation_tranform = kalman_pose_estimator.estimated_reference_to_current_transform_.matrix();
    std::cout << "Estimated pose relative to reference" << std::endl;
    std::cout << kalman_estimation_tranform << std::endl;

    std::cout << std::endl;
    const double error_distance = std::sqrt(pow(reference_tranform(0, 3) - kalman_estimation_tranform(0, 3), 2.0) +
                                            pow(reference_tranform(1, 3) - kalman_estimation_tranform(1, 3), 2.0) +
                                            pow(reference_tranform(2, 3) - kalman_estimation_tranform(2, 3), 2.0));
    std::cout << "Error distance: " << error_distance << std::endl;
    std::cout << std::endl;

    cv::imshow("Current frame", current_frame);
    cv::imshow("Reference frame", reference_frame_);
    cv::imshow("Reference frame depth", reference_frame_depth);
    cv::waitKey(30);

    // If there depth information for the current image, then
    // make the current image the new reference.
    if (fileExist(scene_depth_path) || false) {
      std::cout << "Reference frame found, filter reference: " << scene_depth_path << std::endl;
      std::cout << std::endl;

      cv::FileStorage fs(scene_depth_path, cv::FileStorage::READ);
      fs["idepth"] >> reference_frame_depth;

      // Updates the keyframe to always do the update relative to the latest
      // image this is needed because the scene might change too much (this
      // dataset does not really go that far)
      kalman_pose_estimator.setReferenceFrame(current_frame.clone());
      kalman_pose_estimator.setReferenceDepth(reference_frame_depth);
      kalman_pose_estimator.reset();

      reference_frame_ = current_frame.clone();
      reference_camera_world_pose = scene_camera_world_pose;
    }
  }

  return 1;
}

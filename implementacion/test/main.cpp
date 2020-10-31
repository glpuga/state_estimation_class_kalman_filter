#include <iostream>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <stdio.h>

#include "utils/convertAhandaPovRayToStandard.h"
// #include "DepthEstimatorCostVolume.h"
// #include "PoseEstimatorGradient.h"
#include "PoseEstimatorKalman.h"
// #include "PoseEstimatorParticle.h"
//#include "DepthEstimator/Common/se3.h"
#include "sophus/se3.hpp"
#include <Eigen/Core>

#define dataset_path_ "../../dataset/"

inline bool fileExist(const std::string &name) {
  std::ifstream f(name.c_str());
  return f.good();
}

int main(void) {
  int frameNumber = 0;
  int frameCounterDirection = 1;

  int framesToTrack = 20; // rand() % 10 + 50;
  int framesTracked = 0;

  int width_per_level_ = 640;
  int height_per_level_ = 480;

  float fx, fy, cx, cy;
  fx = 481.20;
  fy = 480.0;
  cx = 319.5;
  cy = 239.5;

  cv::Mat base_frame_ =
      cv::imread(dataset_path_ "scene_000.png", cv::IMREAD_GRAYSCALE);
  Sophus::SE3f keyframePose = readPose(dataset_path_ "scene_000.txt");

  cv::Mat iDepth;
  cv::FileStorage fs(dataset_path_ "scene_depth_000.yml",
                     cv::FileStorage::READ);
  fs["base_depth_"] >> iDepth;

  // PoseEstimatorGradient
  // poseEstimator(fx,fy,cx,cy,width_per_level_,height_per_level_);
  PoseEstimatorKalman poseEstimator(fx, fy, cx, cy, width_per_level_,
                                    height_per_level_);
  // PoseEstimatorParticle
  // poseEstimator(fx,fy,cx,cy,width_per_level_,height_per_level_);

  poseEstimator.setKeyFrame(base_frame_.clone());
  poseEstimator.setIdepth(iDepth);

  while (1) {
    framesTracked++;
    frameNumber += frameCounterDirection;
    if (frameNumber > 598)
      frameCounterDirection = -1;
    if (frameNumber < 2)
      frameCounterDirection = 1;

    char image_filename[500];
    char RT_filename[500];

    // file name
    sprintf(image_filename, dataset_path_ "scene_%03d.png", frameNumber);
    sprintf(RT_filename, dataset_path_ "scene_%03d.txt", frameNumber);

    Sophus::SE3f pose = readPose(RT_filename);
    cv::Mat frame = cv::imread(image_filename, cv::IMREAD_GRAYSCALE);

    Sophus::SE3f realPose = pose * keyframePose.inverse();

    poseEstimator.updatePose(frame);

    std::cout << "real pose " << std::endl;
    std::cout << realPose.matrix() << std::endl;
    std::cout << "est pose " << std::endl;
    std::cout << poseEstimator.framePose.matrix() << std::endl;

    cv::imshow("image (input)", frame);
    cv::imshow("keyframe (reference)", base_frame_);
    cv::imshow("base_depth_", iDepth);
    cv::waitKey(30);

    if (framesTracked >= framesToTrack) {

      char depth_filename[500];
      sprintf(depth_filename, dataset_path_ "scene_depth_%03d.yml",
              frameNumber);

      if (!fileExist(depth_filename))
        continue;

      cv::FileStorage fs(depth_filename, cv::FileStorage::READ);
      fs["base_depth_"] >> iDepth;

      printf("frames tracked %d\n", framesTracked);
      framesTracked = 0;
      framesToTrack = rand() % 5 + 25; // 25;

      poseEstimator.setKeyFrame(frame.clone());
      poseEstimator.setIdepth(iDepth);
      poseEstimator.reset();

      base_frame_ = frame.clone();
      keyframePose = pose;

      // save depth
      // char depth_filename[500];
      // sprintf(depth_filename,dataset_path_  "scene_depth_%03d.yml",
      // keyframeNumber);

      // cv::FileStorage fs(depth_filename, cv::FileStorage::WRITE );
      // fs << "base_depth_" << iDepth;  //choose any key here, just be
      // consistant with the one below

      // Mat fm;
      // FileStorage fs("my.yml", FileStorage::READ );
      // fs["base_depth_"] >> fm;
    }
  }

  return 1;
}

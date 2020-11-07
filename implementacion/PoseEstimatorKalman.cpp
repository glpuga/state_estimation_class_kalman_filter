#include "PoseEstimatorKalman.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

PoseEstimatorKalman::PoseEstimatorKalman(double fx, double fy, double cx, double cy, int32_t width, int32_t height)
    : autocov_Q_{Eigen::MatrixXf::Identity(6, 6) * pow(initial_R_standard_dev_, 2)},
      autocov_R_{Eigen::MatrixXf::Identity(NUM_SAMPLES, NUM_SAMPLES) * pow(initial_Q_standard_dev_, 2.0)} {
  for (int lvl = 0; lvl < MAX_LEVELS; ++lvl) {
    double scale = std::pow(2.0f, static_cast<double>(lvl));
    width_per_level_[lvl] = static_cast<int32_t>(width / scale);
    height_per_level_[lvl] = static_cast<int32_t>(height / scale);

    camera_matrix_K_[lvl] = Eigen::Matrix3f::Zero();
    camera_matrix_K_[lvl](0, 0) = fx / scale;
    camera_matrix_K_[lvl](1, 1) = fy / scale;
    camera_matrix_K_[lvl](2, 2) = 1.0f;
    camera_matrix_K_[lvl](0, 2) = cx / scale;
    camera_matrix_K_[lvl](1, 2) = cy / scale;

    inverse_camera_matrix_K_[lvl] = camera_matrix_K_[lvl].inverse();
  }

  state_x_ = Sophus::SE3f::log(estimated_reference_to_current_transform_);
  autocov_P_ = Eigen::MatrixXf::Identity(6, 6) * pow(0.01, 2);
}

void PoseEstimatorKalman::reset() {
  estimated_reference_to_current_transform_ = Sophus::SE3f();
  state_x_ = Sophus::SE3f::log(estimated_reference_to_current_transform_);
  autocov_P_ = Eigen::MatrixXf::Identity(6, 6) * pow(0.01, 2);
}

void PoseEstimatorKalman::setReferenceFrame(const cv::Mat &keyFrame) {
  for (int lvl = 0; lvl < MAX_LEVELS; ++lvl) {
    cv::resize(keyFrame, reference_frame_[lvl], cv::Size(width_per_level_[lvl], height_per_level_[lvl]), 0, 0,
               cv::INTER_AREA);
  }
}

void PoseEstimatorKalman::setReferenceDepth(const cv::Mat &image_depth) {
  for (int lvl = 0; lvl < MAX_LEVELS; ++lvl) {
    cv::resize(image_depth, reference_depth_[lvl], cv::Size(width_per_level_[lvl], height_per_level_[lvl]), 0, 0,
               cv::INTER_AREA);
  }
}

Sophus::SE3f PoseEstimatorKalman::updateEstimator(const cv::Mat &observer_frame) {
  cv::Mat frame_level[MAX_LEVELS];
  cv::Mat observer_frame_derivative[MAX_LEVELS];

  // Resize the current observer_frame to the scales used to gradually reduce
  // the error
  for (int lvl = 0; lvl < MAX_LEVELS; ++lvl) {
    cv::resize(observer_frame, frame_level[lvl], cv::Size(width_per_level_[lvl], height_per_level_[lvl]), 0, 0,
               cv::INTER_AREA);
    observer_frame_derivative[lvl] = calculateFrameDerivative(frame_level[lvl], lvl);
  }

  // Iterate kalman, starting with the image at the coarser scale, and
  // then moving towards the finer resolution to improve estimation as we
  // go
  for (int lvl = MAX_LEVELS - 1; lvl >= 0; --lvl) {
    float pre_update_error = calculateResidual(frame_level[lvl], estimated_reference_to_current_transform_, lvl);

    calculateKalmanUpdate(frame_level[lvl], observer_frame_derivative[lvl], lvl);

    estimated_reference_to_current_transform_ = Sophus::SE3f::exp(state_x_) * estimated_reference_to_current_transform_;

    float post_update_error = calculateResidual(frame_level[lvl], estimated_reference_to_current_transform_, lvl);

    if (post_update_error >= pre_update_error) {
      std::cout << "[[[ bad update ]]], error went from " << pre_update_error << " to " << post_update_error
                << std::endl;
    }
  }

  return estimated_reference_to_current_transform_;
}

void PoseEstimatorKalman::calculateKalmanUpdate(const cv::Mat &observer_frame, const cv::Mat &observer_frame_derivative,
                                                const int lvl) {
  // ---
  // Propagation step

  // est_x[k] = f(reference_frame_x_coordinate[k-1], v[k], u[k])
  //
  // We assume the best estimation is that the update is the transform
  // hasn't changed.
  //
  // Curiously, the current state information is thrown every loop of the
  // filter, and the estimated value is always assumed to be null.
  // The previous state "informs" the current iteration one through the current
  // estimate for the tranform between reference and observer (which is not the
  // state of this filter, the state is the _change_ in the transform), which
  // gets updated after calling this function in updateEstimator() and used
  // here to tranform the coordinates of the points.
  Eigen::Matrix<float, 6, 1> est_state_x = Eigen::MatrixXf::Zero(6, 1);

  // est_P[k] = F[k-1] P[k-1] Ft[k-1] + Q[k]
  //
  // f() is an identity function, so F = I.
  Eigen::Matrix<float, 6, 6> est_autocov_P = autocov_P_ + autocov_Q_;

  // ---
  // Calculation of the linearized output matrix G[k]
  // This is done for each of the NUM_SAMPLE pixel samples,
  // resulting in a (NUM_SAMPLE, 6) matrix.

  Eigen::MatrixXf observed_output_y = Eigen::MatrixXf::Zero(NUM_SAMPLES, 1);
  Eigen::MatrixXf estimated_output_g = Eigen::MatrixXf::Zero(NUM_SAMPLES, 1);
  Eigen::MatrixXf linearized_matrix_G = Eigen::MatrixXf::Zero(NUM_SAMPLES, 6);

  // We don't work on the whole image, but only on a subset of the pixels.
  // We choose randomly NUM_SAMPLES pixels in this set.
  // We don't use a FOR look here because we are dropping some pixels below if
  // they are not within the field of view of both cameras, and we don't want
  // to count them as used.
  int32_t sample = 0;
  while (sample < NUM_SAMPLES) {
    // Select a pair of coordinates within the reference frame randomly
    const int32_t reference_frame_x_coordinate = rand() % width_per_level_[lvl];
    const int32_t reference_frame_y_coordinate = rand() % height_per_level_[lvl];

    // Get a pixel from the reference frame at the coordinates of the
    // pixel
    const uchar reference_pixel_value =
        reference_frame_[lvl].at<uchar>(reference_frame_y_coordinate, reference_frame_x_coordinate);

    // Extract the dept information for that pixel. This dataset stores the
    // inverse depth (1/observed_output_y) information, instead of the depth.
    double reference_pixel_inverse_depth =
        reference_depth_[lvl].at<float>(reference_frame_y_coordinate, reference_frame_x_coordinate);

    // Map projection coordinates to camera coordinate frame
    // coordinates
    Eigen::Vector3f reference_point_projection_coordinates =
        inverse_camera_matrix_K_[lvl] *
        Eigen::Vector3f(reference_frame_x_coordinate, reference_frame_y_coordinate, 1.0);

    // Undo the projection on the reference camera, by introducing the depth
    // information. This gets the coordinates in the reference camera coordinate
    // frame.
    Eigen::Vector3f reference_point_3d_coordinates =
        reference_point_projection_coordinates / reference_pixel_inverse_depth;

    // Tranform the 3D point from the reference camera coord frame to the
    // observer camera coordinate frame
    Eigen::Vector3f observer_point_3d_coordinates =
        estimated_reference_to_current_transform_ * reference_point_3d_coordinates;

    // ignore pixels that fall behind the observer camera
    if (observer_point_3d_coordinates(2) < 0.0) {
      continue;
    }

    // Get the projection coordinates on the virtual image plane of the observer
    // camera. The resulting vector is in homogeneus coordinates (z == 1).
    Eigen::Vector3f observer_point_projection_coordinates =
        observer_point_3d_coordinates / observer_point_3d_coordinates(2);

    // Get the pixel coordinates by multiplying the coordinates and the
    // sensor camera matrix K.
    Eigen::Vector3f observer_frame_uv_coordinates = camera_matrix_K_[lvl] * observer_point_projection_coordinates;

    // Disregard pixels that fall outside of the observer frame that we have
    if (observer_frame_uv_coordinates(0) < 0.0 || observer_frame_uv_coordinates(0) >= width_per_level_[lvl] ||
        observer_frame_uv_coordinates(1) < 0.0 || observer_frame_uv_coordinates(1) > height_per_level_[lvl]) {
      continue;
    }

    // Get the value of the pixel at the coordinates where we expect
    // to see the same image point that we projected from the reference camera
    const uchar observer_pixel_value =
        observer_frame.at<uchar>(observer_frame_uv_coordinates(1), observer_frame_uv_coordinates(0));

    // These are just some convenience variables used to store divisions by gz that
    // are used frequently below.
    const double inv_gz = 1.0 / observer_point_3d_coordinates(2);
    const double inv_gz_sq = 1.0 / (observer_point_3d_coordinates(2) * observer_point_3d_coordinates(2));

    // Get the 2x1 vector that represents the derivative of observer frame. This value
    // is used as part of the chain rule used to determine G[k] below.
    cv::Vec2f dd =
        observer_frame_derivative.at<cv::Vec2f>(observer_frame_uv_coordinates(1), observer_frame_uv_coordinates(0));
    // Get the components of the derivative vector in convenience variables
    const double du = dd.val[0];
    const double dv = dd.val[1];

    // Store the obseved output along with the expected output. This is used later
    // to calculate the innovation of the filter.
    observed_output_y(sample, 0) = float(reference_pixel_value);
    estimated_output_g(sample, 0) = float(observer_pixel_value);

    // The code below does the product
    // G[k] = image_derivative(u, v) * jacobian of the projection for a single
    // point. That is: image_derivative(u, v) calculated elsewhere, multiplied
    // by the 2-by-6 matrix in slide 19 of the Lie Groups powerpoint
    //
    // The result is the 1-by-6 G[k] matrix for this particular pixel sample,
    // which gets added to the overall NUM_SAMPLES-by-6 G[k] matrix.

    // du * fx / gz
    linearized_matrix_G(sample, 0) = du * camera_matrix_K_[lvl](0, 0) * inv_gz;

    // dv * fy / gz
    linearized_matrix_G(sample, 1) = dv * camera_matrix_K_[lvl](1, 1) * inv_gz;

    // - du * fx * gx / gz^2  * - dv * fy * gy / gz^2
    linearized_matrix_G(sample, 2) =
        du * (-camera_matrix_K_[lvl](0, 0)) * observer_point_3d_coordinates(0) * inv_gz_sq +
        dv * (-camera_matrix_K_[lvl](1, 1)) * observer_point_3d_coordinates(1) * inv_gz_sq;

    // - du * fx * gx * gy / gz^2 - dv * fy * (1 + gy^2 / gz^2)
    linearized_matrix_G(sample, 3) =
        du * (-camera_matrix_K_[lvl](0, 0)) * observer_point_3d_coordinates(0) * observer_point_3d_coordinates(1) *
            inv_gz_sq +
        dv * (-camera_matrix_K_[lvl](1, 1)) *
            (1.0 + observer_point_3d_coordinates(1) * observer_point_3d_coordinates(1) * inv_gz_sq);

    // du * fx * (1 + gx^2 / gz^2) + dv * fy * gx * gy / gz^2
    linearized_matrix_G(sample, 4) =
        du * camera_matrix_K_[lvl](0, 0) *
            (1.0 + observer_point_3d_coordinates(0) * observer_point_3d_coordinates(0) * inv_gz_sq) +
        dv * camera_matrix_K_[lvl](1, 1) * observer_point_3d_coordinates(0) * observer_point_3d_coordinates(1) *
            inv_gz_sq;

    // du * (-fx) * gy / gz + dv * fy * gx * gz
    linearized_matrix_G(sample, 5) = du * (-camera_matrix_K_[lvl](0, 0)) * observer_point_3d_coordinates(1) * inv_gz +
                                     dv * camera_matrix_K_[lvl](1, 1) * observer_point_3d_coordinates(0) * inv_gz;

    sample++;
  }

  // ---
  // Calculate the Kalman Gain

  // S[k] = linearized_matrix_G[k] * est_P[k] * inv(linearized_matrix_G[k]) +
  // R[k] kalman_gain_K[k] = est_P[k] * (trans(linearized_matrix_G[k]) *
  // inv(S[k]))
  //
  // This is the bottleneck of the algorithm, because we are inverting an
  // NUM_SAMPLES x NUM_SAMPLES matrix.
  Eigen::MatrixXf S = linearized_matrix_G * (est_autocov_P * (linearized_matrix_G.transpose())) + autocov_R_;
  Eigen::MatrixXf kalman_gain_K = est_autocov_P * ((linearized_matrix_G.transpose()) * S.inverse());

  // ---
  // Estimate correction

  // x[k] = est_x[k] + kalman_gain_K[k] * (y[k] - g(est_x[k], u[k])),
  // in this problem u[k] is 0
  state_x_ = est_state_x + kalman_gain_K * (observed_output_y - estimated_output_g);

  // ---
  // Estimate autocov correction

  // P[k] = (I - kalman_gain_K[k] * linearized_matrix_G[k]) * est_autocov_P[k]
  autocov_P_ = (Eigen::MatrixXf::Identity(6, 6) - kalman_gain_K * linearized_matrix_G) * est_autocov_P;
}

float PoseEstimatorKalman::calculateResidual(const cv::Mat &observer_frame, const Sophus::SE3f &pose, const int lvl) {
  // std::cout << "entrando calculateResidual" << std::endl;

  float residual = 0;
  int num = 0;

  cv::Mat debug(height_per_level_[lvl], width_per_level_[lvl], CV_32FC1, 0.0);

  for (int reference_frame_y_coordinate = 0; reference_frame_y_coordinate < height_per_level_[lvl];
       reference_frame_y_coordinate++)
    for (int reference_frame_x_coordinate = 0; reference_frame_x_coordinate < width_per_level_[lvl];
         reference_frame_x_coordinate++) {
      uchar reference_pixel_value =
          reference_frame_[lvl].at<uchar>(reference_frame_y_coordinate, reference_frame_x_coordinate);
      float reference_pixel_inverse_depth =
          reference_depth_[lvl].at<float>(reference_frame_y_coordinate, reference_frame_x_coordinate);

      Eigen::Vector3f poinKeyframe =
          (inverse_camera_matrix_K_[lvl] *
           Eigen::Vector3f(reference_frame_x_coordinate, reference_frame_y_coordinate, 1.0)) /
          reference_pixel_inverse_depth;
      Eigen::Vector3f observer_point_3d_coordinates = pose * poinKeyframe;

      if (observer_point_3d_coordinates(2) <= 0.0) continue;

      Eigen::Vector3f observer_frame_uv_coordinates =
          (camera_matrix_K_[lvl] * observer_point_3d_coordinates) / observer_point_3d_coordinates(2);

      if (observer_frame_uv_coordinates(0) < 0.0 || observer_frame_uv_coordinates(0) > width_per_level_[lvl] ||
          observer_frame_uv_coordinates(1) < 0 || observer_frame_uv_coordinates(1) > height_per_level_[lvl])
        continue;

      uchar observer_pixel_value =
          observer_frame.at<uchar>(observer_frame_uv_coordinates(1), observer_frame_uv_coordinates(0));

      float res = (reference_pixel_value - observer_pixel_value);

      residual += res * res;
      num++;

      debug.at<float>(reference_frame_y_coordinate, reference_frame_x_coordinate) = abs(res) * 0.01;
    }

  cv::namedWindow("calculateResidual debug", cv::WINDOW_NORMAL);
  cv::imshow("calculateResidual debug", debug);
  cv::waitKey(30);

  return residual / num;
}

cv::Mat PoseEstimatorKalman::calculateFrameDerivative(const cv::Mat &observer_frame, const int lvl) {
  cv::Mat observer_frame_derivative(height_per_level_[lvl], width_per_level_[lvl], CV_32FC2, cv::Vec2f(0.0, 0.0));

  cv::Vec2f point_derivative;
  for (int reference_frame_y_coordinate = 1; reference_frame_y_coordinate < height_per_level_[lvl] - 1;
       ++reference_frame_y_coordinate)
    for (int reference_frame_x_coordinate = 1; reference_frame_x_coordinate < width_per_level_[lvl] - 1;
         ++reference_frame_x_coordinate) {
      point_derivative.val[0] =
          float((observer_frame.at<uchar>(reference_frame_y_coordinate, reference_frame_x_coordinate + 1)) -
                (observer_frame.at<uchar>(reference_frame_y_coordinate, reference_frame_x_coordinate - 1))) /
          2.0;
      point_derivative.val[1] =
          float((observer_frame.at<uchar>(reference_frame_y_coordinate + 1, reference_frame_x_coordinate)) -
                (observer_frame.at<uchar>(reference_frame_y_coordinate - 1, reference_frame_x_coordinate))) /
          2.0;

      observer_frame_derivative.at<cv::Vec2f>(reference_frame_y_coordinate, reference_frame_x_coordinate) =
          point_derivative;
    }

  return observer_frame_derivative;
}

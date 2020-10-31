#include "PoseEstimatorKalman.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

PoseEstimatorKalman::PoseEstimatorKalman(float fx, float fy, float cx, float cy, int _width, int _height)
{
    for(int lvl = 0; lvl < MAX_LEVELS; lvl++)
    {
        float scale = std::pow(2.0f, float(lvl));

        width[lvl] = int(_width/scale);
        height[lvl] = int(_height/scale);

        K[lvl] = Eigen::Matrix3f::Zero();
        K[lvl](0,0) = fx/scale;
        K[lvl](1,1) = fy/scale;
        K[lvl](2,2) = 1.0f;
        K[lvl](0,2) = cx/scale;
        K[lvl](1,2) = cy/scale;

        KInv[lvl] = K[lvl].inverse();
    }

    mu = Sophus::SE3f::log(framePose);
    sigma = Eigen::MatrixXf::Identity(6,6)*pow(0.01,2);

    R = Eigen::MatrixXf::Identity(6,6)*pow(0.01,2);
    /*
    R <<   0.0168461,  0.000671446,  -0.00099905, -0.000323362,   0.00701273,  -0.00436534,
         0.000671446,   0.00689487,  0.000451642,   -0.0017632,  0.000341444, -0.000483745,
         -0.00099905,  0.000451642,   0.00496891,  0.000918066, -0.000556971, -0.000559598,
        -0.000323362,   -0.0017632,  0.000918066,   0.00129314, -0.000236871, -8.30372e-05,
          0.00701273,  0.000341444, -0.000556971, -0.000236871,   0.00402693,  -0.00226152,
         -0.00436534, -0.000483745, -0.000559598, -8.30372e-05,  -0.00226152,   0.00245291;
         */
    Q = Eigen::MatrixXf::Identity(NUM_SAMPLES, NUM_SAMPLES)*pow(4.0,2.0);
}

void PoseEstimatorKalman::reset()
{
    framePose = Sophus::SE3f();
    mu = Sophus::SE3f::log(framePose);
    sigma = Eigen::MatrixXf::Identity(6,6)*pow(0.01,2);
}

void PoseEstimatorKalman::setKeyFrame(cv::Mat _keyFrame)
{
    for(int lvl = 0; lvl < MAX_LEVELS; lvl++)
    {
        cv::resize(_keyFrame, keyFrame[lvl], cv::Size(width[lvl],height[lvl]),0,0,cv::INTER_AREA);
        //cv::GaussianBlur(_keyFrame, keyFrame[lvl], cv::Size(lvl*2+1,lvl*2+1),0);
        //cv::blur(_keyFrame, keyFrame[lvl], cv::Size(lvl*2+1,lvl*2+1));
    }
}

void PoseEstimatorKalman::setIdepth(cv::Mat _idepth)
{
    for(int lvl = 0; lvl < MAX_LEVELS; lvl++)
    {
        cv::resize(_idepth, idepth[lvl], cv::Size(width[lvl],height[lvl]),0,0,cv::INTER_AREA);
        //cv::GaussianBlur(_idepth, idepth[lvl], cv::Size(lvl*2+1,lvl*2+1),0);
        //cv::blur(_idepth, idepth[lvl], cv::Size(lvl*2+1,lvl*2+1));
        //idepth[lvl] = _idepth;
    }
}

Sophus::SE3f PoseEstimatorKalman::updatePose(cv::Mat _frame)
{
    //std::cout << "entrando updatePose\n";

    //Sophus::SE3f newPose;
    //framePose = newPose;

    cv::Mat frame[MAX_LEVELS];
    cv::Mat frameDer[MAX_LEVELS];

    for(int lvl = 0; lvl < MAX_LEVELS; lvl++)
    {

        //cv::GaussianBlur(_frame, frame[lvl], cv::Size(lvl*2+1,lvl*2+1),0);
        //cv::blur(_frame, frame[lvl], cv::Size(lvl*2+1,lvl*2+1));
        cv::resize(_frame, frame[lvl], cv::Size(width[lvl],height[lvl]),0,0,cv::INTER_AREA);
        //if(lvl == 0)
            frameDer[lvl] = frameDerivative(frame[lvl], lvl);
        //else
            //cv::resize(frameDer[0], frameDer[lvl], cv::Size(width[lvl],height[lvl]),0,0,cv::INTER_LINEAR);
            //cv::GaussianBlur(frameDer[0], frameDer[lvl], cv::Size(lvl*2+1,lvl*2+1),0);

    }

    //calcResidual(frame[0],framePose,0);
    //return framePose;

    //std::cout << "se setearon frames y frameder\n";

    int maxIterations[10] = {5, 20, 100, 100, 100, 100, 100, 100, 100, 100};

    //int lvl = 3;
    for(int lvl=MAX_LEVELS-1; lvl >= 0; lvl--)
    {
        //std::cin.get();

        float last_error = calcResidual(frame[lvl],framePose,lvl);

        //std::cout << "pose for lvl " << lvl << std::endl;
        //std::cout << framePose.matrix() << std::endl;
        std::cout << "lvl " << lvl << " init error " << last_error << std::endl;

        //for(int i = 0; i < maxIterations[lvl]; i++)
        //for(int it = 0; it < maxIterations[lvl]; it++)
        {
            //calcHJ(frame[lvl], frameDer[lvl], framePose ,lvl);
            //calcHJ_2(frame[lvl], frameDer[lvl], framePose ,lvl);
            calcKalmanInc(frame[lvl], frameDer[lvl],lvl);

            framePose = Sophus::SE3f::exp(mu) * framePose;

            float error = calcResidual(frame[lvl], framePose, lvl);

            if(error < last_error)
                std::cout << "good update, error " << error << std::endl;
            else
                std::cout << "bad update, error " << error << std::endl;
        }
    }

    return framePose;
}

void PoseEstimatorKalman::calcKalmanInc(cv::Mat frame, cv::Mat frameDer, int lvl)
{

}



float PoseEstimatorKalman::calcResidual(cv::Mat frame, Sophus::SE3f framePose, int lvl)
{
    //std::cout << "entrando calcResidual" << std::endl;

    float residual = 0;
    int num = 0;

    cv::Mat debug(height[lvl], width[lvl], CV_32FC1, 0.0);

    for(int y = 0; y < height[lvl]; y++)
        for(int x = 0; x < width[lvl]; x++)
        {
            //std::cout << "pixel: " << y << " " << x << std::endl;

            uchar vkf = keyFrame[lvl].at<uchar>(y,x);
            float keyframeId = idepth[lvl].at<float>(y,x);

            //std::cout << "vkf " << vkf << " id " << id << std::endl;

            Eigen::Vector3f poinKeyframe = (KInv[lvl]*Eigen::Vector3f(x,y,1.0))/keyframeId;
            Eigen::Vector3f pointFrame = framePose*poinKeyframe;

            if(pointFrame(2) <= 0.0)
                continue;

            Eigen::Vector3f pixelFrame = (K[lvl]*pointFrame)/pointFrame(2);

            if(pixelFrame(0) < 0.0 || pixelFrame(0) > width[lvl] || pixelFrame(1) < 0 || pixelFrame(1) > height[lvl])
                continue;

            uchar vf = frame.at<uchar>(pixelFrame(1), pixelFrame(0));

            float res = (vkf-vf);

            //std::cout << "pixel " << " " << float(vkf) << " " << float(vf) << " res " << res << std::endl;

            residual += res*res;
            num++;

            //std::cout << "accres " << residual << std::endl;

            debug.at<float>(y,x) = abs(res)*0.01;
        }

    cv::namedWindow("calcResidual debug", cv::WINDOW_NORMAL);
    cv::imshow("calcResidual debug", debug);
    cv::waitKey(30);

    return residual/num;
}

cv::Mat PoseEstimatorKalman::frameDerivative(cv::Mat frame, int lvl)
{
    cv::Mat der(height[lvl], width[lvl], CV_32FC2, cv::Vec2f(0.0,0.0));

    for(int y = 1; y < height[lvl]-1; y++)
        for(int x = 1; x < width[lvl]-1; x++)
        {
            cv::Vec2f d;
            d.val[0] = float((frame.at<uchar>(y,x+1)) - (frame.at<uchar>(y,x-1)))/2.0;
            d.val[1] = float((frame.at<uchar>(y+1,x)) - (frame.at<uchar>(y-1,x)))/2.0;

            der.at<cv::Vec2f>(y,x) = d;
        }

    return der;
}

// This file converts the file format used at http://www.doc.ic.ac.uk/~ahanda/HighFrameRateTracking/downloads.html
// into the standard [R|T] world -> camera format used by OpenCV
// It is based on a file they provided there, but makes the world coordinate system right handed, with z up,
// x right, and y forward.

#include <iostream>
#include <fstream>
#include <string.h>
#include <stdio.h>

//#include "util/SophusUtil.h"
#include "sophus/se3.hpp"

Sophus::SE3f readPose(const char * filename)
{
    std::ifstream cam_pars_file(filename);
    if(!cam_pars_file.is_open())
    {
        std::cout<<"failed opening file!"<<std::endl;
        exit(1);
    }

    char readlinedata[300];

    Eigen::Vector3f direction;
    Eigen::Vector3f upvector;
    Eigen::Vector3f posvector;


    while(1){
        cam_pars_file.getline(readlinedata,300);
//         cout<<readlinedata<<endl;
        if ( cam_pars_file.eof())
            break;

        std::istringstream iss;

        if ( strstr(readlinedata,"cam_dir")!= NULL){

            std::string cam_dir_str(readlinedata);

            cam_dir_str = cam_dir_str.substr(cam_dir_str.find("= [")+3);
            cam_dir_str = cam_dir_str.substr(0,cam_dir_str.find("]"));

            iss.str(cam_dir_str);
            iss >> direction.x() ;
            iss.ignore(1,',');
            iss >> direction.z() ;
            iss.ignore(1,',') ;
            iss >> direction.y();
            iss.ignore(1,',');
        }

        if ( strstr(readlinedata,"cam_up")!= NULL){

            std::string cam_up_str(readlinedata);

            cam_up_str = cam_up_str.substr(cam_up_str.find("= [")+3);
            cam_up_str = cam_up_str.substr(0,cam_up_str.find("]"));

            iss.str(cam_up_str);
            iss >> upvector.x() ;
            iss.ignore(1,',');
            iss >> upvector.z() ;
            iss.ignore(1,',');
            iss >> upvector.y() ;
            iss.ignore(1,',');
        }

        if ( strstr(readlinedata,"cam_pos")!= NULL){
//            cout<< "cam_pos is present!"<<endl;

            std::string cam_pos_str(readlinedata);

            cam_pos_str = cam_pos_str.substr(cam_pos_str.find("= [")+3);
            cam_pos_str = cam_pos_str.substr(0,cam_pos_str.find("]"));

//            cout << "cam pose str = " << endl;
//            cout << cam_pos_str << endl;

            iss.str(cam_pos_str);
            iss >> posvector.x() ;
            iss.ignore(1,',');
            iss >> posvector.z() ;
            iss.ignore(1,',');
            iss >> posvector.y() ;
            iss.ignore(1,',');
//             cout << "position: "<<posvector.x<< ", "<< posvector.y << ", "<< posvector.z << endl;

        }

    }

//    R=Mat(3,3,CV_64F);
//    R.row(0)=Mat(direction.cross(upvector)).t();
//    R.row(1)=Mat(-upvector).t();
//    R.row(2)=Mat(direction).t();

    Eigen::Matrix3f Rot;
    Rot.row(0) = (direction.cross(upvector)).transpose();
    Rot.row(1) = (-upvector).transpose();
    Rot.row(2) = direction.transpose();

    //T=-R*Mat(posvector);

    Eigen::Vector3f Tra;
    Tra = -Rot*posvector*0.01;

    Sophus::SE3f pose = Sophus::SE3f(Rot, Tra);

    /*
    std::ofstream myfile;

    char new_filename[500];

    //file name
    sprintf(new_filename,"%s.new", filename);

    std::cout << "writing " << new_filename << std::endl;

    myfile.open(new_filename);
    //myfile << "Writing this to a file.\n";
    myfile << pose.matrix() << std::endl;
    myfile.close();
    */

    return pose;
}


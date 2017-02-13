/*
Base class for a controller. Controllers take in sensor readings and choose the action.
*/
#pragma once

// Headers.
#include <queue>
#include <cmath>
#include <mutex>
#include <thread>
#include <time.h>
#include <chrono>
#include <vector>
//ros used
#include <ros/ros.h>
#include <std_msgs/Float64.h>
#include <geometry_msgs/Point.h>
//boost unused
#include <boost/scoped_ptr.hpp>
//myne factory headers
#include <ensenso/boost_sender.h>
#include "nn_controller/amfcError.h"
#include "nn_controller/controller.h"
#include "nn_controller/predictor_params.h"
/* ______________________________________________________________________
    *
    *   This code is part of the superchick project. 
    *  
    *
    *   Author: Olalekan Ogunmolu
    *   Date: Nov. 3, 2016
    *   Lab Affiliation: Gans' Lab, Dallas, TX
    *__________________________________________________________________________
*/

#include <ros/time.h>
#include <ros/ros.h>
#include <Eigen/Dense>
    #include <stdlib.h> /*system, NULL, ..*/

#include "nn_controller/amfc.h"
#include "nn_controller/options.h"
#include "nn_controller/predictor.h"
#define BOOST_NO_CXX11_SCOPED_ENUMS     
#include <boost/filesystem.hpp>
#undef BOOST_NO_CXX11_SCOPED_ENUMS
#include <boost/tuple/tuple.hpp>
#include <boost/property_tree/json_parser.hpp>

#define OUT(__o__) std::cout<< __o__ << std::endl;

//forward declaration to use pathfinder function
namespace pathfinder
{
  bool getROSPackagePath(const std::string pkgName, 
                                boost::filesystem::path & pkgPath);

  static bool copyDirectory(const boost::filesystem::path srcPath,
                             const boost::filesystem::path dstPath);  

  bool cloudsAndImagesPath(boost::filesystem::path & imagesPath, \
                            boost::filesystem::path & cloudsPath, 
                            const std::string& pkgName = "ensenso");

  std::tuple<boost::filesystem::path, const std::string&, const std::string&,
            const std::string&, const std::string&, const std::string&, 
            const std::string&> getCurrentPath();

  bool getDataDirectory(boost::filesystem::path data_dir);
}

namespace amfc_control
{
    /*
    * The neural network accounts for the parametric uncertainties and we do not needd
    * a baseline controller for now
    */
    class Controller
    {
    private:
        //these from /vicon/headtwist topic
        geometry_msgs::Vector3 linear, angular;
        std::mutex mutex, pose_mutex, weights_mutex, biases_mutex;
        bool updatePoseInfo, print, updateController, updateWeights, 
             updateBiases, resetController;
        //useSigma adapts the controllaw nased on 
        //sigma modifiocation
        bool useSigma;
        double sigma_y, sigma_r;
        //these are ref model params
        Eigen::VectorXd tracking_error, 
                        tracking_error_delayed, 
                        tracking_error_delta;
        std::chrono::time_point<std::chrono::high_resolution_clock> start, now;
        ros::NodeHandle n_;
        ros::Publisher pub, pred_pub_, control_pub_;
        double k, dt;
        int m, n;
        unsigned counter;
        /* @brief P is symmetric positive definite matrix obtained from the lyapunov functiuon Q
        *
        *  B is the matrix that maps the controls into the state space
        *
        *  Am is the state-mapping matrix of the reference model
        *
        *  Bm is the input-mapping matrix of the reference model
        */
        Eigen::MatrixXd P, B, Am, Bm; 
        /*
        * p = # states of A matrix
        * Gammas are in R^{px p} diagonal, positive definite 
        * matrices of adaptive gains
        */
        Eigen::MatrixXd Gamma_y, Gamma_r;
        Eigen::Vector3d ref_;
        //time-derivative of gains
        Eigen::MatrixXd Ky_hat_dot, Kr_hat_dot; 
        //estimates of real gains kr and ky
        Eigen::MatrixXd Ky_hat, Kr_hat, 
                      Ky_hat_delayed, Kr_hat_delayed, 
                      Ky_hat_delta, Kr_hat_delta; 
        Eigen::VectorXd pose_info, 
                        pose_info_delayed, 
                        pose_info_delta;        //from sensor
        //reference model output, ref_model differential and final value of ref
        //model integral
        Eigen::Vector3d ym, ym_dot, ym_int;     
        //reference model  and integration term
        Eigen::MatrixXd expAmk, expAmk_tau;              

        /*Lambda is an unknown pos def symmetric matrix in R^{m x m}
        * matrix of
        */
        Eigen::MatrixXd Lambda;
        //adaptive control vector
        Eigen::VectorXd u_control;
        std::thread threads, gainsThread;
        //queue to delay the incoming pose message in order to pick delayed y(t-1)
        std::queue<Eigen::VectorXd> pose_queue, tracking_error_queue,
                                    Ky_hat_queue, ref_queue, Kr_hat_queue;
        Eigen::VectorXd pred_ut;
        // trained net predictor input tuple
        nn_controller::predictor pred_;
        //neural net  pre-trained weights
        Eigen::Matrix<double, 6, 6> modelWeights;
        Eigen::VectorXd modelBiases;

        boost::asio::io_service io_service;
        const std::string multicast_address;
        double loss;
        Eigen::VectorXd pred;
        std::mutex net_loss_mutex, pred_mutex;
        bool updateNetLoss, updatePred;

        /* The type of container used to hold the state vector */
        using state = Eigen::Matrix<double, 3, 6>;
        using ym_state = Eigen::Vector3d;
        //we augment the ref_ matrix with 3x1 zero vector to make multiplication non-singular
        Eigen::VectorXd ref_aug;
        Eigen::Matrix<double, 6,6> idealModelWeights;
        Eigen::VectorXd idealBiases;
        Eigen::VectorXd guessControl;

        void initMatrices()
        {   
            n = 3; m = 6; 
            Am.setIdentity(n, n);
            Bm.setIdentity(n, n);       //note from B*Lambda*Kr^T
            B.setZero(n, m);        //R^{3 x 6}
            Gamma_y.setIdentity(n, n); //will be 3 X 3 matrix
            Gamma_r.setIdentity(n, n); //will be 3 X 3 matrix
            //Lambda models controlundertainties by an R^{nxn} diagonal matrix
            //with positive elements
            Lambda.setIdentity(m, m);            
            P.setIdentity(n, n); // R ^{n x n}
            expAmk.setIdentity(n, n);
            expAmk_tau.setIdentity(n,n);

            modelBiases.resize(6);

            P *= -11503./180; //-1705./2668; //
            Am *= -1334./1705;

            //initialize B so that we have the difference between voltages to each IAB
            B(0,0) = 1; B(0, 1) = 64;
            B(1,2) = 1; B(1, 3) = 64;
            B(2,4) = 1; B(2, 5) = 1;

            pose_info.resize(3);

            //initialize y_m based on expected value of y
            ym << 190.898196896, 17.3539236916, 
                  34.7293864688;

            //gamma scaling factor for adaptive gains
            k = 1e-6;

            Gamma_y *= k;
            Gamma_r *= k;

            /*Initialize weights from previously trained model to usebefore
            we converge in the online approximator*/
            idealModelWeights << 3.7133e+01, -1.4934e+00,  3.8134e+01,
                                 2.9362e+01, -3.4522e+01, -3.3090e+01,
                                 1.7951e+00, -1.7639e-01,  2.4107e+00,  
                                 2.7318e+00, -3.7043e+00, -1.1980e+00,
                                 7.1878e+00, -1.3644e+00,  6.5950e+00,  
                                 5.8266e+00, -6.5263e+00, -8.3548e+00,
                                 3.9877e-01,  1.0487e+00, -2.2137e-01,  
                                 6.7413e-01,  8.9013e-04, -2.7893e-01,
                                 1.0266e+00,  3.7444e+00,  1.4412e+00,  
                                 7.9007e-01,  9.8485e-01,  5.4271e-01,
                                 8.4146e-01, -6.5101e-01,  9.8610e-01, 
                                 -3.1613e-01, -6.4775e-01, -2.3919e-01;
            idealBiases.resize(6);
            idealBiases << 54.3523, 1.1706, 10.6537,  0.1493, -0.2866, -0.8298;

            //control law that will be used in first iteration
            guessControl.resize(6);
            guessControl << 0.85, 200, 0.75, 263, 0.53, 0.86;

            //initialize Kr_hat and Ky_hat to dummy values for sigma modification
            Kr_hat.setIdentity(n,m);
            Ky_hat.setIdentity(n,m);
        }

    public:
        // Constructor.
        Controller(ros::NodeHandle nc, 
                    const Eigen::Vector3d& ref, bool print, bool useSigma);
        Controller();
        // Destructor.
        virtual ~Controller();
        //pose callback
        void getRefTraj();

        /*compute control law
        * Ky_hat_dot will be 6x3, 
        * pose_info will be 3x1
        * Kr_hat_dot will be 6x3
        * ref_ will be 3x1
        * modelWights is \theta & will be 3x3
        * phi(x) will be lagged params 3x1
        * u_control will be 6x1
        */
        void ControllerParams(Eigen::VectorXd&& pose_info);
        //transform eigenPose to ensenso::HeadPose format
        void vectorToHeadPose(Eigen::VectorXd&& pose_info, 
                                          ensenso::HeadPose& eig2Pose);
        //predictor from real-time predictor.lua 
        void pred_subscriber(const geometry_msgs::Point& pred);
        //loss from real-time predictor.lua 
        void loss_subscriber(const std_msgs::Float64& net_loss);
        //controller service
        virtual bool configure_controller(
            nn_controller::controller::Request  &req,
            nn_controller::controller::Response  &res);
        //predictor params for pretrained model
        virtual bool configure_predictor_params(
                nn_controller::predictor_params::Request  &req,
                nn_controller::predictor_params::Response  &res);
        void getPoseInfo(const ensenso::HeadPose& headPose, Eigen::VectorXd pose_info);
        ros::Time getTime();
        //subscribe to the weights and biases params
        virtual void weights_sub(const std_msgs::Float64MultiArray::ConstPtr& weights_sub);  
        virtual void bias_sub(const std_msgs::Float64MultiArray::ConstPtr& bias_params);       
        virtual void pose_subscriber(const ensenso::HeadPose& headPose);
    };
}

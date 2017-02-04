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
#include <ros/ros.h>
#include <boost/scoped_ptr.hpp>
#include <geometry_msgs/Vector3.h>
#include "nn_controller/amfcError.h"
#include "nn_controller/controller.h"
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

#include "nn_controller/amfc.h"
#include "nn_controller/options.h"
#include "nn_controller/predictor.h"

#define OUT(__o__) std::cout<< __o__ << std::endl;

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
        std::mutex mutex;
        bool updatePoseInfo, print, updateController;
        //these are ref model params
        Eigen::VectorXd tracking_error;
        std::chrono::time_point<std::chrono::high_resolution_clock> start, now;
        ros::NodeHandle n_;
        ros::Publisher pub, pred_pub_, control_pub_;
        double k;
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
        Eigen::VectorXd ref_;
        Eigen::MatrixXd Ky_hat, Kr_hat;
        Eigen::VectorXd pose_info;  //from sensor
        Eigen::MatrixXd expAmk, ym;         //reference model 

        /*Lambda is an unknown pos def symmetric matrix in R^{m x m}
        * matrix of
        */
        Eigen::MatrixXd Lambda;
        //adaptive control vector
        Eigen::VectorXd u_control;
        std::thread threads, gainsThread;
        //queue to delay the incoming pose message in order to pick delayed y(t-1)
        std::queue<Eigen::VectorXd> pose_queue;
        Eigen::VectorXd pred_ut;
        // trained net predictor input tuple
        nn_controller::predictor pred_;

        void initMatrices()
        {   
            m = 6; n = 6;
            Am.setIdentity(n, n);
            Bm.setIdentity(n, m);
            B.setIdentity(n, m);        //R^{n x m}
            Gamma_y.setIdentity(n, n); //will be 3 X 3 matrix
            Gamma_r.setIdentity(n, n); //will be 3 X 3 matrix
            Lambda.setIdentity(n, n);            
            P.setIdentity(n, n); // R ^{n x n}
            expAmk.setIdentity(n, n);

            P *= -1705./2668; //-0.639055472264; //
            Am *= -1334./1705;

            pose_info.resize(6);

            //gamma scaling factor for adaptive gains
            k = 1e-6;

            Gamma_y *= k;
            Gamma_r *= k;

            OUT("P Matrix: \n " << P);
            OUT("\nB Matrix: \n" << B);
            OUT("\nGamma_y Matrix: \n" << Gamma_y);
            OUT("\nGamma_r Matrix: \n" << Gamma_r);
            OUT("\nref_ : \n" << ref_);
        }

    public:
        // Constructor.
        Controller(ros::NodeHandle nc, 
                    const Eigen::VectorXd& ref);
        Controller();
        // Destructor.
        virtual ~Controller();
        // Update the controller (take an action).
        // virtual void update(RobotPlugin *plugin, ros::Time current_time, boost::scoped_ptr<Sample>& sample, Eigen::VectorXd &torques) = 0;
        // Configure the controller.

        //pose callback
        void getRefTraj();
        void pose_subscriber(const ensenso::HeadPose& headPose);
        void ControllerParams(Eigen::VectorXd&& pose_info);
        void NetPredictorInput(Eigen::VectorXd&& pose_info);
        //controller service
        bool configure_controller(
            nn_controller::controller::Request  &req,
            nn_controller::controller::Response  &res);
        //error service
        bool configure_error(
            nn_controller::amfcError::Request  &req,
            nn_controller::amfcError::Response  &res);
        void getPoseInfo(const ensenso::HeadPose& headPose, Eigen::VectorXd pose_info);
        ros::Time getTime();
        // Set update delay on the controller.
        virtual void set_update_delay(double new_step_length);
        // Get update delay on the controller.
        virtual double get_update_delay();
        // Check if controller is finished with its current task.
        // virtual bool is_finished() const = 0;
        // Reset the controller -- this is typically called when the controller is turned on.
        virtual void reset(ros::Time update_time);
        //subscribe to the reference model parameters
        virtual void ref_model_subscriber(const std_msgs::String::ConstPtr& ref_model_params);
        virtual void ref_model_multisub(const std_msgs::Float64MultiArray::ConstPtr& ref_model_params);
        // virtual void head_twist_subscriber(const geometry_msgs::Twist::ConstPtr& headPose);
    };
}

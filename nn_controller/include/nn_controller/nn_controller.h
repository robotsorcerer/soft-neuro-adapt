/*
Base class for a controller. Controllers take in sensor readings and choose the action.
*/
#pragma once

// Headers.
#include <boost/scoped_ptr.hpp>
#include <geometry_msgs/Vector3.h>
#include <ros/ros.h>
#include <time.h>
#include <chrono>
#include <vector>
#include <queue>
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
        //these are ref model params
        double error;
        std::chrono::time_point<std::chrono::high_resolution_clock> start, now;
        ros::NodeHandle n_;
        ros::Publisher pub;
        size_t k;
        int m, n;
        /* @brief P is symmetric positive definite matrix obtained from the lyapunov functiuon Q
        *
        *  B is the matrix that maps the controls into the state space
        */
        Eigen::MatrixXd P, B; 
        /*
        * p = # states of A matrix
        * Gammas are in R^{px p} diagonal, positive definite 
        * matrices of adaptive gains
        */
        Eigen::MatrixXd Gamma_y, Gamma_r;
        Eigen::Vector3d ref_;
        Eigen::VectorXd Ky, Kr;

        /*Lambda is an unknown pos def symmetric matrix in R^{m x m}
        * matrix of
        */
        Eigen::MatrixXd Lambda;

        void initMatrices()
        {   
            m = 6; n = 6;
            B.setIdentity(n, m);        //R^{n x m}
            Gamma_y.setIdentity(n, n); //will be 3 X 3 matrix
            Gamma_r.setIdentity(n, n); //will be 3 X 3 matrix
            Lambda.setIdentity(n, n);
            
            double p_elem = -0.639055472264; //-1705/2668; 
            P.setIdentity(n, n); // R ^{n x n}
            for(int i = 0; i < n; ++i)
            {
                P(i, i) = p_elem;
            }

            OUT("P Matrix: \n " << P);
            OUT("\nB Matrix: \n" << B);
            OUT("\nGamma_y Matrix: \n" << Gamma_y);
            OUT("\nGamma_r Matrix: \n" << Gamma_r);
            OUT("\nref_ : \n" << ref_);
            OUT("\np_elem : \n" << p_elem);
        }

    public:
        // Constructor.
        Controller(ros::NodeHandle nc, 
                    const Eigen::Vector3d& ref);
        Controller();
        // Destructor.
        virtual ~Controller();
        // Update the controller (take an action).
        // virtual void update(RobotPlugin *plugin, ros::Time current_time, boost::scoped_ptr<Sample>& sample, Eigen::VectorXd &torques) = 0;
        // Configure the controller.

        //pose callback
        void getRefTraj();
        void pose_subscriber(const ensenso::HeadPose& headPose);
        void ControllerParams( );
        bool configure_controller(
            nn_controller::amfcError::Request  &req,
            nn_controller::amfcError::Response  &res);
        void getPoseInfo(const ensenso::HeadPose& headPose, 
                                    Eigen::VectorXd pose_info);
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

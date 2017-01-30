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

namespace amfc_control
{

// Forward declarations.
class Sample;
class RobotPlugin;

class Controller
{
private:
    //these from /vicon/headtwist topic
    geometry_msgs::Vector3 linear, angular;
    //these are ref model params
    double error, k_m, a_m, y_m, y_0, ref, T;
    std::chrono::time_point<std::chrono::high_resolution_clock> start, now;
    ros::NodeHandle n_;
    ros::Publisher pub;
    size_t k;

public:
    // Constructor.
    Controller(ros::NodeHandle n, amfc_control::ActuatorType base_bladder, int ref);
    Controller();
    // Destructor.
    virtual ~Controller();
    // Update the controller (take an action).
    // virtual void update(RobotPlugin *plugin, ros::Time current_time, boost::scoped_ptr<Sample>& sample, Eigen::VectorXd &torques) = 0;
    // Configure the controller.
    bool configure_controller(
        nn_controller::amfcError::Request  &req,
        nn_controller::amfcError::Response  &res);
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
    virtual void head_twist_subscriber(const geometry_msgs::Twist::ConstPtr& headPose);
};

}

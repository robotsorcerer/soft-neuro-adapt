
#include <std_msgs/String.h>
#include <std_msgs/Float64MultiArray.h>
#include <ensenso/HeadPose.h>
#include "nn_controller/amfcError.h"
#include "nn_controller/nn_controller.h"
#include <string>
#include <cmath>

using namespace amfc_control;

//constructor
Controller::Controller(ros::NodeHandle nc, const Eigen::Vector3d& ref)
: n_(nc), ref_(ref)
{	 		 
	initMatrices();
}

//copy constructor
Controller::Controller()
{
	start = std::chrono::high_resolution_clock::now();
}

// Destructor.
Controller::~Controller()
{
}

void Controller::set_update_delay(double new_step_length)
{
}

double Controller::get_update_delay()
{
    return 1.0;
}

void Controller::reset(ros::Time update_time)
{
}

void Controller::pose_subscriber(const ensenso::HeadPose& headPose)
{
	Eigen::VectorXd pose_info;

	pose_info.resize(6);		//CRITICAL OR SEGFAULT

	getPoseInfo(headPose, pose_info);
	
	OUT("\npose_info: \n" << pose_info);

	ControllerParams();
}

void Controller::getPoseInfo(const ensenso::HeadPose& headPose, Eigen::VectorXd pose_info)
{
	pose_info << headPose.x, headPose.y,
				 headPose.z, headPose.pitch,
				 headPose.yaw, 0.0;   //setting roll to zero
				 
}
bool Controller::configure_controller(
	nn_controller::amfcError::Request  &req,
	nn_controller::amfcError::Response  &res)
{
	// this is the reference model. it is always time-varying1
	// k_m = 1.25;  
	// a_m = -0.782404601/k;   
	// y_0 = 0;  //assume zero initial response
 //    now = std::chrono::high_resolution_clock::now();
 //    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count() / 1000.0;
	// y_m = k_m * ref * exp(-a_m *k*T);

	// ROS_INFO_STREAM(" ym: " << y_m);
	// //parametric error between ref. model and plant
	// error = linear.z - y_m;
	// res.error = error;
	// //publish am, km, ref and y as services as well
	// res.am 	= a_m;
	// res.km 	= k_m;
	// res.ref = ref;
	// res.y  	= linear.z;
	// ROS_INFO_STREAM("sending response: " << res);	
	// ++k;

	return true;
}

void Controller::ref_model_subscriber(const std_msgs::String::ConstPtr& ref_model_params)
{
	std::string model_params = ref_model_params->data.c_str();
	ROS_INFO_STREAM("Model Parameters: " << model_params); 
}

void Controller::ref_model_multisub(const std_msgs::Float64MultiArray::ConstPtr& ref_model_params)
{
	// float model_params[10]
	std::vector<double> model_params = ref_model_params->data;
	std::vector<double>::iterator iter;
	for(iter = model_params.begin(); iter!=model_params.end(); ++iter)
	{
		std::cout << "\nmodel parameters are: \n" <<
					*iter << " "; 
	}  	
	Eigen::VectorXf modelParamVector;

}

void Controller::ControllerParams()
{	
	// Ky = -1* Gamma_y * pose_info * error * P * B *
}

void help()
{
	std::cout << "\t Enter the 3DOF desired trajectory separated by a single space" 
				<< std::endl;
}

int main(int argc, char** argv)
{ 
	ros::init(argc, argv, "controller_node", ros::init_options::AnonymousName);
	ros::NodeHandle n;

	Eigen::Vector3f ref;
	if(argc < 2) help();
	for(int i = 0; i < argc; ++i)
	{
		ref << atof(argv[1]), atof(argv[2]), atof(argv[3]);
	}

	Eigen::Vector3d refd; 
	refd = ref.cast<double>();
	Controller c(n, refd);

    ros::Subscriber sub = n.subscribe("/saved_net", 1000, &Controller::ref_model_subscriber, &c );
    ros::Subscriber sub_multi = n.subscribe("/multi_net", 1000, &Controller::ref_model_multisub, &c );
	ros::Subscriber sub_pose = n.subscribe("/mannequine_head/pose", 100, &Controller::pose_subscriber, &c);	
	ros::ServiceServer service = n.advertiseService("/error_srv", &Controller::configure_controller, &c);

	ros::spin();

	ros::shutdown();

	return 0;
}

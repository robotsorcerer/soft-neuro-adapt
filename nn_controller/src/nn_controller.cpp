
#include <std_msgs/String.h>
#include <std_msgs/Float64MultiArray.h>
#include <geometry_msgs/Twist.h>
#include "nn_controller/amfcError.h"
#include "nn_controller/nn_controller.h"
#include <string>

using namespace amfc_control;

//constructor
// default Constructors.
Controller::Controller(ros::NodeHandle n, amfc_control::ActuatorType base_bladder, int ref)
: n_(n), ref(ref), k(1), T(1/(2.2*30))
{
	
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

void Controller::head_twist_subscriber(const geometry_msgs::Twist::ConstPtr& headPose)
{
	linear.x = headPose->linear.x;
	linear.y = headPose->linear.y;
	linear.z = headPose->linear.z;
	angular.x = headPose->angular.x;
	angular.y = headPose->angular.y;
	angular.z = headPose->angular.z;
}

bool Controller::configure_controller(
	nn_controller::amfcError::Request  &req,
	nn_controller::amfcError::Response  &res)
{
	// this is the reference model. it is always time-varying1
	k_m = 1.25;  
	a_m = -0.782404601/k;   
	y_0 = 0;  //assume zero initial response
    now = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count() / 1000.0;
	y_m = k_m * ref * exp(-a_m *k*T);

	ROS_INFO_STREAM(" ym: " << y_m);
	//parametric error between ref. model and plant
	error = linear.z - y_m;
	res.error = error;
	//publish am, km, ref and y as services as well
	res.am 	= a_m;
	res.km 	= k_m;
	res.ref = ref;
	res.y  	= linear.z;
	ROS_INFO_STREAM("sending response: " << res);	
	++k;

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

int main(int argc, char** argv)
{ 
	ros::init(argc, argv, "controller_node", ros::init_options::AnonymousName);
	ros::NodeHandle n;
	amfc_control::ActuatorType at = base_bladder;
	int ref = atoi(argv[1]);
	Controller c(n, base_bladder, ref);

    ros::Subscriber sub = n.subscribe("/saved_net", 1000, &Controller::ref_model_subscriber, &c );
    ros::Subscriber sub_multi = n.subscribe("/multi_net", 1000, &Controller::ref_model_multisub, &c );
	ros::Subscriber sub_twist = n.subscribe("/vicon/headtwist", 1, &Controller::head_twist_subscriber, &c);	
	ros::ServiceServer service = n.advertiseService("/error_srv", &Controller::configure_controller, &c);

	while(ros::ok())
	{		
		ros::spinOnce();
		ros::Rate r(30);
		r.sleep();
	}

	ros::shutdown();

	return 0;
}

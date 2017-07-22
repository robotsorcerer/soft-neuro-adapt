
#include <string>
#include <fstream>
#include <exception>
#include <std_msgs/String.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/Pose.h>
#include <ensenso/pathfinder.h>
#include <std_msgs/Float64MultiArray.h>
#include "nn_controller/nn_controller.h"

#include <boost/numeric/odeint/stepper/runge_kutta4.hpp>
#include <boost/numeric/odeint/stepper/runge_kutta_dopri5.hpp>
#include <boost/numeric/odeint/algebra/vector_space_algebra.hpp>

using namespace amfc_control;
using namespace boost::numeric::odeint;

//constructor
Controller::Controller(ros::NodeHandle nc, const Eigen::Vector3d& ref, bool print,
						bool useSigma, bool save)
: n_(nc), ref_(ref), updatePoseInfo(false), updateController(false), useSigma(useSigma),
  save(save), resetController(false), updateWeights(false), print(print), counter(0), 
  multicast_address("235.255.0.1")
{	 		 
	initMatrices();	
	sigma_y = 0.01;
	sigma_r = 0.01;
	control_pub_ = n_.advertise<ensenso::ValveControl>("/mannequine_head/u_valves", 100);
}

//copy constructor
Controller::Controller()
{

}

// Destructor.
Controller::~Controller()
{
}

ros::Time Controller::getTime() {
	return ros::Time::now();
}

/*Subscribers*/
// pose subscriber from ensenso_seg/vicon_sub
void Controller::pose_subscriber(const geometry_msgs::Pose& headPose) {
	getPoseInfo(headPose, pose_info);

	std::lock_guard<std::mutex> pose_locker(pose_mutex);
	this->pose_info = pose_info;
	updatePoseInfo  = true;		
}

void Controller::net_control_subscriber(const ensenso::ValveControl& net_control_law){
	Eigen::VectorXd net_control;
	net_control.resize(6);

	net_control << net_control_law.left_bladder_pos, net_control_law.left_bladder_neg,
				   net_control_law.right_bladder_pos, net_control_law.right_bladder_neg, 
				   net_control_law.base_bladder_pos, net_control_law.base_bladder_neg;
	this->net_control = net_control;				   
}

void Controller::bias_sub(const std_msgs::Float64MultiArray::ConstPtr& bias_params){
	std::vector<double> model_params = bias_params->data;
	Eigen::VectorXd modelBiases;
	modelBiases.resize(6);

	std::lock_guard<std::mutex> biases_lock(biases_mutex);
	for(auto i = 0; i < 6; ++i){
		modelBiases(i) = model_params[i];
	}

	this->modelBiases  = modelBiases;
	updateBiases	   = true;
}

//net weights subscriber from sample.lua in RAL/farnn
void Controller::weights_sub(const std_msgs::Float64MultiArray::ConstPtr& ref_model_params)
{
	std::vector<double> model_params = ref_model_params->data;

	//retrieve the pre-trained weights and biases 
	Eigen::Matrix<double, 6, 6> modelWeights;
	std::lock_guard<std::mutex> weights_lock(weights_mutex);
	int k = 0;
	for(auto i = 0; i < 6; ++i){
		for(auto j = 0; j < 6; ++j){
			modelWeights(i,j) = model_params[k];
			++k;
		}
	}

	this->modelWeights = modelWeights;
	updateWeights = true;
}

void Controller::getPoseInfo(const geometry_msgs::Pose& headPose, Eigen::VectorXd pose_info)
{
	pose_info << headPose.position.z,// 1,  		//roll to zero
				 headPose.orientation.x, headPose.orientation.y; //headPose.yaw;   //setting roll to zero
	
	this->pose_info = pose_info;
	//set ref's non-controlled states to measurement
	ControllerParams(std::move(pose_info));
}

void Controller::ControllerParams(Eigen::VectorXd&& pose_info)
{	
	// Am = -0.782405        -0        -0
	//       -0          -0.782405     -0
    //       -0              -0    -0.782405
	// Bm = [1 0 0; 0 1 0; 0 0 1]
	// ref_ = [z, roll, pitch ] given by user
	Am.resize(3,3); 	Bm.resize(3,3); 	ym.resize(3); 	ym_dot.resize(3); tracking_error.resize(3);
	if(counter == 0){
		ym = pose_info;
	}
	ym_dot = Am * ym + Bm * ref_;

	if(counter == 0){
		ym = pose_info;		
		prev_ym.push_back(ym);
	}
	else{
		ym_dot = Am * prev_ym.back() + Bm * ref_;
		ym = prev_ym.back() + 0.01 * ym_dot;
	}
	prev_ym.push_back(ym);
	//compute tracking error, e = y - y_m
	tracking_error = pose_info - ym;
	// //use boost ode solver
	runge_kutta_dopri5<state,double,state,double,vector_space_algebra> stepper;
	Ky_hat_dot = -Gamma_y * pose_info * tracking_error.transpose() * P * B  * sgnLambda;
	Kr_hat_dot = -Gamma_r * ref_      * tracking_error.transpose() * P * B  * sgnLambda;

	//use reference for the derivative
	stepper.do_step([](const state& x, state & dxdt, const double t)->void{
		dxdt = x;
	}, Ky_hat_dot, counter, Ky_hat, 0.01);
	//integrate Ky_hat_dot
	stepper.do_step([](const state& x, state & dxdt, const double t)->void{
		dxdt = x;
	}, Kr_hat_dot, counter, Kr_hat, 0.01);

	Eigen::VectorXd pred;
	pred.resize(6);
	if(!updatePred)	{
		pred << pose_info(0), pose_info(0), // note the scaling since two valves do one job in equal but opposite directions
				pose_info(1), pose_info(1),
				pose_info(2), pose_info(2);
	}
	else	{
		std::lock_guard<std::mutex> net_pred_locker(pred_mutex);
		updatePred = false;
		pred = this->pred;
	}

	// Get model biases and weights in real time
	// get weights
	Eigen::Matrix<double, 6, 6> modelWeights;  // has 36 members
	if(updateWeights){
		std::lock_guard<std::mutex> weights_lock (weights_mutex);
		modelWeights = this->modelWeights;
		updateWeights = false;
	}

	//get biases
	Eigen::VectorXd modelBiases;
	modelBiases.resize(6);
	if(updateBiases)	{
		std::lock_guard<std::mutex> biases_lock (biases_mutex);
		modelBiases = this->modelBiases;
		updateBiases = false;
	}

	/*
	* Calculate Control Law
	*/
	u_control = (Ky_hat.transpose() * pose_info) + 
				(Kr_hat.transpose() * ref_) + this->net_control; 

	u_control(0) /= 322;
	u_control(1) /= 322;
	std::lock_guard<std::mutex> lock(mutex);
	{
		this->u_control = u_control;
	}

	resetController = true;

	std::string filename;
	ros::param::get("/nn_controller/Utils/filename", filename);

	pathfinder::getROSPackagePath("nn_controller", nn_controller_path_);
	ss << nn_controller_path_.c_str() << filename;
	std::string ref_pose_file = ss.str();

	if(save) {
		std::ofstream file_handle;
		file_handle.open(ref_pose_file, /*std::fstream::in |*/ std::ofstream::out | std::ofstream::app);

		file_handle  << ref_(0) <<"\t" <<ref_(1) << "\t" << ref_(2) << "\t" <<
					pose_info(0) <<"\t" <<pose_info(1) << "\t" << pose_info(2) << "\n"; 

		file_handle.close();
	}

	// changed the order here because I rearranged hardware two days before cam ready
	u_valves_.left_bladder_pos  = u_control(0);
	u_valves_.left_bladder_neg  = u_control(1);
	u_valves_.right_bladder_pos = u_control(2);
	u_valves_.right_bladder_neg = u_control(3);
	u_valves_.base_bladder_pos  = u_control(4);
	u_valves_.base_bladder_neg  = u_control(5);

	control_pub_.publish(u_valves_);
	// convert from eigen to headpose
	vectorToHeadPose(std::move(pose_info), pose_);
	// fallback since rosrio is messing up
	udp::sender s(io_service, boost::asio::ip::address::from_string(multicast_address), 
	        	  u_valves_, ref_, pose_);

	if(print)	{	
		OUT("\nref_: " 			<< ref_.transpose());
		OUT("y  (z, pitch, roll): " 		 << pose_info.transpose());
		OUT("ym (z, pitch, roll): " 		 << ym.transpose());
		OUT("e  (y-ym): " << tracking_error.transpose());
		OUT("pred (z, z, pitch, pitch, roll, roll): " << pred.transpose());
		OUT("net_control: " << net_control.transpose());
		OUT("Control Law: " << u_control.transpose());
		ROS_INFO_STREAM("\nKr_hat^T: \n" << Kr_hat.transpose());
		ROS_INFO_STREAM("\nKy_hat^T: \n" << Ky_hat.transpose());

	}
	++counter;
}

void Controller::vectorToHeadPose(Eigen::VectorXd&& pose_info, geometry_msgs::Pose& eig2Pose)
{
    eig2Pose.position.z = pose_info(0);
    eig2Pose.orientation.x = pose_info(1);
    eig2Pose.orientation.y = pose_info(2);
}

void help()
{
	OUT("Add the 3DOF desired trajectory separated by a single space");
	OUT("Like so: rosrun nn_controller nn_controller <z> <pitch> <yaw>" << 
			 "\n[<print> <useSigma> <save>]");
	OUT("where the last three arguments are optional");
	OUT("to print, use \"1\" in place of <print> etc");
}

int main(int argc, char** argv)
{ 
	ros::init(argc, argv, "controller_node", ros::init_options::AnonymousName);
	ros::NodeHandle n;
	bool print, useSigma, save, useVicon(true);

	help();

	Eigen::Vector3d ref;
	ref.resize(3);

	try{		
		//supply values from the cmd line or retrieve them 
		//from the ros parameter server
		n.getParam("/nn_controller/Reference/z", ref(0));    	//ref z
		n.getParam("/nn_controller/Reference/pitch", ref(1));	//ref pitch
		n.getParam("/nn_controller/Reference/roll", ref(2));	    //ref roll
		n.getParam("/nn_controller/Utils/print", print);
		n.getParam("/nn_controller/Utils/useSigma", useSigma);
		save = n.getParam("/nn_controller/Utils/save", save);
	}
	catch(std::exception& e){
		e.what();
	}

	Controller c(n, ref, print, useSigma, save);

    ros::Subscriber sub_weights = n.subscribe("/mannequine_pred/net_weights", 1000, &Controller::weights_sub, &c );
    ros::Subscriber sub_bias  = n.subscribe("/mannequine_pred/net_biases", 100, &Controller::bias_sub, &c);
	ros::Subscriber sub_pose = n.subscribe("/mannequine_head/pose", 100, &Controller::pose_subscriber, &c);	
	ros::Subscriber sub_pred = n.subscribe("/mannequine_pred/preds", 100, &Controller::net_control_subscriber, &c);
	ros::spin();

	ros::shutdown();

	return 0;
}

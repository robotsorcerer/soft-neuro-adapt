
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
	// BladderTypeEnum bladder_type_;
	ros::param::get("/nn_controller/Control/with_net", with_net_);
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


void Controller::vectorToHeadPose(Eigen::VectorXd&& pose_info, geometry_msgs::Pose& eig2Pose)
{
    eig2Pose.orientation.x = pose_info(0); // roll
    eig2Pose.position.z = pose_info(1);	// z
    eig2Pose.orientation.y = pose_info(2);	// pitch
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
				   net_control_law.base_bladder_pos, net_control_law.base_bladder_neg,
				   net_control_law.right_bladder_pos, net_control_law.right_bladder_neg;
	update_net_law = true;
	this->net_control.resize(6);
	this->net_control = net_control;				   
}


void Controller::getPoseInfo(const geometry_msgs::Pose& headPose, Eigen::VectorXd pose_info)
{
	pose_info << headPose.orientation.x, // roll = [left and right]
				 headPose.position.z, 	// base  = [base actuator]	
				 headPose.orientation.y; // pitch = [right actuator]
	
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
	if(counter == 0){
		ym = pose_info;		// will be 3x1; pose_info is also 3x1
		ym_dot = Am * ym + Bm * ref_;	// will be 3x1
		prev_ym.push_back(ym);

		tracking_error = pose_info - ym; 	// will be 3x1


		ros::param::get("/nn_controller/Utils/filename", filename_);
		pathfinder::getROSPackagePath("nn_controller", nn_controller_path_);
		ss << nn_controller_path_.c_str() << filename_;
		ref_pose_file_ = ss.str();
	}
	else{
		// find ym
		ym_dot = Am * prev_ym.back() + Bm * ref_;  // will be 3x1
		ym = prev_ym.back() + 0.01 * ym_dot;		// will be 3x1
		prev_ym.push_back(ym);    // don't leve the linked list empty

		tracking_error = pose_info - ym;			// will be 3x1

		// // find Ky_hat 
		// Ky_hat_dot = -Gamma_y * pose_info * tracking_error.transpose() * P * B  * sgnLambda;
		// Ky_hat = prev_Ky_hat_.back() + 0.01 * Ky_hat_dot;
		// prev_Ky_hat_.push_back(Ky_hat);

		// // find Kr_hat
		// Kr_hat_dot = -Gamma_r * ref_      * tracking_error.transpose() * P * B  * sgnLambda;
		// Kr_hat = prev_Kr_hat_.back() + 0.01 * Kr_hat_dot;
		// prev_Kr_hat_.push_back(Kr_hat);
	}
	// tracking_error = pose_info - ym;

	//use boost ode solver to compute Ky_hat and Kr_hat:: lot more stable than trapezoidal rule
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

	Eigen::VectorXd net_control;
	net_control.resize(m);
	if(update_net_law)	{
		std::lock_guard<std::mutex> net_pred_locker(pred_mutex);
		update_net_law = false;
		net_control = this->net_control;
	}
	/*
	* Calculate Control Law
	*/
	if(with_net_){
		u_control = (Ky_hat.transpose() * pose_info) + 
					(Kr_hat.transpose() * ref_) + this->net_control; 
	}
	else{
		u_control = (Ky_hat.transpose() * pose_info) + 
					(Kr_hat.transpose() * ref_); 	
	}
	// u_control = this->net_control;
	/*
	Here are the rules that govern the bladders
	l_i --> Roll+	r_i -->Roll-
	l_o --> Roll-   r_o -->Roll+
	b_i --> Pitch+, Z+   b_o -->Pitch-, Z-
	u_{o+} is a suitable controller magnitude; u_+ is a +ve input
	------------------------------------------------------------
	DOF     |  Control Law
	------------------------------------------------------------
	Roll+   |  if f_{li} = u_{+}:
			|	f_{ri} = 0,
			|   f_{lo} = |u_{lo}|
			| 	f_{ro} = |u_{ro}|
	------------------------------------------------------------
	Roll-   | if f_{ri} = u_{+}:
			|  	f_{li} = 0; 
			|   f_{ro} = 0  or u_{o+}
	------------------------------------------------------------
	Pitch+  | if f_{bi} = u_{+}
			|    f_{bo} = u_{o+} and f_{li} =f_{ri} = u_{head+}
	------------------------------------------------------------
	Pitch-  | f_{bo} = u_{max-}
			| f_{bi} = 0 or < f_{bo}
	------------------------------------------------------------
	Z+      | f_{li} = f_{ri} = u_{head+}
			| f_{bi} = u_{+} f_{bo} = u_{o+}
	------------------------------------------------------------
	Z-      | f_{bo} = u_{-}
			| f_{bi} = 0 or f_{bo}
	*/
	// // saturate control signals
	// for(auto i = 0; i < 6; ++i){
	// 	if(u_control[i] < 0)
	// 		u_control[i] = 0;
	// 	else if (u_control[i] > 1)
	// 		u_control[i] = 1;
	// 	else
	// 		u_control[i] = u_control[i];
	// }
	u_valves_.left_bladder_pos  = u_control(0);
	u_valves_.left_bladder_neg  = u_control(1);
	u_valves_.base_bladder_pos  = u_control(2);
	u_valves_.base_bladder_neg  = u_control(3);	
	u_valves_.right_bladder_pos = u_control(4);
	u_valves_.right_bladder_neg = u_control(5);	

	if(save) {
		std::ofstream file_handle;
		file_handle.open(ref_pose_file_, std::ofstream::out | std::ofstream::app);
		file_handle  << ref_(0) <<"\t" <<ref_(1) << "\t" << ref_(2) << "\t" << pose_info(0) <<"\t" <<pose_info(1) << "\t" << pose_info(2) << "\n"; 
		file_handle.close();
	}
	ros::Rate sleeper(2);
	sleeper.sleep();
	control_pub_.publish(u_valves_);
	vectorToHeadPose(std::move(pose_info), pose_);	// convert from eigen to headpose
	udp::sender s(io_service, boost::asio::ip::address::from_string(multicast_address), u_valves_, ref_, pose_);
	// pose is  [roll, z, pitch]
	// udp::sender s(io_service, boost::asio::ip::address::from_string(multicast_address), pose_); // used for identification

	if(print)	{	
		OUT("\nref_: " 			<< ref_.transpose());
		OUT("y  (roll, z,  pitch): " 		 << pose_info.transpose());
		OUT("ym (roll, z,  pitch): " 		 << ym.transpose());
		OUT("e  (y-ym): " << tracking_error.transpose());
		OUT("pred (z, z, pitch, pitch, roll, roll): " << pred.transpose());
		OUT("net_control: " << net_control.transpose());
		OUT("Control Law: " << u_control.transpose());
		OUT("Kr_hat: \n" << Kr_hat);
		OUT("Ky_hat: \n" << Ky_hat);
	}
	++counter;
}

int main(int argc, char** argv)
{ 
	ros::init(argc, argv, "controller_node", ros::init_options::AnonymousName);
	ros::NodeHandle n;
	bool print, useSigma, save, useVicon(true);

	Eigen::Vector3d ref;
	ref.resize(3);

	try{		
		//supply values from the cmd line or retrieve them 
		//from the ros parameter server
		n.getParam("/nn_controller/Reference/z", ref(1));    	//ref z
		n.getParam("/nn_controller/Reference/pitch", ref(2));	//ref pitch
		n.getParam("/nn_controller/Reference/roll", ref(0));	    //ref roll
		n.getParam("/nn_controller/Utils/print", print);
		n.getParam("/nn_controller/Utils/useSigma", useSigma);
		save = n.getParam("/nn_controller/Utils/save", save);
	}
	catch(std::exception& e){
		e.what();
	}

	Controller c(n, ref, print, useSigma, save);

	ros::Subscriber sub_pose = n.subscribe("/mannequine_head/pose", 100, &Controller::pose_subscriber, &c);	
	ros::Subscriber sub_pred = n.subscribe("/mannequine_pred/preds", 100, &Controller::net_control_subscriber, &c);
	ros::spin();

	if(!ros::ok())
		ros::shutdown();

	return 0;
}

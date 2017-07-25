
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
	pathfinder::getROSPackagePath("nn_controller", nn_controller_path_);
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
	Eigen::VectorXd pose_info;
	getPoseInfo(headPose, pose_info);

	std::lock_guard<std::mutex> pose_locker(pose_mutex);
	this->pose_info_ = pose_info;
	updatePoseInfo  = true;		
}

void Controller::net_control_subscriber(const ensenso::ValveControl& net_control_law){
	Eigen::VectorXd net_control;
	net_control.resize(n);

	net_control <<  net_control_law.left_bladder, 	// left
					net_control_law.base_bladder, 	// base
					net_control_law.right_bladder;	// right
	update_net_control_ = true;					
	this->net_control_ = net_control;				   
}

void Controller::getPoseInfo(const geometry_msgs::Pose& headPose, Eigen::VectorXd pose_info)
{
	pose_info << headPose.orientation.x, // roll = [left and right]
				 headPose.position.z, 	// base  = [base actuator]	
				  headPose.orientation.y; // pitch = [right actuator]
	
	this->pose_info_ = pose_info;
	//set ref's non-controlled states to measurement
	ControllerParams();
}

void Controller::ControllerParams()
{	
	// Am = -0.782405        -0        -0
	//       -0          -0.782405     -0
    //       -0              -0    -0.782405
	// Bm = [1 0 0; 0 1 0; 0 0 1]
	// ref_ = [z, roll, pitch ] given by user
	Am.resize(n, n); 	Bm.resize(n, m); 	ym.resize(n); 	ym_dot.resize(n); tracking_error_.resize(n);
	
	Eigen::VectorXd pose_info = this->pose_info_;

	if(counter == 0){
		ym = pose_info;		
		ym_dot = Am * ym + Bm * ref_;
		prev_ym.push_back(ym);
	}
	else{
		ym_dot = Am * prev_ym.back() + Bm * ref_;
		ym = prev_ym.back() + 0.01 * ym_dot;
	}
	//compute tracking error, e = y - y_m
	tracking_error_ = pose_info - ym;
	// //use boost ode solver for Ky_hat and Kr_hat
	runge_kutta_dopri5<state,double,state,double,vector_space_algebra> stepper;
	Ky_hat_dot = -Gamma_y * pose_info * tracking_error_.transpose() * P * B  * sgnLambda;
	Kr_hat_dot = -Gamma_r * ref_      * tracking_error_.transpose() * P * B  * sgnLambda;

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
	if(!update_net_control_)	{
		net_control << pose_info(0), // roll = [left and right actuator]
					   pose_info(1), // base  = [base actuator]
					   pose_info(2); // pitch = [right actuator]
	}
	else	{
		std::lock_guard<std::mutex> net_pred_locker(pred_mutex);
		update_net_control_ = false;
		net_control = this->net_control_;
	}
	/*
	* Calculate Control Law
	*/
	if(with_net_){
		u_control = (Ky_hat.transpose() * pose_info) + 
					(Kr_hat.transpose() * ref_) + net_control_; 
	}
	else{
		u_control = (Ky_hat.transpose() * pose_info) + 
					(Kr_hat.transpose() * ref_); 	
	}
	// u_control(0) /= 322;
	// u_control(1) /= 322;
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
	u_valves_.left_bladder  = u_control(0); // roll actuator +ve
	u_valves_.base_bladder  = u_control(1);	// base actuator
	u_valves_.right_bladder = u_control(2);	// roll actuator -ve

	bool roll_pos, roll_neg, pitch_pos, pitch_neg, z_pos, z_neg, all;
	if(ros::param::get("/nn_controller/DOF/ROLL_POS", roll_pos))
		this->dof_motion_type_ = DOF_MOTION_ENUM::ROLL_POS;
	if(ros::param::get("/nn_controller/DOF/ROLL_NEG", roll_neg))
		this->dof_motion_type_ = DOF_MOTION_ENUM::ROLL_NEG;
	if(ros::param::get("/nn_controller/DOF/PITCH_POS", pitch_pos))
		this->dof_motion_type_ = DOF_MOTION_ENUM::PITCH_POS;
	if(ros::param::get("/nn_controller/DOF/PITCH_NEG", pitch_neg))
		this->dof_motion_type_ = DOF_MOTION_ENUM::PITCH_NEG;
	if(ros::param::get("/nn_controller/DOF/Z_POS", z_pos))
		this->dof_motion_type_ = DOF_MOTION_ENUM::Z_POS;
	if(ros::param::get("/nn_controller/DOF/Z_NEG", z_neg))
		this->dof_motion_type_ = DOF_MOTION_ENUM::Z_NEG;
	if(ros::param::get("/nn_controller/DOF/ALL", all))
		this->dof_motion_type_ = DOF_MOTION_ENUM::ALL;

	switch (this->dof_motion_type_){
		case DOF_MOTION_ENUM::ROLL_POS:  	// controlled principlally by left inlet torque
			u_valves_.right_bladder_pos = 0; //std::fabs(u_valves_.right_bladder_pos);  // right inlet has to be zero
			u_valves_.right_bladder_neg = std::fabs(u_valves_.right_bladder_neg); // right outlet has to be positive
			u_valves_.left_bladder_neg  = 0; //std::fabs(u_valves_.left_bladder_neg); // left outlet has to be positive
			break;

		case DOF_MOTION_ENUM::ROLL_NEG:  // controlled principally by right inlet torque
			u_valves_.left_bladder_pos = 0; // dampen the negative torque
			u_valves_.left_bladder_neg = std::fabs(u_valves_.left_bladder_neg); // dampen the negative torque
			u_valves_.right_bladder_neg = 0;  // we should not excite right outlet bladder
			break;

		case DOF_MOTION_ENUM::PITCH_POS:  // this is controlled principally by the base bladder inlet
			u_valves_.base_bladder_neg = 0;  // we should not excite base outlet bladder
			break;

		case DOF_MOTION_ENUM::PITCH_NEG:	// this is controlled principally by the base bladder outlet
			u_valves_.base_bladder_pos = 0;  // we should not excite base outlet bladder
			break;

		case DOF_MOTION_ENUM::Z_POS:   // this is controlled principally by the base bladder
			u_valves_.base_bladder_neg = 0;  // we should not excite base outlet bladder
			break;

		case DOF_MOTION_ENUM::Z_NEG: // this is controlled principally by the base bladder outlet
			u_valves_.base_bladder_pos = 0;  // we should not excite base outlet bladder
			break;

		case DOF_MOTION_ENUM::ALL:
			if(u_valves_.left_bladder>0)
				u_valves_.right_bladder = 0;
			if (u_valves_.right_bladder > 0)
				u_valves_.left_bladder = 0;			
	}

	if(save) {
		std::ofstream file_handle;

		ros::param::get("/nn_controller/Utils/filename", filename_);
		ss << nn_controller_path_.c_str() << filename_;
		std::string ref_pose_file = ss.str();

		file_handle.open(ref_pose_file, /*std::fstream::in |*/ std::ofstream::out | std::ofstream::app);

		file_handle  << ref_(0) <<"\t" <<ref_(1) << "\t" << ref_(2) << "\t" <<
					pose_info(0) <<"\t" <<pose_info(1) << "\t" << pose_info(2) << "\n"; 

		file_handle.close();
	}

	control_pub_.publish(u_valves_);
	vectorToHeadPose(std::move(pose_info), pose_);	// convert from eigen to headpose
	udp::sender s(io_service, boost::asio::ip::address::from_string(multicast_address), u_valves_, ref_, pose_);

	if(print)	{	
		OUT("\nref_: " 			<< ref_.transpose());
		OUT("y  (z, roll, pitch): " 		 << pose_info.transpose());
		OUT("ym (z, roll, pitch): " 		 << ym.transpose());
		OUT("e  (y-ym): " << tracking_error_.transpose());
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

	ros::Subscriber sub_pose = n.subscribe("/mannequine_head/pose", 100, &Controller::pose_subscriber, &c);	
	ros::Subscriber sub_pred = n.subscribe("/mannequine_pred/preds", 100, &Controller::net_control_subscriber, &c);
	ros::spin();

	ros::shutdown();

	return 0;
}

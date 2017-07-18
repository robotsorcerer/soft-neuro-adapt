
#include <string>
#include <fstream>
#include <exception>
#include <std_msgs/String.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/Pose.h>
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

void Controller::bias_sub(const std_msgs::Float64MultiArray::ConstPtr& bias_params)
{
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
	pose_info << //headPose.x, headPose.y,
				 headPose.position.z,// 1,  		//roll to zero
				 headPose.orientation.x, headPose.orientation.y; //headPose.yaw;   //setting roll to zero
	
	this->pose_info = pose_info;
	//set ref's non-controlled states to measurement
	ControllerParams(std::move(pose_info));
}

//real-time predictor inputs service request response ::DEPRECATED
bool Controller::configure_predictor_params(
	nn_controller::predictor_params::Request  &req,
	nn_controller::predictor_params::Response  &res)
{
	Eigen::VectorXd u_control_local;
	if(updateController)	{
		u_control_local = this->u_control;
		updateController = false;
	}
	
	Eigen::Vector3d pose_info;
	pose_info.resize(3);
	pose_info = this->pose_info;

	if(updatePoseInfo)	{
		pose_info = this->pose_info;
		updatePoseInfo = false;
	}

	res.u1 		=	u_control_local(0);
	res.u2 		=	u_control_local(1);
	res.u3		=	u_control_local(2);
	res.u4		=	u_control_local(3);
	res.u5		=	u_control_local(4);
	res.u6		=	u_control_local(5);
	// measurements
	res.z		=	pose_info(0);
	res.pitch	=	pose_info(1);
	res.roll		=	pose_info(2);

	return true;
}

void Controller::pred_subscriber(const geometry_msgs::Pose& pred)
{
	std::lock_guard<std::mutex> net_pred_locker(pred_mutex);
	this->pred.resize(6);
	this->pred << pred.position.x, pred.position.y, pred.position.z,
				  pred.orientation.x, pred.orientation.y, pred.orientation.z;
	updatePred = true;
}

void Controller::loss_subscriber(const std_msgs::Float64& net_loss)
{
	std::lock_guard<std::mutex> net_loss_locker(net_loss_mutex);
	this->loss = net_loss.data;
	updateNetLoss = true;
}

void Controller::ControllerParams(Eigen::VectorXd&& pose_info)
{	
	tracking_error.resize(3);
	expAmk *= std::exp(Am(0,0)*counter);	
	//Bm is initialized to identity
	ym = Bm * expAmk * ref_ ;  //take laplace transform of ref model
	//compute tracking error, e = y - y_m
	tracking_error = pose_info - ym;

	if(useSigma){		
		Ky_hat_dot = -Gamma_y * ((pose_info * tracking_error.transpose() * P * B * sgnLambda) +
								 (sigma_y * Ky_hat));
		Kr_hat_dot = -Gamma_r * ((ref_      * tracking_error.transpose() * P * B * sgnLambda) +
								 (sigma_r * Kr_hat));
	}
	else{
		Ky_hat_dot = -Gamma_y * pose_info * tracking_error.transpose() * P * B  * sgnLambda;
		Kr_hat_dot = -Gamma_r * ref_      * tracking_error.transpose() * P * B  * sgnLambda;
	}
	//use boost ode solver
	runge_kutta_dopri5<state,double,state,double,vector_space_algebra> stepper;
	//use reference for the derivative
	stepper.do_step([](const state& x, state & dxdt, const double t)->void{
		dxdt = x;
	}, Ky_hat_dot, counter, Ky_hat, 0.01);
	//integrate Ky_hat_dot
	stepper.do_step([](const state& x, state & dxdt, const double t)->void{
		dxdt = x;
	}, Kr_hat_dot, counter, Kr_hat, 0.01);
	//will be [3x1]. We are integrating the second part of the rhs soln to the 
	//linear ref_ model
	//retrieve net predictions
	Eigen::VectorXd pred;
	pred.resize(6);
	if(!updatePred)	{
		pred << pose_info(0), pose_info(0), 
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
	Eigen::Matrix<double, 6, 6> modelWeights;
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
	Eigen::VectorXd laggedVector;
	laggedVector.resize(6);
	Eigen::VectorXd wgtsPred;
	wgtsPred.resize(6); //will be regressor vector from neural network
	u_control = (Ky_hat.transpose() * pose_info) + 
				(Kr_hat.transpose() * ref_)  + pred; // + wgtsPred;  //

	u_control = u_control.cwiseAbs();
	std::lock_guard<std::mutex> lock(mutex);
	{
		this->u_control = u_control;
	}

	if(print)
	{	
		OUT("\nref_: " 			<< ref_.transpose());
		OUT("pose: " 		 << pose_info.transpose());
		OUT("pred: " << pred.transpose());

		OUT("tracking_error: " << tracking_error.transpose());
		OUT("Control Law: " << u_control.transpose());
	}
	resetController = true;

	if(save) {
		std::ofstream midface;
		midface.open("ref_pose.csv", std::ofstream::out | std::ofstream::app);
		midface << ref_(0) <<"\t" <<ref_(1) << "\t" << ref_(2) << "\t" <<
				pose_info(0) <<"\t" <<pose_info(1) << "\t" << pose_info(2) << "\n"; 
		midface.close();
	}

	u_valves_.left_bladder_pos  = u_control(0);
	u_valves_.left_bladder_neg  = u_control(1);
	u_valves_.right_bladder_pos  = u_control(2);
	u_valves_.right_bladder_neg = u_control(3);
	u_valves_.base_bladder_pos = u_control(4);
	u_valves_.base_bladder_neg = u_control(5);

	control_pub_.publish(u_valves_);
	//convert from eigen to headpose
	vectorToHeadPose(std::move(pose_info), pose_);
	//fallback since rosrio is messing up
	udp::sender s(io_service, boost::asio::ip::address::from_string(multicast_address), 
	        	  u_valves_, ref_, pose_);
	++counter;
}

//currently unused
bool Controller::configure_controller(
	nn_controller::controller::Request  &req,
	nn_controller::controller::Response  &res)
{
	Eigen::VectorXd u_control_local;
	if(updateController)
	{
		u_control_local = this->u_control;
		updateController = false;
	}

	res.left_in 	=	u_control_local(0);
	res.left_out 	=	u_control_local(1);
	res.right_in	=	u_control_local(2);
	res.right_out	=	u_control_local(3);
	res.base_in		=	u_control_local(4);
	res.base_out	=	u_control_local(5);

	return true;
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
		n.getParam("/nn_controller/Reference/", ref(1));	//ref pitch
		n.getParam("/nn_controller/Reference/", ref(2));	    //ref roll
		// if(atoi(argv[4]) == 1)
		n.getParam("/nn_controller/Utils/print", print);
		// if(atoi(argv[5]) == 1)
		n.getParam("/nn_controller/Utils/useSigma", useSigma);
		// if(atoi(argv[6]) == 1)
		save = n.getParam("/nn_controller/Utils/save", save);
	}
	catch(std::exception& e){
		e.what();
	}


	// Eigen::Vector3d refd;
	// refd = ref.cast<double>(); 
	Controller c(n, ref, print, useSigma, save);

    ros::Subscriber sub_weights = n.subscribe("/mannequine_pred/net_weights", 1000, &Controller::weights_sub, &c );
    ros::Subscriber sub_bias  = n.subscribe("/mannequine_pred/net_biases", 100, &Controller::bias_sub, &c);
	ros::Subscriber sub_vicon = n.subscribe("/mannequine_head/pose", 100, &Controller::pose_subscriber, &c);	
	//subscribe to real -time predictor parameters
	ros::Subscriber sub_pred = n.subscribe("/mannequine_pred/preds", 100, &Controller::pred_subscriber, &c);
	ros::Subscriber sub_loss = n.subscribe("/mannequine_pred/net_loss", 100, &Controller::loss_subscriber, &c);
	ros::ServiceServer control_serv = n.advertiseService("/mannequine_head/controller", &Controller::configure_controller, &c);
	ros::spin();

	ros::shutdown();

	return 0;
}

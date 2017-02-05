
#include <string>
#include <exception>
#include <std_msgs/String.h>
#include <ensenso/HeadPose.h>
#include <std_msgs/Float64MultiArray.h>
#include "nn_controller/nn_controller.h"

using namespace amfc_control;

//constructor
Controller::Controller(ros::NodeHandle nc, const Eigen::VectorXd& ref)
: n_(nc), ref_(ref), updatePoseInfo(false), updateController(false),
 updateWeights(false), print(false), counter(0)
{	 		 
	initMatrices();
	pred_pub_ = n_.advertise<nn_controller::predictor>("/osa_pred", 100);
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

ros::Time Controller::getTime()
{
	return ros::Time::now();
}

void Controller::pose_subscriber(const ensenso::HeadPose& headPose)
{
	Eigen::VectorXd pose_info;
	pose_info.resize(6);		//CRITICAL OR SEGFAULT
	getPoseInfo(headPose, pose_info);
	{
		std::lock_guard<std::mutex> lock(mutex);
		this->pose_info = pose_info;
		updatePoseInfo  = true;		
	}
}

void Controller::getPoseInfo(const ensenso::HeadPose& headPose, Eigen::VectorXd pose_info)
{
	pose_info << headPose.x, headPose.y,
				 headPose.z, 1,  		//roll to zero
				 headPose.pitch, headPose.yaw;   //setting roll to zero
	//set ref's non-controlled states to measurement

	ControllerParams(std::move(pose_info));
}

//no loner used
void Controller::NetPredictorInput(Eigen::VectorXd&& pose_info)
{
	Eigen::VectorXd pose_delayed;
	// pose_info.resize(6);		//CRITICAL OR SEGFAULT
	pose_queue.push(pose_info);

	pred_ut = ref_ - pose_info;

	if(pose_queue.size() % 2 == 0)
	{
	    pose_delayed = pose_queue.front();
		//concatenate {u(t), y(t-1)} and broadcast tuple to lua predictor
		pred_.header.stamp 	= getTime();
		pred_.header.seq	= ++counter;
		pred_.header.frame_id	= "osa_pred";
		pred_.u1 		= pred_ut(0);
		pred_.u2    	= pred_ut(1);
		pred_.u3    	= pred_ut(2);
		pred_.u4    	= pred_ut(3);
		pred_.u5    	= pred_ut(4);
		pred_.u6    	= pred_ut(5);
		pred_.z_d    	= pose_delayed(2); 	//z
		pred_.pitch_d   = pose_delayed(4);	//pitch
		pred_.yaw_d    	= pose_delayed(5);	//yaw

		//clear the queue
		pose_queue = {};  	
	}
	if(print)
	{		
		OUT("\npose_delayed: " << pose_delayed.transpose());
		OUT("pose_info: " << pose_info.transpose());
		OUT("pred_input_u: " << pred_ut.transpose());
	}
	pred_pub_.publish(pred_);
}

void Controller::ControllerParams(Eigen::VectorXd&& pose_info)
{	
	tracking_error.resize(6);

	expAmk *= std::exp(-1334./1705*counter);
	ym = Bm * ref_ * expAmk;

	//compute tracking error, e = y - y_m
	tracking_error = pose_info - ym;
	Ky_hat = -1* Gamma_y * pose_info * tracking_error.transpose().eval() * P * B;
	Kr_hat = -1* Gamma_r * ref_      * tracking_error.transpose().eval() * P * B;

	//retrieve the pre-trained weights and biases 
	Eigen::Matrix<double, 3, 3> modelWeights;
	Eigen::Vector3d modelBiases;

	if(updateWeights)
	{			
		std::lock_guard<std::mutex> weights_lock(weights_mutex);
		modelBiases		= this->modelBiases;
		modelWeights	= this->modelWeights;
		updateWeights 	= false;
	}

	//compute control law
	Eigen::VectorXd u_control;
	u_control = (Ky_hat * pose_info) + (Kr_hat * ref_); // + pred;
	{
		std::lock_guard<std::mutex> lock(mutex);
		this->u_control = u_control;
		updateController = true;
	}

	OUT("\nmodelWeights: \n" << modelWeights);
	OUT("\nmodelBiases: \n" << modelBiases.transpose());
	OUT("\nControl Law: \n" << u_control.transpose());

	++counter;
}

void Controller::ref_model_multisub(const std_msgs::Float64MultiArray::ConstPtr& ref_model_params)
{
	std::vector<double> model_params = ref_model_params->data;

	//retrieve the pre-trained weights and biases 
	Eigen::Matrix<double, 3, 3> modelWeights;
	Eigen::Vector3d modelBiases;
	std::lock_guard<std::mutex> weights_lock(weights_mutex);
	{	
		modelWeights(0,0) = model_params[0];
		modelWeights(0,1) = model_params[1];
		modelWeights(0,2) = model_params[2];
		modelWeights(1,0) = model_params[4];
		modelWeights(1,1) = model_params[5];
		modelWeights(1,2) = model_params[6];
		modelWeights(2,0) = model_params[8];
		modelWeights(2,1) = model_params[9];
		modelWeights(2,2) = model_params[10];

		modelBiases(0)	  = model_params[3];
		modelBiases(1)	  = model_params[7];
		modelBiases(2)	  = model_params[11];

		this->modelWeights = modelWeights;
		this->modelBiases  = modelBiases;

		updateWeights = true;
	}
}

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
	res.right_in	=	u_control_local(5);

	return true;
}

bool Controller::configure_error(
	nn_controller::amfcError::Request  &req,
	nn_controller::amfcError::Response  &res)
{
	// this is the reference model. it is always time-varying1
	// k_m = 1.25;  
	// a_m = -0.782404601/k;   
	// y_0 = 0;  //assume zero initial response
 	// now = std::chrono::high_resolution_clock::now();
 	// double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count() / 1000.0;
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

void help()
{
	OUT("\t\tAdd the 3DOF desired trajectory separated by a single space");
	OUT("\t\tLike so: rosrun nn_controller nn_controller <z> <pitch> <yaw>");
}

int main(int argc, char** argv)
{ 
	ros::init(argc, argv, "controller_node", ros::init_options::AnonymousName);
	ros::NodeHandle n;

	Eigen::VectorXf ref;
	ref.resize(6);
	if(argc < 2)
	{
		help();
		return EXIT_FAILURE;
	} 
	try{
		ref(0) = 0;
		ref(1) = 0;
		ref(2) = atof(argv[1]);    //ref z
		ref(3) = 0;
		ref(4) = atof(argv[2]);	  //ref pitch
		ref(5) = atof(argv[3]);	  //ref yaw
	}
	catch(std::exception& e){
		e.what();
	}


	Eigen::VectorXd refd;
	refd.resize(6); 
	refd = ref.cast<double>();
	Controller c(n, refd);

    ros::Subscriber sub = n.subscribe("/saved_net", 1000, &Controller::ref_model_subscriber, &c );
    ros::Subscriber sub_multi = n.subscribe("/multi_net", 1000, &Controller::ref_model_multisub, &c );
	ros::Subscriber sub_pose = n.subscribe("/mannequine_head/pose", 100, &Controller::pose_subscriber, &c);	
	// ros::ServiceServer service = n.advertiseService("/error_srv", &Controller::configure_controller, &c);	
	ros::ServiceServer control_serv = n.advertiseService("/mannequine_head/controller", 
										&Controller::configure_controller, &c);

	ros::spin();

	ros::shutdown();

	return 0;
}

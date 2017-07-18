/*
        Olalekan Ogunmolu.
        June 01, 2017

        See: A Model for Registration of 3D shapes,
             Paul Besl and Neil D. McKay

             Eqs 23 - 27
*/

#include "ros/ros.h"
#include <ros/spinner.h>
#include "std_msgs/String.h"
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>  //for bt Quaternion
#include <geometry_msgs/Transform.h>

#include <mutex>
#include <vector>
#include <thread>
#include <typeinfo>

#include <vicon_bridge/Markers.h>
#include <geometry_msgs/Point.h>

#include <Eigen/Eigenvalues>
// for sender to RIO
#include <ensenso/boost_sender.h>

using namespace Eigen;
#define OUT (std::cout << __o__ << std::endl;)

class Receiver
{
private:
    int count, num_points; //iterator and number of markers on object
    std::vector<geometry_msgs::Point> headMarkersVector,
                                    firstHeadMarkersVector;
    std::vector<Vector3d> face_vec, first_face_vec;
    // covariance matrix
    Matrix3d sigma_px, temp;
    // identity matrix involved in Q matrix
    Matrix3d I3;

    Matrix3d A_Mat;

    // Form Q from which we compute the rotation quaternion
    Matrix4d Q;
    // Delta
    Vector3d Delta;
    // will contain rotationand translation of the head
    geometry_msgs::Pose pose_info;
    geometry_msgs::Point face_translation;

    ros::NodeHandle nm_;
    std::mutex mutex;
    bool updatePose, running, print_;
    ros::AsyncSpinner spinner;
    unsigned long const hardware_threads;

    // pose vector
    ros::Subscriber sub_markers;
    ros::Publisher pose_pub; //publisher for translation and euler angles

    std::thread rotoTransThread;
    double roll, pitch, yaw;
    double mu_p, mu_x;
    double x, y, z;
    double q1, q2, q3, q4;

    // sender objects 
    boost::asio::io_service io_service;
    const std::string multicast_address;

public:
    Receiver(bool print)
    :  hardware_threads(std::thread::hardware_concurrency()),
       spinner(2), count(0), num_points(4), updatePose(false), print_(print),
       multicast_address("235.255.0.1")
    {
       I3.setIdentity(3, 3);
    }

    ~Receiver()
    {
        // rotoTransThread.detach();
    }

    Receiver(Receiver const&) =delete;
    Receiver& operator=(Receiver const&) = delete;

    void run()
    {
      spawn();
      unspawn();
    }
private:
    void spawn()
    {
        if(spinner.canStart())
            spinner.start();
        running = true;

        sub_markers = nm_.subscribe("/vicon/markers", 10, &Receiver::callback, this);
        while(!updatePose) {
            if(!ros::ok()) {
              return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        pose_pub = nm_.advertise<geometry_msgs::Pose>("/mannequine_head/pose", 1000);
        // spawn the threads
        rotoTransThread = std::thread(&Receiver::processRotoTrans, this);
        if(rotoTransThread.joinable())
            rotoTransThread.join();
    }

    void unspawn()
    {
        spinner.stop();
        rotoTransThread.detach();
        running = false;
    }

    void callback(const vicon_bridge::MarkersConstPtr& markers_msg)
    {
        // solve all vicon markers here
        std::vector<geometry_msgs::Point> headMarkersVector;
        headMarkersVector.resize(num_points);
        for(auto i=0; i < num_points; ++i){
            headMarkersVector[i] = markers_msg -> markers[i].translation;   // fore
        }

        std::lock_guard<std::mutex> lock(mutex);
        this->headMarkersVector = headMarkersVector;
        updatePose          = true;
        ++count;
    }

    void remove_mean(std::vector<geometry_msgs::Point> && vec)
    {
        double mu_x, mu_y, mu_z = 0;
        for(auto i = 0; i < num_points; ++i)
        {
            mu_x += headMarkersVector[i].x;
            mu_y += headMarkersVector[i].y;
            mu_z += headMarkersVector[i].z;
        }

        mu_x /= num_points;
        mu_y /= num_points;
        mu_z /= num_points;

        for(auto i = 0; i < num_points; ++i)
        {
            vec[i].x -= mu_x;
            vec[i].y -= mu_y;
            vec[i].z -= mu_z;
        }
    }

    void point_to_eigen(std::vector<geometry_msgs::Point>&& pt, std::vector<Vector3d>&& face_vec)
    {
        for(auto i=0; i < num_points; ++i){
            face_vec[i] << pt[i].x, pt[i].y, pt[i].z;
        }

        // compute average of markers
        face_translation.x = 0.25*(pt[0].x, pt[1].x, pt[2].x, pt[3].x);
        face_translation.y = 0.25*(pt[0].y, pt[1].y, pt[2].y, pt[3].y);
        face_translation.z = 0.25*(pt[0].z, pt[1].z, pt[2].z, pt[3].z);
    }


    // this closely follows pg 243 of the ICP paper by Besl and McKay
    void processRotoTrans()
    {

        std::vector<geometry_msgs::Point> headMarkersVector;
        headMarkersVector.resize(num_points);
        firstHeadMarkersVector.resize(num_points);
        first_face_vec.resize(num_points);

        for(; running && ros::ok() ;)
        {    
            if(count == 5)         //store away zero pose of head // count=5 is a stable number
            {
                std::lock_guard<std::mutex> lock(mutex);
                firstHeadMarkersVector = this->headMarkersVector;
                remove_mean(std::move(firstHeadMarkersVector));
                //convert from geometry points to eigen
                point_to_eigen(std::move(firstHeadMarkersVector), std::move(first_face_vec));
            }            
            if(updatePose)
            {
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    headMarkersVector = this->headMarkersVector;
                    updatePose = false;                    
                }

                //compute center of mass of model and measured point set
                remove_mean(std::move(headMarkersVector));
                //convert from geometry points to eigen
                face_vec.resize(num_points);
                point_to_eigen(std::move(headMarkersVector), std::move(face_vec));
                //compute the cross covariance matrix of the points sets P and X
                sigma_px.resize(3, 3);

                sigma_px =  first_face_vec[0] * face_vec[0].transpose() +
                            first_face_vec[1] * face_vec[1].transpose() +
                            first_face_vec[2] * face_vec[2].transpose() +
                            first_face_vec[3] * face_vec[3].transpose() ;
                sigma_px /= num_points;

                // std::cout<<"\n" << first_face_vec[0] << ", " << first_face_vec[1] << ", " << first_face_vec[2] << ", " << first_face_vec[3] << std::endl;
                // std::cout << std::endl;

                A_Mat.resize(3, 3);
                // find the cyclic components of the skew symmetric matric A_{ij} = (sigma_{px} + sigma_{px}^T )_{ij}
                A_Mat = sigma_px - sigma_px.transpose();

                //collect cyclic components of skew symmetric matrix
                Delta << A_Mat(1,2), A_Mat(2, 0), A_Mat(0, 1);

                temp.resize(3,3);
                temp = sigma_px + sigma_px.transpose() - (sigma_px.trace() * I3);

                // Form the symmetric 4x4 Q matrix
                Q(0, 0) =  sigma_px.trace();      Q(0, 1) = A_Mat(1,2);         Q(0, 2) = A_Mat(2, 0);   Q(0, 3) = A_Mat(0, 1);
                Q(1, 0) =  A_Mat(1,2);            Q(1, 1) = temp(0, 0);         Q(1, 2) = temp(0, 1);    Q(1, 3) = temp(0, 2);
                Q(2, 0) =  A_Mat(2,0);            Q(2, 1) = temp(1, 0);         Q(2, 2) = temp(1, 1);    Q(2, 3) = temp(1, 2);
                Q(3, 0) =  A_Mat(0,1);            Q(3, 1) = temp(2, 0);         Q(3, 2) = temp(2, 1);    Q(3, 3) = temp(2, 2);

                // ROS_INFO_STREAM("Q Matrix: " <<  Q);
                //
                // we now find the maximum eigen value of the matrix Q
                EigenSolver<Matrix4d> eig(Q);

                // Note that eigVal and eigVec are std::complex types. To access their
                // real or imaginary parts, call real or imag
                EigenSolver< Matrix4d >::EigenvalueType eigVals = eig.eigenvalues();
                EigenSolver< Matrix4d >::EigenvectorsType eigVecs = eig.eigenvectors();

                // std::cout << "vector types: " << eigVecs.col(0)  << typeid(eigVecs.col(0)).name() << "\n";
                findQuaternion(std::move(eigVals), std::move(eigVecs));
            }
        }
    }

    inline void rad2deg(double&& rad) const{
        rad = (M_PI * rad)/180;
    }

    void findQuaternion(EigenSolver< Matrix4d >::EigenvalueType&& eigVals, EigenSolver< Matrix4d >::EigenvectorsType && eigVecs)
    {
        //create a look-up table of eig vectors and values
        std::vector<double> valueVectors {eigVals[0].real(), eigVals[1].real(), eigVals[2].real(), eigVals[3].real()};

        auto max = valueVectors[0];
        int magicIdx = 0;
        for(int i = 0; i < valueVectors.size(); ++i)        {
            if(valueVectors[i] > max){
              max = valueVectors[i];
              magicIdx = i;
            }
        }
        // find the eigen vector with the largest eigen value, This would be the optimal rotation quaternion
        auto optimalEigVec = eigVecs.col(magicIdx);
        // compute optimal translation vector

        pose_info.position.x = face_translation.x;
        pose_info.position.y = face_translation.y;
        pose_info.position.z = face_translation.z;
        
        pose_info.orientation.x = optimalEigVec[1].real();
        pose_info.orientation.y = optimalEigVec[2].real();
        pose_info.orientation.z = optimalEigVec[3].real();
        pose_info.orientation.w = optimalEigVec[0].real();
        
        tf::Quaternion quart(pose_info.orientation.x, \
                        pose_info.orientation.y, \
                        pose_info.orientation.z, \
                        pose_info.orientation.w);
        
        tf::Matrix3x3 Rot(quart);
        Rot.getRPY(roll, pitch, yaw);
        // convert rads to degrees
        rad2deg(std::move(roll));
        rad2deg(std::move(pitch));
        rad2deg(std::move(yaw));

        // publish the head pose
        pose_pub.publish(pose_info);
        if(print_){            
            ROS_INFO_STREAM("Publishing pose info as ");
            printf("x: %.3f | y: %.3f | z: %.3f | roll: %.3f | pitch: %.3f | yaw: %.3f \n", pose_info.position.x, \
                                                                            pose_info.position.y, \
                                                                            pose_info.position.z, \
                                                                            roll, pitch, yaw);
        }
        ros::Rate looper(30);
        looper.sleep();
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "vicon_icp_node");

    if(!ros::ok())   
      return EXIT_SUCCESS;

    ROS_INFO_STREAM("Started node " << ros::this_node::getName().c_str());

    bool print;
    if(!ros::param::get("/vicon_icp/print", print))
        ROS_DEBUG("could not retrieve [%d] from ros parameter server",  print);

    Receiver rcvr(print);
    rcvr.run();

    ros::shutdown();
    return EXIT_SUCCESS;
}

/*
Olalekan Ogunmolu. 
SeRViCe Lab, 
Feb. 18, 2017*/

#include "ros/ros.h"
#include <ros/spinner.h>
#include "std_msgs/String.h"
#include <ros/console.h>
#include <tf/transform_broadcaster.h>
#include <thread>
#include <string>
#include <fstream>
#include <mutex>
#include <chrono>

#include <vicon_bridge/Markers.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/Pose.h>

#include <tf/transform_datatypes.h>
#include <tf_conversions/tf_eigen.h>

#include "boost/bind.hpp"
#include <boost/thread.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>

const std::string Superchick_name = "Superchicko";
std::string listener = "vicon_listener";

std::string subject, segment;
const std::string globalTopicName = "/vicon/Superdude/head";
using namespace Eigen;

class Receiver
{ 
private:
    float xm, ym, zm;
    bool save, print, sim, running, firstIter; 
    double headRoll, headPitch, headYaw;
    double panelRoll, panelPitch, panelYaw;
    int count;

    Vector3d rpy;

    std::vector<geometry_msgs::Point> headMarkersVector, 
                                      panelMarkersVector;
    
    std::vector<std::thread> threads;
    ros::NodeHandle nm_;
    ros::Publisher pub;
    Matrix3d R; //basis rotation matrix with respect to camera
    boost::mutex rotation_mutex_, markers_mutex;
    geometry_msgs::Vector3 headTrans; 
    geometry_msgs::Quaternion headQuat;
    std::mutex mutex;
    bool updatePose;
    ros::AsyncSpinner spinner;
    unsigned long const hardware_threads;

    //for rigidTransforms
    geometry_msgs::Vector3 panelTrans;
    geometry_msgs::Quaternion panelQuat;
    //pose vector
    geometry_msgs::Pose pose;
    ros::Publisher pose_pub_;

    std::vector<std::thread> threadsVector;
    std::thread testQuatThread, modGramScmidtThread,
                rotationMatrixThread;

    Matrix3d headMGS, tableMGS, rotationMatrix;

    //use exactTimePolicy to process panel markers and head markers
    // using namespace message_filters;

    using vicon_sub = message_filters::Subscriber<vicon_bridge::Markers> ;
    using head_sub  = message_filters::Subscriber<geometry_msgs::TransformStamped>;
    using pane_sub  = message_filters::Subscriber<geometry_msgs::TransformStamped>;
    using headSyncPolicy = message_filters::sync_policies::ExactTime<vicon_bridge::Markers, 
                                geometry_msgs::TransformStamped, geometry_msgs::TransformStamped>;

    vicon_sub subVicon;
    head_sub  subHead;
    pane_sub  subPanel;  

    message_filters::Synchronizer<headSyncPolicy> sync;                            

public:
    Receiver(ros::NodeHandle nm, bool save, bool print, bool sim)
    :  nm_(nm), save(save), print(print), sim(sim), hardware_threads(std::thread::hardware_concurrency()),
       subVicon(nm_, "/vicon/markers", 1), subHead(nm_, "vicon/Superdude/head", 1), 
       subPanel(nm_,"/vicon/Panel/rigid", 1), spinner(2), count(0), firstIter(false),
       sync(headSyncPolicy(10), subVicon, subHead, subPanel), updatePose(false)
    {      
        // ExactTime takes a queue size as its constructor argument, hence SyncPolicy(10)
       sync.registerCallback(boost::bind(&Receiver::callback, this, _1, _2, _3));
       pose_pub_ = nm_.advertise<geometry_msgs::Pose>("/mannequine_head/pose", 10);
    }

    ~Receiver()
    { }

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
        
        while(!updatePose)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));     

        //spawn the threads
        modGramScmidtThread = std::thread(&Receiver::modGramSchmidt, this);
        // rotationMatrixThread = std::thread(&Receiver::computeRotationMatrix,
                                                             // this);
        
        if(modGramScmidtThread.joinable())
            modGramScmidtThread.join();
        // if(rotationMatrixThread.joinable())
            // rotationMatrixThread.join();
    }

    void unspawn()
    {
        spinner.stop();
        modGramScmidtThread.detach();
        // rotationMatrixThread.detach();
        running = false;
    }

    void callback(const vicon_bridge::MarkersConstPtr& markers_msg, 
                const geometry_msgs::TransformStampedConstPtr& panel_msg,
                const geometry_msgs::TransformStampedConstPtr& head_msg)
    {   
        //solve all vicon markers here
        //Retrieve geometry_msgs translation for four markers on superchicko 
        std::vector<geometry_msgs::Point> headMarkersVector;
        headMarkersVector.resize(4);
        headMarkersVector[0] = markers_msg -> markers[0].translation;  //fore
        headMarkersVector[1] = markers_msg -> markers[1].translation;   //left
        headMarkersVector[2] = markers_msg -> markers[2].translation;   //chin
        headMarkersVector[3] = markers_msg -> markers[3].translation;   //right

        std::vector<geometry_msgs::Point> panelMarkersVector;
        panelMarkersVector.resize(4);
        panelMarkersVector[0] = markers_msg->markers[4].translation;    //tabfore
        panelMarkersVector[1] = markers_msg->markers[5].translation;    //tableft
        panelMarkersVector[2] = markers_msg->markers[6].translation;    //tabchin
        panelMarkersVector[3] = markers_msg->markers[7].translation;    //tabright

        //solve head transformedstamp markers here
        geometry_msgs::Vector3 headTrans = head_msg->transform.translation;
        geometry_msgs::Quaternion headQuat = head_msg->transform.rotation;

        //solve panel transformedstamp markers here
        geometry_msgs::Vector3 panelTrans = panel_msg->transform.translation;
        geometry_msgs::Quaternion panelQuat = panel_msg->transform.rotation;

        //convert translations to millimeters
        metersTomilli(std::move(headTrans)); 
        metersTomilli(std::move(panelTrans));

        boost::mutex::scoped_lock lock(markers_mutex);
        this->headTrans   = headTrans;
        this->headQuat    = headQuat;

        this->panelTrans    = panelTrans;
        this->panelQuat  = panelQuat;

        this->headMarkersVector = headMarkersVector;
        this->panelMarkersVector= panelMarkersVector;

        updatePose          = true;   
        lock.unlock();
    }

    void modGramSchmidt() noexcept
    {
        std::vector<geometry_msgs::Point> headMarkersVector, 
                                          panelMarkersVector;
        
        //attached frame to face
        Vector3d fore, left, right, chin;  
        //base markers
        Vector3d tabfore, tableft, tabright, tabchin; 

        Matrix3d headMGS, tableMGS;

        Vector3d rpy;
        ros::Rate looper(10);

        for(; running && ros::ok() ;)
        {            
            if(updatePose)
            {   
                boost::mutex::scoped_lock lock(markers_mutex);
                headMarkersVector  = this->headMarkersVector;
                panelMarkersVector = this->panelMarkersVector;
                updatePose = false;
                lock.unlock();
            }

            fore << headMarkersVector[0].x, headMarkersVector[0].y,
                    headMarkersVector[0].z;
            left << headMarkersVector[1].x, headMarkersVector[1].y,
                    headMarkersVector[1].z;  
            chin << headMarkersVector[2].x, headMarkersVector[2].y,
                    headMarkersVector[2].z;
            right << headMarkersVector[3].x, headMarkersVector[3].y,
                    headMarkersVector[3].z;   

            tabfore << panelMarkersVector[0].x, panelMarkersVector[0].y,
                       panelMarkersVector[0].z;
            tableft << panelMarkersVector[1].x, panelMarkersVector[1].y,
                       panelMarkersVector[1].z;
            tabchin << panelMarkersVector[2].x, panelMarkersVector[2].y,
                       panelMarkersVector[2].z;
            tabright << panelMarkersVector[3].x, panelMarkersVector[3].y,
                       panelMarkersVector[3].z;

            //subtract the markers to create axes
            left     -= right;
            fore     -= left;
            //table markers
            tableft -= tabright;
            tabfore  -= tableft;
            // botRight -= topRight;
            //define the unit axis vectors kx, ky, kz for head frame
            Vector3d kx, ky, kz;

            /*adapted from Teddy's code
            * Gets the orientation quaternion of a planar model such that the 
            * z-axis is normal to the plane, and the x-y
            * axis are as close as possible to the the table frame
            */
            //http://answers.ros.org/question/31006/how-can-a-vector3-axis-be-used-to-produce-a-quaternion/
            Vector3d item_z((left(0)+right(0)+fore(0)+chin(0))/4, 
                            (left(1)+right(1)+fore(1)+chin(1))/4, 
                            (left(2)+right(2)+fore(2)+chin(2))/4);
            item_z.normalize();

            Vector3d table_z(0.0, 0.0, 1.0);
            // z_vector.normalize();

            double normAcrossB = item_z.cross(table_z).norm(); // this gives us A cross B norm
            double normBcrossA = table_z.cross(item_z).norm(); // this gives us B cross A norm
            double AdotB = item_z.dot(table_z);

            Matrix3d G; // this is matrix G for manipulation
            G <<    AdotB, -1*normAcrossB, 0,
                    normAcrossB, AdotB, 0,
                    0, 0, 1;

            //Calc U
            Vector3d U = (AdotB * item_z).normalized();
            Vector3d V = (table_z - AdotB * item_z).normalized();
            Vector3d W = table_z.cross(item_z);
            Matrix3d Fproto;
            Fproto << U, V, W;

            Matrix3d Rotation_Matrix = (Fproto * G * Fproto.inverse()).inverse();
            // Quaterniond q(Rotation_Matrix);
            // tf::Quaternion tf_q;
            // tf::quaternionEigenToTF(q, tf_q);

            // tf::Vector3 y_vector = axis_vector.cross(z_vector);
            kx = fore.cross(chin)/(fore.cross(chin)).norm();
            ky = left.cross(right)/(left.cross(right)).norm();
            kz = kx.cross(ky);
            //define the unit axes vectors kxp, kyp, kzp forthe table frame'
            Vector3d kxp, kyp, kzp;
            kxp = tabfore.cross(tabchin)/(tabfore.cross(tabchin)).norm();
            kyp = tableft.cross(tabright)/(tableft.cross(tabfore)).norm();
            kyp = kxp.cross(kyp);
            
            //see https://ocw.mit.edu/courses/mathematics/18-335j-introduction-to-numerical-methods-fall-2010/lecture-notes/MIT18_335JF10_lec10a_hand.pdf
            std::vector<Vector3d> v(4), q(4);   
            //compute orthonormal basis vectors for face markers       
            // v[0] = kx;   v[1] = ky;   v[2] = kz; 
            v[0] = left; v[1] = right; v[2] = left.cross(right);
            double r[3][3] = {};    
            for(auto i = 0; i < 3; ++i)
            {
                r[i][i] = v[i].norm();
                q[i]    = v[i]/r[i][i];
                for(auto j = i +1; j < 3; ++j)
                {
                    r[i][j] = q[i].transpose() * v[j];
                    v[j]    = v[j] - r[i][j] * q[i];
                }
            }  
            
            // populate headMGS rotation basis matrix
            for(auto i =0; i < 3; ++i)
            { 
                headMGS.col(i) = q[i];
            }
            this->headMGS = headMGS;

            //compute basis vectors for table markers
            v.clear(); q.clear();
            v[0] = kxp; v[1] = kyp; v[2] = kzp; 
            v[0] = tableft; v[1] = tabright; v[2] = tableft.cross(tabright); 
            for(auto i = 0; i < 3; ++i)
            {
                r[i][i] = v[i].norm();
                q[i]    = v[i]/r[i][i];
                for(auto j = i +1; j < 4; ++j)
                {
                    r[i][j] = q[i].transpose() * v[j];
                    v[j]    = v[j] - r[i][j] * q[i];
                }
            }  
            // populate tableMGS rotation basis matrix
            for(auto i =0; i < 3; ++i)
            { 
                tableMGS.col(i) = q[i];
            }
            this->tableMGS = tableMGS;  

            /*
            The rotation matrix of the head with respect to the table follows
            John Craig's Introduction to Robotics Book convention[p.22] and it 
            is denoted by 

            TRH = [ TXH    TYH     TZH ]   (H denotes head)

                = [XH.XT  YH.XT   ZH.XT]    (T denotes table frame)
                  [XH.YT  YH.YT   ZH.YT]
                  [XH.ZT  YH.ZT   ZH.ZT]
            where
            H   = [XH  YH  ZH]
            T   = [XT  YT  ZT]
            */
            //First Row
            rotationMatrix(0, 0) = headMGS.col(0).dot(tableMGS.col(0));
            rotationMatrix(0, 1) = headMGS.col(1).dot(tableMGS.col(0));
            rotationMatrix(0, 2) = headMGS.col(2).dot(tableMGS.col(0));
            //Second Row
            rotationMatrix(1, 0) = headMGS.col(0).dot(tableMGS.col(1));
            rotationMatrix(1, 1) = headMGS.col(1).dot(tableMGS.col(1));
            rotationMatrix(1, 2) = headMGS.col(2).dot(tableMGS.col(1));
            //Third Row
            rotationMatrix(2, 0) = headMGS.col(0).dot(tableMGS.col(2));
            rotationMatrix(2, 1) = headMGS.col(1).dot(tableMGS.col(2));
            rotationMatrix(2, 2) = headMGS.col(2).dot(tableMGS.col(2));

            this->rotationMatrix = rotationMatrix;
            // rollpy(Rotation_Matrix);
            rollpy(rotationMatrix);

            // std::cout << "\nheadMGS: \n" << headMGS << std::endl;
            // std::cout << "tableMGS: \n" << tableMGS << std::endl;                        
            // std::cout <<"rpy in mgs: " << this->rpy.transpose() << std::endl;

            pose.position.x = item_z(0) * 1000;
            pose.position.y = item_z(1) * 1000;
            pose.position.z = item_z(2) * 1000;
            pose.orientation.x = this->rpy(0); //roll
            pose.orientation.y = this->rpy(1); //pitch
            pose.orientation.z = this->rpy(2); //yaw
            pose.orientation.w = 1.0;
            pose_pub_.publish(pose);

            looper.sleep();                  
        } 
    }

    //From Rotation Matrix, find rpy
    void rollpy(Matrix3d R) //const
    {   
        Vector3d rpy;
        if (  (std::fabs(R(0,0)) < .001) && (std::fabs(R(1,0)) < .001) )           
        {
            // singularity
            rpy(0) = 0;
            rpy(1) = std::atan2(-R(2,0), R(0,0));
            rpy(2) = std::atan2(-R(1,2), R(1,1));
        }
        else
        {   
            rpy(0) = std::atan2(R(1,0), R(0,0));
            rpy(1) = std::atan2(
                                -R(2,0), (
                                         (std::cos(rpy(0)) * R(0,0)) + 
                                         (std::sin(rpy(0)) * R(1,0))
                                         )
                                );
            rpy(2) = std::atan2(
                                (std::sin(rpy(0)) * R(0,2) - std::cos(rpy(0)) * R(1,2)), 
                                (std::cos(rpy(0))*R(1,1) - std::sin(rpy(0))*R(0,1))
                                );
        }   
        //convert to degree
        rad2deg(std::forward<Vector3d>(rpy));
        // ROS_INFO_STREAM("rpy: " << rpy.transpose()); 
        this->rpy = rpy;
    }

    void testQuat() noexcept
    {
        geometry_msgs::Quaternion headQuat, panelQuat;
        // if(updatePose)
        // {                        

        boost::mutex::scoped_lock lock(markers_mutex);
        // std::lock_guard<std::mutex> lock(mutex);
        {                
            headQuat = this->headQuat;
            panelQuat = this->panelQuat;
            updatePose = false;
        }
        lock.unlock();
        // }
        double roll, rolla, pitch, pitcha, yaw, yawa;

        getRPYFromQuaternion(std::forward<geometry_msgs::Quaternion>(headQuat), 
                             std::forward<double>(roll),
                             std::forward<double>(pitch), 
                             std::forward<double>(yaw));

        // rad2deg(std::move(roll));
        // rad2deg(std::move(pitch));
        // rad2deg(std::move(yaw));

        printf("\n|\troll \t|\tpitch \t|\tyaw\n %f , \t%f, \t%f", 
                        roll, pitch, yaw); 

    }

    void getRPYFromQuaternion(geometry_msgs::Quaternion&& rotQuat, double&& roll,
                             double&& pitch, double&& yaw)
    {

        tf::Quaternion q(rotQuat.x, rotQuat.y, rotQuat.z, rotQuat.w);
        tf::Matrix3x3 m(q);
        m.getRPY(roll, pitch, yaw);
        rad2deg(std::move(roll)); rad2deg(std::move(pitch)); rad2deg(std::move(yaw)); 
    }

    inline void metersTomilli(geometry_msgs::Vector3&& translation)
    {
        translation.x   *= 1000;
        translation.y   *= 1000;
        translation.z   *= 1000;
    }

    inline void rad2deg(double&& x)
    {
        x  *= 180/M_PI;
    }

    inline void rad2deg(Vector3d&& x)
    {
        rad2deg(std::move(x(0)));
        rad2deg(std::move(x(1)));
        rad2deg(std::move(x(2)));
    }   

    void savepoints()
    {
        //Now we write the points to a text file for visualization processing
        std::ofstream midface;
        midface.open("midface.csv", std::ofstream::out | std::ofstream::app);
        midface << xm <<"\t" <<ym << "\t" << zm << "\n";
        midface.close();
    }    
};


int main(int argc, char **argv)
{
    
    uint32_t options = 0;

    ros::init(argc, argv, listener, options);

    bool save, print, sim;

    ros::NodeHandle nm;

    save = nm.getParam("save", save) ;
    print = nm.getParam("print", print);
    sim = nm.getParam("sim", sim);

    Receiver  r(nm, save, print, sim);
    r.run();

    if(!ros::ok())
    {
      return 0;
    }

    ros::shutdown();
}
    
    
    
    
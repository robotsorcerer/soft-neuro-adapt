import rospy
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose


class Listener(object):

    def __init__(self, Pose, Twist):
        super(Listener, self).__init__()

        self.pose, self.controls = Pose, Twist
        self.pose_export, self.controls_export = dict(), dict()
        self.listen()

    def pose_callback(self, pose):
      self.pose = pose
        
      self.pose_export["x"] = pose.position.x
      self.pose_export["y"] = pose.position.y
      self.pose_export["z"] = pose.position.z

      self.pose_export["roll"]  = pose.orientation.x
      self.pose_export["pitch"] = pose.orientation.y
      self.pose_export["yaw"]   = pose.orientation.z

    def control_callback(self, controls):
        self.controls = controls

        #it goes from the bottom-most to top-most
        self.controls_export["lo"] = controls.linear.x
        self.controls_export["ro"] = controls.linear.y
        self.controls_export["bo"] = controls.linear.z
        self.controls_export["ri"] = controls.angular.x
        self.controls_export["bi"] = controls.angular.y
        self.controls_export["li"] = controls.angular.z

    def listen(self):
        rospy.init_node('Listener')

        y_sub = rospy.Subscriber('/mannequine_head/pose', Pose, self.pose_callback)
        u_sub = rospy.Subscriber('mannequine_head/u_valves', Twist, self.control_callback)

        # rospy.spin()

        return self.pose_export, self.controls_export

# l = Listener(Pose, Twist)

# while not rospy.is_shutdown():
#     print l.controls_export.get("lo", None)

#     # rospy.sleep(10)
import rospy
import roslib
from ensenso.msg import ValveControl
from geometry_msgs.msg import Pose

roslib.load_manifest('pyrnn')

class Listener(object):

    def __init__(self, Pose, ValveControl):
      super(Listener, self).__init__()

      self.pose, self.controls = Pose, ValveControl
      self.pose_export, self.controls_export = dict(), dict()
      self.listen()

    def pose_callback(self, pose):        
      self.pose_export = {
        'x': pose.position.x,
        'y': pose.position.y,
        'z': pose.position.z,
        'roll': pose.orientation.x,
        'pitch': pose.orientation.y,
        'yaw': pose.orientation.z
      }
      # print 'self.pose: ', self.pose_export

    def control_callback(self, controls):
      # example: lo = left bladder output while li = left bladder input
      self.controls_export = {
        'lo': controls.left_bladder_pos,
        'li': controls.left_bladder_neg,
        'ro': controls.right_bladder_pos,
        'ri': controls.left_bladder_neg,
        'bo': controls.base_bladder_pos,
        'bi': controls.base_bladder_neg
      }
      # print '\nself.controls_export', self.controls_export

    def listen(self):
      rospy.init_node('pose_control_listener', anonymous=True)    

      rospy.Subscriber('/mannequine_head/pose', Pose, self.pose_callback)
      rospy.Subscriber('/mannequine_head/u_valves', ValveControl, self.control_callback)
      
      # rospy.spin()

    def __getattr__(self, controls_export):
      if hasattr(self.controls_export, controls_export):
        return self.controls_export
      else:
        raise AttributeError

    def __getattr__(self, pose_export):
      if hasattr(self.pose_export, pose_export):
        return self.pose_export
      else:
        raise AttributeError



# l = Listener(Pose, ValveControl)

import rospy

from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose

class ROSCommEmulator():
    """
    All communication between algorithms and ros is herein handled.

    Emulates a ROS service (request-response) from a
    publisher-subscriber pair.
    Args:
       pub_topic: Publisher topic.
       pub_type: Publisher message type.
       sub_topic: Subscriber topic.
       sub_type: Subscriber message type.
    """

    def __init__(self, pub_topic, pub_type, sub_topic, sub_type):
        
        if pub_topic:
            self._pub = rospy.Publisher(pub_topic, pub_type, queue_size=10)
        if sub_topic:
            self._sub = rospy.Subscriber(sub_topic, sub_type, self._callback)

        self._waiting = False
        self._subscriber_msg = None

    def _callback(self, message):
        if self._waiting:
            self._subscriber_msg = message
            self._waiting = False

    def publish(self, pub_msg):
        """ Publish a message without waiting for response. """
        self._pub.publish(pub_msg)

    def publish_and_wait(self, pub_msg, timeout=5.0, poll_delay=0.01,
                         check_id=False):
        """
        Publish a message and wait for the response.
        Args:
            pub_msg: Message to publish.
            timeout: Timeout in seconds.
            poll_delay: Speed of polling for the subscriber message in
                seconds.
            check_id: If enabled, will only return messages with a
                matching id field.
        Returns:
            sub_msg: Subscriber message.
        """
        if check_id:  # This is not yet implemented in C++.
            raise NotImplementedError()

        self._waiting = True
        self.publish(pub_msg)

        time_waited = 0
        while self._waiting:
            rospy.sleep(poll_delay)
            time_waited += 0.01
            if time_waited > timeout:
                raise TimeoutException(time_waited)
        return self._subscriber_msg

    #   if init_node:
    #       rospy.init_node('agent_ros', anonymous=True)
    #       self.control_msg = control_msg
    #       self.pose_msg = pose_msg

    #   self._init_pubs_and_subs()
    #   self._seq_id = 0  # Used for setting seq in ROS commands.


    # def _init_pubs_and_subs(self):
    #     self._control_sub = ServiceEmulator(
    #         self._hyperparams['trial_command_topic'], TrialCommand,
    #         self._hyperparams['sample_result_topic'], SampleResult
    #     )
    #     self._pose_sub = ServiceEmulator(
    #         self._hyperparams['reset_command_topic'], PositionCommand,
    #         self._hyperparams['sample_result_topic'], SampleResult
    #     )

    # These two callbacks must be called after vicon launch has been called
    # def _pose_callback(self, message):
    #   """
    #   callback for /mannequine_head/pose
    #   """


    # def _control_law_callback(self, message):
    #   """
    #   callback for /mannequine_head/u_valves
    #   """

    # def _init_tf(self)

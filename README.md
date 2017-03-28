### Intro

Code for the Self-Correcting Neuro Adaptive Controller

### Dependencies

- [PyTorch](http://pytorch.org/)
	<pre><code> conda install pytorch torchvision -c soumith </code></pre>

- [ROS](http://wiki.ros.org/indigo/Installation/Ubuntu)

- The [Superchick package](https://github.com/lakehanne/superchicko)

- Vision-based tracker
	- My clone of the [vicon package](https://github.com/lakehanne/superchicko/tree/indigo-devel/vicon)

	- The [Ensenso package](https://github.com/lakehanne/ensenso)

### Running the Package

#### Vision

-	Ensenso
	If you plan to use ensenso, do this in terminal

	```bash
		<pre class="terminal"><code> Terminal 1</pre></code>:	rosrun ensenso ensenso_bridge
		<pre class="terminal"><code> Terminal 2</pre></code>:	rosrun ensenso ensenso_seg
	```

	This should open up the face scene and segment out the face as well as compute the cartesian coordinates and roll, pitch, yaw angles of the face in the scene.

	The pose tuple of the face is broadcast on the topic `mannequine_head/pose`.

- 	Vicon

	With the vicon system, you get a more accurate representation. We would want four markers on the face in a rhombic manner (preferrably named `fore`, `left` , `right`, and `chin`); make sure the subject and segment are appropriately named `Superdude/head` in Nexus. We would also want four markers on the base panel from which the rotation of the face with respect to the panel frame is computed (call these markers `tabfore`, `tabright`, `tableft` and `tabchin` respectively). Make sure the subject and segment are named `Panel/rigid` in Nexus. In terminal, bring up the vicon system

	```bash		
		<pre class="terminal"><code> Terminal 1</pre></code>:	rosrun vicon_bridge vicon.launch
	```

- 	Neural Network Function Aproximator


	Previously written in Torch7 as the [farnn](/farnn) package, this code has been migrated to [pyrnn](/pyrnn) in the recently released [PyTorch](pytorch) language to take advantage of python libraries, cvx and quadratic convex programming.

	- Farnn

	Running in Farnn is done by `roscd` ing into the `farnn src` folder and running `real_time_predictor.lua` while [nn_controller](/nn_controller) is running(t)his is automatically done in the launch file).

	- PyRNN

	`roscd` into `pyrnn src` folder and do python main.py



### Intro

Source codes for my Robotics and Automation Letters paper: A 3-DoF Neuro-Adaptive Pose Correcting System For Frameless and Maskless Cancer Radiotherapy.  arXiv:1703.03821

### QPProof

Please see my [blog post](http://lakehanne.github.io/QP-Layer-MRAS).

#### Activating Environments 

The code is finally phased into python 3.6+. If you have a base ROS installation that uses
 Python 2.7 and you want to keep your native pythonpath, the best way to change your python version 
 without messing stuff up is to install python3.6 with conda and activate the py36 environment
 everytime you use this code.

To install e.g. a python 3.6 environment around your python skin, do

```bash
	conda create -n py36 python=3.6 anaconda
```

To activate conda 3.6  environment, do:

```bash
# > source activate py36
```

To deactivate this environment, use:

```bash
# > source deactivate py36
```
#### Core dependencies
- python 3.5+
- pytorch
- ROS


- [PyTorch](http://pytorch.org/)
	
	Only works on gpu. Tested with cuda 8.0 and cuda run time version 367.49
	
	```bash
		conda install pytorch torchvision cuda80 -c soumith </code></pre>
	```

- [ROS](http://wiki.ros.org/indigo/Installation/Ubuntu)

#### PyPI dependencies 
- numpy>=1.12.1
- scipy>=0.19.0
- qpth>=0.0.5
- cvxpy>=0.4.9
- matplotlib>=2.0.0
- ipython>=5.3.0
- h5py>=2.7.0
- setproctitle>=1.1.10
- setGPU>=0.0.7
- tqdm>=4.11.2
- catkin_pkg>=0.3.1
- block>=0.0.4
- rospkg>=1.1.0
- netifaces>=0.10.5

These can be installed with 

	```bash
		pip install -r requirements.txt
	```

### Vision processing

- [Option 1] Vision-based tracker
	- My clone of the [vicon package](https://github.com/lakehanne/superchicko/tree/indigo-devel/vicon).

		With the vicon system, you get a more accurate world representation. We would want four markers on the face in a rhombic manner (preferrably named `fore`, `left` , `right`, and `chin` to conform with the direction cosines code that extracts the facial pose); make sure the `subject` and `segment` are appropriately named `Superdude/head` in `Nexus`. We would also want four markers on the base panel from which the rotation of the face with respect to the panel frame is computed (call these markers `tabfore`, `tabright`, `tableft` and `tabchin` respectively). Make sure the `subject` and `segment` are named `Panel/rigid` in `Nexus`. In terminal, bring up the vicon system
		
		<pre class="terminal"><code> Terminal$:	rosrun vicon_bridge vicon.launch</pre></code>

		This launches the [adaptive model-following control algorithm](/nn_controller), drection_cosines computation of head rotation about the table frame and the vicon ros subscriber node.
		
- [Option 2] The [Ensenso package](https://github.com/lakehanne/ensenso).

	If you plan to use ensenso, do this in terminal

	<pre class="terminal"><code> Terminal 1$	rosrun ensenso ensenso_bridge </pre></code>
	<pre class="terminal"><code> Terminal 2$:	rosrun ensenso ensenso_seg </pre></code>
	
	This should open up the face scene and segment out the face as well as compute the cartesian coordinates and roll, pitch, yaw angles of the face in the scene.

	The pose tuple of the face is broadcast on the topic `/mannequine_head/pose`.

- 	Neural Network Function Aproximator

	Previously written in Torch7 as the [farnn](/farnn) package, this code has been migrated to [pyrnn](/pyrnn) in the recently released [PyTorch](pytorch) deep nets framework to take advantage of python libraries, cvx and quadratic convex programming for contraint-based adaptive control.

	- farnn

	Running in Farnn is done by `roscd` ing into the `farnn src` folder and running `th real_time_predictor.lua` while [nn_controller](/nn_controller) is running).

	- pyrnn

	`roscd` into `pyrnn src` folder and do `./main.py`



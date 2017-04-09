### Intro

Code for the Self-Correcting Neuro Adaptive Controller

### QPProof

Please see my [blog post](http://lakehanne.github.io/QP-Layer-MRAS).

### Dependencies 
- tqdm
- python 3.5+
- ros pkg
- catkin pkg
- pytorch

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

### Install Dependencies

```bash
	pip install -r requirements.txt
```

- [PyTorch](http://pytorch.org/)

	<pre><code> conda install pytorch torchvision cuda80 -c soumith </code></pre>

- [ROS](http://wiki.ros.org/indigo/Installation/Ubuntu)

Additionally, since we are using python 3.6 and this code runs on ros indigo, we would want to install necessary ros
packages for the py36 environment using conda without messing up our base installation.

- [Conda Catkin Package](https://anaconda.org/auto/catkin_pkg)
	
	```bash
		conda install -c auto catkin_pkg
	```

- [Conda ROS Package](https://anaconda.org/jdh88/rospkg)
	
	It's pypi installable and it's in the [requirements.txt](/requirements.txt) file.

	Or download the latest tgz from [here](http://download.ros.org/downloads/rospkg/)

- [Conda netifaces](https://anaconda.org/bcbio/netifaces)

	This is very critical as rospy uses this to communicate across the underlying unix socket protocol.

	```bash
		conda install -c bcbio netifaces=0.10.4
	```

- Vision-based tracker
	- My clone of the [vicon package](https://github.com/lakehanne/superchicko/tree/indigo-devel/vicon).

	- The [Ensenso package](https://github.com/lakehanne/ensenso).

### Running the Package

#### Vision

-	Ensenso
	If you plan to use ensenso, do this in terminal

	
	<pre class="terminal"><code> Terminal 1$	rosrun ensenso ensenso_bridge </pre></code>
	<pre class="terminal"><code> Terminal 2$:	rosrun ensenso ensenso_seg </pre></code>
	

	This should open up the face scene and segment out the face as well as compute the cartesian coordinates and roll, pitch, yaw angles of the face in the scene.

	The pose tuple of the face is broadcast on the topic `mannequine_head/pose`.

- 	Vicon

	With the vicon system, you get a more accurate representation. We would want four markers on the face in a rhombic manner (preferrably named `fore`, `left` , `right`, and `chin`); make sure the subject and segment are appropriately named `Superdude/head` in Nexus. We would also want four markers on the base panel from which the rotation of the face with respect to the panel frame is computed (call these markers `tabfore`, `tabright`, `tableft` and `tabchin` respectively). Make sure the subject and segment are named `Panel/rigid` in Nexus. In terminal, bring up the vicon system

		
	<pre class="terminal"><code> Terminal$:	rosrun vicon_bridge vicon.launch</pre></code>
	

- 	Neural Network Function Aproximator


	Previously written in Torch7 as the [farnn](/farnn) package, this code has been migrated to [pyrnn](/pyrnn) in the recently released [PyTorch](pytorch) language to take advantage of python libraries, cvx and quadratic convex programming for contraint-based adaptive control.

	- Farnn

	Running in Farnn is done by `roscd` ing into the `farnn src` folder and running `real_time_predictor.lua` while [nn_controller](/nn_controller) is running(t)his is automatically done in the launch file).

	- PyRNN

	`roscd` into `pyrnn src` folder and do `python main.py`



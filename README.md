### Intro

Source codes for my **IROS paper: A 3-DoF Neuro-Adaptive Pose Correcting System For Frameless and Maskless Cancer Radiotherapy**.  Arxiv ID: arXiv:1703.03821

#### Dependencies

** update: March 2017: ** The neural network estimator is now phased into [pytorch](pytorch.org) and python 3.5+. If you have a base ROS installation, e.g. indigo, that uses
 Python 2.7 and you want to keep your native `pythonpath`, the best way to change your python version  without messing stuff up is to install, for example a `python3.5+` with `conda` and activate the py35+ environment whenever you use this code.

To install e.g. a python 3.5+ environment around your base python skin, do

```bash
	conda create -n py35 python=3.5 anaconda
```

To activate the `conda 3.5`  environment, do:

```bash
# > source activate py35
```

To deactivate this environment, use:

```bash
# > source deactivate py35
```

#### Core dependencies
- python 3.5+
- pytorch
- ros
	
The pytorch version of this code only works on a gpu. Tested with cuda 8.0 and cuda run time version 367.49. 
To install, do		
	<code><pre class="terminal"><code>Termnal x:$ conda install pytorch torchvision cuda80 -c soumith </code></pre>

Instructions for installing ros can be found [here]](http://wiki.ros.org/indigo/Installation/Ubuntu).

#### PyPI dependencies 

PyPI installable using:
	<pre class="terminal"><code> Terminal:$	pip install -r requirements.txt </code></pre>

### QP Proof

Please see my [blog post](http://lakehanne.github.io/QP-Layer-MRAS).

##### Vision processing

You could use the vicon system or the ensenso package. Follow these steps to get up and running

* Clone this code: git clone <this package name> --recursive

If you did not clone the package recursively as shown above, you can initialize the `ensenso` and `vicon` submodules as follows:

* Initialize the ensenso and vicon submodules: `git init submodules`
* Update submodules: `git submodules update`

Then compile the codebase with either `catkin_make` or `catkin build`.

* The Vicon System Option
 * cd inside the vicon directory and follow the readme instructions there. 

   With the vicon system, you get a more accurate world representation. We would want four markers on the face in a rhombic manner (preferrably named `fore`, `left` , `right`, and `chin` to conform with the direction cosines/vicon icp codes that extracts the facial pose with respect to the scene); make sure the `subject` and `segment` are appropriately named `Superdude/head` in `Nexus`. We would also want four markers on the base panel from which the rotation of the face with respect to the panel frame is computed (call these markers `tabfore`, `tabright`, `tableft` and `tabchin` respectively). Make sure the `subject` and `segment` are named `Panel/rigid` in `Nexus`. In terminal, bring up the vicon system.
	
<pre class="terminal"><code> Terminal$:	rosrun vicon_bridge vicon.launch</pre></code>

This launches the [adaptive model-following control algorithm](/nn_controller), [icp computation](/vicon_icp) of head rotation about the table frame and the vicon ros subscriber node.
		
##### The [Ensenso](https://github.com/lakehanne/ensenso) option.

	`cd` inside the ensenso package and follow the README instructions therein. When done, do this in terminal

	<pre class="terminal"><code> Terminal 1$	rosrun ensenso ensenso_bridge </pre></code>
	<pre class="terminal"><code> Terminal 2$:	rosrun ensenso ensenso_seg </pre></code>
	
	This should open up the face scene and segment out the face as well as compute the cartesian coordinates of the face centroid as well as Euler angles that represent the orientation of the face with respect to the scene.

	The pose tuple of the face is broadcast on the topic `/mannequine_head/pose`. To generate the adaptive gains, we would need to bring up the [nn_controller node](/nn_controller). Do this,

	<pre class="terminal"><code> Terminal 3:$ rosrun nn_controller nn_controller ref_z  ref_pitch ref_roll </code></pre>

	Where <ref_x> represents the desired trajectory we want to raise the head. Otherwise, you could fill out the 3-DOF reference positions in the [controller launch file](/nn_controller/launch/controller.launch)

* Neural Network Function Aproximator

 Previously written in Torch7 as the [farnn](/farnn) package, this portion of the codebase has been migrated to [pyrnn](/pyrnn) in the recently released [pytorch](pytorch) deep nets framework to take advantage of python libraries, cvx and quadratic convex programming for contraints-based adaptive quadratic programming (useful for our adaptive controller).

 * farnn
	
	Running `farnn` would consist of `roscd ing` into the `farnn src` folder and running `th real_time_predictor.lua` command while the [nn_controller](/nn_controller) is running).

 * pyrnn

	`roscd` into `pyrnn src` folder and do `./main.py`



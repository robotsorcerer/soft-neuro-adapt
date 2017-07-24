### Intro

Source codes for my **IROS paper: A 3-DoF Neuro-Adaptive Pose Correcting System For Frameless and Maskless Cancer Radiotherapy**.  Arxiv ID: arXiv:1703.03821

#### Dependencies

The neural network estimator runs in [pytorch](pytorch.org). If you have a ROS distro installation that uses
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
- python 2.7+

Install with 

```bash
	sudo apt-get install python-dev
```

- pytorch
	
The pytorch version of this code only works on a gpu. Tested with cuda 8.0 and cuda run time version 367.49. 
Install with <code><pre class="terminal"><code>Termnal x:$ conda install pytorch torchvision cuda80 -c soumith </code></pre>

- ros

Instructions for installing ros can be found [on the osrf ros website]](http://wiki.ros.org/indigo/Installation/Ubuntu).

- PyPI dependencies 

PyPI installable using:
	<pre class="terminal"><code> Terminal:$	pip install -r requirements.txt </code></pre>


### Vision processing

You could use the vicon system or the ensenso package. Follow these steps to get up and running

* Clone this code: git clone https://github.com/lakehanne/iros2017 --recursive

If you did not clone the package recursively as shown above, you can initialize the `ensenso` and `vicon` submodules as follows:

* Initialize submodules: `git init submodules`
* Update submodules: `git submodules update`

Then compile the codebase with either `catkin_make` or `catkin build`.

#### The Vicon System Option
cd inside the vicon directory and follow the readme instructions there. With the vicon system, you get a more accurate representation of the face. We would want four markers on the face in a rhombic manner (preferrably named `fore`, `left` , `right`, and `chin` to conform with the direction cosines code. With the vicon icp code, which is what we run by default, you would not need this. These two codes extract the facial pose with respect to the scene); make sure the `subject` and `segment` are appropriately named `Superdude/head` in `Nexus`. 
	
<pre class="terminal"><code> Terminal$:	rosrun vicon_bridge vicon.launch</pre></code>

This launches the [adaptive model-following control algorithm](/nn_controller), [icp computation](/vicon_icp/src/vicon_icp.cpp) of head rotation about the table frame and the vicon ros subscriber node.
		
#### The [Ensenso](https://github.com/lakehanne/ensenso) option.

cd inside the ensenso package and follow the README instructions therein. When done, run the controller and sensor face pose collector nodes as

### Running the code

```bash
	 roslaunch nn_controller controller.launch
```

The pose tuple of the face, {z, pitch, roll},  is broadcast on the topic `/mannequine_head/pose`. 

The reference pose that the head should track is located in the [traj.yaml file](/nn_controller/config/traj.yaml). Amend this as you wish.

### Neural Network Function Aproximator

 Previously written in Torch7 as the [farnn](/farnn) package, this portion of the codebase has been migrated to [pyrnn](/pyrnn) in the recently released [pytorch](pytorch) deep nets framework to take advantage of python libraries, cvx and quadratic convex programming for contraints-based adaptive quadratic programming.

 #### farnn
	
	Running `farnn` would consist of `roscd ing` into the `farnn src` folder and running `th real_time_predictor.lua` command while the [nn_controller](/nn_controller) is running).

 #### pyrnn

	`roscd` into `pyrnn src` folder and do `./main.py`



# The TRI-STAR framework for Robot Tool Use

This project developed a framework that learns tool-use rapidly (Star 1), generalize the skills to novel objects Star 2), and transfer the skills to other robot platforms (Star 3).

This is the source code for this paper "Rapidly Learning Generalizable and Robot-Agnostic Tool-Use Skills for a Wide Range of Tasks". Please cite the paper if you use the source code or part of it.

@ARTICLE{qin2021rapidly, \
    AUTHOR={Qin, Meiying and Brawer, Jake and Scassellati, Brian}, \
    TITLE={Rapidly Learning Generalizable and Robot-Agnostic Tool-Use Skills for a Wide Range of Tasks}, \     
    JOURNAL={Frontiers in Robotics and AI}, \
    VOLUME={8}, \   
    PAGES={380}, \
    YEAR={2021}, \
    PUBLISHER={Frontiers}, \
    URL={https://www.frontiersin.org/article/10.3389/frobt.2021.726463}, \
    DOI={10.3389/frobt.2021.726463}, \
    ISSN={2296-9144} \
}

## Publication

*Qin M, *Brawer J and Scassellati B (2021) Rapidly Learning Generalizable and Robot-Agnostic Tool-Use Skills for a Wide Range of Tasks. Front. Robot. AI 8:726463. doi: 10.3389/frobt.2021.726463

\* Equal contribution

## Overview

`control_wrapper`, `tool_substitution` and `tri_star` are ROS repositories, and all python scripts have been tested with python 2 and might not work with python 3. 

## control_wrapper

This repo contains the code for motion control of the robot (e.g., set/get the joint angles, set/get the end effector pose). It has been tested on Ubuntu 14.04 with ROS Indigo and on Ubuntu 18.04 with ROS Kinetic. It uses the moveit lib. For instructions of how to run the code, please see the readme file in the repo.

To work with Baxter, the Baxter API is required other than moveit.

If you need to run tri_star and control_wrapper on different computers. You need to create a directory called "pointcloud" and save all ply files there.

## tri_star

This repo contains the visual perception, object 3D model obtaining, user interface to train and test the robot, as well as the code of learning tool-use (star 1). It has been tested on Ubuntu 18.04 with Ros Kinetic. The code will not be compatible with Ros Indigo on Ubuntu 14.04 as it requires the Open3D lib. For other libs required and detailed instructions of how to run the code, please see the readme file in the repo.

To work with control_wrapper that required to run on Ubuntu 14.04, multiple computers are needed such that one of them is set as the ros master computer. To communicate between multiple computers, please refer to the instructions: <http://wiki.ros.org/ROS/Tutorials/MultipleMachines>


## tool_substitution

This repo contains the logic of object subsitution. The tri-star will call this repo, so that this repo doesn't need to be run separately. It has been tested on Ubuntu 18.04 with Ros Kinetic. For other libs required, please see the readme file in the repo.

## batch_segmentation

This repo contains the code that segment the tool triangulated mesh. It is adapted from <https://github.com/pauloabelha/batch_segmentation> with modifications. For other libs required and detailed instructions of how to run the code, please see the readme file in the repo.

# 2_wheels_RL

Works in ROS Noetic - Gazebo 11 - build tools. 

After clone this repo, go to /final/launch/gazebo.launch and change the repo of include file in line 4 to appropriate directory to rb3.launch.

Run balance_train.py for training the model. 

Create folder named /models_B in the same directory to save the .pth file.

Modify GAZEBO_RESOURCE_PATH to path to rb3.launch if needed. 

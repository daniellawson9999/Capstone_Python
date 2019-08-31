# Capstone_Python
![a project graphic](https://res.cloudinary.com/dpanlycrj/image/upload/v1556238157/Copy_of_visual_rf4hqc.png)

Python Reinforcement Learning for FTC

## Project Description

This project was originally created for a senior year capstone project. My inititial goal was to create a system that would create the foundation for a fully autonomous teleop program. Now that the school project has ended, I plan on creating a little bit of documentation in case other people would like to contribute to this project. 

Currently, this project contains a large portion of code unrelated to the project and test and legacy code. Additionally, the sctructure of the code could be much cleaner. If other individuals are interesting in working on the project, this will become an a major area to work on. 

Right now, some results have been scucesfully transfered to real life. There are a lot of different approaches I would like to try, which I will be working on while I still have time. 

## Framework

This project is also a reinforcement learning framework using tensorflow-keras where I implement new algorithms as I learn them. Currently, the project is just
compatible with the custom mayavi environment, however there will soon be suppport for OpenAI gym environments. Currently, gym environments
with single discrete action spaces that are epoch-based should work, however this has not been tested. 

The framework supports training arugument loading from files and arguments for the mayavi environment from files. Addtionally,
you can asynchronously train multiple agents at once to test differnet hyperparameters and algorithms. 

Here is list of implemented algorithms:

* Deep Q Networks (DQN) with state-action, state, and multi-state (frame stacking) model input
* Improvements such as Dueling and Double (target) networks
* Deep Recurrent Q Networks (DQRN) 
* Combining algorithms, such as Deep Double Dueling Recurrent Q Networks (DDDQRN)


## Project Structure 
coming soon, main project code is contained within: https://github.com/daniellawson9999/Capstone_Python/tree/master/Robot/Rotation

## Resources

For now, here are some slides and videos about the project. I can attach scripts soon:

* [First Presentation](https://drive.google.com/open?id=1XcURH9AAJknkbxcPnv8Txodlpo6b9Jzu)
* [Most Recent Presentation](https://drive.google.com/open?id=1J4Uh5nRSOMGiJ5obVCKwYNi3b4-Cobms)
* [Final Presentation](https://drive.google.com/file/d/1aDaDzX2CszAm8dJNNlae2QLGeEyzgKza/view?usp=sharing)
* [Final Presentation Script](https://drive.google.com/open?id=1Bq-ELTKPgTIAyaEwiDUgwDpyd1QiJiBh)
* [Google Drive folder of real life and simulated photos and videos](https://drive.google.com/open?id=1ko3aLJ-0wM7GeuNC-15_1sUnDqcl2LVh)

If you would like to know more about deep q learning, I would recommend the [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) which presented this approach. 
## Setup
Currently, it can be difficult to properly setup your dependencies. I will make a recommended setup guide and will eventually use a service such as docker. 

### robot code repository:
https://github.com/daniellawson9999/RL-Robot-Code

## To-do List
Support for most gym environments
Improved training metrics (reward and performance graphs) and additional tensorboard metrics
Testing in linux in headless environments

Algorithms:
Adding prioritized experience replay (with importance sampling)
Implementing policy based algorithms, starting from:
	Monte Carlo Policy Gradients with REINFORCE
	Actor Critic
	Advantage Actor Critic (A2C)

## Contact 
Discord: Daniel_Lawson9999#7898 (preferred)
Email: daniellawson9999@gmail.com

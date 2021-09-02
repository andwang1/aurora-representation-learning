## Overview

This repository collects two experiments conducted to explore the effect of environment distractions on the representations learned by an AE and the proposed Encoder-Decoder structure (RAED). In both experiments we use the AURORA [1] algorithm to learn robot behaviours.

<code>airhockey</code> simulates an air-hockey table on which a multi-joint planar robot arm is fixed. The robot tries to explore its capability to create distinct air-hockey puck trajectories by interacting with the puck.
The environment distractions are modelled by the appearance of an additional puck that spawns and moves according to a random force vector.

<img src="https://github.com/andwang1/aurora-representation-learning/blob/master/media/two_pucks.gif?raw=true"/>


<code>objectarrangement</code> simulates a room in which the same robot arm instead tries to move a heavy object to different positions in a room. Environment distractions materialise in the appearance of additional objects.

[1] Antoine Cully.  Autonomous skill discovery with Quality-Diversity and Unsu-pervised Descriptors. 9, 2019. pages 1, 2, 5, 23

### Analysis

For a detailled analysis of the experiment results, please refer to the [PDF](https://github.com/andwang1/aurora-representation-learning/tree/master/analysis.pdf) in the repository.


<img src="https://github.com/andwang1/aurora-representation-learning/blob/master/media/ISD_trajectories.png?raw=true"/>


<img src="https://github.com/andwang1/aurora-representation-learning/blob/master/media/ASD_positions_100stoch.png?raw=true"/>

# An HsMM-based forward recursion algorithm for real time indoor localization
# (Development Python Code)

Author
- Shuai Sun  (shuai.sun@student.rmit.edu.au  sunshuai198903@163.com)

## Description

This code is developed as an illustrative example for the simulation section

## Dataset specifications

The dataset consists a training and testing folder with received signal strength (RSS) data collected

There are 8 anchor nodes and 12 regions (see Figure 4 in the paper)

- In the folder _training, it contains 8 * 12 = 96 separate .p2m files, each file
corresponds to a RSS data collected from a grid number of user locations (coordinate 
is given in the .p2m file) with respect to a certain Anchor node

- In the folder _testing, it contains 8 separate .p2m files corresponds to 
a RSS data set collected from a moving user trajectory locations (coordinate is given in 
the .p2m file) with respect to each Anchor node

- The ground truth location region of the user is shown in Figure 4, with
details provided in python file 'HsMM-FRvsFingerprintingSimulation.py'

## Naming Convention (training dataset)

- Notation in the .p2m file: **power.t001_Anchor index_rRegion Index.p2m
Note that the region index is offset by 8, for instance, r009 is region 1
r015 is region 7, and r020 is region 12.

## Naming Convention (Evaluation dataset)
- Notation in the .p2m file: **power.t001_Anchor index_r001.p2m
Note that the region index is not shown in the name as each file
corresponds to all the trajectory data (location is provided in x y z)
in the .p2m file

## File structure

```
Data
└─── _testing		
│   │  	HsMM_Experiment_power.t001_01.r001.p2m
│   │	...
│   │  	HsMM_Experiment_power.t001_01.r001.p2m
│
└─── _training
│   │  	HsMM_Experiment_power.t001_01.r009.p2m
│   │   ...
│   │  	HsMM_Experiment_power.t001_02.r009.p2m
│   │   ...
│   │  	HsMM_Experiment_power.t001_03.r009.p2m
│   │   ...
│   │   ...
│   │  	HsMM_Experiment_power.t001_08.r020.p2m


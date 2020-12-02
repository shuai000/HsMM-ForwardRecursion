# An HsMM-based forward recursion algorithm for real time indoor localization
# (Development Python Code)

Author
- Shuai Sun  (shuai.sun@student.rmit.edu.au,  sunshuai198903@163.com)

## Description

This code is developed as an illustrative example for the simulation of applying
the HsMM-FR algorithm for indoor user region localization

## Dataset specifications

The dataset consists a training and testing folder with received signal strength (RSS) data, obtained
from a ray-tracing software (Wireless Insite)

There are K=8 anchor nodes and N=12 regions 

- In the folder _training, it contains 8 * 12 = 96 separate .p2m files, each file
corresponds to a RSS data collected from a grid number of user locations (coordinate 
is given in the .p2m file) with respect to a certain Anchor node

- In the folder _testing, it contains 8 separate .p2m files corresponds to 
a RSS data set collected from a moving user trajectory locations (coordinate is given in 
the .p2m file) with respect to each Anchor node

- The ground truth location region of the user is shown in Figure 4, with
st_ground = [[1, 8], [2, 18], [3, 23], [4, 36], [5, 46], [6, 62], [8, 78], [7, 120], [9, 129], [10, 138]].
To interpret it, it means the user stays in region 1 for 8 time steps, 
then transits to region 2 with a sojourn time of (18-8)=10 steps, then transits
to region 3 and stay there for (23-18)=5 steps, ... 

## Naming Convention (training dataset)

- Notation in the .p2m file: **power.t001_Anchor index_rRegion Index.p2m
Note that the region index is offset by 8, for instance, r009 is region 1
r015 is region 7, and r020 is region 12.

## Naming Convention (testing dataset)
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


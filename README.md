# Python JSON RPC Server with Crazyflie

This repository contains the source code for a Python JsonRPC server implementation.
Furthermore, this repository provides some functionality of the crazyflie python library via ZMQ and the B Machines that can interface with it.

The code for the JSON RPC server is implemented in the files `zmq_rpc.py` and `cf_rpc.py`.
The server can be started with:

```
python3 cf_rpc.py
```

## Real World Scenarios

### HTML Scenarios with Videos

[Scenario 3](https://hhu-stups.github.io/cf-srv/DroneMainController_3_video)

[Scenario 22](https://hhu-stups.github.io/cf-srv/DroneMainController_22_video)

### More Scenarios

[Scenario 1](https://hhu-stups.github.io/cf-srv/DroneMainController_1_video)

[Scenario 2](https://hhu-stups.github.io/cf-srv/DroneMainController_2_video)

[Scenario 4](https://hhu-stups.github.io/cf-srv/DroneMainController_4_video)

[Scenario 5](https://hhu-stups.github.io/cf-srv/DroneMainController_5_video)

[Scenario 6](https://hhu-stups.github.io/cf-srv/DroneMainController_6_video)

[Scenario 7](https://hhu-stups.github.io/cf-srv/DroneMainController_7_video)

[Scenario 8](https://hhu-stups.github.io/cf-srv/DroneMainController_8_video)

[Scenario 9](https://hhu-stups.github.io/cf-srv/DroneMainController_9_video)

[Scenario 10](https://hhu-stups.github.io/cf-srv/DroneMainController_10_video)

[Scenario 11](https://hhu-stups.github.io/cf-srv/DroneMainController_11_video)

[Scenario 12](https://hhu-stups.github.io/cf-srv/DroneMainController_12_video)

[Scenario 13](https://hhu-stups.github.io/cf-srv/DroneMainController_13_video)

[Scenario 14](https://hhu-stups.github.io/cf-srv/DroneMainController_14_video)

[Scenario 17](https://hhu-stups.github.io/cf-srv/DroneMainController_17_video)

[Scenario 18](https://hhu-stups.github.io/cf-srv/DroneMainController_18_video)

[Scenario 19](https://hhu-stups.github.io/cf-srv/DroneMainController_19_video)

[Scenario 20](https://hhu-stups.github.io/cf-srv/DroneMainController_20_video)

[Scenario 21](https://hhu-stups.github.io/cf-srv/DroneMainController_21_video)

[Scenario 22](https://hhu-stups.github.io/cf-srv/DroneMainController_22_video)





## Interaction with Crazyflie Server

One can interact with the Crazyflie server through the `DroneCommunicator.mch` B model, which can be loaded in the formal method tool 
`ProB2-UI` (https://github.com/hhu-stups/prob2_ui).
The available commands to control the drones are as follows:

| Command                     | Description                                             |
|-----------------------------|---------------------------------------------------------|
| Init                        | Initialize communication with Crazyflie server          |
| Destroy                     | Destroys connection to socket                           |
| open_link                   | Opens connection to a drone                             |
| close_link                  | Closes connection to a drone                            |
| register_sensors            | Registers sensors of drone                              |
|                             |                                                         |
| Drone_Takeoff               | Sends command to drone for taking off                   |
| Drone_Land                  | Sends command to drone to land                          |
| Drone_Left(dist)            | Sends command to drone to fly left                      |
| Drone_Right(dist)           | Sends command to drone to fly right                     |
| Drone_Up(dist)              | Sends command to drone to fly upward                    |
| Drone_Downward(dist)        | Sends command to drone to fly downward                  |
| Drone_Forward(dist)         | Sends command to drone to fly forward                   |
| Drone_Backward(dist)        | Sends command to drone to fly backward                  |
|                             |                                                         |
| Drone_GetLeftDistance       | Read distance sensor to the left                        |
| Drone_GetRightDistance      | Read distance sensor to the right                       |
| Drone_GetUpDistance         | Read distance sensor to the up                          |
| Drone_GetDownDistance       | Read distance sensor to the down                        |
| Drone_GetForwardDistance    | Read distance sensor to the forward                     |
| Drone_GetBackwardDistance   | Read distance sensor to the backward                    |
|                             |                                                         |
| Drone_GetX                  | Read x position of drone                                |
| Drone_GetY                  | Read y position of drone                                |
| Drone_GetZ                  | Read z position of drone                                |


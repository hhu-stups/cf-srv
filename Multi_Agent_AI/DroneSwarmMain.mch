/*
Developed by Fabian Vu
*/

MACHINE DroneSwarmMain
INCLUDES DroneSwarm, DroneCommunicator_Multi_Mockup
CONSTANTS DroneAddresses
PROPERTIES DroneAddresses = {1 |-> "radio://0/80/2M/5700B500F4", 2 |-> "radio://0/80/2M/5700B500F7"}
VARIABLES flying
INVARIANT flying : BOOL

INITIALISATION
       Init;
       open_link(DroneAddresses(1));
       open_link(DroneAddresses(2));
       register_sensors(DroneAddresses(1));
       register_sensors(DroneAddresses(2));
       flying := FALSE

OPERATIONS

    MAIN_TAKEOFF =
    SELECT flying = FALSE THEN
      Drone_Takeoff(DroneAddresses(current_drone));
      flying := TRUE
    END;

    MAIN_LAND =
    SELECT flying = TRUE THEN
      Drone_Takeoff(DroneAddresses(current_drone));
      flying := FALSE
    END;

    MAIN_LEFT =
    SELECT flying = TRUE THEN
      Move_Left;
      Drone_Left(DroneAddresses(current_drone), 500)
    END;

    MAIN_RIGHT =
    SELECT flying = TRUE THEN
      Move_Right;
      Drone_Right(DroneAddresses(current_drone), 500)
    END;

    MAIN_FORWARD =
    SELECT flying = TRUE THEN
      Move_Forward;
      Drone_Forward(DroneAddresses(current_drone), 500)
    END;

    MAIN_DOWN =
    SELECT flying = TRUE THEN
      Move_Backward;
      Drone_Backward(DroneAddresses(current_drone), 500)
    END;

    MAIN_DESTORY =
    SELECT flying = FALSE THEN
      Destroy
    END


END
/*
Developed by Fabian Vu
*/

MACHINE DroneSwarmMain
INCLUDES DroneSwarm, DroneCommunicator_Multi_Mockup
CONSTANTS DroneAddresses
PROPERTIES DroneAddresses = {1 |-> "radio://0/80/2M/5700B500F4", 2 |-> "radio://0/80/2M/5700B500F7"}
VARIABLES flying
INVARIANT 
	flying : 1..n --> BOOL

INITIALISATION
       Init;
       open_link(DroneAddresses(1));
       open_link(DroneAddresses(2));
       register_sensors(DroneAddresses(1));
       register_sensors(DroneAddresses(2));
       flying := (1..n) * {FALSE}

OPERATIONS

    MAIN_TAKEOFF(current_drone) =
    SELECT current_drone : 1..n & flying(current_drone) = FALSE THEN
      Drone_Takeoff(DroneAddresses(current_drone));
      flying(current_drone) := TRUE
    END;

    MAIN_LAND(current_drone) =
    SELECT current_drone : 1..n & flying(current_drone) = TRUE THEN
      Drone_Takeoff(DroneAddresses(current_drone));
      flying(current_drone) := FALSE
    END;

    MAIN_LEFT(current_drone) =
    SELECT current_drone : 1..n & flying(current_drone) = TRUE THEN
      Move_Left(current_drone);
      Drone_Left(DroneAddresses(current_drone), 500)
    END;

    MAIN_RIGHT(current_drone) =
    SELECT current_drone : 1..n & flying(current_drone) = TRUE THEN
      Move_Right(current_drone);
      Drone_Right(DroneAddresses(current_drone), 500)
    END;

    MAIN_FORWARD(current_drone) =
    SELECT current_drone : 1..n & flying(current_drone) = TRUE THEN
      Move_Forward(current_drone);
      Drone_Forward(DroneAddresses(current_drone), 500)
    END;

    MAIN_DOWN(current_drone) =
    SELECT current_drone : 1..n & flying(current_drone) = TRUE THEN
      Move_Backward(current_drone);
      Drone_Backward(DroneAddresses(current_drone), 500)
    END;

    MAIN_DESTORY =
    SELECT ran(flying) = {FALSE} THEN
      Destroy
    END


END
/*
Developed by Fabian Vu
*/

MACHINE DroneSwarmMain
SETS DIRECTIONS = {BACKWARD, FORWARD, LEFT, RIGHT, UP}
DEFINITIONS
  "LibraryZMQ_RPC.def";
  "LibraryReals.def";
  SET_PREF_MAX_OPERATIONS == 0
INCLUDES DroneSwarm, DroneCommunicator_Multi
CONSTANTS DroneAddresses, DISTANCE_ONE_STEP, SAFETY_DISTANCE
PROPERTIES 
	DroneAddresses = {1 |-> "radio://0/80/2M/5700B500F5", 2 |-> "radio://0/80/2M/5700B500FA"} &
	DISTANCE_ONE_STEP = 200 &
	SAFETY_DISTANCE = 400
VARIABLES flying, observations, current_drone, maybe_obstacles
INVARIANT 
	flying : 1..n --> BOOL &
	observations : POW((1..n) * POW(DIRECTIONS * INTEGER)) &
	current_drone : 1..n &
	maybe_obstacles : POW((0..m-1) * (0..m-1))

INITIALISATION
	flying := (1..n) * {FALSE} ||
	observations := {} ||
	current_drone := 1 ||
	maybe_obstacles := {}

OPERATIONS

    MAIN_INIT =
    SELECT init = FALSE THEN
       Init;
       open_link(DroneAddresses(1));
       open_link(DroneAddresses(2));
       register_sensors(DroneAddresses(1));
       register_sensors(DroneAddresses(2))
    END;

    MAIN_CONTROL_DRONE(drone) =
    SELECT drone : 1..n THEN
        current_drone := drone
    END;

    MAIN_TAKEOFF =
    SELECT
      flying(current_drone) = FALSE &
      current_drone : dom(observations) &
      dom(observations(current_drone)) = DIRECTIONS &
      observations(current_drone)(UP) > SAFETY_DISTANCE + DISTANCE_ONE_STEP
    THEN
      Drone_Takeoff(DroneAddresses(current_drone));
      flying(current_drone) := TRUE;
      observations := {current_drone} <<| observations
    END;

    MAIN_LAND =
    SELECT
      flying(current_drone) = TRUE
    THEN
      Drone_Land(DroneAddresses(current_drone));
      flying(current_drone) := FALSE;
      observations := {current_drone} <<| observations
    END;

    MAIN_LEFT =
    SELECT
      flying(current_drone) = TRUE &
      current_drone : dom(observations) &
      dom(observations(current_drone)) = DIRECTIONS &
      observations(current_drone)(LEFT) > SAFETY_DISTANCE + DISTANCE_ONE_STEP
    THEN
      Move_Left(current_drone);
      Drone_Left(DroneAddresses(current_drone), DISTANCE_ONE_STEP);
      observations := {current_drone} <<| observations;
      maybe_obstacles := maybe_obstacles - {prj1(drone_positions(current_drone)) |-> prj2(drone_positions(current_drone))}
    END;

    MAIN_RIGHT =
    SELECT
      flying(current_drone) = TRUE &
      current_drone : dom(observations) &
      dom(observations(current_drone)) = DIRECTIONS &
      observations(current_drone)(RIGHT) > SAFETY_DISTANCE + DISTANCE_ONE_STEP
    THEN
      Move_Right(current_drone);
      Drone_Right(DroneAddresses(current_drone), DISTANCE_ONE_STEP);
      observations := {current_drone} <<| observations;
      maybe_obstacles := maybe_obstacles - {prj1(drone_positions(current_drone)) |-> prj2(drone_positions(current_drone))}
    END;

    MAIN_FORWARD =
    SELECT
      flying(current_drone) = TRUE &
      current_drone : dom(observations) &
      dom(observations(current_drone)) = DIRECTIONS &
      observations(current_drone)(FORWARD) > SAFETY_DISTANCE + DISTANCE_ONE_STEP
    THEN
      Move_Forward(current_drone);
      Drone_Forward(DroneAddresses(current_drone), DISTANCE_ONE_STEP);
      observations := {current_drone} <<| observations;
      maybe_obstacles := maybe_obstacles - {prj1(drone_positions(current_drone)) |-> prj2(drone_positions(current_drone))}
    END;

    MAIN_BACKWARD =
    SELECT
      flying(current_drone) = TRUE &
      current_drone : dom(observations) &
      dom(observations(current_drone)) = DIRECTIONS &
      observations(current_drone)(BACKWARD) > SAFETY_DISTANCE + DISTANCE_ONE_STEP
    THEN
      Move_Backward(current_drone);
      Drone_Backward(DroneAddresses(current_drone), DISTANCE_ONE_STEP);
      observations := {current_drone} <<| observations;
      maybe_obstacles := maybe_obstacles - {prj1(drone_positions(current_drone)) |-> prj2(drone_positions(current_drone))}
    END;

    MAIN_OBSERVE =
    VAR up_dist, left_dist, right_dist, forward_dist, backward_dist IN
      up_dist <-- Drone_GetUpDistance(DroneAddresses(current_drone));
      left_dist <-- Drone_GetLeftDistance(DroneAddresses(current_drone));
      right_dist <-- Drone_GetRightDistance(DroneAddresses(current_drone));
      forward_dist <-- Drone_GetFrontDistance(DroneAddresses(current_drone));
      backward_dist <-- Drone_GetBackDistance(DroneAddresses(current_drone));
      observations(current_drone) := {UP |-> up_dist, LEFT |-> left_dist, RIGHT |-> right_dist, FORWARD |-> forward_dist, BACKWARD |-> backward_dist};
      LET maybe_obstacle_pos_left, maybe_obstacle_pos_right, maybe_obstacle_pos_forward, maybe_obstacle_pos_backward BE 
        maybe_obstacle_pos_left = prj1(drone_positions(current_drone)) - 1 |-> prj2(drone_positions(current_drone)) &
	maybe_obstacle_pos_right = prj1(drone_positions(current_drone)) + 1 |-> prj2(drone_positions(current_drone)) &
        maybe_obstacle_pos_forward = prj1(drone_positions(current_drone)) |-> prj2(drone_positions(current_drone)) - 1 &
	maybe_obstacle_pos_backward = prj1(drone_positions(current_drone)) |-> prj2(drone_positions(current_drone)) + 1
      IN
        IF prj1(drone_positions(current_drone)) > 0 & left_dist <= SAFETY_DISTANCE + DISTANCE_ONE_STEP & maybe_obstacle_pos_left /: ran(visited) THEN
          maybe_obstacles := maybe_obstacles \/ {maybe_obstacle_pos_left}
        END;
        IF prj1(drone_positions(current_drone)) < m-1 & right_dist <= SAFETY_DISTANCE + DISTANCE_ONE_STEP & maybe_obstacle_pos_right /: ran(visited) THEN
          maybe_obstacles := maybe_obstacles \/ {maybe_obstacle_pos_right}
        END;
        IF prj2(drone_positions(current_drone)) > 0 & forward_dist <= SAFETY_DISTANCE + DISTANCE_ONE_STEP & maybe_obstacle_pos_forward /: ran(visited) THEN
          maybe_obstacles := maybe_obstacles \/ {maybe_obstacle_pos_forward}
        END;
        IF prj2(drone_positions(current_drone)) < m-1 & backward_dist <= SAFETY_DISTANCE + DISTANCE_ONE_STEP & maybe_obstacle_pos_backward /: ran(visited) THEN
          maybe_obstacles := maybe_obstacles \/ {maybe_obstacle_pos_backward}
        END
      END
    END;

    MAIN_DESTORY =
    SELECT ran(flying) = {FALSE}
    THEN
      Destroy
    END


END
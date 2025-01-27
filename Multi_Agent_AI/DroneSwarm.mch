/*
Developed by Fabian Vu
*/
MACHINE DroneSwarm

VARIABLES 
  visited,
  drone_positions

CONSTANTS m, n
PROPERTIES m = 4 & n = 2

INVARIANT
  visited : POW((1..n) * ((0..m-1) * (0..m-1))) &
  drone_positions : 1..n --> (0..m-1) * (0..m-1)


INITIALISATION
  drone_positions :: 1..n --> (0..m-1) * (0..m-1);
  visited := drone_positions

OPERATIONS
  Move_Left(current_drone) =
  SELECT
    current_drone : 1..n &
    prj1(drone_positions(current_drone)) > 0 & 
    prj1(drone_positions(current_drone)) - 1 |-> prj2(drone_positions(current_drone)) /: ran(drone_positions)
  THEN
    drone_positions(current_drone) := prj1(drone_positions(current_drone)) - 1 |-> prj2(drone_positions(current_drone));
    visited := visited \/ (drone_positions |>> ran(visited))
  END;

  Move_Backward(current_drone) =
  SELECT
    current_drone : 1..n &
    prj2(drone_positions(current_drone)) < m-1 &
    prj1(drone_positions(current_drone)) |-> prj2(drone_positions(current_drone)) + 1 /: ran(drone_positions)
  THEN
    drone_positions(current_drone) := prj1(drone_positions(current_drone)) |-> prj2(drone_positions(current_drone)) + 1;
    visited := visited \/ (drone_positions |>> ran(visited))
  END;

  Move_Right(current_drone) =
  SELECT
    current_drone : 1..n &
    prj1(drone_positions(current_drone)) < m-1 &
    prj1(drone_positions(current_drone)) + 1 |-> prj2(drone_positions(current_drone)) /: ran(drone_positions)
  THEN
    drone_positions(current_drone) := prj1(drone_positions(current_drone)) + 1 |-> prj2(drone_positions(current_drone));
    visited := visited \/ (drone_positions |>> ran(visited))
  END;

  Move_Forward(current_drone) =
  SELECT
    current_drone : 1..n &
    prj2(drone_positions(current_drone)) > 0 &
    prj1(drone_positions(current_drone)) |-> prj2(drone_positions(current_drone)) - 1 /: ran(drone_positions)
  THEN
    drone_positions(current_drone) := prj1(drone_positions(current_drone)) |-> prj2(drone_positions(current_drone)) - 1;
    visited := visited \/ (drone_positions |>> ran(visited))
  END


END
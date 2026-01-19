MACHINE DroneCommunicator_Param_replay(drone_url)
CONSTRAINTS
    drone_url : STRING
DEFINITIONS
  "LibraryReals.def";
  SET_PREF_OPERATION_REUSE == FALSE;
  SET_PREF_MAX_OPERATIONS == 0
VARIABLES init, socket, cycle
INVARIANT
  init:BOOL &
  socket:INTEGER &
  cycle:INTEGER
INITIALISATION
  init := FALSE;
  socket :: NATURAL;
  cycle := 1

OPERATIONS
  Init = 
  SELECT init = FALSE THEN
    init := TRUE
  END;
  
  open_link = 
  SELECT init = TRUE THEN
    skip
  END;

  /*
  Available commanders:
  "motion" - cflib.positioning.motion_commander.MotionCommander
  "high_level" - cflib.crazyflie.high_level_commander.HighLevelCommander
  "position_high_level" (the default) - cflib.positioning.position_hl_commander.PositionHlCommander
  */
  open_link_with_commander_type(commander) = 
  SELECT
    init = TRUE
    & commander : STRING
  THEN
    skip
  END;

  close_link = 
  SELECT init = TRUE THEN
    skip
  END;

  close_link_emergency_stop = 
  SELECT init = TRUE THEN
    skip
  END;

  register_sensors = 
  SELECT init = TRUE THEN
    skip
  END;

  Drone_Takeoff = 
  SELECT init = TRUE THEN
    skip
  END;

  Drone_Land = 
  SELECT init = TRUE THEN
    skip
  END;

  Drone_Left(dist) = 
  SELECT
    init = TRUE &
    dist : 0..2000
  THEN
    skip
  END;

  Drone_Right(dist) = 
  SELECT
    init = TRUE &
    dist : 0..2000
  THEN
    skip
  END;

  Drone_Up(dist) = 
  SELECT
    init = TRUE &
    dist : 0..2000
  THEN
    skip
  END;

  Drone_Down(dist) = 
  SELECT
    init = TRUE &
    dist : 0..2000
  THEN
    skip
  END;

  Drone_Forward(dist) = 
  SELECT
    init = TRUE &
    dist : 0..2000
  THEN
    skip
  END;

  Drone_Backward(dist) = 
  SELECT
    init = TRUE &
    dist : 0..2000
  THEN
    skip
  END;

  Drone_Turn_Left(degrees) = 
  SELECT
    init = TRUE &
    degrees : -360..360
  THEN
    skip
  END;

  Drone_Turn_Right(degrees) = 
  SELECT
    init = TRUE &
    degrees : -360..360
  THEN
    skip
  END;

  out <-- Drone_GetX = 
  SELECT init = TRUE THEN
    out :: NATURAL;
    cycle := cycle + 1
  END;

  out <-- Drone_GetY = 
  SELECT init = TRUE THEN
    out :: NATURAL;
    cycle := cycle + 1
  END;

  out <-- Drone_GetZ = 
  SELECT init = TRUE THEN
    out :: NATURAL;
    cycle := cycle + 1
  END;

  out <-- Drone_GetBattery = 
  SELECT init = TRUE THEN
    out :: NATURAL;
    cycle := cycle + 1
  END;

  out <-- Drone_GetFrontDistance = 
  SELECT init = TRUE THEN
    out :: NATURAL;
    cycle := cycle + 1
  END;

  out <-- Drone_GetBackDistance = 
  SELECT init = TRUE THEN
    out :: NATURAL;
    cycle := cycle + 1
  END;

  out <-- Drone_GetUpDistance = 
  SELECT init = TRUE THEN
    out :: NATURAL;
    cycle := cycle + 1
  END;

  out <-- Drone_GetDownDistance = 
  SELECT init = TRUE THEN
    out :: NATURAL;
    cycle := cycle + 1
  END;

  out <-- Drone_GetLeftDistance = 
  SELECT init = TRUE THEN
    out :: NATURAL;
    cycle := cycle + 1
  END;

  out <-- Drone_GetRightDistance = 
  SELECT init = TRUE THEN
    out :: NATURAL;
    cycle := cycle + 1
  END;

  out <-- Drone_GetRoll = 
  SELECT init = TRUE THEN
    out :: NATURAL;
    cycle := cycle + 1
  END;

  out <-- Drone_GetPitch = 
  SELECT init = TRUE THEN
    out :: NATURAL;
    cycle := cycle + 1
  END;

  out <-- Drone_GetYaw = 
  SELECT init = TRUE THEN
    out :: NATURAL;
    cycle := cycle + 1
  END;

  Drone_Beep(effect) =
  SELECT
    init = TRUE &
    effect : NATURAL
  THEN
    cycle := cycle + 1
  END;

  Destroy =
  SELECT init = TRUE THEN
    init := FALSE
  END
END

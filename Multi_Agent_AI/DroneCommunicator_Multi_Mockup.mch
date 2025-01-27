MACHINE DroneCommunicator_Multi_Mockup USES LibraryJSON, LibraryZMQ_RPC
DEFINITIONS
  "LibraryZMQ_RPC.def";
  "LibraryReals.def";
  SET_PREF_MAX_OPERATIONS == 0;
VARIABLES init, socket, cycle
INVARIANT
  init:BOOL &
  socket:SOCKET &
  cycle:INTEGER
INITIALISATION
  init:=FALSE;
  socket:=0;
  cycle:=1

OPERATIONS
  Init = 
  SELECT init = FALSE THEN
    init := TRUE
  END;

  open_link(DRONE_URL) =
  SELECT init = TRUE & DRONE_URL : STRING THEN
    skip
  END;

  /*
  Available commanders:
  "motion" - cflib.positioning.motion_commander.MotionCommander
  "high_level" - cflib.crazyflie.high_level_commander.HighLevelCommander
  "position_high_level" (the default) - cflib.positioning.position_hl_commander.PositionHlCommander
  */
  open_link_with_commander_type(DRONE_URL, commander) =
  SELECT
    init = TRUE
    & DRONE_URL : STRING
    & commander : STRING
  THEN
    skip
  END;

  close_link(DRONE_URL) =
  SELECT init = TRUE & DRONE_URL : STRING THEN
    skip
  END;

  register_sensors(DRONE_URL) =
  SELECT init = TRUE & DRONE_URL : STRING THEN
    skip
  END;

  Drone_Takeoff(DRONE_URL) =
  SELECT init = TRUE & DRONE_URL : STRING THEN
    cycle := cycle + 1
  END;

  Drone_Land(DRONE_URL) =
  SELECT init = TRUE & DRONE_URL : STRING THEN
    cycle := cycle + 1
  END;

  Drone_Left(DRONE_URL, dist) =
  SELECT
    init = TRUE
    & DRONE_URL : STRING
    & dist : 0..2000
  THEN
    cycle := cycle + 1
  END;

  Drone_Right(DRONE_URL, dist) =
  SELECT
    init = TRUE
    & DRONE_URL : STRING
    & dist : 0..2000
  THEN
    cycle := cycle + 1
  END;

  Drone_Up(DRONE_URL, dist) =
  SELECT
    init = TRUE
    & DRONE_URL : STRING
    & dist : 0..2000
  THEN
    cycle := cycle + 1
  END;

  Drone_Down(DRONE_URL, dist) =
  SELECT
    init = TRUE
    & DRONE_URL : STRING
    & dist : 0..2000
  THEN
    cycle := cycle + 1
  END;

  Drone_Forward(DRONE_URL, dist) =
  SELECT
    init = TRUE
    & DRONE_URL : STRING
    & dist : 0..2000
  THEN
    cycle := cycle + 1
  END;

  Drone_Backward(DRONE_URL, dist) =
  SELECT
    init = TRUE
    & DRONE_URL : STRING
    & dist : 0..2000
  THEN
    cycle := cycle + 1
  END;

  Drone_Turn_Left(DRONE_URL, degrees) = 
  SELECT
    init = TRUE &
    DRONE_URL : STRING &
    degrees : -360..360
  THEN
    cycle := cycle + 1
  END;

  Drone_Turn_Right(DRONE_URL, degrees) = 
  SELECT
    init = TRUE &
    DRONE_URL : STRING &
    degrees : -360..360
  THEN
    cycle := cycle + 1
  END;

  out <-- Drone_GetX(DRONE_URL) =
  SELECT init = TRUE & DRONE_URL : STRING THEN
    out := 2000;
    cycle := cycle + 1
  END;

  out <-- Drone_GetY(DRONE_URL) =
  SELECT init = TRUE & DRONE_URL : STRING THEN
    out := 2000;
    cycle := cycle + 1
  END;

  out <-- Drone_GetZ(DRONE_URL) =
  SELECT init = TRUE & DRONE_URL : STRING THEN
    out := 2000;
    cycle := cycle + 1
  END;

  out <-- Drone_GetBattery(DRONE_URL) =
  SELECT init = TRUE & DRONE_URL : STRING THEN
    out := 100;
    cycle := cycle + 1
  END;

  out <-- Drone_GetFrontDistance(DRONE_URL) =
  SELECT init = TRUE & DRONE_URL : STRING THEN
    out := 2000;
    cycle := cycle + 1
  END;

  out <-- Drone_GetBackDistance(DRONE_URL) =
  SELECT init = TRUE & DRONE_URL : STRING THEN
    out := 2000;
    cycle := cycle + 1
  END;

  out <-- Drone_GetUpDistance(DRONE_URL) =
  SELECT init = TRUE & DRONE_URL : STRING THEN
    out := 2000;
    cycle := cycle + 1
  END;

  out <-- Drone_GetDownDistance(DRONE_URL) =
  SELECT init = TRUE & DRONE_URL : STRING THEN
    out := 2000;
    cycle := cycle + 1
  END;

  out <-- Drone_GetLeftDistance(DRONE_URL) =
  SELECT init = TRUE & DRONE_URL : STRING THEN
    out := 2000;
    cycle := cycle + 1
  END;

  out <-- Drone_GetRightDistance(DRONE_URL) =
  SELECT init = TRUE & DRONE_URL : STRING THEN
    out := 2000;
    cycle := cycle + 1
  END;

  out <-- Drone_GetRoll(DRONE_URL) = 
  SELECT init = TRUE & DRONE_URL : STRING THEN
    out := 0;
    cycle := cycle + 1
  END;

  out <-- Drone_GetPitch(DRONE_URL) = 
  SELECT init = TRUE & DRONE_URL : STRING THEN
    out := 0;
    cycle := cycle + 1
  END;

  out <-- Drone_GetYaw(DRONE_URL) = 
  SELECT init = TRUE & DRONE_URL : STRING THEN
    out := 0;
    cycle := cycle + 1
  END;

  Drone_Beep(DRONE_URL, effect) =
  SELECT
    init = TRUE &
    DRONE_URL : STRING &
    effect : NATURAL
  THEN
    cycle := cycle + 1
  END;

  Destroy = 
  SELECT init = TRUE THEN
    init := FALSE
  END
END

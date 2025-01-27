MACHINE DroneCommunicator_Multi USES LibraryJSON, LibraryZMQ_RPC
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
    socket := ZMQ_RPC_INIT("tcp://localhost:22272");
    init := TRUE
  END;

  open_link(DRONE_URL) =
  SELECT init = TRUE & DRONE_URL : STRING THEN
    VAR res IN
      res := RpcSuccess~(ZMQ_RPC_SEND(socket, "open_link", {("url" |-> JsonString(DRONE_URL)), ("reinit" |-> JsonBoolean(TRUE))}));
      cycle := cycle + 1
    END
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
    VAR res IN
      res := RpcSuccess~(ZMQ_RPC_SEND(socket, "open_link", {("url" |-> JsonString(DRONE_URL)), ("reinit" |-> JsonBoolean(TRUE)), ("commander" |-> JsonString(commander))}));
      cycle := cycle + 1
    END
  END;

  close_link(DRONE_URL) =
  SELECT init = TRUE & DRONE_URL : STRING THEN
    VAR res IN
      res := RpcSuccess~(ZMQ_RPC_SEND(socket, "close_link", {("url" |-> JsonString(DRONE_URL))}));
      cycle := cycle + 1 
    END
  END;

  register_sensors(DRONE_URL) =
  SELECT init = TRUE & DRONE_URL : STRING THEN
    VAR res1, res2, res3 IN
      res1 := RpcSuccess~(ZMQ_RPC_SEND(socket, "register_log", {
        ("url" |-> JsonString(DRONE_URL)),
        ("name" |-> JsonString("Telemetry")),
        ("variables" |-> JsonArray([
          JsonString("stateEstimate.x"),
          JsonString("stateEstimate.y"),
          JsonString("stateEstimate.z"),
          JsonString("stateEstimate.roll"),
          JsonString("stateEstimate.pitch"),
          JsonString("stateEstimate.yaw")
        ]))}));
        res2 := RpcSuccess~(ZMQ_RPC_SEND(socket, "register_log", {
        ("url" |-> JsonString(DRONE_URL)),
        ("name" |-> JsonString("Power")),
        ("variables" |-> JsonArray([
          JsonString("pm.batteryLevel")
        ]))}));
        res3 := RpcSuccess~(ZMQ_RPC_SEND(socket, "register_log", {
        ("url" |-> JsonString(DRONE_URL)),
        ("name" |-> JsonString("Ranging")),
        ("variables" |-> JsonArray([
          JsonString("range.front"),
          JsonString("range.back"),
          JsonString("range.up"),
          JsonString("range.left"),
          JsonString("range.right"),
          JsonString("range.zrange")
        ]))}))
    END
  END;

  Drone_Takeoff(DRONE_URL) =
  SELECT init = TRUE & DRONE_URL : STRING THEN
    VAR res IN
      res := RpcSuccess~(ZMQ_RPC_SEND(socket, "takeoff", {("url" |-> JsonString(DRONE_URL))}));
      cycle := cycle + 1
    END
  END;

  Drone_Land(DRONE_URL) =
  SELECT init = TRUE & DRONE_URL : STRING THEN
    VAR res IN
      res := RpcSuccess~(ZMQ_RPC_SEND(socket, "land", {("url" |-> JsonString(DRONE_URL))}));
      cycle := cycle + 1
    END
  END;

  Drone_Left(DRONE_URL, dist) =
  SELECT
    init = TRUE
    & DRONE_URL : STRING
    & dist : 0..2000
  THEN
    VAR res IN
      res := RpcSuccess~(ZMQ_RPC_SEND(socket, "left", {("url" |-> JsonString(DRONE_URL)), ("distance" |-> JsonNumber(RDIV(real(dist), real(1000))))}));
      cycle := cycle + 1
    END
  END;

  Drone_Right(DRONE_URL, dist) =
  SELECT
    init = TRUE
    & DRONE_URL : STRING
    & dist : 0..2000
  THEN
    VAR res IN
      res := RpcSuccess~(ZMQ_RPC_SEND(socket, "right", {("url" |-> JsonString(DRONE_URL)), ("distance" |-> JsonNumber(RDIV(real(dist), real(1000))))}));
      cycle := cycle + 1
    END
  END;

  Drone_Up(DRONE_URL, dist) =
  SELECT
    init = TRUE
    & DRONE_URL : STRING
    & dist : 0..2000
  THEN
    VAR res IN
      res := RpcSuccess~(ZMQ_RPC_SEND(socket, "up", {("url" |-> JsonString(DRONE_URL)), ("distance" |-> JsonNumber(RDIV(real(dist), real(1000))))}));
      cycle := cycle + 1
    END
  END;

  Drone_Down(DRONE_URL, dist) =
  SELECT
    init = TRUE
    & DRONE_URL : STRING
    & dist : 0..2000
  THEN
    VAR res IN
      res := RpcSuccess~(ZMQ_RPC_SEND(socket, "down", {("url" |-> JsonString(DRONE_URL)), ("distance" |-> JsonNumber(RDIV(real(dist), real(1000))))}));
      cycle := cycle + 1
    END
  END;

  Drone_Forward(DRONE_URL, dist) =
  SELECT
    init = TRUE
    & DRONE_URL : STRING
    & dist : 0..2000
  THEN
    VAR res IN
      res := RpcSuccess~(ZMQ_RPC_SEND(socket, "forward", {("url" |-> JsonString(DRONE_URL)), ("distance" |-> JsonNumber(RDIV(real(dist), real(1000))))}));
      cycle := cycle + 1
    END
  END;

  Drone_Backward(DRONE_URL, dist) =
  SELECT
    init = TRUE
    & DRONE_URL : STRING
    & dist : 0..2000
  THEN
    VAR res IN
      res := RpcSuccess~(ZMQ_RPC_SEND(socket, "backward", {("url" |-> JsonString(DRONE_URL)), ("distance" |-> JsonNumber(RDIV(real(dist), real(1000))))}));
      cycle := cycle + 1
    END
  END;

  Drone_Turn_Left(DRONE_URL, degrees) = 
  SELECT
    init = TRUE &
    degrees : -360..360
  THEN
    VAR res IN
      res := RpcSuccess~(ZMQ_RPC_SEND(socket, "turn_left", {("url" |-> JsonString(DRONE_URL)), ("degrees" |-> JsonNumber(real(degrees)))}));
      cycle := cycle + 1
    END
  END;

  Drone_Turn_Right(DRONE_URL, degrees) = 
  SELECT
    init = TRUE &
    degrees : -360..360
  THEN
    VAR res IN
      res := RpcSuccess~(ZMQ_RPC_SEND(socket, "turn_right", {("url" |-> JsonString(DRONE_URL)), ("degrees" |-> JsonNumber(real(degrees)))}));
      cycle := cycle + 1
    END
  END;

  out <-- Drone_GetX(DRONE_URL) =
  SELECT init = TRUE & DRONE_URL : STRING THEN
    out := floor(1000.0 * JsonNumber~(RpcSuccess~(ZMQ_RPC_SEND(socket, "get_log_var", {("url" |-> JsonString(DRONE_URL)), ("name" |-> JsonString("stateEstimate.x"))}))));
    cycle := cycle + 1
  END;

  out <-- Drone_GetY(DRONE_URL) =
  SELECT init = TRUE & DRONE_URL : STRING THEN
    out := floor(1000.0 *JsonNumber~(RpcSuccess~(ZMQ_RPC_SEND(socket, "get_log_var", {("url" |-> JsonString(DRONE_URL)), ("name" |-> JsonString("stateEstimate.y"))}))));
    cycle := cycle + 1
  END;

  out <-- Drone_GetZ(DRONE_URL) =
  SELECT init = TRUE & DRONE_URL : STRING THEN
    out := floor(1000.0 * JsonNumber~(RpcSuccess~(ZMQ_RPC_SEND(socket, "get_log_var", {("url" |-> JsonString(DRONE_URL)), ("name" |-> JsonString("stateEstimate.z"))}))));
    cycle := cycle + 1
  END;

  out <-- Drone_GetBattery(DRONE_URL) =
  SELECT init = TRUE & DRONE_URL : STRING THEN
    out := floor(JsonNumber~(RpcSuccess~(ZMQ_RPC_SEND(socket, "get_log_var", {("url" |-> JsonString(DRONE_URL)), ("name" |-> JsonString("pm.batteryLevel"))}))));
    cycle := cycle + 1
  END;

  out <-- Drone_GetFrontDistance(DRONE_URL) =
  SELECT init = TRUE & DRONE_URL : STRING THEN
    out := floor(JsonNumber~(RpcSuccess~(ZMQ_RPC_SEND(socket, "get_log_var", {("url" |-> JsonString(DRONE_URL)), ("name" |-> JsonString("range.front"))}))));
    cycle := cycle + 1
  END;

  out <-- Drone_GetBackDistance(DRONE_URL) =
  SELECT init = TRUE & DRONE_URL : STRING THEN
    out := floor(JsonNumber~(RpcSuccess~(ZMQ_RPC_SEND(socket, "get_log_var", {("url" |-> JsonString(DRONE_URL)), ("name" |-> JsonString("range.back"))}))));
    cycle := cycle + 1
  END;

  out <-- Drone_GetUpDistance(DRONE_URL) =
  SELECT init = TRUE & DRONE_URL : STRING THEN
    out := floor(JsonNumber~(RpcSuccess~(ZMQ_RPC_SEND(socket, "get_log_var", {("url" |-> JsonString(DRONE_URL)), ("name" |-> JsonString("range.up"))}))));
    cycle := cycle + 1
  END;

  out <-- Drone_GetDownDistance(DRONE_URL) =
  SELECT init = TRUE & DRONE_URL : STRING THEN
    out := floor(JsonNumber~(RpcSuccess~(ZMQ_RPC_SEND(socket, "get_log_var", {("url" |-> JsonString(DRONE_URL)), ("name" |-> JsonString("range.zrange"))}))));
    cycle := cycle + 1
  END;

  out <-- Drone_GetLeftDistance(DRONE_URL) =
  SELECT init = TRUE & DRONE_URL : STRING THEN
    out := floor(JsonNumber~(RpcSuccess~(ZMQ_RPC_SEND(socket, "get_log_var", {("url" |-> JsonString(DRONE_URL)), ("name" |-> JsonString("range.left"))}))));
    cycle := cycle + 1
  END;

  out <-- Drone_GetRightDistance(DRONE_URL) =
  SELECT init = TRUE & DRONE_URL : STRING THEN
    out := floor(JsonNumber~(RpcSuccess~(ZMQ_RPC_SEND(socket, "get_log_var", {("url" |-> JsonString(DRONE_URL)), ("name" |-> JsonString("range.right"))}))));
    cycle := cycle + 1
  END;

  out <-- Drone_GetRoll(DRONE_URL) = 
  SELECT init = TRUE THEN
    out := floor(1000.0 * JsonNumber~(RpcSuccess~(ZMQ_RPC_SEND(socket, "get_log_var", {("url" |-> JsonString(DRONE_URL)), ("name" |-> JsonString("stateEstimate.roll"))}))));
    cycle := cycle + 1
  END;

  out <-- Drone_GetPitch(DRONE_URL) = 
  SELECT init = TRUE THEN
    out := floor(1000.0 * JsonNumber~(RpcSuccess~(ZMQ_RPC_SEND(socket, "get_log_var", {("url" |-> JsonString(DRONE_URL)), ("name" |-> JsonString("stateEstimate.pitch"))}))));
    cycle := cycle + 1
  END;

  out <-- Drone_GetYaw(DRONE_URL) = 
  SELECT init = TRUE THEN
    out := floor(1000.0 * JsonNumber~(RpcSuccess~(ZMQ_RPC_SEND(socket, "get_log_var", {("url" |-> JsonString(DRONE_URL)), ("name" |-> JsonString("stateEstimate.yaw"))}))));
    cycle := cycle + 1
  END;

  Drone_Beep(DRONE_URL, effect) =
  SELECT
    init = TRUE &
    effect : NATURAL
  THEN
    VAR res IN
      res := RpcSuccess~(ZMQ_RPC_SEND(socket, "set_value", {("url" |-> JsonString(DRONE_URL)), ("name" |-> JsonString("sound.effect")), ("value" |-> JsonNumber(real(effect)))}));
      cycle := cycle + 1
    END
  END;


  Destroy = 
  SELECT init = TRUE THEN
    ZMQ_RPC_DESTROY(socket);
    init := FALSE
  END
END

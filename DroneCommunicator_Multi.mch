MACHINE DroneCommunicator_Multi USES LibraryZMQ_RPC
DEFINITIONS
  "LibraryZMQ_RPC.def";
  "LibraryIO.def";
  "LibraryReals.def";
  SET_PREF_MAX_OPERATIONS == 0;
VARIABLES init, socket, xx, cycle
INVARIANT
  init:BOOL &
  socket:SOCKET &
  xx:INTEGER &
  cycle : INTEGER
INITIALISATION init:=FALSE; socket:=0; xx:=0; cycle := 1
OPERATIONS
  Init = PRE init=FALSE THEN socket := ZMQ_RPC_INIT("tcp://localhost:22272"); init := TRUE END;
  
  open_link(DRONE_URL) =
  SELECT init = TRUE & DRONE_URL : STRING THEN
    VAR res IN
      res := RpcSuccess~(ZMQ_RPC_SEND(socket, "open_link", {("url" |-> RpcString(DRONE_URL)), ("reinit" |-> RpcBoolean(TRUE))}));
      cycle := cycle + 1
    END
  END;

  close_link(DRONE_URL) =
  SELECT init = TRUE & DRONE_URL : STRING THEN
    VAR res IN
      res := RpcSuccess~(ZMQ_RPC_SEND(socket, "close_link", {("url" |-> RpcString(DRONE_URL)), ("reinit" |-> RpcBoolean(TRUE))}));
      cycle := cycle + 1 
    END
  END;

  register_sensors(DRONE_URL) =
  SELECT init = TRUE & DRONE_URL : STRING THEN
    VAR res IN
      res := RpcSuccess~(ZMQ_RPC_SEND(socket, "register_log", {("url" |-> RpcString(DRONE_URL)), ("name" |-> RpcString("Position")), ("variables" |-> RpcString("stateEstimate.x"))}));
      res := RpcSuccess~(ZMQ_RPC_SEND(socket, "register_log", {("url" |-> RpcString(DRONE_URL)), ("name" |-> RpcString("Position")), ("variables" |-> RpcString("stateEstimate.y"))}));
      res := RpcSuccess~(ZMQ_RPC_SEND(socket, "register_log", {("url" |-> RpcString(DRONE_URL)), ("name" |-> RpcString("Position")), ("variables" |-> RpcString("stateEstimate.z"))}));
      res := RpcSuccess~(ZMQ_RPC_SEND(socket, "register_log", {("url" |-> RpcString(DRONE_URL)), ("name" |-> RpcString("Position")), ("variables" |-> RpcString("pm.batteryLevel"))}));
      res := RpcSuccess~(ZMQ_RPC_SEND(socket, "register_log", {("url" |-> RpcString(DRONE_URL)), ("name" |-> RpcString("Position")), ("variables" |-> RpcString("range.front"))}));
      res := RpcSuccess~(ZMQ_RPC_SEND(socket, "register_log", {("url" |-> RpcString(DRONE_URL)), ("name" |-> RpcString("Position")), ("variables" |-> RpcString("range.back"))}));
      res := RpcSuccess~(ZMQ_RPC_SEND(socket, "register_log", {("url" |-> RpcString(DRONE_URL)), ("name" |-> RpcString("Position")), ("variables" |-> RpcString("range.up"))}));
      res := RpcSuccess~(ZMQ_RPC_SEND(socket, "register_log", {("url" |-> RpcString(DRONE_URL)), ("name" |-> RpcString("Position")), ("variables" |-> RpcString("range.left"))}));
      res := RpcSuccess~(ZMQ_RPC_SEND(socket, "register_log", {("url" |-> RpcString(DRONE_URL)), ("name" |-> RpcString("Position")), ("variables" |-> RpcString("range.right"))}));
      res := RpcSuccess~(ZMQ_RPC_SEND(socket, "register_log", {("url" |-> RpcString(DRONE_URL)), ("name" |-> RpcString("Position")), ("variables" |-> RpcString("range.zrange"))}));
      cycle := cycle + 1
    END
  END;

  Drone_Takeoff(DRONE_URL) =
  SELECT init = TRUE & DRONE_URL : STRING THEN
    VAR res IN
      res := RpcSuccess~(ZMQ_RPC_SEND(socket, "takeoff", {"url" |-> RpcString(DRONE_URL)}));
      cycle := cycle + 1
    END
  END;

  Drone_Land(DRONE_URL) =
  SELECT init = TRUE & DRONE_URL : STRING THEN
    VAR res IN
      res := RpcSuccess~(ZMQ_RPC_SEND(socket, "land", {"url" |-> RpcString(DRONE_URL)}));
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
      res := RpcSuccess~(ZMQ_RPC_SEND(socket, "left", {("url" |-> RpcString(DRONE_URL)), ("distance" |-> RpcFloat(RDIV(real(dist), real(1000))))}));
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
      res := RpcSuccess~(ZMQ_RPC_SEND(socket, "right", {("url" |-> RpcString(DRONE_URL)), ("distance" |-> RpcFloat(RDIV(real(dist), real(1000))))}));
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
      res := RpcSuccess~(ZMQ_RPC_SEND(socket, "up", {("url" |-> RpcString(DRONE_URL)), ("distance" |-> RpcFloat(RDIV(real(dist), real(1000))))}));
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
      res := RpcSuccess~(ZMQ_RPC_SEND(socket, "down", {("url" |-> RpcString(DRONE_URL)), ("distance" |-> RpcFloat(RDIV(real(dist), real(1000))))}));
      cycle := cycle + 1
    END
  END;

  Drone_Forward(DRONE_URL, dist) =
  SELECT
    init = TRUE
    & DRONE_URL : STRING
    & dist : 0..2000
  THEN
    PRINT("B: send forward");
    VAR res IN
      res := RpcSuccess~(ZMQ_RPC_SEND(socket, "forward", {("url" |-> RpcString(DRONE_URL)), ("distance" |-> RpcFloat(RDIV(real(dist), real(1000))))}));
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
      res := RpcSuccess~(ZMQ_RPC_SEND(socket, "backward", {("url" |-> RpcString(DRONE_URL)), ("distance" |-> RpcFloat(RDIV(real(dist), real(1000))))}));
      cycle := cycle + 1
    END
  END;

  out <-- Drone_GetX(DRONE_URL) =
  SELECT init = TRUE & DRONE_URL : STRING THEN
    out := floor(1000.0 * RpcFloat~(RpcSuccess~(ZMQ_RPC_SEND(socket, "get_log_var", {("url" |-> RpcString(DRONE_URL)), ("name" |-> RpcString("stateEstimate.x"))}))));
    cycle := cycle + 1
  END;

  out <-- Drone_GetY(DRONE_URL) =
  SELECT init = TRUE & DRONE_URL : STRING THEN
    out := floor(1000.0 *RpcFloat~(RpcSuccess~(ZMQ_RPC_SEND(socket, "get_log_var", {("url" |-> RpcString(DRONE_URL)), ("name" |-> RpcString("stateEstimate.y"))}))));
    cycle := cycle + 1
  END;


  out <-- Drone_GetZ(DRONE_URL) =
  SELECT init = TRUE & DRONE_URL : STRING THEN
    out := floor(1000.0 * RpcFloat~(RpcSuccess~(ZMQ_RPC_SEND(socket, "get_log_var", {("url" |-> RpcString(DRONE_URL)), ("name" |-> RpcString("stateEstimate.z"))}))));
    cycle := cycle + 1
  END;

  out <-- Drone_GetBattery(DRONE_URL) =
  SELECT init = TRUE & DRONE_URL : STRING THEN
    out := RpcInteger~(RpcSuccess~(ZMQ_RPC_SEND(socket, "get_log_var", {("url" |-> RpcString(DRONE_URL)), ("name" |-> RpcString("pm.batteryLevel"))})));
    cycle := cycle + 1
  END;

  out <-- Drone_GetFrontDistance(DRONE_URL) =
  SELECT init = TRUE & DRONE_URL : STRING THEN
    out := RpcInteger~(RpcSuccess~(ZMQ_RPC_SEND(socket, "get_log_var", {("url" |-> RpcString(DRONE_URL)), ("name" |-> RpcString("range.front"))})));
    cycle := cycle + 1
  END;

  out <-- Drone_GetBackDistance(DRONE_URL) =
  SELECT init = TRUE & DRONE_URL : STRING THEN
    out := RpcInteger~(RpcSuccess~(ZMQ_RPC_SEND(socket, "get_log_var", {("url" |-> RpcString(DRONE_URL)), ("name" |-> RpcString("range.back"))})));
    cycle := cycle + 1
  END;

  out <-- Drone_GetUpDistance(DRONE_URL) =
  SELECT init = TRUE & DRONE_URL : STRING THEN
    out := RpcInteger~(RpcSuccess~(ZMQ_RPC_SEND(socket, "get_log_var", {("url" |-> RpcString(DRONE_URL)), ("name" |-> RpcString("range.up"))})));
    cycle := cycle + 1
  END;

  out <-- Drone_GetDownDistance(DRONE_URL) =
  SELECT init = TRUE & DRONE_URL : STRING THEN
    out := RpcInteger~(RpcSuccess~(ZMQ_RPC_SEND(socket, "get_log_var", {("url" |-> RpcString(DRONE_URL)), ("name" |-> RpcString("range.zrange"))})));
    cycle := cycle + 1
  END;

  out <-- Drone_GetLeftDistance(DRONE_URL) =
  SELECT init = TRUE & DRONE_URL : STRING THEN
    out := RpcInteger~(RpcSuccess~(ZMQ_RPC_SEND(socket, "get_log_var", {("url" |-> RpcString(DRONE_URL)), ("name" |-> RpcString("range.left"))})));
    cycle := cycle + 1
  END;

  out <-- Drone_GetRightDistance(DRONE_URL) =
  SELECT init = TRUE & DRONE_URL : STRING THEN
    out := RpcInteger~(RpcSuccess~(ZMQ_RPC_SEND(socket, "get_log_var", {("url" |-> RpcString(DRONE_URL)), ("name" |-> RpcString("range.right"))})));
    cycle := cycle + 1
  END;

  Destroy = PRE init=TRUE THEN ZMQ_RPC_DESTROY(socket); init := FALSE END
END

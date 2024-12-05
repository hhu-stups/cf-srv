MACHINE DroneCommunicator USES LibraryZMQ_RPC
DEFINITIONS
  "LibraryZMQ_RPC.def";
  "LibraryReals.def";
  SET_PREF_MAX_OPERATIONS == 0;
CONSTANTS DRONE_URL
PROPERTIES DRONE_URL:STRING & DRONE_URL = "radio://0/80/2M/5700B500F4"
VARIABLES init, socket, cycle
INVARIANT
  init:BOOL &
  socket:SOCKET &
  cycle:INTEGER
INITIALISATION
  init := FALSE;
  socket := 0;
  cycle := 1

OPERATIONS
  Init = 
  SELECT init = FALSE THEN
    socket := ZMQ_RPC_INIT("tcp://localhost:22272");
    init := TRUE
  END;
  
  open_link = 
  SELECT init = TRUE THEN
    VAR res IN
      res := RpcSuccess~(ZMQ_RPC_SEND(socket, "open_link", {("url" |-> RpcString(DRONE_URL)), ("reinit" |-> RpcBoolean(TRUE))}))
    END
  END;

  close_link = 
  SELECT init = TRUE THEN
    VAR res IN
      res := RpcSuccess~(ZMQ_RPC_SEND(socket, "close_link", {("url" |-> RpcString(DRONE_URL))}))
    END
  END;

  register_sensors = 
  SELECT init = TRUE THEN
    VAR res IN
      res := RpcSuccess~(ZMQ_RPC_SEND(socket, "register_log", {
        ("url" |-> RpcString(DRONE_URL)),
        ("name" |-> RpcString("Telemetry")),
        ("variables" |-> RpcArray([
          RpcString("stateEstimate.x"),
          RpcString("stateEstimate.y"),
          RpcString("stateEstimate.z"),
          RpcString("pm.batteryLevel"),
          RpcString("range.front"),
          RpcString("range.back"),
          RpcString("range.up"),
          RpcString("range.left"),
          RpcString("range.right"),
          RpcString("range.zrange")
        ]))}))
    END
  END;

  Drone_Takeoff = 
  SELECT init = TRUE THEN
    VAR res IN
      res := RpcSuccess~(ZMQ_RPC_SEND(socket, "takeoff", {("url" |-> RpcString(DRONE_URL))}))
    END
  END;

  Drone_Land = 
  SELECT init = TRUE THEN
    VAR res IN
      res := RpcSuccess~(ZMQ_RPC_SEND(socket, "land", {("url" |-> RpcString(DRONE_URL))}))
    END
  END;

  Drone_Left(dist) = 
  SELECT
    init = TRUE &
    dist : 0..2000
  THEN
    VAR res IN
      res := RpcSuccess~(ZMQ_RPC_SEND(socket, "left", {("url" |-> RpcString(DRONE_URL)), ("distance" |-> RpcFloat(RDIV(real(dist), real(1000))))}))
    END
  END;

  Drone_Right(dist) = 
  SELECT
    init = TRUE &
    dist : 0..2000
  THEN
    VAR res IN
      res := RpcSuccess~(ZMQ_RPC_SEND(socket, "right", {("url" |-> RpcString(DRONE_URL)), ("distance" |-> RpcFloat(RDIV(real(dist), real(1000))))}))
    END
  END;

  Drone_Up(dist) = 
  SELECT
    init = TRUE &
    dist : 0..2000
  THEN
    VAR res IN
      res := RpcSuccess~(ZMQ_RPC_SEND(socket, "up", {("url" |-> RpcString(DRONE_URL)), ("distance" |-> RpcFloat(RDIV(real(dist), real(1000))))}))
    END
  END;

  Drone_Down(dist) = 
  SELECT
    init = TRUE &
    dist : 0..2000
  THEN
    VAR res IN
      res := RpcSuccess~(ZMQ_RPC_SEND(socket, "down", {("url" |-> RpcString(DRONE_URL)), ("distance" |-> RpcFloat(RDIV(real(dist), real(1000))))}))
    END
  END;

  Drone_Forward(dist) = 
  SELECT
    init = TRUE &
    dist : 0..2000
  THEN
    VAR res IN
      res := RpcSuccess~(ZMQ_RPC_SEND(socket, "forward", {("url" |-> RpcString(DRONE_URL)), ("distance" |-> RpcFloat(RDIV(real(dist), real(1000))))}))
    END
  END;

  Drone_Backward(dist) = 
  SELECT
    init = TRUE &
    dist : 0..2000
  THEN
    VAR res IN
      res := RpcSuccess~(ZMQ_RPC_SEND(socket, "backward", {("url" |-> RpcString(DRONE_URL)), ("distance" |-> RpcFloat(RDIV(real(dist), real(1000))))}))
    END
  END;

  out <-- Drone_GetX = 
  SELECT init = TRUE THEN
    out := floor(1000.0 * RpcFloat~(RpcSuccess~(ZMQ_RPC_SEND(socket, "get_log_var", {("url" |-> RpcString(DRONE_URL)), ("name" |-> RpcString("stateEstimate.x"))}))));
    cycle := cycle + 1
  END;

  out <-- Drone_GetY = 
  SELECT init = TRUE THEN
    out := floor(1000.0 *RpcFloat~(RpcSuccess~(ZMQ_RPC_SEND(socket, "get_log_var", {("url" |-> RpcString(DRONE_URL)), ("name" |-> RpcString("stateEstimate.y"))}))));
    cycle := cycle + 1
  END;

  out <-- Drone_GetZ = 
  SELECT init = TRUE THEN
    out := floor(1000.0 * RpcFloat~(RpcSuccess~(ZMQ_RPC_SEND(socket, "get_log_var", {("url" |-> RpcString(DRONE_URL)), ("name" |-> RpcString("stateEstimate.z"))}))));
    cycle := cycle + 1
  END;

  out <-- Drone_GetBattery = 
  SELECT init = TRUE THEN
    out := RpcInteger~(RpcSuccess~(ZMQ_RPC_SEND(socket, "get_log_var", {("url" |-> RpcString(DRONE_URL)), ("name" |-> RpcString("pm.batteryLevel"))})));
    cycle := cycle + 1
  END;

  out <-- Drone_GetFrontDistance = 
  SELECT init = TRUE THEN
    out := RpcInteger~(RpcSuccess~(ZMQ_RPC_SEND(socket, "get_log_var", {("url" |-> RpcString(DRONE_URL)), ("name" |-> RpcString("range.front"))})));
    cycle := cycle + 1
  END;

  out <-- Drone_GetBackDistance = 
  SELECT init = TRUE THEN
    out := RpcInteger~(RpcSuccess~(ZMQ_RPC_SEND(socket, "get_log_var", {("url" |-> RpcString(DRONE_URL)), ("name" |-> RpcString("range.back"))})));
    cycle := cycle + 1
  END;

  out <-- Drone_GetUpDistance = 
  SELECT init = TRUE THEN
    out := RpcInteger~(RpcSuccess~(ZMQ_RPC_SEND(socket, "get_log_var", {("url" |-> RpcString(DRONE_URL)), ("name" |-> RpcString("range.up"))})));
    cycle := cycle + 1
  END;

  out <-- Drone_GetDownDistance = 
  SELECT init = TRUE THEN
    out := RpcInteger~(RpcSuccess~(ZMQ_RPC_SEND(socket, "get_log_var", {("url" |-> RpcString(DRONE_URL)), ("name" |-> RpcString("range.zrange"))})));
    cycle := cycle + 1
  END;

  out <-- Drone_GetLeftDistance = 
  SELECT init = TRUE THEN
    out := RpcInteger~(RpcSuccess~(ZMQ_RPC_SEND(socket, "get_log_var", {("url" |-> RpcString(DRONE_URL)), ("name" |-> RpcString("range.left"))})));
    cycle := cycle + 1
  END;

  out <-- Drone_GetRightDistance = 
  SELECT init = TRUE THEN
    out := RpcInteger~(RpcSuccess~(ZMQ_RPC_SEND(socket, "get_log_var", {("url" |-> RpcString(DRONE_URL)), ("name" |-> RpcString("range.right"))})));
    cycle := cycle + 1
  END;

  Destroy =
  SELECT init = TRUE THEN
    ZMQ_RPC_DESTROY(socket);
    init := FALSE
  END
END

MACHINE DroneMainController INCLUDES DroneMovement_Real, DroneCommunicator_Param("radio://0/80/2M/5700B500F3")
DEFINITIONS
    "LibraryIO.def";
    "LibraryMeta.def";
    SET_PREF_OPERATION_REUSE == FALSE;
    MAX_OPERATIONS_INITIALISATION  == 0;
    SET_PREF_MAX_OPERATIONS  == 0;

    //VISB_SVG_FILE == "";
    //VISB_SVG_BOX == rec(width:1000, height:1000);
VARIABLES
    last_time, current_time
INVARIANT
    last_time : NATURAL & current_time : NATURAL
INITIALISATION 
  last_time, current_time := 0, DELTA_WALLTIME;
  Init;
  //open_link;
  open_link_with_commander_type("motion");
  register_sensors
  

OPERATIONS

  MAIN_TAKEOFF = 
  SELECT GET_GUARD_STATUS("TAKEOFF") = TRUE THEN
    TAKEOFF;
    Drone_Takeoff;
    register_time
  END;

  MAIN_LEFT = 
  SELECT GET_GUARD_STATUS("MOVE_LEFT") = TRUE THEN
    MOVE_LEFT;
    Drone_Left(GRID_SIZE);
    register_time
  END;

  MAIN_RIGHT = 
  SELECT GET_GUARD_STATUS("MOVE_RIGHT") = TRUE THEN
    MOVE_RIGHT; 
    Drone_Right(GRID_SIZE);
    register_time
  END;
  
  MAIN_FORWARD = 
  SELECT GET_GUARD_STATUS("MOVE_FORWARD") = TRUE THEN
    MOVE_FORWARD; 
    Drone_Forward(GRID_SIZE);
    register_time
  END;

  MAIN_BACKWARD = 
  SELECT GET_GUARD_STATUS("MOVE_BACKWARD") = TRUE THEN
    MOVE_BACKWARD;
    Drone_Backward(GRID_SIZE);
    register_time
  END;

  MAIN_LAND = 
    SELECT GET_GUARD_STATUS("LAND") = TRUE THEN 
      LAND;
      Drone_Land;
      register_time
    END;

  MAIN_OBSERVE =
    SELECT GET_GUARD_STATUS("OBSERVE") = TRUE THEN
      VAR LEFT_SENSOR, RIGHT_SENSOR, FORWARD_SENSOR, BACKWARD_SENSOR IN
        LEFT_SENSOR <-- Drone_GetLeftDistance;
        RIGHT_SENSOR <-- Drone_GetRightDistance;
        //SELECT LEFT_SENSOR = GRID_SIZE * (DEPTH - prj1(current_position)) THEN skip END;
        FORWARD_SENSOR <-- Drone_GetFrontDistance;
        BACKWARD_SENSOR <-- Drone_GetBackDistance;
        OBSERVE(BACKWARD_SENSOR, FORWARD_SENSOR, LEFT_SENSOR, RIGHT_SENSOR);
        register_time
      END
    END;

  MAIN_SYNCHRONIZE_BATTERY = 
  SELECT GET_GUARD_STATUS("SYNCHRONIZE_BATTERY") = TRUE THEN
      //BATTERY <-- Drone_GetBattery;
      SYNCHRONIZE_BATTERY(battery);
      register_time
  END;

  MAIN_UPDATE_POSITION =
  SELECT GET_GUARD_STATUS("UPDATE_POSITION") = TRUE THEN
    VAR X_POS, Y_POS IN
      X_POS <-- Drone_GetX;
      Y_POS <-- Drone_GetY;
      UPDATE_POSITION(Y_POS, X_POS);
      register_time
    END
  END

LOCAL_OPERATIONS
    register_time = VAR t IN t := DELTA_WALLTIME; last_time := current_time; current_time := last_time + t END

END
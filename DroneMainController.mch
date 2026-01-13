MACHINE DroneMainController INCLUDES DroneMovement_Real, /*DroneCommunicator_Param_mock("Drone") //,*/ DroneCommunicator_Param("radio://0/80/2M/5700B500FA")
DEFINITIONS
    "LibraryIO.def";
    SET_PREF_OPERATION_REUSE == FALSE;
    MAX_OPERATIONS_INITIALISATION  == 0;
    SET_PREF_MAX_OPERATIONS  == 0;

    //VISB_SVG_FILE == "";
    //VISB_SVG_BOX == rec(width:1000, height:1000);
    VISB_SVG_OBJECTS1 == rec(svg_class:"foreignObject", x:0, y:0, width:800, height:450, `font-size`:7,
         text:```<video id="vid1" style="width:100%; height:100%; box-sizing:border-box; font-size: 7px;" autoplay muted>
                    <source src="./drone_visb_real.mp4#t=${(last_time)/1000},${(current_time)/1000}" type="video/mp4">Unsupported
                 </video>```);
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

  MAIN_TAKEOFF = BEGIN
    TAKEOFF;
    Drone_Takeoff;
    register_time
  END;

  MAIN_LEFT = 
  BEGIN
    MOVE_LEFT;
    Drone_Left(GRID_SIZE);
    register_time
  END;

  MAIN_RIGHT = 
  BEGIN
    MOVE_RIGHT; 
    Drone_Right(GRID_SIZE);
    register_time
  END;
  
  MAIN_FORWARD = 
  BEGIN
    MOVE_FORWARD; 
    Drone_Forward(GRID_SIZE);
    register_time
  END;

  MAIN_BACKWARD = 
  BEGIN
    MOVE_BACKWARD;
    Drone_Backward(GRID_SIZE);
    register_time
  END;

  MAIN_LAND = 
    BEGIN 
      LAND;
      Drone_Land;
      register_time
    END;

  MAIN_OBSERVE =
    VAR LEFT_SENSOR, RIGHT_SENSOR, FORWARD_SENSOR, BACKWARD_SENSOR IN
      LEFT_SENSOR <-- Drone_GetLeftDistance;
      RIGHT_SENSOR <-- Drone_GetRightDistance;
      //SELECT LEFT_SENSOR = GRID_SIZE * (DEPTH - prj1(current_position)) THEN skip END;
      FORWARD_SENSOR <-- Drone_GetFrontDistance;
      BACKWARD_SENSOR <-- Drone_GetBackDistance;
      OBSERVE(BACKWARD_SENSOR, FORWARD_SENSOR, LEFT_SENSOR, RIGHT_SENSOR);
      register_time
    END;

  MAIN_SYNCHRONIZE_BATTERY = 
  BEGIN
      //BATTERY <-- Drone_GetBattery;
      SYNCHRONIZE_BATTERY(battery);
      register_time
  END;

  MAIN_UPDATE_POSITION =
  VAR X_POS, Y_POS IN
      X_POS <-- Drone_GetX;
      Y_POS <-- Drone_GetY;
      UPDATE_POSITION(Y_POS, X_POS);
      register_time
  END

LOCAL_OPERATIONS
    register_time = VAR t IN t := DELTA_WALLTIME; last_time := current_time; current_time := last_time + t END

END
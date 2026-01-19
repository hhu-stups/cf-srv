MACHINE DroneMainController_replay INCLUDES DroneMovement_Real_replay, DroneCommunicator_Param_replay("Drone")
DEFINITIONS
    "LibraryIO.def";
    "LibraryMeta.def";
    SET_PREF_OPERATION_REUSE == FALSE;
    //MAX_OPERATIONS_INITIALISATION  == 0;
    SET_PREF_MAX_OPERATIONS  == 0;

    //VISB_SVG_FILE == "";
    //VISB_SVG_BOX == rec(height:max({(DEPTH+5)*100,900}),width:max({(WIDTH+5)*100,2000}));
    VISB_SVG_OBJECTS1 == rec(svg_class:"foreignObject", x:700, y:100, width:432, height:768, `font-size`:7,
         text:'''<video id="vid1" style="width:100%; height:100%; box-sizing:border-box; font-size: 7px;" autoplay>
                    <source src="./vid/flight_top_3.mp4" type="video/mp4">Unsupported
                 </video>''');
    VISB_SVG_OBJECTS2 == rec(svg_class:"foreignObject", x:700+432+50, y:100, width:768, height:432, `font-size`:7,
         text:'''<video id="vid2" style="width:100%; height:100%; box-sizing:border-box; font-size: 7px;" autoplay>
                    <source src="./vid/flight_front_3.mp4" type="video/mp4">Unsupported
                 </video>''');

    VISB_SVG_OBJECTS101 == rec(svg_class:"script", text:```playVideo(${real(last_time)/1000.0},${real(current_time)/1000.0});```);

    VISB_SVG_OBJECTS100 == rec(svg_class:"script", text:
// https://stackoverflow.com/questions/47643091/html5-video-start-video-at-certain-time-and-play-for-x-amount-of-time
'''function playVideo(from,to) {

    function checkTime() {
        if (vid1.currentTime >= to || vid2.currentTime >= to) {
           vid1.pause();
           vid2.pause();
        } else {
           setTimeout(checkTime, 100);
        }
    }

    vid1.currentTime = from;
    vid2.currentTime = from;
    vid1.play();
    vid2.play();
    checkTime();
}''');
VARIABLES
    last_time, current_time
INVARIANT
    last_time : NATURAL & current_time : NATURAL
INITIALISATION 
  last_time :: NATURAL;
  current_time :: NATURAL;
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
    register_time;
    MOVE_FORWARD; 
    Drone_Forward(GRID_SIZE)
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
    SELECT GET_GUARD_STATUS("OBSERVE") = TRUE THEN
      VAR LEFT_SENSOR, RIGHT_SENSOR, FORWARD_SENSOR, BACKWARD_SENSOR IN
        LEFT_SENSOR <-- Drone_GetLeftDistance;
        RIGHT_SENSOR <-- Drone_GetRightDistance;
        //SELECT LEFT_SENSOR = GRID_SIZE * (WIDTH - prj1(current_position)) + 50 THEN skip END;
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
    register_time = BEGIN last_time :: NATURAL; current_time :: NATURAL END
END
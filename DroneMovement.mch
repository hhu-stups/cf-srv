
MACHINE DroneMovement
SETS /* enumerated */
  MODES={LANDED,FLYING,FINISHED}
CONCRETE_CONSTANTS
  BATTERY_STEP,
  WIDTH,
  DEPTH,
  BASE_POSITION
ABSTRACT_VARIABLES
  mission_failed,
  visited,
  emergency_mode,
  current_position,
  field,
  battery_actual,
  battery,
  position_actual,
  sensor_data_actual,
  mode,
  reward
PROPERTIES
    /*@label M3_Quadcopter_Battery_Ctx:axm_battery_step */ BATTERY_STEP : 1 .. 100
  & /*@label M4_Quadcopter_Movement_Ctx:DEPTH */ DEPTH : NATURAL1
  & /*@label M4_Quadcopter_Movement_Ctx:WIDTH */ WIDTH : NATURAL1
  & /*@label M4_Quadcopter_Movement_Ctx:BASE_POSITION */ BASE_POSITION : (1 .. WIDTH) * (1 .. DEPTH)
  & /*@label M5_Quadcopter_Mission_Status_Ctx_inst:inst_DEPTH */ DEPTH = 5
  & /*@label M5_Quadcopter_Mission_Status_Ctx_inst:inst_WIDTH */ WIDTH = 5
  & /*@label M5_Quadcopter_Mission_Status_Ctx_inst:inst_BATTERY_STEP */ BATTERY_STEP = 2
INVARIANT
    mission_failed : BOOL
  & emergency_mode : BOOL
  & current_position : INTEGER * INTEGER
  & battery_actual : BOOL
  & position_actual : BOOL
  & sensor_data_actual : BOOL
  & mode : MODES
  & /*@label M5_Quadcopter_Mission_Status:inv_mission_status */ mission_failed = FALSE
  & /*@label M4_Quadcopter_Movement:current_pos_not_outside_grid */ (
     emergency_mode = FALSE
     =>
     current_position : (1 .. WIDTH) * (1 .. DEPTH)
    )
  & /*@label M4_Quadcopter_Movement:inv_field */ field : (1 .. WIDTH) * (1 .. DEPTH) --> 0 .. 2
  & /*@label M4_Quadcopter_Movement:no_collision */ (
     emergency_mode = FALSE
     =>
     field(current_position) /= 2
    )
  & /*@label M4_Quadcopter_Movement:inv_visited */ visited <: (1 .. WIDTH) * (1 .. DEPTH)
  & /*@label M4_Quadcopter_Movement:inv_current_pos_in_visited */ (
     current_position : (1 .. WIDTH) * (1 .. DEPTH)
     =>
     current_position : visited
    )
  & /*@label M3_Quadcopter_Battery:battery */ battery : NATURAL
  & reward : REAL
INITIALISATION
    BEGIN
         mode := LANDED
      ||
         sensor_data_actual := FALSE
      ||
         position_actual := FALSE
      ||
         battery := 100
      ||
         battery_actual := FALSE
      ||
         current_position := BASE_POSITION
      ||
         visited := {BASE_POSITION}
      ||
         field := (1 .. WIDTH) * (1 .. DEPTH) * {0} <+ {BASE_POSITION |-> 1}
      ||
         emergency_mode := FALSE
      ||
         mission_failed := FALSE
      ||
         reward := 0.0
    END
OPERATIONS
  TAKEOFF = 
    SELECT 
        /*@label M5_Quadcopter_Mission_Status_inst:grd_mode */ mode = LANDED
      & /*@label M5_Quadcopter_Mission_Status_inst:grd_battery */ battery > BATTERY_STEP
      & /*@label M5_Quadcopter_Mission_Status_inst:grd_battery_actual */ battery_actual = TRUE
      & /*@label M5_Quadcopter_Mission_Status_inst:grd_emergency_mode */ emergency_mode = FALSE
    THEN 
         mode := FLYING
      ||
         battery_actual := FALSE
    END;
  
  MOVE_LEFT = 
    SELECT 
        /*@label M5_Quadcopter_Mission_Status_inst:grd_mode */ mode = FLYING
      & /*@label M5_Quadcopter_Mission_Status_inst:grd_sensor_data_actual */ sensor_data_actual = TRUE
      & /*@label M5_Quadcopter_Mission_Status_inst:grd_position */ position_actual = TRUE
      & /*@label M5_Quadcopter_Mission_Status_inst:grd_battery */ battery > BATTERY_STEP
      & /*@label M5_Quadcopter_Mission_Status_inst:grd_battery_actual */ battery_actual = TRUE
      & /*@label M5_Quadcopter_Mission_Status_inst:grd_emergency_mode */ emergency_mode = FALSE
      & /*@label M5_Quadcopter_Mission_Status_inst:grd_does_not_move_outside */ prj1(INTEGER,INTEGER)(current_position) + 1 <= WIDTH
      & /*@label M5_Quadcopter_Mission_Status_inst:grd_does_not_collide */ field(prj1(INTEGER,INTEGER)(current_position) + 1 |-> prj2(INTEGER,INTEGER)(current_position)) /= 2
      & /*@label M5_Quadcopter_Mission_Status_inst:grd_mission_status */ mission_failed = FALSE
    THEN 
         sensor_data_actual := FALSE
      ||
         position_actual := FALSE
      ||
         battery_actual := FALSE
      ||
         battery := battery - BATTERY_STEP
      ||
         current_position := prj1(INTEGER,INTEGER)(current_position) + 1 |-> prj2(INTEGER,INTEGER)(current_position)
      ||
         visited := visited \/ {prj1(INTEGER,INTEGER)(current_position) + 1 |-> prj2(INTEGER,INTEGER)(current_position)}
      ||
         mission_failed := bool(
             (
              battery - BATTERY_STEP = 0
              or
              (prj1(INTEGER,INTEGER)(current_position) + 1 |-> prj2(INTEGER,INTEGER)(current_position)) /: (1 .. WIDTH) * (1 .. DEPTH)
              or
              field(prj1(INTEGER,INTEGER)(current_position) + 1 |-> prj2(INTEGER,INTEGER)(current_position)) = 2
             )
         )
      ||
         reward :: REAL
    END;
  
  MOVE_RIGHT = 
    SELECT 
        /*@label M5_Quadcopter_Mission_Status_inst:grd_mode */ mode = FLYING
      & /*@label M5_Quadcopter_Mission_Status_inst:grd_sensor_data_actual */ sensor_data_actual = TRUE
      & /*@label M5_Quadcopter_Mission_Status_inst:grd_position */ position_actual = TRUE
      & /*@label M5_Quadcopter_Mission_Status_inst:grd_battery */ battery > BATTERY_STEP
      & /*@label M5_Quadcopter_Mission_Status_inst:grd_battery_actual */ battery_actual = TRUE
      & /*@label M5_Quadcopter_Mission_Status_inst:grd_emergency_mode */ emergency_mode = FALSE
      & /*@label M5_Quadcopter_Mission_Status_inst:grd_does_not_move_outside */ prj1(INTEGER,INTEGER)(current_position) - 1 >= 1
      & /*@label M5_Quadcopter_Mission_Status_inst:grd_does_not_collide */ field(prj1(INTEGER,INTEGER)(current_position) - 1 |-> prj2(INTEGER,INTEGER)(current_position)) /= 2
      & /*@label M5_Quadcopter_Mission_Status_inst:grd_mission_status */ mission_failed = FALSE
    THEN 
         sensor_data_actual := FALSE
      ||
         position_actual := FALSE
      ||
         battery_actual := FALSE
      ||
         battery := battery - BATTERY_STEP
      ||
         current_position := prj1(INTEGER,INTEGER)(current_position) - 1 |-> prj2(INTEGER,INTEGER)(current_position)
      ||
         visited := visited \/ {prj1(INTEGER,INTEGER)(current_position) - 1 |-> prj2(INTEGER,INTEGER)(current_position)}
      ||
         mission_failed := bool(
             (
              battery - BATTERY_STEP = 0
              or
              (prj1(INTEGER,INTEGER)(current_position) - 1 |-> prj2(INTEGER,INTEGER)(current_position)) /: (1 .. WIDTH) * (1 .. DEPTH)
              or
              field(prj1(INTEGER,INTEGER)(current_position) - 1 |-> prj2(INTEGER,INTEGER)(current_position)) = 2
             )
         )
      ||
         reward :: REAL
    END;
  
  MOVE_FORWARD = 
    SELECT 
        /*@label M5_Quadcopter_Mission_Status_inst:grd_mode */ mode = FLYING
      & /*@label M5_Quadcopter_Mission_Status_inst:grd_sensor_data_actual */ sensor_data_actual = TRUE
      & /*@label M5_Quadcopter_Mission_Status_inst:grd_position */ position_actual = TRUE
      & /*@label M5_Quadcopter_Mission_Status_inst:grd_battery */ battery > BATTERY_STEP
      & /*@label M5_Quadcopter_Mission_Status_inst:grd_battery_actual */ battery_actual = TRUE
      & /*@label M5_Quadcopter_Mission_Status_inst:grd_emergency_mode */ emergency_mode = FALSE
      & /*@label M5_Quadcopter_Mission_Status_inst:grd_does_not_move_outside */ prj2(INTEGER,INTEGER)(current_position) + 1 <= DEPTH
      & /*@label M5_Quadcopter_Mission_Status_inst:grd_does_not_collide */ field(prj1(INTEGER,INTEGER)(current_position) |-> prj2(INTEGER,INTEGER)(current_position) + 1) /= 2
      & /*@label M5_Quadcopter_Mission_Status_inst:grd_mission_status */ mission_failed = FALSE
    THEN 
         sensor_data_actual := FALSE
      ||
         position_actual := FALSE
      ||
         battery_actual := FALSE
      ||
         battery := battery - BATTERY_STEP
      ||
         current_position := prj1(INTEGER,INTEGER)(current_position) |-> prj2(INTEGER,INTEGER)(current_position) + 1
      ||
         visited := visited \/ {prj1(INTEGER,INTEGER)(current_position) |-> prj2(INTEGER,INTEGER)(current_position) + 1}
      ||
         mission_failed := bool(
             (
              battery - BATTERY_STEP = 0
              or
              (prj1(INTEGER,INTEGER)(current_position) |-> prj2(INTEGER,INTEGER)(current_position) + 1) /: (1 .. WIDTH) * (1 .. DEPTH)
              or
              field(prj1(INTEGER,INTEGER)(current_position) |-> prj2(INTEGER,INTEGER)(current_position) + 1) = 2
             )
         )
      ||
         reward :: REAL
    END;
  
  MOVE_BACKWARD = 
    SELECT 
        /*@label M5_Quadcopter_Mission_Status_inst:grd_mode */ mode = FLYING
      & /*@label M5_Quadcopter_Mission_Status_inst:grd_sensor_data_actual */ sensor_data_actual = TRUE
      & /*@label M5_Quadcopter_Mission_Status_inst:grd_position */ position_actual = TRUE
      & /*@label M5_Quadcopter_Mission_Status_inst:grd_battery */ battery > BATTERY_STEP
      & /*@label M5_Quadcopter_Mission_Status_inst:grd_battery_actual */ battery_actual = TRUE
      & /*@label M5_Quadcopter_Mission_Status_inst:grd_emergency_mode */ emergency_mode = FALSE
      & /*@label M5_Quadcopter_Mission_Status_inst:grd_does_not_move_outside */ prj2(INTEGER,INTEGER)(current_position) - 1 >= 1
      & /*@label M5_Quadcopter_Mission_Status_inst:grd_does_not_collide */ field(prj1(INTEGER,INTEGER)(current_position) |-> prj2(INTEGER,INTEGER)(current_position) - 1) /= 2
      & /*@label M5_Quadcopter_Mission_Status_inst:grd_mission_status */ mission_failed = FALSE
    THEN 
         sensor_data_actual := FALSE
      ||
         position_actual := FALSE
      ||
         battery_actual := FALSE
      ||
         battery := battery - BATTERY_STEP
      ||
         current_position := prj1(INTEGER,INTEGER)(current_position) |-> prj2(INTEGER,INTEGER)(current_position) - 1
      ||
         visited := visited \/ {prj1(INTEGER,INTEGER)(current_position) |-> prj2(INTEGER,INTEGER)(current_position) - 1}
      ||
         mission_failed := bool(
             (
              battery - BATTERY_STEP = 0
              or
              (prj1(INTEGER,INTEGER)(current_position) |-> prj2(INTEGER,INTEGER)(current_position) - 1) /: (1 .. WIDTH) * (1 .. DEPTH)
              or
              field(prj1(INTEGER,INTEGER)(current_position) |-> prj2(INTEGER,INTEGER)(current_position) - 1) = 2
             )
         )
      ||
         reward :: REAL
    END;
  
  LAND = 
    SELECT 
        /*@label M5_Quadcopter_Mission_Status_inst:grd_mode */ mode = FLYING
    THEN 
      mode := FINISHED ||
      reward :: REAL
    END;
  
  OBSERVE(left_sensor,backward_sensor,right_sensor,forward_sensor) =
    SELECT 
        /*@label M5_Quadcopter_Mission_Status_inst:grd_mode */ mode = FLYING
      & /*@label M5_Quadcopter_Mission_Status_inst:grd_position */ position_actual = TRUE
      & /*@label M5_Quadcopter_Mission_Status_inst:grd_Battery */ battery > BATTERY_STEP
      & /*@label M5_Quadcopter_Mission_Status_inst:grd_battery_actual */ battery_actual = TRUE
      & /*@label M5_Quadcopter_Mission_Status_inst:grd_left */ left_sensor : {1,2}
      & /*@label M5_Quadcopter_Mission_Status_inst:grd_right */ right_sensor : {1,2}
      & /*@label M5_Quadcopter_Mission_Status_inst:grd_forward */ forward_sensor : {1,2}
      & /*@label M5_Quadcopter_Mission_Status_inst:grd_backward */ backward_sensor : {1,2}
      & /*@label M5_Quadcopter_Mission_Status_inst:grd_emergency */ emergency_mode = FALSE
    THEN
         sensor_data_actual := TRUE
      ||
         battery_actual := FALSE
      ||
         field := field <+ ({x,y,val|val : INTEGER & (x = prj1(INTEGER,INTEGER)(current_position) - 1 & (y = prj2(INTEGER,INTEGER)(current_position) & (x : 1 .. WIDTH & (y : 1 .. DEPTH & val = right_sensor))))} \/ ({x,y,val|val : INTEGER & (x = prj1(INTEGER,INTEGER)(current_position) + 1 & (y = prj2(INTEGER,INTEGER)(current_position) & (x : 1 .. WIDTH & (y : 1 .. DEPTH & val = left_sensor))))} \/ ({x,y,val|val : INTEGER & (x = prj1(INTEGER,INTEGER)(current_position) & (y = prj2(INTEGER,INTEGER)(current_position) + 1 & (x : 1 .. WIDTH & (y : 1 .. DEPTH & val = forward_sensor))))} \/ {x,y,val|val : INTEGER & (x = prj1(INTEGER,INTEGER)(current_position) & (y = prj2(INTEGER,INTEGER)(current_position) - 1 & (x : 1 .. WIDTH & (y : 1 .. DEPTH & val = backward_sensor))))})))
    END;
  
  SYNCHRONIZE_BATTERY(pbattery) =
    SELECT 
        /*@label M5_Quadcopter_Mission_Status_inst:grd_battery */ pbattery : NATURAL
      & /*@label M5_Quadcopter_Mission_Status_inst:grd_emergency_mode */ emergency_mode = FALSE
    THEN
         battery := pbattery
      ||
         battery_actual := TRUE
    END;
  
  UPDATE_POSITION(x,y) =
    SELECT 
        x : INTEGER
      & y : INTEGER
      & /*@label M5_Quadcopter_Mission_Status_inst:grd_Battery */ battery > BATTERY_STEP
      & /*@label M5_Quadcopter_Mission_Status_inst:grd_battery_actual */ battery_actual = TRUE
      & /*@label M5_Quadcopter_Mission_Status_inst:grd_emergency_mode */ emergency_mode = FALSE
    THEN
         position_actual := TRUE
      ||
         battery_actual := FALSE
      ||
         current_position := x |-> y
      ||
         visited := IF
             x |-> y : (1 .. WIDTH) * (1 .. DEPTH)
         THEN
           visited \/ {x |-> y}
         ELSE
           visited
         END
      ||
         emergency_mode := bool(
             (
              (x |-> y) /: (1 .. WIDTH) * (1 .. DEPTH)
              or
              (
               x |-> y : dom(field)
               &
               field(x |-> y) = 2
              )
             )
         )
    END
END

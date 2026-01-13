
MACHINE DroneMovement_Real_Mockup
SETS /* enumerated */
  MODES={LANDED,FLYING,FINISHED}
CONCRETE_CONSTANTS
  BATTERY_STEP,
  WIDTH,
  DEPTH,
  BASE_POSITION,
  CONCRETE_DEPTH,
  MAP_TO_ABSTRACT_POS,
  SAFETY_DISTANCE,
  GRID_SIZE,
  CONCRETE_WIDTH
ABSTRACT_VARIABLES
  concrete_position,
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
    CONCRETE_DEPTH : INTEGER
  & MAP_TO_ABSTRACT_POS : POW(INTEGER * INTEGER)
  & CONCRETE_WIDTH : INTEGER
  & /*@label M3_Quadcopter_Battery_Ctx:axm_battery_step */ BATTERY_STEP : 1 .. 100
  & /*@label M4_Quadcopter_Movement_Ctx:DEPTH */ DEPTH : NATURAL1
  & /*@label M4_Quadcopter_Movement_Ctx:WIDTH */ WIDTH : NATURAL1
  & /*@label M4_Quadcopter_Movement_Ctx:BASE_POSITION */ BASE_POSITION : (1 .. WIDTH) * (1 .. DEPTH)
  & /*@label M6_Quadcopter_Concrete_Movement_Ctx:axm_grid_size */ GRID_SIZE : NATURAL1
  & /*@label M6_Quadcopter_Concrete_Movement_Ctx:axm_grid_size_minimum */ GRID_SIZE >= 2
  & /*@label M6_Quadcopter_Concrete_Movement_Ctx:axm_concrete_width */ CONCRETE_WIDTH = WIDTH * GRID_SIZE
  & /*@label M6_Quadcopter_Concrete_Movement_Ctx:axm_concrete_depth */ CONCRETE_DEPTH = DEPTH * GRID_SIZE
  & /*@label M6_Quadcopter_Concrete_Movement_Ctx:axm_safety_distance */ SAFETY_DISTANCE : NATURAL
  & /*@label M6_Quadcopter_Concrete_Movement_Ctx:axm_map_abstract_pos */ MAP_TO_ABSTRACT_POS = /*@symbolic*/ %p.(p : INTEGER & p >= 0|p / GRID_SIZE + 1) \/ /*@symbolic*/ %p.(p : INTEGER & p < 0|p / GRID_SIZE)
  & /*@label M6_Quadcopter_Concrete_Movement_Ctx_inst2:inst_DEPTH */ DEPTH = 5
  & /*@label M6_Quadcopter_Concrete_Movement_Ctx_inst2:inst_WIDTH */ WIDTH = 5
  & /*@label M6_Quadcopter_Concrete_Movement_Ctx_inst2:inst_BASE */ BASE_POSITION = WIDTH / 2 |-> DEPTH / 2
  & /*@label M6_Quadcopter_Concrete_Movement_Ctx_inst2:inst_grid_size */ GRID_SIZE = 500
  & /*@label M6_Quadcopter_Concrete_Movement_Ctx_inst2:inst_safety_distance */ SAFETY_DISTANCE = 0
  & BATTERY_STEP = 2
INVARIANT
    concrete_position : INTEGER * INTEGER
  & mission_failed : BOOL
  & emergency_mode : BOOL
  & current_position : INTEGER * INTEGER
  & battery_actual : BOOL
  & position_actual : BOOL
  & sensor_data_actual : BOOL
  & mode : MODES
  & /*@label M6_Quadcopter_Concrete_Movement:glueing_inv_1 */ prj1(INTEGER,INTEGER)(concrete_position) |-> prj1(INTEGER,INTEGER)(current_position) : MAP_TO_ABSTRACT_POS
  & /*@label M6_Quadcopter_Concrete_Movement:glueing_inv_2 */ prj2(INTEGER,INTEGER)(concrete_position) |-> prj2(INTEGER,INTEGER)(current_position) : MAP_TO_ABSTRACT_POS
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
         concrete_position : (concrete_position : INTEGER * INTEGER & (prj1(INTEGER,INTEGER)(concrete_position) |-> prj1(INTEGER,INTEGER)(BASE_POSITION) : MAP_TO_ABSTRACT_POS & prj2(INTEGER,INTEGER)(concrete_position) |-> prj2(INTEGER,INTEGER)(BASE_POSITION) : MAP_TO_ABSTRACT_POS))
      ||
         reward := 0.0
    END
OPERATIONS
  TAKEOFF = 
    SELECT 
        /*@label M6_Quadcopter_Concrete_Movement_inst2:grd_mode */ mode = LANDED
      & /*@label M6_Quadcopter_Concrete_Movement_inst2:grd_battery */ battery > BATTERY_STEP
      & /*@label M6_Quadcopter_Concrete_Movement_inst2:grd_battery_actual */ battery_actual = TRUE
      & /*@label M6_Quadcopter_Concrete_Movement_inst2:grd_emergency_mode */ emergency_mode = FALSE
    THEN 
         mode := FLYING
      ||
         battery_actual := FALSE
    END;
  
  MOVE_LEFT = 
    SELECT 
        /*@label M6_Quadcopter_Concrete_Movement_inst2:grd_mode */ mode = FLYING
      & /*@label M6_Quadcopter_Concrete_Movement_inst2:grd_sensor_data_actual */ sensor_data_actual = TRUE
      & /*@label M6_Quadcopter_Concrete_Movement_inst2:grd_position */ position_actual = TRUE
      & /*@label M6_Quadcopter_Concrete_Movement_inst2:grd_battery */ battery > BATTERY_STEP
      & /*@label M6_Quadcopter_Concrete_Movement_inst2:grd_battery_actual */ battery_actual = TRUE
      & /*@label M6_Quadcopter_Concrete_Movement_inst2:grd_emergency_mode */ emergency_mode = FALSE
      & /*@label M6_Quadcopter_Concrete_Movement_inst2:grd_does_not_move_outside */ prj1(INTEGER,INTEGER)(current_position) + 1 <= WIDTH
      & /*@label M6_Quadcopter_Concrete_Movement_inst2:grd_does_not_collide */ field(prj1(INTEGER,INTEGER)(current_position) + 1 |-> prj2(INTEGER,INTEGER)(current_position)) /= 2
      & /*@label M6_Quadcopter_Concrete_Movement_inst2:grd_mission_status */ mission_failed = FALSE
      & /*@label M6_Quadcopter_Concrete_Movement_inst2:grd_concrete_pos_does_not_move_outside */ prj1(INTEGER,INTEGER)(concrete_position) <= CONCRETE_WIDTH - GRID_SIZE
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
         concrete_position : (concrete_position : INTEGER * INTEGER & (prj1(INTEGER,INTEGER)(concrete_position) |-> prj1(INTEGER,INTEGER)(current_position) + 1 : MAP_TO_ABSTRACT_POS & prj2(INTEGER,INTEGER)(concrete_position) = prj2(INTEGER,INTEGER)(concrete_position$0)))
      ||
         reward :: REAL
    END;
  
  MOVE_RIGHT = 
    SELECT 
        /*@label M6_Quadcopter_Concrete_Movement_inst2:grd_mode */ mode = FLYING
      & /*@label M6_Quadcopter_Concrete_Movement_inst2:grd_sensor_data_actual */ sensor_data_actual = TRUE
      & /*@label M6_Quadcopter_Concrete_Movement_inst2:grd_position */ position_actual = TRUE
      & /*@label M6_Quadcopter_Concrete_Movement_inst2:grd_battery */ battery > BATTERY_STEP
      & /*@label M6_Quadcopter_Concrete_Movement_inst2:grd_battery_actual */ battery_actual = TRUE
      & /*@label M6_Quadcopter_Concrete_Movement_inst2:grd_emergency_mode */ emergency_mode = FALSE
      & /*@label M6_Quadcopter_Concrete_Movement_inst2:grd_does_not_move_outside */ prj1(INTEGER,INTEGER)(current_position) - 1 >= 1
      & /*@label M6_Quadcopter_Concrete_Movement_inst2:grd_does_not_collide */ field(prj1(INTEGER,INTEGER)(current_position) - 1 |-> prj2(INTEGER,INTEGER)(current_position)) /= 2
      & /*@label M6_Quadcopter_Concrete_Movement_inst2:grd_mission_status */ mission_failed = FALSE
      & /*@label M6_Quadcopter_Concrete_Movement_inst2:grd_concrete_pos_does_not_move_outside */ prj1(INTEGER,INTEGER)(concrete_position) >= GRID_SIZE
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
         concrete_position : (concrete_position : INTEGER * INTEGER & (prj1(INTEGER,INTEGER)(concrete_position) |-> prj1(INTEGER,INTEGER)(current_position) - 1 : MAP_TO_ABSTRACT_POS & prj2(INTEGER,INTEGER)(concrete_position) = prj2(INTEGER,INTEGER)(concrete_position$0)))
      ||
         reward :: REAL
    END;
  
  MOVE_FORWARD = 
    SELECT 
        /*@label M6_Quadcopter_Concrete_Movement_inst2:grd_mode */ mode = FLYING
      & /*@label M6_Quadcopter_Concrete_Movement_inst2:grd_sensor_data_actual */ sensor_data_actual = TRUE
      & /*@label M6_Quadcopter_Concrete_Movement_inst2:grd_position */ position_actual = TRUE
      & /*@label M6_Quadcopter_Concrete_Movement_inst2:grd_battery */ battery > BATTERY_STEP
      & /*@label M6_Quadcopter_Concrete_Movement_inst2:grd_battery_actual */ battery_actual = TRUE
      & /*@label M6_Quadcopter_Concrete_Movement_inst2:grd_emergency_mode */ emergency_mode = FALSE
      & /*@label M6_Quadcopter_Concrete_Movement_inst2:grd_does_not_move_outside */ prj2(INTEGER,INTEGER)(current_position) + 1 <= DEPTH
      & /*@label M6_Quadcopter_Concrete_Movement_inst2:grd_does_not_collide */ field(prj1(INTEGER,INTEGER)(current_position) |-> prj2(INTEGER,INTEGER)(current_position) + 1) /= 2
      & /*@label M6_Quadcopter_Concrete_Movement_inst2:grd_mission_status */ mission_failed = FALSE
      & /*@label M6_Quadcopter_Concrete_Movement_inst2:grd_concrete_pos_does_not_move_outside */ prj2(INTEGER,INTEGER)(concrete_position) <= CONCRETE_DEPTH - GRID_SIZE
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
         concrete_position : (concrete_position : INTEGER * INTEGER & (prj1(INTEGER,INTEGER)(concrete_position) = prj1(INTEGER,INTEGER)(concrete_position$0) & prj2(INTEGER,INTEGER)(concrete_position) |-> prj2(INTEGER,INTEGER)(current_position) + 1 : MAP_TO_ABSTRACT_POS))
      ||
         reward :: REAL
    END;
  
  MOVE_BACKWARD = 
    SELECT 
        /*@label M6_Quadcopter_Concrete_Movement_inst2:grd_mode */ mode = FLYING
      & /*@label M6_Quadcopter_Concrete_Movement_inst2:grd_sensor_data_actual */ sensor_data_actual = TRUE
      & /*@label M6_Quadcopter_Concrete_Movement_inst2:grd_position */ position_actual = TRUE
      & /*@label M6_Quadcopter_Concrete_Movement_inst2:grd_battery */ battery > BATTERY_STEP
      & /*@label M6_Quadcopter_Concrete_Movement_inst2:grd_battery_actual */ battery_actual = TRUE
      & /*@label M6_Quadcopter_Concrete_Movement_inst2:grd_emergency_mode */ emergency_mode = FALSE
      & /*@label M6_Quadcopter_Concrete_Movement_inst2:grd_does_not_move_outside */ prj2(INTEGER,INTEGER)(current_position) - 1 >= 1
      & /*@label M6_Quadcopter_Concrete_Movement_inst2:grd_does_not_collide */ field(prj1(INTEGER,INTEGER)(current_position) |-> prj2(INTEGER,INTEGER)(current_position) - 1) /= 2
      & /*@label M6_Quadcopter_Concrete_Movement_inst2:grd_mission_status */ mission_failed = FALSE
      & /*@label M6_Quadcopter_Concrete_Movement_inst2:grd_concrete_pos_does_not_move_outside */ prj2(INTEGER,INTEGER)(concrete_position) >= GRID_SIZE
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
         concrete_position : (concrete_position : INTEGER * INTEGER & (prj1(INTEGER,INTEGER)(concrete_position) = prj1(INTEGER,INTEGER)(concrete_position$0) & prj2(INTEGER,INTEGER)(concrete_position) |-> prj2(INTEGER,INTEGER)(current_position) - 1 : MAP_TO_ABSTRACT_POS))
      ||
         reward :: REAL
    END;
  
  LAND = 
    SELECT 
        /*@label M6_Quadcopter_Concrete_Movement_inst2:grd_mode */ mode = FLYING
    THEN 
      mode := FINISHED ||
      reward :: REAL
    END;
  
  OBSERVE(concrete_backward_sensor,concrete_forward_sensor,concrete_left_sensor,concrete_right_sensor) =
    SELECT 
        /*@label M6_Quadcopter_Concrete_Movement_inst2:grd_concrete_left_sensor */ concrete_left_sensor : NATURAL
      & /*@label M6_Quadcopter_Concrete_Movement_inst2:grd_concrete_right_sensor */ concrete_right_sensor : NATURAL
      & /*@label M6_Quadcopter_Concrete_Movement_inst2:grd_concrete_forward_sensor */ concrete_forward_sensor : NATURAL
      & /*@label M6_Quadcopter_Concrete_Movement_inst2:grd_concrete_backward_sensor */ concrete_backward_sensor : NATURAL
      & /*@label M6_Quadcopter_Concrete_Movement_inst2:grd_emergency */ emergency_mode = FALSE
      & /*@label M6_Quadcopter_Concrete_Movement_inst2:grd_mode */ mode = FLYING
      & /*@label M6_Quadcopter_Concrete_Movement_inst2:grd_position */ position_actual = TRUE
      & /*@label M6_Quadcopter_Concrete_Movement_inst2:grd_Battery */ battery > BATTERY_STEP
      & /*@label M6_Quadcopter_Concrete_Movement_inst2:grd_battery_actual */ battery_actual = TRUE
    THEN
         sensor_data_actual := TRUE
      ||
         battery_actual := FALSE
      ||
         field := field <+ ({x,y,val|val : INTEGER & (x = prj1(INTEGER,INTEGER)(current_position) - 1 & (y = prj2(INTEGER,INTEGER)(current_position) & (x : 1 .. WIDTH & (y : 1 .. DEPTH & ((concrete_right_sensor <= GRID_SIZE + prj1(INTEGER,INTEGER)(concrete_position) mod GRID_SIZE & val = 2) or (not(concrete_right_sensor <= GRID_SIZE + prj1(INTEGER,INTEGER)(concrete_position) mod GRID_SIZE) & val = 1))))))} \/ ({x,y,val|val : INTEGER & (x = prj1(INTEGER,INTEGER)(current_position) + 1 & (y = prj2(INTEGER,INTEGER)(current_position) & (x : 1 .. WIDTH & (y : 1 .. DEPTH & ((concrete_left_sensor <= GRID_SIZE + (GRID_SIZE - prj1(INTEGER,INTEGER)(concrete_position) mod GRID_SIZE) & val = 2) or (not(concrete_left_sensor <= GRID_SIZE + (GRID_SIZE - prj1(INTEGER,INTEGER)(concrete_position) mod GRID_SIZE)) & val = 1))))))} \/ ({x,y,val|val : INTEGER & (x = prj1(INTEGER,INTEGER)(current_position) & (y = prj2(INTEGER,INTEGER)(current_position) + 1 & (x : 1 .. WIDTH & (y : 1 .. DEPTH & ((concrete_forward_sensor <= GRID_SIZE + (GRID_SIZE - prj2(INTEGER,INTEGER)(concrete_position) mod GRID_SIZE) & val = 2) or (not(concrete_forward_sensor <= GRID_SIZE + (GRID_SIZE - prj2(INTEGER,INTEGER)(concrete_position) mod GRID_SIZE)) & val = 1))))))} \/ {x,y,val|val : INTEGER & (x = prj1(INTEGER,INTEGER)(current_position) & (y = prj2(INTEGER,INTEGER)(current_position) - 1 & (x : 1 .. WIDTH & (y : 1 .. DEPTH & ((concrete_backward_sensor <= GRID_SIZE + prj2(INTEGER,INTEGER)(concrete_position) mod GRID_SIZE & val = 2) or (not(concrete_backward_sensor <= GRID_SIZE + prj2(INTEGER,INTEGER)(concrete_position) mod GRID_SIZE) & val = 1))))))})))
    END;
  
  SYNCHRONIZE_BATTERY(pbattery) =
    SELECT 
        /*@label M6_Quadcopter_Concrete_Movement_inst2:grd_battery */ pbattery : NATURAL
      & /*@label M6_Quadcopter_Concrete_Movement_inst2:grd_emergency_mode */ emergency_mode = FALSE
    THEN
         battery := pbattery
      ||
         battery_actual := TRUE
    END;
  
  UPDATE_POSITION(concrete_x,concrete_y) =
    SELECT 
        concrete_y : INTEGER
      & concrete_x : INTEGER
      & /*@label M6_Quadcopter_Concrete_Movement_inst2:grd_emergency_mode */ emergency_mode = FALSE
      & /*@label M6_Quadcopter_Concrete_Movement_inst2:grd_Battery */ battery > BATTERY_STEP
      & /*@label M6_Quadcopter_Concrete_Movement_inst2:grd_battery_actual */ battery_actual = TRUE
    THEN
         position_actual := TRUE
      ||
         battery_actual := FALSE
      ||
         current_position := MAP_TO_ABSTRACT_POS(concrete_x) |-> MAP_TO_ABSTRACT_POS(concrete_y)
      ||
         visited := IF
             MAP_TO_ABSTRACT_POS(concrete_x) |-> MAP_TO_ABSTRACT_POS(concrete_y) : (1 .. WIDTH) * (1 .. DEPTH)
         THEN
           visited \/ {MAP_TO_ABSTRACT_POS(concrete_x) |-> MAP_TO_ABSTRACT_POS(concrete_y)}
         ELSE
           visited
         END
      ||
         concrete_position := concrete_x |-> concrete_y
      ||
         emergency_mode := bool(
             (
              (MAP_TO_ABSTRACT_POS(concrete_x) |-> MAP_TO_ABSTRACT_POS(concrete_y)) /: (1 .. WIDTH) * (1 .. DEPTH)
              or
              (
               MAP_TO_ABSTRACT_POS(concrete_x) |-> MAP_TO_ABSTRACT_POS(concrete_y) : dom(field)
               &
               field(MAP_TO_ABSTRACT_POS(concrete_x) |-> MAP_TO_ABSTRACT_POS(concrete_y)) = 2
              )
             )
         )
    END
END

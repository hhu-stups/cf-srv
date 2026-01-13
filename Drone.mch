MACHINE Drone

CONSTANTS
    DEPTH, WIDTH, BASE_POSITION

PROPERTIES
    DEPTH = 5 &
    WIDTH = 5 &
    BASE_POSITION : (1.. WIDTH) * (1..DEPTH) &
    BASE_POSITION = (WIDTH /2) |-> (DEPTH/2) // always start in the center

VARIABLES
    field,
    battery,
    current_position,
    mission_failed,
    visited

INVARIANT
    field : (1..WIDTH) * (1..DEPTH) --> 0..2 & // 2 = obstacle, 0 = unexplored, 1 = explored
    battery : 0..100 &
    battery > 0 & // battery level should never be zero
    current_position : (1.. WIDTH) * (1..DEPTH) &
    field(current_position) /= 2 & // drone shall not collide with an obstacle
    // 0: unexplored, 1: covered, 2: obstacle
    mission_failed = FALSE &
    visited <: (1..WIDTH) * (1..DEPTH)

INITIALISATION
    //field := (1..WIDTH) * (1..DEPTH) * {0} <+ {BASE_POSITION |-> 1} ||
    battery := 100 ||
    current_position := BASE_POSITION ||
    visited := {BASE_POSITION} ||
    mission_failed := FALSE;
    field := (1..WIDTH) * (1..DEPTH) * {0} <+ {BASE_POSITION |-> 1}

OPERATIONS
    MOVE_FORWARD =
    SELECT 1=1 THEN
        SELECT mission_failed = FALSE THEN

            current_position := prj1(current_position) |-> prj2(current_position) - 1;
	        battery := battery - 2;
            visited := visited \/ {current_position};
            //field :: (1..WIDTH) * (1..DEPTH) --> (0..2);
            ANY update WHERE
		//        ! pos.(pos : dom(update) => update(pos) = REAL_ENVIRONMENT(pos)) &
                update : {x,y | x : prj1(current_position) - 1 .. prj1(current_position) + 1 &
                    y : prj2(current_position) - 1 .. prj2(current_position) + 1 &
                    x : 1.. WIDTH & y : 1.. DEPTH &
                    (x = prj1(current_position) or y = prj2(current_position)) &
                    field(x |-> y) = 0} --> {1, 2}
            THEN
                field := field <+ update
            END;
            mission_failed := bool(battery = 0 or current_position /: (1.. DEPTH) * (1..WIDTH) or field(current_position) = 2)
        END
    END;

    MOVE_BACKWARD =
    SELECT 1=1 THEN
        SELECT mission_failed = FALSE THEN
            current_position := prj1(current_position) |-> prj2(current_position) + 1;
	        battery := battery - 2;
            visited := visited \/ {current_position};
            //field :: (1..WIDTH) * (1..DEPTH) --> (0..2);
            ANY update WHERE
                update : {x,y | x : prj1(current_position) - 1 .. prj1(current_position) + 1 &
                    y : prj2(current_position) - 1 .. prj2(current_position) + 1 &
                    x : 1.. WIDTH & y : 1.. DEPTH &
                    (x = prj1(current_position) or y = prj2(current_position)) &
                    field(x |-> y) = 0} --> {1, 2}
            THEN
                field := field <+ update
            END;
            mission_failed := bool(battery = 0 or current_position /: (1.. DEPTH) * (1..WIDTH) or field(current_position) = 2)
        END
    END;

    MOVE_LEFT =
    SELECT 1=1 THEN
        SELECT mission_failed = FALSE THEN
            current_position := prj1(current_position) - 1 |-> prj2(current_position);
	        battery := battery - 2;
            visited := visited \/ {current_position};
            //field :: (1..WIDTH) * (1..DEPTH) --> (0..2);
            ANY update WHERE
                update : {x,y | x : prj1(current_position) - 1 .. prj1(current_position) + 1 &
                    y : prj2(current_position) - 1 .. prj2(current_position) + 1 &
                    x : 1.. WIDTH & y : 1.. DEPTH &
                    (x = prj1(current_position) or y = prj2(current_position)) &
                    field(x |-> y) = 0} --> {1, 2}
            THEN
                field := field <+ update
            END;
            mission_failed := bool(battery = 0 or current_position /: (1.. DEPTH) * (1..WIDTH) or field(current_position) = 2)
        END
    END;

    MOVE_RIGHT =
    SELECT 1=1 THEN
        SELECT mission_failed = FALSE THEN
            current_position := prj1(current_position) + 1 |-> prj2(current_position);
	        battery := battery - 2;
            visited := visited \/ {current_position};
            //field :: (1..WIDTH) * (1..DEPTH) --> (0..2);
            ANY update WHERE
		//        ! pos.(pos : dom(update) => update(pos) = REAL_ENVIRONMENT(pos)) &
                update : {x,y | x : prj1(current_position) - 1 .. prj1(current_position) + 1 &
                    y : prj2(current_position) - 1 .. prj2(current_position) + 1 &
                    x : 1.. WIDTH & y : 1.. DEPTH &
                    (x = prj1(current_position) or y = prj2(current_position)) &
                    field(x |-> y) = 0} --> {1, 2}
            THEN
                field := field <+ update
            END;
            mission_failed := bool(battery = 0 or current_position /: (1.. DEPTH) * (1..WIDTH) or field(current_position) = 2)
        END
    END

END
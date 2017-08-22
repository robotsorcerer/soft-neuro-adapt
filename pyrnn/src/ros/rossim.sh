rostopic pub /mannequine_head/pose -r 30 geometry_msgs/Pose '{position: {x: 17.0548143387, y: -635.534881592, z: 240.301490784}, orientation: {x: 3.0, y: -3.14159265359, z: 3.14159265359} }'

rostopic pub /mannequine_head/u_valves -r 30 ensenso/ValveControl '{left_bladder_pos: 1, left_bladder_neg: 1, right_bladder_pos: 1, right_bladder_neg: 1, base_bladder_pos: 1, base_bladder_neg: 1}'
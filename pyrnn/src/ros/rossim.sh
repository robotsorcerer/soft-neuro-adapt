rostopic pub /mannequine_head/pose -r 30 geometry_msgs/Pose '{position: {x: 12, y: 14, z: 16}, orientation: {x: 12, y: 14, z: 16} }'

rostopic pub /mannequine_head/u_valves -r 30 geometry_msgs/Twist '{linear: {x: 12, y: 14, z: 16}, angular: {x: 12, y: 14, z: 16} }'
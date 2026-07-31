# New MPC Isaac
This project uses an NN-based car dynamics model and an MPC method to control the car to follow a path.

## Environment
- Ubuntu
- Docker
- Isaac Sim

## Start Docker & Codes
```
git clone https://github.com/lianghhhhh/new_mpc_isaac.git
cd new_mpc_isaac
./docker/scripts/mpc.sh
r
ros2 run mlp_control_pkg mlp_control_node
```

## Start Isaac Sim
1. Open Isaac Sim
2. Confirm that ros2_bridge can be used (Window → Extensions → ros2_bridge)
3. Select both small_car->Cube & BasisCurves->BasisCurves at Stage
4. Press the Play button

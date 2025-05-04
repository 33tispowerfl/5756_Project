import pybullet as p
import time
import pybullet_data


# Initialize PyBullet
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.setTimeStep(1./60.)

# Load the car URDF
planeId = p.loadURDF("plane.urdf")


mazeID = p.loadURDF("/home/beta_frame/Documents/5756_project/asset/maze/maze_generator_ros/maze/model.urdf")

cubeStartPos = [0.5,-0.5,1]
cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
carId = p.loadURDF("./asset/mentorpi_description/urdf/ack.xacro",cubeStartPos, cubeStartOrientation)
num_joints = p.getNumJoints(carId)

# Identify the driving wheel joint names (you might need to adjust these)
wheel_joint_names = ["wheel_rf_Joint", "wheel_lf_Joint", "wheel_rb_Joint", "wheel_lb_Joint"]
wheel_joint_indices = []

for name in wheel_joint_names:
    joint_index = -1
    for i in range(p.getNumJoints(carId)):
        info = p.getJointInfo(carId, i)
        if info[1].decode('utf-8') == name:
            joint_index = info[0]
            break
    if joint_index != -1:
        wheel_joint_indices.append(joint_index)
        print(f"Found joint index for {name}: {joint_index}")
    else:
        print(f"Warning: Joint {name} not found!")

# Set a target velocity for the driving wheels
target_velocity = 10  # Radians per second (adjust as needed)
max_force = 100      # Newton-meters (adjust as needed)

# Simulation loop
while True:
    # Apply torque (or set velocity) to the driving wheels
    for joint_index in wheel_joint_indices:
        p.setJointMotorControl2(bodyIndex=carId,
                                jointIndex=joint_index,
                                controlMode=p.VELOCITY_CONTROL,
                                targetVelocity=target_velocity,
                                force=max_force)

    p.stepSimulation()
    time.sleep(1./240.) # Optional: Add a small delay for visualization
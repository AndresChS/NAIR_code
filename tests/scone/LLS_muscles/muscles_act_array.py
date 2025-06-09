import matplotlib.pyplot as plt
import numpy as np
import time
from datetime import datetime
from sconetools import sconepy
import sys
import os
from read_csv import merge_csvs
import pandas as pd

# add scone gym using a relative path
sconegym_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../NAIR_envs')) 
sys.path.append(sconegym_path)
today = datetime.now().strftime('%Y-%m-%d')  # Format: YYYY-MM-DD



# Helper function that shows various model properties
def print_model_info(model):
	# iterate over bodies and print some properties
	# other available objects are: model.dofs(), model.actuators(), model.state()
	for bod in model.bodies():
		if bod.name().startswith('t'): # only items starting with 't' to avoid clutter
			print(f"body {bod.name()} mass={bod.mass():.3f} inertia={bod.inertia_diag()}")
	for mus in model.muscles():
		if mus.name().startswith('g'): # only items starting with 'g' to avoid clutter
			print(f"muscle {mus.name()} L={mus.fiber_length_norm():.3f} V={mus.fiber_velocity_norm():.3f} F={mus.force_norm():.3f}")


# ====================================================================
# PID controller definition
# --------------------------------------------------------------------
import numpy as np

class PIDController:
    def __init__(self, kp, ki, kd, setpoint):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, current_position, dt, setpoint):
        # Update setpoint if it changes during the simulation
        self.setpoint = setpoint

        # Compute error
        error = self.setpoint - current_position

        # Update integral term
        self.integral += error * dt

        # Protect against division by zero in derivative term
        if dt == 0:
            derivative = 0.0
        else:
            derivative = (error - self.prev_error) / dt

        # PID output
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)

        # Clip output to range [-100, 100]
        output = np.clip(output, -100, 100)

        # Update previous error
        self.prev_error = error

        return output, error

# ====================================================================
# Scone step simulation definition
# --------------------------------------------------------------------
def scone_step(model, muscles_actions, motor_torque, use_neural_delays=True, step=0):
	muscle_activations = muscles_actions
	model.init_muscle_activations(muscle_activations)
	#dof_positions = model.dof_position_array()
	#model.set_dof_positions(dof_positions)
	#model.init_state_from_dofs()
	# The model inputs are computed here
	# We use an example controller that mimics basic monosynaptic reflexes
	# by simply adding the muscle force, length and velocity of a muscle
	# and feeding it back as input to the same muscle
	if use_neural_delays:
		# Read sensor information WITH neural delays
		mus_in = model.delayed_muscle_force_array()
		mus_in += model.delayed_muscle_fiber_length_array() - 1.2
		mus_in += 0.1 * model.delayed_muscle_fiber_velocity_array()
	else:
		# Read sensor information WITHOUT neural delays
		mus_in = model.muscle_force_array()
		mus_in += model.muscle_fiber_length_array() - 1
		mus_in += 0.2 * model.muscle_fiber_velocity_array()
	
	motor_torque = np.array([motor_torque])
	#print("torque: ", motor_torque, "	mus_in: ", mus_in)
	mus_in = np.concatenate((mus_in,motor_torque))
	# The muscle inputs (excitations) are set here
	if use_neural_delays:
		# Set muscle excitation WITH neural delays
		model.set_delayed_actuator_inputs(mus_in)
	else:
		# Set muscle excitation WITHOUT neural delays
		model.set_actuator_inputs(mus_in)
	
	model.advance_simulation_to(step)

	return model.com_pos(), model.time()

# ====================================================================
# Sconepy model initialitation
# --------------------------------------------------------------------
model = sconepy.load_model(sconegym_path+"/sconegym/nair_envs/H0910_LLS/H0910_LLSV0.scone")
model.reset()
store_data = True
use_neural_delays = True
model.set_store_data(store_data)

dirname = "sconetest_" + "LLS" + "_" + model.name() + "_" + today

# Read csvs and merge
path1 = 'csvs/time.csv'
path2 = 'csvs/knee_angle.csv'
path3 = 'csvs/activations.csv'

df1 = pd.read_csv(path1)
df2 = pd.read_csv(path2)
df3 = pd.read_csv(path3)
merged = pd.concat([df1.reset_index(drop=True), df2.reset_index(drop=True)], axis=1)
merged = pd.concat([merged.reset_index(drop=True), df3.reset_index(drop=True)], axis=1) 
print(merged)
# Configuration  of time steps and simulation time
max_time = 2 # In seconds
timestep = 0.01
timesteps = int(max_time / timestep)
kp = 100
ki = 0
kd = 50
setpoint = 0.2
params_list = [ki, kp, kd]
pid_controller = PIDController(kp, ki, kd, setpoint)
# ====================================================================
# Controller loop and main function
# --------------------------------------------------------------------
# Extract column names (assumes first = time, second = pos)
time_col = merged.columns[0]
pos_col = merged.columns[1]
bifemlh_r_col = merged.columns[2]
bifemsh_r_col = merged.columns[3]
rect_fem_r_col = merged.columns[4]
vas_int_r_col = merged.columns[5]
med_gas_r_col = merged.columns[6]
print(med_gas_r_col)
dt = 0

time_list = []
pos_list = []
setpoint_list = []
PID_error_list = []
torque_list = []
interval = 500
low_setpoint = -1.4
high_setpoint = -0.1
prev_time = merged.iloc[0][time_col] 
print(prev_time)
# Controller loop: iterate directly over the data
for step in range(len(merged)):
	current_time = merged.iloc[step][time_col]
	current_setpoint = merged.iloc[step][pos_col]
	bifemlh_r_r = merged.loc[step, bifemlh_r_col]
	bifemsh_r_r  = merged.loc[step, bifemsh_r_col]
	rect_fem_r = merged.loc[step, rect_fem_r_col]
	vas_int_r  = merged.loc[step, vas_int_r_col]
	med_gas_r  = merged.loc[step, med_gas_r_col]

	dofs = model.dofs()
	current_pos = dofs[1].pos()
	dt = current_time - prev_time
	prev_time = current_time
	steps = step
	rng = np.random.default_rng(1)
	actions = np.zeros(len(model.muscles()))

	actions[0]= merged.loc[step, bifemlh_r_col]
	actions[1]= merged.loc[step, bifemsh_r_col]
	actions[4]= merged.loc[step, rect_fem_r_col]
	actions[5]= merged.loc[step, vas_int_r_col]
	actions[6]= merged.loc[step, med_gas_r_col]
	#print(rect_fem_r)
	#actions = 0 * rng.random((len(model.muscles())))
	torque, PID_error = pid_controller.compute(current_pos, dt, setpoint = current_setpoint)
	# Use current_time and current_pos in your logic
	#print(f"Step {dt}: time={current_time:.4f}, pos={current_pos:.4f}, setpoint={current_setpoint:.4f}, torque={torque:.4f}", )
	model_com_pos, model_time = scone_step(model, motor_torque=0, muscles_actions=actions, use_neural_delays=True, step=current_time)
    
	time_list.append(current_time)
	pos_list.append(current_pos)
	setpoint_list.append(current_setpoint)
	PID_error_list.append(PID_error)
	torque_list.append(torque/1000)
# --------------------------------------------
# Plotear resultados
# --------------------------------------------
mus = model.muscles()   
print([m.name() for m in mus])
plt.figure(figsize=(10, 5))
plt.plot(time_list, pos_list, label='Position', linewidth=2)
plt.plot(time_list, PID_error_list, label='Error', linewidth=2)
plt.plot(time_list, setpoint_list, label='Setpoint', linestyle='--', linewidth=2)
#plt.plot(time_list, torque_list, label='torque', linestyle='--', linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Position (rad)')
plt.title('Position vs Setpoint Over Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
"""   
print(f"Episode completed in {current_time} steps")
 
print(mus[0].name(),"0")
print(mus[1].name(),"1")
print(mus[2].name(),"2")
print(mus[3].name(),"3")
print(mus[4].name(),"4")
print(mus[5].name(),"5")
print(mus[6].name(),"6")
print([m.name() for m in mus])

act = model.actuators()    
print([m.name() for m in act])
joints = model.dofs()
print([m.name() for m in joints])
print(joints[1].name(), joints[1].pos())

"""
if store_data:
	filename = model.name() + f'_{model.time():0.2f}_'+ today + "test_2"
    
	if use_neural_delays: dirname += "_delay"
	model.write_results(dirname, filename)
	print(f"Results written to {dirname}/{filename}", flush=True)



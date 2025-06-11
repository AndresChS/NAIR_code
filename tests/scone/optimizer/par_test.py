import matplotlib.pyplot as plt
import numpy as np
import time
from datetime import datetime
from sconetools import sconepy
import sys
import os
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
# Scone step simulation definition
# --------------------------------------------------------------------
def scone_step(model, muscles_actions, motor_torque, use_neural_delays=True, step=0):
	param = model.get_control_parameter_names()
	#print([model.get_control_parameter(m) for m in param])
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
	#mus_in = np.concatenate((mus_in,motor_torque))
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
model = sconepy.load_model(sconegym_path+"/sconegym/nair_envs/H0918_KneeExo/H0918_KneeExoRLV0.scone", sconegym_path+"/sconegym/nair_envs/H0918_KneeExo/par/0069_64.442_0.596.par")
model.reset()
#par.import_values( par_file );
#model = mo->CreateModelFromParams( par );
sconepy.evaluate_par_file(sconegym_path+"/sconegym/nair_envs/H0918_KneeExo/par/0069_64.442_0.596.par")
param = model.get_control_parameter_names()
print([model.get_control_parameter(m) for m in param])
store_data = True
use_neural_delays = False
model.set_store_data(store_data)

dirname = "sconetest_" + "LLS" + "_" + model.name() + "_" + today


# Configuration  of time steps and simulation time
max_time = 5 # In seconds
timestep = 0.005
timesteps = int(max_time / timestep)
# ====================================================================
# Controller loop and main function
# --------------------------------------------------------------------
com_y_list = []
time_list = []
pos_list = []
# Controller loop: iterate directly over the data
for step in range(timesteps):

	dofs = model.dofs()
	current_pos = dofs[2].pos()
	steps = step
	rng = np.random.default_rng(1)
	actions = np.zeros(len(model.muscles()))
	#actions = 0 * rng.random((len(model.muscles())))
	# Use current_time and current_pos in your logic
	#print(f"Step {dt}: time={current_time:.4f}, pos={current_pos:.4f}, setpoint={current_setpoint:.4f}, torque={torque:.4f}", )
	model_com_pos, model_time = scone_step(model, motor_torque=0, muscles_actions=actions, use_neural_delays=False, step=step*timestep)
    
	pos_list.append(current_pos)
	com_y_list.append(model.com_pos().y)
	#print(dofs[2].name(), dofs[2].pos())
	time_list.append(step*timestep)
# --------------------------------------------
# Plotear resultados
# --------------------------------------------
mus = model.dofs()   
print([m.name() for m in mus])
plt.figure(figsize=(10, 5))
plt.plot(time_list, com_y_list, label='pos_pelvis_y', linewidth=2)
plt.plot(time_list, pos_list, label='com_y', linestyle='--', linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Position pelvis y (m)')
plt.title('Position vs Setpoint Over Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
act = model.actuators()    
print([m.name() for m in act])
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


joints = model.dofs()
print([m.name() for m in joints])
print(joints[1].name(), joints[1].pos())

"""
if store_data:
	filename = model.name() + f'_{model.time():0.2f}_'+ today + "test_exo_gait_par_no_musin"
    
	if use_neural_delays: dirname += "_delay"
	model.write_results(dirname, filename)
	print(f"Results written to {dirname}/{filename}", flush=True)



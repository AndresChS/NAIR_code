import numpy as np
import time
from datetime import datetime
from sconetools import sconepy
import sys
import os
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
def scone_step(model, actions, use_neural_delays=True, step=0):
	muscle_activations = actions
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
	#motor_torque = np.array([0])
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
model = sconepy.load_model(sconegym_path+"/sconegym/nair_envs/H0910_LLS/H0910_LLSV0.scone")
model.reset()
store_data = True
use_neural_delays = True
model.set_store_data(store_data)

dirname = "sconetest_" + "LLS" + "_" + model.name() + "_" + today

# Configuration  of time steps and simulation time
max_time = 2 # In seconds
timestep = 0.01
timesteps = int(max_time / timestep)

# ====================================================================
# Controller loop and main function
# --------------------------------------------------------------------
for step in range(timesteps):

	steps = step
	step= step*timestep
	rng = np.random.default_rng(1)
	actions = 0.4 * rng.random((len(model.muscles())))
	model_com_pos, model_time = scone_step(model, actions, use_neural_delays=True, step=step)
    
    
print(f"Episode completed in {steps} steps")
mus = model.muscles()    
print(mus[0].name(),"0")
print(mus[1].name(),"1")
print(mus[2].name(),"2")
print(mus[3].name(),"3")
print(mus[4].name(),"4")
print(mus[5].name(),"5")
print(mus[6].name(),"6")
#for m in mus[0:1]:
#    print(m.name())

if store_data:
	filename = model.name() + f'_{model.time():0.2f}_'+ today + "test_3"
    
	if use_neural_delays: dirname += "_delay"
	model.write_results(dirname, filename)
	print(f"Results written to {dirname}/{filename}", flush=True)



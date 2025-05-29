# NAIR_Code

Build library: #Duplicate folder
  
  cd /Users/achs/Documents/PHD/code/NAIR_Code/GymRL/nair_envs
  
  python3 setup.py bdist_wheel

Instal - Reinstall library
 
 pip3 install --force-reinstall /Users/achs/Documents/PHD/code/NAIR_Code/GymRL/nair_envs/
 
 dist/nair_envs-0.0-py3-none-any.whl

 # Register NAIRGroup Suite
 import myosuite.envs.nair # noqa
 
 myosuite_nair_suite = set(gym_registry_specs().keys())
 
 myosuite_env_suite-_current_gym_envs
 
 myosuite_env_suite  = myosuite_env_suite | myosuite_nair_suite
 
 myosuite_nair_suite = sorted(myosuite_nair_suite)

tensorboard --logdir .code/RL/SB3/tensorlogs

# Github sentences
git add .
git commit -m "Comentario del commit"
git push or git pull



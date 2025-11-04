import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gymnasium as gym
import time
from stable_baselines3 import PPO
from stable_baselines3 import SAC

# Importar el entorno personalizado
from env_quadruped import QuadrupedEnv

# Crear el entorno con renderizado (GUI) activado
env = QuadrupedEnv(render=True)

# Ruta al modelo guardado (sin .zip)
MODEL_PATH = "./Robotica/ppo_model/rl_model_150000_steps" 
#MODEL_PATH = "./Robotica/model_results/rl_model_350000_steps" 

print(f"Cargando modelo desde {MODEL_PATH}.zip...")
#model = PPO.load(MODEL_PATH, env=env)
model = SAC.load(MODEL_PATH, env=env, device="cpu")
print("¡Modelo cargado!")

vec_env = model.get_env()
obs = vec_env.reset()

sim_speed = 1./60. 

for _ in range(5000):
    action, _states = model.predict(obs, deterministic=True)

    obs, reward, done, info = vec_env.step(action)
    
    time.sleep(sim_speed) 
    
    if done:
        print("Episodio terminado, reiniciando.")
        obs = vec_env.reset()

print("Prueba finalizada.")
env.close()
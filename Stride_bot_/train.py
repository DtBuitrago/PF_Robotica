import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gymnasium as gym
import os
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback

# Importar el entorno personalizado
from env_quadruped import QuadrupedEnv

# Directorios para logs y modelos
log_dir = "./ppo_log/"
model_dir = "./ppo_model/"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

print("Iniciando creación del entorno...")
env = QuadrupedEnv(render=False) # render=False para entrenamiento rápido

# Verificar que el entorno cumple con la API de Gym
print("Verificando el entorno...")
check_env(env)
print("Verificación superada.")

# Guardar un checkpoint del modelo cada 50,000 pasos
checkpoint_callback = CheckpointCallback(
  save_freq=50000,
  save_path=model_dir,
  name_prefix="rl_model"
)

"""
policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=log_dir,
    policy_kwargs=policy_kwargs,
    batch_size=2048,
    n_steps=4096,
    gamma=0.99,
    gae_lambda=0.95,
    n_epochs=10,
    ent_coef=0.0,
    learning_rate=3e-4,
    clip_range=0.2,
    device="cuda"
)
"""
# Define la arquitectura de la red neuronal (2 capas de 256 neuronas)
policy_kwargs = dict(net_arch=dict(pi=[256, 256], qf=[256, 256]))

# Configuración del modelo SAC (Soft Actor-Critic)
model = SAC(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=log_dir,
    policy_kwargs=policy_kwargs,
    
    # Hiperparámetros de SAC
    buffer_size=1_000_000,  # Tamaño del buffer de memoria
    batch_size=256,         # Tamaño del lote para cada actualización
    ent_coef="auto",        # Control automático de la exploración
    gamma=0.99,
    learning_rate=3e-4,
    learning_starts=10000,  # Pasos antes de empezar a aprender
    tau=0.005,              # Suavizado de la actualización de la red
    
    device="cuda"
)

# Iniciar el entrenamiento
TIMESTEPS = 2_000_000
print(f"Iniciando entrenamiento por {TIMESTEPS} pasos...")

model.learn(
    total_timesteps=TIMESTEPS,
    callback=checkpoint_callback,
    progress_bar=True,
    tb_log_name="PPO_SAC"
)

# Guardar el modelo final
model_path = os.path.join(model_dir, "ppo_quadruped_IK_final")
print(f"Entrenamiento completado. Guardando modelo final en {model_path}")
model.save(model_path)

env.close()
print("¡Entrenamiento finalizado!")
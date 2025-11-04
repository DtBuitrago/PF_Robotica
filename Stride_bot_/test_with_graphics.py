import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gymnasium as gym
import time
from stable_baselines3 import PPO
from stable_baselines3 import SAC
import pybullet as p
import matplotlib.pyplot as plt
import numpy as np

# Importar el entorno personalizado
from env_quadruped import QuadrupedEnv

#  Cargar el entorno y el modelo

# Crear el entorno con renderizado (GUI) activado
env = QuadrupedEnv(render=True)

# AÑADIR CONTROLES DE CÁMARA
pybullet_client_id = env.client 
print("Añadiendo sliders de cámara...")
cam_dist_slider = p.addUserDebugParameter("Distancia", 0.1, 2.0, 0.8, physicsClientId=pybullet_client_id)
cam_yaw_slider = p.addUserDebugParameter("Yaw (Rotación)", -180, 180, 45, physicsClientId=pybullet_client_id)
cam_pitch_slider = p.addUserDebugParameter("Pitch (Ángulo)", -90, 90, -30, physicsClientId=pybullet_client_id)

# MODEL_PATH = "./Robotica/ppo_model/rl_model_100000_steps" 
MODEL_PATH = "./Robotica/model_results/rl_model_350000_steps" 

print(f"Cargando modelo desde {MODEL_PATH}.zip...")
#model = PPO.load(MODEL_PATH, env=env)
model = SAC.load(MODEL_PATH, env=env, device="cpu")
print("¡Modelo cargado!")

# --- Inicialización de variables para gráficos ---
print("Iniciando simulación y recolección de datos para las 4 patas...")
time_data = []
com_z_data = []

# Listas para las 4 patas (FR, FL, BR, BL)
legs = ['fr', 'fl', 'br', 'bl']

# Diccionarios para almacenar los datos de cada pata
coxa_data = {leg: [] for leg in legs}
femur_data = {leg: [] for leg in legs}
tibia_data = {leg: [] for leg in legs}
foot_x_data = {leg: [] for leg in legs}
foot_z_data = {leg: [] for leg in legs}

link_indices = {'fr': 3, 'fl': 7, 'br': 11, 'bl': 15}

# Índices de los ángulos en el vector de observación (obs)
angle_indices = {
    'fr': (10, 11, 12), # coxa, femur, tibia
    'fl': (13, 14, 15),
    'br': (16, 17, 18),
    'bl': (19, 20, 21)
}

vec_env = model.get_env()
obs = vec_env.reset()

robot_id = env.robot_id
client_id = env.client

sim_speed = 1./60.
num_steps_para_graficar = 1000

for i in range(num_steps_para_graficar):
    # Control de Cámara
    dist = p.readUserDebugParameter(cam_dist_slider, physicsClientId=client_id)
    yaw = p.readUserDebugParameter(cam_yaw_slider, physicsClientId=client_id)
    pitch = p.readUserDebugParameter(cam_pitch_slider, physicsClientId=client_id)
    base_pos_cam, _ = p.getBasePositionAndOrientation(robot_id, physicsClientId=client_id)
    
    p.resetDebugVisualizerCamera(
        cameraDistance=dist,
        cameraYaw=yaw,
        cameraPitch=pitch,
        cameraTargetPosition=base_pos_cam,
        physicsClientId=client_id
    )

    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    
    # RECOPILACIÓN DE DATOS PARA GRÁFICOS
    
    current_time = i * env.dT
    time_data.append(current_time)
    
    base_pos, base_orn = p.getBasePositionAndOrientation(robot_id, physicsClientId=client_id)
    com_z_data.append(base_pos[2])

    base_world_to_local_pos, base_world_to_local_orn = p.invertTransform(base_pos, base_orn)

    for leg in legs:
        idx_c, idx_f, idx_t = angle_indices[leg]
        coxa_data[leg].append(obs[0][idx_c])
        femur_data[leg].append(obs[0][idx_f])
        tibia_data[leg].append(obs[0][idx_t])

        link_idx = link_indices[leg]
        link_state = p.getLinkState(robot_id, link_idx, physicsClientId=client_id)
        foot_world_pos = link_state[0]

        foot_local_pos, _ = p.multiplyTransforms(base_world_to_local_pos, base_world_to_local_orn,
                                               foot_world_pos, [0,0,0,1])
        
        foot_x_data[leg].append(foot_local_pos[0])
        foot_z_data[leg].append(foot_local_pos[2])
    
    time.sleep(sim_speed) 
    
    if done:
        print("Episodio terminado, reiniciando.")
        obs = vec_env.reset()

print("Prueba finalizada. Generando gráficos...")
env.close()

# --- GENERACIÓN DE GRÁFICOS ---

try:
    # === Gráfico 1: Ángulos Articulares (Cuadrícula 2x2) ===
    fig1, axs1 = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
    fig1.suptitle("Ángulos Articulares vs. Tiempo", fontsize=16)

    # FR
    axs1[0, 0].set_title("Pata Delantera Derecha (FR)")
    axs1[0, 0].plot(time_data, coxa_data['fr'], label="Coxa (q1)")
    axs1[0, 0].plot(time_data, femur_data['fr'], label="Femur (q2)")
    axs1[0, 0].plot(time_data, tibia_data['fr'], label="Tibia (q3)")
    axs1[0, 0].set_ylabel("Ángulo (radianes)")
    axs1[0, 0].grid(True)
    axs1[0, 0].legend()

    # FL
    axs1[0, 1].set_title("Pata Delantera Izquierda (FL)")
    axs1[0, 1].plot(time_data, coxa_data['fl'], label="Coxa (q1)")
    axs1[0, 1].plot(time_data, femur_data['fl'], label="Femur (q2)")
    axs1[0, 1].plot(time_data, tibia_data['fl'], label="Tibia (q3)")
    axs1[0, 1].grid(True)
    axs1[0, 1].legend()

    # BR
    axs1[1, 0].set_title("Pata Trasera Derecha (BR)")
    axs1[1, 0].plot(time_data, coxa_data['br'], label="Coxa (q1)")
    axs1[1, 0].plot(time_data, femur_data['br'], label="Femur (q2)")
    axs1[1, 0].plot(time_data, tibia_data['br'], label="Tibia (q3)")
    axs1[1, 0].set_xlabel("Tiempo (s)")
    axs1[1, 0].set_ylabel("Ángulo (radianes)")
    axs1[1, 0].grid(True)
    axs1[1, 0].legend()

    # BL
    axs1[1, 1].set_title("Pata Trasera Izquierda (BL)")
    axs1[1, 1].plot(time_data, coxa_data['bl'], label="Coxa (q1)")
    axs1[1, 1].plot(time_data, femur_data['bl'], label="Femur (q2)")
    axs1[1, 1].plot(time_data, tibia_data['bl'], label="Tibia (q3)")
    axs1[1, 1].set_xlabel("Tiempo (s)")
    axs1[1, 1].grid(True)
    axs1[1, 1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("grafico_angulos_articulares_4patas.png")


    # === Gráfico 2: Altura del Centro de Masa (CoM) ===
    plt.figure(figsize=(10, 4))
    plt.title("Altura del CoM (Base del Robot) vs. Tiempo")
    plt.plot(time_data, com_z_data, label="Altura Z")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Altura (m)")
    plt.legend()
    plt.grid(True)
    plt.savefig("grafico_altura_com.png")


    # === Gráfico 3: Trayectoria de la Pata (Cuadrícula 2x2) ===
    fig3, axs3 = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
    fig3.suptitle("Trayectoria de Patas (Relativa al Cuerpo, Plano X-Z)", fontsize=16)

    # FR
    axs3[0, 0].set_title("Pata FR")
    axs3[0, 0].plot(foot_x_data['fr'], foot_z_data['fr'])
    axs3[0, 0].set_ylabel("Eje Z (Arriba/Abajo)")
    axs3[0, 0].grid(True)
    axs3[0, 0].set_aspect('equal', adjustable='box')

    # FL
    axs3[0, 1].set_title("Pata FL")
    axs3[0, 1].plot(foot_x_data['fl'], foot_z_data['fl'])
    axs3[0, 1].grid(True)
    axs3[0, 1].set_aspect('equal', adjustable='box')

    # BR
    axs3[1, 0].set_title("Pata BR")
    axs3[1, 0].plot(foot_x_data['br'], foot_z_data['br'])
    axs3[1, 0].set_xlabel("Eje X (Adelante/Atrás)")
    axs3[1, 0].set_ylabel("Eje Z (Arriba/Abajo)")
    axs3[1, 0].grid(True)
    axs3[1, 0].set_aspect('equal', adjustable='box')

    # BL
    axs3[1, 1].set_title("Pata BL")
    axs3[1, 1].plot(foot_x_data['bl'], foot_z_data['bl'])
    axs3[1, 1].set_xlabel("Eje X (Adelante/Atrás)")
    axs3[1, 1].grid(True)
    axs3[1, 1].set_aspect('equal', adjustable='box')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("grafico_trayectoria_patas_4patas.png")

    plt.show()

except Exception as e:
    print(f"Error al generar gráficos: {e}")
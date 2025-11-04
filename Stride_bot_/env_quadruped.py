import gymnasium as gym
import pybullet as p
import pybullet_data
import numpy as np
import time
import os

from src.kinematic_model import robotKinematics
from src.IK_solver import has_ik_failed, reset_ik_failure_flag

class QuadrupedEnv(gym.Env):
    """
    Entorno de Gymnasium para el robot cuadrúpedo StrideBot.
    La red neuronal controla las posiciones objetivo de las patas (IK).
    """
    
    def __init__(self, render=False):
        """Inicializa el entorno, PyBullet, y define los espacios de acción/observación."""
        super(QuadrupedEnv, self).__init__()

        self.render = render
        if self.render:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)

        self.dT = 0.005
        self.max_episode_steps = 1000
        
        
        self.action_scale = np.array([0.05, 0.03, 0.03] * 4) # Escala de la acción [x,y,z] * 4 patas
        
        # Rango de la acción [x, y, z] * 4 patas (en metros)
        #self.action_scale = np.array([0.10, 0.03, 0.07] * 4)
        
        self.kinematics = robotKinematics()

        # 12 deltas de posición de pata [-1, 1]
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)

        # 34 valores (orientación, velocidad, articulaciones)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(34,), dtype=np.float32)

        # Posición neutral de reposo de las patas
        Xdist, Ydist, height = 0.18, 0.15, 0.10
        self.bodytoFeet0 = np.array([
            [ Xdist/2., -Ydist/2., height],  # FR
            [ Xdist/2.,  Ydist/2., height],  # FL
            [-Xdist/2., -Ydist/2., height],  # BR
            [-Xdist/2.,  Ydist/2., height]   # BL
        ]).astype(np.float32)

        self.setup_simulation()

        self.last_good_obs = np.zeros(self.observation_space.shape, dtype=np.float32)

    def setup_simulation(self):
        """Configura la simulación de PyBullet (gravedad, motor, carga de URDFs)."""
        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        
        p.setPhysicsEngineParameter(
            fixedTimeStep=self.dT,
            numSolverIterations=100,
            numSubSteps=1,
            physicsClientId=self.client
        )
        
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client)
        
        urdf_path = os.path.join(os.path.dirname(__file__), "stridebot.urdf")
        
        self.robot_id = p.loadURDF(
            urdf_path,
            basePosition=[0, 0, 0.18],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=False,
            physicsClientId=self.client
        )

        self.joint_name_to_index = {}
        for i in range(p.getNumJoints(self.robot_id, physicsClientId=self.client)):
            info = p.getJointInfo(self.robot_id, i, physicsClientId=self.client)
            self.joint_name_to_index[info[1].decode('UTF-8')] = i
            
            # Desactivar motores por defecto para control de posición
            p.setJointMotorControl2(self.robot_id, i, p.VELOCITY_CONTROL, force=0, physicsClientId=self.client)

        # Orden de las 12 articulaciones (FR, FL, BR, BL)
        self.joint_names = [
            'coxaF_FR', 'femurF_FR', 'tibiaF_FR',
            'coxaF_FL', 'femurF_FL', 'tibiaF_FL',
            'coxaF_BR', 'femurF_BR', 'tibiaF_BR',
            'coxaF_BL', 'femurF_BL', 'tibiaF_BL'
        ]
        self.joint_indices = [self.joint_name_to_index[name] for name in self.joint_names]


    def _get_observation(self):
        """Recopila el estado actual del robot (sensores) y lo devuelve como un vector."""
        pos, orn_q = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client)
        lin_vel, ang_vel = p.getBaseVelocity(self.robot_id, physicsClientId=self.client)
        
        joint_states = p.getJointStates(self.robot_id, self.joint_indices, physicsClientId=self.client)
        joint_pos = [state[0] for state in joint_states]
        joint_vel = [state[1] for state in joint_states]
        self.joint_torques = [state[3] for state in joint_states]
        
        observation = np.concatenate([
            orn_q,
            lin_vel,
            ang_vel,
            joint_pos,
            joint_vel
        ]).astype(np.float32)
        
        return observation

    def _check_done(self):
        """Comprueba si el episodio debe terminar (por caída o voltereta)."""
        pos, orn_q = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client)
        
        # Terminar si el robot se cae
        if pos[2] < 0.05:
            return True
            
        # Terminar si el robot se voltea
        orn_euler = p.getEulerFromQuaternion(orn_q)
        if abs(orn_euler[0]) > 1.0 or abs(orn_euler[1]) > 1.0:
            return True
            
        return False
    
    """
    def _calculate_reward(self):
        pos, _ = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client)
        lin_vel, ang_vel = p.getBaseVelocity(self.robot_id, physicsClientId=self.client)
        
        # 1. Recompensa por avanzar (velocidad en X)
        R_forward = 2.0 * lin_vel[0] # Queremos que vaya en dirección +X
        
        # 2. Penalización por desviarse (velocidad en Y)
        R_lateral = -1.0 * abs(lin_vel[1])
 
        # 3. Penalización por girar (velocidad angular Z)
        R_rotation = -0.5 * abs(ang_vel[2])
        
        # 4. Recompensa por "seguir vivo"
        R_alive = 0.1
        
        total_reward = R_forward + R_lateral + R_rotation + R_alive
        
        # Guardamos la posición X para el próximo cálculo        
        self.last_x_pos = pos[0]
        return total_reward
    """

    def _calculate_reward(self):
        """Calcula la recompensa (el "puntaje") para el último paso dado."""
        pos, _ = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client)
        lin_vel, ang_vel = p.getBaseVelocity(self.robot_id, physicsClientId=self.client)

        # Cortafuegos NaN: Si la física colapsa, penalizar y terminar
        if (np.isnan(pos).any() or np.isnan(lin_vel).any() or np.isnan(ang_vel).any()):
            print("Error: NaN en física (pos/vel). Penalizando.")
            return -1000.0

        R_forward = 2.0 * lin_vel[0]      # Recompensa por avanzar
        R_lateral = -1.0 * abs(lin_vel[1])  # Penalización por desviarse
        R_alive = 0.1                     # Recompensa por seguir vivo

        # Penalización por esfuerzo (torque)
        if np.isnan(self.joint_torques).any():
            torque_penalty = 10000.0
            print("Alerta: NaN en torques. Penalizando.")
        else:
            torque_penalty = sum(abs(t) for t in self.joint_torques)
        R_effort = -0.001 * torque_penalty 

        # Penalización por velocidad vertical (saltos)
        R_vertical = -1.0 * abs(lin_vel[2])

        # Penalización por rotación (roll, pitch, yaw)
        R_rotation = -0.5 * abs(ang_vel[0]) -0.5 * abs(ang_vel[1]) -0.5 * abs(ang_vel[2])

        # Recompensa por levantar patas
        z_actions = self.last_scaled_action[[2, 5, 8, 11]]
        positive_z_actions = z_actions[z_actions > 0]
        R_foot_lift = 0.0
        if len(positive_z_actions) > 0:
            R_foot_lift = 0.2 * np.mean(positive_z_actions)

        # Penalización por fallo de IK ("Out of Domain")
        R_ik_fail = 0.0
        if self.ik_failed_this_step:
            R_ik_fail = -100.0

        total_reward = R_forward + R_lateral + R_alive + R_effort + R_vertical + R_rotation + R_foot_lift + R_ik_fail

        # Cortafuegos NaN final
        if np.isnan(total_reward):
            print("Error: NaN en recompensa total. Penalizando.")
            total_reward = -1000.0
            
        return total_reward

    def step(self, action):
        """Avanza la simulación un paso, aplicando la acción de la NN."""
        scaled_action = action * self.action_scale
        self.last_scaled_action = scaled_action
        
        target_foot_pos = self.bodytoFeet0 + scaled_action.reshape(4, 3)
        
        reset_ik_failure_flag()

        fr_angles, fl_angles, br_angles, bl_angles, _ = self.kinematics.solve(
            np.array([0, 0, 0]),     # orn
            np.array([0, 0, 0]),     # pos
            np.asmatrix(target_foot_pos) # bodytoFeet
        )
        
        self.ik_failed_this_step = has_ik_failed()

        target_angles = np.concatenate([
            fr_angles, fl_angles, br_angles, bl_angles
        ]).astype(np.float32)
        
        p.setJointMotorControlArray(
            bodyUniqueId=self.robot_id,
            jointIndices=self.joint_indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=target_angles,
            forces=[200] * 12,
            physicsClientId=self.client
        )
        
        p.stepSimulation(physicsClientId=self.client)
        self.current_step += 1
        
        obs = self._get_observation()
        reward = self._calculate_reward()
        done = self._check_done()
        
        truncated = self.current_step >= self.max_episode_steps
        
        # Cortafuegos NaN: Si la observación o recompensa es NaN, terminar
        nan_detected = False
        if np.isnan(obs).any():
            print("Error: NaN en OBSERVACIÓN. Terminando episodio.")
            nan_detected = True
        elif np.isnan(reward):
             print("Error: NaN en RECOMPENSA. Terminando episodio.")
             nan_detected = True
        
        if nan_detected:
            done = True
            reward = -1000.0
            obs = self.last_good_obs
        else:
            self.last_good_obs = obs
        
        if done:
            truncated = False

        return obs, reward, done, truncated, {}

    def reset(self, seed=None, options=None):
        """Reinicia el entorno para un nuevo episodio."""
        if seed is not None:
            np.random.seed(seed)
            
        self.setup_simulation()
        self.current_step = 0
        self.last_x_pos = 0.0
        self.joint_torques = [0.0] * 12
        self.last_scaled_action = np.zeros(12)

        reset_ik_failure_flag()
        self.ik_failed_this_step = False
        
        # Aplicar fuerza aleatoria para variar inicio
        force = [np.random.uniform(-5, 5) for _ in range(3)]
        p.applyExternalForce(self.robot_id, -1, force, [0,0,0], p.WORLD_FRAME, physicsClientId=self.client)
        
        obs = self._get_observation()
        self.last_good_obs = obs
        return obs, {}

    def close(self):
        """Cierra la conexión con PyBullet."""
        p.disconnect(physicsClientId=self.client)
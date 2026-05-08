import argparse
import json
import os
from datetime import datetime, timezone
from evosax.algorithms import CMA_ES

from structs import *
from functions import *

import jax.numpy as jnp
import jax.random as random
import jax
import numpy as np
from flax import struct
from sensor import *


#simulation update params
DEFAULT_SEED = 42
KEY = jax.random.PRNGKey(DEFAULT_SEED)
NOISE_SCALE = 0.05
Dt = 0.1 # discrete time increments
EP_LEN = 6000

# world params
WORLD_SIZE_X = 500.0 # max travel distance in x direction
WORLD_SIZE_Y = 500.0 # max travel distance in y direction
MAX_SPAWN_X = 250.0
MAX_SPAWN_Y = 250.0

GRASS_RADIUS = 200.0 # decrease for evolved agents, increase for random agents
MAX_ENERGY = 100.0

# death
DEATH_ENERGY_THRESHOLD = 0.02 * MAX_ENERGY
DEATH_TIME_THRESHOLD = 0.01 * EP_LEN

# reproduction
REPRODUCTION_ENERGY_THRESHOLD = 0.2 * MAX_ENERGY
REPRODUCTION_TIME_THRESHOLD = 0.01 * EP_LEN
OFFSPRING_SPREAD= 30.0 # when an agent reproduces, the offspring will spawn within this radius of the parent


# sheep params
MAX_SHEEP = 400
INIT_SHEEP = 150
SHEEP_RADIUS = 5.0
SHEEP_ENERGY_BEGIN_MAX = 50.0
SHEEP_AGENT_TYPE = 1

# wolf params
MAX_WOLF = 400
INIT_WOLF = 80
WOLF_RADIUS = 5.0
WOLF_ENERGY_BEGIN_MAX = 50.0
WOLF_AGENT_TYPE = 2

# metabolism
METABOLIC_COST_SPEED = 0.02 # not being used
METABOLIC_COST_ANGULAR = 0.02 # not being used
BASIC_METABOLIC_COST_SHEEP = 0.04
BASIC_METABOLIC_COST_WOLF = 0.04

# energy params
BASE_ENERGY_RATE = 0.5  # base energy gain for sheep per timestep in grass radius
EAT_RATE_SHEEP = 0.6 # rate at which wolves consume sheep

# action params (sheep)
SHEEP_SPEED_MULTIPLIER = 2.0
SHEEP_LINEAR_ACTION_SCALE = SHEEP_SPEED_MULTIPLIER * SHEEP_RADIUS / Dt
SHEEP_ANGULAR_SPEED_SCALE = 2.0

# action params (wolves)
WOLF_SPEED_MULTIPLIER = 2.0
WOLF_LINEAR_ACTION_SCALE = WOLF_SPEED_MULTIPLIER * WOLF_RADIUS / Dt
WOLF_ANGULAR_SPEED_SCALE = 2.0

# sensor params
SHEEP_RAY_MAX_LENGTH = 300.0
WOLF_RAY_MAX_LENGTH = 300.0
RAY_RESOLUTION = 13  # W&B update
RAY_SPAN = jnp.pi # W&B update

# controller params
NUM_OBS = RAY_RESOLUTION*2 + 8
NUM_NEURONS = 60
NUM_ACTIONS = 2
ACTION_SCALE = 1.0
LINEAR_ACTION_OFFSET = 0.0
TIME_CONSTANT_SCALE = 10.0 # speed of the neuron dynamics
NUM_ES_PARAMS = NUM_NEURONS * (NUM_NEURONS + NUM_OBS + NUM_ACTIONS + 2) # total number of parameters the Evolutionary Strategy needs to optimize

# training params
NUM_WORLDS = 16 # needs to match POP_SIZE
POP_SIZE = 16  # the number of worlds we evaluate in parallel
PHASE3_GENERATIONS = 150

INACTIVE_POS = -5000.0  # moves inactive agents off the map
INACTIVE_ENERGY = -1.0  # placeholder for inactive agents
INIT_LOW = -2.0         # lower bound for initial ES weights
INIT_HIGH = 2.0         # upper bound for initial ES weights

# ----------------------------------------------------------------------------------------------------------------

# save data
DATA_PATH = "./data/evolved/training2/"
# load agent data
SHEEP_BRAIN_PATH = "./data_s/sheep_vs_rand/sheep_vs_rand4/"
WOLF_BRAIN_PATH = "./data_w/wolf_vs_rand/wolf_vs_rand13/"

# ----------------------------------------------------------------------------------------------------------------

# Predator-prey world parameters
PP_WORLD_PARAMS = Params(content= {"sheep_params": {"x_max": MAX_SPAWN_X,
                                                    "y_max": MAX_SPAWN_Y,
                                                    "energy_begin_max": SHEEP_ENERGY_BEGIN_MAX,
                                                    "eat_rate": EAT_RATE_SHEEP,
                                                    "radius": SHEEP_RADIUS,
                                                    "agent_type": SHEEP_AGENT_TYPE,
                                                    "num_agents": MAX_SHEEP,
                                                    "num_active_agents": INIT_SHEEP
                                                    },
                                   "wolf_params": {"x_max": MAX_SPAWN_X,
                                                   "y_max": MAX_SPAWN_Y,
                                                   "energy_begin_max": WOLF_ENERGY_BEGIN_MAX,
                                                   "radius": WOLF_RADIUS,
                                                   "agent_type": WOLF_AGENT_TYPE,
                                                   "num_agents": MAX_WOLF,
                                                   "num_active_agents": INIT_WOLF
                                                   },
                                   "policy_params_sheep": {"num_neurons": NUM_NEURONS,
                                                     "num_obs": NUM_OBS,
                                                     "num_actions": NUM_ACTIONS
                                                    },
                                   "policy_params_wolf": {"num_neurons": NUM_NEURONS,
                                                     "num_obs": NUM_OBS,
                                                     "num_actions": NUM_ACTIONS
                                                    }})

# Agents ----------------------------------------------------------------------------------------------------------

@struct.dataclass
class Sheep(Agent):
    @staticmethod
    def create_agent(type, params, id, active_state, key):
        policy = params.content["policy"]

        key, *subkeys = random.split(key, 5)

        x_max = params.content["x_max"]
        y_max = params.content["y_max"]
        energy_begin_max = params.content["energy_begin_max"]
        eat_rate = params.content["eat_rate"]
        radius = params.content["radius"]

        params_content = {"radius": radius,
                          "x_max": x_max, "y_max": y_max,
                          "energy_begin_max": energy_begin_max,
                          "eat_rate": eat_rate,
                          "immortal_flag": False}
        params = Params(content=params_content)

        def create_active_agent():
            state_content = {"x": random.uniform(subkeys[0], shape=(1,), minval=-x_max, maxval=x_max),
                             "y": random.uniform(subkeys[1], shape=(1,), minval=-y_max, maxval=y_max),
                             "ang":  random.uniform(subkeys[2], shape=(1,), minval=-jnp.pi, maxval=jnp.pi),

                             "x_dot": jnp.zeros((1,), dtype=jnp.float32),
                             "y_dot": jnp.zeros((1,), dtype=jnp.float32),
                             "ang_dot": jnp.zeros((1,), dtype=jnp.float32),

                             "energy": random.uniform(subkeys[3], shape=(1,), minval=0.0, maxval=energy_begin_max),
                             "fitness": jnp.array([0.0]),

                             "time_death": jnp.zeros((1,), dtype=jnp.float32), # time below death energy threshold
                             "time_reproduction": jnp.zeros((1,), dtype=jnp.float32), # time above reproduction energy threshold

                             "reproduce_flag": False, # flag to indicate if the agent is ready to reproduce
                             "death_flag": False, # flag to indicate if the agent is ready to die
            }
            state = State(content=state_content)
            return state

        def create_inactive_agent():
            state_content = {"x": jnp.array([INACTIVE_POS]),
                             "y": jnp.array([INACTIVE_POS]),
                             "ang": jnp.array([0.0]),
                             "x_dot": jnp.zeros((1,)),
                             "y_dot": jnp.zeros((1,)),
                             "ang_dot": jnp.zeros((1,)),
                             "energy": jnp.array([INACTIVE_ENERGY]),
                             "fitness": jnp.array([0.0]),
                             "time_death": jnp.zeros((1,), dtype=jnp.float32),
                             "time_reproduction": jnp.zeros((1,), dtype=jnp.float32),
                             "reproduce_flag": False, "death_flag": False
            }
            state = State(content=state_content)
            return state

        agent_state = jax.lax.cond(active_state, lambda _: create_active_agent(),
                                   lambda _: create_inactive_agent(), None)

        return Sheep(id=id, state=agent_state, params=params, active_state=active_state, agent_type=type, age=0.0, key=key ,policy=policy)


    @staticmethod
    def step_agent(agent, input, step_params):
        def step_active_agent():
            # input data
            obs_rays = input.content["obs"]
            energy_intake = input.content["energy_intake"] # also handles energy output (if eaten by wolves)
            is_in_flag = input.content["is_in_flag"]

            # current agent state
            x = agent.state.content["x"]
            y = agent.state.content["y"]
            ang = agent.state.content["ang"]
            x_dot = agent.state.content["x_dot"]
            y_dot = agent.state.content["y_dot"]
            ang_dot = agent.state.content["ang_dot"]

            energy = agent.state.content["energy"]
            fitness = agent.state.content["fitness"]

            time_death = agent.state.content['time_death']
            time_reproduction = agent.state.content['time_reproduction']
            reproduce_flag = agent.state.content['reproduce_flag']
            death_flag = agent.state.content['death_flag']

            death_energy_threshold = step_params.content['death_energy_threshold']
            death_time_threshold = step_params.content['death_time_threshold']
            repro_energy_threshold = step_params.content['repro_energy_threshold']
            repro_time_threshold = step_params.content['repro_time_threshold']


            dist_from_center = jnp.sqrt(jnp.square(x) + jnp.square(y))

            obs_content = {'obs': jnp.concatenate((obs_rays,
                                                   energy,
                                                   jnp.array([energy_intake]).reshape(1),
                                                   jnp.array([is_in_flag]).reshape(1),
                                                   dist_from_center,
                                                   x_dot, y_dot, ang_dot,
                                                   ang), axis=0)}

            obs = Signal(content=obs_content)

            # new policy
            new_policy = CTRNN.step_policy(agent.policy, obs, step_params)

            x_max_arena = step_params.content["x_max_arena"]
            y_max_arena = step_params.content["y_max_arena"]

            action = new_policy.state.content["action"]
            forward_action = action[0]  # tanh (-1 to 1)
            angular_action = action[1]  # tanh (-1 to 1)

            key, *noise_keys = random.split(agent.key, 3)

            # fixed base speed (with noise)
            speed = ((LINEAR_ACTION_OFFSET + SHEEP_LINEAR_ACTION_SCALE * forward_action) *
                     (1 + NOISE_SCALE * jax.random.normal(noise_keys[0], ())))
            ang_speed = SHEEP_ANGULAR_SPEED_SCALE * angular_action * \
                        (1 + NOISE_SCALE * jax.random.normal(noise_keys[1], ()))

            # update positions
            x_new = jnp.clip(x + Dt * x_dot, -x_max_arena, x_max_arena)
            y_new = jnp.clip(y + Dt * y_dot, -y_max_arena, y_max_arena)
            ang_new = jnp.mod(ang + Dt * ang_dot + jnp.pi, 2 * jnp.pi) - jnp.pi

            x_dot_new = jnp.array([speed * jnp.cos(ang[0])])
            y_dot_new = jnp.array([speed * jnp.sin(ang[0])])
            ang_dot_new = jnp.array([ang_speed])

            # energy and fitness update
            metabolic_cost = step_params.content["metabolic_cost"]
            energy_new = energy + energy_intake - metabolic_cost  # energy_intake already includes loss to wolves
            energy_new = jnp.clip(energy_new, 0.0, MAX_ENERGY) # clip energy to max energy
            fitness_new = fitness + energy_intake - metabolic_cost

            # update death and reproduction
            time_death_new = jax.lax.cond(energy_new[0] < death_energy_threshold, lambda _: time_death + Dt,
                                          lambda _: jnp.zeros((1,), dtype=jnp.float32), None)
            death_flag_new = jax.lax.cond(death_flag, lambda _: True,
                                          lambda _: time_death_new[0] >= death_time_threshold,
                                          None)

            death_flag_new = jax.lax.cond(agent.params.content['immortal_flag'], lambda _: False,
                                          lambda _: death_flag_new,
                                          None)  # if immortal flag is true, agent cannot die, else it can die based on energy and time below threshold


            time_reproduction_new = jax.lax.cond(energy_new[0] > repro_energy_threshold,
                                                 lambda _: time_reproduction + Dt,
                                                 lambda _: jnp.zeros((1,), dtype=jnp.float32), None)
            reproduce_flag_new = jax.lax.cond(reproduce_flag, lambda _: True,
                                              lambda _: time_reproduction_new[0] >= repro_time_threshold,
                                              None)  # can become false, only from the outside when the agent is reset, otherwise once true it stays



            new_state_content = {"x": x_new, "y": y_new, "x_dot": x_dot_new, "y_dot": y_dot_new, "ang": ang_new,
                                 "ang_dot": ang_dot_new,
                                 "energy": energy_new, "fitness": fitness_new,
                                 "time_death": time_death_new,
                                 "death_flag": death_flag_new,
                                 "time_reproduction": time_reproduction_new,
                                 "reproduce_flag": reproduce_flag_new}

            new_state = State(content=new_state_content)

            return agent.replace(state=new_state, key=key, age=agent.age + Dt, policy=new_policy)


        def step_inactive_agent():
            return agent

        return jax.lax.cond(agent.active_state, lambda _: step_active_agent(), lambda _: step_inactive_agent(), None)

    @staticmethod
    def reset_agent(agent):
        x_max = agent.params.content["x_max"]
        y_max = agent.params.content["y_max"]
        energy_begin_max = agent.params.content["energy_begin_max"]
        policy = CTRNN.reset_policy(agent.policy)
        key = agent.key

        #init_count = reset_params.content["init_count"]
        active_state = jnp.where(agent.id < INIT_SHEEP, 1, 0)

        key, *subkeys = random.split(key, 5)
        x = random.uniform(subkeys[0], shape=(1,), minval=-x_max, maxval=x_max)
        y = random.uniform(subkeys[1], shape=(1,), minval=-y_max, maxval=y_max)
        ang = random.uniform(subkeys[2], shape=(1,), minval=-jnp.pi, maxval=jnp.pi)

        x = jnp.where(active_state == 1, x, jnp.array([INACTIVE_POS]))
        y = jnp.where(active_state == 1, y, jnp.array([INACTIVE_POS]))

        x_dot = jnp.zeros((1,), dtype=jnp.float32)
        y_dot = jnp.zeros((1,), dtype=jnp.float32)
        ang_dot = jnp.zeros((1,), dtype=jnp.float32)

        energy = random.uniform(subkeys[3], shape=(1,), minval=0.0, maxval=energy_begin_max)
        energy = jnp.where(active_state == 1, energy, jnp.array([INACTIVE_ENERGY]))
        fitness = jnp.array([0.0])

        state_content = {"x": x, "y": y, "ang": ang, "x_dot": x_dot, "y_dot": y_dot, "ang_dot": ang_dot,
                         "energy": energy, "fitness": fitness,
                         "time_death": jnp.zeros((1,), dtype=jnp.float32),
                         "time_reproduction": jnp.zeros((1,), dtype=jnp.float32),
                         "reproduce_flag": False,
                         "death_flag": False
                         }
        state = State(content=state_content)

        return agent.replace(state=state, age=0.0, active_state=active_state, key=key, policy=policy)


@struct.dataclass
class Wolf(Agent):
    @staticmethod
    def create_agent(type, params, id, active_state, key):
        policy = params.content["policy"]
        key, *subkeys = random.split(key, 5)

        x_max = params.content["x_max"]
        y_max = params.content["y_max"]
        energy_begin_max = params.content["energy_begin_max"]
        radius = params.content["radius"]

        params_content = {"radius": radius, "x_max": x_max, "y_max": y_max,
                          "energy_begin_max": energy_begin_max,
                          "immortal_flag": False}
        params = Params(content=params_content)

        def create_active_agent():
            state_content = {"x": random.uniform(subkeys[0], shape=(1,), minval=-x_max, maxval=x_max),
                             "y": random.uniform(subkeys[1], shape=(1,), minval=-y_max, maxval=y_max),
                             "ang": random.uniform(subkeys[2], shape=(1,), minval=-jnp.pi, maxval=jnp.pi),

                             "x_dot": jnp.zeros((1,), dtype=jnp.float32),
                             "y_dot": jnp.zeros((1,), dtype=jnp.float32),
                             "ang_dot": jnp.zeros((1,), dtype=jnp.float32),

                             "energy": random.uniform(subkeys[3], shape=(1,), minval=0.0, maxval=energy_begin_max),
                             "fitness": jnp.array([0.0]),

                             "time_death": jnp.zeros((1,), dtype=jnp.float32),  # time below death energy threshold
                             "time_reproduction": jnp.zeros((1,), dtype=jnp.float32), # time above reproduction energy threshold

                             "reproduce_flag": False,  # flag to indicate if the agent is ready to reproduce
                             "death_flag": False,  # flag to indicate if the agent is ready to die
                             }
            state = State(content=state_content)
            return state

        def create_inactive_agent():
            state_content = {"x": jnp.array([INACTIVE_POS]),
                             "y": jnp.array([INACTIVE_POS]),
                             "ang": jnp.array([0.0]),
                             "x_dot": jnp.zeros((1,)),
                             "y_dot": jnp.zeros((1,)),
                             "ang_dot": jnp.zeros((1,)),
                             "energy": jnp.array([INACTIVE_ENERGY]),
                             "fitness": jnp.array([0.0]),
                             "time_death": jnp.zeros((1,), dtype=jnp.float32),
                             "time_reproduction": jnp.zeros((1,), dtype=jnp.float32),
                             "reproduce_flag": False, "death_flag": False
                             }
            state = State(content=state_content)
            return state

        agent_state = jax.lax.cond(active_state, lambda _: create_active_agent(),
                                   lambda _: create_inactive_agent(), None)
        return Wolf(id=id, state=agent_state, params=params, active_state=active_state, agent_type=type, age=0.0,
                    key=key, policy=policy)

    @staticmethod
    def step_agent(agent, input, step_params):
        def step_active_agent():
            # input data
            obs_rays = input.content["obs"]
            energy_intake = input.content["energy_intake"]
            is_in_flag = input.content["is_in_flag"]

            # current agent state
            x = agent.state.content["x"]
            y = agent.state.content["y"]
            ang = agent.state.content["ang"]
            x_dot = agent.state.content["x_dot"]
            y_dot = agent.state.content["y_dot"]
            ang_dot = agent.state.content["ang_dot"]

            energy = agent.state.content["energy"]
            fitness = agent.state.content["fitness"]

            time_death = agent.state.content['time_death']
            time_reproduction = agent.state.content['time_reproduction']
            reproduce_flag = agent.state.content['reproduce_flag']
            death_flag = agent.state.content['death_flag']

            death_energy_threshold = step_params.content['death_energy_threshold']
            death_time_threshold = step_params.content['death_time_threshold']
            repro_energy_threshold = step_params.content['repro_energy_threshold']
            repro_time_threshold = step_params.content['repro_time_threshold']


            distance_to_center = jnp.sqrt(jnp.square(x) + jnp.square(y))

            obs_content = {'obs': jnp.concatenate((obs_rays,
                                                   energy,
                                                   jnp.array([energy_intake]).reshape(1),
                                                   jnp.array([is_in_flag]).reshape(1),
                                                   distance_to_center,
                                                   x_dot, y_dot, ang_dot,
                                                   ang), axis=0)}

            obs = Signal(content=obs_content)

            new_policy = CTRNN.step_policy(agent.policy, obs, step_params)

            x_max_arena = step_params.content["x_max_arena"]
            y_max_arena = step_params.content["y_max_arena"]

            action = new_policy.state.content["action"]
            forward_action = action[0]  # tanh (-1 to 1)
            angular_action = action[1]  # tanh (-1 to 1)

            key, *noise_keys = random.split(agent.key, 3)

            # fixed base speed (with noise)
            speed = (LINEAR_ACTION_OFFSET + WOLF_LINEAR_ACTION_SCALE * forward_action) * (
                    1 + NOISE_SCALE * jax.random.normal(noise_keys[0], ()))
            ang_speed = WOLF_ANGULAR_SPEED_SCALE * angular_action * (
                        1 + NOISE_SCALE * jax.random.normal(noise_keys[1], ()))

            # update positions
            x_new = jnp.clip(x + Dt * x_dot, -x_max_arena, x_max_arena)
            y_new = jnp.clip(y + Dt * y_dot, -y_max_arena, y_max_arena)
            ang_new = jnp.mod(ang + Dt * ang_dot + jnp.pi, 2 * jnp.pi) - jnp.pi

            x_dot_new = jnp.array([speed * jnp.cos(ang[0])])
            y_dot_new = jnp.array([speed * jnp.sin(ang[0])])
            ang_dot_new = jnp.array([ang_speed])

            # energy and fitness update
            metabolic_cost = step_params.content["metabolic_cost"]
            energy_new = energy + energy_intake - metabolic_cost
            energy_new = jnp.clip(energy_new, 0.0, MAX_ENERGY)
            fitness_new = fitness + energy_intake - metabolic_cost

            # update death and reproduction
            time_death_new = jax.lax.cond(energy_new[0] < death_energy_threshold, lambda _: time_death + Dt,
                                          lambda _: jnp.zeros((1,), dtype=jnp.float32), None)
            death_flag_new = jax.lax.cond(death_flag, lambda _: True,
                                          lambda _: time_death_new[0] >= death_time_threshold,
                                          None)

            death_flag_new = jax.lax.cond(agent.params.content['immortal_flag'], lambda _: False,
                                          lambda _: death_flag_new,
                                          None)  # if immortal flag is true, agent cannot die, else it can die based on energy and time below threshold
            time_reproduction_new = jax.lax.cond(energy_new[0] > repro_energy_threshold,
                                                 lambda _: time_reproduction + Dt,
                                                 lambda _: jnp.zeros((1,), dtype=jnp.float32), None)
            reproduce_flag_new = jax.lax.cond(reproduce_flag, lambda _: True,
                                              lambda _: time_reproduction_new[0] >= repro_time_threshold,
                                              None)  # can become false, only from the outside when the agent is reset, otherwise once true it stays


            new_state_content = {"x": x_new, "y": y_new, "x_dot": x_dot_new, "y_dot": y_dot_new,
                                 "ang": ang_new, "ang_dot": ang_dot_new,
                                 "energy": energy_new, "fitness": fitness_new,
                                 "time_death": time_death_new,
                                 "death_flag": death_flag_new,
                                 "time_reproduction": time_reproduction_new,
                                 "reproduce_flag": reproduce_flag_new
                                 }
            new_state = State(content=new_state_content)

            return agent.replace(state=new_state, key=key, age=agent.age + Dt, policy=new_policy)

        def step_inactive_agent():
            return agent

        return jax.lax.cond(agent.active_state, lambda _: step_active_agent(), lambda _: step_inactive_agent(), None)

    @staticmethod
    def reset_agent(agent):
        x_max = agent.params.content["x_max"]
        y_max = agent.params.content["y_max"]
        energy_begin_max = agent.params.content["energy_begin_max"]

        policy = CTRNN.reset_policy(agent.policy)
        key = agent.key

        #init_count = reset_params.content["init_count"]
        active_state = jnp.where(agent.id < INIT_WOLF, 1, 0)

        key, *subkeys = random.split(key, 5)
        x = random.uniform(subkeys[0], shape=(1,), minval=-x_max, maxval=x_max)
        y = random.uniform(subkeys[1], shape=(1,), minval=-y_max, maxval=y_max)
        ang = random.uniform(subkeys[2], shape=(1,), minval=-jnp.pi, maxval=jnp.pi)

        x = jnp.where(active_state == 1, x, jnp.array([INACTIVE_POS]))
        y = jnp.where(active_state == 1, y, jnp.array([INACTIVE_POS]))

        x_dot = jnp.zeros((1,), dtype=jnp.float32)
        y_dot = jnp.zeros((1,), dtype=jnp.float32)
        ang_dot = jnp.zeros((1,), dtype=jnp.float32)

        energy = random.uniform(subkeys[3], shape=(1,), minval=0.0, maxval=energy_begin_max)
        energy = jnp.where(active_state == 1, energy, jnp.array([INACTIVE_ENERGY]))
        fitness = jnp.array([0.0])

        state_content = {"x": x, "y": y, "ang": ang, "x_dot": x_dot, "y_dot": y_dot, "ang_dot": ang_dot,
                         "energy": energy, "fitness": fitness,
                         "time_death": jnp.zeros((1,), dtype=jnp.float32),
                         "time_reproduction": jnp.zeros((1,), dtype=jnp.float32),
                         "reproduce_flag": False,
                         "death_flag": False
                         }
        state = State(content=state_content)

        return agent.replace(state=state, age=0.0, active_state=active_state, key=key, policy=policy)

# Functions for death and reproduction -----------------------------------------------------------------------------

def add_agent(agent, add_params):
    """
    Overwrites an inactive agent slot (agent) with data from a parent (agent_to_copy).
    """
    parent = add_params.content['agent_to_copy']
    child_J = add_params.content['child_J']  # inherited weight matrix

    key, *sub_keys = random.split(agent.key, 4)

    #spawn child near parent
    x = parent.state.content['x'] + random.uniform(sub_keys[0], shape=(1,), minval=-OFFSPRING_SPREAD, maxval=OFFSPRING_SPREAD)
    y = parent.state.content['y'] + random.uniform(sub_keys[1], shape=(1,), minval=-OFFSPRING_SPREAD, maxval=OFFSPRING_SPREAD)
    ang = random.uniform(sub_keys[2], shape=(1,), minval=-jnp.pi, maxval=jnp.pi)
    energy = parent.state.content['energy']/2.0 # split energy between parent and offspring

    state_content = {'x': x,
                     'y': y,
                     'ang': ang,

                     "x_dot": jnp.zeros((1,), dtype=jnp.float32),
                     "y_dot": jnp.zeros((1,), dtype=jnp.float32),
                     "ang_dot": jnp.zeros((1,), dtype=jnp.float32),

                     'energy': energy,
                     "fitness": jnp.array([0.0]),

                     "time_death": jnp.zeros((1,), dtype=jnp.float32),
                     "time_reproduction": jnp.zeros((1,), dtype=jnp.float32),
                     "reproduce_flag": False,
                     "death_flag": False
    }
    state = State(content=state_content)

    policy_params_content = { 'J': child_J,
                                 'tau': parent.policy.params.content['tau'], # everyone has the same time constants
                                 'E': parent.policy.params.content['E'], # everyone has the same mapping from observations to neurons
                                 'B': parent.policy.params.content['B'], # everyone has the same bias for each neuron
                                 'D': parent.policy.params.content['D']  # everyone has the same readout from neurons to actions
    }
    policy_params = Params(content = policy_params_content)
    child_policy = CTRNN.set_policy(agent.policy, policy_params)
    child_policy = CTRNN.reset_policy(child_policy)  # reset child's policy state

    return agent.replace(state=state, policy=child_policy, age=0.0, active_state=1, key=key)


def remove_agent(agent, remove_params):
    """
    Cleans up an agent slot on death --> moves it off-map and resets life-cycle flags.
    """
    state_content = {
        **agent.state.content,
        'energy': jnp.array([INACTIVE_ENERGY]),
        'fitness': jnp.array([0.0]),
        'x': jnp.array([INACTIVE_POS]),
        'y': jnp.array([INACTIVE_POS]),
        'x_dot': jnp.zeros((1,), dtype=jnp.float32),
        'y_dot': jnp.zeros((1,), dtype=jnp.float32),
        'ang_dot': jnp.zeros((1,), dtype=jnp.float32),
        'time_reproduction': jnp.zeros((1,), dtype=jnp.float32),
        'reproduce_flag': False,
        'time_death': jnp.zeros((1,), dtype=jnp.float32),
        'death_flag': False
    }
    new_state = State(content=state_content)
    return agent.replace(state=new_state, age=0.0, active_state=0)

def half_parent_energy(agent, params):
    """
    Called on the parent agent after birth to half its energy
    and reset its reproduction timer.
    """
    energy = agent.state.content['energy']
    new_energy = energy / 2.0  # parent gives half to the child

    state_content = {**agent.state.content,
                     'time_reproduction': jnp.zeros((1,), dtype=jnp.float32),
                     'reproduce_flag': False,
                     'energy': new_energy
    }
    new_state = State(content=state_content)
    return agent.replace(state=new_state)

# Agent interaction functions --------------------------------------------------------------------------------

def calculate_sheep_energy_intake(sheep: Sheep, base_energy_rate):
    """
    If a sheep is within the grass radius, it gets a fixed amount of resource.
    Returns the energy intake for all sheep.
    """
    xs_sheep = sheep.state.content["x"].reshape(-1)
    ys_sheep = sheep.state.content["y"].reshape(-1)
    active_sheep = sheep.active_state.astype(jnp.float32).reshape(-1)

    # dist to center
    dist = jnp.sqrt(jnp.square(xs_sheep) + jnp.square(ys_sheep))

    # check if sheep is within the grass patch (returns 1.0 if inside, 0.0 if outside)
    in_zone_mask = jnp.where(dist <= GRASS_RADIUS, 1.0, 0.0)

    # calculate fixed energy intake: BASE_ENERGY_RATE only if active and in the zone
    energy_intake = base_energy_rate * active_sheep * in_zone_mask
    return energy_intake

jit_calculate_sheep_energy_intake = jax.jit(calculate_sheep_energy_intake)

def wolves_sheep_interactions(sheep: Sheep, wolves: Wolf, eat_rate):
    """
    Wolves can only catch one sheep at a time.
    When multiple wolves catch the same sheep, each wolf gets EAT_RATE * energy_sheep
    """
    def wolf_sheep_interaction(one_wolf, sheep):
        xs_sheep = sheep.state.content["x"]
        ys_sheep = sheep.state.content["y"]
        x_wolf = one_wolf.state.content["x"]
        y_wolf = one_wolf.state.content["y"]

        active_sheep = sheep.active_state

        wolf_radius = one_wolf.params.content["radius"]

        distances = jnp.linalg.norm(jnp.stack((xs_sheep - x_wolf, ys_sheep - y_wolf), axis=1), axis=1).reshape(-1)
        is_in_range = jnp.where(jnp.logical_and(distances <= wolf_radius, active_sheep), 1.0, 0.0) # only consider active sheep

        # find the closest sheep; wolf can only catch one sheep at a time
        distances_masked = jnp.where(is_in_range > 0, distances, jnp.inf)
        closest_sheep_idx = jnp.argmin(distances_masked)

        is_catching_sheep = jnp.zeros_like(is_in_range)
        is_catching_sheep = is_catching_sheep.at[closest_sheep_idx].set(
            jnp.where(distances_masked[closest_sheep_idx] < jnp.inf, 1.0, 0.0)
        )
        return is_catching_sheep, is_in_range

    is_catching_matrix, touch_matrix = jax.vmap(wolf_sheep_interaction, in_axes=(0, None))(wolves, sheep) # shape (num_wolves, num_sheep)
    # is_being_fed_on = jnp.any(is_catching_matrix, axis=0)  # shape (num_sheep,) - t/f if sheep is being fed on by any wolf

    #split energy among wolves if multiple wolves target same sheep
    #num_wolves_at_sheep = jnp.maximum(jnp.sum(is_catching_matrix, axis=0), 1.0)
    #energy_sharing_matrix = jnp.divide(is_catching_matrix, num_wolves_at_sheep)

    energy_offer_per_sheep = sheep.state.content["energy"].reshape(-1) * eat_rate

    # calculate energy intake for each wolf
    #energy_intake_wolves = jnp.matmul(energy_sharing_matrix, energy_offer_per_sheep)
    energy_intake_wolves = jnp.matmul(is_catching_matrix, energy_offer_per_sheep)

    # calculate energy loss for each sheep
    #energy_loss_sheep = jnp.where(is_being_fed_on, energy_offer_per_sheep, 0.0)
    num_wolves_at_sheep = jnp.sum(is_catching_matrix, axis=0)
    energy_loss_sheep = num_wolves_at_sheep * energy_offer_per_sheep

    return energy_loss_sheep, energy_intake_wolves

jit_wolves_sheep_interactions = jax.jit(wolves_sheep_interactions)


def calculate_overlap_flags(agents, all_sheep, all_wolves):
    """
    Checks if an agent is overlapping with any other agent.
    Returns 1.0 if true, 0.0 if false.
    """
    def check_overlap(one_agent):
        xs_all = jnp.concatenate((all_sheep.state.content["x"].reshape(-1), all_wolves.state.content["x"].reshape(-1)))
        ys_all = jnp.concatenate((all_sheep.state.content["y"].reshape(-1), all_wolves.state.content["y"].reshape(-1)))
        active_all = jnp.concatenate((all_sheep.active_state.astype(jnp.float32).reshape(-1),
                                      all_wolves.active_state.astype(jnp.float32).reshape(-1)))
        radii_all = jnp.concatenate(
            (all_sheep.params.content["radius"].reshape(-1), all_wolves.params.content["radius"].reshape(-1)))

        x = one_agent.state.content["x"]
        y = one_agent.state.content["y"]
        own_radius = one_agent.params.content["radius"].reshape(-1)

        distances = jnp.linalg.norm(jnp.stack((xs_all - x, ys_all - y), axis=1), axis=1).reshape(-1)
        is_touching = jnp.where(jnp.logical_and(distances < (own_radius + radii_all), active_all), 1.0, 0.0)

        total_overlaps = jnp.sum(is_touching)
        self_active = one_agent.active_state.astype(jnp.float32).reshape(-1)
        overlaps_minus_self = total_overlaps - self_active

        # convert to boolean flag: 1.0 if overlapping with anyone else, 0.0 if not
        return jnp.where(overlaps_minus_self > 0.0, 1.0, 0.0)

    return jax.vmap(check_overlap)(agents).reshape(-1)

jit_calculate_overlap_flags = jax.jit(calculate_overlap_flags)

# CTRNN ---------------------------------------------------------------------------------------------------------

@struct.dataclass
class CTRNN(Policy):
    @staticmethod
    def create_policy(params: Params, key: jax.random.PRNGKey):
        num_neurons = params.content["num_neurons"]
        num_obs = params.content["num_obs"]
        num_actions = params.content["num_actions"]

        Z = jnp.zeros((num_neurons,), dtype=jnp.float32)
        action = jnp.zeros((num_actions,), dtype=jnp.float32)

        state = State(content={'Z': Z, 'action': action})

        J = jnp.zeros((num_neurons, num_neurons), dtype=jnp.float32)
        E = jnp.zeros((num_neurons, num_obs), dtype=jnp.float32)
        D = jnp.zeros((num_actions, num_neurons), dtype=jnp.float32)
        tau = jnp.zeros((num_neurons,), dtype=jnp.float32)
        B = jnp.zeros((num_neurons,), dtype=jnp.float32)

        params = Params(content={'J': J, 'E': E, 'D': D, 'tau': tau, 'B': B})
        return Policy(params=params, state=state, key=key)

    @staticmethod
    @jax.jit
    def step_policy(policy: Policy, input: Signal, params: Params):
        dt = params.content["dt"]
        action_scale = params.content["action_scale"]
        time_constant_scale = params.content["time_constant_scale"]

        J = policy.params.content["J"]
        E = policy.params.content["E"]
        D = policy.params.content["D"]
        tau = policy.params.content["tau"]
        B = policy.params.content["B"]

        Z = policy.state.content["Z"]

        obs = input.content["obs"]

        #step the policy
        z_dot = jnp.tanh(jnp.matmul(J, Z) + jnp.matmul(E, obs) + B) - Z
        z_dot = jnp.multiply(z_dot, time_constant_scale * jax.nn.sigmoid(tau))

        new_Z = Z + dt * z_dot # euler integration
        readout = jnp.matmul(D, new_Z)
        actions = action_scale * jnp.array([jax.nn.tanh(readout[0]), jax.nn.tanh(readout[1])]) # 0; speed, 1; angular speed

        new_policy_state = State(content={'Z': new_Z, 'action': actions})
        new_policy = policy.replace(state=new_policy_state)
        return new_policy

    @staticmethod
    @jax.jit
    def reset_policy(policy: Policy):
        Z = jnp.zeros_like(policy.state.content['Z'])
        action = jnp.zeros_like(policy.state.content['action'])

        new_policy_state = State(content={'Z': Z, 'action': action})
        new_policy = policy.replace(state=new_policy_state)

        return new_policy

    @staticmethod
    @jax.jit
    def set_policy(policy: Policy, set_params: Params):
        J = set_params.content['J']
        tau = set_params.content['tau']
        E = set_params.content['E']
        B = set_params.content['B']
        D = set_params.content['D']
        new_policy_params = Params(content={'J': J, 'tau': tau, 'E': E, 'B': B, 'D': D})
        return policy.replace(params=new_policy_params)

# CMA-ES --------------------------------------------------------------------------------------------------------

def set_CMAES_params(CMAES_params, agents):
    """
    copy the CMAES_params to the agents while manipulating the shape of the parameters
    Args:
        - CMAES_params: The parameters to set with shape (NUM_FORAGERS, NUM_ES_PARAMS)
        - agents: The agents to set the parameters to
    Returns:
        The updated agents
    """
    J = CMAES_params[:,:NUM_NEURONS*NUM_NEURONS].reshape((-1, NUM_NEURONS, NUM_NEURONS))
    last_index = NUM_NEURONS*NUM_NEURONS

    tau = CMAES_params[:, last_index:last_index + NUM_NEURONS].reshape((-1, NUM_NEURONS))
    last_index += NUM_NEURONS

    E = CMAES_params[:, last_index:last_index + NUM_NEURONS * NUM_OBS].reshape((-1, NUM_NEURONS, NUM_OBS))
    last_index += NUM_NEURONS*NUM_OBS

    B = CMAES_params[:, last_index:last_index + NUM_NEURONS].reshape((-1, NUM_NEURONS))
    last_index += NUM_NEURONS

    D = CMAES_params[:, last_index:last_index + NUM_NEURONS * NUM_ACTIONS].reshape((-1, NUM_ACTIONS, NUM_NEURONS))

    policy_params = Params(content={'J': J, 'tau': tau, 'E': E, 'B': B, 'D': D})
    new_policies = jax.vmap(CTRNN.set_policy)(agents.policy, policy_params)
    return agents.replace(policy=new_policies)

jit_set_CMAES_params = jax.jit(set_CMAES_params)

# Predator-prey world --------------------------------------------------------------------------------------------

@struct.dataclass
class PredatorPreyWorld:
    sheep_set: Set
    wolf_set: Set
    key: jax.random.PRNGKey

    @staticmethod
    def create_world(params, key):
        sheep_params = params.content["sheep_params"]
        wolf_params = params.content["wolf_params"]
        policy_params_sheep = params.content["policy_params_sheep"]
        policy_params_wolf = params.content["policy_params_wolf"]

        num_sheep_buffer = sheep_params["num_agents"]
        init_sheep_count = sheep_params["num_active_agents"]

        num_wolf_buffer = wolf_params["num_agents"]
        init_wolf_count = wolf_params["num_active_agents"]

        key, *policy_keys = jax.random.split(key, num_sheep_buffer + 1)
        policy_keys = jnp.array(policy_keys)

        policy_create_params_sheep = Params(content={'num_neurons': policy_params_sheep['num_neurons'],
                                               'num_obs': policy_params_sheep['num_obs'],
                                               'num_actions': policy_params_sheep['num_actions']})
        policies_sheep = jax.vmap(CTRNN.create_policy, in_axes=(None, 0))(policy_create_params_sheep, policy_keys)

        key, *policy_keys = jax.random.split(key, num_wolf_buffer + 1)
        policy_keys = jnp.array(policy_keys)
        policy_create_params_wolf = Params(content={'num_neurons': policy_params_wolf['num_neurons'],
                                               'num_obs': policy_params_wolf['num_obs'],
                                               'num_actions': policy_params_wolf['num_actions']})

        policies_wolf = jax.vmap(CTRNN.create_policy, in_axes=(None, 0))(policy_create_params_wolf, policy_keys)

        key, sheep_key = random.split(key, 2)

        x_max_array = jnp.tile(jnp.array([sheep_params["x_max"]]), (num_sheep_buffer,))
        y_max_array = jnp.tile(jnp.array([sheep_params["y_max"]]), (num_sheep_buffer,))
        energy_begin_max_array = jnp.tile(jnp.array([sheep_params["energy_begin_max"]]), (num_sheep_buffer,))
        eat_rate_array = jnp.tile(jnp.array([sheep_params["eat_rate"]]), (num_sheep_buffer,))
        radius_array = jnp.tile(jnp.array([sheep_params["radius"]]), (num_sheep_buffer,))


        sheep_create_params = Params(content= {
            "x_max": x_max_array,
            "y_max": y_max_array,
            "energy_begin_max": energy_begin_max_array,
            "eat_rate": eat_rate_array,
            "radius": radius_array,
            "policy": policies_sheep
        })

        sheep = create_agents(agent=Sheep, params=sheep_create_params, num_agents=num_sheep_buffer, num_active_agents=init_sheep_count,
                              agent_type=sheep_params["agent_type"], key=sheep_key)

        sheep_set = Set(num_agents=num_sheep_buffer, num_active_agents=init_sheep_count, agents=sheep, id=0, set_type=sheep_params["agent_type"],
                        params=None, state=None, policy=None, key=None)

        key, wolf_key = random.split(key, 2)
        x_max_array = jnp.tile(jnp.array([wolf_params["x_max"]]), (num_wolf_buffer,))
        y_max_array = jnp.tile(jnp.array([wolf_params["y_max"]]), (num_wolf_buffer,))
        energy_begin_max_array = jnp.tile(jnp.array([wolf_params["energy_begin_max"]]), (num_wolf_buffer,))
        radius_array = jnp.tile(jnp.array([wolf_params["radius"]]), (num_wolf_buffer,))

        wolf_create_params = Params(content= {"x_max": x_max_array,
                                              "y_max": y_max_array,
                                              "energy_begin_max": energy_begin_max_array,
                                              "radius": radius_array,
                                              "policy": policies_wolf
        })

        wolves = create_agents(agent=Wolf, params=wolf_create_params, num_agents=num_wolf_buffer, num_active_agents=init_wolf_count,
                               agent_type=wolf_params["agent_type"], key=wolf_key)

        wolf_set = Set(num_agents=num_wolf_buffer, num_active_agents=init_wolf_count, agents=wolves, id=2, set_type=wolf_params["agent_type"],
                       params=None, state=None, policy=None, key=None)


        return PredatorPreyWorld(sheep_set=sheep_set, wolf_set=wolf_set, key=key)




def step_world(pred_prey_world, _t, base_energy_rate, eat_rate, metab_sheep, metab_wolf,
               repro_energy_threshold_sheep, death_energy_threshold_sheep,
               repro_time_threshold_sheep, death_time_threshold_sheep,
               repro_energy_threshold_wolf, death_energy_threshold_wolf,
               repro_time_threshold_wolf, death_time_threshold_wolf):
    key, subkey = random.split(pred_prey_world.key)

    sheep_set = pred_prey_world.sheep_set
    wolf_set = pred_prey_world.wolf_set

    sheep_sensor_data, wolf_sensor_data = jit_get_all_agent_sensors(sheep_set.agents, wolf_set.agents, SHEEP_AGENT_TYPE, WOLF_AGENT_TYPE)

    energy_intake_from_environment = jit_calculate_sheep_energy_intake(sheep_set.agents, base_energy_rate)
    energy_loss_sheep, energy_intake_wolves = jit_wolves_sheep_interactions(sheep_set.agents, wolf_set.agents, eat_rate)

    sheep_overlap_flags = jit_calculate_overlap_flags(sheep_set.agents, sheep_set.agents, wolf_set.agents)
    wolf_overlap_flags = jit_calculate_overlap_flags(wolf_set.agents, sheep_set.agents, wolf_set.agents)

    patch_to_sheep_transfer = jnp.sum(energy_intake_from_environment)
    sheep_to_wolf_transfer = jnp.sum(energy_intake_wolves)

    sheep_step_input = Signal(content={"obs": sheep_sensor_data,
                                       "energy_intake": energy_intake_from_environment - energy_loss_sheep,
                                       "is_in_flag": sheep_overlap_flags # num of agents that are in the sheep
                                       })
    sheep_step_params = Params(content={"dt": Dt,
                                        "x_max_arena": WORLD_SIZE_X,
                                        "y_max_arena": WORLD_SIZE_Y,
                                        "action_scale": ACTION_SCALE,
                                        "time_constant_scale": TIME_CONSTANT_SCALE,
                                        "metabolic_cost": metab_sheep,
                                        "repro_energy_threshold": repro_energy_threshold_sheep,
                                        "death_energy_threshold": death_energy_threshold_sheep,
                                        "repro_time_threshold": repro_time_threshold_sheep,
                                        "death_time_threshold": death_time_threshold_sheep
    })

    wolf_step_input = Signal(content={"obs": wolf_sensor_data,
                                      "energy_intake": energy_intake_wolves,
                                      "is_in_flag": wolf_overlap_flags # num of agents that are in the wolf
                                      })
    wolf_step_params = Params(content={"dt": Dt,
                                       "x_max_arena": WORLD_SIZE_X,
                                       "y_max_arena": WORLD_SIZE_Y,
                                       "action_scale": ACTION_SCALE,
                                       "time_constant_scale": TIME_CONSTANT_SCALE,
                                       "metabolic_cost": metab_wolf,
                                       "repro_energy_threshold": repro_energy_threshold_wolf,
                                       "death_energy_threshold": death_energy_threshold_wolf,
                                       "repro_time_threshold": repro_time_threshold_wolf,
                                       "death_time_threshold": death_time_threshold_wolf
    })

    sheep_set = jit_step_agents(Sheep.step_agent, sheep_step_params, sheep_step_input, sheep_set)
    wolf_set = jit_step_agents(Wolf.step_agent, wolf_step_params, wolf_step_input, wolf_set)

    # remove agents
    dead_sheep_mask = jnp.where(sheep_set.agents.state.content['death_flag'] == True, 1,0)
    dead_wolf_mask = jnp.where(wolf_set.agents.state.content['death_flag'] == True, 1,0)

    sheep_mask_params = Params(content={'set_mask': dead_sheep_mask})
    wolf_mask_params = Params(content={'set_mask': dead_wolf_mask})

    sheep_set = jit_set_agents_mask(remove_agent,
                                    set_params = None,
                                    mask_params = sheep_mask_params,
                                    num_agents=-1,
                                    set=sheep_set)

    wolf_set = jit_set_agents_mask(remove_agent,
                                   set_params = None,
                                   mask_params = wolf_mask_params,
                                   num_agents=-1,
                                   set=wolf_set)

    # add agents
    select_sheep_mask = jnp.where(sheep_set.agents.active_state == False, 1,0) # inactive sheep
    change_sheep_mask = sheep_set.agents.state.content['reproduce_flag'] # sheep that can reproduce

    select_wolf_mask = jnp.where(wolf_set.agents.active_state == False, 1,0) # inactive wolves
    change_wolf_mask = wolf_set.agents.state.content['reproduce_flag'] # wolves that can reproduce

    sheep_child_Js = sheep_set.agents.policy.params.content['J'] # children inherit the parent's weights
    wolf_child_Js = wolf_set.agents.policy.params.content['J']


    sheep_add_params = Params(content={'agent_to_copy': sheep_set.agents, 'child_J': sheep_child_Js})
    sheep_repro_mask_params = Params(content={'select_mask': select_sheep_mask, 'change_mask': change_sheep_mask})

    wolf_add_params = Params(content={'agent_to_copy': wolf_set.agents, 'child_J': wolf_child_Js})
    wolf_repro_mask_params = Params(content={'select_mask': select_wolf_mask, 'change_mask': change_wolf_mask})

    sheep_set, num_sheep_births = jit_set_agents_rank_match(
        add_agent,
        set_params=sheep_add_params,
        mask_params=sheep_repro_mask_params,
        num_agents=-1,
        set=sheep_set
    )

    wolf_set, num_wolf_births = jit_set_agents_rank_match(
        add_agent,
        set_params=wolf_add_params,
        mask_params=wolf_repro_mask_params,
        num_agents=-1,
        set=wolf_set
    )

    sheep_set = jit_set_agents_mask(
        half_parent_energy,
        set_params = None,
        mask_params=Params(content={'set_mask': change_sheep_mask}),
        num_agents=num_sheep_births,
        set=sheep_set
    )

    wolf_set = jit_set_agents_mask(
        half_parent_energy,
        set_params = None,
        mask_params=Params(content={'set_mask': change_wolf_mask}),
        num_agents=num_wolf_births,
        set=wolf_set
    )

    num_active_sheep = jnp.sum(sheep_set.agents.active_state)
    num_active_wolves = jnp.sum(wolf_set.agents.active_state)


    render_data = Signal(content={"sheep_count": num_active_sheep,
                                  "wolf_count": num_active_wolves,

                                  "sheep_xs": sheep_set.agents.state.content["x"].reshape(-1, 1),
                                  "sheep_ys": sheep_set.agents.state.content["y"].reshape(-1, 1),
                                  "sheep_angles": sheep_set.agents.state.content["ang"].reshape(-1, 1),
                                  "sheep_energy": sheep_set.agents.state.content["energy"].reshape(-1, 1),
                                  "sheep_fitness": sheep_set.agents.state.content["fitness"].reshape(-1, 1),

                                  "wolf_xs": wolf_set.agents.state.content["x"].reshape(-1, 1),
                                  "wolf_ys": wolf_set.agents.state.content["y"].reshape(-1, 1),
                                  "wolf_angles": wolf_set.agents.state.content["ang"].reshape(-1, 1),
                                  "wolf_energy": wolf_set.agents.state.content["energy"].reshape(-1, 1),
                                  "wolf_fitness": wolf_set.agents.state.content["fitness"].reshape(-1, 1),

                                  "patch_sheep_transfer": patch_to_sheep_transfer,
                                  "sheep_wolf_transfer": sheep_to_wolf_transfer
    })

    return pred_prey_world.replace(sheep_set=sheep_set, wolf_set=wolf_set, key=key), render_data

jit_step_world = jax.jit(step_world)

# new run_dynamic_episode fn ------------------------------------------------------------------------------------

def run_dynamic_episode(pred_prey_world, base_energy_rate, eat_rate, metab_sheep, metab_wolf,
                        repro_energy_threshold_sheep, death_energy_threshold_sheep,
                        repro_time_threshold_sheep, death_time_threshold_sheep,
                        repro_energy_threshold_wolf, death_energy_threshold_wolf,
                        repro_time_threshold_wolf, death_time_threshold_wolf):
    
    sheep_set_agents = jax.vmap(Sheep.reset_agent)(pred_prey_world.sheep_set.agents)
    wolf_set_agents = jax.vmap(Wolf.reset_agent)(pred_prey_world.wolf_set.agents)

    pred_prey_world = pred_prey_world.replace(
        sheep_set=pred_prey_world.sheep_set.replace(agents=sheep_set_agents),
        wolf_set=pred_prey_world.wolf_set.replace(agents=wolf_set_agents)
    )

    def scan_step(world, t):
        new_world, render_data = step_world(
            world, t, base_energy_rate, eat_rate, metab_sheep, metab_wolf,
            repro_energy_threshold_sheep, death_energy_threshold_sheep,
            repro_time_threshold_sheep, death_time_threshold_sheep,
            repro_energy_threshold_wolf, death_energy_threshold_wolf,
            repro_time_threshold_wolf, death_time_threshold_wolf
        )
        return new_world, render_data

    ts = jnp.arange(EP_LEN)
    final_world, render_data = jax.lax.scan(scan_step, pred_prey_world, ts)

    render_data = Signal(content={
        "sheep_xs": render_data.content["sheep_xs"],
        "sheep_ys": render_data.content["sheep_ys"],
        "sheep_angles": render_data.content["sheep_angles"],
        "sheep_energy": render_data.content["sheep_energy"],
        "sheep_fitness": render_data.content["sheep_fitness"],
        "wolf_xs": render_data.content["wolf_xs"],
        "wolf_ys": render_data.content["wolf_ys"],
        "wolf_angles": render_data.content["wolf_angles"],
        "wolf_energy": render_data.content["wolf_energy"],
        "wolf_fitness": render_data.content["wolf_fitness"],
        "sheep_count": render_data.content["sheep_count"],
        "wolf_count": render_data.content["wolf_count"]
    })

    return final_world, render_data

jit_run_dynamic_episode = jax.jit(run_dynamic_episode)

# --------------------------------------------------------------------------------------------------------
def get_fitness(CMAES_params_sheep, CMAES_params_wolf, pred_prey_worlds):
    """
    Args:
        - CMAES_params_sheep: shape (NUM_SHEEP, NUM_ES_PARAMS)
        - CMAES_params_wolf: shape (NUM_WOLF, NUM_ES_PARAMS)
        - pred_prey_worlds: array of worlds, shape (NUM_WORLDS,)
    Returns:
        - sheep_fitness: Mean accumulated fitness per sheep
        - wolf_fitness: Mean accumulated fitness per wolf
        - render_data
        - pred_prey_worlds: updated worlds
    """
    def update_single_world(world):
        # apply the new evolved parameters to the agents
        new_sheep_agents = jit_set_CMAES_params(CMAES_params_sheep, world.sheep_set.agents)
        new_sheep_set = world.sheep_set.replace(agents=new_sheep_agents)

        new_wolf_agents = jit_set_CMAES_params(CMAES_params_wolf, world.wolf_set.agents)
        new_wolf_set = world.wolf_set.replace(agents=new_wolf_agents)

        return world.replace(sheep_set=new_sheep_set, wolf_set=new_wolf_set)

    # parallelise the parameter update across all worlds
    pred_prey_worlds = jax.vmap(update_single_world)(pred_prey_worlds)

    # run the simulation for specified EP_LEN using the default shared lifecycle settings
    def run_default_episode(world):
        return jit_run_dynamic_episode(
            world,
            BASE_ENERGY_RATE,
            EAT_RATE_SHEEP,
            BASIC_METABOLIC_COST_SHEEP,
            BASIC_METABOLIC_COST_WOLF,
            REPRODUCTION_ENERGY_THRESHOLD,
            DEATH_ENERGY_THRESHOLD,
            REPRODUCTION_TIME_THRESHOLD,
            DEATH_TIME_THRESHOLD,
            REPRODUCTION_ENERGY_THRESHOLD,
            DEATH_ENERGY_THRESHOLD,
            REPRODUCTION_TIME_THRESHOLD,
            DEATH_TIME_THRESHOLD
        )

    pred_prey_worlds, render_data = jax.vmap(run_default_episode)(pred_prey_worlds)

    # get agents fitness; average the total accumulated fitness across all parallel worlds
    sheep_fitness = jnp.mean(pred_prey_worlds.sheep_set.agents.state.content["fitness"], axis=0).reshape(-1)
    wolf_fitness = jnp.mean(pred_prey_worlds.wolf_set.agents.state.content["fitness"], axis=0).reshape(-1)

    return sheep_fitness, wolf_fitness, pred_prey_worlds, render_data

jit_get_fitness = jax.jit(get_fitness)

# phase 3 functions ----------------------------------------------------------------------------------------

LOSS_COMPONENT_NAMES = (
    "phase",
    "extinction",
    "ceiling",
    "amp",
    "turning",
    "crossing",
    "drift",
    #"terminal_edge",
)

def calculate_phase_loss(abm_sheep_traj: jnp.ndarray, abm_wolf_traj: jnp.ndarray, max_sheep: float, max_wolf: float):
    """
    Loss for LV-like predator-prey curves where sheep lead wolves.

    The key signal is lagged alignment: sheep population/slope now should predict
    wolf population/slope a little later. Boundary terms keep both populations
    alive and away from the hard capacity ceiling.
    """
    eps = 1e-8 # avoid dividing by 0

    # convert raw integers into floats between 0.0 and 1.0
    s = abm_sheep_traj.astype(jnp.float32) / max_sheep
    w = abm_wolf_traj.astype(jnp.float32) / max_wolf

    # Smooth integer birth/death jumps so the loss sees population trends.
    # Edge padding avoids artificial end-of-episode drops from zero padding.
    SMOOTH_WINDOW = 51
    kernel = jnp.ones((SMOOTH_WINDOW,), dtype=jnp.float32) / SMOOTH_WINDOW
    pad = SMOOTH_WINDOW // 2
    s_smooth = jnp.convolve(jnp.pad(s, (pad, pad), mode="edge"), kernel, mode="valid")
    w_smooth = jnp.convolve(jnp.pad(w, (pad, pad), mode="edge"), kernel, mode="valid")

    # pearson correlation coefficient
    def corrcoef(x, y):
        x = x - jnp.mean(x)
        y = y - jnp.mean(y)
        return jnp.mean(x * y) / ((jnp.std(x) + eps) * (jnp.std(y) + eps))

    # sheep should be the leading curve: sheep now resembles wolves later
    LEAD_LAG = 600 # 80
    s_lead = s_smooth[:-LEAD_LAG]
    w_follow = w_smooth[LEAD_LAG:]
    level_follow_corr = corrcoef(s_lead, w_follow)

    ds = s_smooth[1:] - s_smooth[:-1]
    dw = w_smooth[1:] - w_smooth[:-1]
    ds_lead = ds[:-LEAD_LAG]
    dw_follow = dw[LEAD_LAG:]
    slope_follow_corr = corrcoef(ds_lead, dw_follow)

    # keep the classical LV pressure: many sheep should allow wolf growth, and
    # many wolves should push sheep downward --> Weighted lower than the lead terms
    s_centered = s_smooth[:-1] - jnp.mean(s_smooth)
    w_centered = w_smooth[:-1] - jnp.mean(w_smooth)
    prey_supports_predator = corrcoef(s_centered, dw)
    predator_suppresses_prey = corrcoef(w_centered, -ds)

    phase_loss = (1.0 - level_follow_corr) + \
                 (1.0 - slope_follow_corr) + \
                 (0.5 * (1.0 - prey_supports_predator)) + \
                 (0.5 * (1.0 - predator_suppresses_prey))

    # stay within bounds; no extinction and no ceiling saturation
    LOWER_FRAC = 0.05
    UPPER_FRAC = 0.88
    lower_violation_s = jax.nn.relu(LOWER_FRAC - s)
    lower_violation_w = jax.nn.relu(LOWER_FRAC - w)
    upper_violation_s = jax.nn.relu(s - UPPER_FRAC)
    upper_violation_w = jax.nn.relu(w - UPPER_FRAC)

    extinction_loss = jnp.mean(lower_violation_s) + \
                      jnp.mean(lower_violation_w) + \
                      (2.0 * jax.nn.relu(LOWER_FRAC - jnp.min(s))) + \
                      (2.0 * jax.nn.relu(LOWER_FRAC - jnp.min(w)))

    hard_ceiling_s = s > 0.97
    hard_ceiling_w = w > 0.97
    ceiling_loss = jnp.mean(upper_violation_s) + \
                   jnp.mean(upper_violation_w) + \
                   (0.5 * jnp.mean(jnp.square(upper_violation_s))) + \
                   (0.5 * jnp.mean(jnp.square(upper_violation_w))) + \
                   jax.nn.relu(jnp.max(s) - 0.98) + \
                   jax.nn.relu(jnp.max(w) - 0.98) + \
                   (2.0 * jnp.mean(hard_ceiling_s.astype(jnp.float32))) + \
                   (2.0 * jnp.mean(hard_ceiling_w.astype(jnp.float32)))

    # require sustained waves in both halves of the episode
    midpoint = s_smooth.shape[0] // 2
    amp_s_first = jnp.max(s_smooth[:midpoint]) - jnp.min(s_smooth[:midpoint])
    amp_s_second = jnp.max(s_smooth[midpoint:]) - jnp.min(s_smooth[midpoint:])
    amp_w_first = jnp.max(w_smooth[:midpoint]) - jnp.min(w_smooth[:midpoint])
    amp_w_second = jnp.max(w_smooth[midpoint:]) - jnp.min(w_smooth[midpoint:])

    MIN_AMP = 0.20 # demands that the population swings by at least 20% in both halves
    amp_loss = jnp.square(jax.nn.relu(MIN_AMP - amp_s_first)) + \
               jnp.square(jax.nn.relu(MIN_AMP - amp_s_second)) + \
               jnp.square(jax.nn.relu(MIN_AMP - amp_w_first)) + \
               jnp.square(jax.nn.relu(MIN_AMP - amp_w_second))

    # A one-way climb can have large amplitude without being a cycle. Require
    # both upward and downward movement in each half of each population curve.
    def half_turning_loss(x):
        first = x[:midpoint]
        second = x[midpoint:]

        def segment_loss(segment):
            dx = segment[1:] - segment[:-1]
            total_up = jnp.sum(jax.nn.relu(dx))
            total_down = jnp.sum(jax.nn.relu(-dx))
            MIN_HALF_MOVE = 0.12
            return jnp.square(jax.nn.relu(MIN_HALF_MOVE - total_up)) + \
                   jnp.square(jax.nn.relu(MIN_HALF_MOVE - total_down))

        return segment_loss(first) + segment_loss(second)

    turning_loss = half_turning_loss(s_smooth) + half_turning_loss(w_smooth)

    # mean crossings to reject monotonic climb/crash trajectories.
    crossings_s = jnp.sum(jnp.abs(jnp.diff(jnp.sign(s_smooth - jnp.mean(s_smooth))))) / 2.0
    crossings_w = jnp.sum(jnp.abs(jnp.diff(jnp.sign(w_smooth - jnp.mean(w_smooth))))) / 2.0
    crossing_loss = jnp.square(jax.nn.relu(3.0 - crossings_s)) + \
                    jnp.square(jax.nn.relu(3.0 - crossings_w))

    # Penalize long-horizon drift using windows that scale with episode length.
    # This catches trajectories that look fine early but are still trending
    # toward extinction or the population ceiling near the end of the rollout.
    # compare the first ep_len/8 with the last ep_len/8 ts
    DRIFT_WINDOW = max(250, s_smooth.shape[0] // 8)
    early_s = jnp.mean(s_smooth[:DRIFT_WINDOW])
    late_s = jnp.mean(s_smooth[-DRIFT_WINDOW:])
    early_w = jnp.mean(w_smooth[:DRIFT_WINDOW])
    late_w = jnp.mean(w_smooth[-DRIFT_WINDOW:])

    DRIFT_TOL = 0.12 # allows a 12% difference before adding penalty
    mean_drift_loss = jnp.square(jax.nn.relu(jnp.abs(late_s - early_s) - DRIFT_TOL)) + \
                      jnp.square(jax.nn.relu(jnp.abs(late_w - early_w) - DRIFT_TOL))

    drift_loss = mean_drift_loss 

    # The episode may end in any cycle phase, including low or high population.
    # Only reject terminal extinction or exact capacity saturation.
    # final_edge_loss = (abm_sheep_traj[-1] <= 0).astype(jnp.float32) + \
    #                   (abm_wolf_traj[-1] <= 0).astype(jnp.float32) + \
    #                   (abm_sheep_traj[-1] >= max_sheep).astype(jnp.float32) + \
    #                   (abm_wolf_traj[-1] >= max_wolf).astype(jnp.float32)
    terminal_window = 250
    terminal_s = s[-terminal_window:]
    terminal_w = w[-terminal_window:]

    terminal_edge_loss = (
        jnp.mean(jax.nn.relu(LOWER_FRAC - terminal_s)) +
        jnp.mean(jax.nn.relu(LOWER_FRAC - terminal_w)) +
        jnp.mean(jax.nn.relu(terminal_s - UPPER_FRAC)) +
        jnp.mean(jax.nn.relu(terminal_w - UPPER_FRAC)) +
        (2.0 * jnp.mean((terminal_s > 0.97).astype(jnp.float32))) +
        (2.0 * jnp.mean((terminal_w > 0.97).astype(jnp.float32)))
    )

    loss_components = jnp.array([
        35.0 * phase_loss,
        1200.0 * extinction_loss,
        1000.0 * ceiling_loss,
        500.0 * amp_loss,
        800.0 * turning_loss,
        50.0 * crossing_loss,
        500.0 * drift_loss,
        #500.0 * terminal_edge_loss
    ])
    total_loss = jnp.sum(loss_components)

    return total_loss, loss_components

def get_fitness_phase3(env_cmaes_params, fixed_sheep_brains, fixed_wolf_brains, pred_prey_worlds):
    """
    Evaluates the ABM parameters using phase-space correlation and boundary penalties.
    Args:
        - env_cmaes_params: shape (NUM_WORLDS, 12) -> the parameters proposed by CMA-ES
        - fixed_*_brains: the frozen CTRNN weights from Phase 1 & 2
    """
    def eval_single_world(world, env_params):

        # ensure proposed params are within the bounds of the param ranges
        constrained_params = MIN_BOUNDS + (MAX_BOUNDS - MIN_BOUNDS) * jax.nn.sigmoid(env_params)

        base_energy_rate = constrained_params[0]
        eat_rate = constrained_params[1]
        metab_sheep = constrained_params[2]
        metab_wolf = constrained_params[3]
        repro_energy_threshold_sheep = constrained_params[4] * MAX_ENERGY
        death_energy_threshold_sheep = constrained_params[5] * MAX_ENERGY
        repro_time_threshold_sheep = constrained_params[6] * EP_LEN
        death_time_threshold_sheep = constrained_params[7] * EP_LEN
        repro_energy_threshold_wolf = constrained_params[8] * MAX_ENERGY
        death_energy_threshold_wolf = constrained_params[9] * MAX_ENERGY
        repro_time_threshold_wolf = constrained_params[10] * EP_LEN
        death_time_threshold_wolf = constrained_params[11] * EP_LEN

        # inject (evolved) brains into the agents
        new_sheep_agents = jit_set_CMAES_params(fixed_sheep_brains, world.sheep_set.agents)
        new_wolf_agents = jit_set_CMAES_params(fixed_wolf_brains, world.wolf_set.agents)

        world = world.replace(
            sheep_set=world.sheep_set.replace(agents=new_sheep_agents),
            wolf_set=world.wolf_set.replace(agents=new_wolf_agents)
        )

        # run the episode with the dynamic environmental parameters
        final_world, render_data = jit_run_dynamic_episode(
            world, base_energy_rate, eat_rate, metab_sheep, metab_wolf,
            repro_energy_threshold_sheep, death_energy_threshold_sheep,
            repro_time_threshold_sheep, death_time_threshold_sheep,
            repro_energy_threshold_wolf, death_energy_threshold_wolf,
            repro_time_threshold_wolf, death_time_threshold_wolf
        )

        # extract simulated trajectories
        abm_sheep_traj = render_data.content["sheep_count"]
        abm_wolf_traj = render_data.content["wolf_count"]

        # loss function call
        total_loss, loss_components = calculate_phase_loss(abm_sheep_traj, abm_wolf_traj, MAX_SHEEP, MAX_WOLF)

        return total_loss, loss_components, final_world, render_data

    # Vmap across all worlds in the CMA-ES population
    loss_scores, loss_components, final_worlds, render_data = jax.vmap(eval_single_world)(pred_prey_worlds, env_cmaes_params)
    return loss_scores, loss_components, final_worlds, render_data

jit_get_fitness_phase3 = jax.jit(get_fitness_phase3)

# MAIN -------------------------------------------------------------------------------------------------------
# parameter ranges to find the solution in
# [Base Energy, Eat Rate, Metab Sheep, Metab Wolf,
#  Sheep Repro Energy, Sheep Death Energy, Sheep Repro Time, Sheep Death Time,
#  Wolf Repro Energy, Wolf Death Energy, Wolf Repro Time, Wolf Death Time]
MIN_BOUNDS = jnp.array([
    0.02, 0.45, 0.006, 0.008, 0.30, 0.005, 0.003, 0.001, 0.30, 0.005, 0.003, 0.001
])

MAX_BOUNDS = jnp.array([
    0.50, 0.90, 0.090, 0.090, 0.60, 0.10, 0.020, 0.040, 0.60, 0.10, 0.040, 0.020
])

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--generations", type=int, default=PHASE3_GENERATIONS)
    parser.add_argument("--data-path", default=DATA_PATH)
    return parser.parse_args()

def normalize_data_path(data_path):
    return data_path if data_path.endswith(os.sep) else data_path + os.sep

def save_run_metadata(data_path, seed, generations, agent_condition, status, best_generation=None, best_loss=None):
    metadata = {
        "script": os.path.basename(__file__),
        "agent_condition": agent_condition,
        "seed": int(seed),
        "phase3_generations": int(generations),
        "ep_len": int(EP_LEN),
        "pop_size": int(POP_SIZE),
        "num_worlds": int(NUM_WORLDS),
        "max_sheep": int(MAX_SHEEP),
        "max_wolf": int(MAX_WOLF),
        "data_path": data_path,
        "loss_component_names": list(LOSS_COMPONENT_NAMES),
        "loss_component_count": len(LOSS_COMPONENT_NAMES),
        "status": status,
        "saved_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    if best_generation is not None:
        metadata["best_generation"] = int(best_generation)
    if best_loss is not None:
        metadata["best_loss"] = float(np.asarray(best_loss))

    with open(os.path.join(data_path, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

def main():
    args = parse_args()
    data_path = normalize_data_path(args.data_path)
    os.makedirs(data_path, exist_ok=True)
    save_run_metadata(data_path, args.seed, args.generations, "evolved", "running")
    jnp.save(data_path + 'phase3_seed.npy', jnp.array(args.seed))

    key = jax.random.PRNGKey(args.seed)
    key, world_key = random.split(key)
    eval_worlds = jax.vmap(PredatorPreyWorld.create_world, in_axes=(None, 0))(PP_WORLD_PARAMS, random.split(world_key, POP_SIZE))

    print("Loading frozen brains...")
    evolved_sheep_params = jnp.load(SHEEP_BRAIN_PATH + 'sheep_param_list_mean.npy')[-1]
    evolved_wolf_params = jnp.load(WOLF_BRAIN_PATH + 'wolf_param_list_mean.npy')[-1]
    
    sheep_params_tiled = jnp.tile(evolved_sheep_params, (MAX_SHEEP, 1))
    wolf_params_tiled = jnp.tile(evolved_wolf_params, (MAX_WOLF, 1))

    # print("generating random brains for testing")
    # key, sheep_rand_key, wolf_rand_key = jax.random.split(key, 3)

    # random_sheep_params = jax.random.uniform(sheep_rand_key, (NUM_ES_PARAMS,), minval=INIT_LOW, maxval=INIT_HIGH)
    # random_wolf_params = jax.random.uniform(wolf_rand_key, (NUM_ES_PARAMS,), minval=INIT_LOW, maxval=INIT_HIGH)

    # sheep_params_tiled = jnp.tile(random_sheep_params, (MAX_SHEEP, 1))
    # wolf_params_tiled = jnp.tile(random_wolf_params, (MAX_WOLF, 1))
    
    jnp.save(data_path + 'phase3_checkpoint_sheep_brain_params.npy', evolved_sheep_params)
    jnp.save(data_path + 'phase3_checkpoint_wolf_brain_params.npy', evolved_wolf_params)

    print("\n--- PHASE 3: Fitting AB Parameters to LV Equations ---")
    key, env_init_key = random.split(key)
    initial_env_mean = jax.random.uniform(env_init_key, (12,), minval=-1.0, maxval=1.0)

    strategy_env = CMA_ES(population_size=POP_SIZE, solution=initial_env_mean)
    es_params_env = strategy_env.default_params

    key, env_key = random.split(key)
    env_state = strategy_env.init(env_key, mean=initial_env_mean, params=es_params_env)

    env_param_history = []
    best_loss_history = []
    mean_loss_history = []
    best_component_history = []
    mean_component_history = []
    best_raw_params = initial_env_mean
    best_loss = jnp.inf
    best_generation = -1

    for gen in range(args.generations):
        key, ask_key, tell_key = jax.random.split(key, 3)
        x_env, env_state = strategy_env.ask(ask_key, env_state, es_params_env)

        losses, loss_components, _, _ = jit_get_fitness_phase3(x_env, sheep_params_tiled, wolf_params_tiled, eval_worlds)

        gen_best_idx = jnp.argmin(losses)
        gen_best_raw_params = x_env[gen_best_idx]
        gen_best_loss = losses[gen_best_idx]
        gen_best_components = loss_components[gen_best_idx]
        gen_mean_components = jnp.mean(loss_components, axis=0)
        if gen_best_loss < best_loss:
            best_loss = gen_best_loss
            best_raw_params = gen_best_raw_params
            best_generation = gen
            best_actual_params = MIN_BOUNDS + (MAX_BOUNDS - MIN_BOUNDS) * jax.nn.sigmoid(best_raw_params)
            jnp.save(data_path + 'phase3_checkpoint_best_raw_params.npy', best_raw_params)
            jnp.save(data_path + 'phase3_checkpoint_best_actual_params.npy', best_actual_params)
            jnp.save(data_path + 'phase3_checkpoint_best_loss.npy', jnp.array(best_loss))
            jnp.save(data_path + 'phase3_checkpoint_best_generation.npy', jnp.array(gen))
            jnp.save(data_path + 'phase3_checkpoint_best_loss_components.npy', gen_best_components)

        env_state, _ = strategy_env.tell(tell_key, x_env, losses, env_state, es_params_env)

        env_param_history.append(env_state.mean)
        best_loss_history.append(jnp.min(losses))
        mean_loss_history.append(jnp.mean(losses))
        best_component_history.append(gen_best_components)
        mean_component_history.append(gen_mean_components)

        if gen % 10 == 0 or gen == args.generations - 1:
            jnp.save(data_path + 'phase3_checkpoint_best_losses.npy', jnp.array(best_loss_history))
            jnp.save(data_path + 'phase3_checkpoint_mean_losses.npy', jnp.array(mean_loss_history))
            jnp.save(data_path + 'phase3_checkpoint_best_loss_components_history.npy', jnp.array(best_component_history))
            jnp.save(data_path + 'phase3_checkpoint_mean_loss_components_history.npy', jnp.array(mean_component_history))

            # apply the same constraints used in the fitness function for the display
            raw_best_env = env_state.mean
            actual_best_env = MIN_BOUNDS + (MAX_BOUNDS - MIN_BOUNDS) * jax.nn.sigmoid(raw_best_env)

            low_loss = jnp.min(losses)
            mean_loss = jnp.mean(losses)

            print(f'Gen {gen:04d} | Low Loss: {low_loss:.2f} | Mean Loss: {mean_loss:.2f}')
            print(f'          | Best Guess Params -> '
                  f'Grass:{actual_best_env[0]:.3f}, Eat:{actual_best_env[1]:.3f}, '
                  f'Meta_S:{actual_best_env[2]:.3f}, Meta_W:{actual_best_env[3]:.3f}, '
                  f'S_RepE:{actual_best_env[4]:.2f}, S_DthE:{actual_best_env[5]:.2f}, '
                  f'S_RepT:{actual_best_env[6] * EP_LEN:.0f}, S_DthT:{actual_best_env[7] * EP_LEN:.0f}, '
                  f'W_RepE:{actual_best_env[8]:.2f}, W_DthE:{actual_best_env[9]:.2f}, '
                  f'W_RepT:{actual_best_env[10] * EP_LEN:.0f}, W_DthT:{actual_best_env[11] * EP_LEN:.0f}')
            best_component_text = ", ".join(
                f"{name}:{value:.1f}" for name, value in zip(LOSS_COMPONENT_NAMES, gen_best_components)
            )
            mean_component_text = ", ".join(
                f"{name}:{value:.1f}" for name, value in zip(LOSS_COMPONENT_NAMES, gen_mean_components)
            )
            print(f'          | Best Loss Parts -> {best_component_text}')
            print(f'          | Mean Loss Parts -> {mean_component_text}')

    print("\nPhase 3 Training Complete. Saving data...")
    jnp.save(data_path + 'phase3_env_params_history.npy', jnp.array(env_param_history))
    jnp.save(data_path + 'phase3_best_losses.npy', jnp.array(best_loss_history))
    jnp.save(data_path + 'phase3_mean_losses.npy', jnp.array(mean_loss_history))
    jnp.save(data_path + 'phase3_best_loss_components_history.npy', jnp.array(best_component_history))
    jnp.save(data_path + 'phase3_mean_loss_components_history.npy', jnp.array(mean_component_history))
    np.save(data_path + 'phase3_loss_component_names.npy', np.array(LOSS_COMPONENT_NAMES))
    jnp.save(data_path + 'phase3_best_raw_params.npy', best_raw_params)
    jnp.save(data_path + 'phase3_best_actual_params.npy',
             MIN_BOUNDS + (MAX_BOUNDS - MIN_BOUNDS) * jax.nn.sigmoid(best_raw_params))
    save_run_metadata(data_path, args.seed, args.generations, "evolved", "complete", best_generation, best_loss)

    


    print(f"Data saved to {data_path}")

if __name__ == "__main__":
    main()

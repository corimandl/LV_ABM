"""
Clean simplified version of AB predator-prey simulation - equation 1

Characteristics:
- Two species (prey; sheep, predator; wolves)
    * same metabolism
    * same speed parameters
- No Energy/Mass/Speed trade-offs: fixed speed, fixed metabolic cost
- No reproduction and death dynamics
- Storing separate files for render data

updates:
- only 2 sensor channels: distance and agent type
- is_in number now calculated and passed to step_input in step_world function
- cleaned up hyperparameters
- removed sheep resource competition


obs_content back to normal 
changed sigmoid to tanh in ctrnn readout 
saving transfer rates 
    - step_world returns transfer data in render date
    - saving transfer data in the eval run

training only wolves

running eval episode to save data
"""
import os

from evosax.algorithms import CMA_ES

from structs import *
from functions import *

import jax.numpy as jnp
import jax.random as random
import jax
from flax import struct
from sensor import jit_get_all_agent_sensors


WORLD_SIZE_X = 500.0
WORLD_SIZE_Y = 500.0

MAX_SPAWN_X = 300.0
MAX_SPAWN_Y = 300.0

Dt = 0.1 # discrete time increments
KEY = jax.random.PRNGKey(42)
NOISE_SCALE = 0.05

# Sheep parameters
NUM_SHEEP = 10
SHEEP_RADIUS = 5.0
SHEEP_ENERGY_BEGIN_MAX = 50.0
SHEEP_MASS_BEGIN = 5.0 # initial sheep mass at birth
SHEEP_AGENT_TYPE = 1

GRASS_RADIUS = 700.0

# Wolf parameters
NUM_WOLF = 10
WOLF_RADIUS = 5.0
WOLF_ENERGY_BEGIN_MAX = 50.0
WOLF_MASS_BEGIN = 5.0
WOLF_AGENT_TYPE = 2

# Metabolism
METABOLIC_COST_SPEED = 0.02 # not being used
METABOLIC_COST_ANGULAR = 0.02 # not being used
BASIC_METABOLIC_COST_SHEEP = 0.04
BASIC_METABOLIC_COST_WOLF = 0.04

# Energy parameters (replaces grass)
#BASE_ENERGY_RATE = 0.5  # base energy gain for sheep per timestep in grass radius
#EAT_RATE_SHEEP = 0.6 # rate at which wolves consume sheep

MIN_BASE_ENERGY = 0.
MAX_BASE_ENERGY = 1.0
MIN_EAT_RATE = 0.3
MAX_EAT_RATE = 0.9

# Action parameters (sheep)
SHEEP_SPEED_MULTIPLIER = 2.0
SHEEP_LINEAR_ACTION_SCALE = SHEEP_SPEED_MULTIPLIER * SHEEP_RADIUS / Dt
SHEEP_ANGULAR_SPEED_SCALE = 2.0

# Action parameters (wolves)
WOLF_SPEED_MULTIPLIER = 2.0
WOLF_LINEAR_ACTION_SCALE = WOLF_SPEED_MULTIPLIER * WOLF_RADIUS / Dt
WOLF_ANGULAR_SPEED_SCALE = 2.0

# Sensors parameters
SHEEP_RAY_MAX_LENGTH = 300.0
WOLF_RAY_MAX_LENGTH = 300.0
RAY_RESOLUTION = 13  # W&B update
RAY_SPAN = jnp.pi # W&B update

# Controller parameters
NUM_OBS = RAY_RESOLUTION*2 + 8
NUM_NEURONS = 30
NUM_ACTIONS = 2
ACTION_SCALE = 1.0
LINEAR_ACTION_OFFSET = 0.0
TIME_CONSTANT_SCALE = 10.0 # speed of the neuron dynamics
NUM_ES_PARAMS = NUM_NEURONS * (NUM_NEURONS + NUM_OBS + NUM_ACTIONS + 2) # total number of parameters the Evolutionary Strategy needs to optimize


# Training parameters
NUM_WORLDS = 8
NUM_GENERATIONS = 2000
EP_LEN = 2000

SHEEP_TRAINING_STEPS = 50
WOLF_TRAINING_STEPS = 50

INACTIVE_POS = -5000.0  # moves inactive agents off the map
INACTIVE_ENERGY = -1.0  # placeholder for inactive agents
INIT_LOW = -1.0         # lower bound for initial ES weights
INIT_HIGH = 1.0         # upper bound for initial ES weights


#FITNESS_THRESH_SAVE = 150.0 # threshold for saving render data
#FITNESS_THRESH_SAVE_STEP = 10.0 # the amount by which we increase the threshold for saving render data

# save data
DATA_PATH = "./data/wolf_vs_rand7/"


# Predator-prey world parameters
PP_WORLD_PARAMS = Params(content= {"sheep_params": {"x_max": MAX_SPAWN_X,
                                                    "y_max": MAX_SPAWN_Y,
                                                    "energy_begin_max": SHEEP_ENERGY_BEGIN_MAX,
                                                    "mass_begin": SHEEP_MASS_BEGIN,
                                                    "radius": SHEEP_RADIUS,
                                                    "agent_type": SHEEP_AGENT_TYPE,
                                                    "num_sheep": NUM_SHEEP
                                                    },
                                   "wolf_params": {"x_max": MAX_SPAWN_X,
                                                   "y_max": MAX_SPAWN_Y,
                                                   "energy_begin_max": WOLF_ENERGY_BEGIN_MAX,
                                                   "mass_begin": WOLF_MASS_BEGIN,
                                                   "radius": WOLF_RADIUS,
                                                   "agent_type": WOLF_AGENT_TYPE,
                                                   "num_wolf": NUM_WOLF
                                                   },
                                   "policy_params_sheep": {"num_neurons": NUM_NEURONS,
                                                     "num_obs": NUM_OBS,
                                                     "num_actions": NUM_ACTIONS
                                                    },
                                   "policy_params_wolf": {"num_neurons": NUM_NEURONS,
                                                     "num_obs": NUM_OBS,
                                                     "num_actions": NUM_ACTIONS
                                                    }})


# Sheep dataclass
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
        base_energy_rate = params.content["base_energy_rate"]

        radius = params.content["radius"]
        mass_begin = params.content["mass_begin"]  # initial mass

        params_content = {"radius": radius, "x_max": x_max, "y_max": y_max, "energy_begin_max": energy_begin_max, "mass": mass_begin, "eat_rate": eat_rate, "base_energy_rate": base_energy_rate}
        params = Params(content=params_content)

        def create_active_agent():
            x = random.uniform(subkeys[0], shape=(1,), minval=-x_max, maxval=x_max) # random initial position (x,y)
            y = random.uniform(subkeys[1], shape=(1,), minval=-y_max, maxval=y_max)
            ang = random.uniform(subkeys[2], shape=(1,), minval=-jnp.pi, maxval=jnp.pi)  # random initial angle
            x_dot = jnp.zeros((1,), dtype=jnp.float32)  # initialize x velocity as 0
            y_dot = jnp.zeros((1,), dtype=jnp.float32)  # initialize y velocity as 0
            ang_dot = jnp.zeros((1,), dtype=jnp.float32)  # initialize angular velocity as 0

            energy = random.uniform(subkeys[3], shape=(1,), minval=0.5 * energy_begin_max, maxval=energy_begin_max)
            fitness = jnp.array([0.0])

            state_content = {"x": x, "y": y, "ang": ang, "x_dot": x_dot, "y_dot": y_dot, "ang_dot": ang_dot, "energy": energy, "fitness": fitness}
            state = State(content=state_content)
            return state

        def create_inactive_agent():
            state_content = {"x": jnp.array([INACTIVE_POS]), "y": jnp.array([INACTIVE_POS]), "ang": jnp.array([0.0]),
                             "x_dot": jnp.zeros((1,)), "y_dot": jnp.zeros((1,)), "ang_dot": jnp.zeros((1,)),
                             "energy": jnp.array([INACTIVE_ENERGY]), "fitness": jnp.array([0.0])} # placeholder values
            state = State(content=state_content)
            return state

        agent_state = jax.lax.cond(active_state, lambda _: create_active_agent(),
                                   lambda _: create_inactive_agent(), None)

        return Sheep(id=id, state=agent_state, params=params, active_state=active_state, agent_type=type, age=0.0, key=key,policy=policy)

    @staticmethod
    def step_agent(agent, input, step_params):
        def step_active_agent():
            # input
            obs_rays = input.content["obs"]
            energy_intake = input.content["energy_intake"] # also handles energy output (if eaten by wolves)
            is_in_flag = input.content["is_in_flag"]

            # current agent state
            energy = agent.state.content["energy"]
            fitness = agent.state.content["fitness"]
            x = agent.state.content["x"]
            y = agent.state.content["y"]
            ang = agent.state.content["ang"]
            x_dot = agent.state.content["x_dot"] # current x_velocity
            y_dot = agent.state.content["y_dot"] # current y_velocity
            ang_dot = agent.state.content["ang_dot"]

            #new
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

            dt = step_params.content["dt"]
            x_max_arena = step_params.content["x_max_arena"]
            y_max_arena = step_params.content["y_max_arena"]


            action = new_policy.state.content["action"]
            forward_action = action[0]  # sigmoid (0 to 1)
            angular_action = action[1]  # tanh (-1 to 1)

            key, *noise_keys = random.split(agent.key, 3)

            # fixed base speed (with noise)
            speed = ((LINEAR_ACTION_OFFSET + SHEEP_LINEAR_ACTION_SCALE * forward_action) *
                     (1 + NOISE_SCALE * jax.random.normal(noise_keys[0], ())))
            ang_speed = SHEEP_ANGULAR_SPEED_SCALE * angular_action * (1 + NOISE_SCALE * jax.random.normal(noise_keys[1], ()))

            # update positions
            x_new = jnp.clip(x + dt * x_dot, -x_max_arena, x_max_arena)
            y_new = jnp.clip(y + dt * y_dot, -y_max_arena, y_max_arena)
            ang_new = jnp.mod(ang + dt * ang_dot + jnp.pi, 2 * jnp.pi) - jnp.pi

            x_dot_new = jnp.array([speed * jnp.cos(ang[0])])
            y_dot_new = jnp.array([speed * jnp.sin(ang[0])])
            ang_dot_new = jnp.array([ang_speed])

            # fixed metabolic cost
            metabolic_cost = BASIC_METABOLIC_COST_SHEEP
            energy_new = energy + energy_intake - metabolic_cost # energy_intake already includes loss to wolves
            fitness_new = fitness + energy_intake - metabolic_cost

            agent_is_dead = energy_new[0] <= 0.0

            new_state_content = {"x": x_new, "y": y_new, "x_dot": x_dot_new, "y_dot": y_dot_new, "ang": ang_new, "ang_dot": ang_dot_new,
                                 "energy": energy_new, "fitness": fitness_new}
            new_state = State(content=new_state_content)

            return jax.lax.cond(
                agent_is_dead,
                lambda _: agent.replace(state=new_state, active_state=0),  # mark as dead/inactive
                lambda _: agent.replace(state=new_state, key=key, age=agent.age + dt, policy=new_policy), # add new policy
                None
            )
        def step_inactive_agent():
            return agent

        return jax.lax.cond(agent.active_state, lambda _: step_active_agent(), lambda _: step_inactive_agent(), None)

    @staticmethod
    def reset_agent(agent, reset_params):
        x_max = agent.params.content["x_max"]
        y_max = agent.params.content["y_max"]
        energy_begin_max = agent.params.content["energy_begin_max"]
        eat_rate = agent.params.content["eat_rate"]
        policy = CTRNN.reset_policy(agent.policy)
        key = agent.key

        key, *subkeys = random.split(key, 5)
        x = random.uniform(subkeys[0], shape=(1,), minval=-x_max, maxval=x_max)
        y = random.uniform(subkeys[1], shape=(1,), minval=-y_max, maxval=y_max)
        ang = random.uniform(subkeys[2], shape=(1,), minval=-jnp.pi, maxval=jnp.pi)
        x_dot = jnp.zeros((1,), dtype=jnp.float32)
        y_dot = jnp.zeros((1,), dtype=jnp.float32)
        ang_dot = jnp.zeros((1,), dtype=jnp.float32)

        energy = random.uniform(subkeys[3], shape=(1,), minval=0.5 * energy_begin_max, maxval=energy_begin_max)
        fitness = jnp.array([0.0])

        state_content = {"x": x, "y": y, "ang": ang, "x_dot": x_dot, "y_dot": y_dot, "ang_dot": ang_dot, "energy": energy, "fitness": fitness}
        state = State(content=state_content)

        return agent.replace(state=state, age=0.0, active_state=1, key=key, policy=policy)


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
        mass_begin = params.content["mass_begin"]

        params_content = {"radius": radius, "x_max": x_max, "y_max": y_max, "energy_begin_max": energy_begin_max, "mass": mass_begin}
        params = Params(content=params_content)

        def create_active_agent():
            x = random.uniform(subkeys[0], shape=(1,), minval=-x_max, maxval=x_max)  # random initial position (x,y)
            y = random.uniform(subkeys[1], shape=(1,), minval=-y_max, maxval=y_max)
            ang = random.uniform(subkeys[2], shape=(1,), minval=-jnp.pi, maxval=jnp.pi)  # random initial angle
            x_dot = jnp.zeros((1,), dtype=jnp.float32)  # initialize x velocity as 0
            y_dot = jnp.zeros((1,), dtype=jnp.float32)  # initialize y velocity as 0
            ang_dot = jnp.zeros((1,), dtype=jnp.float32)  # initialize angular velocity as 0

            energy = random.uniform(subkeys[3], shape=(1,), minval=0.5 * energy_begin_max, maxval=energy_begin_max)
            fitness = jnp.array([0.0])

            state_content = {"x": x, "y": y, "ang": ang, "x_dot": x_dot, "y_dot": y_dot, "ang_dot": ang_dot,
                             "energy": energy, "fitness": fitness}
            state = State(content=state_content)
            return state

        def create_inactive_agent():
            state_content = {"x": jnp.array([INACTIVE_POS]), "y": jnp.array([INACTIVE_POS]), "ang": jnp.array([0.0]),
                             "x_dot": jnp.zeros((1,)), "y_dot": jnp.zeros((1,)), "ang_dot": jnp.zeros((1,)),
                             "energy": jnp.array([INACTIVE_ENERGY]), "fitness": jnp.array([0.0])}
            state = State(content=state_content)
            return state

        agent_state = jax.lax.cond(active_state, lambda _: create_active_agent(),
                                   lambda _: create_inactive_agent(), None)
        return Wolf(id=id, state=agent_state, params=params, active_state=active_state, agent_type=type, age=0.0, key=key,policy=policy)

    @staticmethod
    def step_agent(agent, input, step_params):
        def step_active_agent():
            energy_intake = input.content["energy_intake"]
            obs_rays = input.content["obs"]
            is_in_flag = input.content["is_in_flag"]

            energy = agent.state.content["energy"]
            fitness = agent.state.content["fitness"]
            x = agent.state.content["x"]
            y = agent.state.content["y"]
            ang = agent.state.content["ang"]
            x_dot = agent.state.content["x_dot"]  # current x_velocity
            y_dot = agent.state.content["y_dot"]  # current y_velocity
            ang_dot = agent.state.content["ang_dot"]

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

            dt = step_params.content["dt"]
            x_max_arena = step_params.content["x_max_arena"]
            y_max_arena = step_params.content["y_max_arena"]

            action = new_policy.state.content["action"]
            forward_action = action[0]  # sigmoid (0 to 1)
            angular_action = action[1]  # tanh (-1 to 1)

            key, *noise_keys = random.split(agent.key, 3)

            # fixed base speed (with noise)
            speed = (LINEAR_ACTION_OFFSET + WOLF_LINEAR_ACTION_SCALE * forward_action) * (
                        1 + NOISE_SCALE * jax.random.normal(noise_keys[0], ()))
            ang_speed = WOLF_ANGULAR_SPEED_SCALE * angular_action * (1 + NOISE_SCALE * jax.random.normal(noise_keys[1], ()))

            # update positions
            x_new = jnp.clip(x + dt * x_dot, -x_max_arena, x_max_arena)
            y_new = jnp.clip(y + dt * y_dot, -y_max_arena, y_max_arena)
            ang_new = jnp.mod(ang + dt * ang_dot + jnp.pi, 2 * jnp.pi) - jnp.pi

            x_dot_new = jnp.array([speed * jnp.cos(ang[0])])
            y_dot_new = jnp.array([speed * jnp.sin(ang[0])])
            ang_dot_new = jnp.array([ang_speed])

            # metabolic cost
            metabolic_cost = BASIC_METABOLIC_COST_WOLF
            energy_new = energy + energy_intake - metabolic_cost
            fitness_new = fitness + energy_intake - metabolic_cost

            agent_is_dead = energy_new[0] <= 0.0

            new_state_content = {"x": x_new, "y": y_new, "x_dot": x_dot_new, "y_dot": y_dot_new, "ang": ang_new,
                                 "ang_dot": ang_dot_new, "energy": energy_new, "fitness": fitness_new}
            new_state = State(content=new_state_content)
            return jax.lax.cond(
                agent_is_dead,
                lambda _: agent.replace(state=new_state, active_state=0),  # mark as dead/inactive
                lambda _: agent.replace(state=new_state, key=key, age=agent.age + dt, policy=new_policy),
                None
            )
        def step_inactive_agent():
            return agent

        return jax.lax.cond(agent.active_state, lambda _: step_active_agent(), lambda _: step_inactive_agent(), None)

    @staticmethod
    def reset_agent(agent, reset_params):
        x_max = agent.params.content["x_max"]
        y_max = agent.params.content["y_max"]
        energy_begin_max = agent.params.content["energy_begin_max"]

        policy = CTRNN.reset_policy(agent.policy)
        key = agent.key

        key, *subkeys = random.split(key, 5)
        x = random.uniform(subkeys[0], shape=(1,), minval=-x_max, maxval=x_max)
        y = random.uniform(subkeys[1], shape=(1,), minval=-y_max, maxval=y_max)
        ang = random.uniform(subkeys[2], shape=(1,), minval=-jnp.pi, maxval=jnp.pi)
        x_dot = jnp.zeros((1,), dtype=jnp.float32)
        y_dot = jnp.zeros((1,), dtype=jnp.float32)
        ang_dot = jnp.zeros((1,), dtype=jnp.float32)

        energy = random.uniform(subkeys[3], shape=(1,), minval=0.5 * energy_begin_max, maxval=energy_begin_max)
        fitness = jnp.array([0.0])

        state_content = {"x": x, "y": y, "ang": ang, "x_dot": x_dot, "y_dot": y_dot, "ang_dot": ang_dot,
                         "energy": energy, "fitness": fitness}
        state = State(content=state_content)

        return agent.replace(state=state, age=0.0, active_state=1, key=key, policy=policy)



def calculate_sheep_energy_intake(sheep: Sheep):
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
    base_energy_rate = sheep.params.content["base_energy_rate"].reshape(-1)

    # calculate fixed energy intake: BASE_ENERGY_RATE only if active and in the zone
    energy_intake = base_energy_rate * active_sheep * in_zone_mask

    return energy_intake

jit_calculate_sheep_energy_intake = jax.jit(calculate_sheep_energy_intake)

def wolves_sheep_interactions(sheep: Sheep, wolves: Wolf):
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
    is_being_fed_on = jnp.any(is_catching_matrix, axis=0)  # shape (num_sheep,) - t/f if sheep is being fed on by any wolf

    #split energy among wolves if multiple wolves target same sheep
    num_wolves_at_sheep = jnp.maximum(jnp.sum(is_catching_matrix, axis=0), 1.0)
    energy_sharing_matrix = jnp.divide(is_catching_matrix, num_wolves_at_sheep)

    eat_rate = sheep.params.content["eat_rate"].reshape(-1)
    energy_offer_per_sheep = sheep.state.content["energy"].reshape(-1) * eat_rate

    # calculate energy intake for each wolf
    energy_intake_wolves = jnp.matmul(energy_sharing_matrix, energy_offer_per_sheep)

    # calculate energy loss for each sheep
    energy_loss_sheep = jnp.where(is_being_fed_on, energy_offer_per_sheep, 0.0)

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
        own_radius = one_agent.params.content["radius"].reshape()

        distances = jnp.linalg.norm(jnp.stack((xs_all - x, ys_all - y), axis=1), axis=1).reshape(-1)
        is_touching = jnp.where(jnp.logical_and(distances < (own_radius + radii_all), active_all), 1.0, 0.0)

        total_overlaps = jnp.sum(is_touching)
        self_active = one_agent.active_state.astype(jnp.float32).reshape()
        overlaps_minus_self = total_overlaps - self_active

        # convert to boolean flag: 1.0 if overlapping with anyone else, 0.0 if not
        return jnp.where(overlaps_minus_self > 0.0, 1.0, 0.0)

    return jax.vmap(check_overlap)(agents).reshape(-1)

jit_calculate_overlap_flags = jax.jit(calculate_overlap_flags)


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




@struct.dataclass
class PredatorPreyWorld:
    sheep_set: Set
    wolf_set: Set

    @staticmethod
    def create_world(params, env_rates, key):
        sheep_params = params.content["sheep_params"]
        wolf_params = params.content["wolf_params"]
        policy_params_sheep = params.content["policy_params_sheep"]
        policy_params_wolf = params.content["policy_params_wolf"]

        num_sheep = sheep_params["num_sheep"]
        num_wolf = wolf_params["num_wolf"]

        world_base_energy = env_rates[0]
        world_eat_rate = env_rates[1]

        key, *policy_keys = jax.random.split(key, num_sheep + 1)
        policy_keys = jnp.array(policy_keys)

        policy_create_params_sheep = Params(content={'num_neurons': policy_params_sheep['num_neurons'],
                                               'num_obs': policy_params_sheep['num_obs'],
                                               'num_actions': policy_params_sheep['num_actions']})
        policies_sheep = jax.vmap(CTRNN.create_policy, in_axes=(None, 0))(policy_create_params_sheep, policy_keys)

        key, *policy_keys = jax.random.split(key, num_wolf + 1)
        policy_keys = jnp.array(policy_keys)
        policy_create_params_wolf = Params(content={'num_neurons': policy_params_wolf['num_neurons'],
                                               'num_obs': policy_params_wolf['num_obs'],
                                               'num_actions': policy_params_wolf['num_actions']})

        policies_wolf = jax.vmap(CTRNN.create_policy, in_axes=(None, 0))(policy_create_params_wolf, policy_keys)

        key, sheep_key = random.split(key, 2)

        x_max_array = jnp.tile(jnp.array([sheep_params["x_max"]]), (num_sheep,))
        y_max_array = jnp.tile(jnp.array([sheep_params["y_max"]]), (num_sheep,))
        energy_begin_max_array = jnp.tile(jnp.array([sheep_params["energy_begin_max"]]), (num_sheep,))

        eat_rate_array = jnp.tile(jnp.array([world_eat_rate]), (num_sheep,))
        base_energy_rate_array = jnp.tile(jnp.array([world_base_energy]), (num_sheep,))

        radius_array = jnp.tile(jnp.array([sheep_params["radius"]]), (num_sheep,))
        mass_array = jnp.tile(jnp.array([sheep_params["mass_begin"]]), (num_sheep,))

        sheep_create_params = Params(content= {
            "x_max": x_max_array,
            "y_max": y_max_array,
            "energy_begin_max": energy_begin_max_array,

            "eat_rate": eat_rate_array,
            "base_energy_rate": base_energy_rate_array,

            "radius": radius_array,
            "mass_begin": mass_array,
            "policy": policies_sheep
        })

        sheep = create_agents(agent=Sheep, params=sheep_create_params, num_agents=num_sheep, num_active_agents=num_sheep,
                              agent_type=sheep_params["agent_type"], key=sheep_key)

        sheep_set = Set(num_agents=num_sheep, num_active_agents=num_sheep, agents=sheep, id=0, set_type=sheep_params["agent_type"],
                        params=None, state=None, policy=None, key=None)

        key, wolf_key = random.split(key, 2)
        x_max_array = jnp.tile(jnp.array([wolf_params["x_max"]]), (num_wolf,))
        y_max_array = jnp.tile(jnp.array([wolf_params["y_max"]]), (num_wolf,))
        energy_begin_max_array = jnp.tile(jnp.array([wolf_params["energy_begin_max"]]), (num_wolf,))
        radius_array = jnp.tile(jnp.array([wolf_params["radius"]]), (num_wolf,))
        mass_array = jnp.tile(jnp.array([wolf_params["mass_begin"]]), (num_wolf,))

        wolf_create_params = Params(content= {"x_max": x_max_array,
                                              "y_max": y_max_array,
                                              "energy_begin_max": energy_begin_max_array,
                                              "radius": radius_array,
                                              "mass_begin": mass_array,
                                              "policy": policies_wolf
        })

        wolves = create_agents(agent=Wolf, params=wolf_create_params, num_agents=num_wolf, num_active_agents=num_wolf,
                               agent_type=wolf_params["agent_type"], key=wolf_key)

        wolf_set = Set(num_agents=num_wolf, num_active_agents=num_wolf, agents=wolves, id=2, set_type=wolf_params["agent_type"],
                       params=None, state=None, policy=None, key=None)


        return PredatorPreyWorld(sheep_set=sheep_set, wolf_set=wolf_set)




def step_world(pred_prey_world, _t):
    sheep_set = pred_prey_world.sheep_set
    wolf_set = pred_prey_world.wolf_set

    sheep_sensor_data, wolf_sensor_data = jit_get_all_agent_sensors(
        sheep_set.agents, wolf_set.agents, SHEEP_AGENT_TYPE, WOLF_AGENT_TYPE)

    energy_intake_from_environment = jit_calculate_sheep_energy_intake(sheep_set.agents)
    energy_loss_sheep, energy_intake_wolves = jit_wolves_sheep_interactions(sheep_set.agents, wolf_set.agents)

    sheep_overlap_flags = jit_calculate_overlap_flags(sheep_set.agents, sheep_set.agents, wolf_set.agents)
    wolf_overlap_flags = jit_calculate_overlap_flags(wolf_set.agents, sheep_set.agents, wolf_set.agents)

    patch_to_sheep_transfer = jnp.sum(energy_intake_from_environment)
    sheep_to_wolf_transfer = jnp.sum(energy_intake_wolves)



    sheep_step_input = Signal(content={"obs": sheep_sensor_data,
                                       "energy_intake": energy_intake_from_environment - energy_loss_sheep,
                                       "is_in_flag": sheep_overlap_flags # num of agents that are in the sheep
                                       })
    sheep_step_params = Params(content={"dt": Dt,
                                        "metabolic_cost_speed": METABOLIC_COST_SPEED,
                                        "metabolic_cost_angular": METABOLIC_COST_ANGULAR,
                                        "x_max_arena": WORLD_SIZE_X,
                                        "y_max_arena": WORLD_SIZE_Y,
                                        "action_scale": ACTION_SCALE,
                                        "time_constant_scale": TIME_CONSTANT_SCALE
    })
    sheep_set = jit_step_agents(Sheep.step_agent, sheep_step_params, sheep_step_input, sheep_set)


    wolf_step_input = Signal(content={"obs": wolf_sensor_data,
                                      "energy_intake": energy_intake_wolves,
                                      "is_in_flag": wolf_overlap_flags # num of agents that are in the wolf
                                      })
    wolf_step_params = Params(content={"dt": Dt,
                                       "metabolic_cost_speed": METABOLIC_COST_SPEED,
                                       "metabolic_cost_angular": METABOLIC_COST_ANGULAR,
                                       "x_max_arena": WORLD_SIZE_X,
                                       "y_max_arena": WORLD_SIZE_Y,
                                       "action_scale": ACTION_SCALE,
                                       "time_constant_scale": TIME_CONSTANT_SCALE
    })
    wolf_set = jit_step_agents(Wolf.step_agent, wolf_step_params, wolf_step_input, wolf_set)

    # render_data = None
    # comment this out for training
    render_data = Signal(content={"sheep_xs": sheep_set.agents.state.content["x"].reshape(-1, 1),
                                  "sheep_ys": sheep_set.agents.state.content["y"].reshape(-1, 1),
                                  "sheep_angles": sheep_set.agents.state.content["ang"].reshape(-1, 1),
                                  "sheep_energy": sheep_set.agents.state.content["energy"].reshape(-1, 1),
                                  "sheep_fitness": sheep_set.agents.state.content["fitness"].reshape(-1, 1),
                                  "patch_sheep_transfer": patch_to_sheep_transfer,
                                  "wolf_xs": wolf_set.agents.state.content["x"].reshape(-1, 1),
                                  "wolf_ys": wolf_set.agents.state.content["y"].reshape(-1, 1),
                                  "wolf_angles": wolf_set.agents.state.content["ang"].reshape(-1, 1),
                                  "wolf_energy": wolf_set.agents.state.content["energy"].reshape(-1, 1),
                                  "wolf_fitness": wolf_set.agents.state.content["fitness"].reshape(-1, 1),
                                  "sheep_wolf_transfer": sheep_to_wolf_transfer
    })

    return pred_prey_world.replace(sheep_set=sheep_set, wolf_set=wolf_set), render_data

jit_step_world = jax.jit(step_world)


def reset_world(pred_prey_world):
    sheep_set_agents = pred_prey_world.sheep_set.agents
    wolf_set_agents = pred_prey_world.wolf_set.agents

    sheep_set_agents = jax.vmap(Sheep.reset_agent)(sheep_set_agents, None)
    wolf_set_agents = jax.vmap(Wolf.reset_agent)(wolf_set_agents, None)

    sheep_set = pred_prey_world.sheep_set.replace(agents=sheep_set_agents)
    wolf_set = pred_prey_world.wolf_set.replace(agents=wolf_set_agents)

    return pred_prey_world.replace(sheep_set=sheep_set, wolf_set=wolf_set)

jit_reset_world = jax.jit(reset_world)


def scan_episode(pred_prey_world: PredatorPreyWorld, ts):
    return jax.lax.scan(jit_step_world, pred_prey_world, ts)

jit_scan_episode = jax.jit(scan_episode)

def run_episode(pred_prey_world: PredatorPreyWorld):
    ts = jnp.arange(EP_LEN)
    pred_prey_world = jit_reset_world(pred_prey_world)
    pred_prey_world, render_data = jit_scan_episode(pred_prey_world, ts)
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
        "wolf_fitness": render_data.content["wolf_fitness"]
    })
    return pred_prey_world, render_data

jit_run_episode = jax.jit(run_episode)

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

    # run the simulation for specified EP_LEN
    pred_prey_worlds, render_data = jax.vmap(jit_run_episode)(pred_prey_worlds)

    # get agents fitness; average the total accumulated fitness across all parallel worlds
    sheep_fitness = jnp.mean(pred_prey_worlds.sheep_set.agents.state.content["fitness"], axis=0).reshape(-1)
    wolf_fitness = jnp.mean(pred_prey_worlds.wolf_set.agents.state.content["fitness"], axis=0).reshape(-1)

    return sheep_fitness, wolf_fitness, pred_prey_worlds, render_data

jit_get_fitness = jax.jit(get_fitness)




def main():
    key, *pred_prey_world_keys = random.split(KEY, NUM_WORLDS+1)
    pred_prey_world_keys = jnp.array(pred_prey_world_keys)

    key, energy_key, eat_key = jax.random.split(key, 3)
    base_energy_rates = jax.random.uniform(energy_key, shape=(NUM_WORLDS,), minval=MIN_BASE_ENERGY, maxval=MAX_BASE_ENERGY)
    eat_rates = jax.random.uniform(eat_key, shape=(NUM_WORLDS,), minval=MIN_EAT_RATE, maxval=MAX_EAT_RATE)

    env_rates_matrix = jnp.stack([base_energy_rates, eat_rates], axis=1)
    pred_prey_worlds = jax.vmap(PredatorPreyWorld.create_world, in_axes=(None, 0, 0))(PP_WORLD_PARAMS, env_rates_matrix,
                                                                                      pred_prey_world_keys)

    # dummy_solution = jnp.zeros(NUM_ES_PARAMS)
    key, sheep_init_key, wolf_init_key = random.split(key, 3)

    initial_sheep_mean = jax.random.uniform(sheep_init_key, (NUM_ES_PARAMS,), minval=INIT_LOW, maxval=INIT_HIGH)
    initial_wolf_mean = jax.random.uniform(wolf_init_key, (NUM_ES_PARAMS,), minval=INIT_LOW, maxval=INIT_HIGH)

    strategy_sheep = CMA_ES(population_size=NUM_SHEEP, solution=initial_sheep_mean)
    es_params_sheep = strategy_sheep.default_params

    strategy_wolf = CMA_ES(population_size=NUM_WOLF, solution=initial_wolf_mean)
    es_params_wolf = strategy_wolf.default_params

    key, sheep_key, wolf_key = random.split(key, 3)
    sheep_state = strategy_sheep.init(key=sheep_key, mean=initial_sheep_mean, params=es_params_sheep)
    sheep_state_mean = sheep_state.mean
    #sheep_state_best = sheep_state.mean

    wolf_state = strategy_wolf.init(key=wolf_key, mean=initial_wolf_mean, params=es_params_wolf)
    wolf_state_mean = wolf_state.mean
    #wolf_state_best = wolf_state.mean

    mean_sheep_fitness_list = []
    mean_wolf_fitness_list = []

    sheep_param_list_mean = []
    sheep_param_list_best = []
    wolf_param_list_mean = []
    wolf_param_list_best = []

    sheep_xs_list, sheep_ys_list, sheep_angles_list, sheep_fitness_list, sheep_energy_list, sheep_transfer_list = [], [], [], [], [], []
    wolf_xs_list, wolf_ys_list, wolf_angles_list, wolf_fitness_list, wolf_energy_list, wolf_transfer_list = [], [], [], [], [], []

    print(f"Starting Hierarchical alternating evolution with {NUM_WORLDS} worlds, {NUM_GENERATIONS} generations")
    print(f"Sheep population: {NUM_SHEEP}, Wolf population: {NUM_WOLF}")

    # phase 1: train sheep (wolves are static)
    # for generation in range(NUM_GENERATIONS):
    #     print(f"Generation {generation}: Training sheep")
    #
    #     #key, sheep_ask_key, sheep_tell_key = jax.random.split(key, 3)
    #     key, sheep_ask_key, sheep_tell_key, wolf_rand_key = jax.random.split(key, 4)
    #
    #     x_sheep, sheep_state = strategy_sheep.ask(sheep_ask_key, sheep_state, es_params_sheep)
    #
    #     # tile the current mean of the wolves for the whole population -> ensures sheep train against a stable average predator
    #     #x_wolf_fixed = jnp.tile(wolf_state_mean, (NUM_WOLF, 1))
    #     x_wolf_random = jax.random.uniform(wolf_rand_key, (NUM_WOLF, NUM_ES_PARAMS),
    #                                    minval=INIT_LOW, maxval=INIT_HIGH)
    #
    #     # evaluate the fitness
    #     sheep_fitness, wolf_fitness, pred_prey_worlds, _ = jit_get_fitness(x_sheep, x_wolf_random, pred_prey_worlds)
    #
    #     # update the sheep strategy
    #     sheep_state, _ = strategy_sheep.tell(
    #         sheep_tell_key,
    #         x_sheep,
    #         jnp.array(-1.0 * sheep_fitness).reshape(-1),
    #         sheep_state,
    #         es_params_sheep
    #     )
    #
    #     sheep_state_mean = sheep_state.mean
    #     sheep_param_list_mean.append(sheep_state_mean)
    #     mean_sheep_fitness_list.append(jnp.mean(sheep_fitness))
    #
    #     if generation % 10 == 0:
    #         print(f'Sheep Gen {generation} - Mean: {jnp.mean(sheep_fitness):.2f}, Best: {jnp.max(sheep_fitness):.2f}')
    #

    # phase 2: train wolves (sheep are static)
    for generation in range(NUM_GENERATIONS):
        print(f"Generation {generation}: Training wolves")

        #key, wolf_ask_key, wolf_tell_key = jax.random.split(key, 3)
        key, wolf_ask_key, wolf_tell_key, sheep_rand_key = jax.random.split(key, 4)

        x_wolf, wolf_state = strategy_wolf.ask(wolf_ask_key, wolf_state, es_params_wolf)

        # x_sheep_fixed = jnp.tile(initial_sheep_mean, (NUM_SHEEP, 1))
        x_sheep_random = jax.random.uniform(sheep_rand_key, (NUM_SHEEP, NUM_ES_PARAMS),
                                        minval=INIT_LOW, maxval=INIT_HIGH)

        sheep_fitness, wolf_fitness, pred_prey_worlds, _ = jit_get_fitness(x_sheep_random, x_wolf, pred_prey_worlds)

        wolf_state, _ = strategy_wolf.tell(
            wolf_tell_key,
            x_wolf,
            jnp.array(-1.0 * wolf_fitness).reshape(-1),
            wolf_state,
            es_params_wolf
        )

        wolf_state_mean = wolf_state.mean
        wolf_param_list_mean.append(wolf_state_mean)
        mean_wolf_fitness_list.append(jnp.mean(wolf_fitness))
        if generation % 10 == 0:
          print(f'Wolf Gen {generation} - Mean: {jnp.mean(wolf_fitness):.2f}, Best: {jnp.max(wolf_fitness):.2f}')

        # save render data every generation
        # sheep_xs_list.append(render_data_all.content["sheep_xs"])
        # sheep_ys_list.append(render_data_all.content["sheep_ys"])
        # sheep_angles_list.append(render_data_all.content["sheep_angles"])
        # sheep_fitness_list.append(render_data_all.content["sheep_fitness"])
        # sheep_energy_list.append(render_data_all.content["sheep_energy"])

        # wolf_xs_list.append(render_data_all.content["wolf_xs"])
        # wolf_ys_list.append(render_data_all.content["wolf_ys"])
        # wolf_angles_list.append(render_data_all.content["wolf_angles"])
        # wolf_fitness_list.append(render_data_all.content["wolf_fitness"])
        # wolf_energy_list.append(render_data_all.content["wolf_energy"])

    
    # evaluation run
    # 1. prepare NUM_WORLDS different evaluation keys
    key, eval_master_key = jax.random.split(key)
    eval_keys = jax.random.split(eval_master_key, NUM_WORLDS)

    # 2. create NUM_WORLDS independent worlds (vmapped create_world)
    eval_worlds = jax.vmap(PredatorPreyWorld.create_world, in_axes=(None, 0, 0))(
        PP_WORLD_PARAMS,
        env_rates_matrix,
        eval_keys
    )

    # 3. inject the best sheep parameters into all agents in all worlds
    # We tile final_params for the population, then vmap the set_params over the worlds
    # comment this for training wolves:
    # final_sheep_params_tiled = jnp.tile(sheep_state.mean, (NUM_SHEEP, 1))
    #
    # key, eval_wolf_key = jax.random.split(key)
    # random_wolf_params = jax.random.uniform(eval_wolf_key, (NUM_ES_PARAMS,), minval=INIT_LOW, maxval=INIT_HIGH)
    # final_wolf_params_tiled = jnp.tile(random_wolf_params, (NUM_WOLF, 1))


    # uncomment this for training wolves:
    final_wolf_params_tiled = jnp.tile(wolf_state.mean, (NUM_WOLF, 1))

    key, eval_sheep_key = jax.random.split(key)
    random_sheep_params = jax.random.uniform(eval_sheep_key, (NUM_ES_PARAMS,), minval=INIT_LOW, maxval=INIT_HIGH)
    final_sheep_params_tiled = jnp.tile(random_sheep_params, (NUM_SHEEP, 1))

    def apply_params_to_world(world):
        # Set Sheep brains
        new_sheep = jit_set_CMAES_params(final_sheep_params_tiled, world.sheep_set.agents)
        # Set Wolf brains (crucial for JIT consistency)
        new_wolves = jit_set_CMAES_params(final_wolf_params_tiled, world.wolf_set.agents)
        
        return world.replace(
            sheep_set=world.sheep_set.replace(agents=new_sheep),
            wolf_set=world.wolf_set.replace(agents=new_wolves)
        )

    eval_worlds = jax.vmap(apply_params_to_world)(eval_worlds)

    ts_array = jnp.arange(EP_LEN)
    final_states, eval_render_data = jax.vmap(jit_scan_episode, in_axes=(0, None))(eval_worlds, ts_array)

    # save render data of eval run
    sheep_xs_list.append(eval_render_data.content["sheep_xs"])
    sheep_ys_list.append(eval_render_data.content["sheep_ys"])
    sheep_angles_list.append(eval_render_data.content["sheep_angles"])
    sheep_fitness_list.append(eval_render_data.content["sheep_fitness"])
    sheep_energy_list.append(eval_render_data.content["sheep_energy"])
    sheep_transfer_list.append(eval_render_data.content["patch_sheep_transfer"])


    wolf_xs_list.append(eval_render_data.content["wolf_xs"])
    wolf_ys_list.append(eval_render_data.content["wolf_ys"])
    wolf_angles_list.append(eval_render_data.content["wolf_angles"])
    wolf_fitness_list.append(eval_render_data.content["wolf_fitness"])
    wolf_energy_list.append(eval_render_data.content["wolf_energy"])
    wolf_transfer_list.append(eval_render_data.content["sheep_wolf_transfer"])


    # convert to arrays and save
    os.makedirs(DATA_PATH, exist_ok=True)

    jnp.save(DATA_PATH + 'mean_sheep_fitness_list.npy',jnp.array(mean_sheep_fitness_list))
    jnp.save(DATA_PATH + 'mean_wolf_fitness_list.npy', jnp.array(mean_wolf_fitness_list))
    jnp.save(DATA_PATH + 'final_key.npy', jnp.array(key))

    # save sheep render data
    jnp.save(DATA_PATH + 'rendering_sheep_xs.npy', jnp.array(sheep_xs_list))
    jnp.save(DATA_PATH + 'rendering_sheep_ys.npy', jnp.array(sheep_ys_list))
    jnp.save(DATA_PATH + 'rendering_sheep_angs.npy', jnp.array(sheep_angles_list))
    jnp.save(DATA_PATH + 'rendering_sheep_fitness.npy',  jnp.array(sheep_fitness_list))
    jnp.save(DATA_PATH + 'rendering_sheep_energy.npy', jnp.array(sheep_energy_list))
    jnp.save(DATA_PATH + 'rendering_sheep_transfer_rate.npy', jnp.array(sheep_transfer_list))

    # save wolf render data
    jnp.save(DATA_PATH + 'rendering_wolf_xs.npy', jnp.array(wolf_xs_list))
    jnp.save(DATA_PATH + 'rendering_wolf_ys.npy', jnp.array(wolf_ys_list))
    jnp.save(DATA_PATH + 'rendering_wolf_angs.npy', jnp.array(wolf_angles_list))
    jnp.save(DATA_PATH + 'rendering_wolf_fitness.npy', jnp.array(wolf_fitness_list))
    jnp.save(DATA_PATH + 'rendering_wolf_energy.npy', jnp.array(wolf_energy_list))
    jnp.save(DATA_PATH + 'rendering_wolf_energy_transfer.npy', jnp.array(wolf_transfer_list))

    # params
    jnp.save(DATA_PATH + 'sheep_param_list_mean.npy', jnp.array(sheep_param_list_mean))
    #jnp.save(DATA_PATH + 'sheep_param_list_best.npy', jnp.array(sheep_param_list_best))
    jnp.save(DATA_PATH + 'wolf_param_list_mean.npy', jnp.array(wolf_param_list_mean))
    #jnp.save(DATA_PATH + 'wolf_param_list_best.npy', jnp.array(wolf_param_list_best))

    print(f"Simulation completed. Data saved to {DATA_PATH}")



if __name__ == "__main__":
    main()

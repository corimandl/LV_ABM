

import sys
import os
import jax
import jax.numpy as jnp
from structs import *
from functions import *
from train_w import * # change this

TEST_RENDER_PATH = "./data/test_results_render_data/"
os.makedirs(TEST_RENDER_PATH, exist_ok=True)


def test_one_param_pair(sheep_ES_params, wolf_ES_params, world):
    """
    Tests a specific pair of sheep and wolf parameters (usually from the same generation).
    """
    # tile the single parameter vector across the entire population
    sheep_params_batch = jnp.tile(sheep_ES_params, (NUM_SHEEP, 1))
    wolf_params_batch = jnp.tile(wolf_ES_params, (NUM_WOLF, 1))

    # inject parameters into the agents
    es_sheep = jit_set_CMAES_params(sheep_params_batch, world.sheep_set.agents)
    es_wolf = jit_set_CMAES_params(wolf_params_batch, world.wolf_set.agents)

    # update the world sets
    world = world.replace(
        sheep_set=world.sheep_set.replace(agents=es_sheep),
        wolf_set=world.wolf_set.replace(agents=es_wolf)
    )

    final_world, rendering_data = jit_run_episode(world)

    # extract Fitness (net energy gain accumulated in state)
    sheep_fitness = final_world.sheep_set.agents.state.content['fitness'].reshape(-1)
    wolf_fitness = final_world.wolf_set.agents.state.content['fitness'].reshape(-1)

    stats = jnp.array([
        jnp.mean(sheep_fitness), jnp.min(sheep_fitness), jnp.max(sheep_fitness), # sheep mean/min/max fitness
        jnp.mean(wolf_fitness), jnp.min(wolf_fitness), jnp.max(wolf_fitness) # wolf mean/min/max fitness
    ])

    return rendering_data, stats

jit_test_one_param_pair = jax.jit(test_one_param_pair)


def test_all_params(all_sheep_params, all_wolf_params, key):
    # num_test_worlds = ES_Params.shape[0]
    key, subkey = jax.random.split(key)

    test_env_rates = jnp.array([0.5, 0.6])  # [base_energy_rate, eat_rate]

    world = PredatorPreyWorld.create_world(PP_WORLD_PARAMS, test_env_rates, subkey)

    render_data, fitness_stats = jax.vmap(jit_test_one_param_pair, in_axes=(0, 0, None))(all_sheep_params, all_wolf_params, world)

    # save results
    jnp.save(os.path.join(TEST_RENDER_PATH, "test_sheep_xs.npy"), render_data.content['sheep_xs'])
    jnp.save(os.path.join(TEST_RENDER_PATH, "test_sheep_ys.npy"), render_data.content['sheep_ys'])
    jnp.save(os.path.join(TEST_RENDER_PATH, "test_sheep_angs.npy"), render_data.content['sheep_angles'])
    jnp.save(os.path.join(TEST_RENDER_PATH, "test_wolf_xs.npy"), render_data.content['wolf_xs'])
    jnp.save(os.path.join(TEST_RENDER_PATH, "test_wolf_ys.npy"), render_data.content['wolf_ys'])
    jnp.save(os.path.join(TEST_RENDER_PATH, "test_wolf_angs.npy"), render_data.content['wolf_angles'])
    jnp.save(os.path.join(TEST_RENDER_PATH, "test_sheep_fitness.npy"), render_data.content['sheep_fitness']) # save fitness at every ts
    jnp.save(os.path.join(TEST_RENDER_PATH, "test_wolf_fitness.npy"), render_data.content['wolf_fitness'])
    jnp.save(os.path.join(TEST_RENDER_PATH, "test_fitness_stats.npy"), fitness_stats) # mean, min, and max fitness values reached at end of each episode
    print(f"Testing complete. Data saved to {TEST_RENDER_PATH}")


if __name__ == "__main__":
    # load the parameters saved by train_old.py
    sheep_params = jnp.load(DATA_PATH + "sheep_param_list_mean.npy")
    wolf_params = jnp.load(DATA_PATH + "wolf_param_list_mean.npy")

    key = jnp.load(DATA_PATH + "final_key.npy")

    test_all_params(sheep_params, wolf_params, key)
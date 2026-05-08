import argparse
import os

import jax
import jax.numpy as jnp
import numpy as np

import train_rand as sim 


def configure_world(max_sheep: int, max_wolf: int):
    sim.INIT_SHEEP = 150 #300 #120 #150 
    sim.INIT_WOLF = 80 #150 #50 #80
    sim.MAX_SHEEP = max_sheep
    sim.MAX_WOLF = max_wolf

    sim.WORLD_SIZE_X = 500.0
    sim.WORLD_SIZE_Y = 500.0
    sim.MAX_SPAWN_X = 250.0
    sim.MAX_SPAWN_Y = 250.0

    sim.PP_WORLD_PARAMS.content["sheep_params"]["num_agents"] = max_sheep
    sim.PP_WORLD_PARAMS.content["sheep_params"]["num_active_agents"] = sim.INIT_SHEEP
    sim.PP_WORLD_PARAMS.content["wolf_params"]["num_agents"] = max_wolf
    sim.PP_WORLD_PARAMS.content["wolf_params"]["num_active_agents"] = sim.INIT_WOLF


def make_long_episode_runner(eval_steps: int):
    @jax.jit
    def run_long_episode(
        pred_prey_world,
        base_energy_rate,
        eat_rate,
        metab_sheep,
        metab_wolf,
        repro_energy_threshold_sheep,
        death_energy_threshold_sheep,
        repro_time_threshold_sheep,
        death_time_threshold_sheep,
        repro_energy_threshold_wolf,
        death_energy_threshold_wolf,
        repro_time_threshold_wolf,
        death_time_threshold_wolf,
    ):
        sheep_set_agents = jax.vmap(sim.Sheep.reset_agent)(pred_prey_world.sheep_set.agents)
        wolf_set_agents = jax.vmap(sim.Wolf.reset_agent)(pred_prey_world.wolf_set.agents)

        pred_prey_world = pred_prey_world.replace(
            sheep_set=pred_prey_world.sheep_set.replace(agents=sheep_set_agents),
            wolf_set=pred_prey_world.wolf_set.replace(agents=wolf_set_agents),
        )

        def scan_step(world, t):
            new_world, render_data = sim.step_world(
                world,
                t,
                base_energy_rate,
                eat_rate,
                metab_sheep,
                metab_wolf,
                repro_energy_threshold_sheep,
                death_energy_threshold_sheep,
                repro_time_threshold_sheep,
                death_time_threshold_sheep,
                repro_energy_threshold_wolf,
                death_energy_threshold_wolf,
                repro_time_threshold_wolf,
                death_time_threshold_wolf,
            )
            return new_world, render_data

        ts = jnp.arange(eval_steps)
        final_world, render_data = jax.lax.scan(scan_step, pred_prey_world, ts)
        return final_world, render_data

    return run_long_episode


def load_best_raw_params(data_path: str):
    best_path = os.path.join(data_path, "phase3_best_raw_params.npy")
    fallback_path = os.path.join(data_path, "phase3_env_params_history.npy")

    if os.path.exists(best_path):
        return np.load(best_path)

    print("phase3_best_raw_params.npy not found; falling back to final CMA-ES mean.")
    env_params_history = np.load(fallback_path)
    return env_params_history[-1]


def resolve_sim_path(path: str):
    if os.path.isabs(path):
        return path
    return os.path.join(os.path.dirname(os.path.abspath(sim.__file__)), path)


def load_random_brain_params(data_path: str):
    sheep_candidates = (
        os.path.join(data_path, "phase3_sheep_brain_params.npy"),
        os.path.join(data_path, "phase3_checkpoint_sheep_brain_params.npy"),
    )
    wolf_candidates = (
        os.path.join(data_path, "phase3_wolf_brain_params.npy"),
        os.path.join(data_path, "phase3_checkpoint_wolf_brain_params.npy"),
    )

    sheep_path = next((path for path in sheep_candidates if os.path.exists(path)), None)
    wolf_path = next((path for path in wolf_candidates if os.path.exists(path)), None)

    if sheep_path is None:
        raise FileNotFoundError(
            f"Could not find saved random sheep brain params in {data_path}. "
            f"Expected one of: {', '.join(sheep_candidates)}"
        )
    if wolf_path is None:
        raise FileNotFoundError(
            f"Could not find saved random wolf brain params in {data_path}. "
            f"Expected one of: {', '.join(wolf_candidates)}"
        )

    print(f"Loading random sheep brain parameters from {sheep_path}")
    print(f"Loading random wolf brain parameters from {wolf_path}")
    sheep_params = jnp.array(np.load(sheep_path))
    wolf_params = jnp.array(np.load(wolf_path))
    return sheep_params, wolf_params


def generate_eval_random_brain_params(key):
    key, sheep_key, wolf_key = jax.random.split(key, 3)
    sheep_params = jax.random.uniform(
        sheep_key, (sim.MAX_SHEEP, sim.NUM_ES_PARAMS), minval=sim.INIT_LOW, maxval=sim.INIT_HIGH
    )
    wolf_params = jax.random.uniform(
        wolf_key, (sim.MAX_WOLF, sim.NUM_ES_PARAMS), minval=sim.INIT_LOW, maxval=sim.INIT_HIGH
    )
    return key, sheep_params, wolf_params


def expand_brain_params(brain_params, num_agents: int, label: str):
    if brain_params.ndim == 1:
        if brain_params.shape[0] != sim.NUM_ES_PARAMS:
            raise ValueError(
                f"Expected {label} brain vector of length {sim.NUM_ES_PARAMS}, got shape {brain_params.shape}"
            )
        return jnp.tile(brain_params, (num_agents, 1))

    if brain_params.ndim == 2:
        if brain_params.shape == (num_agents, sim.NUM_ES_PARAMS):
            return brain_params
        if brain_params.shape == (1, sim.NUM_ES_PARAMS):
            return jnp.tile(brain_params[0], (num_agents, 1))

    raise ValueError(
        f"Expected {label} brain params with shape ({sim.NUM_ES_PARAMS},) "
        f"or ({num_agents}, {sim.NUM_ES_PARAMS}), got {brain_params.shape}"
    )


def inject_brains(worlds, sheep_params_tiled, wolf_params_tiled):
    def inject_single_world(world):
        new_sheep = sim.jit_set_CMAES_params(sheep_params_tiled, world.sheep_set.agents)
        new_wolf = sim.jit_set_CMAES_params(wolf_params_tiled, world.wolf_set.agents)
        return world.replace(
            sheep_set=world.sheep_set.replace(agents=new_sheep),
            wolf_set=world.wolf_set.replace(agents=new_wolf),
        )

    return jax.vmap(inject_single_world)(worlds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default=sim.DATA_PATH)
    parser.add_argument("--eval-steps", type=int, default=14000)
    parser.add_argument("--num-worlds", type=int, default=32)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-sheep", type=int, default=400)
    parser.add_argument("--max-wolf", type=int, default=400)
    parser.add_argument(
        "--use-saved-random-brains",
        action="store_true",
        help="Debug option: evaluate with the random brains saved by train_rand.py instead of new random brains.",
    )
    args = parser.parse_args()

    data_path = args.data_path
    if not data_path.endswith("/"):
        data_path += "/"

    configure_world(args.max_sheep, args.max_wolf)

    print(f"Loading best environment parameters from {data_path}...")
    best_raw_params = load_best_raw_params(data_path)

    if best_raw_params.shape[0] != 12:
        raise ValueError(f"Expected 12 learned environment parameters, got shape {best_raw_params.shape}")

    constrained_params = sim.MIN_BOUNDS + (sim.MAX_BOUNDS - sim.MIN_BOUNDS) * jax.nn.sigmoid(best_raw_params)

    base_energy_rate = constrained_params[0]
    eat_rate = constrained_params[1]
    metab_sheep = constrained_params[2]
    metab_wolf = constrained_params[3]
    repro_energy_threshold_sheep = constrained_params[4] * sim.MAX_ENERGY
    death_energy_threshold_sheep = constrained_params[5] * sim.MAX_ENERGY
    repro_time_threshold_sheep = constrained_params[6] * sim.EP_LEN
    death_time_threshold_sheep = constrained_params[7] * sim.EP_LEN
    repro_energy_threshold_wolf = constrained_params[8] * sim.MAX_ENERGY
    death_energy_threshold_wolf = constrained_params[9] * sim.MAX_ENERGY
    repro_time_threshold_wolf = constrained_params[10] * sim.EP_LEN
    death_time_threshold_wolf = constrained_params[11] * sim.EP_LEN

    print("Actual parameters:")
    print(
        f"Grass:{constrained_params[0]:.4f}, Eat:{constrained_params[1]:.4f}, "
        f"Meta_S:{constrained_params[2]:.4f}, Meta_W:{constrained_params[3]:.4f}, "
        f"S_RepE:{constrained_params[4]:.4f}, S_DthE:{constrained_params[5]:.4f}, "
        f"S_RepT:{constrained_params[6] * sim.EP_LEN:.0f}, "
        f"S_DthT:{constrained_params[7] * sim.EP_LEN:.0f}, "
        f"W_RepE:{constrained_params[8]:.4f}, W_DthE:{constrained_params[9]:.4f}, "
        f"W_RepT:{constrained_params[10] * sim.EP_LEN:.0f}, "
        f"W_DthT:{constrained_params[11] * sim.EP_LEN:.0f}"
    )

    key = jax.random.PRNGKey(args.seed)
    if args.use_saved_random_brains:
        eval_sheep_params, eval_wolf_params = load_random_brain_params(data_path)
    else:
        print(f"Generating new random brain parameters for evaluation from seed {args.seed}")
        key, eval_sheep_params, eval_wolf_params = generate_eval_random_brain_params(key)

    eval_sheep_tiled = expand_brain_params(eval_sheep_params, sim.MAX_SHEEP, "sheep")
    eval_wolf_tiled = expand_brain_params(eval_wolf_params, sim.MAX_WOLF, "wolf")

    key, eval_world_key = jax.random.split(key)
    eval_world_keys = jax.random.split(eval_world_key, args.num_worlds)
    test_worlds = jax.vmap(sim.PredatorPreyWorld.create_world, in_axes=(None, 0))(
        sim.PP_WORLD_PARAMS,
        eval_world_keys,
    )
    test_worlds = inject_brains(test_worlds, eval_sheep_tiled, eval_wolf_tiled)

    print(f"\nRunning {args.num_worlds} evaluation worlds for {args.eval_steps} timesteps...")
    run_long_episode = make_long_episode_runner(args.eval_steps)
    run_many_episodes = jax.vmap(run_long_episode, in_axes=(0, None, None, None, None, None, None, None, None, None, None, None, None))
    _, final_render_data = run_many_episodes(
        test_worlds,
        base_energy_rate,
        eat_rate,
        metab_sheep,
        metab_wolf,
        repro_energy_threshold_sheep,
        death_energy_threshold_sheep,
        repro_time_threshold_sheep,
        death_time_threshold_sheep,
        repro_energy_threshold_wolf,
        death_energy_threshold_wolf,
        repro_time_threshold_wolf,
        death_time_threshold_wolf,
    )

    sheep_pop = final_render_data.content["sheep_count"]
    wolf_pop = final_render_data.content["wolf_count"]
    losses, loss_components = jax.vmap(
        lambda sheep_traj, wolf_traj: sim.calculate_phase_loss(
            sheep_traj,
            wolf_traj,
            sim.MAX_SHEEP,
            sim.MAX_WOLF,
        )
    )(sheep_pop, wolf_pop)

    component_means = jnp.mean(loss_components, axis=0)
    component_stds = jnp.std(loss_components, axis=0)

    os.makedirs(data_path, exist_ok=True)
    jnp.save(data_path + f"{args.eval_steps}_sheep_trajs.npy", sheep_pop)
    jnp.save(data_path + f"{args.eval_steps}_wolf_trajs.npy", wolf_pop)
    render_keys = (
        "sheep_xs",
        "sheep_ys",
        "sheep_angles",
        "wolf_xs",
        "wolf_ys",
        "wolf_angles",
    )
    for render_key in render_keys:
        values = final_render_data.content[render_key]
        jnp.save(data_path + f"{args.eval_steps}_{render_key}.npy", values)
        jnp.save(data_path + f"{args.eval_steps}_{render_key}_traj.npy", values[0])
    jnp.save(data_path + f"{args.eval_steps}_robustness_losses.npy", losses)
    jnp.save(data_path + f"{args.eval_steps}_robustness_loss_components.npy", loss_components)
    np.save(data_path + "phase3_loss_component_names.npy", np.array(sim.LOSS_COMPONENT_NAMES))

    jnp.save(data_path + f"{args.eval_steps}_sheep_traj.npy", sheep_pop[0])
    jnp.save(data_path + f"{args.eval_steps}_wolf_traj.npy", wolf_pop[0])

    component_summary = ", ".join(
        f"{name}:{mean:.1f}+/-{std:.1f}"
        for name, mean, std in zip(sim.LOSS_COMPONENT_NAMES, component_means, component_stds)
    )

    print(
        f"Robustness Loss -> mean:{jnp.mean(losses):.2f}, "
        f"std:{jnp.std(losses):.2f}, min:{jnp.min(losses):.2f}, max:{jnp.max(losses):.2f}"
    )
    print(f"Robustness Parts -> {component_summary}")
    print(f"Final sheep count mean: {jnp.mean(sheep_pop[:, -1]):.1f}")
    print(f"Final wolf count mean: {jnp.mean(wolf_pop[:, -1]):.1f}")
    print(f"Results saved to {data_path}")


if __name__ == "__main__":
    main()

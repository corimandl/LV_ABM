"""
Sensor functions for predator-prey simulation -> import this in simulation
Handles ray casting and collision detection for agent vision
"""

import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class Point:
    x: jnp.float32
    y: jnp.float32

@struct.dataclass
class Line:
    p1: Point
    p2: Point

@struct.dataclass
class Circle:
    center: Point
    radius: jnp.float32

@struct.dataclass
class Ray:
    origin: Point
    direction: Point # cos, sin
    length: jnp.float32


def generate_rays(agent, ray_span: jnp.float32, ray_length: jnp.float32, ray_resolution: int):
    """
    Generate rays for vision system given agent position and orientation.
    Args:
        - agent: The agent (Sheep or Wolf) with state containing x, y, ang
        - ray_span: Total angular span of vision (will be split: ±ray_span/2)
        - ray_length: How far the agent can see
        - ray_resolution: Number of rays to cast
    Returns:
        Array of Ray objects
    """
    x = agent.state.content["x"][0]
    y = agent.state.content["y"][0]
    angle = agent.state.content["ang"][0]

    ray_angles = jnp.linspace(angle - ray_span, angle + ray_span, ray_resolution)
    cos_ray_angles = jnp.cos(ray_angles)
    sin_ray_angles = jnp.sin(ray_angles)
    ray_directions = jax.vmap(Point)(cos_ray_angles, sin_ray_angles) # each point represents the direction vector (cos, sin) for a single ray

    ray_origin = Point(x,y)

    rays = jax.vmap(Ray, in_axes=(None, 0, None))(ray_origin, ray_directions, ray_length)
    return rays

jit_generate_rays = jax.jit(generate_rays)

def get_ray_agent_collision(ray, agent_x: jnp.float32, agent_y: jnp.float32, agent_radius: jnp.float32):
    """
    Ray casting algorithm to check for collision between a ray and an agent (represented as a circle)
    Adapted from https://www.youtube.com/watch?v=ebzlMOw79Yw&ab_channel=MagellanicMath
    Args:
        - ray: Ray, The ray to check for collision
        - agent_x: x-coordinate of agent center
        - agent_y: y-coordinate of agent center
        - agent_radius: radius of the circle
    Returns:
        The distance of the collision of the ray with the agent along the ray
    """
    # extract ray information
    agent_center = jnp.array([agent_x, agent_y])
    ray_origin = jnp.array([ray.origin.x, ray.origin.y])
    ray_direction = jnp.array([ray.direction.x, ray.direction.y])

    s = ray_origin - agent_center
    b = jnp.dot(s, ray_direction)
    c = jnp.dot(s, s) - agent_radius**2
    h = b**2 - c
    h = jax.lax.cond(h < 0, lambda _: -1.0, lambda _: jnp.sqrt(h), None) # no intersection
    t = jax.lax.cond(h >= 0, lambda _: -b - h, lambda _: ray.length, None) # ray intersects agent
    t = jax.lax.cond(t < 0, lambda _: ray.length, lambda _: t, None)  # intersection occurs behind the ray's origin; no valid collision
    return jnp.minimum(t, ray.length)

jit_get_ray_agent_collision = jax.jit(get_ray_agent_collision)


def get_sensor_data(agent, all_sheep, all_wolves, ray_span, sheep_ray_length, wolf_ray_length,
                    ray_resolution, sheep_agent_type, wolf_agent_type):
    """
    Get sensor data for a single agent (sheep or wolf) using ray casting
    Each ray has 2 channels:
        1. distance from other entity
        2. type indicator: is_sheep (1) or is_wolf (-1)

    Args:
        - agent: Single agent (Sheep or Wolf) to generate sensors for
        - all_sheep: All sheep in the environment
        - all_wolves: All wolves in the environment
        - ray_span, sheep_ray_length, wolf_ray_length, ray_resolution: sensor parameters
        - sheep_agent_type, wolf_agent_type: agent type identifiers
    Returns:
        Sensor data array of shape (RAY_RESOLUTION * 2,) - flattened ray observations
    """
    agent_xs = jnp.concatenate((all_sheep.state.content['x'].reshape(-1), all_wolves.state.content['x'].reshape(-1))) # combines the x-coordinates of all sheep and all wolves into a single 1D array
    agent_ys = jnp.concatenate((all_sheep.state.content['y'].reshape(-1), all_wolves.state.content['y'].reshape(-1)))
    agent_rads = jnp.concatenate((all_sheep.params.content['radius'].reshape(-1), all_wolves.params.content['radius'].reshape(-1)))
    #agent_energies = jnp.concatenate((all_sheep.state.content['energy'].reshape(-1), all_wolves.state.content['energy'].reshape(-1)))
    agent_types = jnp.concatenate((all_sheep.agent_type, all_wolves.agent_type)) # agent types: 1 for sheep, 2 for wolves
    agent_active = jnp.concatenate((all_sheep.active_state, all_wolves.active_state)) # active: 1 or inactive: 0

    # type sensor values: [empty, sheep, wolf]
    # index 0: nothing (0), index 1: sheep (1), index 2: wolf (-1)
    type_sensor_values = jnp.array([0.0, 1.0, -1.0])

    # determine ray parameters based on agent type
    ray_length = jax.lax.cond(agent.agent_type == sheep_agent_type, lambda _: sheep_ray_length, lambda _: wolf_ray_length,None)

    def for_each_ray(ray):
        # check collisions with all agents
        def check_collision(agent_x, agent_y, agent_rad, is_active):
            distance = jax.lax.cond(is_active,
                lambda _: jit_get_ray_agent_collision(ray, agent_x, agent_y, agent_rad), # get the actual hit distance
                lambda _: ray.length,  # return max distance if inactive
                None
            )
            return distance

        intercepts = jax.vmap(check_collision)(agent_xs, agent_ys, agent_rads, agent_active) # apply check_collision across all targets simultaneously for the current ray
        min_dist_indx = jnp.argmin(intercepts) # idx of the closest hit agent
        min_dist = intercepts[min_dist_indx] # distance to that agent

        sensed_type = jax.lax.cond(min_dist < ray.length,
            lambda _: (agent_types[min_dist_indx]), # retrieve the type of the closest hit agent
            lambda _: 0,  # 0 if no agent was hit
            None
        )

        # get one-hot encoded type
        sensed_type_value = type_sensor_values[sensed_type]

        # return: [distance, energy, type]
        return jnp.concatenate((
            jnp.array([min_dist]),
            jnp.array([sensed_type_value])  # [is_sheep/is_wolf]
        ))

    rays = generate_rays(agent, ray_span, ray_length, ray_resolution)

    sensor_data = jax.vmap(for_each_ray)(rays).reshape(-1)
    return sensor_data

jit_get_sensor_data = jax.jit(get_sensor_data)


SHEEP_RAY_MAX_LENGTH = 300.0
WOLF_RAY_MAX_LENGTH = 300.0
RAY_RESOLUTION = 13  # W&B update
RAY_SPAN = jnp.pi # W&B update

def get_all_agent_sensors(all_sheep, all_wolves, sheep_agent_type, wolf_agent_type):
    """
    Get sensor data for all agents
    Args:
        - all_sheep: All sheep agents
        - all_wolves: All wolf agents
        - sensor parameters
    Returns:
        sheep_sensors: Sensor data for all sheep, shape (num_sheep, RAY_RESOLUTION * 2)
        wolf_sensors: Sensor data for all wolves, shape (num_wolves, RAY_RESOLUTION * 2)
    """
    def get_sensors_for_one_agent(agent):
        return jax.lax.cond(agent.active_state,
            lambda _: get_sensor_data(agent, all_sheep, all_wolves, RAY_SPAN, SHEEP_RAY_MAX_LENGTH,
                                     WOLF_RAY_MAX_LENGTH, RAY_RESOLUTION, sheep_agent_type,
                                     wolf_agent_type),
            lambda _: jnp.zeros((RAY_RESOLUTION * 2,)),  # return zeros if inactive
            None
        )

    sheep_sensors = jax.vmap(get_sensors_for_one_agent)(all_sheep)
    wolf_sensors = jax.vmap(get_sensors_for_one_agent)(all_wolves)

    return sheep_sensors, wolf_sensors

jit_get_all_agent_sensors = jax.jit(get_all_agent_sensors) # treat the arguments as a static used only for compilation, not as a dynamic array


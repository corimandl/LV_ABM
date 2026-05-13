# Stabilising Agent-based Predation Model using Lotka-Volterra Equations
Honours BSc Thesis 2026 — Radboud University

Can a continuous predator-prey agent-based model be tuned to produce 
Lotka-Volterra-like population cycles?

This project models sheep and wolves as active agents in a 2D arena, 
equipped with ray-based sensors and CTRNN neural controllers. Agents gain 
and lose energy through foraging and predation, with death and reproduction 
driving population dynamics. The model is implemented in ABMax, a JAX-based 
ABM framework enabling parallelised simulation on hardware accelerators.

A loss function is defined over population trajectories to 
reward sustained oscillations, correct predator-prey phase lag, bounded 
populations, and long-term stability. Environmental and demographic parameters 
are optimised using CMA-ES against this objective. Agent controllers are 
trained separately in a two-phase evolutionary procedure before the 
full system is tuned toward Lotka-Volterra dynamics.

## Key components
- Continuous predator-prey ABM built in ABMax / JAX
- Ray-based occlusion-aware sensing + CTRNN controllers
- 7-component Lotka-Volterra loss function
- CMA-ES optimisation of both agent controllers and environmental parameters

## Simulations
**Random agents** — population dynamics with untrained controllers, 
after tuning environmental parameters toward Lotka-Volterra cycles.\
**Trained agents** — same setup with evolved CTRNN controllers, 
sheep forage from a sparser patch.

| Random agents | Trained agents |
|---|---|
| ![](./assets/rand_agents.gif) | ![](./assets/trained_agents.gif) |

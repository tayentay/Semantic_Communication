# Semantic Communication over SAGIN (Reference Implementation)

This repository contains a lightweight, runnable implementation of the alternating optimization framework described in **“Energy-Efficient Probabilistic Semantic Communication Over Space-Air-Ground Integrated Networks.”** It models probabilistic semantic compression (PSCom) with piecewise computation overhead and iteratively optimizes six coupled subproblems for a satellite–UAV–ground-terminal network.

## Contents
- `Energy-Efficient_Probabilistic_Semantic_Communication_Over_Space-Air-Ground_Integrated_Networks.pdf`: Reference paper.
- `semantic_comm/`: Python package with data models and optimizer.
- `config/default.yaml`: Example scenario and algorithm hyperparameters.
- `run.py`: Entry point to execute the solver.
- `train_drl.py`: PPO (vwxyzjn/ppo-implementation-details style) trainer over a Gymnasium wrapper of the optimizer.

## Quickstart
1) Install dependencies (Python 3.10+ recommended):
```bash
pip install -r requirements.txt
```
2) Run the example scenario:
```bash
python run.py --config config/default.yaml
```
The script prints per-iteration energy values and final decisions (semantic compression placement, compression ratios, computation/power/bandwidth allocations, UAV altitude/beamwidth/location).

To visualize the optimization history, enable plotting:
```bash
python run.py --config config/default.yaml --plot
# plot saved to outputs/pscom_history.png by default
```

## Configuration
Edit `config/default.yaml` or provide another YAML file. Key fields:
- `constants`: Physical constants (e.g., τ, κ, noise PSD, wavelength, beam gains).
- `satellite`, `uav`: Power/computation budgets and antenna limits.
- `distance_su`, `sat_bandwidth`, `latency_budget`: Link distance, satellite–UAV bandwidth, and per-GT latency bound.
- `ground_terminals`: Positions, data sizes, minimum compression ratios, and piecewise overhead parameters (slopes/intercepts/boundaries).
- `simulation`: Iteration counts, grid/beamwidth steps, and verbosity.

The defaults are illustrative; adjust to match the exact parameters of the target deployment or paper reproductions (e.g., Table II).

## DRL (PPO) training
- The environment `semantic_comm.envs.SemanticComEnv` exposes the optimizer as a Gymnasium task.
- Train with PPO adapted from [vwxyzjn/ppo-implementation-details](https://github.com/vwxyzjn/ppo-implementation-details):
  ```bash
  python train_drl.py --config config/default.yaml --total-timesteps 20000 --num-envs 2
  ```
- Rewards are negative energy with latency and coverage penalties; the trained policy checkpoint is saved under `runs/`.

## Notes
- The solver follows the paper’s subproblem structure but uses practical heuristics (e.g., midpoint segment selection, grid search for location, scaled power from the closed-form relation) to remain lightweight and easily configurable.
- Computation and communication latency constraints are enforced through iterative updates with subgradient tuning; tighten tolerances or increase iteration budgets for higher fidelity.
- Piecewise overheads are defined per-GT, enabling heterogeneous semantic graphs.

## License
This implementation is provided for academic experimentation aligned with the referenced paper. Review and adapt parameter choices before deployment in other contexts.

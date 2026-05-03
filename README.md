# Smart Home Appliance and Task Scheduling using Reinforcement Learning

This repository contains the code artefact for the MSc project:

**Smart Home Appliance and Task Scheduling using Reinforcement Learning**

The project models a smart home supplied by solar panels, battery storage, and external grid electricity. It compares reinforcement learning-based appliance scheduling methods against a rule-based baseline.

## Project Overview

The smart home environment includes:

- solar power generation with stochastic variation
- battery storage with fixed capacity
- external grid charging
- household appliances with fixed duration and deadline constraints
- electric vehicle charging
- AC/heater comfort operation
- evaluation of PPO, DQN, and rule-based scheduling

The aim is to minimise external grid electricity usage, maximise renewable utilisation, satisfy appliance deadlines, maintain comfort, and avoid battery depletion.

## Repository Contents

| File | Description |
|---|---|
| `smart_home_rl_scheduling.py` | Main Python implementation for the smart home environment, PPO/DQN setup, rule-based baseline, evaluation, CSV output, and figures |
| `smart_home_rl_scheduling.ipynb` | Notebook version of the main implementation |
| `supervisor_summary_table.csv` | Summary of model performance across 1000 test cases |
| `supervisor_completion_rates.csv` | Task and deadline completion rates |
| `supervisor_all_test_results.csv` | Per-test-case evaluation results |
| `figure_4_1_model_comparison.png` | Bar chart comparing PPO, DQN, and rule-based scheduling |
| `figure_4_2_ppo_episode.png` | Sample PPO episode showing battery, solar, load, and grid charging |

## Required Libraries

The main code is written in Python and uses:

```bash
numpy
pandas
matplotlib
gymnasium
stable-baselines3
torch
```

For Google Colab, install missing libraries using:

```bash
pip install gymnasium stable-baselines3
```

## How to Run

### Option 1: Google Colab

1. Open `smart_home_rl_scheduling.ipynb`.
2. Run all cells from top to bottom.
3. The notebook will generate result CSV files and figures.

### Option 2: Local Python

Clone the repository:

```bash
git clone https://github.com/abdullahfaiz10/smart-home-rl-scheduling.git
cd smart-home-rl-scheduling
```

Install dependencies:

```bash
pip install numpy pandas matplotlib gymnasium stable-baselines3 torch
```

Run the script:

```bash
python smart_home_rl_scheduling.py
```

## Outputs Produced

Running the code produces:

- `supervisor_summary_table.csv`
- `supervisor_completion_rates.csv`
- `supervisor_all_test_results.csv`
- `figure_4_1_model_comparison.png`
- `figure_4_2_ppo_episode.png`

## Accessibility

This repository is public and accessible to the project supervisor and markers.

## Note

The experiments use synthetic stochastic solar and appliance scheduling scenarios following the supervisor-defined project setup.

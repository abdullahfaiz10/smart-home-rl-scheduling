# Smart Home Appliance and Task Scheduling using Reinforcement Learning

This repository contains the code artefact for the MSc project:

**Smart Home Appliance and Task Scheduling using Reinforcement Learning**

The project models a smart home supplied by solar panels, battery storage, and external grid electricity. It compares reinforcement learning-based appliance scheduling methods against a rule-based baseline.

## Project Overview

The smart home environment includes:

- solar power generation with stochastic variation
- battery storage with fixed capacity
- external grid charging
- household appliances with fixed-duration and deadline constraints
- electric vehicle charging
- AC/heater comfort operation
- evaluation of PPO, DQN, and rule-based scheduling

The objective is to minimise external grid electricity usage, maximise renewable energy utilisation, satisfy appliance deadlines, maintain comfort, and avoid battery depletion.

## Repository Contents

| File | Description |
|---|---|
| `smart_home_rl_scheduling.py` | Main Python implementation for the smart home environment, PPO and DQN training, rule-based baseline, evaluation, CSV output, figure generation, and model saving |
| `smart_home_rl_scheduling.ipynb` | Notebook version of the main implementation, suitable for Google Colab |
| `supervisor_summary_table.csv` | Clean summary table of model performance across 1000 test cases |
| `supervisor_completion_rates.csv` | Task and deadline completion rates for PPO, DQN, and rule-based scheduling |
| `supervisor_all_test_results.csv` | Full per-test-case evaluation results for 3000 cases: 1000 PPO, 1000 DQN, and 1000 rule-based |
| `figure_4_1_model_comparison.png` | Bar chart comparing PPO, DQN, and rule-based scheduling across key evaluation metrics |
| `figure_4_2_ppo_episode.png` | Sample PPO episode showing battery level, solar generation, load demand, and grid charging |
| `ppo_smart_home_model.zip` | Saved trained PPO model |
| `dqn_smart_home_model.zip` | Saved trained DQN model |
| `.gitignore` | Python Git ignore file |

## Required Libraries

The code is written in Python and uses the following libraries:

```bash
numpy
pandas
matplotlib
gymnasium
stable-baselines3
torch
```

For Google Colab, install the required libraries using:

```bash
pip install gymnasium stable-baselines3 pandas matplotlib numpy
```

## How to Run

### Option 1: Google Colab

1. Open `smart_home_rl_scheduling.ipynb`.
2. Run all cells from top to bottom.
3. The notebook trains PPO and DQN, evaluates PPO, DQN, and the rule-based baseline, and generates output files.

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
- `ppo_smart_home_model.zip`
- `dqn_smart_home_model.zip`

## Experiment Setup

The experiment follows the supervisor-defined smart home scheduling setup:

- one-day scheduling horizon
- 30-minute decision intervals
- solar panel maximum power: 2 kW
- battery capacity: 30 kWh
- initial battery charge: 20 kWh
- external grid charging power: 2.5 kW
- stochastic solar coefficient with 5% resolution and minimum 10%
- household appliances: refrigerator, laundry, dishwasher, oven, AC/heater, and electric vehicle
- PPO, DQN, and rule-based baseline comparison
- evaluation on 1000 independent test cases per method

## Key Evaluation Metrics

The models are evaluated using:

- total energy used
- external grid energy purchased
- renewable utilisation ratio
- deadline satisfaction rate
- AC/heater operation duration
- battery depletion events
- unfinished EV charging energy
- individual task completion rates

## Results Summary

The main experimental results show that PPO achieved the best overall balance between cost reduction, renewable utilisation, and system stability. PPO purchased less external grid energy than both DQN and the rule-based scheduler, while maintaining high renewable utilisation and low battery depletion. DQN achieved full deadline satisfaction but used more grid energy than PPO. The rule-based scheduler achieved full task reliability and comfort but required substantially more grid electricity.

## Accessibility

This repository is public and accessible to the project supervisor and markers.

## Note

The experiments use synthetic stochastic solar and appliance scheduling scenarios following the supervisor-defined project setup. The saved result files and figures correspond to the outputs used in the MSc project report.

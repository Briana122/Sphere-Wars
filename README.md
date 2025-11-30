## Sphere-Wars
Sphere Wars: Reinforcement Learning for Territory Capture on a Hexasphere Board.
This project uses a custom Gymnasium environment.

## Installation
**Windows:**
- *python -m venv .venv* to create the virtual environment
- If using VS Code, use **Ctrl+Shift+P**, choose **Python: Select Interpreter** and then choose **Python 3.9.2 (.venv)**
- Otherwise if using the terminal, navigate to the project directory and use *.\venv\Scripts\activate*
- *pip install -r requirements.txt* to install all required dependencies

**Linux:**
- *python python3 -m venv .venv && source .venv/bin/activate* to create and activate virtual environment
- *python pip install -r requirements.txt* to install all required dependencies
- *python pip install -e .* to run the setup

## Usage
To run a simple game with the rendering,
*python -m scripts.play_game*


## Project Structure
```text
├── .gitattributes
├── .gitignore
├── .pre-commit-config.yaml
├── README.md
├── RL_model_plans
│   ├── actor_critic.txt
│   ├── deep_q_network.txt
│   └── mcts_with_policy_value_network.txt
├── gymnasium_env
│   ├── __init__.py
│   ├── agents
│   │   ├── __init__.py
│   │   ├── actor_critic
│   │   │   ├── ac_agent.py
│   │   │   ├── ac_final_model.pt
│   │   │   ├── ac_model.py
│   │   │   ├── checkpoints_final
│   │   │   │   ├── lr1e-4_ent0.01
│   │   │   │   │   ├── lr1e-4_ent0.01_ep12000.pt
│   │   │   │   │   ├── lr1e-4_ent0.01_ep16000.pt
│   │   │   │   │   ├── lr1e-4_ent0.01_ep20000.pt
│   │   │   │   │   ├── lr1e-4_ent0.01_ep24000.pt
│   │   │   │   │   ├── lr1e-4_ent0.01_ep28000.pt
│   │   │   │   │   ├── lr1e-4_ent0.01_ep32000.pt
│   │   │   │   │   ├── lr1e-4_ent0.01_ep36000.pt
│   │   │   │   │   └── ... (+9 more)
│   │   │   │   └── plots
│   │   │   │       ├── avg_return_comparison.png
│   │   │   │       └── winrate_comparison.png
│   │   │   └── old_models
│   │   │       ├── ac_final_model_old.pt
│   │   │       ├── lr1e-4_ent0.01
│   │   │       │   ├── finetune_greedy
│   │   │       │   │   ├── logs.txt
│   │   │       │   │   ├── lr1e-4_ent0.01_finetune_greedy_ep1000.pt
│   │   │       │   │   ├── lr1e-4_ent0.01_finetune_greedy_ep1500.pt
│   │   │       │   │   ├── lr1e-4_ent0.01_finetune_greedy_ep2000.pt
│   │   │       │   │   ├── lr1e-4_ent0.01_finetune_greedy_ep2500.pt
│   │   │       │   │   ├── lr1e-4_ent0.01_finetune_greedy_ep3000.pt
│   │   │       │   │   ├── lr1e-4_ent0.01_finetune_greedy_ep3500.pt
│   │   │       │   │   └── ... (+10 more)
│   │   │       │   ├── finetune_greedy_2000_more
│   │   │       │   │   ├── logs.txt
│   │   │       │   │   ├── lr1e-4_ent0.01_finetune_greedy_2000_ep1000.pt
│   │   │       │   │   ├── lr1e-4_ent0.01_finetune_greedy_2000_ep1200.pt
│   │   │       │   │   ├── lr1e-4_ent0.01_finetune_greedy_2000_ep1400.pt
│   │   │       │   │   ├── lr1e-4_ent0.01_finetune_greedy_2000_ep1600.pt
│   │   │       │   │   ├── lr1e-4_ent0.01_finetune_greedy_2000_ep1800.pt
│   │   │       │   │   ├── lr1e-4_ent0.01_finetune_greedy_2000_ep200.pt
│   │   │       │   │   └── ... (+7 more)
│   │   │       │   ├── lr1e-4_ent0.01_ep10000.pt
│   │   │       │   ├── lr1e-4_ent0.01_ep12000.pt
│   │   │       │   ├── lr1e-4_ent0.01_ep14000.pt
│   │   │       │   ├── lr1e-4_ent0.01_ep16000.pt
│   │   │       │   ├── lr1e-4_ent0.01_ep18000.pt
│   │   │       │   └── ... (+11 more)
│   │   │       └── plots
│   │   │           ├── avg_return_comparison.png
│   │   │           ├── avg_return_comparison_single.png
│   │   │           ├── winrate_comparison.png
│   │   │           └── winrate_comparison_single.png
│   │   ├── base_agent.py
│   │   ├── dqn
│   │   │   ├── __init__.py
│   │   │   ├── checkpoints
│   │   │   │   ├── dqn_checkpoint_ep1000.pt
│   │   │   │   ├── dqn_checkpoint_ep10000.pt
│   │   │   │   ├── dqn_checkpoint_ep10500.pt
│   │   │   │   ├── dqn_checkpoint_ep11000.pt
│   │   │   │   ├── dqn_checkpoint_ep11500.pt
│   │   │   │   ├── dqn_checkpoint_ep12000.pt
│   │   │   │   ├── dqn_checkpoint_ep12500.pt
│   │   │   │   └── ... (+73 more)
│   │   │   ├── dqn_agent.py
│   │   │   ├── dqn_final_model.pt
│   │   │   ├── dqn_model.py
│   │   │   ├── replay_buffer.py
│   │   │   └── utils.py
│   │   ├── dyna_q_plus
│   │   │   ├── __init__.py
│   │   │   ├── agent.py
│   │   │   ├── dyna_gym_agent.py
│   │   │   └── model.py
│   │   └── random_agent.py
│   ├── envs
│   │   ├── __init__.py
│   │   └── game_env.py
│   ├── game
│   │   ├── Game.py
│   │   ├── Piece.py
│   │   ├── Player.py
│   │   ├── PlayerUI.py
│   │   ├── Tile.py
│   │   ├── __init__.py
│   │   └── visual_game.py
│   └── utils
│       ├── __init__.py
│       ├── action_utils.py
│       ├── constants.py
│       ├── evaluation.py
│       ├── evaluation_ac_vs_random.py
│       ├── evaluation_dqn_vs_random.py
│       └── game_board.py
├── progress_reports
│   ├── Progress_Report_Nov15.txt
│   ├── Progress_Report_Oct15.txt
│   ├── Progress_Report_Oct30.txt
│   └── Progress_report_Nov30.txt
├── pyproject.toml
├── requirements.txt
├── scripts
│   └── play_game.py
└── training
    ├── fine_tune_actor_critic.py
    ├── train_actor_critic.py
    ├── train_dqn.py
    └── train_dyna_q_plus.py
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
├── README.md
├── gymnasium_env
│   ├── __init__.py
│   ├── agents
│   │   ├── dqn
│   │   │   ├── dqn_agent.py
│   │   │   ├── dqn_model.py
│   │   │   └── replay_buffer.py
│   │   └── base_agent.py
│   ├── envs
│   │   ├── __init__.py
│   │   ├── game_env.py
│   ├── game
│   │   ├── Game.py
│   │   ├── Piece.py
│   │   ├── Player.py
│   │   ├── PlayerUI.py
│   │   ├── Tile.py
│   │   ├── __init__.py
│   │   └── visual_game.py
│   ├── utils
│   │   ├── __init__.py
│   │   ├── evaluation.py
│   │   └── training_game.py
├── progress_reports
│   ├── Progress_Report_Nov15.txt
│   ├── Progress_Report_Oct15.txt
│   └── Progress_Report_Oct30.txt
├── RL_model_plans
│   ├── deep_q_network.txt
│   ├── mcts_with_policy_value_network.txt
│   └── q-learning_with_function_approximation.txt
├── pyproject.toml
├── requirements.txt
├── scripts
│   └── play_game.py
└── tests
    ├── test_envs.py
    └── test_game.py
## Sphere-Wars
Sphere Wars: Reinforcement Learning for Territory Capture on a Hexasphere Board.
This project uses a custom Gymnasium environment.

## Installation
**Windows:**
- *python -m venv .venv* to create the virtual enviornment
- If using VS Code, use **Ctrl+Shift+P**, choose **Python: Select Interpreter** and then choose **Python 3.9.2 (.venv)**
- Otherwise if using the terminal, navigate to the project directory and use *.\venv\Scripts\activate*
- *python pip install -r requirements.txt* to install all required dependancies

**Mac OS:**
- *python python3 -m venv .venv source .venv/bin/activate* to create and activate virtual environment
- *python pip install -r requirements.txt* to install all required dependancies

## Usage
To run a simple game with the rendering,
'''python
python3 -m scripts.play_game
'''

## Project Structure
```text
├── README.md
├── gymnasium_env
│   ├── __init__.py
│   ├── envs
│   │   ├── __init__.py
│   │   ├── game_env.py
│   │   └── grid_world.py
│   ├── game
│   │   ├── Game.py
│   │   ├── Piece.py
│   │   ├── Tile.py
│   │   ├── __init__.py
│   │   └── visual_game.py
│   ├── utils
│   │   ├── __init__.py
│   │   └── training_game.py
│   └── wrappers
│       ├── __init__.py
│       ├── clip_reward.py
│       ├── discrete_actions.py
│       ├── reacher_weighted_reward.py
│       └── relative_position.py
├── progress_reports
│   └── Progress_Report_Oct15.txt
├── pyproject.toml
├── requirements.txt
├── scripts
│   └── play_game.py
└── tests
    ├── test_envs.py
    └── test_game.py
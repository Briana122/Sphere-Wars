## Sphere-Wars
Sphere Wars: Reinforcement Learning for Territory Capture on a Hexasphere Board.
This project uses a custom Gymnasium environment.

## Installation
First, create a virtual environment:
'''python
python3 -m venv .venv
source .venv/bin/activate
'''

Then install the required dependencies:
'''python
pip install -r requirements.txt
'''

## Usage
To run a simple game with the rendering,
'''python
python3 -m scripts.play_game
'''

## Project Structure
.
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
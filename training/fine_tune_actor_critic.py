# fine_tune_actor_critic.py

import os

from gymnasium_env.utils.constants import MAX_STEPS, SUBDIV
from training.train_actor_critic import train_stage

# Path to the pre-trained model
BASE_MODEL_DIR = r"checkpoints\lr1e-4_ent0.01"
BASE_MODEL_PATH = os.path.join(
    BASE_MODEL_DIR,
    "finetune_greedy",
    "lr1e-4_ent0.01_finetune_greedy_final.pt",
)

# Fine-tuning configuration
FINE_TUNE_EPISODES = 2000

SAVE_DIR   = os.path.join(BASE_MODEL_DIR, "finetune_greedy_2000")
SAVE_PREFIX = "lr1e-4_ent0.01_finetune_greedy_2000"

LR = 1e-4      # same as before
GAMMA = 0.99
VALUE_COEF = 0.5

# We pass entropy_coef=0.0 so inside train_stage becomes: initial_entropy = 0.0 and final_entropy = 1e-4
# Entropy is essentially off (very tiny near the end).
ENTROPY_COEF = 0.0

def main():
    if not os.path.exists(BASE_MODEL_PATH):
        raise FileNotFoundError(f"Base model not found: {BASE_MODEL_PATH}")

    print("=== Fine-tuning Actor-Critic ===")
    print("Starting from:", BASE_MODEL_PATH)
    print("Saving to dir:", SAVE_DIR)
    print("Episodes:", FINE_TUNE_EPISODES)

    history, final_model_path = train_stage(
        subdiv=SUBDIV,
        num_episodes=FINE_TUNE_EPISODES,
        max_steps_per_episode=MAX_STEPS,
        lr=LR,
        gamma=GAMMA,
        value_coef=VALUE_COEF,
        entropy_coef=ENTROPY_COEF,
        save_dir=SAVE_DIR,
        save_prefix=SAVE_PREFIX,
        load_model_path=BASE_MODEL_PATH,  # continue from pre-trained model
        snapshot_interval=max(1000, FINE_TUNE_EPISODES // 5),
    )

    print("\nFine-tuning done.")
    print("Fine-tuned model saved at:", final_model_path)


if __name__ == "__main__":
    main()

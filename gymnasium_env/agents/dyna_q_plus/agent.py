# DYNA Q+ AGENT
# Overview:
# -------------------------------
# - Q[s][a] stores the Q-value of taking action a in state s.
# - Model[(s,a)] = (r, s_next, done) stores the one-step tabular model.
# - For each real step you take in the environment, do k "imagined" steps from the model.
#   These imagined rewards get an exploration bonus that grows with how long it’s been since (s,a) was last tried.
#
# Real Step
# --------------------------------------------------------
# 1) Choose action with ε-greedy over LEGAL actions:
#       a = agent.select_action(obs, legal_mask)
# 2) Step the environment (you do this outside the agent):
# 3) Let the agent learn from that transition:
#       agent.step(
#           s=obs, a=a, r=r, s_next=obs2, done=done,
#           next_legal_mask=info2.get("legal_mask")   
#       )
#   3.5) Imagined Updates inside agent.step():
#       - Sample a previously seen (ŝ, â) from model.
#       - Look up (r̂, ŝ′, donê) = Model[(ŝ, â)]
#       - Compute τ = t_now − last_seen[(ŝ, â)]
#       - Do The Dyna-Q+ bonus:
#             r_plus = r̂ + c * f(τ)
#       - Do Q-learning update with r_plus:
#             target_hat = r_plus                      if donê
#                          r_plus + γ * max_a Q(ŝ′,a)  otherwise
#             Q(ŝ,â) ← Q(ŝ,â) + α * (target_hat − Q(ŝ,â))
#
# Notes:
# -----------------------------
# - Call once at episode start:
#       agent.start_episode()
#
# - Call once at episode end (optional):
#       agent.end_episode()

from typing import Any, Optional, Dict
import math
import numpy as np

from .model import TabularModel

# -------------------- Helper Functions --------------------
def default_state_encoder(state: Any) -> Any:
    '''
    Make observations hashable for tabular Q/model tables.
    If 'state' is a NumPy vector, store its bytes; otherwise pass through.
    '''
    if isinstance(state, np.ndarray):
        return state.tobytes()
    return state


class DynaQPlusAgent:
    '''
    Parameters
    ----------
    action_dim (int): Number of discrete actions.
    state_encoder : Callable
        Maps the raw observation into a hashable key for the tables.
    alpha : float
        Step size for Q-learning updates.
    gamma : float
        Discount factor.
    epsilon : float
        ε for ε-greedy behavior.
    plan_n : int
        Planning updates per real step (e.g., 5..50).
    bonus_c : float
        Exploration bonus scale c in r_plus = r_hat + c * f(τ).
    bonus_mode : str
        'sqrt' (default) or 'linear' for f(τ).
    seed : int
        For random selection.
    '''

    # -------------------- Initialization --------------------
    def __init__(
        self,
        action_dim: int,
        state_encoder = default_state_encoder,
        *,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.9,
        epsilon_min: float = 0.1,
        epsilon_decay: float = 0.999,
        plan_n: int = 20,
        bonus_c: float = 0.01,
        bonus_mode: str = "sqrt",
        seed: int = 0,
    ):
        # Initialize parameters
        self.action_dim = int(action_dim)
        self.encode = state_encoder
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.plan_n = int(plan_n)
        self.bonus_c = float(bonus_c)
        self.bonus_mode = str(bonus_mode)
        self.rng = np.random.default_rng(seed)

        # Create Q Table and Model for Learning
        self.Q: Dict[Any, np.ndarray] = {}
        self.model = TabularModel()

        # To track time for Dyna-Q+ bonuse
        self.t_now = 0

    # -------------------- Utility Functions --------------------
    def _q_row(self, s_key: Any) -> np.ndarray:
        '''Return the Q-values row for s_key. If s_key isnt in Q, create new zero row'''
        row = self.Q.get(s_key)
        if row is None:
            row = np.zeros(self.action_dim, dtype=np.float32)
            self.Q[s_key] = row
        return row

    def _epsilon_greedy(self, s_key: Any, legal_mask: Optional[np.ndarray]) -> int:
        '''Select action via ε-greedy from Q(s_key)'''
        # If there aren't any legal actions, assume all actions are legal
        # Otherwise, get indices of legal actions
        # We won't need this, but it is here just in case
        if legal_mask is None:
            legal_idx = np.arange(self.action_dim)
        else:
            legal_idx = np.flatnonzero(legal_mask.astype(bool))

        # If our random number is less than epsilon, Explore
        # Otherwise, Exploit and choose the best action from Q
        if self.rng.random() < self.epsilon:
            return int(self.rng.choice(legal_idx))
        q = self._q_row(s_key)
        return int(legal_idx[np.argmax(q[legal_idx])])

    def _bonus(self, tau: Optional[int]) -> float:
        '''Compute r_bonus = c * f(τ). If time since last tried (τ) is None (never seen), be optimistic.'''
        if tau is None: tau = 10  # Optimism
        # Compute f(τ)
        # Will be either linear or sqrt based on bonus_mode
        if self.bonus_mode == "linear": f = float(tau)
        else: f = math.sqrt(float(tau))
        return self.bonus_c * f

    # -------------------- public API --------------------
    def start_episode(self) -> None:
        '''Start a new episode. We don't need to do anything here but useful for consistency.'''
        pass

    def select_action(self, state: Any, legal_mask: Optional[np.ndarray]) -> int:
        '''Select action for given state from Q via ε-greedy'''
        s_key = self.encode(state)
        return self._epsilon_greedy(s_key, legal_mask)

    def step(
        self,
        s: Any,
        a: int,
        r: float,
        s_next: Any,
        done: bool,
        next_legal_mask: Optional[np.ndarray] = None,  
    ) -> None:
        '''
        Perform one *real* Dyna-Q+ iteration:
          (b) Take A, observe R, S' [Done outside this function]
          (c) Real Q-learning update on (S,A,R,S')
          (d) Model(S,A) <- (R, S'); last_seen[(S,A)] <- t_now
          (e) Do 'plan_n' planning updates with exploration bonus
        '''
        # Increment time step
        self.t_now += 1

        # Encode states to keys so you can store in dictionaries
        s_key = self.encode(s)
        s_next_key = self.encode(s_next)

        # Fetch the Q-rows for current and next states (creates if missing)
        q_s = self._q_row(s_key)
        q_s_next = self._q_row(s_next_key)

        # If the episode has ended, the target is the immideate reward
        if done:
            target = r
        # Otherwise, make sure that we're considering a legal action
        # Then use the Bellman Optimality Equation to compute target
        else:
            if next_legal_mask is not None and next_legal_mask.any():
                legal_idx = np.flatnonzero(next_legal_mask.astype(bool))    # Get indices of legal actions
                target = r + self.gamma * float(np.max(q_s_next[legal_idx]))
            else:
                # Incase there are no legal actions, consider all actions legal
                print("Warning: No legal actions provided; considering all actions legal.")
                target = r + self.gamma * float(np.max(q_s_next))

        # Store TD Error and update Q(S,A)   
        td = target - float(q_s[a])
        q_s[a] += self.alpha * td

        # Update Our Model
        self.model.update(s_key, int(a), float(r), s_next_key, bool(done), t_now=self.t_now)

        # Gain Imagined Experience via Planning with Exploration Bonus
        for _ in range(self.plan_n):
            pair = self.model.sample_pair()
            if pair is None:
                break
            sh_key, ah = pair
            entry = self.model.get(sh_key, ah)
            if entry is None:
                continue
            r_hat, sh_next_key, done_hat = entry

            tau = self.model.time_since_last_seen(sh_key, ah, self.t_now)
            r_plus = r_hat + self._bonus(tau)

            q_sh = self._q_row(sh_key)
            q_sh_next = self._q_row(sh_next_key)

            if done_hat:
                target_hat = r_plus
            else:
                target_hat = r_plus + self.gamma * float(np.max(q_sh_next))

            td_hat = target_hat - float(q_sh[ah])
            q_sh[ah] += self.alpha * td_hat

    def end_episode(self) -> None:
        '''Decay epsilon after each episode'''
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # -------------------- Saving/Loading --------------------
    def save(self, path_npz: str) -> None:
        keys = list(self.Q.keys())
        mats = (
            np.stack([self.Q[k] for k in keys], axis=0)
            if keys
            else np.zeros((0, self.action_dim), dtype=np.float32)
        )

        # Store keys as an object array
        keys_arr = np.empty(len(keys), dtype=object)
        keys_arr[:] = keys

        np.savez(
            path_npz,
            keys=keys_arr,
            mats=mats,
            action_dim=self.action_dim,
            alpha=self.alpha,
            gamma=self.gamma,
            epsilon=self.epsilon,
            plan_n=self.plan_n,
            bonus_c=self.bonus_c,
            bonus_mode=self.bonus_mode,
            t_now=self.t_now,
        )


    def load(self, filename):
        data = np.load(filename, allow_pickle=True)
        raw_keys = data["keys"]
        mats = data["mats"]

        self.Q.clear()

        for key, row in zip(raw_keys, mats):
            # If we ever load an old checkpoint that stored repr() strings,
            # fall back to eval; otherwise just use the key directly.
            if isinstance(key, (bytes, tuple, int, float)):
                real_key = key
            else:
                # old-format compatibility (repr-ed strings)
                txt = key.decode("utf-8") if isinstance(key, (bytes, np.bytes_)) else str(key)
                try:
                    real_key = eval(txt)
                except Exception:
                    real_key = txt

            self.Q[real_key] = row.astype(np.float32, copy=True)


    def save_model(self, filename: str) -> None:
        self.save(filename)

    def load_model(self, filename: str) -> None:
        self.load(filename)
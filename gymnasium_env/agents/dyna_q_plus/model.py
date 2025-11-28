from typing import Dict, Tuple, Any, Optional
import random


class TabularModel:
    '''
    Dyna-Q+ model lecture pseudocode:
      - Model(S, A) <- (R, S')   (deterministic; overwrite ok)
      - Track last_seen[(S,A)] = t_now  (for bonus on stale pairs)
      - sample_pair(): uniformly sample a previously seen (S, A)
    '''

    def __init__(self):
        self.transitions: Dict[Tuple[Any, int], Tuple[float, Any, bool]] = {}
        self.last_seen: Dict[Tuple[Any, int], int] = {}
        self._keys: list[Tuple[Any, int]] = []

    def update(self, s_key: Any, a: int, r: float, s_next_key: Any, done: bool, t_now: int) -> None:
        key = (s_key, a)
        self.transitions[key] = (float(r), s_next_key, bool(done))
        self.last_seen[key] = int(t_now)
        if key not in self._keys:
            self._keys.append(key)

    def sample_pair(self) -> Optional[Tuple[Any, int]]:
        if not self._keys:
            return None
        return random.choice(self._keys)

    def get(self, s_key: Any, a: int) -> Optional[Tuple[float, Any, bool]]:
        return self.transitions.get((s_key, a))

    def time_since_last_seen(self, s_key: Any, a: int, t_now: int) -> Optional[int]:
        ts = self.last_seen.get((s_key, a))
        if ts is None:
            return None
        return max(0, int(t_now) - int(ts))
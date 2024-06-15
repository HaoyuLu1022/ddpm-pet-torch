from collections import deque
import numpy as np

class PerPromptStatTracker:
    def __init__(self, buffer_size, min_count):
        self.buffer_size = buffer_size
        self.min_count = min_count
        self.stats = {}

    def update(self, prompts, rewards):
        # unique = np.unique(prompts)
        unique = np.split(prompts, prompts.shape[0], axis=0)
        advantages = np.empty_like(rewards)
        for prompt in unique:
            prompt_rewards = rewards[np.where(np.all(prompts==prompt, axis=(1, 2, 3)))]
            prompt_key = prompt.tobytes()
            if prompt_key not in self.stats:
                self.stats[prompt_key] = deque(maxlen=self.buffer_size)
            self.stats[prompt_key].extend(prompt_rewards)

            if len(self.stats[prompt_key]) < self.min_count:
                mean = np.mean(rewards)
                std = np.std(rewards) + 1e-6
            else:
                mean = np.mean(self.stats[prompt_key])
                std = np.std(self.stats[prompt_key]) + 1e-6
            advantages[np.where(np.all(prompts==prompt, axis=(1, 2, 3)))] = (prompt_rewards - mean) / std

        return advantages
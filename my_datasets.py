import numpy as np
from typing import Tuple

class SinNoise(np.ndarray):
    def __new__(
        cls,
        n_samples: int,
        std: float = 0.2,
        amp: float = 1.,
        pi_per_sample: int = 100,
    ) -> None:
        y = np.sin((np.pi / pi_per_sample) * np.arange(n_samples))
        noises = np.random.normal(scale=std, size=n_samples)
        return (amp * (y + noises)).view(cls)

    def split(self, n_features: int) -> Tuple[np.array, np.array]:
        data = np.array([self[i:i + n_features] \
            for i in range(len(self) - n_features)])
        target = self[n_features:]
        return data, np.array(target)
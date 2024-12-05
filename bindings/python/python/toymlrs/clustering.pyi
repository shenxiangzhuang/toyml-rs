from typing import Optional


class Kmeans:
    def __init__(self, k: int, max_iter: int,
                 centroids_init_method: str = "random",
                 distance_metric: str = "euclidean",
                 random_seed: Optional[int] = None,
                 ) -> None: ...
    def fit(self, point_values: list[list[float]]) -> None: ...
    def fit_predict(self, point_values: list[list[float]]) -> list[int]: ...
    @property
    def labels(self) -> list[int]: ...


__all__ = [
    "Kmeans",
]

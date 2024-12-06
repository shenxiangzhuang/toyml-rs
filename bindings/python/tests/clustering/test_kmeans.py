import pytest

from toymlrs.clustering import Kmeans


class TestKMeansSimple:
    """
    Test the k-means algorithm completely.
    """

    @pytest.mark.parametrize("k,max_iter,centroids_init_method,distance_metric",
                             [
                                 (2, 10, "random", "euclidean"),
                                 (5, 100, "kmeans++", "euclidean"),
                             ],
                             )
    def test_repr(self,
                  k: int,
                  max_iter: int,
                  centroids_init_method: str,
                  distance_metric: str,
                  ) -> None:
        sut = Kmeans(k, max_iter, centroids_init_method, distance_metric)

        assert repr(sut) == str(sut)
        assert repr(
            sut) == f"Kmeans({k=}, {max_iter=}, centroids_init_method={centroids_init_method}, distance_metric={distance_metric})"

    # @pytest.mark.parametrize("centroids_init_method", ["random", "kmeans++"])
    # def test_fit(
    #         self,
    #         centroids_init_method: Literal["random", "kmeans++"],
    #         simple_dataset: list[list[float]],
    # ) -> None:
    #     k = 2
    #     max_iter = 10
    #     dataset = simple_dataset
    #
    #     kmeans = Kmeans(k, max_iter, 1e-5, centroids_init_method)
    #     assert len(kmeans.clusters_) == 0
    #     assert len(kmeans.centroids_) == 0
    #     kmeans.fit(dataset)
    #     assert len(kmeans.clusters_) == 2
    #     assert len(kmeans.centroids_) == 2
    #
    #     # check clusters
    #     first_sample_cluster, last_sample_cluster = kmeans.predict(dataset[0]), kmeans.predict(dataset[-1])
    #     assert kmeans.clusters_[first_sample_cluster] == [0, 1, 2]
    #     assert kmeans.clusters_[last_sample_cluster] == [3, 4, 5]
    #
    #     # check prediction
    #     x_sample = [0.0] * len(dataset[0])
    #     assert kmeans.predict(x_sample) == first_sample_cluster

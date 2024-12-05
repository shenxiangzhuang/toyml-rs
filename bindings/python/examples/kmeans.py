from toymlrs.clustering import Kmeans


def run():
    xs = [[0.0], [0.0], [10.0], [10.0]]
    kmeans = Kmeans(k=2, max_iter=100, centroids_init_method="kmeans++", distance_metric="euclidean", random_seed=42)
    print(kmeans.fit_predict(xs))


if __name__ == '__main__':
    run()

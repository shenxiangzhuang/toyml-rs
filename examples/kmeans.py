from toymlrs import KmeansRust


def run():
    xs = [[0], [0], [10], [10]]
    kmeans = KmeansRust(k=2, max_iter=100)
    print(kmeans.fit_predict(xs))


if __name__ == '__main__':
    run()

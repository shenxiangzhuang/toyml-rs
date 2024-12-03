from toymlrs import KmeansRust


def run():
    xs = [[0], [0], [10], [10]]
    kmeans = KmeansRust(k=2)
    kmeans.fit(xs)
    print(kmeans.labels)


if __name__ == '__main__':
    run()
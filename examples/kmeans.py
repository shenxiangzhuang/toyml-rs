from toymlrs import Kmeans


def run():
    xs = [[1], [1], [10], [10]]
    kmeans = Kmeans(k=2)
    kmeans.fit(xs)
    print(kmeans.labels)


if __name__ == '__main__':
    run()
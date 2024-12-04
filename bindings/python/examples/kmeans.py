from toymlrs import Kmeans


def run():
    xs = [[0.0], [0.0], [10.0], [10.0]]
    kmeans = Kmeans(k=2, max_iter=100)
    print(kmeans.fit_predict(xs))


if __name__ == '__main__':
    run()

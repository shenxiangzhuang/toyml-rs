import init, {greet, Kmeans} from "toymlrs";

async function run() {
    await init().then(() => console.log("toymlrs kmeans initialized"));
    greet("ToymlRS");
    const options = {
        k: 2,
        centroidsInitMethod: "kmeans++",
        maxIter: 100,
        randomSeed: 42,
    };

    const kmeans = new Kmeans(options);

    // Test data: two clusters of points
    const points = [
        [0, 0],
        [0.1, 0.1],
        [10, 10],
        [10.1, 10.1],
    ];

    let labels = kmeans.fit_predict([[0], [0], [1], [1]]);
    console.log("Point labels:", labels);

    const clusters = kmeans.cluster_();
    console.log("Cluster clusters:", clusters);

    const centroids = kmeans.centroids_();
    console.log("Cluster centroids:", centroids);
}

run().then(r => console.log("Hello toymlrs"));

import init, { greet, Kmeans, KmeansOptions } from "toymlrs";

async function run() {
  await init().then(() => console.log("toymlrs kmeans initialized"));
  const options = {
    k: 2,
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

  // Use fit_predict to get cluster labels
  const labels = kmeans.fit_predict(points);
  console.log("Cluster labels:", Array.from(labels));

  // Or use fit separately
  kmeans.fit(points);
}

run();

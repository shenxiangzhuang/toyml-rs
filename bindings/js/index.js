import {greet, Kmeans} from './pkg';

greet('World');

let opts = {k: 2, centroidsInitMethod: "kmeans++", maxIter: 100, randomSeed: 42};
let kmeans = new Kmeans(opts)

let labels = kmeans.fit_predict([[0], [0], [1], [1]]);
console.log("Point labels:", labels);

const clusters = kmeans.cluster_();
console.log("Cluster clusters:", clusters);

const centroids = kmeans.centroids_();
console.log("Cluster centroids:", centroids);

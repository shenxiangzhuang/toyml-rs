import {greet, Kmeans} from './pkg';

greet('World');

let opts = {k: 2, maxIter: 100, randomSeed: 42, centroidsInitMethod: "random"};
let kmeans = new Kmeans(opts)
let labels = kmeans.fit_predict([[0], [0], [1], [1]]);
console.log(labels);
const centroids_ = kmeans.centroids_();
console.log("Cluster centroids_:", centroids_);

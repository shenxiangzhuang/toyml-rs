import {greet, Kmeans} from './pkg';

greet('World');

let opts = {k: 2, maxIter: 100, randomSeed: 42};
let kmeans = new Kmeans(opts)
let labels = kmeans.fit_predict([[0], [0], [1], [1]]);
console.log(labels);

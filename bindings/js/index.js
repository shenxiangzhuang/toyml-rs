import {greet, Kmeans, KmeansOptions} from './pkg';

greet('World');

let opts: KmeansOptions = {k: 2, maxIter: 100, randomSeed: 42}
let kmeans: Kmeans = {opts: opts};
let labels = kmeans.fit_predict([[0], [0], [1], [1]])
console.log(labels)

use std::collections::HashMap;
use std::f64;
use rand::prelude::SeedableRng;

/// Dataset structs

#[derive(Default, Debug, Clone, PartialEq)]
struct Point(Vec<f64>);

impl Point {
    pub fn get_nearest_cluster_index(&self, centroids: &Centroids) -> usize {
        unimplemented!("get_nearest_cluster_index() is not implemented yet")
    }
}

#[derive(Default, Debug, PartialEq)]
struct Points(Vec<Point>);

impl Points {
    pub fn get_init_centroids(&self, centroids_init_method: &str, k: usize, random_seed: usize) -> Centroids {
        match centroids_init_method {
            "random" => Centroids {
                centroid_map: HashMap::from_iter(
                    self.sample(k, random_seed).0
                    .into_iter()
                    .enumerate()
                    .map(|(i, point)| (i, point))
                ),
            },
            "kmeans++" => {
                unimplemented!("kmeans++ init centroids is not implemented yet");
            }
            _ => panic!("Don't support the initialize method: {:?}!", centroids_init_method)
        }
    }

    pub fn mean(&self) -> Point {
        unimplemented!("mean() is not implemented yet");
    }

    fn sample(&self, k: usize, random_seed: usize) -> Points {
        use rand::seq::SliceRandom;
        let mut rng = rand::rngs::StdRng::seed_from_u64(random_seed as u64);
        Points((0..self.0.len()).collect::<Vec<_>>().choose_multiple(&mut rng, k).into_iter().map(|i| self.0[*i].clone()).collect())
    }
}


/// K-means structs
#[derive(Default, Debug)]
pub struct Labels(Vec<usize>);


#[derive(Default, Debug)]
pub struct Cluster {
    point_indices: Vec<usize>,
}

impl Cluster {
    pub fn get_points(&self, points: &Points) -> &Points {
        unimplemented!("get_cluster_points() is not implemented yet");
    }

}

#[derive(Debug)]
pub struct Clusters {
    pub cluster_map: HashMap<usize, Cluster>
}

impl Default for Clusters {
    fn default() -> Self {
        Self {
            cluster_map: HashMap::new()
        }
    }
}

impl Clusters {
    pub fn get_centroids(&self, points: &Points) -> Centroids {
        Centroids{
            centroid_map: HashMap::from_iter(
                self.cluster_map
                    .iter()
                    .map(|(&cluster_index, cluster)| {
                        (cluster_index, cluster.get_points(points).mean())
                    })
            )
        }
    }
}

#[derive(Debug)]
pub struct Centroids {
    pub centroid_map: HashMap<usize, Point>,
}

impl Default for Centroids {
    fn default() -> Self {
        Self {
            centroid_map: HashMap::new()
        }
    }
}


impl Centroids {
    fn get_clusters(&self, points: &Points) -> Clusters {
        let mut clusters = Clusters::default();
        let _ = points.0
            .iter()
            .enumerate()
            .map(|(index, point)|
                {
                    clusters.cluster_map
                        .entry(point.get_nearest_cluster_index(&self))
                        .or_insert(Cluster::default()).point_indices.push(index);
                });
        clusters
    }
}


#[derive(Debug)]
pub struct Kmeans {
    pub k: usize,
    pub max_iter: usize,
    pub tolerance: f64,
    pub centroids_init_method: &'static str,
    pub random_seed: usize,
    pub distance_metric: &'static str,
    clusters: Clusters,
    centroids: Centroids,
    labels: Labels,
}

impl Default for Kmeans {
    fn default() -> Self {
        Kmeans {
            k: 5,
            max_iter: 500,
            tolerance: 1e-5,
            centroids_init_method: "random",
            random_seed: 42,
            distance_metric: "euclidean",
            clusters: Clusters::default(),
            centroids: Centroids::default(),
            labels: Labels::default(),
        }
    }
}


impl Kmeans {
    pub fn fit(&mut self, points: Points) {
        self.centroids = points.get_init_centroids(self.centroids_init_method,
                                                    self.k,
                                                    self.random_seed);
        let mut iter: usize = 0;
        while iter < self.max_iter {
            println!("{:?}", self.centroids);
            self.clusters = self.centroids.get_clusters(&points);
            self.centroids = self.clusters.get_centroids(&points);
            // TODO: Early stop check here
            iter += 1;
        }

    }

    pub fn get_clusters(&self) -> &Clusters {
        &self.clusters
    }

    pub fn get_centroids(&self) -> &Centroids {
        &self.centroids
    }

    pub fn get_labels(&self) -> &Labels {
        &self.labels
    }
}



#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_points() -> Points {
        Points(vec![
            Point(vec![1.0, 2.0]),
            Point(vec![3.0, 4.0]),
            Point(vec![5.0, 6.0]),
            Point(vec![7.0, 8.0]),
        ])
    }

    #[test]
    fn test_kmeans_fit() {
        let mut kmeans = Kmeans{k: 2, ..Default::default() };
        let dataset = create_test_points();
        kmeans.fit(dataset);
        println!("{:?}", kmeans.centroids);
    }

    #[test]
    fn test_dataset_get_init_centroids() {
        let dataset = create_test_points();
        let centroids = dataset.get_init_centroids("random", 2, 42);

        assert_eq!(centroids.centroid_map.len(), 2);
        for (_, centroid) in centroids.centroid_map.iter() {
            assert_eq!(centroid.0.len(), 2);
        }
    }

    #[test]
    fn test_dataset_sample() {
        let dataset = create_test_points();
        let sampled = dataset.sample(2, 42);

        assert_eq!(sampled.0.len(), 2);
        for point in sampled.0.iter() {
            assert!(dataset.0.contains(point));
        }
    }
}



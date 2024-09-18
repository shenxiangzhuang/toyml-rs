use rand::prelude::SeedableRng;
use std::collections::HashMap;
use std::f64;

/// Dataset structs
#[derive(Default, Debug, Clone, PartialEq)]
pub struct Point {
    pub values: Vec<f64>,
}

impl Point {
    pub fn dim(&self) -> usize {
        self.values.len()
    }

    pub fn distance(&self, other: &Point, distance_metric: &str) -> f64 {
        if self.values.len() != other.values.len() {
            panic!(
                "Points with different dimensions are not supported: {:?}, {:?}",
                self.values, other.values
            );
        }
        match distance_metric {
            "euclidean" => self
                .values
                .iter()
                .zip(other.values.iter())
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f64>()
                .powf(1.0 / 2.0),
            _ => {
                panic!("Only euclidean distance metric is supported")
            }
        }
    }
}

#[derive(Default, Debug, PartialEq)]
pub struct Points(pub Vec<Point>);

impl Points {
    pub fn get_init_centroids(
        &self,
        centroids_init_method: &str,
        k: usize,
        random_seed: usize,
    ) -> Centroids {
        match centroids_init_method {
            "random" => Centroids {
                centroid_map: HashMap::from_iter(
                    self.sample(k, random_seed).0.into_iter().enumerate(),
                ),
            },
            "kmeans++" => {
                panic!("kmeans++ init centroids is not implemented yet");
            }
            _ => panic!(
                "Don't support the initialize method: {:?}!",
                centroids_init_method
            ),
        }
    }

    fn sample(&self, k: usize, random_seed: usize) -> Points {
        use rand::seq::SliceRandom;
        let mut rng = rand::rngs::StdRng::seed_from_u64(random_seed as u64);
        Points(
            (0..self.0.len())
                .collect::<Vec<_>>()
                .choose_multiple(&mut rng, k)
                .into_iter()
                .map(|i| self.0[*i].clone())
                .collect(),
        )
    }
}

/// K-means structs
#[derive(Default, Debug)]
pub struct Labels(Vec<usize>);

impl Labels {
    pub fn set(&mut self, sample_index: usize, label: usize) {
        self.0[sample_index] = label
    }
}

#[derive(Default, Debug, Eq, PartialEq, Clone)]
pub struct Cluster {
    point_indices: Vec<usize>,
}

#[derive(Debug, Eq, PartialEq, Clone)]
pub struct Clusters {
    pub cluster_map: HashMap<usize, Cluster>,
}

impl Default for Clusters {
    fn default() -> Self {
        Self {
            cluster_map: HashMap::new(),
        }
    }
}

impl Clusters {
    pub fn get_centroids(&self, points: &Points) -> Centroids {
        Centroids {
            centroid_map: self
                .cluster_map
                .iter()
                .map(|(&cluster_index, cluster)| {
                    let sum: Vec<f64> = cluster
                        .point_indices
                        .iter()
                        .map(|&i| &points.0[i].values)
                        .fold(vec![0.0; points.0[0].dim()], |acc, p| {
                            acc.iter().zip(p).map(|(a, b)| a + b).collect()
                        });
                    let centroid = Point {
                        values: sum
                            .into_iter()
                            .map(|x| x / cluster.point_indices.len() as f64)
                            .collect(),
                    };
                    (cluster_index, centroid)
                })
                .collect(),
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
            centroid_map: HashMap::new(),
        }
    }
}

impl Centroids {
    fn get_clusters(&self, points: &Points) -> Clusters {
        let mut clusters = Clusters::default();
        points.0.iter().enumerate().for_each(|(index, point)| {
            clusters
                .cluster_map
                .entry(self.get_nearest_cluster_index(point))
                .or_insert(Cluster::default())
                .point_indices
                .push(index);
        });
        clusters
    }

    pub fn get_nearest_cluster_index(&self, point: &Point) -> usize {
        let index = self
            .centroid_map
            .iter()
            .map(|(&centroid_index, centroid_point)| {
                (centroid_index, centroid_point.distance(point, "euclidean"))
            })
            .fold(
                (0, f64::MAX),
                |(current_index, current_distance), (centroid_index, distance)| {
                    if distance < current_distance {
                        (centroid_index, distance)
                    } else {
                        (current_index, current_distance)
                    }
                },
            )
            .0;
        // println!("Point {:?}, Nearest Cluster {:?}", self, index);
        index
    }
}

#[derive(Debug)]
pub struct Kmeans {
    pub k: usize,
    pub max_iter: usize,
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
            k: 2,
            max_iter: 500,
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
    pub fn fit(&mut self, points: &Points) {
        self.centroids =
            points.get_init_centroids(self.centroids_init_method, self.k, self.random_seed);
        let mut iter: usize = 0;
        while iter < self.max_iter {
            let old_clusters = self.clusters.clone();
            self.fit_one_step(points);
            // Early stop
            if self.clusters == old_clusters {
                println!("Early stop");
                break;
            }
            iter += 1;
        }
        // set labels
        self.labels = Labels(vec![0; points.0.len()]);
        for (&cluster_index, cluster) in &self.clusters.cluster_map {
            for &point_index in &cluster.point_indices {
                self.labels.set(point_index, cluster_index);
            }
        }
    }

    pub fn fit_one_step(&mut self, points: &Points) {
        self.clusters = self.centroids.get_clusters(&points);
        self.centroids = self.clusters.get_centroids(&points);
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
            Point {
                values: vec![1.0, 0.0],
            },
            Point {
                values: vec![1.0, 1.0],
            },
            Point {
                values: vec![1.0, 2.0],
            },
            Point {
                values: vec![10.0, 0.0],
            },
            Point {
                values: vec![10.0, 1.0],
            },
            Point {
                values: vec![10.0, 2.0],
            },
        ])
    }

    #[test]
    fn test_kmeans_fit() {
        let mut kmeans = Kmeans {
            k: 2,
            max_iter: 100,
            ..Default::default()
        };
        let dataset = create_test_points();
        kmeans.fit(&dataset);
        assert_eq!(kmeans.centroids.centroid_map.len(), 2);
    }

    #[test]
    fn test_dataset_get_init_centroids() {
        let dataset = create_test_points();
        let centroids = dataset.get_init_centroids("random", 2, 42);

        assert_eq!(centroids.centroid_map.len(), 2);
        for (_, centroid) in centroids.centroid_map.iter() {
            assert_eq!(centroid.values.len(), 2);
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

    #[test]
    fn test_point_distance() {
        let p1 = Point {
            values: vec![1.0, 2.0, 3.0],
        };
        let p2 = Point {
            values: vec![4.0, 5.0, 6.0],
        };
        assert_eq!(p1.distance(&p2, "euclidean"), 5.196152422706632);
    }

    #[test]
    #[should_panic(expected = "Points with different dimensions are not supported")]
    fn test_point_distance_different_dimensions() {
        let p1 = Point {
            values: vec![1.0, 2.0],
        };
        let p2 = Point {
            values: vec![4.0, 5.0, 6.0],
        };
        p1.distance(&p2, "euclidean");
    }

    #[test]
    #[should_panic(expected = "Only euclidean distance metric is supported")]
    fn test_point_distance_unsupported_metric() {
        let p1 = Point {
            values: vec![1.0, 2.0],
        };
        let p2 = Point {
            values: vec![4.0, 5.0],
        };
        p1.distance(&p2, "manhattan");
    }

    #[test]
    fn test_clusters_get_centroids() {
        let points = Points(vec![
            Point {
                values: vec![1.0, 1.0],
            },
            Point {
                values: vec![2.0, 2.0],
            },
            Point {
                values: vec![3.0, 3.0],
            },
            Point {
                values: vec![10.0, 10.0],
            },
            Point {
                values: vec![11.0, 11.0],
            },
        ]);
        let mut clusters = Clusters::default();
        clusters.cluster_map.insert(
            0,
            Cluster {
                point_indices: vec![0, 1, 2],
            },
        );
        clusters.cluster_map.insert(
            1,
            Cluster {
                point_indices: vec![3, 4],
            },
        );

        let centroids = clusters.get_centroids(&points);
        assert_eq!(centroids.centroid_map.len(), 2);
        assert_eq!(centroids.centroid_map[&0].values, vec![2.0, 2.0]);
        assert_eq!(centroids.centroid_map[&1].values, vec![10.5, 10.5]);
    }

    #[test]
    fn test_centroids_get_nearest_cluster_index() {
        let centroids = Centroids {
            centroid_map: HashMap::from([
                (
                    0,
                    Point {
                        values: vec![1.0, 1.0],
                    },
                ),
                (
                    1,
                    Point {
                        values: vec![5.0, 5.0],
                    },
                ),
            ]),
        };
        let point = Point {
            values: vec![2.0, 2.0],
        };
        assert_eq!(centroids.get_nearest_cluster_index(&point), 0);

        let point = Point {
            values: vec![4.0, 4.0],
        };
        assert_eq!(centroids.get_nearest_cluster_index(&point), 1);
    }

    #[test]
    fn test_kmeans_fit_convergence() {
        let mut kmeans = Kmeans {
            k: 2,
            max_iter: 1000,
            ..Default::default()
        };
        let dataset = create_test_points();
        kmeans.fit(&dataset);

        // Check if the clusters are as expected
        let clusters = kmeans.get_clusters();
        assert_eq!(clusters.cluster_map.len(), 2);

        // The exact cluster assignments may vary due to random initialization,
        // so we'll check if the points are grouped correctly
        let cluster1 = &clusters.cluster_map[&0].point_indices;
        let cluster2 = &clusters.cluster_map[&1].point_indices;

        assert!(
            (cluster1.len() == 3 && cluster2.len() == 3)
                || (cluster1.len() == 6 && cluster2.is_empty())
                || (cluster1.is_empty() && cluster2.len() == 6)
        );

        if cluster1.len() == 3 && cluster2.len() == 3 {
            assert!(
                (cluster1.contains(&0) && cluster1.contains(&1) && cluster1.contains(&2))
                    || (cluster1.contains(&3) && cluster1.contains(&4) && cluster1.contains(&5))
            );
        }
    }
}

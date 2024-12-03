use pyo3::prelude::*;
use rand::prelude::SeedableRng;
use std::collections::HashMap;

#[derive(Debug)]
pub enum DistanceMetric {
    Euclidean,
}

#[derive(Debug, Copy, Clone)]
pub enum CentroidsInitMethod {
    Random,
    KmeansPlusPlus,
}

/// Dataset structs
#[derive(Default, Debug, Clone, PartialEq)]
pub struct Point {
    pub values: Vec<f64>,
}

impl Point {
    pub fn dim(&self) -> usize {
        self.values.len()
    }

    pub fn distance(&self, other: &Point, metric: Option<DistanceMetric>) -> f64 {
        if self.values.len() != other.values.len() {
            panic!(
                "Points with different dimensions are not supported: {:?}, {:?}",
                self.values, other.values
            );
        }
        match metric {
            None | Some(DistanceMetric::Euclidean) => self
                .values
                .iter()
                .zip(other.values.iter())
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f64>()
                .powf(1.0 / 2.0),
        }
    }
}

#[derive(Default, Debug, PartialEq)]
pub struct Points(pub Vec<Point>);

impl Points {
    pub fn get_init_centroids(
        &self,
        centroids_init_method: CentroidsInitMethod,
        k: usize,
        random_seed: usize,
    ) -> Centroids {
        match centroids_init_method {
            CentroidsInitMethod::Random => Centroids {
                centroid_map: HashMap::from_iter(
                    self.sample(k, random_seed).0.into_iter().enumerate(),
                ),
            },
            CentroidsInitMethod::KmeansPlusPlus => {
                panic!("kmeans++ init centroids is not implemented yet");
            }
        }
    }

    fn sample(&self, k: usize, random_seed: usize) -> Points {
        use rand::seq::SliceRandom;
        let mut rng = rand::rngs::StdRng::seed_from_u64(random_seed as u64);
        Points(
            (0..self.0.len())
                .collect::<Vec<_>>()
                .choose_multiple(&mut rng, k)
                .map(|i| self.0[*i].clone())
                .collect(),
        )
    }
}

/// K-means structs
#[pyclass]
#[derive(Default, Debug)]
pub struct Labels(pub Vec<usize>);

impl Labels {
    pub fn set(&mut self, sample_index: usize, label: usize) {
        self.0[sample_index] = label
    }
}

#[derive(Default, Debug, Eq, PartialEq, Clone)]
pub struct Cluster {
    pub point_indices: Vec<usize>,
}

#[derive(Debug, Default, Eq, PartialEq, Clone)]
pub struct Clusters {
    pub cluster_map: HashMap<usize, Cluster>,
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

#[derive(Debug, Default)]
pub struct Centroids {
    pub centroid_map: HashMap<usize, Point>,
}

impl Centroids {
    pub fn get_clusters(&self, points: &Points) -> Clusters {
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
                (centroid_index, centroid_point.distance(point, None))
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

#[pyclass(name = "KmeansRust")]
#[derive(Debug)]
pub struct Kmeans {
    pub k: usize,
    pub max_iter: usize,
    pub centroids_init_method: CentroidsInitMethod,
    pub random_seed: usize,
    pub distance_metric: DistanceMetric,
    clusters: Clusters,
    centroids: Centroids,
    labels: Labels,
}

impl Default for Kmeans {
    fn default() -> Self {
        Kmeans {
            k: 2,
            max_iter: 500,
            centroids_init_method: CentroidsInitMethod::Random,
            random_seed: 42,
            distance_metric: DistanceMetric::Euclidean,
            clusters: Clusters::default(),
            centroids: Centroids::default(),
            labels: Labels::default(),
        }
    }
}

impl Kmeans {
    pub fn fit_one_step(&mut self, points: &Points) {
        self.clusters = self.centroids.get_clusters(points);
        self.centroids = self.clusters.get_centroids(points);
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

#[pymethods]
impl Kmeans {
    #[new]
    fn py_new(k: usize, max_iter: usize) -> PyResult<Self> {
        Ok(Kmeans {
            k,
            max_iter,
            centroids_init_method: CentroidsInitMethod::Random,
            random_seed: rand::random::<usize>(),
            distance_metric: DistanceMetric::Euclidean,
            clusters: Clusters::default(),
            centroids: Centroids::default(),
            labels: Labels::default(),
        })
    }

    pub fn fit(&mut self, point_values: Vec<Vec<f64>>) {
        let points = &Points(
            point_values
                .into_iter()
                .map(|v| Point { values: v })
                .collect(),
        );
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

    pub fn fit_predict(&mut self, point_values: Vec<Vec<f64>>) -> PyResult<Vec<usize>> {
        self.fit(point_values);
        self.labels()
    }

    #[getter]
    fn labels(&self) -> PyResult<Vec<usize>> {
        Ok(self.labels.0.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_points() -> Vec<Vec<f64>> {
        vec![
            vec![1.0, 0.0],
            vec![1.0, 1.0],
            vec![1.0, 2.0],
            vec![10.0, 0.0],
            vec![10.0, 1.0],
            vec![10.0, 2.0],
        ]
    }

    #[test]
    fn test_kmeans_fit() {
        let mut kmeans = Kmeans {
            k: 2,
            max_iter: 100,
            ..Default::default()
        };
        let dataset = create_test_points();
        kmeans.fit(dataset);
        assert_eq!(kmeans.centroids.centroid_map.len(), 2);
    }

    #[test]
    fn test_dataset_get_init_centroids() {
        let point_values = create_test_points();
        let dataset = Points(
            point_values
                .into_iter()
                .map(|v| Point { values: v })
                .collect(),
        );
        let centroids = dataset.get_init_centroids(CentroidsInitMethod::Random, 2, 42);

        assert_eq!(centroids.centroid_map.len(), 2);
        for (_, centroid) in centroids.centroid_map.iter() {
            assert_eq!(centroid.values.len(), 2);
        }
    }

    #[test]
    fn test_dataset_sample() {
        let points_values = create_test_points();
        let dataset = Points(
            points_values
                .into_iter()
                .map(|v| Point { values: v })
                .collect(),
        );
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
        assert_eq!(
            p1.distance(&p2, Some(DistanceMetric::Euclidean)),
            5.196152422706632
        );
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
        p1.distance(&p2, Some(DistanceMetric::Euclidean));
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
        kmeans.fit(dataset);

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

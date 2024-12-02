mod kmeans;

use std::collections::HashMap;
use kmeans::*;

use pyo3::prelude::*;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}


#[pyclass]
struct PyKmeans {
    pub k: usize,
    pub max_iter: usize,
    pub centroids_init_method: CentroidsInitMethod,
    pub random_seed: usize,
    pub distance_metric: DistanceMetric,
    clusters: Clusters,
    centroids: Centroids,
    labels: Labels,
}


#[pymethods]
impl PyKmeans {
    #[new]
    fn py_new(k: usize) -> PyResult<Self> {
        Ok(PyKmeans {
            k,
            max_iter: 100,
            centroids_init_method: CentroidsInitMethod::Random,
            random_seed: rand::random::<usize>(),
            distance_metric: DistanceMetric::Euclidean,
            clusters: Clusters::default(),
            centroids: Centroids::default(),
            labels: Labels::default(),
        })
    }
    
    pub fn fit(&mut self, points: &Points) {
        // TODO: deduplicated of code
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
        self.clusters = self.centroids.get_clusters(points);
        self.centroids = self.clusters.get_centroids(points);
    }

    #[getter]
    fn labels(&self) -> PyResult<Vec<usize>> {
        Ok(self.labels.0.clone())
    }

}

/// A Python module implemented in Rust.
#[pymodule]
fn _toymlrs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    let _ = m.add_class::<PyKmeans>();
    let _ = m.add_class::<Points>();
    Ok(())
}

use pyo3::prelude::*;

#[derive(Debug)]
#[pyclass]
pub struct Kmeans {
    inner: toymlrs_clustering::kmeans::Kmeans,
}

#[pymethods]
impl Kmeans {
    #[new]
    #[pyo3(signature = (k, max_iter, centroids_init_method="random", distance_metric="euclidean", random_seed=None))]
    fn py_new(
        k: usize,
        max_iter: usize,
        centroids_init_method: &str,
        distance_metric: &str,
        random_seed: Option<u64>,
    ) -> PyResult<Self> {
        Ok(Kmeans {
            inner: toymlrs_clustering::kmeans::Kmeans::new(
                k,
                max_iter,
                centroids_init_method
                    .parse()
                    .expect("Centroids method should be random or kmeans++"),
                distance_metric
                    .parse()
                    .expect("Distance method should be euclidean"),
                random_seed,
            ),
        })
    }

    pub fn fit(&mut self, point_values: Vec<Vec<f64>>) {
        self.inner.fit(point_values);
    }

    pub fn fit_predict(&mut self, point_values: Vec<Vec<f64>>) -> PyResult<Vec<usize>> {
        self.fit(point_values);
        self.labels()
    }

    #[getter]
    fn labels(&self) -> PyResult<Vec<usize>> {
        Ok(self.inner.get_labels().0.clone())
    }
}

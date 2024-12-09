//! Bindings for clustering algorithms.
use crate::core::*;
use serde::Deserialize;
use tsify_next::Tsify;
use wasm_bindgen::prelude::*;

/// Options for the dynamic time warping calculation.
#[derive(Clone, Debug, Default, Deserialize, Tsify)]
#[serde(rename_all = "camelCase")]
#[tsify(from_wasm_abi)]
pub struct KmeansOptions {
    /// The k in kmeans
    pub k: usize,
    pub max_iter: usize,
    pub random_seed: Option<u64>,
}

/// A DBSCAN clustering algorithm.
#[derive(Debug)]
#[wasm_bindgen]
pub struct Kmeans {
    inner: toymlrs_clustering::kmeans::Kmeans,
}

#[wasm_bindgen]
impl Kmeans {
    /// Create a new Kmeans instance.
    #[wasm_bindgen(constructor)]
    pub fn new(opts: KmeansOptions) -> Self {
        Self {
            inner: toymlrs_clustering::kmeans::Kmeans::new(
                opts.k,
                opts.max_iter,
                toymlrs_clustering::kmeans::CentroidsInitMethod::KmeansPlusPlus,
                toymlrs_clustering::kmeans::DistanceMetric::Euclidean,
                opts.random_seed,
            ),
        }
    }

    /// Fit the Kmeans clustering algorithm to the given data points.
    #[wasm_bindgen]
    pub fn fit(&mut self, point_values: VecVecF64) {
        self.inner.fit(point_values.convert().unwrap())
    }

    #[wasm_bindgen]
    pub fn fit_predict(&mut self, point_values: VecVecF64) -> Result<Vec<usize>, JsError> {
        self.fit(point_values);
        Ok(self.inner.get_labels().0.to_vec())
    }
}

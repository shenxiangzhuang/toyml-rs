//! Bindings for clustering algorithms.

use crate::core::*;
use serde::Deserialize;
use std::collections::HashMap;
use tsify_next::Tsify;
use wasm_bindgen::prelude::*;

/// The distance function to use for point distance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Tsify, Default)]
#[serde(rename_all = "lowercase")]
#[tsify(from_wasm_abi)]
pub enum CentroidsInitMethod {
    Random,
    #[default]
    KmeansPlusPlus,
}

/// The kmeans options.
#[derive(Clone, Debug, Default, Deserialize, Tsify)]
#[serde(rename_all = "camelCase")]
#[tsify(from_wasm_abi)]
pub struct KmeansOptions {
    /// The k in kmeans
    pub k: usize,
    pub max_iter: usize,
    pub centroids_init_method: CentroidsInitMethod,
    pub random_seed: Option<u64>,
}

/// A Kmeans clustering algorithm.
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
                match opts.centroids_init_method {
                    CentroidsInitMethod::Random => {
                        toymlrs_clustering::kmeans::CentroidsInitMethod::Random
                    }
                    CentroidsInitMethod::KmeansPlusPlus => {
                        toymlrs_clustering::kmeans::CentroidsInitMethod::KmeansPlusPlus
                    }
                },
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
        self.labels_()
    }

    #[wasm_bindgen]
    pub fn labels_(&self) -> Result<Vec<usize>, JsError> {
        Ok(self.inner.get_labels().0.to_vec())
    }

    #[wasm_bindgen]
    pub fn centroids_(&self) -> Result<JsValue, JsError> {
        let centroids: HashMap<usize, Vec<f64>> = self
            .inner
            .get_centroids()
            .centroid_map
            .iter()
            .map(|(k, v)| (*k, v.values.to_vec()))
            .collect();
        Ok(serde_wasm_bindgen::to_value(&centroids)?)
    }
}

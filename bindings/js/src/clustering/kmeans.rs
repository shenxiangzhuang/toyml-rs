//! Bindings for kmeans clustering algorithms.

use crate::core::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tsify_next::Tsify;
use wasm_bindgen::prelude::*;

/// The distance function to use for point distance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Tsify, Default)]
#[tsify(from_wasm_abi)]
pub enum CentroidsInitMethod {
    #[serde(rename = "random")]
    Random,
    #[default]
    #[serde(rename = "kmeans++")]
    KmeansPlusPlus,
}

impl From<CentroidsInitMethod> for toymlrs_clustering::kmeans::CentroidsInitMethod {
    fn from(method: CentroidsInitMethod) -> Self {
        match method {
            CentroidsInitMethod::Random => toymlrs_clustering::kmeans::CentroidsInitMethod::Random,
            CentroidsInitMethod::KmeansPlusPlus => {
                toymlrs_clustering::kmeans::CentroidsInitMethod::KmeansPlusPlus
            }
        }
    }
}

/// The kmeans options.
#[derive(Clone, Debug, Default, Deserialize, Tsify)]
#[serde(rename_all = "camelCase")]
#[tsify(from_wasm_abi)]
pub struct KmeansOptions {
    /// The k in kmeans
    pub k: usize,
    pub centroids_init_method: CentroidsInitMethod,
    pub max_iter: usize,
    pub random_seed: Option<u64>,
}

/// A Kmeans clustering algorithm.
#[derive(Debug)]
#[wasm_bindgen]
pub struct Kmeans {
    inner: toymlrs_clustering::kmeans::Kmeans,
}

#[derive(Debug, Clone, Serialize, Tsify)]
#[tsify(into_wasm_abi)]
pub struct Centroids {
    pub centroid_map: HashMap<usize, Vec<f64>>,
}

impl From<&toymlrs_clustering::kmeans::Centroids> for Centroids {
    fn from(c: &toymlrs_clustering::kmeans::Centroids) -> Self {
        Self {
            centroid_map: c
                .centroid_map
                .iter()
                .map(|(k, v)| (*k, v.values.clone()))
                .collect(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Tsify)]
#[tsify(into_wasm_abi)]
pub struct Clusters {
    pub cluster_map: HashMap<usize, Vec<usize>>,
}

impl From<&toymlrs_clustering::kmeans::Clusters> for Clusters {
    fn from(c: &toymlrs_clustering::kmeans::Clusters) -> Self {
        Self {
            cluster_map: c
                .cluster_map
                .iter()
                .map(|(k, v)| (*k, v.point_indices.clone()))
                .collect(),
        }
    }
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
                opts.centroids_init_method.into(),
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
    pub fn centroids_(&self) -> Result<Centroids, JsError> {
        Ok(self.inner.get_centroids().into())
    }

    #[wasm_bindgen]
    pub fn cluster_(&self) -> Result<Clusters, JsError> {
        Ok(self.inner.get_clusters().into())
    }
}

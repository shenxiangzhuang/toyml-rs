//! Bindings for clustering algorithms.
use serde::Deserialize;
use tsify_next::Tsify;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    fn alert(s: &str);

    /// Custom type for `Vec<usize>`.
    #[wasm_bindgen(typescript_type = "number[] | Uint32Array")]
    #[derive(Debug)]
    pub type VecUsize;

    /// Custom type for `Vec<f64>`.
    #[wasm_bindgen(typescript_type = "number[] | Float64Array")]
    #[derive(Debug)]
    pub type VecF64;

    /// Custom type for `Vec<Vec<f64>>`.
    #[wasm_bindgen(typescript_type = "number[][] | Float64Array[]")]
    #[derive(Debug)]
    pub type VecVecF64;
}

#[wasm_bindgen]
pub fn test_kmeans() {
    use js_sys::Float64Array;
    use wasm_bindgen::prelude::*;
    use web_sys::console;

    let opts = KmeansOptions { k: 2, max_iter: 100, random_seed: Some(42) };
    let mut model = Kmeans::new(opts);
    let points = vec![
        vec![1.0, 0.0],
        vec![1.0, 1.0],
        vec![1.0, 2.0],
        vec![10.0, 0.0],
        vec![10.0, 1.0],
        vec![10.0, 2.0],
    ];

    // Convert and log each row with its values
    let js_array = points.into_iter()
        .enumerate()
        .map(|(i, row)| {
            let float_array = Float64Array::from(&row[..]);

            // Convert Float64Array to Vec to display values
            let values: Vec<f64> = (0..float_array.length())
                .map(|j| float_array.get_index(j))
                .collect();

            console::log_2(
                &format!("Row {}: ", i).into(),
                &format!("{:?}", values).into(),
            );
            float_array
        })
        .collect::<js_sys::Array>();

    // Log all values in a structured way
    let all_values: Vec<Vec<f64>> = (0..js_array.length())
        .map(|i| {
            let array = js_sys::Float64Array::unchecked_from_js(js_array.get(i));
            (0..array.length())
                .map(|j| array.get_index(j))
                .collect()
        })
        .collect();

    console::log_2(
        &"Complete array values:".into(),
        &format!("{:?}", all_values).into(),
    );

    let points_array: JsValue = js_array.into();
    let points_js: VecVecF64 = points_array.into();

    alert(&format!("Kmeans: {:?}", model.fit_predict(points_js)));
}


impl VecUsize {
    /// Convert to a `Vec<usize>`.
    pub fn convert(self) -> Result<Vec<usize>, JsError> {
        serde_wasm_bindgen::from_value(self.into())
            .map_err(|_| JsError::new("TypeError: expected array of integers or Uint32Array"))
    }
}

impl VecF64 {
    /// Convert to a `Vec<f64>`.
    pub fn convert(self) -> Result<Vec<f64>, JsError> {
        serde_wasm_bindgen::from_value(self.into())
            .map_err(|_| JsError::new("TypeError: expected array of numbers or Float64Array"))
    }
}

impl VecVecF64 {
    /// Convert to a `Vec<Vec<f64>>`.
    pub fn convert(self) -> Result<Vec<Vec<f64>>, JsError> {
        serde_wasm_bindgen::from_value(self.into()).map_err(|_| {
            JsError::new("TypeError: expected array of number arrays or array of Float64Array")
        })
    }
}

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

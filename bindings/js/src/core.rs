use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    // https://github.com/rustwasm/wasm-bindgen/issues/201
    #[wasm_bindgen]
    pub fn alert(s: &str);

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

#[wasm_bindgen]
pub fn greet(name: &str) {
    alert(&format!("Hello, {}!", name));
}

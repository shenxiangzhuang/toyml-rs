mod kmeans;

use crate::kmeans::*;
use kmeans::Kmeans;

fn main() {
    let points = Points(vec![
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
    ]);
    let mut kmeans = Kmeans::default();
    kmeans.fit(&points);
    println!("Clusters: {:?}", kmeans.get_labels());
}

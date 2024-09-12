mod kmeans;

use kmeans::{Point, kmeans};

fn main() {
    let points = vec![
        Point { x: 1.0, y: 1.0 },
        Point { x: 1.5, y: 2.0 },
        Point { x: 3.0, y: 4.0 },
        Point { x: 5.0, y: 7.0 },
        Point { x: 3.5, y: 5.0 },
        Point { x: 4.5, y: 5.0 },
        Point { x: 3.5, y: 4.5 },
    ];

    let k = 2;
    let max_iterations = 100;

    let clusters = kmeans(&points, k, max_iterations);

    println!("Clusters: {:?}", clusters);
}
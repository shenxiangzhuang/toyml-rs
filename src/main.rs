mod kmeans;

use kmeans::{Kmeans};

fn main() {
    let kmeans = Kmeans::default();
    println!("Clusters: {:?}", kmeans.get_labels());
}

use toymlrs_clustering::kmeans::{CentroidsInitMethod, Kmeans};

fn main() {
    let points = vec![
        vec![1.0, 0.0],
        vec![1.0, 1.0],
        vec![1.0, 2.0],
        vec![10.0, 0.0],
        vec![10.0, 1.0],
        vec![10.0, 2.0],
    ];
    let mut kmeans = Kmeans::default();
    kmeans.centroids_init_method = CentroidsInitMethod::KmeansPlusPlus;
    kmeans.fit(points);
    println!("Clusters: {:?}", kmeans.get_labels());
}

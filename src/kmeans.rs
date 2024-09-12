use rand::Rng;
use std::f64;

#[derive(Debug, Clone)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

impl Point {
    fn distance(&self, other: &Point) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }
}

pub fn kmeans(points: &[Point], k: usize, max_iterations: usize) -> Vec<Vec<usize>> {
    let mut rng = rand::thread_rng();
    let mut centroids: Vec<Point> = (0..k)
        .map(|_| Point {
            x: rng.gen_range(0.0..100.0),
            y: rng.gen_range(0.0..100.0),
        })
        .collect();

    let mut clusters: Vec<Vec<usize>> = vec![Vec::new(); k];

    for _ in 0..max_iterations {
        // Assign points to clusters
        clusters.iter_mut().for_each(|c| c.clear());
        for (i, point) in points.iter().enumerate() {
            let closest_centroid = centroids
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    point.distance(a).partial_cmp(&point.distance(b)).unwrap()
                })
                .map(|(index, _)| index)
                .unwrap();
            clusters[closest_centroid].push(i);
        }

        // Update centroids
        let mut new_centroids = vec![Point { x: 0.0, y: 0.0 }; k];
        for (i, cluster) in clusters.iter().enumerate() {
            if !cluster.is_empty() {
                let (sum_x, sum_y) = cluster.iter().fold((0.0, 0.0), |(sx, sy), &idx| {
                    (sx + points[idx].x, sy + points[idx].y)
                });
                new_centroids[i] = Point {
                    x: sum_x / cluster.len() as f64,
                    y: sum_y / cluster.len() as f64,
                };
            }
        }

        // Check for convergence
        if centroids
            .iter()
            .zip(new_centroids.iter())
            .all(|(a, b)| a.distance(b) < 1e-6)
        {
            break;
        }

        centroids = new_centroids;
    }

    clusters
}

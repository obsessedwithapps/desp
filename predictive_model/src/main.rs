#![allow(warnings)]
use linfa::prelude::*;
use linfa_datasets::*;
use linfa_trees::*;
use ndarray::{array, Array};
use std::{fs::File, io};
use text_io::read;

fn main() -> io::Result<()> {
    // Training data ...
    // The features
    let training_file = File::open("datasets/training.csv")?;
    let training_set = linfa_datasets::array_from_csv(training_file, false, b',').unwrap();

    // Scores - High or low risk
    let lines = std::fs::read_to_string("datasets/training_scores.txt")?;
    let mut scores = lines.lines().collect::<Vec<&str>>(); 

    // Train the model
    let dataset = Dataset::new(training_set, Array::from_vec(scores));
    let model = DecisionTree::params().fit(&dataset).unwrap();

    // Input
    println!("Please input data (neuroticism, sleep, social scores), a space between each score:");
    let mut buffer: String = String::new();
    io::stdin().read_line(&mut buffer)?;

    let sample = buffer
        .split_ascii_whitespace()
        .map(|x| x.parse::<f64>().unwrap())
        .collect::<Vec<f64>>();

    let new_sample = array![[sample[0], sample[1], sample[2]]];

    let prediction = model.predict(&new_sample);
    println!("Prediction: {:?}", prediction[0]);

    Ok(())
}

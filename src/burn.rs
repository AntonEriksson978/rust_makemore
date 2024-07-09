#![allow(dead_code)]
use anyhow::Result;
use burn::backend::candle::CandleDevice;
use burn::backend::Candle;
use burn::tensor::{Data, Distribution, Int, Shape, Tensor};
use std::path::Path;

use crate::utils::{self, char_to_index};
type BE = Candle;

pub fn makemore_probability_neural_net() -> Result<()> {
    let device = CandleDevice::default();
    let input_path = Path::new("data/names.txt");

    let words = utils::read_words_from_file(&input_path)?;

    // # Generate dataset

    // first word as bigram
    let first_word = utils::extract_bigrams(&words[0]);
    for (ch1, ch2) in &first_word {
        println!("{}{}", ch1, ch2);
    }

    // input
    let xs = &first_word
        .iter()
        .map(|(ch1, _)| ch1.clone())
        .collect::<Vec<char>>();
    // training answer, every x should have high probability to be y
    let ys = &first_word
        .iter()
        .map(|(_, ch2)| ch2.clone())
        .collect::<Vec<char>>();
    let y_indices: Vec<i64> = ys.iter().map(|y| char_to_index(y) as i64).collect();
    let ys_len = y_indices.len();
    let ys_data = Data::new(y_indices, Shape::new([ys_len]));
    let ys_tensor = Tensor::<BE, 1, Int>::from(ys_data);
    let _y_range = ys.iter().map(|y| {
        let s = char_to_index(y);
        s..s + 1
    });
    println!("ys_tensor: {}", ys_tensor);

    println!("{:?}", xs);
    println!("{:?}", ys);

    // # Neural network with one layer of neurons

    // initialize neuron layer weights with random values
    let neuron_weighs = Tensor::<BE, 2>::random([27, 27], Distribution::Normal(0.0, 1.0), &device);

    // skipping bias for now

    // # Forward pass

    // encode input character indexes to one-hot representation
    let x_one_hots = xs
        .iter()
        .map(|x| Tensor::<BE, 2>::one_hot(char_to_index(x), 27, &device))
        .collect::<Vec<Tensor<BE, 2>>>();
    let x_one_hot_tensor = Tensor::cat(x_one_hots, 0);
    println!("x_one_hot: {}", &x_one_hot_tensor);

    // run input through the neuron layer

    // logits are the rows of makemore_probability_table that correspond to the weight of each character in the word
    // this is what the one_hot_tensor multiplication is doing.
    let logits = x_one_hot_tensor.clone().matmul(neuron_weighs);

    // apply softmax to get probabilities (softmax turns numbers to postive and then normalizes them)
    // also worth noting is that this counts tensor will, with training, converge to the actual counts
    // like the bigram_counts tensor in makemore_probability_table.
    let counts = logits.exp();
    let probabilites = counts.clone() / counts.sum_dim(1);
    println!("probabilities: {}", &probabilites);

    // let relevant_probabilites = probabilites.clone().slice(y_range).to_data();
    // println!("relevant_probabilties: {}", &relevant_probabilites);

    // let loss = println!("Negative log likelihood: {}", -loss);
    // println!("Averge negative log likelihood: {}", -(loss / n));

    Ok(())
}

pub fn makemore_probability_table(words: Vec<String>, alphabet: &[(usize, char); 27]) {
    let device = CandleDevice::default();
    let bigram_counts = calculate_bigram_counts(&words, &alphabet);
    let counts_tensor = Tensor::<BE, 2>::from_floats(bigram_counts, &device);
    let prob_tensor = counts_tensor.clone() / counts_tensor.clone().sum_dim(1);
    let log_prob_tensor = prob_tensor.clone().log();

    let (log_prob, n) = calculate_log_prob(&words, &alphabet, &log_prob_tensor);

    println!("Negative log likelihood: {}", -log_prob);
    println!("Averge negative log likelihood: {}", -(log_prob / n));
}

pub fn calculate_bigram_counts(
    words: &[String],
    alphabet: &[(usize, char); 27],
) -> [[f32; 27]; 27] {
    let mut bigram_counts = [[1.0; 27]; 27];
    for word in words {
        let bigrams = utils::extract_bigrams(word);
        for (ch1, ch2) in bigrams {
            let ch1_index = alphabet.iter().position(|&(_, c)| c == ch1).unwrap();
            let ch2_index = alphabet.iter().position(|&(_, c)| c == ch2).unwrap();
            bigram_counts[ch1_index][ch2_index] += 1.0;
        }
    }
    bigram_counts
}

pub fn calculate_log_prob(
    words: &[String],
    alphabet: &[(usize, char); 27],
    log_prob_tensor: &Tensor<BE, 2>,
) -> (f32, f32) {
    let mut log_prob = 0.0;
    let mut n = 0.0;
    for word in &words[0..3] {
        let bigrams = utils::extract_bigrams(word);
        for (ch1, ch2) in bigrams {
            let ch1_index = alphabet.iter().position(|&(_, c)| c == ch1).unwrap();
            let ch2_index = alphabet.iter().position(|&(_, c)| c == ch2).unwrap();
            log_prob += log_prob_tensor
                .clone()
                .slice([ch1_index..ch1_index + 1, ch2_index..ch2_index + 1])
                .to_data()
                .value[0];
            n += 1.0;
        }
    }
    (log_prob, n)
}

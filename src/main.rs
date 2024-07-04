#![allow(dead_code)]
use burn::backend::candle::CandleDevice;
use burn::backend::Candle;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::iter;
use std::path::Path;

use burn::tensor::{Distribution, Tensor};

type BE = Candle;
fn main() -> io::Result<()> {
    let device = CandleDevice::default();
    let input_path = Path::new("data/names.txt");

    let words = read_words_from_file(&input_path)?;

    // # Generate dataset

    // first word as bigram
    let first_word = extract_bigrams(&words[0]);
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
    let logits = x_one_hot_tensor.matmul(neuron_weighs);
    println!("logits: {}", &logits);

    // apply softmax to get probabilities (softmax turns numbers to postive and then normalizes them)
    let counts = logits.exp();
    println!("counts: {}", &counts);
    let probabilites = counts.clone() / counts.sum_dim(1);
    println!("probabilities: {}", &probabilites);

    Ok(())
}

fn makemore_probability_table(words: Vec<String>) {
    let bigram_counts = calculate_bigram_counts(&words, &ALPHABET);

    let device = CandleDevice::default();
    let counts_tensor = Tensor::<BE, 2>::from_floats(bigram_counts, &device);
    let prob_tensor = counts_tensor.clone() / counts_tensor.clone().sum_dim(1);
    let log_prob_tensor = prob_tensor.clone().log();

    let (log_prob, n) = calculate_log_prob(&words, &ALPHABET, &log_prob_tensor);

    println!("Negative log likelihood: {}", -log_prob);
    println!("Averge negative log likelihood: {}", -(log_prob / n));
}

fn read_words_from_file(input_path: &Path) -> io::Result<Vec<String>> {
    let input_file = File::open(input_path)?;
    let reader = BufReader::new(input_file);
    let words = reader
        .lines()
        .map(|line| line.unwrap())
        .collect::<Vec<String>>();
    Ok(words)
}

fn calculate_bigram_counts(words: &[String], alphabet: &[(usize, char); 27]) -> [[f32; 27]; 27] {
    let mut bigram_counts = [[1.0; 27]; 27];
    for word in words {
        let bigrams = extract_bigrams(word);
        for (ch1, ch2) in bigrams {
            let ch1_index = alphabet.iter().position(|&(_, c)| c == ch1).unwrap();
            let ch2_index = alphabet.iter().position(|&(_, c)| c == ch2).unwrap();
            bigram_counts[ch1_index][ch2_index] += 1.0;
        }
    }
    bigram_counts
}

fn calculate_log_prob(
    words: &[String],
    alphabet: &[(usize, char); 27],
    log_prob_tensor: &Tensor<BE, 2>,
) -> (f32, f32) {
    let mut log_prob = 0.0;
    let mut n = 0.0;
    for word in &words[0..3] {
        let bigrams = extract_bigrams(word);
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

fn extract_bigrams(word: &str) -> Vec<(char, char)> {
    let chars: Vec<char> = iter::once('.')
        .chain(word.chars())
        .chain(iter::once('.'))
        .collect();
    chars.windows(2).map(|w| (w[0], w[1])).collect()
}

const INDEX_TO_CHAR: [char; 27] = [
    '.', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
    's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
];

fn char_to_index(c: &char) -> usize {
    INDEX_TO_CHAR.iter().position(|x| x == c).unwrap()
}

// const CHAR_TO_INDEX: HashMap<char, usize> = INDEX_TO_CHAR
//     .iter()
//     .enumerate()
//     .map(|(i, c)| (*c, i))
//     .collect();

const ALPHABET: [(usize, char); 27] = [
    (0, '.'),
    (1, 'a'),
    (2, 'b'),
    (3, 'c'),
    (4, 'd'),
    (5, 'e'),
    (6, 'f'),
    (7, 'g'),
    (8, 'h'),
    (9, 'i'),
    (10, 'j'),
    (11, 'k'),
    (12, 'l'),
    (13, 'm'),
    (14, 'n'),
    (15, 'o'),
    (16, 'p'),
    (17, 'q'),
    (18, 'r'),
    (19, 's'),
    (20, 't'),
    (21, 'u'),
    (22, 'v'),
    (23, 'w'),
    (24, 'x'),
    (25, 'y'),
    (26, 'z'),
];

use burn::backend::candle::CandleDevice;
use burn::backend::Candle;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::iter;
use std::path::Path;

use burn::tensor::Tensor;

type BE = Candle;
fn main() -> io::Result<()> {
    let input_path = Path::new("data/names.txt");

    let words = read_words_from_file(&input_path)?;

    let bigram_counts = calculate_bigram_counts(&words, &ALPHABET);

    let device = CandleDevice::default();
    let counts_tensor = Tensor::<BE, 2>::from_floats(bigram_counts, &device);
    let prob_tensor = counts_tensor.clone() / counts_tensor.clone().sum_dim(1);
    let log_prob_tensor = prob_tensor.clone().log();

    let (log_prob, n) = calculate_log_prob(&words, &ALPHABET, &log_prob_tensor);

    println!("Negative log likelihood: {}", -log_prob);
    println!("Averge negative log likelihood: {}", -(log_prob / n));

    Ok(())
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

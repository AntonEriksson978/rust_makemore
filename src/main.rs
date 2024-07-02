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

    let input_file = File::open(&input_path)?;
    let reader = BufReader::new(input_file);
    let words = reader
        .lines()
        .map(|line| line.unwrap())
        .collect::<Vec<String>>();

    let device = CandleDevice::default();
    let alphabet = [
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

    let mut bigram_counts = [[0.0; 27]; 27];
    for word in words.clone() {
        let chars: Vec<char> = iter::once('.')
            .chain(word.chars())
            .chain(iter::once('.'))
            .collect();
        let bigrams = chars.windows(2).map(|w| (w[0], w[1]));
        for bigram in bigrams {
            let start = alphabet.iter().find(|(_, c)| *c == bigram.0).unwrap().0 as usize;
            let end = alphabet.iter().find(|(_, c)| *c == bigram.1).unwrap().0 as usize;
            bigram_counts[start][end] += 1.0;
        }
    }

    let counts_tensor = Tensor::<BE, 2>::from_floats(bigram_counts, &device);
    let prob_tensor = counts_tensor.clone() / counts_tensor.clone().sum_dim(1);
    let log_prob_tensor = prob_tensor.clone().log();

    // println!("{}", prob_tensor);
    // println!("{}", prob_tensor.to_data().shape.num_elements());
    // println!("{}", prob_tensor.to_data().value[3 + 27 * 4]);
    // println!("{}", prob_tensor.clone().slice([0..1, 0..1]).to_data());
    // // println!("{}", prob_tensor.slice([0..1, 0..1]).to_data());

    let mut log_prob = 0.0;
    let mut n = 0.0;
    for word in words.iter().take(3) {
        let chars: Vec<char> = iter::once('.')
            .chain(word.chars())
            .chain(iter::once('.'))
            .collect();
        let bigrams = chars.windows(2).map(|w| (w[0], w[1]));
        for (ch1, ch2) in bigrams {
            let ch1_index = alphabet.iter().position(|&(_, c)| c == ch1).unwrap();
            let ch2_index = alphabet.iter().position(|&(_, c)| c == ch2).unwrap();
            let prop = prob_tensor
                .clone()
                .slice([ch1_index..ch1_index + 1, ch2_index..ch2_index + 1]);
            println!(
                "{}{}: {:.4}",
                alphabet[ch1_index].1,
                alphabet[ch2_index].1,
                prop.to_data().value[0]
            );
            log_prob += log_prob_tensor
                .clone()
                .slice([ch1_index..ch1_index + 1, ch2_index..ch2_index + 1])
                .to_data()
                .value[0];
            n += 1.0;
        }
        println!("{}: {:.4}", word, log_prob);
    }
    println!("Negative log likelihood: {}", -log_prob);
    println!("Averge negative log likelihood: {}", -(log_prob / n));

    Ok(())
}

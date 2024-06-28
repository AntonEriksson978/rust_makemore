use burn::backend::candle::CandleDevice;
use burn::backend::Candle;
use std::collections::HashMap;
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
        .take(10)
        .map(|line| line.unwrap())
        .collect::<Vec<String>>();

    let device = CandleDevice::default();
    let a = Tensor::<BE, 2>::full([28, 28], 0, &device);

    println!("{}", a);

    let mut bigram_counts = HashMap::<(char, char), i32>::new();
    for word in words {
        let chars: Vec<char> = iter::once('S')
            .chain(word.chars())
            .chain(iter::once('E'))
            .collect();
        let bigrams = chars.windows(2).map(|w| (w[0], w[1]));
        for bigram in bigrams {
            let count = bigram_counts.get(&bigram).unwrap_or(&0) + 1;
            bigram_counts.insert(bigram, count);
        }
    }
    println!("{:?}", bigram_counts);
    Ok(())
}

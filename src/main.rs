use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::iter;
use std::path::Path;

fn main() -> io::Result<()> {
    let input_path = Path::new("data/names.txt");

    let input_file = File::open(&input_path)?;
    let reader = BufReader::new(input_file);
    let words = reader
        .lines()
        .map(|line| line.unwrap())
        .collect::<Vec<String>>();

    let mut bigram_counts = HashMap::<(char, char), i32>::new();
    for word in words {
        let chars: Vec<char> = iter::once('S')
            .chain(word.chars())
            .chain(iter::once('E'))
            .collect();
        for (ch1, ch2) in chars.windows(2).map(|w| (w[0], w[1])) {
            let bigram = (ch1, ch2);
            bigram_counts.insert(bigram, bigram_counts.get(&bigram).unwrap_or(&0) + 1);
        }
    }
    println!("{:?}", bigram_counts);
    Ok(())
}

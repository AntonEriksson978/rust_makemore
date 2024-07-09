use std::{
    fs::File,
    io::{self, BufRead, BufReader},
    iter,
    path::Path,
};

pub const INDEX_TO_CHAR: [char; 27] = [
    '.', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
    's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
];

pub fn char_to_index(c: &char) -> usize {
    INDEX_TO_CHAR.iter().position(|x| x == c).unwrap()
}

// const CHAR_TO_INDEX: HashMap<char, usize> = INDEX_TO_CHAR
//     .iter()
//     .enumerate()
//     .map(|(i, c)| (*c, i))
//     .collect();

pub const ALPHABET: [(usize, char); 27] = [
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

pub fn read_words_from_file(input_path: &Path) -> io::Result<Vec<String>> {
    let input_file = File::open(input_path)?;
    let reader = BufReader::new(input_file);
    let words = reader
        .lines()
        .map(|line| line.unwrap())
        .collect::<Vec<String>>();
    Ok(words)
}

pub fn extract_bigrams(word: &str) -> Vec<(char, char)> {
    let chars: Vec<char> = iter::once('.')
        .chain(word.chars())
        .chain(iter::once('.'))
        .collect();
    chars.windows(2).map(|w| (w[0], w[1])).collect()
}

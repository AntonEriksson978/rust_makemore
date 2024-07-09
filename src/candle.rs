use std::path::Path;

use anyhow::Result;
use burn::nn;
use candle_core::{DType, Device, Shape, Tensor};
use candle_nn::encoding::one_hot;

use crate::utils::{self, char_to_index};
pub fn makemore_neural_net_candle() -> Result<()> {
    let device = Device::Cpu;
    let input_path = Path::new("data/names.txt");

    let words = utils::read_words_from_file(&input_path)?;

    // # Generate dataset

    // first word as bigram
    let first_word = utils::extract_bigrams(&words[0]);
    // for (ch1, ch2) in &first_word {
    //     println!("{}{}", ch1, ch2);
    // }

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
    let neuron_weighs = Tensor::randn(0f32, 1.0, (27, 27), &device)?;
    // println!("weights: {:?}", neuron_weighs.to_vec2::<f32>()?);
    // println!("weights: {}", neuron_weighs);

    // skipping bias for now

    // # Forward pass

    // encode input character indexes to one-hot representation
    let x_one_hot = xs
        .iter()
        .map(|x| utils::char_to_index(x) as u32)
        .collect::<Vec<u32>>();
    let hot = one_hot(
        Tensor::from_vec(x_one_hot, Shape::from_dims(&[5]), &device)?,
        27,
        1u32,
        0,
    )?
    .to_dtype(DType::F32)?;

    // run input through the neuron layer

    // logits are the rows of makemore_probability_table that correspond to the weight of each character in the word
    // this is what the one_hot_tensor multiplication is doing.
    let logits = hot.matmul(&neuron_weighs)?;
    // println!("logits: {}", &logits);

    // apply softmax to get probabilities (softmax turns numbers to postive and then normalizes them)
    // also worth noting is that this counts tensor will, with training, converge to the actual counts
    // like the bigram_counts tensor in makemore_probability_table.
    let counts = logits.exp()?;
    let probabilites = counts.clone().broadcast_div(&counts.sum_keepdim(1)?)?;
    println!("probabilities: {}", &probabilites);
    // println!(
    //     "PROB: {}",
    //     &probabilites.index_select(&Tensor::new(&[0u32, 5, 13, 13, 1], &device)?, 1)?
    // );
    let len = xs.len();
    // let mut loss = Tensor::zeros(len, DType::F32, &device)?;
    let mut loss = Vec::<f32>::new();
    for (i, c) in xs.iter().enumerate() {
        let first = probabilites.get_on_dim(0, i)?;
        let p = first.get_on_dim(0, char_to_index(c))?;
        let ps = p.to_scalar()?;
        loss.push(ps);
    }

    let loss = probabilites.log()?.mean_all()?.neg()?;
    println!("loss: {}", &loss.to_scalar::<f32>()?);

    Ok(())
}

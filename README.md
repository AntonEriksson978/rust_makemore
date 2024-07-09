# Andrej Karpathy's makemore written in rust

## Purpose
This is a rust implementation of Andrej Karpathy's makemore. The purpose of this project is to learn neural networks and rust.

## What is makemore?

> "Makemore takes one text file as input, where each line is assumed to be one training thing, and generates more things like it. Under the hood, it is an autoregressive character-level language model, with a wide choice of models from bigrams all the way to a Transformer (exactly as seen in GPT). For example, we can feed it a database of names, and makemore will generate cool baby name ideas that all sound name-like, but are not already existing names. Or if we feed it a database of company names then we can generate new ideas for a name of a company. Or we can just feed it valid scrabble words and generate english-like babble."

[Link to Andrejs github repo](https://github.com/karpathy/makemore?tab=readme-ov-file)


## Libraries
- candle
- burn

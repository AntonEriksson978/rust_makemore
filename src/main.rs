#![allow(dead_code)]
use anyhow::Result;
use rust_makemore::candle;

fn main() -> Result<()> {
    candle::makemore_neural_net_candle()?;
    Ok(())
}

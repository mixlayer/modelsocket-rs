use anyhow::{Context, Result};
use clap::Parser;
use modelsocket::client::{AppendOpts, GenOpts, ModelSocket};

mod utils;

#[derive(Parser, Debug)]
struct Args {
    #[clap(short, long, default_value = "wss://models.mixlayer.ai/ws")]
    url: String,
    #[clap(short, long, env = "MODELSOCKET_API_KEY")]
    api_key: String,
    #[clap(long, long, default_value_t = false)]
    hidden: bool,
    #[clap(long, long, default_value_t = true)]
    colors: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    let socket = ModelSocket::connect(&args.url, Some(&args.api_key))
        .await
        .context("websocket connection failed")?;

    let seq = socket
        .open_seq("meta/llama3.1-8b-instruct-free", None)
        .await
        .context("seq open failed")?;

    seq.append("Hello there.", AppendOpts::user()).await?;

    let stream = seq
        .generate(Some(GenOpts::assistant()))
        .await
        .context("generate failed")?;

    utils::print_stream(stream, args.hidden, args.colors).await?;

    Ok(())
}

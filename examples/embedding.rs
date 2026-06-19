use anyhow::{Context, Result};
use clap::Parser;
use modelsocket::{
    client::{EmbedOpts, ModelSocket},
    EmbeddingInput, EmbeddingInputType,
};

#[derive(Parser, Debug)]
struct Args {
    #[clap(short, long, default_value = "wss://models.mixlayer.ai/ws")]
    url: String,
    #[clap(short, long, env = "MODELSOCKET_API_KEY")]
    api_key: String,
    #[clap(short, long, env = "MODELSOCKET_EMBED_MODEL")]
    model: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    let socket = ModelSocket::connect(&args.url, Some(&args.api_key))
        .await
        .context("websocket connection failed")?;

    let seq = socket
        .open(&args.model, None)
        .await
        .context("seq open failed")?;

    let inputs = vec![
        "Rust is a systems programming language focused on safety and performance.",
        "Memory-safe systems programming is a core strength of Rust.",
        "Fresh basil and tomatoes make a simple pasta sauce.",
    ];

    let opts = EmbedOpts {
        input_type: Some(EmbeddingInputType::SemanticSimilarity),
        normalize: Some(true),
        ..Default::default()
    };

    let result = seq
        .embed(
            inputs
                .iter()
                .map(|input| EmbeddingInput::Text((*input).to_string()))
                .collect(),
            Some(opts),
        )
        .await
        .context("embed failed")?;

    for (idx, embedding) in result.embeddings.iter().enumerate() {
        println!(
            "{}: dimensions={}, input_tokens={}",
            idx,
            embedding.len(),
            result.input_tokens[idx]
        );
    }

    println!("prompt_tokens={}", result.prompt_tokens);

    Ok(())
}

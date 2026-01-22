use std::{future::Future, pin::Pin};

use anyhow::{Context, Result};
use clap::Parser;
use modelsocket::{
    client::{AppendOpts, GenOpts, ModelSocket},
    tools::{SimpleToolbox, Tool, ToolDefinition, Toolbox},
    OpenOpts,
};
use serde_json::json;

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
    tracing_subscriber::fmt::init();

    let socket = ModelSocket::connect(&args.url, Some(&args.api_key))
        .await
        .context("websocket connection failed")?;

    let mut opts = OpenOpts::default();
    let mut toolbox = SimpleToolbox::new();
    toolbox.add_tool(WeatherTool);
    toolbox.add_tool(CalculatorTool);
    opts.toolbox = Some(Box::new(toolbox) as Box<dyn Toolbox>);

    let seq = socket
        .open("qwen/qwen3-8b", Some(opts))
        .await
        .context("seq open failed")?;

    seq.append(
        "What's the weather in SF and what is 1.352 + 4.4442?",
        AppendOpts::user(),
    )
    .await?;

    let stream = seq
        .generate(Some(GenOpts::assistant()))
        .await
        .context("generate failed")?;

    utils::print_stream(stream, args.hidden, args.colors).await?;

    Ok(())
}

#[derive(Debug)]
pub struct WeatherTool;

impl Tool for WeatherTool {
    fn definition(&self) -> ToolDefinition {
        let parameters = json!({
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": r#"City and State and Country to retrieve the weather for. Do not use nicknames (such as "NYC"), use "<CITY>, <TWO LETTER STATE>, <COUNTRY>", Example: "New York, NY, USA". If the user provides an abbreviation, try to figure out the full name by yourself."#
                }
            },
            "required": ["location"]
        });

        let parameters = serde_json::from_value(parameters).unwrap();

        ToolDefinition {
            name: "get_current_weather".to_string(),
            description: "Get the current weather for a given location".to_string(),
            parameters,
        }
    }

    fn call(&self, _args: &str) -> Pin<Box<dyn Future<Output = Result<String>> + Send>> {
        Box::pin(async move { Ok("The weather is overcast and 52 F.".to_string()) })
    }
}

#[derive(Debug)]
pub struct CalculatorTool;

impl Tool for CalculatorTool {
    fn definition(&self) -> ToolDefinition {
        let parameters = json!({
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A mathematical expression to evaluate (e.g. 1 + 1)"
                }
            },
        });

        let parameters = serde_json::from_value(parameters).unwrap();

        ToolDefinition {
            name: "calculator".to_string(),
            description: "Evaluate a mathematical expression".to_string(),
            parameters,
        }
    }

    fn call(&self, args: &str) -> Pin<Box<dyn Future<Output = Result<String>> + Send>> {
        let args = args.to_string();
        let fut = async move {
            let expr_obj = serde_json::from_str::<serde_json::Value>(&args)?;
            let expression = expr_obj
                .get("expression")
                .ok_or_else(|| anyhow::anyhow!("expression field not found"))?
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("expression field is not a string"))?;

            let result = meval::eval_str(expression)?;
            Ok(result.to_string())
        };

        Box::pin(fut)
    }
}

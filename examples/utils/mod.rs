use anyhow::Result;
use futures::StreamExt;
use modelsocket::GenStream;
use std::io::{stdout, Write};

pub async fn print_stream(mut stream: GenStream, hidden: bool, colors: bool) -> Result<()> {
    use colorize::AnsiColor;
    let mut stdout = stdout();

    while let Some(event) = stream.next().await {
        if !hidden && event.hidden {
            continue;
        }

        if colors {
            print!(
                "{}",
                if event.hidden {
                    event.text.b_black()
                } else {
                    event.text.default()
                }
            );
        } else {
            print!("{}", event.text);
        }

        stdout.flush()?;
    }

    println!();

    Ok(())
}

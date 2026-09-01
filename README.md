# modelsocket.rs

Modelsocket.rs provides a Rust client and protocol types for talking to [Modelsocket](https://modelsocket.com/) model hosting endpoints. It exposes a WebSocket based `ModelSocket` client for opening sequences, streaming generations, and coordinating multiple forks of a conversation. The crate is designed so that consumers can either embed the client directly in Rust projects or build higher level integrations, such as Python bindings.

## Crate layout

- `protocol` contains serde-powered request/response structures that mirror the Modelsocket wire format.
- `client` (enabled with the `client` feature flag) implements a Tokio/WebSocket powered sequence client with helpers for opening sequences, appending messages, streaming tokens, and forking conversations.
- `python` (enabled with the `python` feature flag) re-exports PyO3 bindings that expose the blocking client API to Python.

## Feature flags

The crate ships with two opt-in feature flags so it can be kept lightweight when only the protocol types are needed:

| Feature  | Description                                                                                 |
| -------- | ------------------------------------------------------------------------------------------- |
| `client` | Pulls in the asynchronous WebSocket client and its Tokio/futures dependencies.              |
| `python` | Builds the PyO3 powered blocking bindings. This automatically enables the `client` feature. |

Enable the feature(s) you need when running Cargo commands, for example:

```bash
cargo check --features client
cargo test --features client
```

## Python bindings

Enabling the `python` feature exposes a `modelsocket` extension module that wraps the Rust client in a "blocking" API. The bindings reuse a shared single-threaded Tokio runtime under the hood so each call feels natural to synchronous Python code while still delegating the work to the async Rust implementation.

### Installing the bindings locally

The easiest way to experiment with the bindings is to install them into a virtual environment with [maturin](https://github.com/PyO3/maturin):

```bash
python -m venv .venv
source .venv/bin/activate
pip install maturin
maturin develop --release -F python
```

The last command builds the extension module (`modelsocket`) in-place using the `python` feature flag and makes it importable inside the virtual environment.

### Example

Once the module is installed you can drive the blocking client just like the Rust version. The snippet below opens a sequence, sends a user message, and prints the generated response:

```python
from modelsocket import BlockingModelSocketClient

client = BlockingModelSocketClient.connect(
    "wss://models.mixlayer.ai/ws",
    api_key="sk_example_123"
)
seq = client.open("meta/llama-3.1-8b-instruct-free")

seq.append("Hello there!", role="user")

# blocking generate
reply = seq.gen_text()
print(reply)

# ...or streaming generate
stream = seq.gen_text_stream(
    role="assistant",
    max_tokens=200,
    reasoning_level="medium",
)
for chunk in stream:
    if chunk.reasoning:
        print(f"[thinking]{chunk.text}", end="", flush=True)
    else:
        print(chunk, end="", flush=True)

seq.close()
```

### Tool calling from Python

Define each tool with the `Tool` helper by providing its schema dictionary and a callable that accepts the raw JSON arguments string. Pass the list of tools to `BlockingModelSocketClient.open` and tool calls will be routed to the handlers automatically:

```python
import json
from modelsocket import BlockingModelSocketClient, Tool


def get_weather(args_json: str) -> str:
    args = json.loads(args_json)
    location = args["location"]
    return f"The weather in {location} is 72F and sunny."


tools = [
    Tool(
        name="get_current_weather",
        description="Lookup the current weather for a location",
        parameters={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City, state, and country to retrieve the forecast for.",
                }
            },
            "required": ["location"],
        },
        function=get_weather,
    )
]

client = BlockingModelSocketClient.connect("wss://models.mixlayer.ai/ws")
seq = client.open("qwen/qwen3-8b", tools=tools)
```

Each handler receives the JSON payload for the tool call and should return a string that is forwarded back to the model sequence.

### Reasoning fields

Generation requests can carry reasoning controls:

```json
{
  "command": "gen",
  "max_tokens": 256,
  "reasoning": {
    "level": "medium",
    "max_tokens": 128
  }
}
```

Generated text events can carry an explicit `reasoning` flag - use this to determine when reasoning output is beingh generated:

```json
{
  "event": "seq_text",
  "seq_id": "seq_1",
  "cid": "gen_1",
  "text": "thinking step",
  "hidden": false,
  "reasoning": true,
  "num_input_tokens": 12,
  "num_output_tokens": 34
}
```

## License

The crate is distributed under the terms of the Apache 2.0 license.

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
seq = client.open(
    "meta/llama-3.1-8b-instruct-free",
    tools_enabled=False,
)

seq.append("Hello there!", role="user")
reply = sequence.gen_text()
print(reply)

seq.close()
```

## License

The crate is distributed under the terms of the Apache 2.0 license.

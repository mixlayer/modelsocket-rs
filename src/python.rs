use std::sync::Arc;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyModule, PyType};

use crate::client::{AppendOpts, GenOpts, ModelSocket, ModelSocketError, OpenOpts, Seq};

fn map_err(err: ModelSocketError) -> PyErr {
    PyRuntimeError::new_err(err.to_string())
}

#[pyclass(name = "BlockingModelSocketClient", module = "modelsocket_py")]
pub struct BlockingModelSocketClient {
    runtime: Arc<tokio::runtime::Runtime>,
    inner: ModelSocket,
}

#[pymethods]
impl BlockingModelSocketClient {
    #[classmethod]
    #[pyo3(
        name = "connect",
        text_signature = "(url, api_key=None)",
        signature = (url, api_key=None)
    )]
    pub fn connect(_cls: &Bound<'_, PyType>, url: &str, api_key: Option<&str>) -> PyResult<Self> {
        let runtime = tokio::runtime::Runtime::new().map_err(|err| {
            PyRuntimeError::new_err(format!("failed to create tokio runtime: {err}"))
        })?;

        let inner = runtime
            .block_on(ModelSocket::connect(url, api_key))
            .map_err(map_err)?;

        Ok(Self {
            runtime: Arc::new(runtime),
            inner,
        })
    }

    #[pyo3(
        text_signature = "($self, model, /, *, tools_enabled=False, tool_prompt=None, skip_prelude=False)",
        signature = (model, tools_enabled=None, tool_prompt=None, skip_prelude=None)
    )]
    pub fn open(
        &self,
        model: &str,
        tools_enabled: Option<bool>,
        tool_prompt: Option<&str>,
        skip_prelude: Option<bool>,
    ) -> PyResult<BlockingSequence> {
        let runtime = self.runtime.clone();
        let client = self.inner.clone();

        let seq = runtime
            .block_on(async move {
                let mut opts = OpenOpts::default();
                opts.tools_enabled = tools_enabled.unwrap_or(false);
                opts.tool_prompt = tool_prompt.map(|s| s.to_string());
                opts.skip_prelude = skip_prelude.unwrap_or(false);
                client.open(model, Some(opts)).await
            })
            .map_err(map_err)?;

        Ok(BlockingSequence {
            runtime,
            inner: seq,
        })
    }
}

#[pyclass(name = "BlockingSequence", module = "modelsocket_py")]
pub struct BlockingSequence {
    runtime: Arc<tokio::runtime::Runtime>,
    inner: Seq,
}

#[pymethods]
impl BlockingSequence {
    #[pyo3(text_signature = "($self, text, /, *, role=None)", signature = (text, role=None))]
    pub fn append(&self, text: &str, role: Option<&str>) -> PyResult<()> {
        let runtime = self.runtime.clone();
        let seq = self.inner.clone();
        runtime
            .block_on(async move {
                let mut opts = AppendOpts::default();
                opts.role = role.map(|r| r.to_string());
                seq.append(text, opts).await
            })
            .map_err(map_err)
    }

    #[pyo3(
        text_signature = "($self, /, *, role=None, stop_strings=None, max_length=None, max_tokens=None, hidden=None, temperature=None, top_p=None, top_k=None, repeat_penalty=None, seed=None)",
        signature = (
            role=None,
            stop_strings=None,
            max_length=None,
            max_tokens=None,
            hidden=None,
            temperature=None,
            top_p=None,
            top_k=None,
            repeat_penalty=None,
            seed=None
        )
    )]
    pub fn generate_text(
        &self,
        role: Option<&str>,
        stop_strings: Option<Vec<String>>,
        max_length: Option<u32>,
        max_tokens: Option<u32>,
        hidden: Option<bool>,
        temperature: Option<f32>,
        top_p: Option<f32>,
        top_k: Option<i32>,
        repeat_penalty: Option<f32>,
        seed: Option<u64>,
    ) -> PyResult<String> {
        let runtime = self.runtime.clone();
        let seq = self.inner.clone();

        runtime
            .block_on(async move {
                let opts = build_gen_opts(
                    role,
                    stop_strings,
                    max_length,
                    max_tokens,
                    hidden,
                    temperature,
                    top_p,
                    top_k,
                    repeat_penalty,
                    seed,
                );
                let stream = seq.generate(Some(opts)).await?;
                stream.text().await
            })
            .map_err(map_err)
    }

    #[pyo3(
        text_signature = "($self, /, *, role=None, stop_strings=None, max_length=None, max_tokens=None, hidden=None, temperature=None, top_p=None, top_k=None, repeat_penalty=None, seed=None)",
        signature = (
            role=None,
            stop_strings=None,
            max_length=None,
            max_tokens=None,
            hidden=None,
            temperature=None,
            top_p=None,
            top_k=None,
            repeat_penalty=None,
            seed=None
        )
    )]
    pub fn generate_text_and_tokens(
        &self,
        role: Option<&str>,
        stop_strings: Option<Vec<String>>,
        max_length: Option<u32>,
        max_tokens: Option<u32>,
        hidden: Option<bool>,
        temperature: Option<f32>,
        top_p: Option<f32>,
        top_k: Option<i32>,
        repeat_penalty: Option<f32>,
        seed: Option<u64>,
    ) -> PyResult<(String, Vec<u32>)> {
        let runtime = self.runtime.clone();
        let seq = self.inner.clone();

        runtime
            .block_on(async move {
                let opts = build_gen_opts(
                    role,
                    stop_strings,
                    max_length,
                    max_tokens,
                    hidden,
                    temperature,
                    top_p,
                    top_k,
                    repeat_penalty,
                    seed,
                );
                let stream = seq.generate(Some(opts)).await?;
                stream.text_and_tokens().await
            })
            .map_err(map_err)
    }

    #[pyo3(text_signature = "($self)")]
    pub fn close(&self) -> PyResult<()> {
        let runtime = self.runtime.clone();
        let seq = self.inner.clone();
        runtime
            .block_on(async move { seq.close().await })
            .map_err(map_err)
    }

    #[pyo3(text_signature = "($self)")]
    pub fn fork(&self) -> PyResult<BlockingSequence> {
        let runtime = self.runtime.clone();
        let seq = self.inner.clone();

        let child = runtime
            .block_on(async move { seq.fork().await })
            .map_err(map_err)?;

        Ok(BlockingSequence {
            runtime,
            inner: child,
        })
    }
}

fn build_gen_opts(
    role: Option<&str>,
    stop_strings: Option<Vec<String>>,
    max_length: Option<u32>,
    max_tokens: Option<u32>,
    hidden: Option<bool>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<i32>,
    repeat_penalty: Option<f32>,
    seed: Option<u64>,
) -> GenOpts {
    let mut opts = GenOpts::default();
    opts.role = role.map(|r| r.to_string());
    opts.stop_strings = stop_strings;
    opts.max_length = max_length;
    opts.max_tokens = max_tokens;
    opts.hidden = hidden;
    opts.temperature = temperature;
    opts.top_p = top_p;
    opts.top_k = top_k;
    opts.repeat_penalty = repeat_penalty;
    opts.seed = seed;
    opts
}

#[pymodule]
fn modelsocket_py(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BlockingModelSocketClient>()?;
    m.add_class::<BlockingSequence>()?;
    Ok(())
}

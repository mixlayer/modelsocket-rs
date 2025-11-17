use std::future::Future;
use std::time::Duration;

use once_cell::sync::OnceCell;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyModule, PyType};
use tokio::runtime::{Builder, Runtime};

use crate::client::{
    AppendOpts, GenChunk, GenOpts, GenStream, ModelSocket, ModelSocketError, OpenOpts, Seq,
};

fn map_err(err: ModelSocketError) -> PyErr {
    PyRuntimeError::new_err(err.to_string())
}

static RUNTIME: OnceCell<Runtime> = OnceCell::new();

fn runtime() -> PyResult<&'static Runtime> {
    RUNTIME.get_or_try_init(|| {
        Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|err| {
                PyRuntimeError::new_err(format!("failed to create tokio runtime: {err}"))
            })
    })
}

/// Block on a future, checking for signals every 100ms.
/// Allows a future being blocked on to be interrupted by ctrl+c.
fn block_on<F, T>(py: Python<'_>, future: F) -> PyResult<T>
where
    F: Future<Output = Result<T, ModelSocketError>> + Send,
    T: Send,
{
    let runtime = runtime()?;

    let signal_fut = {
        async move {
            let mut tick = tokio::time::interval(Duration::from_millis(100));
            loop {
                tick.tick().await;
                if let Err(err) = Python::attach(|py| py.check_signals()) {
                    return err;
                }
            }
        }
    };

    let f = async move {
        tokio::pin! {
            let work = future;
            let signals = signal_fut;
        };

        tokio::select! {
            f = &mut work => f.map_err(map_err),
            _signal = &mut signals => {
                return Err(PyRuntimeError::new_err("interrupted by signal"));
            },
        }
    };

    py.detach(|| runtime.block_on(f))
}

#[pyclass(name = "BlockingModelSocketClient", module = "modelsocket")]
pub struct PyBlockingModelSocketClient {
    inner: ModelSocket,
}

#[pymethods]
impl PyBlockingModelSocketClient {
    #[classmethod]
    #[pyo3(
        name = "connect",
        text_signature = "(url, api_key=None)",
        signature = (url, api_key=None)
    )]
    pub fn connect(_cls: &Bound<'_, PyType>, url: &str, api_key: Option<&str>) -> PyResult<Self> {
        let inner = block_on(_cls.py(), ModelSocket::connect(url, api_key))?;

        Ok(Self { inner })
    }

    #[pyo3(
        text_signature = "($self, model, /, *, tools_enabled=False, tool_prompt=None, skip_prelude=False)",
        signature = (model, tool_prompt=None, skip_prelude=None)
    )]
    pub fn open(
        &self,
        model: &str,
        // tools_enabled: Option<bool>,
        tool_prompt: Option<&str>,
        skip_prelude: Option<bool>,
    ) -> PyResult<PyBlockingSeq> {
        let client = self.inner.clone();

        let seq = Python::attach(|py| {
            block_on(py, async move {
                let mut opts = OpenOpts::default();
                // opts.tools_enabled = tools_enabled.unwrap_or(false);
                opts.tool_prompt = tool_prompt.map(|s| s.to_string());
                opts.skip_prelude = skip_prelude.unwrap_or(false);
                client.open(model, Some(opts)).await
            })
        })?;

        Ok(PyBlockingSeq { inner: seq })
    }
}

#[pyclass(name = "BlockingSeq", module = "modelsocket")]
pub struct PyBlockingSeq {
    inner: Seq,
}

#[pymethods]
impl PyBlockingSeq {
    #[pyo3(text_signature = "($self, text, /, *, role=None)", signature = (text, role=None))]
    pub fn append(&self, text: &str, role: Option<&str>) -> PyResult<()> {
        let seq = self.inner.clone();
        Python::attach(|py| {
            block_on(py, async move {
                let mut opts = AppendOpts::default();
                opts.role = role.map(|r| r.to_string());
                seq.append(text, opts).await
            })
        })
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
    pub fn gen_text(
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
        let seq = self.inner.clone();

        Python::attach(|py| {
            block_on(py, async move {
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
        })
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
    pub fn gen_text_and_tokens(
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
        let seq = self.inner.clone();

        Python::attach(|py| {
            block_on(py, async move {
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
        })
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
    pub fn gen_text_stream(
        &self,
        py: Python<'_>,
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
    ) -> PyResult<Py<PyBlockingGenStream>> {
        let seq = self.inner.clone();

        let stream = block_on(py, async move {
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
            seq.generate(Some(opts)).await
        })?;

        Py::new(py, PyBlockingGenStream::new(stream))
    }

    #[pyo3(text_signature = "($self)")]
    pub fn close(&self) -> PyResult<()> {
        let seq = self.inner.clone();
        Python::attach(|py| block_on(py, async move { seq.close().await }))
    }

    #[pyo3(text_signature = "($self)")]
    pub fn fork(&self) -> PyResult<PyBlockingSeq> {
        let seq = self.inner.clone();

        let child = Python::attach(|py| block_on(py, async move { seq.fork().await }))?;

        Ok(PyBlockingSeq { inner: child })
    }
}

#[pyclass(name = "BlockingGenStream", module = "modelsocket")]
pub struct PyBlockingGenStream {
    stream: Option<GenStream>,
}

impl PyBlockingGenStream {
    fn new(stream: GenStream) -> Self {
        Self {
            stream: Some(stream),
        }
    }
}

#[pymethods]
impl PyBlockingGenStream {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<Py<PyGenEvent>>> {
        use futures::StreamExt;

        let Some(stream) = self.stream.take() else {
            return Ok(None);
        };

        let (stream, chunk) = block_on(py, async move {
            let mut stream = stream;
            let next = stream.next().await;
            Ok((stream, next))
        })?;

        if let Some(chunk) = chunk {
            let Ok(chunk) = chunk else {
                return Err(map_err(chunk.unwrap_err()));
            };

            let event = Py::new(py, PyGenEvent::from(chunk))?;
            self.stream = Some(stream);
            Ok(Some(event))
        } else {
            self.stream = None;
            Ok(None)
        }
    }
}

#[pyclass(name = "GenEvent", module = "modelsocket")]
pub struct PyGenEvent {
    #[pyo3(get)]
    text: String,
    #[pyo3(get)]
    hidden: bool,
    #[pyo3(get)]
    tokens: Option<Vec<u32>>,
}

impl From<GenChunk> for PyGenEvent {
    fn from(chunk: GenChunk) -> Self {
        Self {
            text: chunk.text,
            hidden: chunk.hidden,
            tokens: chunk.tokens,
        }
    }
}

#[pymethods]
impl PyGenEvent {
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "GenEvent(text={:?}, hidden={}, tokens={})",
            self.text,
            self.hidden,
            match &self.tokens {
                Some(tokens) => format!("{:?}", tokens),
                None => "None".to_string(),
            }
        ))
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(self.text.clone())
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
fn modelsocket(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBlockingModelSocketClient>()?;
    m.add_class::<PyBlockingSeq>()?;
    m.add_class::<PyBlockingGenStream>()?;
    m.add_class::<PyGenEvent>()?;
    Ok(())
}

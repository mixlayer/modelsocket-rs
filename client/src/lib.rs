mod seq;
pub mod transport;

use futures::{Sink, Stream};
use futures_util::{SinkExt, StreamExt};
use modelsocket_common::{MSEvent, MSRequest, SeqGenReq, SeqOpenReq};
pub use seq::{GenChunk, GenStream, Seq};
use std::{collections::HashMap, pin::Pin, sync::Arc};
use thiserror::Error;
use tokio::sync::{mpsc, Mutex};
use tokio_tungstenite::tungstenite::{protocol::Message, Error as WsError};
use tracing::{debug, error};
use uuid::Uuid;

#[derive(Error, Debug)]
pub enum ModelSocketError {
    #[error("ws error: {0}")]
    WebSocket(#[from] WsError),

    #[error("invalid url: {0}")]
    Url(#[from] url::ParseError),

    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("protocol error: {0}")]
    Protocol(String),

    #[error("state error: {0}")]
    State(String),

    #[error("open error: {0}")]
    Open(String),

    #[error("channel closed")]
    Send(#[from] mpsc::error::SendError<Message>),

    #[error("command error: {0}")]
    Command(String),

    #[error("other: {0}")]
    Other(anyhow::Error),
}

#[derive(Clone)]
pub struct ModelSocket {
    ws_sink: Arc<Mutex<Pin<Box<dyn Sink<MSRequest, Error = ModelSocketError> + Send>>>>,
    opening_seqs: Arc<Mutex<HashMap<String, mpsc::Sender<Result<String, ModelSocketError>>>>>,
    seqs: Arc<Mutex<HashMap<String, Seq>>>,
}

impl ModelSocket {
    pub fn new<RX, TX, T>(transport: T) -> Result<Self, ModelSocketError>
    where
        RX: Stream<Item = Result<MSEvent, ModelSocketError>> + Send + Unpin + 'static,
        TX: Sink<MSRequest, Error = ModelSocketError> + Send + 'static,
        T: transport::MSTransport<RX, TX>,
    {
        let (ws_sink, ws_stream) = transport.split();

        let socket = Self {
            ws_sink: Arc::new(Mutex::new(Box::pin(ws_sink))),
            opening_seqs: Arc::new(Mutex::new(HashMap::new())),
            seqs: Arc::new(Mutex::new(HashMap::new())),
        };

        let socket_clone = socket.clone_components();

        tokio::spawn(async move {
            socket_clone.read_loop(ws_stream).await;
        });

        Ok(socket)
    }

    pub async fn connect(url: &str, api_key: Option<&str>) -> Result<Self, ModelSocketError> {
        let ws_transport = transport::ws::WebSocketTransport::connect(url, api_key).await?;
        Self::new(ws_transport)
    }

    fn clone_components(&self) -> Self {
        Self {
            ws_sink: self.ws_sink.clone(),
            opening_seqs: self.opening_seqs.clone(),
            seqs: self.seqs.clone(),
        }
    }

    async fn read_loop<S: Stream<Item = Result<MSEvent, ModelSocketError>> + Unpin>(
        self,
        mut events: S,
    ) {
        while let Some(event) = events.next().await {
            match event {
                Ok(event) => self.on_event(event).await,
                Err(err) => {
                    error!("ms error: {}", err);
                    break;
                }
            }
        }
    }

    async fn on_event(&self, event: MSEvent) {
        debug!("<- {:?}", event);
        match &event {
            MSEvent::SeqOpened { seq_id, cid } => self.on_seq_opened(seq_id, cid).await,
            MSEvent::Error { cid, message, .. } => self.on_error(cid, message).await,
            MSEvent::SeqClosed { seq_id, .. } => {
                self.forward_to_seq(&event).await;
                self.on_seq_closed(seq_id).await
            }
            _ => self.forward_to_seq(&event).await,
        }
    }

    async fn forward_to_seq(&self, event: &MSEvent) {
        let seq_id = match event.seq_id() {
            Some(id) => id,
            None => {
                error!("unroutable event forwarded to seq: {:?}", event);
                return;
            }
        };

        let mut seqs = self.seqs.lock().await;

        if let Some(seq) = seqs.get_mut(seq_id) {
            seq.on_event(event).await;
        } else {
            error!("state error: unknown seq_id {}", seq_id);
        }
    }

    async fn on_error(&self, cid: &Option<String>, message: &str) {
        error!("error: {}", message);

        if let Some(cid) = cid {
            if let Some(sender) = self.opening_seqs.lock().await.remove(cid) {
                let _ = sender
                    .send(Err(ModelSocketError::Open(format!(
                        "open error: {}",
                        message
                    ))))
                    .await;
            }
        }
    }

    async fn on_seq_closed(&self, seq_id: &str) {
        if let Some(seq) = self.seqs.lock().await.remove(seq_id) {
            debug!(seq = seq_id, "removing seq from client state");
            drop(seq);
        } else {
            error!("state error: unknown seq_id {}", seq_id);
        }
    }

    async fn on_seq_opened(&self, seq_id: &str, cid: &str) {
        let mut opening_seqs = self.opening_seqs.lock().await;
        if let Some(sender) = opening_seqs.remove(cid) {
            if let Err(e) = sender.send(Ok(seq_id.to_owned())).await {
                error!("Failed to send seq_id: {}", e);
            }
        } else {
            error!("unknown opened seq cid {}", cid);
        }
    }

    pub async fn open(&self, model: &str, opts: Option<OpenOpts>) -> Result<Seq, ModelSocketError> {
        let cid = Uuid::new_v4().to_string();
        let (tx, mut rx) = mpsc::channel(1);

        {
            let mut opening_seqs = self.opening_seqs.lock().await;
            opening_seqs.insert(cid.clone(), tx);
        }

        let opts = opts.unwrap_or_default();

        self.send(MSRequest::SeqOpen {
            cid: cid.clone(),
            data: SeqOpenReq {
                model: model.to_string(),
                tools_enabled: opts.tools_enabled,
                tool_prompt: opts.tool_prompt,
                skip_prelude: opts.skip_prelude,
            },
        })
        .await?;

        let seq_id = rx.recv().await.unwrap_or_else(|| {
            Err(ModelSocketError::Open(
                "Failed to receive seq_id".to_string(),
            ))
        })?;

        let seq = Seq::new(
            seq_id.clone(),
            model.to_string(),
            self.clone_components(),
            None,
        );

        {
            let mut seqs = self.seqs.lock().await;
            let mut seq_event_handler = seq.clone();
            seq_event_handler.event_tx = opts.event_sink;

            seqs.insert(seq_id, seq_event_handler);
        }

        Ok(seq)
    }

    async fn send(&self, req: MSRequest) -> Result<(), ModelSocketError> {
        debug!("-> {:?}", req);
        let mut sink = self.ws_sink.lock().await;
        sink.send(req).await?;
        Ok(())
    }
}

#[derive(Default, Debug, Clone)]
pub struct AppendOpts {
    pub role: Option<String>,
}

#[derive(Default, Debug, Clone)]
pub struct GenOpts {
    pub role: Option<String>,
    pub stop_strings: Option<Vec<String>>,
    pub max_length: Option<u32>,
    pub max_tokens: Option<u32>,
    pub hidden: Option<bool>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<i32>,
    pub repeat_penalty: Option<f32>,
    pub seed: Option<u64>,
}

impl GenOpts {
    pub fn assistant() -> Self {
        Self {
            role: Some("assistant".to_string()),
            ..Default::default()
        }
    }

    pub fn user() -> Self {
        Self {
            role: Some("user".to_string()),
            ..Default::default()
        }
    }

    pub fn system() -> Self {
        Self {
            role: Some("system".to_string()),
            ..Default::default()
        }
    }
}

impl Into<SeqGenReq> for GenOpts {
    fn into(self) -> SeqGenReq {
        SeqGenReq {
            role: self.role,
            stop_strings: self.stop_strings,
            max_length: self.max_length,
            max_tokens: self.max_tokens,
            hidden: self.hidden.unwrap_or(false),
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            repeat_penalty: self.repeat_penalty,
            seed: self.seed,
            ..Default::default()
        }
    }
}

impl AppendOpts {
    pub fn assistant() -> Self {
        Self {
            role: Some("assistant".to_string()),
        }
    }

    pub fn user() -> Self {
        Self {
            role: Some("user".to_string()),
        }
    }

    pub fn system() -> Self {
        Self {
            role: Some("system".to_string()),
        }
    }
}

#[derive(Default, Debug)]
pub struct OpenOpts {
    /// Enable too use on the sequence. Will instruct the engine
    /// to append tool instructions to the system prompt.
    pub tools_enabled: bool,

    /// Optional system prompt to use when tools are enabled
    pub tool_prompt: Option<String>,

    /// Skips system prompt prescribed by model authors
    pub skip_prelude: bool,

    /// Optional sink for listening to all events related
    /// to this sequence
    pub event_sink: Option<mpsc::Sender<Result<MSEvent, ModelSocketError>>>,
}

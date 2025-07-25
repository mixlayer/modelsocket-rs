use futures_util::{
    stream::{SplitSink, SplitStream},
    SinkExt, StreamExt,
};
use http::{Request, Uri};
use modelsocket_common::{MSEvent, MSRequest, SeqGenReq, SeqOpenReq};
use std::{collections::HashMap, sync::Arc};
use thiserror::Error;
use tokio::{
    net::TcpStream,
    sync::{mpsc, Mutex},
};
use tokio_tungstenite::{
    connect_async,
    tungstenite::{self, protocol::Message, Error as WsError},
    MaybeTlsStream, WebSocketStream,
};

use tracing::{debug, error};
use uuid::Uuid;

mod seq;

pub use seq::{GenChunk, GenStream, Seq};

#[derive(Error, Debug)]
pub enum ModelSocketError {
    #[error("WebSocket error: {0}")]
    WebSocket(#[from] WsError),
    #[error("URL parsing error: {0}")]
    Url(#[from] url::ParseError),
    #[error("JSON serialization/deserialization error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("Protocol error: {0}")]
    Protocol(String),
    #[error("State error: {0}")]
    State(String),
    #[error("Open error: {0}")]
    Open(String),
    #[error("Send error: {0}")]
    Send(#[from] mpsc::error::SendError<Message>),
    #[error("Command error: {0}")]
    Command(String),
}

type WsSink = SplitSink<WebSocketStream<MaybeTlsStream<TcpStream>>, Message>;
type WsStream = SplitStream<WebSocketStream<MaybeTlsStream<TcpStream>>>;

#[derive(Clone)]
pub struct ModelSocket {
    ws_sink: Arc<Mutex<WsSink>>,
    opening_seqs: Arc<Mutex<HashMap<String, mpsc::Sender<Result<String, ModelSocketError>>>>>,
    seqs: Arc<Mutex<HashMap<String, Seq>>>,
}

impl ModelSocket {
    pub async fn connect(url: &str, api_key: Option<&str>) -> Result<Self, ModelSocketError> {
        let uri: Uri = url.parse().unwrap();

        let mut request_builder = Request::builder().uri(&uri);

        request_builder = request_builder
            .header(
                "Sec-WebSocket-Key",
                tungstenite::handshake::client::generate_key(),
            )
            .header("host", uri.host().unwrap())
            .header("upgrade", "websocket")
            .header("connection", "upgrade")
            .header("sec-websocket-version", 13);

        if let Some(key) = api_key {
            request_builder = request_builder.header("Authorization", format!("Bearer {key}"));
        }

        let request = request_builder
            .body(())
            .map_err(|e| ModelSocketError::Protocol(e.to_string()))?;

        let (ws_stream, _) = connect_async(request).await?;
        let (ws_sink, ws_stream) = ws_stream.split();

        let socket = Self {
            ws_sink: Arc::new(Mutex::new(ws_sink)),
            opening_seqs: Arc::new(Mutex::new(HashMap::new())),
            seqs: Arc::new(Mutex::new(HashMap::new())),
        };

        let socket_clone = socket.clone_components();

        tokio::spawn(async move {
            socket_clone.read_loop(ws_stream).await;
        });

        Ok(socket)
    }

    fn clone_components(&self) -> Self {
        Self {
            ws_sink: self.ws_sink.clone(),
            opening_seqs: self.opening_seqs.clone(),
            seqs: self.seqs.clone(),
        }
    }

    async fn read_loop(self, mut ws_stream: WsStream) {
        while let Some(msg) = ws_stream.next().await {
            match msg {
                Ok(Message::Text(text)) => {
                    debug!("<- {}", text);
                    self.on_message(&text).await;
                }
                Ok(_) => { /* Ignore other message types */ }
                Err(e) => {
                    error!("WebSocket read error: {}", e);
                    break;
                }
            }
        }
    }

    async fn on_message(&self, msg: &str) {
        match serde_json::from_str::<MSEvent>(msg) {
            Ok(event) => match event {
                MSEvent::SeqOpened { seq_id, cid } => self.on_seq_opened(seq_id, cid).await,
                MSEvent::Error { cid, message, .. } => self.on_error(cid, message).await,
                MSEvent::SeqClosed { cid, seq_id, .. } => self.on_seq_closed(cid, seq_id).await,
                _ => self.forward_to_seq(&event).await,
            },
            Err(e) => {
                error!("deserialization error: {}", e);
            }
        }
    }

    async fn forward_to_seq(&self, event: &MSEvent) {
        let seq_id = match event {
            MSEvent::SeqAppendFinish { seq_id, .. } => seq_id,
            MSEvent::SeqGenFinish { seq_id, .. } => seq_id,
            MSEvent::SeqForkFinish { seq_id, .. } => seq_id,
            MSEvent::SeqText { seq_id, .. } => seq_id,
            MSEvent::SeqToolCall { seq_id, .. } => seq_id,
            _ => {
                error!("unhandled event forwarded to seq: {:?}", event);
                return;
            }
        };

        let mut seqs = self.seqs.lock().await;

        if let Some(seq) = seqs.get_mut(seq_id) {
            // This part will be completed in the next steps
            seq.on_event(event).await;
        } else {
            error!("state error: unknown seq_id {}", seq_id);
        }
    }

    async fn on_error(&self, cid: Option<String>, message: String) {
        error!("error: {}", message);
        if let Some(cid) = cid {
            if let Some(sender) = self.opening_seqs.lock().await.remove(&cid) {
                let _ = sender
                    .send(Err(ModelSocketError::Open(format!(
                        "open error: {}",
                        message
                    ))))
                    .await;
            }
        }
    }

    async fn on_seq_closed(&self, _cid: Option<String>, seq_id: String) {
        if let Some(_seq) = self.seqs.lock().await.remove(&seq_id) {
            // This part will be completed in the next steps
            // seq.on_close().await;
        } else {
            error!("state error: unknown seq_id {}", seq_id);
        }
    }

    async fn on_seq_opened(&self, seq_id: String, cid: String) {
        let mut opening_seqs = self.opening_seqs.lock().await;
        if let Some(sender) = opening_seqs.remove(&cid) {
            if let Err(e) = sender.send(Ok(seq_id)).await {
                error!("Failed to send seq_id: {}", e);
            }
        } else {
            error!("unknown opened seq cid {}", cid);
        }
    }

    pub async fn open(&self, model: &str) -> Result<Seq, ModelSocketError> {
        let cid = Uuid::new_v4().to_string();
        let (tx, mut rx) = mpsc::channel(1);

        {
            let mut opening_seqs = self.opening_seqs.lock().await;
            opening_seqs.insert(cid.clone(), tx);
        }

        self.send(MSRequest::SeqOpen {
            cid: cid.clone(),
            data: SeqOpenReq {
                model: model.to_string(),
                tools_enabled: false,
                tool_prompt: None,
                skip_prelude: false,
            },
        })
        .await?;

        let seq_id = rx.recv().await.unwrap_or_else(|| {
            Err(ModelSocketError::Open(
                "Failed to receive seq_id".to_string(),
            ))
        })?;

        let seq = Seq::new(seq_id.clone(), model.to_string(), self.clone_components());

        {
            let mut seqs = self.seqs.lock().await;
            seqs.insert(seq_id, seq.clone());
        }

        Ok(seq)
    }

    async fn send(&self, req: MSRequest) -> Result<(), ModelSocketError> {
        let msg = serde_json::to_string(&req)?;
        debug!("-> {}", msg);
        let mut sink = self.ws_sink.lock().await;
        sink.send(Message::Text(msg)).await?;
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

mod seq;
pub mod tools;
pub mod transport;

use crate::{
    protocol::{MSEvent, MSRequest, SeqGenReq, SeqOpenReq},
    tools::Toolbox,
};
use futures::{Sink, Stream};
use futures_util::{SinkExt, StreamExt};
use seq::SeqState;
pub use seq::{EmbedOpts, EmbeddingResult, GenChunk, GenStream, Seq};
use std::{
    collections::{HashMap, HashSet},
    pin::Pin,
    sync::Arc,
};
use thiserror::Error;
use tokio::sync::{mpsc, Mutex, RwLock};
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

    #[error("seq closed")]
    SeqClosed,

    #[error("command error: {0}")]
    Command(String),

    #[error("{message}")]
    Remote {
        message: String,
        code: Option<String>,
        details: Option<serde_json::Map<String, serde_json::Value>>,
    },

    #[error("{0}")]
    Other(#[from] anyhow::Error),
}

#[derive(Clone)]
struct ModelSocketEventHandler {
    /// Seqs that are being opened by this socket
    opening_seqs: Arc<Mutex<HashMap<String, mpsc::Sender<Result<String, ModelSocketError>>>>>,

    /// Seqs being managed by this socket
    seqs: Arc<Mutex<HashMap<String, Arc<SeqState>>>>,

    /// Records a set of tombstoned seqs that have been closed
    closed_seqs: Arc<RwLock<HashSet<String>>>,
}

/// A connection kept alive by caller-owned sockets, sequences, and generation streams.
#[derive(Clone)]
pub struct ModelSocket {
    inner: Arc<Connection>,
}

/// Connection state shared by every `ModelSocket` handle.
struct Connection {
    /// Channel for sending requests to the model server
    ws_sink: Mutex<Pin<Box<dyn Sink<MSRequest, Error = ModelSocketError> + Send>>>,

    /// Seq tables, shared with the receive task
    handler: ModelSocketEventHandler,

    receive_task: tokio::task::AbortHandle,
}

impl Drop for Connection {
    fn drop(&mut self) {
        // The receive task only holds the seq tables, never the connection, so it
        // would otherwise run until the server hangs up.
        self.receive_task.abort();
    }
}

impl ModelSocket {
    pub fn new<RX, TX, T>(transport: T) -> Result<Self, ModelSocketError>
    where
        RX: Stream<Item = Result<MSEvent, ModelSocketError>> + Send + Unpin + 'static,
        TX: Sink<MSRequest, Error = ModelSocketError> + Send + 'static,
        T: transport::MSTransport<RX, TX>,
    {
        super::ensure_rustls_provider();

        let (ws_sink, ws_stream) = transport.split();

        let handler = ModelSocketEventHandler {
            opening_seqs: Default::default(),
            seqs: Default::default(),
            closed_seqs: Default::default(),
        };

        let receive_task = tokio::spawn(handler.clone().event_recv_loop(ws_stream)).abort_handle();

        Ok(Self {
            inner: Arc::new(Connection {
                ws_sink: Mutex::new(Box::pin(ws_sink)),
                handler,
                receive_task,
            }),
        })
    }

    pub async fn connect(url: &str, api_key: Option<&str>) -> Result<Self, ModelSocketError> {
        let ws_transport = transport::ws::WebSocketTransport::connect(url, api_key).await?;
        Self::new(ws_transport)
    }

    async fn registered_seq(&self, seq_id: &str) -> Option<Arc<SeqState>> {
        self.inner.handler.seqs.lock().await.get(seq_id).cloned()
    }
}

impl ModelSocketEventHandler {
    /// Receives events from the model server and forwards them to handler
    async fn event_recv_loop<S: Stream<Item = Result<MSEvent, ModelSocketError>> + Unpin>(
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

    /// Handles events received from the model
    async fn on_event(&self, event: MSEvent) {
        debug!("<- {:?}", event);
        match &event {
            MSEvent::SeqOpened { seq_id, cid } => self.on_seq_opened(seq_id, cid).await,
            MSEvent::Error {
                cid,
                seq_id,
                message,
                code,
                details,
            } => {
                self.on_error(cid, message, code, details).await;
                if seq_id.is_some() {
                    self.forward_to_seq(&event).await;
                }
            }
            MSEvent::SeqForkFinish {
                seq_id,
                child_seq_id,
                ..
            } => {
                // register the child before the parent's fork() call resumes
                self.on_seq_forked(seq_id, child_seq_id).await;
                self.forward_to_seq(&event).await
            }
            MSEvent::SeqClosed { seq_id, .. } => {
                // delegate to seq first to allow it to clean up
                self.forward_to_seq(&event).await;

                // then clean up the seq from the socket state and remove it
                // from seq table
                self.on_seq_closed(seq_id).await
            }
            _ => self.forward_to_seq(&event).await,
        }
    }

    async fn is_seq_closed(&self, seq_id: &str) -> bool {
        self.closed_seqs.read().await.contains(seq_id)
    }

    /// Forwards events from the model to the approriate seq handler
    async fn forward_to_seq(&self, event: &MSEvent) {
        let seq_id = match event.seq_id() {
            Some(id) => id,
            None => {
                error!("unroutable event forwarded to seq: {:?}", event);
                return;
            }
        };

        if self.is_seq_closed(seq_id).await {
            debug!(seq = seq_id, "skipping event for closed seq");
            return;
        }

        // release the table lock first so a slow event sink can't block open() or fork()
        let seq = self.seqs.lock().await.get(seq_id).cloned();
        if let Some(seq) = seq {
            seq.on_event(event).await;
        } else {
            error!(seq_id, "state error: unknown seq_id");
        }
    }

    async fn on_error(
        &self,
        cid: &Option<String>,
        message: &str,
        code: &Option<String>,
        details: &Option<serde_json::Map<String, serde_json::Value>>,
    ) {
        error!("error: {}", message);

        if let Some(cid) = cid {
            if let Some(sender) = self.opening_seqs.lock().await.remove(cid) {
                let _ = sender.send(Err(remote_error(message, code, details))).await;
            }
        }
    }

    async fn on_seq_forked(&self, parent_seq_id: &str, child_seq_id: &str) {
        let mut seqs = self.seqs.lock().await;
        if let Some(parent) = seqs.get(parent_seq_id) {
            let child = parent.fork_child(child_seq_id);
            seqs.insert(child_seq_id.to_owned(), child);
        }
    }

    async fn on_seq_closed(&self, seq_id: &str) {
        if let Some(_seq) = self.seqs.lock().await.remove(seq_id) {
            debug!(seq = seq_id, "removing seq from client socket state");
            self.closed_seqs.write().await.insert(seq_id.to_owned());
        } else {
            // if the seq was not in the seqs table and we don't have a tombstone for it,
            // then emit an error
            if !self.is_seq_closed(seq_id).await {
                error!(seq_id, "unknown seq id during on close event");
            }
        }
    }

    async fn on_seq_opened(&self, seq_id: &str, cid: &str) {
        let mut opening_seqs = self.opening_seqs.lock().await;
        if let Some(sender) = opening_seqs.remove(cid) {
            if let Err(e) = sender.send(Ok(seq_id.to_owned())).await {
                error!("failed to send seq_id: {}", e);
            }
        } else {
            error!(cid, "unknown opened seq cid");
        }
    }
}

fn remote_error(
    message: &str,
    code: &Option<String>,
    details: &Option<serde_json::Map<String, serde_json::Value>>,
) -> ModelSocketError {
    ModelSocketError::Remote {
        message: message.to_string(),
        code: code.clone(),
        details: details.clone(),
    }
}

impl ModelSocket {
    pub async fn open(&self, model: &str, opts: Option<OpenOpts>) -> Result<Seq, ModelSocketError> {
        let cid = Uuid::new_v4().to_string();
        let (tx, mut rx) = mpsc::channel(1);

        {
            let mut opening_seqs = self.inner.handler.opening_seqs.lock().await;
            opening_seqs.insert(cid.clone(), tx);
        }

        let opts = opts.unwrap_or_default();
        let tools_enabled = opts.toolbox.is_some();

        self.send_request(MSRequest::SeqOpen {
            cid: cid.clone(),
            data: SeqOpenReq {
                model: model.to_string(),
                tools_enabled,
                tool_prompt: opts.tool_prompt,
                tool_schemas: opts.tool_schemas,
                skip_prelude: opts.skip_prelude,
            },
        })
        .await?;

        let seq_id = rx.recv().await.unwrap_or_else(|| {
            Err(ModelSocketError::Open(
                "Failed to receive seq_id".to_string(),
            ))
        })?;

        let tool_def_prompt = opts.toolbox.as_ref().and_then(|t| t.tool_def_prompt());
        let toolbox = Arc::new(opts.toolbox.map(|t| Mutex::new(t)));

        let seq = Seq::new(
            seq_id.clone(),
            model.to_string(),
            self.clone(),
            toolbox,
            opts.event_sink,
        );

        self.inner
            .handler
            .seqs
            .lock()
            .await
            .insert(seq_id, seq.state.clone());

        // append tool definition prompts to the sequence
        if let Some(tool_def_prompt) = tool_def_prompt {
            seq.append(tool_def_prompt, AppendOpts::system()).await?;
        }

        Ok(seq)
    }

    async fn send_request(&self, req: MSRequest) -> Result<(), ModelSocketError> {
        debug!("-> {:?}", req);
        let mut sink = self.inner.ws_sink.lock().await;
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
    pub json_schema: Option<serde_json::Value>,
    pub json_schema_strict: Option<bool>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub max_emitted_tools: Option<u32>,
    pub reasoning: Option<crate::ReasoningConfig>,
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
            json_schema: self.json_schema,
            json_schema_strict: self.json_schema_strict,
            frequency_penalty: self.frequency_penalty,
            presence_penalty: self.presence_penalty,
            max_emitted_tools: self.max_emitted_tools,
            reasoning: self.reasoning,
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
    /// Optional system prompt to use when tools are enabled
    pub tool_prompt: Option<String>,

    /// Skips system prompt prescribed by model authors
    pub skip_prelude: bool,

    /// Optional map of tool names to JSON Schema fragments used for tool arg typing.
    pub tool_schemas: Option<HashMap<String, serde_json::Value>>,

    /// Optional sink for listening to all events related
    /// to this sequence
    pub event_sink: Option<mpsc::Sender<Result<MSEvent, ModelSocketError>>>,

    /// Tools available for use on this sequence
    pub toolbox: Option<Box<dyn Toolbox>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::task::{Context, Poll};
    use tokio::time::{timeout, Duration};

    struct ChannelTransport {
        requests: tokio::sync::mpsc::UnboundedSender<MSRequest>,
        events: tokio::sync::mpsc::UnboundedReceiver<Result<MSEvent, ModelSocketError>>,
    }

    struct RequestSink(tokio::sync::mpsc::UnboundedSender<MSRequest>);

    impl Sink<MSRequest> for RequestSink {
        type Error = ModelSocketError;

        fn poll_ready(
            self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
        ) -> Poll<Result<(), Self::Error>> {
            Poll::Ready(Ok(()))
        }

        fn start_send(self: Pin<&mut Self>, item: MSRequest) -> Result<(), Self::Error> {
            self.0
                .send(item)
                .map_err(|_| ModelSocketError::State("request receiver closed".to_string()))
        }

        fn poll_flush(
            self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
        ) -> Poll<Result<(), Self::Error>> {
            Poll::Ready(Ok(()))
        }

        fn poll_close(
            self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
        ) -> Poll<Result<(), Self::Error>> {
            Poll::Ready(Ok(()))
        }
    }

    struct EventStream(tokio::sync::mpsc::UnboundedReceiver<Result<MSEvent, ModelSocketError>>);

    impl Stream for EventStream {
        type Item = Result<MSEvent, ModelSocketError>;

        fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            self.0.poll_recv(cx)
        }
    }

    impl transport::MSTransport<EventStream, RequestSink> for ChannelTransport {
        fn split(self) -> (RequestSink, EventStream) {
            (RequestSink(self.requests), EventStream(self.events))
        }
    }

    #[tokio::test]
    async fn dropping_socket_closes_request_channel_while_event_stream_is_open() {
        let (request_tx, mut request_rx) = tokio::sync::mpsc::unbounded_channel();
        let (event_tx, event_rx) = tokio::sync::mpsc::unbounded_channel();

        let socket = ModelSocket::new(ChannelTransport {
            requests: request_tx,
            events: event_rx,
        })
        .unwrap();

        drop(socket);

        let closed = timeout(Duration::from_millis(100), request_rx.recv())
            .await
            .expect("request receiver should observe closure after dropping ModelSocket");

        assert!(closed.is_none());

        drop(event_tx);
    }

    pub(super) fn test_connection() -> (
        ModelSocket,
        mpsc::UnboundedReceiver<MSRequest>,
        mpsc::UnboundedSender<Result<MSEvent, ModelSocketError>>,
    ) {
        let (request_tx, request_rx) = mpsc::unbounded_channel();
        let (event_tx, event_rx) = mpsc::unbounded_channel();
        let socket = ModelSocket::new(ChannelTransport {
            requests: request_tx,
            events: event_rx,
        })
        .unwrap();
        (socket, request_rx, event_tx)
    }

    async fn open_test_sequence(
        socket: &ModelSocket,
        requests: &mut mpsc::UnboundedReceiver<MSRequest>,
        events: &mpsc::UnboundedSender<Result<MSEvent, ModelSocketError>>,
        opts: Option<OpenOpts>,
    ) -> Seq {
        let mut open = Box::pin(socket.open("model", opts));
        assert!(futures::poll!(&mut open).is_pending());
        let MSRequest::SeqOpen { cid, .. } = requests.recv().await.unwrap() else {
            panic!("expected sequence open");
        };
        events
            .send(Ok(MSEvent::SeqOpened {
                seq_id: "seq-1".into(),
                cid,
            }))
            .unwrap();
        timeout(Duration::from_secs(1), open)
            .await
            .unwrap()
            .unwrap()
    }

    async fn assert_connection_dropped(
        requests: &mut mpsc::UnboundedReceiver<MSRequest>,
        events: &mpsc::UnboundedSender<Result<MSEvent, ModelSocketError>>,
    ) {
        assert!(timeout(Duration::from_secs(1), requests.recv())
            .await
            .expect("last client handle must release the request sink without SeqClosed")
            .is_none());
        timeout(Duration::from_secs(1), events.closed())
            .await
            .expect("last client handle must stop the receive task");
    }

    #[tokio::test]
    async fn dropping_open_sequence_and_all_socket_clones_releases_transport() {
        let (socket, mut requests, events) = test_connection();
        let seq = open_test_sequence(&socket, &mut requests, &events, None).await;
        let socket_clone = socket.clone();
        let seq_clone = seq.clone();
        drop(socket);
        drop(seq);
        drop(socket_clone);

        // The final external sequence still owns a usable connection.
        seq_clone
            .send_cmd("gen", crate::SeqCommand::Gen(Default::default()))
            .await
            .unwrap();
        assert!(matches!(
            requests.recv().await,
            Some(MSRequest::SeqCommand { .. })
        ));
        drop(seq_clone);
        assert_connection_dropped(&mut requests, &events).await;
    }

    #[tokio::test]
    async fn cancelling_pending_open_releases_transport() {
        let (socket, mut requests, events) = test_connection();
        let mut open = Box::pin(async move { socket.open("model", None).await });
        assert!(futures::poll!(&mut open).is_pending());
        assert!(matches!(
            requests.recv().await,
            Some(MSRequest::SeqOpen { .. })
        ));
        drop(open);
        assert_connection_dropped(&mut requests, &events).await;
    }

    #[tokio::test]
    async fn cancelling_pending_prefill_releases_transport_without_close_ack() {
        let (socket, mut requests, events) = test_connection();
        let seq = open_test_sequence(&socket, &mut requests, &events, None).await;
        drop(socket);
        let mut prefill = Box::pin(async move { seq.append("prompt", AppendOpts::user()).await });
        assert!(futures::poll!(&mut prefill).is_pending());
        assert!(matches!(
            requests.recv().await,
            Some(MSRequest::SeqCommand {
                data: crate::SeqCommand::Append(_),
                ..
            })
        ));
        drop(prefill);
        assert_connection_dropped(&mut requests, &events).await;
    }

    #[tokio::test]
    async fn generation_stream_keeps_connection_alive_until_dropped() {
        let (socket, mut requests, events) = test_connection();
        let seq = open_test_sequence(&socket, &mut requests, &events, None).await;
        let mut stream = seq.generate::<SeqGenReq>(None).await.unwrap();
        let Some(MSRequest::SeqCommand { cid, .. }) = requests.recv().await else {
            panic!("expected generation command");
        };
        drop(seq);
        drop(socket);
        events
            .send(Ok(MSEvent::SeqText {
                seq_id: "seq-1".into(),
                cid,
                text: "hello".into(),
                hidden: false,
                reasoning: false,
                num_input_tokens: 1,
                num_output_tokens: 1,
                tokens: None,
            }))
            .unwrap();
        assert_eq!(
            timeout(Duration::from_secs(1), stream.next())
                .await
                .unwrap()
                .unwrap()
                .unwrap()
                .text,
            "hello"
        );
        drop(stream);
        assert_connection_dropped(&mut requests, &events).await;
    }

    #[tokio::test]
    async fn dropping_client_stops_receive_task_with_full_event_sink() {
        let (socket, mut requests, events) = test_connection();
        let (sink, mut receiver) = mpsc::channel(1);
        let event = MSEvent::SeqAppendFinish {
            seq_id: "seq-1".into(),
            cid: "append".into(),
        };
        sink.send(Ok(event.clone())).await.unwrap();
        let seq = open_test_sequence(
            &socket,
            &mut requests,
            &events,
            Some(OpenOpts {
                event_sink: Some(sink),
                ..Default::default()
            }),
        )
        .await;
        events.send(Ok(event)).unwrap();
        tokio::task::yield_now().await;
        drop(seq);
        drop(socket);
        assert_connection_dropped(&mut requests, &events).await;
        assert!(receiver.recv().await.is_some());
        assert!(receiver.recv().await.is_none());
    }

    #[tokio::test]
    async fn close_acknowledgement_removes_sequence_event_handler() {
        let (socket, mut requests, events) = test_connection();
        let (sink, mut receiver) = mpsc::channel(1);
        let seq = open_test_sequence(
            &socket,
            &mut requests,
            &events,
            Some(OpenOpts {
                event_sink: Some(sink),
                ..Default::default()
            }),
        )
        .await;
        seq.close().await.unwrap();
        let Some(MSRequest::SeqCommand {
            cid,
            data: crate::SeqCommand::Close(_),
            ..
        }) = requests.recv().await
        else {
            panic!("expected close request");
        };
        assert!(matches!(
            receiver.try_recv(),
            Err(mpsc::error::TryRecvError::Empty)
        ));
        events
            .send(Ok(MSEvent::SeqClosed {
                seq_id: seq.id().into(),
                cid: Some(cid),
                input_tokens: 0,
                cached_input_tokens: 0,
                output_tokens: 0,
                reasoning_tokens: None,
                duration_ms: 0,
                error: None,
            }))
            .unwrap();
        assert!(matches!(
            timeout(Duration::from_secs(1), receiver.recv())
                .await
                .unwrap(),
            Some(Ok(MSEvent::SeqClosed { .. }))
        ));
        assert!(timeout(Duration::from_secs(1), receiver.recv())
            .await
            .unwrap()
            .is_none());
        drop(seq);
        drop(socket);
        assert_connection_dropped(&mut requests, &events).await;
    }

    #[tokio::test]
    async fn forked_sequence_keeps_connection_alive_and_releases_transport_on_drop() {
        let (socket, mut requests, events) = test_connection();
        let parent = open_test_sequence(&socket, &mut requests, &events, None).await;
        let mut fork = Box::pin(parent.fork());
        assert!(futures::poll!(&mut fork).is_pending());
        let Some(MSRequest::SeqCommand { cid, .. }) = requests.recv().await else {
            panic!("expected fork command");
        };
        events
            .send(Ok(MSEvent::SeqForkFinish {
                seq_id: parent.id().into(),
                cid,
                child_seq_id: "child".into(),
            }))
            .unwrap();
        let child = timeout(Duration::from_secs(1), fork)
            .await
            .unwrap()
            .unwrap();
        drop(parent);
        drop(socket);
        child
            .send_cmd("gen", crate::SeqCommand::Gen(Default::default()))
            .await
            .unwrap();
        assert!(
            matches!(requests.recv().await, Some(MSRequest::SeqCommand { seq_id, .. }) if seq_id == "child")
        );
        drop(child);
        assert_connection_dropped(&mut requests, &events).await;
    }

    #[tokio::test]
    async fn open_error_preserves_remote_details() {
        let (request_tx, mut request_rx) = tokio::sync::mpsc::unbounded_channel();
        let (event_tx, event_rx) = tokio::sync::mpsc::unbounded_channel();
        let socket = ModelSocket::new(ChannelTransport {
            requests: request_tx,
            events: event_rx,
        })
        .unwrap();
        let open = tokio::spawn(async move { socket.open("model", None).await });
        let MSRequest::SeqOpen { cid, .. } = request_rx.recv().await.unwrap() else {
            panic!("expected sequence open");
        };
        let details = serde_json::json!({
            "rpm_limit": 60,
            "rpm_remaining": 0,
            "tpm_limit": 100_000,
            "tpm_remaining": 0,
            "retry_after_ms": 1_000
        })
        .as_object()
        .unwrap()
        .clone();
        event_tx
            .send(Ok(MSEvent::Error {
                cid: Some(cid),
                seq_id: Some("seq-1".into()),
                message: "Rate limit exceeded".into(),
                code: Some("rate_limit_exceeded".into()),
                details: Some(details.clone()),
            }))
            .unwrap();

        match open.await.unwrap() {
            Err(ModelSocketError::Remote {
                code,
                details: actual,
                ..
            }) => {
                assert_eq!(code.as_deref(), Some("rate_limit_exceeded"));
                assert_eq!(actual, Some(details));
            }
            _ => panic!("expected structured remote error"),
        }
    }
}

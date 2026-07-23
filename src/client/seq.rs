use crate::{
    protocol::{
        EmbeddingInput, MSEvent, MSRequest, SeqAppendMedia, SeqAppendReq, SeqCloseReq, SeqCommand,
        SeqEmbedReq, SeqForkReq, SeqGenReq,
    },
    tools::Toolbox,
    SeqToolCall, SeqToolReturnReq,
};
use futures::Stream;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};
use tokio::sync::{mpsc, Mutex};
use tracing::{debug, error, warn};
use uuid::Uuid;

use crate::{AppendOpts, ModelSocket, ModelSocketError};

#[derive(Clone)]
pub struct Seq {
    /// id for this seq
    seq_id: String,

    /// model this seq is associated with
    model: String,

    /// modelsocket this seq is associated with
    socket: ModelSocket,

    /// open commands waiting for a response and their return channels
    cmds: Arc<Mutex<HashMap<String, mpsc::Sender<Result<serde_json::Value, ModelSocketError>>>>>,

    /// open gen streams waiting for a response and their return channels
    gen_streams: Arc<Mutex<HashMap<String, mpsc::Sender<Result<GenChunk, ModelSocketError>>>>>,

    /// open embedding commands waiting for a complete batch response
    embed_cmds: Arc<Mutex<HashMap<String, EmbedCommandState>>>,

    /// tools active on this seq
    toolbox: Arc<Option<Mutex<Box<dyn Toolbox>>>>,

    // channel for forwarding model events to the client
    pub(crate) event_tx: Option<mpsc::Sender<Result<MSEvent, ModelSocketError>>>,
}

impl Seq {
    pub(crate) fn new(
        seq_id: String,
        model: String,
        socket: ModelSocket,
        toolbox: Arc<Option<Mutex<Box<dyn Toolbox>>>>,
        event_tx: Option<mpsc::Sender<Result<MSEvent, ModelSocketError>>>,
    ) -> Self {
        Self {
            seq_id,
            model,
            socket,
            cmds: Arc::new(Mutex::new(HashMap::new())),
            gen_streams: Arc::new(Mutex::new(HashMap::new())),
            embed_cmds: Arc::new(Mutex::new(HashMap::new())),
            event_tx,
            toolbox,
        }
    }

    pub fn id(&self) -> &str {
        &self.seq_id
    }

    pub async fn on_event(&mut self, event: &MSEvent) {
        if let Some(tx) = &self.event_tx {
            if let Err(_e) = tx.send(Ok(event.clone())).await {
                debug!("seq event sink closed");
                self.event_tx = None;
            }
        }

        match event {
            MSEvent::SeqAppendFinish { cid, .. } => self.on_append_finished(cid).await,
            MSEvent::SeqGenFinish { cid, .. } => self.on_gen_finished(cid).await,
            MSEvent::SeqEmbedding {
                cid,
                index,
                embedding,
                input_tokens,
                ..
            } => {
                self.on_embedding(cid, *index, embedding.clone(), *input_tokens)
                    .await
            }
            MSEvent::SeqEmbedFinish {
                cid, prompt_tokens, ..
            } => self.on_embed_finished(cid, *prompt_tokens).await,
            MSEvent::SeqForkFinish {
                cid, child_seq_id, ..
            } => self.on_fork_finished(cid, child_seq_id).await,
            MSEvent::SeqText {
                cid,
                text,
                hidden,
                tokens,
                ..
            } => {
                self.on_text(cid, text.clone(), *hidden, tokens.clone())
                    .await
            }
            MSEvent::SeqClosed { cid, .. } => {
                self.on_close_event(cid).await;
            }
            MSEvent::SeqToolCall {
                cid, tool_calls, ..
            } => self.on_tool_call(cid, tool_calls).await,
            MSEvent::SeqToolCallStart { .. }
            | MSEvent::SeqToolCallArgsChunk { .. }
            | MSEvent::SeqToolCallEnd { .. }
            | MSEvent::SeqToolCallAborted { .. }
            | MSEvent::SeqState { .. } => {}
            MSEvent::Error {
                cid,
                message,
                code,
                details,
                ..
            } => {
                if let Some(cid) = cid {
                    self.on_command_error(cid, message, code, details).await;
                } else {
                    self.on_seq_error(message, code, details).await;
                }
            }
            _ => {
                warn!("unhandled event in seq: {:?}", event);
            }
        }
    }

    /// handle when the server has closed the seq
    async fn on_close_event(&mut self, cid: &Option<String>) {
        if let Some(cid) = cid {
            if let Some(sender) = self.cmds.lock().await.remove(cid) {
                let _ = sender.send(Ok(serde_json::Value::Null)).await;
            }
        }

        self.fail_cmds_with_close_error().await;
        self.event_tx = None;
    }

    async fn on_text(&mut self, cid: &str, text: String, hidden: bool, tokens: Option<Vec<u32>>) {
        let mut gen_streams = self.gen_streams.lock().await;
        if let Some(sender) = gen_streams.get_mut(cid) {
            let chunk = GenChunk {
                text,
                hidden,
                tokens,
            };

            if sender.send(Ok(chunk)).await.is_err() {
                // Stream closed, remove it
                gen_streams.remove(cid);
            }
        }
    }

    async fn on_gen_finished(&mut self, cid: &str) {
        if let Some(sender) = self.cmds.lock().await.remove(cid) {
            let _ = sender.send(Ok(serde_json::Value::Null)).await;
        }
        if let Some(_stream) = self.gen_streams.lock().await.remove(cid) {
            // The stream is automatically closed when the sender is dropped.
        }
    }

    async fn on_embedding(
        &mut self,
        cid: &str,
        index: u32,
        embedding: Vec<f32>,
        input_tokens: u32,
    ) {
        let mut embed_cmds = self.embed_cmds.lock().await;
        let Some(state) = embed_cmds.get_mut(cid) else {
            warn!("received embedding for unknown cid: {}", cid);
            return;
        };

        let index = index as usize;
        if index >= state.embeddings.len() {
            warn!(
                cid,
                index,
                len = state.embeddings.len(),
                "received embedding index out of bounds"
            );
            return;
        }

        state.embeddings[index] = Some(embedding);
        state.input_tokens[index] = input_tokens;
    }

    async fn on_embed_finished(&mut self, cid: &str, prompt_tokens: u32) {
        let Some(state) = self.embed_cmds.lock().await.remove(cid) else {
            warn!("received embed finish for unknown cid: {}", cid);
            return;
        };

        let mut embeddings = Vec::with_capacity(state.embeddings.len());
        for (idx, embedding) in state.embeddings.into_iter().enumerate() {
            let Some(embedding) = embedding else {
                let _ = state
                    .sender
                    .send(Err(ModelSocketError::Protocol(format!(
                        "missing embedding result for index {}",
                        idx
                    ))))
                    .await;
                return;
            };
            embeddings.push(embedding);
        }

        let _ = state
            .sender
            .send(Ok(EmbeddingResult {
                embeddings,
                input_tokens: state.input_tokens,
                prompt_tokens,
            }))
            .await;
    }

    async fn on_append_finished(&mut self, cid: &str) {
        if let Some(sender) = self.cmds.lock().await.remove(cid) {
            let _ = sender.send(Ok(serde_json::Value::Null)).await;
        }
    }

    async fn on_command_error(
        &mut self,
        cid: &str,
        message: &str,
        code: &Option<String>,
        details: &Option<serde_json::Map<String, serde_json::Value>>,
    ) {
        let err = || super::remote_error(message, code, details);

        let cmd = self.cmds.lock().await.remove(cid);
        let gen_stream = self.gen_streams.lock().await.remove(cid);
        let embed_cmd = self.embed_cmds.lock().await.remove(cid);

        if let Some(sender) = cmd {
            let _ = sender.send(Err(err())).await;
        }

        if let Some(sender) = gen_stream {
            let _ = sender.send(Err(err())).await;
        }

        if let Some(state) = embed_cmd {
            let _ = state.sender.send(Err(err())).await;
        }
    }

    async fn on_seq_error(
        &mut self,
        message: &str,
        code: &Option<String>,
        details: &Option<serde_json::Map<String, serde_json::Value>>,
    ) {
        let err = || super::remote_error(message, code, details);

        let cmds = std::mem::take(&mut *self.cmds.lock().await);
        let gen_streams = std::mem::take(&mut *self.gen_streams.lock().await);
        let embed_cmds = std::mem::take(&mut *self.embed_cmds.lock().await);

        for (_cid, tx) in cmds {
            let _ = tx.send(Err(err())).await;
        }

        for (_cid, tx) in gen_streams {
            let _ = tx.send(Err(err())).await;
        }

        for (_cid, state) in embed_cmds {
            let _ = state.sender.send(Err(err())).await;
        }
    }

    async fn on_fork_finished(&mut self, cid: &str, child_seq_id: &str) {
        if let Some(sender) = self.cmds.lock().await.remove(cid) {
            let child_seq = Seq::new(
                child_seq_id.to_string(),
                self.model.clone(),
                self.socket.clone(),
                self.toolbox.clone(),
                None,
            );
            self.socket
                .seqs
                .lock()
                .await
                .insert(child_seq_id.to_string(), child_seq.clone());

            let child_seq_json = serde_json::to_value(child_seq_id).unwrap();
            let _ = sender.send(Ok(child_seq_json)).await;
        }
    }

    async fn on_tool_call(&mut self, cid: &str, tool_calls: &Vec<SeqToolCall>) {
        let Some(toolbox) = self.toolbox.as_ref() else {
            debug!(
                seq_id = self.seq_id,
                "tool call requested but tools disabled"
            );

            return;
        };

        let results = toolbox.lock().await.call_tools(tool_calls).await;

        match results {
            Ok(Some(results)) => {
                let tool_return_cmd = MSRequest::SeqCommand {
                    cid: cid.to_string(),
                    seq_id: self.seq_id.clone(),
                    data: SeqCommand::ToolReturn(SeqToolReturnReq {
                        results,
                        gen_opts: Default::default(), //TODO resume generating with the same options
                    }),
                };

                if let Err(err) = self.socket.send_request(tool_return_cmd).await {
                    error!("failed to send tool return response: {}", err);
                }
            }
            Ok(None) => {} // toolbox returned None, skip tool return command
            Err(e) => {
                error!("failed to call tools: {}", e);
                if let Err(err) = self.close().await {
                    error!("failed to close seq after tool call error: {}", err);
                }
            }
        }
    }

    /// A low-level method for sending raw commands over the ModelSocket connection.
    /// You probably don't need this, use the higher-level methods like `append`, `generate`, `fork`, and `close`.
    pub async fn send_cmd<S: AsRef<str>>(
        &self,
        cid: S,
        cmd: SeqCommand,
    ) -> Result<(), ModelSocketError> {
        self.socket
            .send_request(MSRequest::SeqCommand {
                cid: cid.as_ref().to_string(),
                seq_id: self.seq_id.clone(),
                data: cmd,
            })
            .await?;

        Ok(())
    }

    pub async fn append(
        &self,
        text: impl AsRef<str>,
        opts: AppendOpts,
    ) -> Result<(), ModelSocketError> {
        let cid = Uuid::new_v4().to_string();
        let (tx, mut rx) = mpsc::channel(1);
        self.cmds.lock().await.insert(cid.clone(), tx);

        self.socket
            .send_request(MSRequest::SeqCommand {
                cid: cid.clone(),
                seq_id: self.seq_id.clone(),
                data: SeqCommand::Append(SeqAppendReq {
                    text: text.as_ref().to_string(),
                    role: opts.role,
                    ..Default::default()
                }),
            })
            .await?;

        rx.recv()
            .await
            .ok_or_else(|| ModelSocketError::Command("failed to receive response".into()))??;

        Ok(())
    }

    pub async fn append_media(
        &self,
        media: SeqAppendMedia,
        opts: AppendOpts,
    ) -> Result<(), ModelSocketError> {
        let cid = Uuid::new_v4().to_string();
        let (tx, mut rx) = mpsc::channel(1);
        self.cmds.lock().await.insert(cid.clone(), tx);

        self.socket
            .send_request(MSRequest::SeqCommand {
                cid: cid.clone(),
                seq_id: self.seq_id.clone(),
                data: SeqCommand::Append(SeqAppendReq {
                    media: Some(media),
                    role: opts.role,
                    ..Default::default()
                }),
            })
            .await?;

        rx.recv()
            .await
            .ok_or_else(|| ModelSocketError::Command("failed to receive response".into()))??;

        Ok(())
    }

    pub async fn generate<O: Into<SeqGenReq>>(
        &self,
        opts: Option<O>,
    ) -> Result<GenStream, ModelSocketError> {
        let cid = Uuid::new_v4().to_string();
        let (cmd_tx, mut cmd_rx) = mpsc::channel(1);
        self.cmds.lock().await.insert(cid.clone(), cmd_tx);

        let (stream_tx, stream_rx) = mpsc::channel(100);
        self.gen_streams.lock().await.insert(cid.clone(), stream_tx);

        self.socket
            .send_request(MSRequest::SeqCommand {
                cid: cid.clone(),
                seq_id: self.seq_id.clone(),
                data: SeqCommand::Gen(opts.map(|o| o.into()).unwrap_or_default()),
            })
            .await?;

        // The command will complete when the generation is finished. We'll spawn a task to wait for it.
        tokio::spawn(async move {
            let _ = cmd_rx.recv().await;
        });

        Ok(GenStream { stream: stream_rx })
    }

    pub async fn fork(&self) -> Result<Seq, ModelSocketError> {
        let cid = Uuid::new_v4().to_string();
        let (tx, mut rx) = mpsc::channel(1);
        self.cmds.lock().await.insert(cid.clone(), tx);

        self.socket
            .send_request(MSRequest::SeqCommand {
                cid: cid.clone(),
                seq_id: self.seq_id.clone(),
                data: SeqCommand::Fork(SeqForkReq {}),
            })
            .await?;

        let child_seq_id_val = rx
            .recv()
            .await
            .ok_or_else(|| ModelSocketError::Command("failed to receive response".into()))??;

        let child_seq_id = child_seq_id_val.as_str().unwrap().to_string();

        let child_seq = self
            .socket
            .seqs
            .lock()
            .await
            .get(&child_seq_id)
            .ok_or_else(|| ModelSocketError::State("child seq not found".into()))?
            .clone();

        Ok(child_seq)
    }

    pub async fn embed(
        &self,
        inputs: Vec<EmbeddingInput>,
        opts: Option<EmbedOpts>,
    ) -> Result<EmbeddingResult, ModelSocketError> {
        let cid = Uuid::new_v4().to_string();
        let (tx, mut rx) = mpsc::channel(1);
        let opts = opts.unwrap_or_default();

        self.embed_cmds.lock().await.insert(
            cid.clone(),
            EmbedCommandState {
                embeddings: vec![None; inputs.len()],
                input_tokens: vec![0; inputs.len()],
                sender: tx,
            },
        );

        self.socket
            .send_request(MSRequest::SeqCommand {
                cid,
                seq_id: self.seq_id.clone(),
                data: SeqCommand::Embed(SeqEmbedReq {
                    inputs,
                    input_type: opts.input_type,
                    dimensions: opts.dimensions,
                    normalize: opts.normalize,
                    truncate: opts.truncate,
                }),
            })
            .await?;

        rx.recv().await.ok_or_else(|| {
            ModelSocketError::Command("failed to receive embedding response".into())
        })?
    }

    /// sends an error to any outstanding commands or streams on this seq
    async fn fail_cmds_with_close_error(&self) {
        let mut cmds = self.cmds.lock().await;
        // send an error to any outstanding commands on this seqs
        for (_cid, tx) in cmds.drain() {
            let _ = tx.send(Err(ModelSocketError::SeqClosed)).await;
        }

        let mut gen_streams = self.gen_streams.lock().await;
        for (_cid, tx) in gen_streams.drain() {
            let _ = tx.send(Err(ModelSocketError::SeqClosed)).await;
        }

        let mut embed_cmds = self.embed_cmds.lock().await;
        for (_cid, state) in embed_cmds.drain() {
            let _ = state.sender.send(Err(ModelSocketError::SeqClosed)).await;
        }
    }

    pub async fn close(&self) -> Result<(), ModelSocketError> {
        let cid = Uuid::new_v4().to_string();

        // if the client has requested a close and there are outstanding
        // commands or streams, send an error to them
        self.fail_cmds_with_close_error().await;

        // fire and forget a close command
        self.socket
            .send_request(MSRequest::SeqCommand {
                cid: cid.clone(),
                seq_id: self.seq_id.clone(),
                data: SeqCommand::Close(SeqCloseReq {}),
            })
            .await?;

        Ok(())
    }
}

#[derive(Debug, Default, Clone)]
pub struct EmbedOpts {
    pub input_type: Option<String>,
    pub dimensions: Option<u32>,
    pub normalize: Option<bool>,
    pub truncate: Option<bool>,
}

pub struct GenStream {
    stream: mpsc::Receiver<Result<GenChunk, ModelSocketError>>,
}

impl Stream for GenStream {
    type Item = Result<GenChunk, ModelSocketError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.stream.poll_recv(cx)
    }
}

impl GenStream {
    pub async fn text(mut self) -> Result<String, ModelSocketError> {
        let mut result = String::new();
        while let Some(chunk) = self.stream.recv().await {
            let chunk = chunk?;
            if !chunk.hidden {
                result.push_str(&chunk.text);
            }
        }
        Ok(result)
    }

    pub async fn text_and_tokens(mut self) -> Result<(String, Vec<u32>), ModelSocketError> {
        let mut text = String::new();
        let mut tokens = Vec::new();
        while let Some(chunk) = self.stream.recv().await {
            let chunk = chunk?;
            if !chunk.hidden {
                text.push_str(&chunk.text);
                if let Some(chunk_tokens) = chunk.tokens {
                    tokens.extend(chunk_tokens);
                }
            }
        }

        Ok((text, tokens))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenChunk {
    pub text: String,
    pub hidden: bool,
    pub tokens: Option<Vec<u32>>,
}

struct EmbedCommandState {
    embeddings: Vec<Option<Vec<f32>>>,
    input_tokens: Vec<u32>,
    sender: mpsc::Sender<Result<EmbeddingResult, ModelSocketError>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingResult {
    pub embeddings: Vec<Vec<f32>>,
    pub input_tokens: Vec<u32>,
    pub prompt_tokens: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::Sink;
    use std::task::{Context, Poll};
    use tokio::time::{timeout, Duration};

    struct RequestSink;

    impl Sink<MSRequest> for RequestSink {
        type Error = ModelSocketError;

        fn poll_ready(
            self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
        ) -> Poll<Result<(), Self::Error>> {
            Poll::Ready(Ok(()))
        }

        fn start_send(self: Pin<&mut Self>, _item: MSRequest) -> Result<(), Self::Error> {
            Ok(())
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

    fn test_socket() -> ModelSocket {
        let ws_sink: Pin<Box<dyn Sink<MSRequest, Error = ModelSocketError> + Send>> =
            Box::pin(RequestSink);

        ModelSocket {
            ws_sink: Arc::new(Mutex::new(ws_sink)),
            opening_seqs: Default::default(),
            seqs: Default::default(),
            closed_seqs: Default::default(),
        }
    }

    fn assert_remote_error<T>(result: Result<T, ModelSocketError>, message: &str) {
        match result {
            Err(ModelSocketError::Remote {
                message: error,
                code: None,
                details: None,
            }) => assert_eq!(error, message),
            Err(error) => panic!("expected remote error, got {error:?}"),
            Ok(_) => panic!("expected remote error, got ok"),
        }
    }

    #[tokio::test]
    async fn seq_level_error_fails_all_pending_commands_and_streams() {
        let mut seq = Seq::new(
            "seq-1".to_string(),
            "model".to_string(),
            test_socket(),
            Arc::new(None),
            None,
        );

        let (cmd_tx, mut cmd_rx) = mpsc::channel(1);
        seq.cmds.lock().await.insert("cmd-1".to_string(), cmd_tx);

        let (gen_tx, mut gen_rx) = mpsc::channel(1);
        seq.gen_streams
            .lock()
            .await
            .insert("gen-1".to_string(), gen_tx);

        let (embed_tx, mut embed_rx) = mpsc::channel(1);
        seq.embed_cmds.lock().await.insert(
            "embed-1".to_string(),
            EmbedCommandState {
                embeddings: vec![None],
                input_tokens: vec![0],
                sender: embed_tx,
            },
        );

        let message = "seq failed";
        seq.on_event(&MSEvent::Error {
            cid: None,
            seq_id: Some("seq-1".to_string()),
            message: message.to_string(),
            code: None,
            details: None,
        })
        .await;

        assert_remote_error(
            timeout(Duration::from_millis(100), cmd_rx.recv())
                .await
                .expect("cmd error should be sent")
                .expect("cmd channel should remain open"),
            message,
        );
        assert_remote_error(
            timeout(Duration::from_millis(100), gen_rx.recv())
                .await
                .expect("gen error should be sent")
                .expect("gen channel should remain open"),
            message,
        );
        assert_remote_error(
            timeout(Duration::from_millis(100), embed_rx.recv())
                .await
                .expect("embed error should be sent")
                .expect("embed channel should remain open"),
            message,
        );

        assert!(seq.cmds.lock().await.is_empty());
        assert!(seq.gen_streams.lock().await.is_empty());
        assert!(seq.embed_cmds.lock().await.is_empty());
    }

    #[tokio::test]
    async fn command_error_preserves_remote_details() {
        let mut seq = Seq::new(
            "seq-1".to_string(),
            "model".to_string(),
            test_socket(),
            Arc::new(None),
            None,
        );
        let (tx, mut rx) = mpsc::channel(1);
        seq.cmds.lock().await.insert("gen".to_string(), tx);
        let details = serde_json::json!({
            "rpm_limit": 60,
            "rpm_remaining": 0,
            "tpm_limit": 100_000,
            "tpm_remaining": -1,
            "retry_after_ms": 1_000
        })
        .as_object()
        .unwrap()
        .clone();

        seq.on_event(&MSEvent::Error {
            cid: Some("gen".into()),
            seq_id: Some("seq-1".into()),
            message: "Rate limit exceeded".into(),
            code: Some("rate_limit_exceeded".into()),
            details: Some(details.clone()),
        })
        .await;

        match rx.recv().await.unwrap() {
            Err(ModelSocketError::Remote {
                code,
                details: actual,
                ..
            }) => {
                assert_eq!(code.as_deref(), Some("rate_limit_exceeded"));
                assert_eq!(actual, Some(details));
            }
            other => panic!("expected structured remote error, got {other:?}"),
        }
    }
}

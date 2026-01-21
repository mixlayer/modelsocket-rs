use crate::{
    protocol::{MSEvent, MSRequest, SeqAppendReq, SeqCloseReq, SeqCommand, SeqForkReq, SeqGenReq},
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

    /// tools active on this seq
    toolbox: Arc<Option<Mutex<Toolbox>>>,

    // channel for forwarding model events to the client
    pub(crate) event_tx: Option<mpsc::Sender<Result<MSEvent, ModelSocketError>>>,
}

impl Seq {
    pub(crate) fn new(
        seq_id: String,
        model: String,
        socket: ModelSocket,
        toolbox: Arc<Option<Mutex<Toolbox>>>,
        event_tx: Option<mpsc::Sender<Result<MSEvent, ModelSocketError>>>,
    ) -> Self {
        Self {
            seq_id,
            model,
            socket,
            cmds: Arc::new(Mutex::new(HashMap::new())),
            gen_streams: Arc::new(Mutex::new(HashMap::new())),
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

    async fn on_append_finished(&mut self, cid: &str) {
        if let Some(sender) = self.cmds.lock().await.remove(cid) {
            let _ = sender.send(Ok(serde_json::Value::Null)).await;
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
            Ok(results) => {
                let response = MSRequest::SeqCommand {
                    cid: cid.to_string(),
                    seq_id: self.seq_id.clone(),
                    data: SeqCommand::ToolReturn(SeqToolReturnReq {
                        results,
                        gen_opts: Default::default(), //TODO resume generating with the same options
                    }),
                };

                if let Err(err) = self.socket.send_request(response).await {
                    error!("failed to send tool return response: {}", err);
                }
            }
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

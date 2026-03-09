use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// A ModelSocket command.
///
/// All the commands that can be sent to a modelsocket server
/// from a client.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "request")]
#[serde(rename_all = "snake_case")]
pub enum MSRequest {
    SeqOpen {
        cid: String,
        data: SeqOpenReq,
    },
    SeqCommand {
        cid: String,
        seq_id: String,
        data: SeqCommand,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "command")]
#[serde(rename_all = "snake_case")]
pub enum SeqCommand {
    Close(SeqCloseReq),
    Append(SeqAppendReq),
    Gen(SeqGenReq),
    ToolReturn(SeqToolReturnReq),
    Fork(SeqForkReq),
}

/// A ModelSocket event.
///
/// All the events that can be sent to a model socket client
/// from a server.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[serde(tag = "event")]
pub enum MSEvent {
    SeqOpened {
        seq_id: String,
        cid: String,
    },
    SeqAppendFinish {
        seq_id: String,
        cid: String,
    },
    SeqGenFinish {
        seq_id: String,
        cid: String,
    },
    SeqForkFinish {
        seq_id: String,
        cid: String,
        child_seq_id: String,
    },
    SeqText {
        seq_id: String,
        cid: String,
        text: String,
        hidden: bool,
        num_input_tokens: u32,
        num_output_tokens: u32,
        tokens: Option<Vec<u32>>,
    },
    SeqToolCall {
        seq_id: String,
        cid: String,
        tool_calls: Vec<SeqToolCall>,
    },
    SeqState {
        seq_id: String,
        state: SeqState,
    },
    SeqClosed {
        cid: Option<String>,
        seq_id: String,
        input_tokens: u32,
        output_tokens: u32,
        duration_ms: u64,
        error: Option<String>,
    },
    Error {
        #[serde(skip_serializing_if = "Option::is_none")]
        cid: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        seq_id: Option<String>,
        message: String,
    },
}

impl MSEvent {
    pub fn cid(&self) -> Option<&str> {
        match self {
            MSEvent::SeqOpened { cid, .. } => Some(cid.as_str()),
            MSEvent::SeqAppendFinish { cid, .. } => Some(cid.as_str()),
            MSEvent::SeqGenFinish { cid, .. } => Some(cid.as_str()),
            MSEvent::SeqForkFinish { cid, .. } => Some(cid.as_str()),
            MSEvent::SeqText { cid, .. } => Some(cid.as_str()),
            MSEvent::SeqToolCall { cid, .. } => Some(cid.as_str()),
            MSEvent::SeqClosed { cid, .. } => cid.as_ref().map(|s| s.as_str()),
            MSEvent::Error { cid, .. } => cid.as_ref().map(|s| s.as_str()),
            _ => None,
        }
    }

    pub fn event_type(&self) -> &'static str {
        match self {
            MSEvent::SeqOpened { .. } => "seq_opened",
            MSEvent::SeqAppendFinish { .. } => "seq_append_finish",
            MSEvent::SeqGenFinish { .. } => "seq_gen_finish",
            MSEvent::SeqForkFinish { .. } => "seq_fork_finish",
            MSEvent::SeqText { .. } => "seq_text",
            MSEvent::SeqToolCall { .. } => "seq_tool_call",
            MSEvent::SeqClosed { .. } => "seq_closed",
            MSEvent::SeqState { .. } => "seq_state",
            MSEvent::Error { .. } => "error",
        }
    }

    pub fn seq_id(&self) -> Option<&str> {
        match self {
            MSEvent::SeqOpened { seq_id, .. } => Some(seq_id.as_str()),
            MSEvent::SeqAppendFinish { seq_id, .. } => Some(seq_id.as_str()),
            MSEvent::SeqGenFinish { seq_id, .. } => Some(seq_id.as_str()),
            MSEvent::SeqForkFinish { seq_id, .. } => Some(seq_id.as_str()),
            MSEvent::SeqText { seq_id, .. } => Some(seq_id.as_str()),
            MSEvent::SeqToolCall { seq_id, .. } => Some(seq_id.as_str()),
            MSEvent::SeqState { seq_id, .. } => Some(seq_id.as_str()),
            MSEvent::SeqClosed { seq_id, .. } => Some(seq_id.as_str()),
            MSEvent::Error { seq_id, .. } => seq_id.as_ref().map(|s| s.as_str()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeqOpenReq {
    pub model: String,

    #[serde(default)]
    pub tools_enabled: bool,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_prompt: Option<String>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_schemas: Option<HashMap<String, serde_json::Value>>,

    #[serde(default)]
    pub skip_prelude: bool,
}

#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct SeqAppendReq {
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokens: Option<Vec<u32>>,
    #[serde(default)]
    pub hidden: bool,
    #[serde(default)]
    pub echo: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
}

/// Sequence capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SeqCaps {
    Fork,
    Regex,
    ToolCall,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SeqState {
    /// A sequence is initialized and ready to append or generate
    Ready,

    /// A sequence is currently prefilling text
    Appending,

    /// A sequence is currently generating text
    Generating,

    /// A sequence has requested a tool call and is
    /// waiting for a response
    ToolCall,

    /// A sequence is currently forking
    Forking,

    /// A sequence is closed
    Closed,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct SeqGenReq {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_strings: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_length: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub regex_mask: Option<String>,
    #[serde(default)]
    pub hidden: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repeat_penalty: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub return_tokens: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefill_text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub name: String,
    pub result: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeqForkReq {}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeqCloseReq {}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeqToolReturnReq {
    pub results: Vec<ToolResult>,
    pub gen_opts: SeqGenReq,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeqToolCall {
    pub name: String,
    pub args: String,
}

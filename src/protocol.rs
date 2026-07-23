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
    Embed(SeqEmbedReq),
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
#[non_exhaustive]
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
    SeqEmbedding {
        seq_id: String,
        cid: String,
        index: u32,
        embedding: Vec<f32>,
        input_tokens: u32,
    },
    SeqEmbedFinish {
        seq_id: String,
        cid: String,
        prompt_tokens: u32,
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
    SeqToolCallStart {
        seq_id: String,
        cid: String,
        index: u32,
        name: String,
        id: Option<String>,
    },
    SeqToolCallArgsChunk {
        seq_id: String,
        cid: String,
        index: u32,
        fragment: String,
        id: Option<String>,
    },
    SeqToolCallEnd {
        seq_id: String,
        cid: String,
        index: u32,
        args: String,
    },
    SeqToolCallAborted {
        seq_id: String,
        cid: String,
        index: u32,
        reason: String,
        id: Option<String>,
    },
    SeqState {
        seq_id: String,
        state: SeqState,
    },
    SeqClosed {
        cid: Option<String>,
        seq_id: String,
        input_tokens: u32,
        #[serde(default)]
        cached_input_tokens: u32,
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
        #[serde(default, skip_serializing_if = "Option::is_none")]
        code: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        rate_limit: Option<RateLimitErrorDetails>,
    },
}

/// Structured rate-limit state attached to a ModelSocket error.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RateLimitErrorDetails {
    pub cause: String,
    pub enforcement_mode: String,
    pub rpm_limit: u32,
    pub rpm_remaining: i64,
    pub tpm_limit: u32,
    pub tpm_remaining: i64,
    pub retry_after_ms: u64,
}

impl MSEvent {
    pub fn cid(&self) -> Option<&str> {
        match self {
            MSEvent::SeqOpened { cid, .. } => Some(cid.as_str()),
            MSEvent::SeqAppendFinish { cid, .. } => Some(cid.as_str()),
            MSEvent::SeqGenFinish { cid, .. } => Some(cid.as_str()),
            MSEvent::SeqEmbedding { cid, .. } => Some(cid.as_str()),
            MSEvent::SeqEmbedFinish { cid, .. } => Some(cid.as_str()),
            MSEvent::SeqForkFinish { cid, .. } => Some(cid.as_str()),
            MSEvent::SeqText { cid, .. } => Some(cid.as_str()),
            MSEvent::SeqToolCall { cid, .. } => Some(cid.as_str()),
            MSEvent::SeqToolCallStart { cid, .. } => Some(cid.as_str()),
            MSEvent::SeqToolCallArgsChunk { cid, .. } => Some(cid.as_str()),
            MSEvent::SeqToolCallEnd { cid, .. } => Some(cid.as_str()),
            MSEvent::SeqToolCallAborted { cid, .. } => Some(cid.as_str()),
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
            MSEvent::SeqEmbedding { .. } => "seq_embedding",
            MSEvent::SeqEmbedFinish { .. } => "seq_embed_finish",
            MSEvent::SeqForkFinish { .. } => "seq_fork_finish",
            MSEvent::SeqText { .. } => "seq_text",
            MSEvent::SeqToolCall { .. } => "seq_tool_call",
            MSEvent::SeqToolCallStart { .. } => "seq_tool_call_start",
            MSEvent::SeqToolCallArgsChunk { .. } => "seq_tool_call_args_chunk",
            MSEvent::SeqToolCallEnd { .. } => "seq_tool_call_end",
            MSEvent::SeqToolCallAborted { .. } => "seq_tool_call_aborted",
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
            MSEvent::SeqEmbedding { seq_id, .. } => Some(seq_id.as_str()),
            MSEvent::SeqEmbedFinish { seq_id, .. } => Some(seq_id.as_str()),
            MSEvent::SeqForkFinish { seq_id, .. } => Some(seq_id.as_str()),
            MSEvent::SeqText { seq_id, .. } => Some(seq_id.as_str()),
            MSEvent::SeqToolCall { seq_id, .. } => Some(seq_id.as_str()),
            MSEvent::SeqToolCallStart { seq_id, .. } => Some(seq_id.as_str()),
            MSEvent::SeqToolCallArgsChunk { seq_id, .. } => Some(seq_id.as_str()),
            MSEvent::SeqToolCallEnd { seq_id, .. } => Some(seq_id.as_str()),
            MSEvent::SeqToolCallAborted { seq_id, .. } => Some(seq_id.as_str()),
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
    #[serde(default)]
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokens: Option<Vec<u32>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub media: Option<SeqAppendMedia>,
    #[serde(default)]
    pub hidden: bool,
    #[serde(default)]
    pub echo: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeqAppendMedia {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uri: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blob: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hash: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub json_schema: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub json_schema_strict: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EmbeddingInput {
    Text(String),
    Tokens(Vec<u32>),
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct SeqEmbedReq {
    pub inputs: Vec<EmbeddingInput>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub normalize: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub truncate: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub name: String,
    pub args: String,
}

#[cfg(test)]
mod tests {
    use super::{
        MSEvent, RateLimitErrorDetails, SeqAppendMedia, SeqAppendReq, SeqCommand, SeqToolCall,
        SeqToolReturnReq, ToolResult,
    };

    #[test]
    fn seq_tool_call_deserializes_without_id() {
        let json = r#"{"name":"search","args":"{\"q\":\"hello\"}"}"#;
        let call: SeqToolCall = serde_json::from_str(json).unwrap();

        assert_eq!(call.id, None);
        assert_eq!(call.name, "search");
        assert_eq!(call.args, r#"{"q":"hello"}"#);
    }

    #[test]
    fn error_details_are_backward_compatible() {
        let old: MSEvent = serde_json::from_str(r#"{"event":"error","message":"nope"}"#).unwrap();
        assert!(matches!(
            old,
            MSEvent::Error {
                code: None,
                rate_limit: None,
                ..
            }
        ));

        let event = MSEvent::Error {
            cid: Some("gen".into()),
            seq_id: Some("seq_1".into()),
            message: "Rate limit exceeded".into(),
            code: Some("rate_limit_exceeded".into()),
            rate_limit: Some(RateLimitErrorDetails {
                cause: "rate_limited".into(),
                enforcement_mode: "enforced".into(),
                rpm_limit: 60,
                rpm_remaining: 0,
                tpm_limit: 100_000,
                tpm_remaining: -5,
                retry_after_ms: 1_000,
            }),
        };
        let json = serde_json::to_string(&event).unwrap();
        let decoded: MSEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(
            serde_json::to_value(decoded).unwrap(),
            serde_json::to_value(event).unwrap()
        );
    }

    #[test]
    fn seq_tool_call_round_trips_with_id() {
        let call = SeqToolCall {
            id: Some("functions.search:0".to_string()),
            name: "search".to_string(),
            args: r#"{"q":"hello"}"#.to_string(),
        };

        let json = serde_json::to_string(&call).unwrap();
        assert!(json.contains(r#""id":"functions.search:0""#));

        let round_trip: SeqToolCall = serde_json::from_str(&json).unwrap();
        assert_eq!(round_trip.id, Some("functions.search:0".to_string()));
        assert_eq!(round_trip.name, "search");
        assert_eq!(round_trip.args, r#"{"q":"hello"}"#);
    }

    #[test]
    fn seq_tool_call_omits_absent_id() {
        let call = SeqToolCall {
            id: None,
            name: "search".to_string(),
            args: r#"{"q":"hello"}"#.to_string(),
        };

        let json = serde_json::to_string(&call).unwrap();
        assert!(!json.contains(r#""id""#));
    }

    #[test]
    fn append_request_deserializes_legacy_text_payload() {
        let json = r#"{"text":"hello","hidden":false,"echo":false,"role":"user"}"#;
        let req: SeqAppendReq = serde_json::from_str(json).unwrap();

        assert_eq!(req.text, "hello");
        assert_eq!(req.role.as_deref(), Some("user"));
        assert!(req.media.is_none());
    }

    #[test]
    fn append_command_round_trips_with_media_uri() {
        let cmd = SeqCommand::Append(SeqAppendReq {
            text: String::new(),
            media: Some(SeqAppendMedia {
                uri: Some("https://example.com/cat.png".to_string()),
                blob: None,
                hash: Some("b3abc".to_string()),
                mime_type: Some("image/png".to_string()),
                detail: Some("auto".to_string()),
            }),
            role: Some("user".to_string()),
            ..Default::default()
        });

        let json = serde_json::to_string(&cmd).unwrap();
        assert!(json.contains(r#""command":"append""#));
        assert!(json.contains(r#""uri":"https://example.com/cat.png""#));

        let parsed: SeqCommand = serde_json::from_str(&json).unwrap();
        let SeqCommand::Append(parsed) = parsed else {
            panic!("expected append command");
        };
        let media = parsed.media.expect("media should roundtrip");

        assert_eq!(media.uri.as_deref(), Some("https://example.com/cat.png"));
        assert_eq!(media.mime_type.as_deref(), Some("image/png"));
        assert_eq!(media.detail.as_deref(), Some("auto"));
    }

    #[test]
    fn append_command_round_trips_with_media_blob() {
        let cmd = SeqCommand::Append(SeqAppendReq {
            text: String::new(),
            media: Some(SeqAppendMedia {
                uri: None,
                blob: Some("aW1hZ2U=".to_string()),
                hash: Some("b3abc".to_string()),
                mime_type: Some("image/png".to_string()),
                detail: Some("low".to_string()),
            }),
            role: Some("user".to_string()),
            ..Default::default()
        });

        let json = serde_json::to_string(&cmd).unwrap();
        assert!(json.contains(r#""blob":"aW1hZ2U=""#));

        let parsed: SeqCommand = serde_json::from_str(&json).unwrap();
        let SeqCommand::Append(parsed) = parsed else {
            panic!("expected append command");
        };
        let media = parsed.media.expect("media should roundtrip");

        assert_eq!(media.blob.as_deref(), Some("aW1hZ2U="));
        assert_eq!(media.detail.as_deref(), Some("low"));
    }

    #[test]
    fn seq_tool_call_event_deserializes_legacy_payload() {
        let json = r#"{
            "event":"seq_tool_call",
            "seq_id":"seq_1",
            "cid":"gen",
            "tool_calls":[{"name":"search","args":"{\"q\":\"hello\"}"}]
        }"#;

        let event: MSEvent = serde_json::from_str(json).unwrap();
        let MSEvent::SeqToolCall { tool_calls, .. } = event else {
            panic!("expected seq_tool_call event");
        };

        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id, None);
        assert_eq!(tool_calls[0].name, "search");
    }

    #[test]
    fn tool_result_deserializes_without_id() {
        let json = r#"{"name":"search","result":"{\"hits\":3}"}"#;
        let result: ToolResult = serde_json::from_str(json).unwrap();

        assert_eq!(result.id, None);
        assert_eq!(result.name, "search");
        assert_eq!(result.result, r#"{"hits":3}"#);
    }

    #[test]
    fn tool_result_round_trips_with_id() {
        let result = ToolResult {
            id: Some("functions.search:0".to_string()),
            name: "search".to_string(),
            result: r#"{"hits":3}"#.to_string(),
        };

        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains(r#""id":"functions.search:0""#));

        let round_trip: ToolResult = serde_json::from_str(&json).unwrap();
        assert_eq!(round_trip.id, Some("functions.search:0".to_string()));
        assert_eq!(round_trip.name, "search");
        assert_eq!(round_trip.result, r#"{"hits":3}"#);
    }

    #[test]
    fn tool_result_omits_absent_id() {
        let result = ToolResult {
            id: None,
            name: "search".to_string(),
            result: r#"{"hits":3}"#.to_string(),
        };

        let json = serde_json::to_string(&result).unwrap();
        assert!(!json.contains(r#""id""#));
    }

    #[test]
    fn tool_return_deserializes_legacy_results_without_id() {
        let json = r#"{
            "results":[{"name":"search","result":"{\"hits\":3}"}],
            "gen_opts":{}
        }"#;

        let req: SeqToolReturnReq = serde_json::from_str(json).unwrap();
        assert_eq!(req.results.len(), 1);
        assert_eq!(req.results[0].id, None);
        assert_eq!(req.results[0].name, "search");
    }
}

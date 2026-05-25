use std::{collections::HashMap, fmt, future::Future, pin::Pin};

use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tracing::debug;

use crate::{SeqToolCall, ToolResult};

#[async_trait]
pub trait Toolbox: Send + Sync + 'static + std::fmt::Debug {
    /// Call tools and return the results. If this method returns None,
    /// the client will not send a tool return command to the server.
    async fn call_tools(&self, calls: &[SeqToolCall]) -> Result<Option<Vec<ToolResult>>>;

    /// Return a prompt that describes the tools available to the server along
    /// with their parameters.
    fn tool_def_prompt(&self) -> Option<String>;
}

/// Simple toolbox that expects a string input
/// and returns a string output. Calls tools sequentially.
pub struct SimpleToolbox {
    tools: HashMap<String, Box<dyn Tool>>,
}

impl fmt::Debug for SimpleToolbox {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SimpleToolbox")
            .field("tools", &self.tools.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl SimpleToolbox {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    pub fn add_tool<T: Tool + 'static>(&mut self, tool: T) {
        let tool = Box::new(tool);
        let definition = tool.definition();
        self.tools.insert(definition.name.clone(), tool);
    }

    pub async fn call_tools(&self, calls: &[SeqToolCall]) -> Result<Option<Vec<ToolResult>>> {
        let mut results = Vec::new();

        for call in calls {
            let result = self.call(call.name.as_str(), call.args.as_str()).await?;
            results.push(ToolResult {
                id: call.id.clone(),
                name: call.name.clone(),
                result,
            });
        }

        Ok(Some(results))
    }

    async fn call(&self, name: &str, args: &str) -> Result<String> {
        let tool = self
            .tools
            .get(name)
            .ok_or(anyhow::anyhow!("tool {} not found", name))?;

        debug!(tool = name, "tool call");

        tool.call(args).await
    }

    pub fn tool_def_prompt(&self) -> Option<String> {
        Some(
            self.tools
                .values()
                .map(|tool| tool_def_prompt(&tool.definition()))
                .collect::<Vec<_>>()
                .join("\n\n"),
        )
    }
}

#[async_trait]
impl Toolbox for SimpleToolbox {
    async fn call_tools(&self, calls: &[SeqToolCall]) -> Result<Option<Vec<ToolResult>>> {
        Self::call_tools(self, calls).await
    }

    fn tool_def_prompt(&self) -> Option<String> {
        Self::tool_def_prompt(self)
    }
}

fn tool_def_prompt(tool: &ToolDefinition) -> String {
    format!(
        "Use the {} function to: {}\n\n{}\n",
        tool.name,
        tool.description,
        serde_json::to_string_pretty(&tool.parameters).unwrap()
    )
}

pub trait Tool: fmt::Debug + Send + Sync + 'static {
    fn definition(&self) -> ToolDefinition;
    fn call(&self, args: &str) -> Pin<Box<dyn Future<Output = Result<String>> + Send>>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: ToolParameters,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolParameters {
    #[serde(rename = "type")]
    pub type_: ToolSchemaType,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub properties: HashMap<String, ToolProperty>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub required: Vec<String>,
}

impl Default for ToolParameters {
    fn default() -> Self {
        Self {
            type_: ToolSchemaType::Object,
            properties: HashMap::new(),
            required: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolProperty {
    #[serde(rename = "type")]
    pub type_: ToolSchemaType,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none", rename = "enum")]
    pub allowed_values: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ToolSchemaType {
    Object,
    String,
    Number,
    Integer,
    Boolean,
    Array,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::collections::HashMap;

    #[test]
    fn tool_definition_serializes_expected_parameters() {
        let mut properties = HashMap::new();
        properties.insert(
            "location".to_string(),
            ToolProperty {
                type_: ToolSchemaType::String,
                description: Some("The city and state, e.g. San Francisco, CA".to_string()),
                allowed_values: None,
            },
        );
        properties.insert(
            "unit".to_string(),
            ToolProperty {
                type_: ToolSchemaType::String,
                description: None,
                allowed_values: Some(vec!["celsius".to_string(), "fahrenheit".to_string()]),
            },
        );

        let definition = ToolDefinition {
            name: "get_weather".to_string(),
            description: "Get the current weather".to_string(),
            parameters: ToolParameters {
                type_: ToolSchemaType::Object,
                properties,
                required: vec!["location".to_string()],
            },
        };

        let value = serde_json::to_value(definition).unwrap();

        assert_eq!(value["parameters"]["type"], "object");
        assert_eq!(value["parameters"]["required"][0], "location");
        assert_eq!(
            value["parameters"]["properties"]["location"]["description"],
            "The city and state, e.g. San Francisco, CA"
        );
        assert_eq!(
            value["parameters"]["properties"]["unit"]["enum"],
            json!(["celsius", "fahrenheit"])
        );
    }
}

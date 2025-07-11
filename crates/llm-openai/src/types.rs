//! OpenAI-specific request and response types.

use llm_core::{ChatResponse, CompletionResponse, FinishReason, Metadata, ToolCall, Usage};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// OpenAI chat completion request.
#[derive(Debug, Clone, Serialize)]
pub struct OpenAIChatRequest {
    pub model: String,
    pub messages: Vec<OpenAIMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub stop: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<OpenAITool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

/// OpenAI message format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIMessage {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OpenAIToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

/// OpenAI tool call format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: OpenAIFunctionCall,
}

/// OpenAI function call format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIFunctionCall {
    pub name: String,
    pub arguments: String,
}

/// OpenAI tool definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAITool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: OpenAIFunction,
}

/// OpenAI function definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIFunction {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

/// OpenAI chat completion response.
#[derive(Debug, Clone, Deserialize)]
pub struct OpenAIChatResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<OpenAIChatChoice>,
    pub usage: Option<OpenAIUsage>,
    pub system_fingerprint: Option<String>,
}

/// OpenAI chat choice.
#[derive(Debug, Clone, Deserialize)]
pub struct OpenAIChatChoice {
    pub index: u32,
    pub message: OpenAIMessage,
    pub finish_reason: Option<String>,
    pub logprobs: Option<serde_json::Value>,
}

/// OpenAI usage statistics.
#[derive(Debug, Clone, Deserialize)]
pub struct OpenAIUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// OpenAI embeddings usage statistics (no completion_tokens).
#[derive(Debug, Clone, Deserialize)]
pub struct OpenAIEmbeddingsUsage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}

/// OpenAI completion request.
#[derive(Debug, Clone, Serialize)]
pub struct OpenAICompletionRequest {
    pub model: String,
    pub prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub stop: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

/// OpenAI completion response.
#[derive(Debug, Clone, Deserialize)]
pub struct OpenAICompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<OpenAICompletionChoice>,
    pub usage: Option<OpenAIUsage>,
}

/// OpenAI completion choice.
#[derive(Debug, Clone, Deserialize)]
pub struct OpenAICompletionChoice {
    pub index: u32,
    pub text: String,
    pub finish_reason: Option<String>,
    pub logprobs: Option<serde_json::Value>,
}

/// OpenAI embeddings request.
#[derive(Debug, Clone, Serialize)]
pub struct OpenAIEmbeddingsRequest {
    pub model: String,
    pub input: serde_json::Value, // Can be string or array of strings
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encoding_format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

/// OpenAI embeddings response.
#[derive(Debug, Clone, Deserialize)]
pub struct OpenAIEmbeddingsResponse {
    pub object: String,
    pub data: Vec<OpenAIEmbedding>,
    pub model: String,
    pub usage: OpenAIEmbeddingsUsage,
}

/// OpenAI embedding data.
#[derive(Debug, Clone, Deserialize)]
pub struct OpenAIEmbedding {
    pub object: String,
    pub index: usize,
    pub embedding: Vec<f32>,
}

/// OpenAI streaming response chunk.
#[derive(Debug, Clone, Deserialize)]
pub struct OpenAIStreamChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<OpenAIStreamChoice>,
}

/// OpenAI streaming choice.
#[derive(Debug, Clone, Deserialize)]
pub struct OpenAIStreamChoice {
    pub index: u32,
    pub delta: OpenAIStreamDelta,
    pub finish_reason: Option<String>,
}

/// OpenAI streaming delta.
#[derive(Debug, Clone, Deserialize)]
pub struct OpenAIStreamDelta {
    pub role: Option<String>,
    pub content: Option<String>,
    pub tool_calls: Option<Vec<OpenAIStreamToolCall>>,
}

/// OpenAI streaming tool call.
#[derive(Debug, Clone, Deserialize)]
pub struct OpenAIStreamToolCall {
    pub index: u32,
    pub id: Option<String>,
    #[serde(rename = "type")]
    pub call_type: Option<String>,
    pub function: Option<OpenAIStreamFunction>,
}

/// OpenAI streaming function.
#[derive(Debug, Clone, Deserialize)]
pub struct OpenAIStreamFunction {
    pub name: Option<String>,
    pub arguments: Option<String>,
}

// Implement ChatResponse for OpenAIChatResponse
impl ChatResponse for OpenAIChatResponse {
    fn content(&self) -> &str {
        self.choices
            .first()
            .and_then(|choice| choice.message.content.as_ref())
            .and_then(|content| content.as_str())
            .unwrap_or("")
    }

    fn usage(&self) -> Option<&Usage> {
        // We need to convert OpenAIUsage to Usage
        // This is a bit tricky with the current design
        // For now, return None and handle conversion elsewhere
        None
    }

    fn finish_reason(&self) -> Option<FinishReason> {
        self.choices
            .first()
            .and_then(|choice| choice.finish_reason.as_ref())
            .and_then(|reason| match reason.as_str() {
                "stop" => Some(FinishReason::Stop),
                "length" => Some(FinishReason::Length),
                "tool_calls" => Some(FinishReason::ToolCalls),
                "content_filter" => Some(FinishReason::ContentFilter),
                _ => None,
            })
    }

    fn metadata(&self) -> &Metadata {
        // This is also tricky - we need to store metadata separately
        // For now, return a static empty metadata
        use std::sync::LazyLock;
        static EMPTY_METADATA: LazyLock<Metadata> = LazyLock::new(|| Metadata {
            extensions: HashMap::new(),
            request_id: None,
            user_id: None,
            created_at: chrono::DateTime::UNIX_EPOCH,
        });
        &EMPTY_METADATA
    }

    fn tool_calls(&self) -> Option<&[ToolCall]> {
        // Similar issue - need conversion from OpenAI format
        None
    }
}

// Implement CompletionResponse for OpenAICompletionResponse
impl CompletionResponse for OpenAICompletionResponse {
    fn text(&self) -> &str {
        self.choices
            .first()
            .map(|choice| choice.text.as_str())
            .unwrap_or("")
    }

    fn usage(&self) -> Option<&Usage> {
        None // Same conversion issue
    }

    fn finish_reason(&self) -> Option<FinishReason> {
        self.choices
            .first()
            .and_then(|choice| choice.finish_reason.as_ref())
            .and_then(|reason| match reason.as_str() {
                "stop" => Some(FinishReason::Stop),
                "length" => Some(FinishReason::Length),
                _ => None,
            })
    }

    fn metadata(&self) -> &Metadata {
        use std::sync::LazyLock;
        static EMPTY_METADATA: LazyLock<Metadata> = LazyLock::new(|| Metadata {
            extensions: HashMap::new(),
            request_id: None,
            user_id: None,
            created_at: chrono::DateTime::UNIX_EPOCH,
        });
        &EMPTY_METADATA
    }
}

// Conversion utilities
impl From<&llm_core::Message> for OpenAIMessage {
    fn from(message: &llm_core::Message) -> Self {
        let role = match message.role {
            llm_core::Role::User => "user",
            llm_core::Role::Assistant => "assistant",
            llm_core::Role::System => "system",
            llm_core::Role::Tool => "tool",
        };

        let content = match &message.content {
            llm_core::MessageContent::Text(text) => Some(serde_json::Value::String(text.clone())),
            llm_core::MessageContent::Multimodal(parts) => {
                // Convert multimodal content to OpenAI format
                let content_parts: Vec<serde_json::Value> = parts
                    .iter()
                    .map(|part| match part {
                        llm_core::ContentPart::Text { text } => serde_json::json!({
                            "type": "text",
                            "text": text
                        }),
                        llm_core::ContentPart::Image { image_url, detail } => serde_json::json!({
                            "type": "image_url",
                            "image_url": {
                                "url": image_url.url,
                                "detail": detail.as_deref().or(image_url.detail.as_deref()).unwrap_or("auto")
                            }
                        }),
                        llm_core::ContentPart::Audio { audio_url, .. } => serde_json::json!({
                            "type": "audio_url",
                            "audio_url": audio_url
                        }),
                    })
                    .collect();
                Some(serde_json::Value::Array(content_parts))
            }
        };

        let tool_calls = message.tool_calls.as_ref().map(|calls| {
            calls
                .iter()
                .map(|call| OpenAIToolCall {
                    id: call.id.clone(),
                    call_type: call.call_type.clone(),
                    function: OpenAIFunctionCall {
                        name: call.function.name.clone(),
                        arguments: call.function.arguments.clone(),
                    },
                })
                .collect()
        });

        Self {
            role: role.to_string(),
            content,
            name: message.name.clone(),
            tool_calls,
            tool_call_id: message.tool_call_id.clone(),
        }
    }
}

impl From<&llm_core::Tool> for OpenAITool {
    fn from(tool: &llm_core::Tool) -> Self {
        Self {
            tool_type: tool.tool_type.clone(),
            function: OpenAIFunction {
                name: tool.function.name.clone(),
                description: tool.function.description.clone(),
                parameters: tool.function.parameters.clone(),
            },
        }
    }
}

impl From<OpenAIUsage> for Usage {
    fn from(usage: OpenAIUsage) -> Self {
        Self {
            prompt_tokens: usage.prompt_tokens,
            completion_tokens: usage.completion_tokens,
            total_tokens: usage.total_tokens,
        }
    }
}

impl From<OpenAIEmbeddingsUsage> for Usage {
    fn from(usage: OpenAIEmbeddingsUsage) -> Self {
        Self {
            prompt_tokens: usage.prompt_tokens,
            completion_tokens: 0, // Embeddings don't have completion tokens
            total_tokens: usage.total_tokens,
        }
    }
}

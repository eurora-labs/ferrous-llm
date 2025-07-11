//! Core types for LLM requests and responses.
//!
//! This module defines standardized types that are used across all providers,
//! including request/response structures, messages, and common data types.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use uuid::Uuid;

/// A chat request containing messages and parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    /// The conversation messages
    pub messages: Vec<Message>,
    /// Request parameters (temperature, max_tokens, etc.)
    pub parameters: Parameters,
    /// Additional metadata and provider-specific extensions
    pub metadata: Metadata,
}

/// A completion request for non-chat text generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionRequest {
    /// The text prompt to complete
    pub prompt: String,
    /// Request parameters
    pub parameters: Parameters,
    /// Additional metadata
    pub metadata: Metadata,
}

/// Common parameters used across providers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Parameters {
    /// Controls randomness in the response (0.0 to 2.0)
    pub temperature: Option<f32>,
    /// Maximum number of tokens to generate
    pub max_tokens: Option<u32>,
    /// Nucleus sampling parameter (0.0 to 1.0)
    pub top_p: Option<f32>,
    /// Alternative to temperature, called Top-k sampling
    pub top_k: Option<u32>,
    /// Sequences where the API will stop generating further tokens
    pub stop_sequences: Vec<String>,
    /// Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency
    pub frequency_penalty: Option<f32>,
    /// Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far
    pub presence_penalty: Option<f32>,
}

/// Metadata for requests, including provider-specific extensions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metadata {
    /// Provider-specific extensions
    pub extensions: HashMap<String, Value>,
    /// Optional request ID for tracking
    pub request_id: Option<String>,
    /// Optional user ID for tracking
    pub user_id: Option<String>,
    /// Timestamp when the request was created
    pub created_at: DateTime<Utc>,
}

/// A message in a conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// The role of the message sender
    pub role: Role,
    /// The content of the message
    pub content: MessageContent,
    /// Optional name of the message sender
    pub name: Option<String>,
    /// Optional tool calls in this message
    pub tool_calls: Option<Vec<ToolCall>>,
    /// Optional tool call ID if this is a tool response
    pub tool_call_id: Option<String>,
    /// Timestamp when the message was created
    pub created_at: DateTime<Utc>,
}

/// The role of a message sender.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// Message from the user
    User,
    /// Message from the AI assistant
    Assistant,
    /// System message (instructions, context)
    System,
    /// Message from a tool/function call
    Tool,
}

/// Content of a message, which can be text or multimodal.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    /// Simple text content
    Text(String),
    /// Multimodal content with text and other media
    Multimodal(Vec<ContentPart>),
}

/// A part of multimodal message content.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    /// Text content
    Text { text: String },
    /// Image content
    Image {
        /// Image data or URL
        image_url: ImageUrl,
        /// Optional detail level for image processing
        detail: Option<String>,
    },
    /// Audio content
    Audio {
        /// Audio data or URL
        audio_url: String,
        /// Audio format (mp3, wav, etc.)
        format: Option<String>,
    },
}

/// Image URL or data for multimodal content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUrl {
    /// The URL or base64-encoded image data
    pub url: String,
    /// Optional detail level (low, high, auto)
    pub detail: Option<String>,
}

/// A tool/function call made by the AI.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Unique identifier for this tool call
    pub id: String,
    /// The type of tool call (usually "function")
    #[serde(rename = "type")]
    pub call_type: String,
    /// The function being called
    pub function: FunctionCall,
}

/// A function call within a tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    /// Name of the function to call
    pub name: String,
    /// Arguments to pass to the function (JSON string)
    pub arguments: String,
}

/// Definition of a tool/function that can be called.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    /// Type of tool (usually "function")
    #[serde(rename = "type")]
    pub tool_type: String,
    /// The function definition
    pub function: Function,
}

/// Definition of a function that can be called.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Function {
    /// Name of the function
    pub name: String,
    /// Description of what the function does
    pub description: String,
    /// JSON schema for the function parameters
    pub parameters: Value,
}

/// Usage statistics for a request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    /// Number of tokens in the prompt
    pub prompt_tokens: u32,
    /// Number of tokens in the completion
    pub completion_tokens: u32,
    /// Total number of tokens used
    pub total_tokens: u32,
}

/// Reason why the model stopped generating.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// The model reached a natural stopping point
    Stop,
    /// The model reached the maximum token limit
    Length,
    /// The model generated a stop sequence
    StopSequence,
    /// The model made a tool call
    ToolCalls,
    /// Content was filtered
    ContentFilter,
    /// An error occurred
    Error,
}

/// An embedding vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Embedding {
    /// The embedding vector
    pub embedding: Vec<f32>,
    /// Index of the input text this embedding corresponds to
    pub index: usize,
}

/// Request for image generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageRequest {
    /// Text prompt for image generation
    pub prompt: String,
    /// Optional negative prompt (what to avoid)
    pub negative_prompt: Option<String>,
    /// Number of images to generate
    pub n: Option<u32>,
    /// Image size specification
    pub size: Option<String>,
    /// Image quality setting
    pub quality: Option<String>,
    /// Response format (url or b64_json)
    pub response_format: Option<String>,
    /// Additional metadata
    pub metadata: Metadata,
}

/// Request for speech-to-text conversion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeechToTextRequest {
    /// Audio data (base64 encoded or file path)
    pub audio: String,
    /// Audio format (mp3, wav, etc.)
    pub format: String,
    /// Language of the audio (optional)
    pub language: Option<String>,
    /// Additional metadata
    pub metadata: Metadata,
}

/// Request for text-to-speech conversion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextToSpeechRequest {
    /// Text to convert to speech
    pub text: String,
    /// Voice to use
    pub voice: String,
    /// Audio format for output
    pub format: Option<String>,
    /// Speed of speech (0.25 to 4.0)
    pub speed: Option<f32>,
    /// Additional metadata
    pub metadata: Metadata,
}

/// Trait for chat response types.
pub trait ChatResponse: Send + Sync {
    /// Get the text content of the response
    fn content(&self) -> &str;

    /// Get usage statistics if available
    fn usage(&self) -> Option<&Usage>;

    /// Get the reason why generation finished
    fn finish_reason(&self) -> Option<FinishReason>;

    /// Get response metadata
    fn metadata(&self) -> &Metadata;

    /// Get tool calls if any were made
    fn tool_calls(&self) -> Option<&[ToolCall]> {
        None
    }

    /// Convert response to a Message for conversation history
    fn as_message(&self) -> Message {
        Message {
            role: Role::Assistant,
            content: MessageContent::Text(self.content().to_string()),
            name: None,
            tool_calls: self.tool_calls().map(|calls| calls.to_vec()),
            tool_call_id: None,
            created_at: Utc::now(),
        }
    }
}

/// Trait for completion response types.
pub trait CompletionResponse: Send + Sync {
    /// Get the completion text
    fn text(&self) -> &str;

    /// Get usage statistics if available
    fn usage(&self) -> Option<&Usage>;

    /// Get the reason why generation finished
    fn finish_reason(&self) -> Option<FinishReason>;

    /// Get response metadata
    fn metadata(&self) -> &Metadata;
}

/// Trait for image generation response types.
pub trait ImageResponse: Send + Sync {
    /// Get the generated images
    fn images(&self) -> &[GeneratedImage];

    /// Get response metadata
    fn metadata(&self) -> &Metadata;
}

/// A generated image.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedImage {
    /// Image URL or base64 data
    pub url: Option<String>,
    /// Base64 encoded image data
    pub b64_json: Option<String>,
    /// Revised prompt used for generation
    pub revised_prompt: Option<String>,
}

/// Trait for speech-to-text response types.
pub trait SpeechToTextResponse: Send + Sync {
    /// Get the transcribed text
    fn text(&self) -> &str;

    /// Get the detected language if available
    fn language(&self) -> Option<&str>;

    /// Get response metadata
    fn metadata(&self) -> &Metadata;
}

/// Trait for text-to-speech response types.
pub trait TextToSpeechResponse: Send + Sync {
    /// Get the audio data
    fn audio_data(&self) -> &[u8];

    /// Get the audio format
    fn format(&self) -> &str;

    /// Get response metadata
    fn metadata(&self) -> &Metadata;
}

// Convenience constructors
impl Message {
    /// Create a user message with text content
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: MessageContent::Text(content.into()),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            created_at: Utc::now(),
        }
    }

    /// Create an assistant message with text content
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: MessageContent::Text(content.into()),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            created_at: Utc::now(),
        }
    }

    /// Create a system message with text content
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: MessageContent::Text(content.into()),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            created_at: Utc::now(),
        }
    }
}

impl Default for Parameters {
    fn default() -> Self {
        Self {
            temperature: None,
            max_tokens: None,
            top_p: None,
            top_k: None,
            stop_sequences: Vec::new(),
            frequency_penalty: None,
            presence_penalty: None,
        }
    }
}

impl Default for Metadata {
    fn default() -> Self {
        Self {
            extensions: HashMap::new(),
            request_id: None,
            user_id: None,
            created_at: Utc::now(),
        }
    }
}

impl ChatRequest {
    /// Create a new chat request builder
    pub fn builder() -> ChatRequestBuilder {
        ChatRequestBuilder::new()
    }
}

/// Builder for ChatRequest
pub struct ChatRequestBuilder {
    messages: Vec<Message>,
    parameters: Parameters,
    metadata: Metadata,
}

impl ChatRequestBuilder {
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
            parameters: Parameters::default(),
            metadata: Metadata::default(),
        }
    }

    pub fn message(mut self, message: Message) -> Self {
        self.messages.push(message);
        self
    }

    pub fn messages(mut self, messages: Vec<Message>) -> Self {
        self.messages = messages;
        self
    }

    pub fn temperature(mut self, temperature: f32) -> Self {
        self.parameters.temperature = Some(temperature);
        self
    }

    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.parameters.max_tokens = Some(max_tokens);
        self
    }

    pub fn top_p(mut self, top_p: f32) -> Self {
        self.parameters.top_p = Some(top_p);
        self
    }

    pub fn stop_sequences(mut self, stop_sequences: Vec<String>) -> Self {
        self.parameters.stop_sequences = stop_sequences;
        self
    }

    pub fn request_id(mut self, request_id: String) -> Self {
        self.metadata.request_id = Some(request_id);
        self
    }

    pub fn user_id(mut self, user_id: String) -> Self {
        self.metadata.user_id = Some(user_id);
        self
    }

    pub fn extension(mut self, key: String, value: Value) -> Self {
        self.metadata.extensions.insert(key, value);
        self
    }

    pub fn build(self) -> ChatRequest {
        ChatRequest {
            messages: self.messages,
            parameters: self.parameters,
            metadata: self.metadata,
        }
    }
}

//! Response types for gRPC providers.

use chrono::{DateTime, Utc};
use ferrous_llm_core::types::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::proto::chat::{ProtoImageSource, proto_image_source::ProtoSourceType};

/// Response from a gRPC chat request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrpcChatResponse {
    /// The response content
    pub content: String,

    /// Usage statistics
    pub usage: Option<Usage>,

    /// Reason why generation finished
    pub finish_reason: Option<FinishReason>,

    /// Response metadata
    pub metadata: Metadata,

    /// Tool calls if any were made
    pub tool_calls: Option<Vec<ToolCall>>,
}

impl ChatResponse for GrpcChatResponse {
    fn content(&self) -> String {
        self.content.clone()
    }

    fn usage(&self) -> Option<Usage> {
        self.usage.clone()
    }

    fn finish_reason(&self) -> Option<FinishReason> {
        self.finish_reason.clone()
    }

    fn metadata(&self) -> Metadata {
        self.metadata.clone()
    }

    fn tool_calls(&self) -> Option<Vec<ToolCall>> {
        self.tool_calls.clone()
    }
}

/// Streaming response chunk from a gRPC chat request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrpcStreamResponse {
    /// Incremental content
    pub content: String,

    /// Whether this is the final chunk
    pub is_final: bool,

    /// Usage statistics (only in final chunk)
    pub usage: Option<Usage>,

    /// Finish reason (only in final chunk)
    pub finish_reason: Option<FinishReason>,

    /// Response metadata
    pub metadata: Metadata,

    /// Tool calls if any were made
    pub tool_calls: Option<Vec<ToolCall>>,
}

impl GrpcStreamResponse {
    /// Create a new streaming response chunk.
    pub fn new(content: String, is_final: bool) -> Self {
        Self {
            content,
            is_final,
            usage: None,
            finish_reason: None,
            metadata: Metadata::default(),
            tool_calls: None,
        }
    }

    /// Create a final streaming response chunk with usage and finish reason.
    pub fn final_chunk(
        content: String,
        usage: Option<Usage>,
        finish_reason: Option<FinishReason>,
        tool_calls: Option<Vec<ToolCall>>,
    ) -> Self {
        Self {
            content,
            is_final: true,
            usage,
            finish_reason,
            metadata: Metadata::default(),
            tool_calls,
        }
    }

    /// Check if this is the final chunk in the stream.
    pub fn is_final(&self) -> bool {
        self.is_final
    }

    /// Get the content of this chunk.
    pub fn content(&self) -> &str {
        &self.content
    }

    /// Get usage statistics if available (typically only in final chunk).
    pub fn usage(&self) -> Option<&Usage> {
        self.usage.as_ref()
    }

    /// Get finish reason if available (typically only in final chunk).
    pub fn finish_reason(&self) -> Option<&FinishReason> {
        self.finish_reason.as_ref()
    }

    /// Get tool calls if any were made.
    pub fn tool_calls(&self) -> Option<&Vec<ToolCall>> {
        self.tool_calls.as_ref()
    }
}

// Conversion functions between proto and core types

/// Convert proto Role to core Role
pub fn proto_role_to_core(role: i32) -> Role {
    match role {
        1 => Role::User,
        2 => Role::Assistant,
        3 => Role::System,
        4 => Role::Tool,
        _ => Role::User, // Default fallback
    }
}

/// Convert core Role to proto Role
pub fn core_role_to_proto(role: &Role) -> i32 {
    match role {
        Role::User => 1,
        Role::Assistant => 2,
        Role::System => 3,
        Role::Tool => 4,
    }
}

/// Convert proto FinishReason to core FinishReason
pub fn proto_finish_reason_to_core(reason: i32) -> FinishReason {
    match reason {
        1 => FinishReason::Stop,
        2 => FinishReason::Length,
        3 => FinishReason::StopSequence,
        4 => FinishReason::ToolCalls,
        5 => FinishReason::ContentFilter,
        6 => FinishReason::Error,
        _ => FinishReason::Stop, // Default fallback
    }
}

/// Convert core FinishReason to proto FinishReason
pub fn core_finish_reason_to_proto(reason: &FinishReason) -> i32 {
    match reason {
        FinishReason::Stop => 1,
        FinishReason::Length => 2,
        FinishReason::StopSequence => 3,
        FinishReason::ToolCalls => 4,
        FinishReason::ContentFilter => 5,
        FinishReason::Error => 6,
    }
}

/// Convert proto Timestamp to DateTime<Utc>
pub fn proto_timestamp_to_datetime(timestamp: Option<prost_types::Timestamp>) -> DateTime<Utc> {
    timestamp
        .map(|ts| {
            DateTime::from_timestamp(ts.seconds, ts.nanos as u32).unwrap_or_else(|| Utc::now())
        })
        .unwrap_or_else(|| Utc::now())
}

/// Convert DateTime<Utc> to proto Timestamp
pub fn datetime_to_proto_timestamp(datetime: &DateTime<Utc>) -> prost_types::Timestamp {
    prost_types::Timestamp {
        seconds: datetime.timestamp(),
        nanos: datetime.timestamp_subsec_nanos() as i32,
    }
}

/// Convert proto Struct to HashMap<String, serde_json::Value>
pub fn proto_struct_to_hashmap(
    proto_struct: Option<prost_types::Struct>,
) -> HashMap<String, serde_json::Value> {
    proto_struct
        .map(|s| {
            s.fields
                .into_iter()
                .filter_map(|(k, v)| proto_value_to_json_value(v).map(|json_val| (k, json_val)))
                .collect()
        })
        .unwrap_or_default()
}

/// Convert HashMap<String, serde_json::Value> to proto Struct
pub fn hashmap_to_proto_struct(map: &HashMap<String, serde_json::Value>) -> prost_types::Struct {
    let fields = map
        .iter()
        .filter_map(|(k, v)| json_value_to_proto_value(v).map(|proto_val| (k.clone(), proto_val)))
        .collect();

    prost_types::Struct { fields }
}

/// Convert proto Value to serde_json::Value
fn proto_value_to_json_value(value: prost_types::Value) -> Option<serde_json::Value> {
    use prost_types::value::Kind;

    match value.kind? {
        Kind::NullValue(_) => Some(serde_json::Value::Null),
        Kind::NumberValue(n) => Some(serde_json::Value::Number(serde_json::Number::from_f64(n)?)),
        Kind::StringValue(s) => Some(serde_json::Value::String(s)),
        Kind::BoolValue(b) => Some(serde_json::Value::Bool(b)),
        Kind::StructValue(s) => {
            let map = proto_struct_to_hashmap(Some(s));
            Some(serde_json::Value::Object(map.into_iter().collect()))
        }
        Kind::ListValue(l) => {
            let values: Option<Vec<_>> = l
                .values
                .into_iter()
                .map(proto_value_to_json_value)
                .collect();
            Some(serde_json::Value::Array(values?))
        }
    }
}

/// Convert serde_json::Value to proto Value
fn json_value_to_proto_value(value: &serde_json::Value) -> Option<prost_types::Value> {
    use prost_types::value::Kind;

    let kind = match value {
        serde_json::Value::Null => Kind::NullValue(0),
        serde_json::Value::Bool(b) => Kind::BoolValue(*b),
        serde_json::Value::Number(n) => Kind::NumberValue(n.as_f64()?),
        serde_json::Value::String(s) => Kind::StringValue(s.clone()),
        serde_json::Value::Array(arr) => {
            let values: Option<Vec<_>> = arr.iter().map(json_value_to_proto_value).collect();
            Kind::ListValue(prost_types::ListValue { values: values? })
        }
        serde_json::Value::Object(obj) => {
            let map: HashMap<String, serde_json::Value> =
                obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
            Kind::StructValue(hashmap_to_proto_struct(&map))
        }
    };

    Some(prost_types::Value { kind: Some(kind) })
}

impl From<ImageSource> for ProtoImageSource {
    fn from(source: ImageSource) -> Self {
        match source {
            ImageSource::Url(url) => ProtoImageSource {
                proto_source_type: Some(ProtoSourceType::Url(url)),
            },
            ImageSource::DynamicImage(image) => ProtoImageSource {
                proto_source_type: Some(ProtoSourceType::Data(image.into_bytes())),
            },
        }
    }
}

# ferrous-llm-grpc

[![Crates.io](https://img.shields.io/crates/v/ferrous-llm-grpc.svg)](https://crates.io/crates/ferrous-llm-grpc)
[![Documentation](https://docs.rs/ferrous-llm-grpc/badge.svg)](https://docs.rs/ferrous-llm-grpc)

A gRPC provider implementation for the ferrous-llm ecosystem, enabling communication with LLM services over gRPC protocol.

## Overview

This crate provides gRPC-based implementations of the [`ChatProvider`](../ferrous-llm-core/src/traits.rs) and [`StreamingProvider`](../ferrous-llm-core/src/traits.rs) traits from [`ferrous-llm-core`](../ferrous-llm-core). It allows you to connect to any LLM service that implements the ferrous-llm gRPC protocol specification.

## Features

-   **Full gRPC Support**: Complete implementation of chat and streaming chat services
-   **Multimodal Content**: Support for text, images, and audio in conversations
-   **Tool/Function Calls**: Native support for AI tool calling capabilities
-   **Flexible Configuration**: Comprehensive configuration options for timeouts, TLS, authentication, and more
-   **Error Handling**: Rich error types with retry logic and detailed error classification
-   **Type Safety**: Strong typing with automatic protobuf code generation
-   **Async/Await**: Full async support with tokio integration

## Protocol Definition

The gRPC service is defined in [`proto/chat.proto`](proto/chat.proto) and includes:

-   **ProtoChatService**: Main service with `Chat` (unary) and `ChatStream` (streaming) methods
-   **Comprehensive Message Types**: Support for all ferrous-llm core types including multimodal content
-   **Tool Integration**: Native support for function/tool calling
-   **Metadata & Usage**: Request tracking and token usage statistics

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
ferrous-llm-grpc = "0.1.0"
ferrous-llm-core = "0.3.0"
tokio = { version = "1.0", features = ["full"] }
```

## Quick Start

### Basic Chat

```rust
use ferrous_llm_grpc::{GrpcConfig, GrpcChatProvider};
use ferrous_llm_core::{
    traits::ChatProvider,
    types::{ChatRequest, Message, MessageContent, Role, Parameters}
};
use url::Url;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Configure the gRPC client
    let config = GrpcConfig::new(Url::parse("http://localhost:50051")?)
        .with_timeout(std::time::Duration::from_secs(30))
        .with_auth_token("your-api-key".to_string());

    // Create the provider
    let provider = GrpcChatProvider::new(config).await?;

    // Create a chat request
    let request = ChatRequest::new(vec![
        Message::new(Role::User, MessageContent::Text("Hello, world!".to_string()))
    ]);

    // Send the request
    let response = provider.chat(request).await?;
    println!("Response: {}", response.content());

    Ok(())
}
```

### Streaming Chat

```rust
use ferrous_llm_grpc::{GrpcConfig, GrpcStreamingProvider};
use ferrous_llm_core::{
    traits::{ChatProvider, StreamingProvider},
    types::{ChatRequest, Message, MessageContent, Role}
};
use futures::StreamExt;
use url::Url;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = GrpcConfig::new(Url::parse("http://localhost:50051")?);
    let provider = GrpcStreamingProvider::new(config).await?;

    let request = ChatRequest::new(vec![
        Message::new(Role::User, MessageContent::Text("Tell me a story".to_string()))
    ]);

    let mut stream = provider.chat_stream(request).await?;

    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(response) => {
                print!("{}", response.content());
                if response.is_final() {
                    println!("\n--- Final chunk ---");
                    if let Some(usage) = response.usage() {
                        println!("Tokens used: {}", usage.total_tokens);
                    }
                }
            }
            Err(e) => eprintln!("Stream error: {}", e),
        }
    }

    Ok(())
}
```

### Multimodal Content

```rust
use ferrous_llm_core::types::{ContentPart, ImageSource, MessageContent};

// Create a multimodal message with text and image
let content = MessageContent::Multimodal(vec![
    ContentPart::Text {
        text: "What's in this image?".to_string(),
    },
    ContentPart::Image {
        image_source: ImageSource::Url("https://example.com/image.jpg".to_string()),
        detail: Some("high".to_string()),
    },
]);

let message = Message::new(Role::User, content);
```

## Configuration

The [`GrpcConfig`](src/config.rs) struct provides comprehensive configuration options:

```rust
use std::time::Duration;
use url::Url;

let config = GrpcConfig::new(Url::parse("https://api.example.com")?)
    .with_auth_token("your-token".to_string())
    .with_tls(Some("api.example.com".to_string()))
    .with_timeout(Duration::from_secs(60))
    .with_connect_timeout(Duration::from_secs(10))
    .with_max_request_size(8 * 1024 * 1024)  // 8MB
    .with_max_response_size(8 * 1024 * 1024) // 8MB
    .with_keep_alive(
        Duration::from_secs(30), // interval
        Duration::from_secs(5),  // timeout
        true                     // while idle
    )
    .with_max_concurrent_requests(50)
    .with_user_agent("my-app/1.0".to_string());
```

### Configuration Options

-   **`endpoint`**: gRPC server URL (required)
-   **`auth_token`**: Bearer token for authentication
-   **`timeout`**: Request timeout duration
-   **`connect_timeout`**: Connection establishment timeout
-   **`use_tls`**: Enable TLS encryption
-   **`tls_domain`**: Domain name for TLS verification
-   **`max_request_size`**: Maximum request message size
-   **`max_response_size`**: Maximum response message size
-   **`keep_alive_*`**: TCP keep-alive settings
-   **`max_concurrent_requests`**: Connection pool size
-   **`user_agent`**: Custom user agent string

## Error Handling

The crate provides comprehensive error handling through [`GrpcError`](src/error.rs):

```rust
use ferrous_llm_grpc::GrpcError;
use ferrous_llm_core::error::ProviderError;

match provider.chat(request).await {
    Ok(response) => println!("Success: {}", response.content()),
    Err(e) => {
        match &e {
            GrpcError::Authentication(_) => println!("Auth failed: {}", e),
            GrpcError::RateLimit => println!("Rate limited, retry after: {:?}", e.retry_after()),
            GrpcError::Timeout => println!("Request timed out"),
            _ => println!("Other error: {}", e),
        }

        // Check if error is retryable
        if e.is_retryable() {
            println!("This error can be retried");
        }
    }
}
```

## Building and Protocol Buffers

This crate uses [`tonic-build`](https://docs.rs/tonic-build) to generate Rust code from Protocol Buffer definitions at build time. The generated code is placed in [`src/gen/`](src/gen/) and is automatically included in the library.

### Build Requirements

-   **Protocol Buffers Compiler**: `protoc` must be installed
-   **Windows**: Additional configuration for protoc include paths
-   **Generated Code**: Automatically created in `src/gen/` (gitignored)

The [`build.rs`](build.rs) script handles:

-   Automatic discovery of `.proto` files in the `proto/` directory
-   Cross-platform protoc configuration
-   Code generation with both client and server support
-   Proto3 optional field support

## Response Types

### GrpcChatResponse

Standard chat response implementing the [`ChatResponse`](../ferrous-llm-core/src/types.rs) trait:

```rust
pub struct GrpcChatResponse {
    pub content: String,
    pub usage: Option<Usage>,
    pub finish_reason: Option<FinishReason>,
    pub metadata: Metadata,
    pub tool_calls: Option<Vec<ToolCall>>,
}
```

### GrpcStreamResponse

Streaming response chunks:

```rust
pub struct GrpcStreamResponse {
    pub content: String,
    pub is_final: bool,
    pub usage: Option<Usage>,        // Only in final chunk
    pub finish_reason: Option<FinishReason>, // Only in final chunk
    pub metadata: Metadata,
    pub tool_calls: Option<Vec<ToolCall>>,
}
```

## Server Implementation

While this crate focuses on the client side, the generated protobuf code includes server traits. You can implement a gRPC server using:

```rust
use ferrous_llm_grpc::proto::chat::{
    proto_chat_service_server::{ProtoChatService, ProtoChatServiceServer},
    ProtoChatRequest, ProtoChatResponse, ProtoChatStreamResponse
};
use tonic::{Request, Response, Status};

#[derive(Default)]
pub struct MyChatService;

#[tonic::async_trait]
impl ProtoChatService for MyChatService {
    async fn chat(
        &self,
        request: Request<ProtoChatRequest>,
    ) -> Result<Response<ProtoChatResponse>, Status> {
        // Implement your chat logic here
        todo!()
    }

    type ChatStreamStream = /* your stream type */;

    async fn chat_stream(
        &self,
        request: Request<ProtoChatRequest>,
    ) -> Result<Response<Self::ChatStreamStream>, Status> {
        // Implement your streaming chat logic here
        todo!()
    }
}
```

## Testing

The crate includes comprehensive tests covering:

-   Configuration validation
-   Provider creation and connection handling
-   Request/response conversion
-   Error handling scenarios
-   Stream processing

Run tests with:

```bash
cargo test -p ferrous-llm-grpc
```

## Dependencies

Key dependencies include:

-   **[`tonic`](https://docs.rs/tonic)**: gRPC implementation for Rust
-   **[`prost`](https://docs.rs/prost)**: Protocol Buffers implementation
-   **[`tokio`](https://docs.rs/tokio)**: Async runtime
-   **[`ferrous-llm-core`](../ferrous-llm-core)**: Core types and traits
-   **[`async-stream`](https://docs.rs/async-stream)**: Stream utilities
-   **[`serde`](https://docs.rs/serde)**: Serialization framework

## License

This project is licensed under the MIT License - see the [LICENSE](../../LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## Related Crates

-   [`ferrous-llm-core`](../ferrous-llm-core): Core types and traits
-   [`ferrous-llm-openai`](../ferrous-llm-openai): OpenAI API provider
-   [`ferrous-llm-anthropic`](../ferrous-llm-anthropic): Anthropic API provider
-   [`ferrous-llm-ollama`](../ferrous-llm-ollama): Ollama local provider

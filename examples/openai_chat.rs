//! Basic chat example using the OpenAI provider.
//!
//! This example demonstrates how to use the ferrous-llm-openai crate to send
//! a simple chat request to OpenAI's API and receive a response.
//!
//! To run this example:
//! ```bash
//! export OPENAI_API_KEY="your-api-key-here"
//! cargo run --example openai_chat
//! ```

use ferrous_llm::{
    ChatProvider, ChatRequest, ChatResponse, Message, MessageContent, Metadata, Parameters, Role,
    openai::{OpenAIConfig, OpenAIProvider},
};
use std::error::Error;
use tracing::info;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    dotenv::dotenv().ok();
    // Initialize tracing for better error reporting
    tracing_subscriber::fmt::init();

    info!("🤖 OpenAI Chat Example");
    info!("======================");

    // Load configuration from environment variables
    // Requires OPENAI_API_KEY to be set
    let config = OpenAIConfig::from_env()
        .map_err(|e| format!("Failed to load config from environment: {}", e))?;

    info!("✅ Configuration loaded successfully");
    info!("📝 Model: {}", config.model);

    // Create the OpenAI provider
    let provider = OpenAIProvider::new(config)
        .map_err(|e| format!("Failed to create OpenAI provider: {}", e))?;

    info!("🔗 Provider created successfully");

    // Create a simple chat request
    let request = ChatRequest {
        messages: vec![
            Message {
                role: Role::System,
                content: MessageContent::Text(
                    "You are a helpful assistant that provides concise and informative responses."
                        .to_string(),
                ),
                name: None,
                tool_calls: None,
                tool_call_id: None,
                created_at: chrono::Utc::now(),
            },
            Message {
                role: Role::User,
                content: MessageContent::Text(
                    "Hello! Can you explain what Rust is in one paragraph?".to_string(),
                ),
                name: None,
                tool_calls: None,
                tool_call_id: None,
                created_at: chrono::Utc::now(),
            },
        ],
        parameters: Parameters {
            temperature: Some(0.7),
            max_tokens: Some(150),
            top_p: Some(1.0),
            top_k: None,
            stop_sequences: vec![],
            frequency_penalty: None,
            presence_penalty: None,
        },
        metadata: Metadata {
            extensions: std::collections::HashMap::new(),
            request_id: Some("example-chat-001".to_string()),
            user_id: Some("example-user".to_string()),
            created_at: chrono::Utc::now(),
        },
    };

    info!("📤 Sending chat request...");

    // Send the chat request
    let response = provider
        .chat(request)
        .await
        .map_err(|e| format!("Chat request failed: {}", e))?;

    info!("📥 Response received!");

    // Display the response
    info!("🤖 Assistant Response:");
    info!("─────────────────────");
    info!("{}", response.content());

    // Display usage information if available
    if let Some(usage) = response.usage() {
        info!("📊 Usage Statistics:");
        info!("   • Prompt tokens: {}", usage.prompt_tokens);
        info!("   • Completion tokens: {}", usage.completion_tokens);
        info!("   • Total tokens: {}", usage.total_tokens);
    }

    // Display finish reason
    if let Some(finish_reason) = response.finish_reason() {
        info!("🏁 Finish reason: {:?}", finish_reason);
    }

    // Display metadata
    let metadata = response.metadata();
    if let Some(request_id) = metadata.request_id {
        info!("🔍 Request ID: {}", request_id);
    }

    info!("✅ Example completed successfully!");

    Ok(())
}

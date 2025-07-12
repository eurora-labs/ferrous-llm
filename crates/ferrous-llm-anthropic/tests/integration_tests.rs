//! Integration tests for the Anthropic provider.

use ferrous_llm_anthropic::{AnthropicConfig, AnthropicProvider};

#[cfg(feature = "e2e-tests")]
mod e2e {
    use super::*;
    use dotenv::dotenv;
    use ferrous_llm_core::{
        ChatProvider, ChatRequest, ChatResponse, Message, Metadata, Parameters, StreamingProvider,
    };
    use futures::StreamExt;

    fn create_test_config() -> AnthropicConfig {
        dotenv().ok();
        AnthropicConfig::from_env().expect("ANTHROPIC_API_KEY must be set for e2e tests")
    }

    #[tokio::test]
    async fn test_basic_chat() {
        let config = create_test_config();
        let provider = AnthropicProvider::new(config).expect("Failed to create provider");

        let request = ChatRequest {
            messages: vec![Message::user("Hello! Please respond with just 'Hi there!'")],
            parameters: Parameters {
                max_tokens: Some(50),
                temperature: Some(0.1),
                ..Default::default()
            },
            metadata: Metadata::default(),
        };

        let response = provider.chat(request).await.expect("Chat request failed");

        // Basic assertions
        assert!(!response.content().is_empty());
        println!("Response: {}", response.content());
    }

    #[tokio::test]
    async fn test_system_message() {
        let config = create_test_config();
        let provider = AnthropicProvider::new(config).expect("Failed to create provider");

        let request = ChatRequest {
            messages: vec![
                Message::system(
                    "You are a helpful assistant that always responds with exactly 3 words.",
                ),
                Message::user("What is the weather?"),
            ],
            parameters: Parameters {
                max_tokens: Some(20),
                temperature: Some(0.1),
                ..Default::default()
            },
            metadata: Metadata::default(),
        };

        let response = provider.chat(request).await.expect("Chat request failed");

        assert!(!response.content().is_empty());
        println!("System message response: {}", response.content());
    }

    #[tokio::test]
    async fn test_conversation() {
        let config = create_test_config();
        let provider = AnthropicProvider::new(config).expect("Failed to create provider");

        let request = ChatRequest {
            messages: vec![
                Message::user("My name is Alice."),
                Message::assistant("Hello Alice! Nice to meet you."),
                Message::user("What's my name?"),
            ],
            parameters: Parameters {
                max_tokens: Some(50),
                temperature: Some(0.1),
                ..Default::default()
            },
            metadata: Metadata::default(),
        };

        let response = provider.chat(request).await.expect("Chat request failed");

        assert!(!response.content().is_empty());
        assert!(response.content().to_lowercase().contains("alice"));
        println!("Conversation response: {}", response.content());
    }

    #[tokio::test]
    async fn test_streaming() {
        let config = create_test_config();
        let provider = AnthropicProvider::new(config).expect("Failed to create provider");

        let request = ChatRequest {
            messages: vec![Message::user("Count from 1 to 10, one number per line.")],
            parameters: Parameters {
                max_tokens: Some(100),
                temperature: Some(0.1),
                ..Default::default()
            },
            metadata: Metadata::default(),
        };

        let mut stream = provider
            .chat_stream(request)
            .await
            .expect("Streaming failed");
        let mut content = String::new();
        let mut chunk_count = 0;

        while let Some(chunk_result) = stream.next().await {
            match chunk_result {
                Ok(chunk) => {
                    content.push_str(&chunk);
                    chunk_count += 1;
                    print!("{}", chunk);
                }
                Err(e) => panic!("Stream error: {:?}", e),
            }
        }

        println!("\nTotal chunks: {}", chunk_count);
        println!("Full content: {}", content);

        assert!(chunk_count > 1, "Should receive multiple chunks");
        assert!(!content.is_empty(), "Should receive content");
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn test_config_creation() {
        let config = AnthropicConfig::new("sk-ant-test123", "claude-3-5-sonnet-20241022");
        assert_eq!(config.model, "claude-3-5-sonnet-20241022");
        assert_eq!(config.version, "2023-06-01");
    }

    #[test]
    fn test_config_builder() {
        let config = AnthropicConfig::builder()
            .api_key("sk-ant-test123")
            .model("claude-3-haiku-20240307")
            .version("2023-06-01")
            .build();

        assert_eq!(config.model, "claude-3-haiku-20240307");
        assert_eq!(config.version, "2023-06-01");
    }

    #[test]
    fn test_provider_creation() {
        let config = AnthropicConfig::new("sk-ant-test123", "claude-3-5-sonnet-20241022");
        let provider = AnthropicProvider::new(config);
        assert!(provider.is_ok());
    }

    #[test]
    fn test_urls() {
        let config = AnthropicConfig::new("sk-ant-test123", "claude-3-5-sonnet-20241022");
        assert_eq!(
            config.messages_url(),
            "https://api.anthropic.com/v1/messages"
        );
    }
}

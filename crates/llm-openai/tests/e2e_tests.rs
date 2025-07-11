#[cfg(feature = "e2e-tests")]
mod e2e_tests {
    use dotenv::dotenv;
    use llm_core::*;
    use llm_openai::*;
    use std::env;
    use std::time::Duration;

    // These tests require a real OpenAI API key and should only run when explicitly enabled
    // Run with: cargo test --features integration-tests

    fn get_test_config() -> Option<OpenAIConfig> {
        dotenv().ok();
        env::var("OPENAI_API_KEY")
            .ok()
            .map(|key| OpenAIConfig::new(key, "gpt-3.5-turbo"))
    }

    #[tokio::test]
    async fn test_live_chat_completion() {
        let Some(config) = get_test_config() else {
            println!("Skipping live test - OPENAI_API_KEY not set");
            return;
        };

        let provider = OpenAIProvider::new(config).unwrap();
        let request = ChatRequest::builder()
            .message(Message::user("Say 'Hello, World!' and nothing else."))
            .temperature(0.0)
            .max_tokens(10)
            .build();

        let response = provider.chat(request).await;
        match response {
            Ok(resp) => {
                assert!(!resp.content().is_empty());
                println!("Live test response: {}", resp.content());
            }
            Err(e) => {
                panic!("Live test failed (this may be expected): {:?}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_live_embeddings() {
        let Some(config) = get_test_config() else {
            println!("Skipping live test - OPENAI_API_KEY not set");
            return;
        };

        let provider = OpenAIProvider::new(config).unwrap();
        let texts = vec!["Hello, world!".to_string()];

        let response = provider.embed(&texts).await;
        match response {
            Ok(embeddings) => {
                assert_eq!(embeddings.len(), 1);
                assert!(!embeddings[0].embedding.is_empty());
                println!(
                    "Live embedding test passed, got {} dimensions",
                    embeddings[0].embedding.len()
                );
            }
            Err(e) => {
                panic!("Live embedding test failed (this may be expected): {:?}", e);
            }
        }
    }
}

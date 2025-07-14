//! gRPC provider for ferrous-llm.
//!
//! This crate provides gRPC-based implementations of the ChatProvider and StreamingProvider traits.

pub mod proto {
    pub mod chat {
        include!("./gen/ferrous_llm.chat.rs");
    }
}

pub mod config;
pub mod error;
pub mod provider;
pub mod types;

// Re-export main types for convenience
pub use config::GrpcConfig;
pub use error::GrpcError;
pub use provider::{GrpcChatProvider, GrpcStreamingProvider};
pub use types::{GrpcChatResponse, GrpcStreamResponse};

#[cfg(test)]
mod tests {
    use super::*;
    use url::Url;

    #[test]
    fn test_grpc_config_creation() {
        let config = GrpcConfig::new(Url::parse("http://localhost:50051").unwrap());
        assert_eq!(config.endpoint.to_string(), "http://localhost:50051/");
        assert!(!config.use_tls);
    }

    #[test]
    fn test_grpc_config_with_tls() {
        let config = GrpcConfig::new(Url::parse("https://api.example.com").unwrap())
            .with_tls(Some("api.example.com".to_string()))
            .with_auth_token("test-token".to_string());

        assert!(config.use_tls);
        assert_eq!(config.tls_domain, Some("api.example.com".to_string()));
        assert_eq!(config.auth_token, Some("test-token".to_string()));
    }

    #[tokio::test]
    async fn test_grpc_chat_provider_creation_fails_with_invalid_endpoint() {
        let config = GrpcConfig::new(Url::parse("http://invalid-endpoint:8080").unwrap());

        // This should fail because the endpoint is not reachable
        let result = GrpcChatProvider::new(config).await;
        assert!(result.is_err());
    }
}

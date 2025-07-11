//! Ollama provider implementation.

use crate::{config::OllamaConfig, error::OllamaError, types::*};
use async_trait::async_trait;
use futures::Stream;
use llm_core::{
    ChatProvider, ChatRequest, CompletionProvider, CompletionRequest, Embedding, EmbeddingProvider,
    ProviderResult, StreamingProvider,
};
use reqwest::{Client, RequestBuilder};
use serde_json::json;
use std::pin::Pin;
use tokio_stream::{StreamExt, wrappers::ReceiverStream};

/// Ollama provider implementation.
pub struct OllamaProvider {
    config: OllamaConfig,
    client: Client,
}

impl OllamaProvider {
    /// Create a new Ollama provider with the given configuration.
    pub fn new(config: OllamaConfig) -> Result<Self, OllamaError> {
        let mut headers = reqwest::header::HeaderMap::new();

        // Add content type header
        headers.insert(
            reqwest::header::CONTENT_TYPE,
            "application/json"
                .parse()
                .map_err(|_| OllamaError::Config {
                    source: llm_core::ConfigError::invalid_value("headers", "Invalid content type"),
                })?,
        );

        // Add user agent
        if let Some(ref user_agent) = config.http.user_agent {
            headers.insert(
                reqwest::header::USER_AGENT,
                user_agent.parse().map_err(|_| OllamaError::Config {
                    source: llm_core::ConfigError::invalid_value(
                        "user_agent",
                        "Invalid user agent format",
                    ),
                })?,
            );
        }

        // Add custom headers
        for (key, value) in &config.http.headers {
            let header_name: reqwest::header::HeaderName =
                key.parse().map_err(|_| OllamaError::Config {
                    source: llm_core::ConfigError::invalid_value("headers", "Invalid header name"),
                })?;
            let header_value: reqwest::header::HeaderValue =
                value.parse().map_err(|_| OllamaError::Config {
                    source: llm_core::ConfigError::invalid_value("headers", "Invalid header value"),
                })?;
            headers.insert(header_name, header_value);
        }

        let mut client_builder = Client::builder()
            .timeout(config.http.timeout)
            .default_headers(headers);

        // Configure compression
        if !config.http.compression {
            client_builder = client_builder.no_gzip();
        }

        // Configure connection pool
        client_builder = client_builder
            .pool_max_idle_per_host(config.http.pool.max_idle_connections)
            .pool_idle_timeout(config.http.pool.idle_timeout)
            .connect_timeout(config.http.pool.connect_timeout);

        let client = client_builder
            .build()
            .map_err(|e| OllamaError::Network { source: e })?;

        Ok(Self { config, client })
    }

    /// Create a request builder with common settings.
    fn request_builder(&self, method: reqwest::Method, url: &str) -> RequestBuilder {
        self.client.request(method, url)
    }

    /// Handle HTTP response and convert to appropriate error.
    async fn handle_response<T>(&self, response: reqwest::Response) -> Result<T, OllamaError>
    where
        T: serde::de::DeserializeOwned,
    {
        let status = response.status();

        if status.is_success() {
            response
                .json()
                .await
                .map_err(|e| OllamaError::Network { source: e })
        } else {
            let body = response.text().await.unwrap_or_default();
            Err(OllamaError::from_response(status.as_u16(), &body))
        }
    }

    /// Convert core ChatRequest to Ollama format.
    fn convert_chat_request(&self, request: &ChatRequest) -> OllamaChatRequest {
        let mut ollama_request = OllamaChatRequest {
            model: self.config.model.clone(),
            messages: request.messages.iter().map(|m| m.into()).collect(),
            stream: Some(false),
            format: None,
            options: self.config.options.clone(),
            keep_alive: self.config.keep_alive.map(|ka| format!("{ka}s")),
        };

        // Apply parameters to options
        if let Some(ref mut options) = ollama_request.options {
            if let Some(temp) = request.parameters.temperature {
                options["temperature"] = json!(temp);
            }
            if let Some(max_tokens) = request.parameters.max_tokens {
                options["num_predict"] = json!(max_tokens);
            }
            if let Some(top_p) = request.parameters.top_p {
                options["top_p"] = json!(top_p);
            }
            if let Some(top_k) = request.parameters.top_k {
                options["top_k"] = json!(top_k);
            }
            if !request.parameters.stop_sequences.is_empty() {
                options["stop"] = json!(request.parameters.stop_sequences);
            }
        } else if request.parameters.temperature.is_some()
            || request.parameters.max_tokens.is_some()
            || request.parameters.top_p.is_some()
            || request.parameters.top_k.is_some()
            || !request.parameters.stop_sequences.is_empty()
        {
            let mut options = serde_json::Map::new();
            if let Some(temp) = request.parameters.temperature {
                options.insert("temperature".to_string(), json!(temp));
            }
            if let Some(max_tokens) = request.parameters.max_tokens {
                options.insert("num_predict".to_string(), json!(max_tokens));
            }
            if let Some(top_p) = request.parameters.top_p {
                options.insert("top_p".to_string(), json!(top_p));
            }
            if let Some(top_k) = request.parameters.top_k {
                options.insert("top_k".to_string(), json!(top_k));
            }
            if !request.parameters.stop_sequences.is_empty() {
                options.insert("stop".to_string(), json!(request.parameters.stop_sequences));
            }
            ollama_request.options = Some(serde_json::Value::Object(options));
        }

        ollama_request
    }

    /// Convert core CompletionRequest to Ollama format.
    fn convert_completion_request(&self, request: &CompletionRequest) -> OllamaCompletionRequest {
        let mut ollama_request = OllamaCompletionRequest {
            model: self.config.model.clone(),
            prompt: request.prompt.clone(),
            stream: Some(false),
            format: None,
            options: self.config.options.clone(),
            keep_alive: self.config.keep_alive.map(|ka| format!("{ka}s")),
            context: None,
        };

        // Apply parameters to options
        if let Some(ref mut options) = ollama_request.options {
            if let Some(temp) = request.parameters.temperature {
                options["temperature"] = json!(temp);
            }
            if let Some(max_tokens) = request.parameters.max_tokens {
                options["num_predict"] = json!(max_tokens);
            }
            if let Some(top_p) = request.parameters.top_p {
                options["top_p"] = json!(top_p);
            }
            if let Some(top_k) = request.parameters.top_k {
                options["top_k"] = json!(top_k);
            }
            if !request.parameters.stop_sequences.is_empty() {
                options["stop"] = json!(request.parameters.stop_sequences);
            }
        } else if request.parameters.temperature.is_some()
            || request.parameters.max_tokens.is_some()
            || request.parameters.top_p.is_some()
            || request.parameters.top_k.is_some()
            || !request.parameters.stop_sequences.is_empty()
        {
            let mut options = serde_json::Map::new();
            if let Some(temp) = request.parameters.temperature {
                options.insert("temperature".to_string(), json!(temp));
            }
            if let Some(max_tokens) = request.parameters.max_tokens {
                options.insert("num_predict".to_string(), json!(max_tokens));
            }
            if let Some(top_p) = request.parameters.top_p {
                options.insert("top_p".to_string(), json!(top_p));
            }
            if let Some(top_k) = request.parameters.top_k {
                options.insert("top_k".to_string(), json!(top_k));
            }
            if !request.parameters.stop_sequences.is_empty() {
                options.insert("stop".to_string(), json!(request.parameters.stop_sequences));
            }
            ollama_request.options = Some(serde_json::Value::Object(options));
        }

        ollama_request
    }
}

#[async_trait]
impl ChatProvider for OllamaProvider {
    type Config = OllamaConfig;
    type Response = OllamaChatResponse;
    type Error = OllamaError;

    async fn chat(&self, request: ChatRequest) -> ProviderResult<Self::Response, Self::Error> {
        let ollama_request = self.convert_chat_request(&request);

        let response = self
            .request_builder(reqwest::Method::POST, &self.config.chat_url())
            .json(&ollama_request)
            .send()
            .await
            .map_err(|e| OllamaError::Network { source: e })?;

        self.handle_response(response).await
    }
}

#[async_trait]
impl CompletionProvider for OllamaProvider {
    type Config = OllamaConfig;
    type Response = OllamaCompletionResponse;
    type Error = OllamaError;

    async fn complete(
        &self,
        request: CompletionRequest,
    ) -> ProviderResult<Self::Response, Self::Error> {
        let ollama_request = self.convert_completion_request(&request);

        let response = self
            .request_builder(reqwest::Method::POST, &self.config.generate_url())
            .json(&ollama_request)
            .send()
            .await
            .map_err(|e| OllamaError::Network { source: e })?;

        self.handle_response(response).await
    }
}

#[async_trait]
impl EmbeddingProvider for OllamaProvider {
    type Config = OllamaConfig;
    type Error = OllamaError;

    async fn embed(&self, texts: &[String]) -> ProviderResult<Vec<Embedding>, Self::Error> {
        let embedding_model = self
            .config
            .embedding_model
            .clone()
            .unwrap_or_else(|| "nomic-embed-text".to_string());

        let mut embeddings = Vec::new();

        // Ollama embeddings API processes one text at a time
        for (index, text) in texts.iter().enumerate() {
            let request = OllamaEmbeddingsRequest {
                model: embedding_model.clone(),
                prompt: text.clone(),
                options: self.config.options.clone(),
                keep_alive: self.config.keep_alive.map(|ka| format!("{ka}s")),
            };

            let response = self
                .request_builder(reqwest::Method::POST, &self.config.embeddings_url())
                .json(&request)
                .send()
                .await
                .map_err(|e| OllamaError::Network { source: e })?;

            let embeddings_response: OllamaEmbeddingsResponse =
                self.handle_response(response).await?;

            embeddings.push(Embedding {
                embedding: embeddings_response.embedding,
                index,
            });
        }

        Ok(embeddings)
    }
}

#[async_trait]
impl StreamingProvider for OllamaProvider {
    type StreamItem = String;
    type Stream = Pin<Box<dyn Stream<Item = Result<Self::StreamItem, Self::Error>> + Send>>;

    async fn chat_stream(&self, request: ChatRequest) -> ProviderResult<Self::Stream, Self::Error> {
        let mut ollama_request = self.convert_chat_request(&request);
        ollama_request.stream = Some(true);

        let response = self
            .request_builder(reqwest::Method::POST, &self.config.chat_url())
            .json(&ollama_request)
            .send()
            .await
            .map_err(|e| OllamaError::Network { source: e })?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let body = response.text().await.unwrap_or_default();
            return Err(OllamaError::from_response(status, &body));
        }

        // Create a tokio channel for streaming
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<String, OllamaError>>(100);

        // Spawn a task to process the streaming response
        let tx_clone = tx.clone();
        tokio::spawn(async move {
            let mut byte_stream = response.bytes_stream();
            let mut buffer = Vec::new();

            while let Some(chunk_result) = byte_stream.next().await {
                match chunk_result {
                    Ok(chunk) => {
                        buffer.extend_from_slice(chunk.as_ref());

                        // Process complete lines (Ollama sends one JSON object per line)
                        let mut start = 0;
                        while let Some(pos) = buffer[start..].iter().position(|&b| b == b'\n') {
                            let line_end = start + pos;
                            let line = String::from_utf8_lossy(&buffer[start..line_end])
                                .trim()
                                .to_string();
                            start = line_end + 1;

                            if !line.is_empty() {
                                // Try to parse the JSON chunk
                                if let Ok(chunk) = serde_json::from_str::<OllamaStreamChunk>(&line)
                                {
                                    // Extract content from the chunk
                                    let content = if let Some(ref message) = chunk.message {
                                        message.content.as_str()
                                    } else {
                                        chunk.response.as_deref().unwrap_or_default()
                                    };

                                    if !content.is_empty()
                                        && tx_clone.send(Ok(content.to_string())).await.is_err()
                                    {
                                        // Receiver dropped
                                        return;
                                    }

                                    // Check if this is the final chunk
                                    if chunk.done {
                                        drop(tx_clone);
                                        return;
                                    }
                                }
                            }
                        }

                        // Keep remaining bytes in buffer
                        buffer.drain(0..start);
                    }
                    Err(e) => {
                        let _ = tx_clone.send(Err(OllamaError::Network { source: e })).await;
                        return;
                    }
                }
            }

            // Close the channel when done
            drop(tx_clone);
        });

        // Convert the receiver to a stream
        let content_stream = ReceiverStream::new(rx);

        Ok(Box::pin(content_stream))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm_core::{Message, Metadata, Parameters};

    fn create_test_config() -> OllamaConfig {
        OllamaConfig::new("llama2")
    }

    #[test]
    fn test_provider_creation() {
        let config = create_test_config();
        let provider = OllamaProvider::new(config);
        assert!(provider.is_ok());
    }

    #[test]
    fn test_convert_chat_request() {
        let config = create_test_config();
        let provider = OllamaProvider::new(config).unwrap();

        let request = ChatRequest {
            messages: vec![Message::user("Hello")],
            parameters: Parameters {
                temperature: Some(0.7),
                max_tokens: Some(100),
                ..Default::default()
            },
            metadata: Metadata::default(),
        };

        let ollama_request = provider.convert_chat_request(&request);
        assert_eq!(ollama_request.model, "llama2");
        assert_eq!(ollama_request.messages.len(), 1);
        assert_eq!(ollama_request.stream, Some(false));
    }

    #[test]
    fn test_convert_completion_request() {
        let config = create_test_config();
        let provider = OllamaProvider::new(config).unwrap();

        let request = CompletionRequest {
            prompt: "Complete this sentence".to_string(),
            parameters: Parameters {
                temperature: Some(0.8),
                max_tokens: Some(50),
                ..Default::default()
            },
            metadata: Metadata::default(),
        };

        let ollama_request = provider.convert_completion_request(&request);
        assert_eq!(ollama_request.model, "llama2");
        assert_eq!(ollama_request.prompt, "Complete this sentence");
        assert_eq!(ollama_request.stream, Some(false));
    }

    #[test]
    fn test_config_urls() {
        let config = create_test_config();
        assert_eq!(config.chat_url(), "http://localhost:11434/api/chat");
        assert_eq!(config.generate_url(), "http://localhost:11434/api/generate");
        assert_eq!(
            config.embeddings_url(),
            "http://localhost:11434/api/embeddings"
        );
    }
}

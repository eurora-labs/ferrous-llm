//! OpenAI provider configuration.

use llm_core::{ConfigError, HttpConfig, ProviderConfig, SecretString, validation};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use url::Url;

/// Configuration for the OpenAI provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIConfig {
    /// OpenAI API key
    pub api_key: SecretString,

    /// Model to use (e.g., "gpt-4", "gpt-3.5-turbo")
    pub model: String,

    /// Base URL for the OpenAI API (defaults to https://api.openai.com/v1)
    pub base_url: Option<Url>,

    /// Organization ID (optional)
    pub organization: Option<String>,

    /// Project ID (optional)
    pub project: Option<String>,

    /// HTTP client configuration
    pub http: HttpConfig,
}

impl Default for OpenAIConfig {
    fn default() -> Self {
        Self {
            api_key: SecretString::new(""),
            model: "gpt-3.5-turbo".to_string(),
            base_url: None,
            organization: None,
            project: None,
            http: HttpConfig::default(),
        }
    }
}

impl ProviderConfig for OpenAIConfig {
    type Provider = crate::provider::OpenAIProvider;

    fn build(self) -> Result<Self::Provider, ConfigError> {
        self.validate()?;
        crate::provider::OpenAIProvider::new(self).map_err(|e| match e {
            crate::error::OpenAIError::Config { source } => source,
            _ => ConfigError::validation_failed("Failed to create provider"),
        })
    }

    fn validate(&self) -> Result<(), ConfigError> {
        // Validate API key
        validation::validate_api_key(&self.api_key, "api_key")?;

        // Validate model name
        validation::validate_model_name(&self.model, "model")?;

        // Validate base URL if provided
        if let Some(ref url) = self.base_url {
            validation::validate_https_url(url, "base_url")?;
        }

        // Validate HTTP configuration
        validation::validate_positive_duration(self.http.timeout, "http.timeout")?;
        validation::validate_range(self.http.max_retries, 0, 10, "http.max_retries")?;

        Ok(())
    }
}

impl OpenAIConfig {
    /// Create a new OpenAI configuration with the given API key and model.
    pub fn new(api_key: impl Into<SecretString>, model: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model: model.into(),
            ..Default::default()
        }
    }

    /// Create a configuration builder.
    pub fn builder() -> OpenAIConfigBuilder {
        OpenAIConfigBuilder::new()
    }

    /// Get the base URL for API requests.
    pub fn base_url(&self) -> &str {
        self.base_url
            .as_ref()
            .map(|u| u.as_str())
            .unwrap_or("https://api.openai.com/v1")
    }

    /// Get the chat completions endpoint URL.
    pub fn chat_url(&self) -> String {
        format!("{}/chat/completions", self.base_url())
    }

    /// Get the completions endpoint URL.
    pub fn completions_url(&self) -> String {
        format!("{}/completions", self.base_url())
    }

    /// Get the embeddings endpoint URL.
    pub fn embeddings_url(&self) -> String {
        format!("{}/embeddings", self.base_url())
    }

    /// Get the images endpoint URL.
    pub fn images_url(&self) -> String {
        format!("{}/images/generations", self.base_url())
    }

    /// Get the audio transcriptions endpoint URL.
    pub fn transcriptions_url(&self) -> String {
        format!("{}/audio/transcriptions", self.base_url())
    }

    /// Get the audio speech endpoint URL.
    pub fn speech_url(&self) -> String {
        format!("{}/audio/speech", self.base_url())
    }

    /// Load configuration from environment variables.
    pub fn from_env() -> Result<Self, ConfigError> {
        use llm_core::env;

        let api_key = env::required_secret("OPENAI_API_KEY")?;
        let model = env::with_default("OPENAI_MODEL", "gpt-3.5-turbo");
        let organization = env::optional("OPENAI_ORGANIZATION");
        let project = env::optional("OPENAI_PROJECT");

        let base_url = if let Some(url_str) = env::optional("OPENAI_BASE_URL") {
            Some(validation::validate_url(&url_str, "OPENAI_BASE_URL")?)
        } else {
            None
        };

        Ok(Self {
            api_key,
            model,
            base_url,
            organization,
            project,
            http: HttpConfig::default(),
        })
    }
}

/// Builder for OpenAI configuration.
pub struct OpenAIConfigBuilder {
    config: OpenAIConfig,
}

impl OpenAIConfigBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            config: OpenAIConfig::default(),
        }
    }

    /// Set the API key.
    pub fn api_key(mut self, api_key: impl Into<SecretString>) -> Self {
        self.config.api_key = api_key.into();
        self
    }

    /// Set the model.
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.config.model = model.into();
        self
    }

    /// Set the base URL.
    pub fn base_url(mut self, base_url: impl Into<String>) -> Result<Self, ConfigError> {
        let url = validation::validate_url(&base_url.into(), "base_url")?;
        self.config.base_url = Some(url);
        Ok(self)
    }

    /// Set the organization.
    pub fn organization(mut self, organization: impl Into<String>) -> Self {
        self.config.organization = Some(organization.into());
        self
    }

    /// Set the project.
    pub fn project(mut self, project: impl Into<String>) -> Self {
        self.config.project = Some(project.into());
        self
    }

    /// Set the request timeout.
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.config.http.timeout = timeout;
        self
    }

    /// Set the maximum number of retries.
    pub fn max_retries(mut self, max_retries: u32) -> Self {
        self.config.http.max_retries = max_retries;
        self
    }

    /// Set a custom HTTP header.
    pub fn header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.config.http.headers.insert(key.into(), value.into());
        self
    }

    /// Build the configuration.
    pub fn build(self) -> OpenAIConfig {
        self.config
    }
}

impl Default for OpenAIConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_validation() {
        let config = OpenAIConfig::new("sk-test123456789", "gpt-4");
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validation_empty_api_key() {
        let config = OpenAIConfig::new("", "gpt-4");
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_builder() {
        let config = OpenAIConfig::builder()
            .api_key("sk-test123456789")
            .model("gpt-4")
            .organization("org-123")
            .timeout(Duration::from_secs(60))
            .build();

        assert_eq!(config.model, "gpt-4");
        assert_eq!(config.organization, Some("org-123".to_string()));
        assert_eq!(config.http.timeout, Duration::from_secs(60));
    }

    #[test]
    fn test_urls() {
        let config = OpenAIConfig::new("sk-test", "gpt-4");
        assert_eq!(
            config.chat_url(),
            "https://api.openai.com/v1/chat/completions"
        );
        assert_eq!(
            config.embeddings_url(),
            "https://api.openai.com/v1/embeddings"
        );
    }

    #[test]
    fn test_custom_base_url() {
        let mut config = OpenAIConfig::new("sk-test", "gpt-4");
        config.base_url = Some("https://custom.openai.com/v1".parse().unwrap());
        assert_eq!(
            config.chat_url(),
            "https://custom.openai.com/v1/chat/completions"
        );
    }
}

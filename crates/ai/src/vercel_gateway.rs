//! Direct, account-free transport to an OpenAI-compatible inference endpoint
//! such as the [Vercel AI Gateway](https://vercel.com/docs/ai-gateway).
//!
//! Warp's normal AI path routes every request through Warp's hosted GraphQL
//! backend (`/graphql/v2`), which requires a Warp account/auth token and
//! performs provider routing server-side. In local-only mode (see
//! [`warp_core::channel::ChannelState::is_local_only`]) we instead talk
//! *directly* to a user-configured OpenAI-compatible endpoint using the user's
//! own API key, so no Warp account is involved and nothing leaves the machine
//! except the request to the user's chosen provider.
//!
//! The Vercel AI Gateway exposes an OpenAI-compatible Chat Completions API, so
//! the same transport works for the gateway, OpenAI itself, OpenRouter, a local
//! Ollama/LM Studio server, or any other compatible endpoint. Users configure
//! these as [`CustomEndpoint`]s in settings; this module turns one of those
//! into a working request.
//!
//! This is intentionally a minimal, non-streaming "one completion" path used as
//! the local-mode scaffold. Streaming and full tool-calling parity with Warp's
//! hosted agent can be layered on top of [`GatewayClient`] later.

use http_client::Client;
use serde::{Deserialize, Serialize};

use crate::api_keys::CustomEndpoint;

/// Default base URL for the Vercel AI Gateway's OpenAI-compatible API.
///
/// Used when a [`CustomEndpoint`] is configured with an empty URL. Users supply
/// their own gateway API key.
pub const VERCEL_AI_GATEWAY_BASE_URL: &str = "https://ai-gateway.vercel.sh/v1";

/// A single chat message in the OpenAI Chat Completions format.
#[derive(Debug, Clone, Serialize)]
pub struct ChatMessage {
    /// One of `system`, `user`, or `assistant`.
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".into(),
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".into(),
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".into(),
            content: content.into(),
        }
    }
}

/// A client that sends OpenAI-compatible chat completions directly to a
/// user-configured endpoint, bypassing Warp's backend entirely.
pub struct GatewayClient {
    base_url: String,
    api_key: String,
    http: Client,
}

impl GatewayClient {
    /// Create a client for the given OpenAI-compatible base URL and API key.
    ///
    /// `base_url` should include the API version segment (e.g. `.../v1`); a
    /// trailing slash is tolerated.
    pub fn new(base_url: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            api_key: api_key.into(),
            http: Client::new(),
        }
    }

    /// Build a client from a user-configured [`CustomEndpoint`].
    ///
    /// An empty endpoint URL falls back to [`VERCEL_AI_GATEWAY_BASE_URL`]. A
    /// missing API key is an error, since BYO-key endpoints require one.
    pub fn from_custom_endpoint(endpoint: &CustomEndpoint) -> anyhow::Result<Self> {
        let api_key = endpoint.api_key.trim();
        if api_key.is_empty() {
            anyhow::bail!("custom endpoint \"{}\" has no API key", endpoint.name);
        }
        let url = endpoint.url.trim();
        let base_url = if url.is_empty() {
            VERCEL_AI_GATEWAY_BASE_URL.to_string()
        } else {
            url.trim_end_matches('/').to_string()
        };
        Ok(Self::new(base_url, api_key.to_string()))
    }

    fn completions_url(&self) -> String {
        format!("{}/chat/completions", self.base_url.trim_end_matches('/'))
    }

    /// Send a chat completion request and return the assistant's reply text.
    pub async fn complete(
        &self,
        model: &str,
        messages: &[ChatMessage],
    ) -> anyhow::Result<String> {
        let body = ChatRequest {
            model,
            messages,
            stream: false,
        };

        let response = self
            .http
            .post(self.completions_url())
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("request to inference endpoint failed: {e}"))?
            .error_for_status_with_body()
            .await?;

        let parsed: ChatResponse = response
            .json()
            .await
            .map_err(|e| anyhow::anyhow!("failed to parse inference response: {e}"))?;

        parsed
            .choices
            .into_iter()
            .next()
            .map(|choice| choice.message.content)
            .ok_or_else(|| anyhow::anyhow!("inference endpoint returned no choices"))
    }
}

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: &'a [ChatMessage],
    stream: bool,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Deserialize)]
struct ChatChoice {
    message: ChatResponseMessage,
}

#[derive(Deserialize)]
struct ChatResponseMessage {
    #[serde(default)]
    content: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completions_url_is_appended_to_base() {
        let client = GatewayClient::new("https://ai-gateway.vercel.sh/v1", "sk-test");
        assert_eq!(
            client.completions_url(),
            "https://ai-gateway.vercel.sh/v1/chat/completions"
        );
    }

    #[test]
    fn trailing_slash_in_base_url_is_tolerated() {
        let client = GatewayClient::new("https://example.com/v1/", "sk-test");
        assert_eq!(
            client.completions_url(),
            "https://example.com/v1/chat/completions"
        );
    }

    #[test]
    fn empty_endpoint_url_defaults_to_vercel_gateway() {
        let endpoint = CustomEndpoint {
            name: "gateway".into(),
            url: String::new(),
            api_key: "sk-test".into(),
            models: Vec::new(),
        };
        let client = GatewayClient::from_custom_endpoint(&endpoint).unwrap();
        assert_eq!(
            client.completions_url(),
            format!("{VERCEL_AI_GATEWAY_BASE_URL}/chat/completions")
        );
    }

    #[test]
    fn missing_api_key_is_rejected() {
        let endpoint = CustomEndpoint {
            name: "gateway".into(),
            url: VERCEL_AI_GATEWAY_BASE_URL.into(),
            api_key: "   ".into(),
            models: Vec::new(),
        };
        assert!(GatewayClient::from_custom_endpoint(&endpoint).is_err());
    }

    #[test]
    fn response_parsing_extracts_first_choice() {
        let raw = serde_json::json!({
            "choices": [
                { "message": { "role": "assistant", "content": "hello from local AI" } }
            ]
        });
        let parsed: ChatResponse = serde_json::from_value(raw).unwrap();
        assert_eq!(parsed.choices.len(), 1);
        assert_eq!(parsed.choices[0].message.content, "hello from local AI");
    }
}

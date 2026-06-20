pub mod agent;
pub mod api_keys;
pub mod aws_credentials;
pub mod geap_credentials;
#[cfg(not(target_family = "wasm"))]
pub mod grok_subscription;
pub mod llm_id;

pub use llm_id::LLMId;
pub mod diff_validation;
pub mod document;
pub mod gfm_table;
pub mod index;
pub mod paths;
pub mod project_context;
pub mod skills;
mod telemetry;
/// Direct, account-free transport to OpenAI-compatible inference endpoints
/// (e.g. the Vercel AI Gateway) for local-only mode.
#[cfg(not(target_family = "wasm"))]
pub mod vercel_gateway;
pub mod workspace;

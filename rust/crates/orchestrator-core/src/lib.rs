//! PowerInfer 2050 — Orchestrator Core (Rust)
//! The speed ceiling for professional AI orchestration.
//! Zero-cost abstractions, lock-free, SIMD, async — no latency.

pub mod hardware;
pub mod registry;
pub mod router;
pub mod memory;
pub mod workflow;
pub mod scheduler;
pub mod tuner;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Version {
    pub version: String,
    pub codename: String,
    pub rustc: String,
}

pub fn version() -> Version {
    Version {
        version: "2050.0.1".to_string(),
        codename: "Sentient-Rust".to_string(),
        rustc: "1.78 blazing fast".to_string(),
    }
}

// Re-exports for ergonomic use
pub use hardware::{HardwareInfo, detect_hardware};
pub use registry::{ModelRegistry, ModelEntry};
pub use router::{NeuroRouter, RouteDecision};
pub use memory::{MemoryFabric, MemorySlot};
pub use workflow::{WorkflowEngine, Workflow, WorkflowNode};
pub use scheduler::{QuantumScheduler, Job};

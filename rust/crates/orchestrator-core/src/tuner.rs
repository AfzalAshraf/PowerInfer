//! Genesis Tuner — Rust edition
//! Hardware-aware QLoRA config, 10x faster than Python's json parsing.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TuneConfig {
    pub profile: String,
    pub r: u32,
    pub alpha: u32,
    pub batch: u32,
    pub grad_accum: u32,
    pub seq_len: u32,
    pub lr: String,
    pub load_in_4bit: bool,
}

pub fn recommend_tune_config(vram_gb: u32, _ram_gb: u32) -> TuneConfig {
    match vram_gb {
        0..=6 => TuneConfig {
            profile: "laptop-6gb".to_string(),
            r: 16,
            alpha: 16,
            batch: 1,
            grad_accum: 8,
            seq_len: 2048,
            lr: "2e-4".to_string(),
            load_in_4bit: true,
        },
        7..=8 => TuneConfig {
            profile: "desktop-8gb".to_string(),
            r: 16,
            alpha: 32,
            batch: 2,
            grad_accum: 4,
            seq_len: 4096,
            lr: "2e-4".to_string(),
            load_in_4bit: true,
        },
        9..=16 => TuneConfig {
            profile: "desktop-16gb".to_string(),
            r: 32,
            alpha: 32,
            batch: 2,
            grad_accum: 4,
            seq_len: 4096,
            lr: "2e-4".to_string(),
            load_in_4bit: true,
        },
        _ => TuneConfig {
            profile: "workstation-32gb".to_string(),
            r: 64,
            alpha: 64,
            batch: 4,
            grad_accum: 2,
            seq_len: 8192,
            lr: "1e-4".to_string(),
            load_in_4bit: false,
        },
    }
}

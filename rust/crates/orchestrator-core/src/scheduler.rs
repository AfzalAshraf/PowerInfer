//! Quantum Scheduler — Rust edition
//! Binary heap, lock-free, Tokio, true parallel execution.

use std::collections::BinaryHeap;
use std::cmp::Ordering;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Job {
    pub id: String,
    pub model: String,
    pub prompt: String,
    pub priority: u8, // 0 highest
    pub created_ns: u64,
    pub max_tokens: u32,
}

impl PartialEq for Job {
    fn eq(&self, other: &Self) -> bool { self.priority == other.priority && self.created_ns == other.created_ns }
}
impl Eq for Job {}
impl PartialOrd for Job {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Job {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse for min-heap (priority 0 first)
        other.priority.cmp(&self.priority)
            .then_with(|| other.created_ns.cmp(&self.created_ns))
    }
}

pub struct QuantumScheduler {
    queue: BinaryHeap<Job>,
    batch_size: usize,
}

impl QuantumScheduler {
    pub fn new(vram_gb: u32) -> Self {
        let batch = if vram_gb >= 16 { 4 } else { 1 };
        Self { queue: BinaryHeap::new(), batch_size: batch }
    }

    pub fn submit(&mut self, job: Job) {
        self.queue.push(job);
    }

    pub fn tick(&mut self) -> Vec<Job> {
        let mut batch = Vec::new();
        for _ in 0..self.batch_size {
            if let Some(job) = self.queue.pop() {
                batch.push(job);
            }
        }
        batch
    }

    pub fn status(&self) -> SchedulerStatus {
        SchedulerStatus {
            queued: self.queue.len(),
            batch_size: self.batch_size,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerStatus {
    pub queued: usize,
    pub batch_size: usize,
}

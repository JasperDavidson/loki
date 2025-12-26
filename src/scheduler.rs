use priority_queue::DoublePriorityQueue;
use std::collections::{BTreeMap, HashMap, HashSet};
use thiserror::Error;

use crate::allocator::{Allocator, CacheBlockHandle, LayerBlockHandle};
use crate::table::{Table, TableError};

#[derive(PartialEq, Eq, Hash, Clone, Copy)]
struct ModelID(u8);

struct Model {
    id: ModelID,
    layer_size: u32,
    layer_count: u32,
    layer_block: LayerBlockHandle,
    cache_block_size: u32,
}

#[derive(PartialEq, Eq, Hash, Clone, Copy)]
struct RequestID(u32);

#[derive(PartialEq, Eq, Hash, Clone, Copy)]
struct RequestMetadata {
    id: RequestID,
    tier: u8,
    age: u32,
}

impl Ord for RequestMetadata {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .tier
            .cmp(&self.tier)
            .then_with(|| self.age.cmp(&other.age))
    }
}

impl PartialOrd for RequestMetadata {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl RequestMetadata {
    pub fn check_priority_boost(&mut self) {
        if self.age > 100 {
            // choose appropriate threshold
            self.tier += 1;
        }
    }
}

#[derive(PartialEq, Eq, Hash)]
struct Request {
    priority: RequestMetadata,
    model_id: ModelID,
    cache_ids: Vec<CacheBlockHandle>, // Use formula when loading request comp with Model to
                                      // calculate initial # blocks needed, then grow later as needed (i.e. this should be determined by the prefill phase -> only LLMs have prefill though...)
}

struct Registry {
    loaded_models: Vec<ModelID>,
    model_map: HashMap<ModelID, Model>,
    req_to_model: HashMap<RequestID, ModelID>,
    priority_request_map: HashMap<RequestID, Request>,
}

struct Scheduler {
    total_mem: u64,
    cache_free_mem: u64,
    layer_free_mem: u64,
    cache_weight_ratio: f32,

    registry: Registry,
    allocator: Allocator,
    table: Table,

    iter_req_models: HashSet<ModelID>,
    request_priority_queue: DoublePriorityQueue<RequestID, RequestMetadata>,
    active_requests: Vec<RequestID>,
    active_size_map: BTreeMap<RequestMetadata, u64>,
}

#[derive(Error, Debug)]
enum SchedulerError {
    #[error("Model ID not mapped to a valid model")]
    UnmappedModel,

    #[error("Request ID not a valid key")]
    FailedRequestID,

    #[error("Table failure: {0}")]
    TableError(#[from] TableError),
}

impl Scheduler {
    // This function gets ran after every decoding cycle (every model has produced one token)
    pub fn step(&mut self) -> Result<(), SchedulerError> {
        self.evict_finished_requests()?;

        let (new_requests, unfulfilled_requests) = self.schedule_pending_requests()?;

        // Allocate physical memory addresses for each request layer block/cache block -> should
        // the scheduler own an allocator? Or maybe they could communicate via channels?
        // Then compute prefill for new requests (need to look into this)
        // Maybe should maintain a state vector of all the requests currently being under prefill
        // which get moved into the active stage afterwards?
        self.allocate_resources(new_requests)?;

        // Perform iteration level scheduling on all active requests
        // 1. Check if the request needs more KV cache blocks to continue token generation
        self.step_active_requests()?;

        // Handle the unfulfilled data, pushing it back into the priority queue
        // Also handles our memory histogram insertion
        self.handle_unfulfilled(unfulfilled_requests)?;

        Ok(())
    }

    fn evict_finished_requests(&mut self) -> Result<(), SchedulerError> {
        // Check if any requests have finished, if so evict them
        // Maybe update a flag in each request and iterate through
        // - I think it would it would be better if we set a flag when receiving a request from the
        // GPU
        for request in self.active_requests.iter() {}
        Ok(())
    }

    fn schedule_pending_requests(
        &mut self,
    ) -> Result<(Vec<RequestMetadata>, Vec<RequestMetadata>), SchedulerError> {
        let mut unfulfilled_requests = Vec::new();
        let mut new_requests = Vec::new();

        // Fetch new requests based on priority and current gpu utilization
        // Note these should actually be *separate* from the current requests - this is the prefill
        // phase -> still not super sure on this, only if it's a LLM?
        while let Some((request_id, mut metadata)) = self.request_priority_queue.pop_max() {
            let request = self
                .registry
                .priority_request_map
                .get(&request_id)
                .ok_or(SchedulerError::FailedRequestID)?;
            let request_model = self
                .registry
                .model_map
                .get(&request.model_id)
                .ok_or(SchedulerError::UnmappedModel)?;
            let cache_needed = request.cache_ids.len() as u32 * request_model.cache_block_size;

            // Need to decide how to handle eviction of requests layers/kv cache blocks if a
            // higher priority new request needs space

            // Check if there are any lower priority active tasks to pre-empty
            //      - If not, store back in idle queue and increase age + check for eligible
            //    priority boost
            //      - If there is an empty task to pre-empty, page the cache blocks back to
            //      physical memory and allocate high priority task in place

            // Check if the model is already scheduled to be computed next cycle, then reuse
            // weights
            let can_allocate_model = match self.iter_req_models.get(&request_model.id) {
                Some(_) => true,
                None => (request_model.layer_size as u64 * 2) < self.layer_free_mem,
            };
            let can_allocate_cache = (cache_needed as u64) < self.cache_free_mem;
            if can_allocate_cache && can_allocate_model {
                new_requests.push(metadata);
            } else if can_allocate_model {
                // Check evicting other KV caches
                // Idea:
                // We need to check if we can free up to the cache memory required by this new
                // request
                //  - Naively we could continuously pop off the bottom, checking if each request
                // is lower priority, until we accumulate enough memory that we could evict
                // However, if we can't accumulate enough memory before reaching a request of
                // higher priority, we have to push all those requests back -> O(N * log(N))
                //  - Instead, we could maintain a sorted hash table such that every entry
                // contains the amount of "potential" memory stored at a certain priority (key
                // to the hash table)
                //  - Then, when searching for a given priority, simply do an O(N) iteration to
                //  sum over all the lower priorities and check if evicting them will be enough
                //  - Note we maintain the invariant that keys are never modified, since the
                //  priority of a currently active request is static (don't change the age of a
                //  request until it is pre-empted or ends (in which case it's done))
                let lowest_priority = self
                    .request_priority_queue
                    .pop_min()
                    .ok_or(SchedulerError::FailedRequestID)?; // TODO: This shouldn't be an
                // error

                let mut possible_size = 0;
                for size in self.active_size_map.iter() {
                    if *size.0 < lowest_priority.1 {
                        possible_size += *size.1;
                    } else {
                        break;
                    }
                }

                if possible_size > 100 { // 100 is placeholder, calc needed cache size similar to below
                    // evict and alloc for new request
                } else {
                    // mark as unfulfilled and try again later
                }
            } else if can_allocate_cache {
                // Check freeing other model
            } else {
                metadata.age += 1;
                metadata.check_priority_boost();
                unfulfilled_requests.push(metadata);
            }
        }

        Ok((new_requests, unfulfilled_requests))
    }

    fn allocate_resources(
        &mut self,
        new_requests: Vec<RequestMetadata>,
    ) -> Result<(), SchedulerError> {
        for new_data in new_requests {
            let request = self
                .registry
                .priority_request_map
                .get(&new_data.id)
                .ok_or(SchedulerError::FailedRequestID)?;

            // Need to add more errors for this sequence and/or it can likely be cleaned up?
            let req_model = self
                .registry
                .req_to_model
                .get(&new_data.id)
                .ok_or(SchedulerError::FailedRequestID)?;
            let cache_mem = request.cache_ids.len() as u32
                * self
                    .registry
                    .model_map
                    .get(req_model)
                    .ok_or(SchedulerError::UnmappedModel)?
                    .cache_block_size;
            self.active_size_map
                .entry(new_data)
                .and_modify(|size| *size += cache_mem as u64)
                .or_insert(cache_mem as u64);
        }

        Ok(())
    }

    fn step_active_requests(&mut self) -> Result<(), SchedulerError> {
        for request in self.active_requests.iter() {}

        Ok(())
    }

    fn handle_unfulfilled(
        &mut self,
        unfulfilled_requests: Vec<RequestMetadata>,
    ) -> Result<(), SchedulerError> {
        // This is quadratic time - can we find a way around this
        for unfulfilled_meta in unfulfilled_requests {
            self.request_priority_queue
                .push(unfulfilled_meta.id, unfulfilled_meta);
        }

        Ok(())
    }
}

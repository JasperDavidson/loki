use std::{collections::HashMap, fs::metadata};
use thiserror::Error;

use crate::scheduler::{ModelID, ModelMetadata};

#[derive(Default, Debug, Clone, Copy, Eq, Hash, PartialEq)]
pub struct PhysicalId(pub u64);

// Make it such that last used is updated to the newest value when accessed for the first time
// (last_used = 0) via the memory translator
// Note the memory translator will store cache bock size to avoid repeating the value across many
// instances of CacheBlock
#[derive(Default, PartialEq, Eq, Hash, Clone, Copy)]
pub struct CacheBlockHandle(pub u64);

// A model only needs to the virtual ID of the layer block to which to map its layers
// The translator will then intercept these virtual IDs and map them to the correct layer slots
// physically
#[derive(Default, PartialEq, Eq, Hash, Clone, Copy)]
pub struct LayerBlockHandle(pub u64);

struct LayerStreamer {
    cur_id: u8,
    layers: Vec<PhysicalId>,
}

impl LayerStreamer {
    fn fetch_next(&mut self) -> PhysicalId {
        let next_layer = self.layers[self.cur_id as usize];

        self.cur_id += 1;
        if self.cur_id == self.layers.len() as u8 {
            self.cur_id = 0;
        }

        next_layer
    }
}

#[derive(Default)]
pub struct ModelMemory {
    pub streaming_ids: (PhysicalId, PhysicalId),
    pub scratchpad_id: PhysicalId,
    pub output_id: PhysicalId,
}

#[derive(Debug, Error)]
pub enum AllocatorError {
    #[error("Not enough memory to accomodate streaming requested models")]
    OutOfMemory,
    #[error("Attempted to fetch layer for invalid model id")]
    InvalidModel,
}

pub struct Allocator {
    total_mem: u64,
    layer_free: HashMap<ModelID, LayerStreamer>,
    model_to_phys: HashMap<ModelID, ModelMemory>,
    cache_free: Vec<PhysicalId>,
    cache_block_size: u32,
    layer_mem_total: u64,
    virtual_counter: u64,
}

impl Allocator {
    // The size of each cache block should ideally divide the physical memory amount given to cache
    // Memory layout:
    // - Beginning of memory is where the page table mapping physical ids -> memory addresses is
    // plaeced
    //      - The page table is implemented as a linear array such that table[phys_id] = addr
    // - Next, each model acquires enough space for two layers worth of parameters + scratchpad
    // memory (between layers) and output space
    // - Finally, KV Cache blocks take up the remaining memory - as many of the specified size
    // that can be allocated
    pub fn new(
        total_mem: u64,
        cache_block_size: u32,
        model_metadata: &HashMap<ModelID, ModelMetadata>,
    ) -> Result<Self, AllocatorError> {
        let total_streaming_size: u64 = model_metadata
            .iter()
            .map(|(_, metadata)| {
                metadata.info_size as u64 * metadata.vocab_size // output size
                    + metadata.max_sequence_len // scratchpad size
                        * metadata.hidden_size
                        * metadata.mlp_expansion_factor as u64
                        * metadata.info_size as u64
                    + metadata.layer_size * 2 // streaming layer size
            })
            .sum::<u64>();
        if total_streaming_size > total_mem {
            return Err(AllocatorError::OutOfMemory);
        }

        // allocate all the layers to physical ids
        let mut phys_id = 0;
        let mut layer_free = HashMap::with_capacity(model_metadata.len());
        let mut model_to_phys = HashMap::with_capacity(model_metadata.len());
        for (model_id, _) in model_metadata {
            let stream_left_id = PhysicalId(phys_id);
            phys_id += 1;
            let stream_right_id = PhysicalId(phys_id);
            phys_id += 1;
            let scratchpad_id = PhysicalId(phys_id);
            phys_id += 1;
            let output_id = PhysicalId(phys_id);
            phys_id += 1;

            // creating the streamer for double buffering
            let streamer = LayerStreamer {
                cur_id: 0,
                layers: vec![stream_left_id, stream_right_id],
            };
            layer_free.insert(*model_id, streamer);

            // mapping from model to layer physical ids for translator table use
            let model_entry = ModelMemory {
                streaming_ids: (stream_left_id, stream_right_id),
                scratchpad_id: scratchpad_id,
                output_id: output_id,
            };
            model_to_phys.insert(*model_id, model_entry);
        }

        // construct the cache free list from the remaining memory
        let cache_mem_size = total_mem - total_streaming_size;
        let num_cache_blocks = cache_mem_size / (cache_block_size as u64);
        let mut cache_free = Vec::with_capacity(num_cache_blocks as usize);

        for _ in 0..num_cache_blocks {
            cache_free.push(PhysicalId(phys_id));
            phys_id += 1;
        }

        Ok(Self {
            total_mem,
            layer_free,
            model_to_phys,
            cache_free,
            cache_block_size,
            layer_mem_total: total_streaming_size,
            virtual_counter: 0,
        })
    }

    // hands out virtual IDs to models that request it
    // note that virtual IDs do not gain associated physical ids until later requested
    // this prevents wastefully handing out physical ids
    pub fn alloc_cache(&mut self) -> CacheBlockHandle {
        self.virtual_counter += 1;

        CacheBlockHandle(self.virtual_counter)
    }

    // Need to handle when there aren't physical blocks available
    // Scheduler can call this when running
    pub fn try_cache_alloc_phys(&mut self) -> Option<PhysicalId> {
        self.cache_free.pop()
    }

    pub fn fetch_layer_phys(&mut self, model_id: &ModelID) -> Result<PhysicalId, AllocatorError> {
        let streamer = self
            .layer_free
            .get_mut(model_id)
            .ok_or(AllocatorError::InvalidModel)?;
        Ok(streamer.fetch_next())
    }

    pub fn free_cache(&mut self, phys_id: PhysicalId) {
        self.cache_free.push(phys_id);
    }

    pub fn cache_mem_remaining(&self) -> u64 {
        self.cache_free.len() as u64 * self.cache_block_size as u64
    }
}

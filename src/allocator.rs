use std::collections::HashMap;
use thiserror::Error;

use crate::scheduler::ModelID;

#[derive(Debug, Clone, Copy)]
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

#[derive(Debug, Error)]
pub enum AllocatorError {
    #[error("Not enough memory to accomodate streaming requested models")]
    OutOfMemory,
    #[error("Attempted to fetch layer for invalid model id")]
    InvalidModel,
}

pub struct Allocator {
    total_mem: u64,
    layer_free_mem: HashMap<ModelID, LayerStreamer>,
    cache_free_mem: Vec<PhysicalId>,
    layer_mem_total: u64,
    virtual_counter: u64,
}

impl Allocator {
    // the size of each cache block should ideally divide the physical memory amount given to cache
    pub fn new(
        total_mem: u64,
        cache_block_size: u32,
        model_layer_sizes: &[(ModelID, u64)],
    ) -> Result<Self, AllocatorError> {
        let total_streaming_size: u64 = model_layer_sizes.iter().map(|&(_, x)| x).sum::<u64>() * 2;
        if total_streaming_size > total_mem {
            return Err(AllocatorError::OutOfMemory);
        }
        let cache_mem_size = total_mem - total_streaming_size;

        // allocate all the layers to physical ids
        let mut phys_id = 0;
        let mut layer_free = HashMap::with_capacity(model_layer_sizes.len() * 2);
        for (model_id, _) in model_layer_sizes {
            let streamer = layer_free.entry(*model_id).or_insert(LayerStreamer {
                cur_id: 0,
                layers: Vec::with_capacity(2),
            });
            streamer.layers.push(PhysicalId(phys_id));
            phys_id += 1;
        }

        // construct the cache free list from the remaining memory
        let num_cache_blocks = cache_mem_size / (cache_block_size as u64);
        let mut cache_free = Vec::with_capacity(num_cache_blocks as usize);

        let mut phys_id = 0;
        for _ in (0..(total_mem - cache_mem_size)).step_by(cache_block_size as usize) {
            cache_free.push(PhysicalId(phys_id));
            phys_id += 1;
        }

        Ok(Self {
            total_mem,
            layer_free_mem: layer_free,
            cache_free_mem: cache_free,
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

    pub fn alloc_layer_block(&mut self) -> LayerBlockHandle {
        self.virtual_counter += 1;

        LayerBlockHandle(self.virtual_counter)
    }

    // Need to handle when there aren't physical blocks available
    // Scheduler can call this when running
    pub fn try_cache_alloc_phys(&mut self) -> Option<PhysicalId> {
        self.cache_free_mem.pop()
    }

    pub fn fetch_layer_phys(&mut self, model_id: &ModelID) -> Result<PhysicalId, AllocatorError> {
        let streamer = self
            .layer_free_mem
            .get_mut(model_id)
            .ok_or(AllocatorError::InvalidModel)?;
        Ok(streamer.fetch_next())
    }

    pub fn free_cache(&mut self, phys_id: PhysicalId) {
        self.cache_free_mem.push(phys_id);
    }
}

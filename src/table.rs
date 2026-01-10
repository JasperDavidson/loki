use std::{
    collections::{HashMap, VecDeque},
    hash::Hash,
};
use thiserror::Error;

use crate::{
    allocator::{
        Allocator, AllocatorError, CacheBlockHandle, LayerBlockHandle, ModelMemory, PhysicalId,
    },
    scheduler::{ModelID, ModelMetadata},
};

trait Evictable {
    fn is_pinned(&self) -> bool;
    fn is_used(&self) -> bool;
    fn set_used(&mut self, used: bool);
    fn set_in_clock(&mut self, clock: bool);
}

trait Swappable {
    fn get_phys_id(&self) -> Option<PhysicalId>;
}

#[derive(Default)]
pub struct CacheBlock {
    pub phys_id: Option<PhysicalId>, // none if swapped to disk
    pinned: bool,                    // true if a model is actively using this block for calculation
    used: bool,                      // true if active in LRU
    in_clock: bool,
}

impl Evictable for CacheBlock {
    fn is_pinned(&self) -> bool {
        self.pinned
    }
    fn is_used(&self) -> bool {
        self.used
    }
    fn set_used(&mut self, used: bool) {
        self.used = used;
    }
    fn set_in_clock(&mut self, clock: bool) {
        self.in_clock = clock;
    }
}

impl Swappable for CacheBlock {
    fn get_phys_id(&self) -> Option<PhysicalId> {
        self.phys_id
    }
}

// the layer size combined will allow the virtual ID mapping to offset to the correct slot
#[derive(Default)]
pub struct LayerBlock {
    pub phys_id: Option<PhysicalId>,
    num_layers: u16,
    cur_layer: u16,
    pinned: bool,
}

fn swap_block<K, V>(
    block_table: &mut HashMap<K, V>,
    allocator: &mut Allocator,
    swap_in: &K,
    swap_out: &K,
) -> Result<PhysicalId, TableError>
where
    K: Eq + std::hash::Hash + Clone,
    V: Swappable,
{
    let swap_out_block = block_table
        .get(swap_out)
        .ok_or(TableError::InvalidLayerHandle)?;
    let swap_out_id = swap_out_block
        .get_phys_id()
        .ok_or(TableError::InvalidSwap)?;

    // TODO: Copy swap out memory from the GPU to the CPU
    // And copy swap in memory from CPU to the GPU

    // Free the memory swapped to disk
    allocator.free_cache(swap_out_id);

    // Allocate new block in place?

    Ok(swap_out_id)
}

// With clock algorithm -> lru cache check for memory that has not been used
// since last clock sweep + is not pinned
// Then swap the block to disk -> TODO: Implement swapping to disk
// A model maintains control over all of virtual cache ids, so access isn't
// invalidated
fn clock_sweep<K, V>(
    block_table: &mut HashMap<K, V>,
    clock_lru: &mut VecDeque<K>,
) -> Result<K, TableError>
where
    K: Eq + std::hash::Hash + Copy + Clone,
    V: Evictable,
{
    // max checks is two times the length since it will have cycled through all possibilities
    let max_checks = clock_lru.len() * 2;
    let mut checks = 0;

    let mut evict_block_handle = None;
    while checks < max_checks {
        // match on pinned/adjust usage of block fetched from lru queue
        let lru_block_handle = clock_lru.pop_back().ok_or(TableError::InvalidSwap)?;
        let lru_block = block_table
            .get_mut(&lru_block_handle)
            .ok_or(TableError::InvalidCacheHandle)?;

        if lru_block.is_pinned() {
            clock_lru.push_front(lru_block_handle);
        } else if lru_block.is_used() {
            lru_block.set_used(false);
            clock_lru.push_front(lru_block_handle);
        } else {
            evict_block_handle = Some(lru_block_handle);
            lru_block.set_in_clock(false);
            clock_lru.push_front(lru_block_handle);
        }

        checks += 1;
    }

    if let Some(handle) = evict_block_handle {
        Ok(handle)
    } else {
        // Cache memory should never be completely utilized since that would mean the Scheduler
        // attempted to request a block even though all were currently pinned - logic error
        Err(TableError::InvalidSwap)
    }
}

#[derive(Clone, Copy)]
pub struct MemoryAddr(pub u64);

impl MemoryAddr {
    pub fn to_gpu_index(&self) -> u32 {
        assert!(
            self.0 % 4 == 0,
            "Address {} is not aligned to f32 stride!",
            self.0
        );
        (self.0 / 4) as u32
    }
}

// enum that represents how a layer activation should continue
// NeedStream -> scheduler must initiate a stream into the physical id, kernel has map from id to
// address in uniform buffer
// Present -> the layer is already present and no i/o is needed
pub enum LayerActivationStatus {
    NeedStream { target_phys: PhysicalId },
    Present,
}

#[derive(Debug, Error)]
pub enum TableError {
    #[error("Ran out of layer memory to allocate pages to")]
    OutOfLayerMemory,
    #[error("Ran out of cache memory to allocate pages to")]
    OutOfCacheMemory,
    #[error("Layer Handle not mappted to a valid Layer Block")]
    InvalidLayerHandle,
    #[error("Cache Handle not mapped to a valid Cache Block")]
    InvalidCacheHandle,
    #[error("Invalid swap: page not in memory or memory is full")]
    InvalidSwap,
    #[error("Invalid model id during creation")]
    InvalidModelID,
    #[error("Allocator failure: {0}")]
    TableError(#[from] AllocatorError),
}

#[derive(Default)]
pub struct Table {
    // Represented as a contiguous layer of memory to allow for efficient access
    // - Note that upon physical ID assignment by the Allocator, they are assigned sequentially,
    // so this alternative mapping will preserve the invariant of phys id -> memory
    // - Further it contains Options so that KV cache blocks can be invalidated
    phys_to_mem: Vec<Option<MemoryAddr>>,

    // pub layer_table: HashMap<LayerBlockHandle, LayerBlock>,
    pub cache_table: HashMap<CacheBlockHandle, CacheBlock>,

    // Note that the LRU queue being empty is a logic error (if asked to swap by the scheduler)
    cache_clock_lru: VecDeque<CacheBlockHandle>, // true if used since last swept by clock
                                                 // layer_clock_lru: VecDeque<LayerBlockHandle>,
}

impl Table {
    pub fn new(
        model_metadata: &HashMap<ModelID, ModelMetadata>,
        model_phys_id: &HashMap<ModelID, ModelMemory>,
        cache_phys_id: &[PhysicalId],
        cache_block_size: u32,
    ) -> Result<Self, TableError> {
        // Pre-initialize to allow for indexing below since HashMap iterator access is
        // non-deterministic
        let mut phys_to_mem: Vec<Option<MemoryAddr>> =
            vec![None; model_phys_id.len() * 2 + cache_phys_id.len()];

        // Set up mappings from physical ids to gpu memory addresses (buffer offsets)
        // When kernels require certain physical ids their addresses will be fetched from here
        // before being sent off
        let mut cur_mem_addr = 0;
        for (model_id, layer_ids) in model_phys_id {
            let metadata = model_metadata
                .get(model_id)
                .ok_or(TableError::InvalidModelID)?;
            let scratchpad_size = metadata.vocab_size * metadata.info_size as u64;
            let output_size = metadata.hidden_size
                * metadata.mlp_expansion_factor as u64
                * metadata.info_size as u64
                * metadata.max_sequence_len;

            // Assign the streaming layer addresses
            phys_to_mem[layer_ids.streaming_ids.0.0 as usize] = Some(MemoryAddr(cur_mem_addr));
            cur_mem_addr += metadata.layer_size;
            phys_to_mem[layer_ids.streaming_ids.1.0 as usize] = Some(MemoryAddr(cur_mem_addr));
            cur_mem_addr += metadata.layer_size;

            // Assign the scratchpad and output addresses
            phys_to_mem[layer_ids.scratchpad_id.0 as usize] = Some(MemoryAddr(cur_mem_addr));
            cur_mem_addr += scratchpad_size;
            phys_to_mem[layer_ids.output_id.0 as usize] = Some(MemoryAddr(cur_mem_addr));
            cur_mem_addr += output_size;
        }

        for cache_id in cache_phys_id {
            phys_to_mem[cache_id.0 as usize] = Some(MemoryAddr(cur_mem_addr));
            cur_mem_addr += cache_block_size as u64;
        }

        Ok(Table {
            phys_to_mem,
            ..Default::default()
        })
    }

    pub fn register_cache_block(&mut self, cache_handle: CacheBlockHandle) {
        let cache_block = CacheBlock::default();
        self.cache_table.insert(cache_handle, cache_block);
    }

    // Activates an existing layer block for a model
    // Takes in the model id from which to fetch the next physical id for
    // LOGIC: This function should only be called when the next layer needs to be streamed in
    // Layer handles are not needed since streaming takes a double buffering approach and only
    // loads one layer at a time
    pub fn activate_next_layer(
        &mut self,
        allocator: &mut Allocator,
        model_id: &ModelID,
    ) -> Result<LayerActivationStatus, TableError> {
        let target_phys = allocator.fetch_layer_phys(model_id)?;

        // Look into if there's any times we would *not* need to stream
        Ok(LayerActivationStatus::NeedStream { target_phys })
    }

    // activates the cache block
    // since no data needs to be transferred, this just updates the data sent over uniform
    // buffers to kernels
    pub fn activate_cache_block(
        &mut self,
        allocator: &mut Allocator,
        cache_handle: CacheBlockHandle,
    ) -> Result<(), TableError> {
        let needs_alloc = self
            .cache_table
            .get(&cache_handle)
            .map(|b| b.phys_id.is_none())
            .ok_or(TableError::InvalidCacheHandle)?;

        let mut new_phys_id = None;
        if needs_alloc {
            new_phys_id = Some(match allocator.try_cache_alloc_phys() {
                Some(id) => id,
                None => {
                    // Find a handle to swap with and execute
                    let swap_handle =
                        clock_sweep(&mut self.cache_table, &mut self.cache_clock_lru)?;
                    swap_block(
                        &mut self.cache_table,
                        allocator,
                        &cache_handle,
                        &swap_handle,
                    )?
                }
            })
        }

        // Acquire the block
        let cache_block = self
            .cache_table
            .get_mut(&cache_handle)
            .ok_or(TableError::InvalidCacheHandle)?;

        // update block and mark it as activated
        if let Some(new_phys_id) = new_phys_id {
            cache_block.phys_id = Some(new_phys_id);
        }

        cache_block.pinned = true;
        cache_block.used = true;

        // Add to the clock queue if not already there
        if !cache_block.in_clock {
            self.cache_clock_lru.push_front(cache_handle);
            cache_block.in_clock = true;
        }

        Ok(())
    }

    pub fn activate_cache_blocks<I>(
        &mut self,
        allocator: &mut Allocator,
        cache_handles: I,
    ) -> Result<(), TableError>
    where
        I: IntoIterator<Item = CacheBlockHandle>,
    {
        for cache_handle in cache_handles {
            self.activate_cache_block(allocator, cache_handle)?;
        }

        Ok(())
    }
}

use std::collections::{HashMap, VecDeque};
use thiserror::Error;

use crate::allocator::{Allocator, CacheBlockHandle, LayerBlockHandle, PhysicalId};

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

// The layer size combined will allow the virtual ID mapping to offset to the correct slot
#[derive(Default)]
pub struct LayerBlock {
    pub phys_id: Option<PhysicalId>,
    // layer_size: u32,
    num_layers: u16,
    cur_layer: u16,
    pinned: bool,
    used: bool,
    in_clock: bool,
}

impl Evictable for LayerBlock {
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

impl Swappable for LayerBlock {
    fn get_phys_id(&self) -> Option<PhysicalId> {
        self.phys_id
    }
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

#[derive(Default)]
pub struct Table {
    pub cache_table: HashMap<CacheBlockHandle, CacheBlock>,
    pub layer_table: HashMap<LayerBlockHandle, LayerBlock>,
    cache_clock_lru: VecDeque<CacheBlockHandle>, // true if used since last swept by clock
    layer_clock_lru: VecDeque<LayerBlockHandle>,
    // Note that either LRU being empty is a logic error (if asked to swap by the scheduler)
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
}

impl Table {
    // do we need a constructor?
    // most of the fields should be defaultable
    //    pub fn new() -> Self {
    //        Table {}
    //    }

    // Registers a models layer block in the map
    pub fn register_layer_block(&mut self, layer_handle: LayerBlockHandle) {
        let layer_block = LayerBlock {
            phys_id: None,
            // layer_size,
            num_layers: 0,
            cur_layer: 0,
            ..Default::default()
        };
        self.layer_table.insert(layer_handle, layer_block);
    }

    pub fn register_cache_block(&mut self, cache_handle: CacheBlockHandle) {
        let cache_block = CacheBlock {
            phys_id: None,
            pinned: false,
            ..Default::default()
        };
        self.cache_table.insert(cache_handle, cache_block);
    }

    // Activates an existing layer block for a model
    pub fn activate_layer_block(
        &mut self,
        allocator: &mut Allocator,
        layer_handle: LayerBlockHandle,
    ) -> Result<(), TableError> {
        // Check if the layer needs to be swapped to VRAM
        let needs_alloc = self
            .layer_table
            .get(&layer_handle)
            .map(|l| l.phys_id.is_some())
            .ok_or(TableError::InvalidLayerHandle)?;

        // Take memory/swap for memory in VRAM if needed
        let mut new_phys_id = None;
        if needs_alloc {
            new_phys_id = Some(match allocator.try_layer_alloc_phys() {
                Some(id) => id,
                None => {
                    // TODO: Think about if we should use LRU cache for weights as well
                    let swap_handle =
                        clock_sweep(&mut self.layer_table, &mut self.layer_clock_lru)?;
                    swap_block(
                        &mut self.layer_table,
                        allocator,
                        &layer_handle,
                        &swap_handle,
                    )?
                }
            })
        }

        // Grab and activate layer block
        let layer_block = self
            .layer_table
            .get_mut(&layer_handle)
            .ok_or(TableError::InvalidLayerHandle)?;

        if let Some(phys_id) = new_phys_id {
            layer_block.phys_id = Some(phys_id);
        }
        layer_block.pinned = true;
        layer_block.used = true;

        if !layer_block.in_clock {
            self.layer_clock_lru.push_front(layer_handle);
            layer_block.in_clock = true;
        }

        Ok(())
    }

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

use std::collections::{HashMap, VecDeque};

use crate::allocator::{Allocator, CacheBlockHandle, LayerBlockHandle};

struct CacheBlock {
    phys_id: Option<u64>, // none if swapped to disk
    used: bool,           // true if used since last swept by clock
    pinned: bool,         // true if a model is actively using this block for calculation
}

// The layer size combined will allow the virtual ID mapping to offset to the correct slot
pub struct LayerBlock {
    phys_id: Option<u64>,
    layer_size: u32,
    num_layers: u16,
    cur_layer: u16,
    pinned: bool,
}

#[derive(Default)]
struct Table {
    cache_table: HashMap<CacheBlockHandle, CacheBlock>,
    layer_table: HashMap<LayerBlockHandle, LayerBlock>,
    clock_lru: VecDeque<CacheBlockHandle>,
}

#[derive(Debug)]
enum TranslatorError {
    OutOfMemory,
    InvalidLayerHandle,
}

impl Table {
    // Registers a models layer block in the map
    pub fn register_layer_block(&mut self, layer_handle: LayerBlockHandle, layer_size: u32) {
        let layer_block = LayerBlock {
            phys_id: None,
            layer_size,
            num_layers: 0,
            cur_layer: 0,
            pinned: false,
        };
        self.layer_table.insert(layer_handle, layer_block);
    }

    pub fn register_cache_block(&mut self, cache_handle: CacheBlockHandle) {
        let cache_block = CacheBlock {
            phys_id: None,
            used: false,
            pinned: false,
        };
        self.cache_table.insert(cache_handle, cache_block);
    }

    // Activates an existing layer block for a model with a new number of layers based on shifting
    // model count
    pub fn activate_layer_block(
        &mut self,
        allocator: &mut Allocator,
        layer_handle: LayerBlockHandle,
        num_layers: u16,
    ) -> Result<(), TranslatorError> {
        if let Some(layer_block) = self.layer_table.get_mut(&layer_handle) {
            if layer_block.phys_id.is_none() {
                layer_block.phys_id = allocator.try_layer_alloc_phys();
            }
            if layer_block.phys_id.is_none() {
                match allocator.try_layer_alloc_phys() {
                    Some(id) => layer_block.phys_id = Some(id),

                    // Return err result, scheduler must perform a free on some layer block (i.e. free
                    // a model) and try again
                    None => return Err(TranslatorError::OutOfMemory),
                }
            }
            layer_block.num_layers = num_layers;
            layer_block.pinned = true;

            Ok(())
        } else {
            Err(TranslatorError::InvalidLayerHandle)
        }
    }

    pub fn activate_cache_block(
        &mut self,
        allocator: &mut Allocator,
        cache_handle: CacheBlockHandle,
    ) -> Result<(), TranslatorError> {
        if let Some(cache_block) = self.cache_table.get_mut(&cache_handle) {
            if cache_block.phys_id.is_none() {
                match allocator.try_cache_alloc_phys() {
                    Some(id) => cache_block.phys_id = Some(id),
                    // With clock algorithm -> lru cache check for memory that has not been used
                    // since last clock sweep + is not pinned
                    // Then swap the block to disk -> TODO: Implement swapping to disk
                    // A model maintains control over all of virtual cache ids, so access isn't
                    // invalidated
                    None => return Err(TranslatorError::OutOfMemory),
                }
            }

            cache_block.pinned = true;
            cache_block.used = true;

            Ok(())
        } else {
            Err(TranslatorError::InvalidLayerHandle)
        }
    }

    pub fn activate_cache_blocks(
        &mut self,
        allocator: &mut Allocator,
        cache_handles: Vec<CacheBlockHandle>,
    ) -> Result<(), TranslatorError> {
        for cache_handle in cache_handles {
            self.activate_cache_block(allocator, cache_handle)?;
        }

        Ok(())
    }
}

use std::collections::HashMap;

use crate::allocator::{Allocator, CacheBlockHandle, LayerBlockHandle};

struct Model {
    layer_block: LayerBlockHandle,
    cache_ids: Vec<CacheBlockHandle>,
}

struct CacheBlock {
    phys_id: Option<u32>, // none if swapped to disk
    last_used: u64,
    modified: bool,
    pinned: bool,
}

// The layer size combined will allow the virtual ID mapping to offset to the correct slot
pub struct LayerBlock {
    phys_id: Option<u32>,
    layer_size: u32,
    num_layers: u16,
    cur_layer: u16,
    last_used: u64,
    pinned: bool,
}

#[derive(Default)]
struct Translator {
    cache_table: HashMap<CacheBlockHandle, CacheBlock>,
    layer_table: HashMap<LayerBlockHandle, LayerBlock>,
    cache_lru_id: u64,
    layer_lru_id: u64,
}

#[derive(Debug)]
enum TranslatorError {
    OutOfMemory,
    InvalidLayerHandle,
}

impl Translator {
    // Registers a models layer block in the map
    pub fn register_layer_block(&mut self, layer_handle: LayerBlockHandle, layer_size: u32) {
        let layer_block = LayerBlock {
            phys_id: None,
            layer_size,
            num_layers: 0,
            cur_layer: 0,
            last_used: 0,
            pinned: false,
        };
        self.layer_table.insert(layer_handle, layer_block);
    }

    pub fn register_cache_block(&mut self, cache_handle: CacheBlockHandle) {
        let cache_block = CacheBlock {
            phys_id: None,
            last_used: self.cache_lru_id,
            modified: false,
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

            self.layer_lru_id += 1;
            layer_block.last_used = self.layer_lru_id;

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
                    None => return Err(TranslatorError::OutOfMemory),
                }
            }

            cache_block.pinned = true;
            cache_block.modified = true;
            cache_block.last_used = self.cache_lru_id;

            Ok(())
        } else {
            Err(TranslatorError::InvalidLayerHandle)
        }
    }
}

// Make it such that last used is updated to the newest value when accessed for the first time
// (last_used = 0) via the memory translator
// Note the memory translator will store cache bock size to avoid repeating the value across many
// instances of CacheBlock
pub struct CacheBlockHandle(pub u64);

struct CacheBlock {
    phys_id: Option<u32>, // none if swapped to disk
    last_used: u32,
    modified: bool,
    pinned: bool,
}

// A model only needs to the virtual ID of the layer block to which to map its layers
// The translator will then intercept these virtual IDs and map them to the correct layer slots
// physically
pub struct LayerBlockHandle(pub u64);

// The layer size combined will allow the virtual ID mapping to offset to the correct slot
pub struct LayerBlock {
    phys_id: Option<u32>,
    layer_size: u32,
    last_used: u32,
    pinned: bool,
}

pub struct Allocator {
    total_mem: u64,
    cache_free: Vec<u32>,
    layer_free: Vec<u32>,
    layer_mem_total: u64,
    virtual_counter: u64,
}

impl Allocator {
    // the size of each cache block should ideally divide the physical memory amount given to cache
    pub fn new(total_mem: u64, cache_weight_ratio: f32, cache_block_size: u32) -> Self {
        let cache_mem_size = (total_mem as f32 * cache_weight_ratio).floor() as u64;

        // Construct the cache free list
        let num_cache_blocks = cache_mem_size / (cache_block_size as u64);
        let mut cache_free = Vec::with_capacity(num_cache_blocks as usize);
        for id in 0..num_cache_blocks {
            cache_free.push(id as u32);
        }

        Self {
            total_mem,
            cache_free,
            layer_free: vec![0], // Assume 1 active model initially
            layer_mem_total: total_mem - cache_mem_size,
            virtual_counter: 0,
        }
    }

    // Hands out virtual IDs to models that request it
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
    fn try_cache_alloc_phys(&mut self) -> Option<u32> {
        self.cache_free.pop()
    }

    fn try_layer_alloc_phys(&mut self) -> Option<u32> {
        self.layer_free.pop()
    }

    pub fn free_cache(&mut self, phys_id: u32) {
        self.cache_free.push(phys_id);
    }

    // Scheduler and translator will handle the other active models have increased layer streaming
    // capacity
    pub fn free_layer(&mut self, phys_id: u32) {
        self.layer_free.push(phys_id);
    }

    // The goal here is to re-partition layer memory such that each model has fair access to layer
    // to an equal amount of layer memory
    // TOOO: Account for pinned models as well, e.g. models that need all weights loaded at once
    // This is effectively a flush, physical ids need to be re-assigned
    // If we didn't flush, we would get fragmentation when shrinking block size and a new model may
    // not be able to fit as many layers in
    //      - Note that technically a model could load in layers in this fragmented space, but
    //      since layer size is variable it's not guaranteed it could fit and fragmentation issues
    //      still occur
    pub fn increase_active_models(&mut self, num_active: u8) {
        let new_slot_size = self.layer_mem_total / num_active as u64;
        self.layer_free.clear();

        for id in 0..num_active {
            self.layer_free.push(id as u32);
        }

        // NOTE: Scheduler is responsible for ensuring IDs get reassigned
    }
}

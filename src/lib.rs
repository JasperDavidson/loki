mod allocator;
mod backend;
mod scheduler;
mod table;

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::allocator::*;
    use crate::scheduler::*;
    use crate::table::*;

    use super::*;

    fn setup_basic_metadata() -> ModelMetadata {
        ModelMetadata {
            layer_size: 1,
            hidden_size: 2,
            vocab_size: 2,
            max_sequence_len: 4,
            info_size: 4, // f32 is 4 bytes
            mlp_expansion_factor: 2,
        }
    }

    #[test]
    fn test_alloc_no_phys() -> anyhow::Result<()> {
        let model_metadata = setup_basic_metadata();
        let mut allocator = Allocator::new(100, 1, &HashMap::from([(ModelID(0), model_metadata)]))?;
        let mut table = Table::default();

        let cache_id_1 = allocator.alloc_cache();
        let cache_id_2 = allocator.alloc_cache();
        let cache_id_3 = allocator.alloc_cache();

        table.register_cache_block(cache_id_1);
        table.register_cache_block(cache_id_2);
        table.activate_cache_block(&mut allocator, cache_id_1)?;
        table.activate_cache_block(&mut allocator, cache_id_2)?;
        assert!(
            table
                .activate_cache_block(&mut allocator, cache_id_3)
                .is_err()
        );

        Ok(())
    }

    #[test]
    fn test_alloc_not_registered() -> anyhow::Result<()> {
        let model_metadata = setup_basic_metadata();
        let mut allocator = Allocator::new(100, 1, &HashMap::from([(ModelID(0), model_metadata)]))?;
        let mut table = Table::default();

        let cache_id = allocator.alloc_cache();
        assert!(
            table
                .activate_cache_block(&mut allocator, cache_id)
                .is_err()
        );

        Ok(())
    }
}

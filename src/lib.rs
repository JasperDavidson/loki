mod allocator;
mod scheduler;
mod table;

#[cfg(test)]
mod tests {
    use crate::allocator::*;
    use crate::scheduler::*;
    use crate::table::*;

    use super::*;

    #[test]
    fn test_alloc_no_phys() -> anyhow::Result<()> {
        let mut allocator = Allocator::new(4, 1, &[(ModelID(1), 1)])?;
        let mut table = Table::default();

        let cache_id_1 = allocator.alloc_cache();
        let cache_id_2 = allocator.alloc_cache();
        let cache_id_3 = allocator.alloc_cache();

        let layer_id_1 = allocator.alloc_layer_block();
        let layer_id_2 = allocator.alloc_layer_block();
        let layer_id_3 = allocator.alloc_layer_block();

        table.register_cache_block(cache_id_1);
        table.register_cache_block(cache_id_2);
        table.register_cache_block(cache_id_3);
        table.activate_cache_block(&mut allocator, cache_id_1)?;
        table.activate_cache_block(&mut allocator, cache_id_2)?;
        assert!(
            table
                .activate_cache_block(&mut allocator, cache_id_3)
                .is_err()
        );

        table.register_layer_block(layer_id_1);
        table.register_layer_block(layer_id_2);
        table.register_layer_block(layer_id_3);
        table.activate_layer_block(&mut allocator, layer_id_1)?;
        table.activate_layer_block(&mut allocator, layer_id_2)?;
        assert!(
            table
                .activate_layer_block(&mut allocator, layer_id_3)
                .is_err()
        );

        Ok(())
    }

    #[test]
    fn test_alloc_not_registered() -> anyhow::Result<()> {
        let mut allocator = Allocator::new(4, 1, &[(ModelID(1), 1)])?;
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

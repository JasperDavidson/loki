mod allocator;
mod backend;
mod scheduler;
mod table;

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use pollster::FutureExt;
    use wgpu::naga::back;

    use crate::allocator::*;
    use crate::backend::*;
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

    #[test]
    fn test_matadd() -> anyhow::Result<()> {
        let backend = BackendHandler::new(80).block_on();
        let data_one: Vec<f32> = vec![1.0, 2.0, 3.0];
        let data_one_id = PhysicalId(0);
        let data_two: Vec<f32> = vec![1.0, 2.0, 3.0];
        let data_two_id = PhysicalId(1);
        let output_id = PhysicalId(2);
        let data_size = std::mem::size_of::<f32>();
        let data_len = data_one.len();
        let data_byte_size = (data_size * data_len) as u64;

        let mut page_table: Vec<Option<MemoryAddr>> = vec![None; 10];
        let base_addr = 0;
        page_table[data_one_id.0 as usize] = Some(MemoryAddr(base_addr));
        page_table[data_two_id.0 as usize] = Some(MemoryAddr(base_addr + data_byte_size));
        page_table[output_id.0 as usize] = Some(MemoryAddr(base_addr + data_byte_size * 2));

        backend.write_slab_mem(bytemuck::cast_slice(&data_one), &data_one_id, &page_table)?;
        backend.write_slab_mem(bytemuck::cast_slice(&data_two), &data_two_id, &page_table)?;

        let matadd_uniform = BinaryOpUniformBuffer::new(
            page_table[data_one_id.0 as usize].unwrap().to_gpu_index(),
            page_table[data_two_id.0 as usize].unwrap().to_gpu_index(),
            page_table[output_id.0 as usize].unwrap().to_gpu_index(),
            [3, 1, 1, 1],
        );
        backend.write_uniform_mem(matadd_uniform, 0);

        let bind_group = backend.create_bind_group(
            "Matadd bind group",
            0,
            std::mem::size_of::<BinaryOpUniformBuffer>(),
            &KernelType::MatrixAddition,
        )?;
        let mut encoder = backend.create_step_encoder();

        backend.commit_actions(
            &mut encoder,
            vec![KernelType::MatrixAddition],
            vec![bind_group],
            vec![(1, 1, 1)],
        )?;
        backend.fetch_result(encoder, &output_id, data_byte_size, &page_table)?;

        Ok(())
    }
}

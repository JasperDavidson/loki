use std::collections::HashMap;
use thiserror::Error;
use wgpu::{BindGroupEntry, BufferUses, PipelineLayoutDescriptor};

use crate::{allocator::PhysicalId, table::MemoryAddr};

struct GpuComputeContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

struct KernelContext {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ModifyBackendUniformBuffer {
    read_id: u32,
    read_offset: u32,
    write_id: u32,
    write_offset: u32,
    dim: [u32; 4],
}

pub enum UniformBuffer {
    ModifyBackend(ModifyBackendUniformBuffer),
}

#[derive(PartialEq, Eq, Hash)]
pub enum KernelType {
    MatrixAddition,
}

#[derive(Debug, Error)]
pub enum BackendError {
    #[error("Kernel type not suppported")]
    KernelDNE,
    #[error("Physical ID not mapped to a memory address")]
    PhysicalDNE,
    #[error("Memory currently not associated with any data")]
    PageDNE,
}

pub struct JobDescription {
    buffer_offsets: Vec<u64>,
}

pub struct BackendHandler {
    uniform_buffers: wgpu::Buffer,
    slab_buffer: wgpu::Buffer,
    compute_context: GpuComputeContext,
    kernel_contexts: HashMap<KernelType, KernelContext>,
}

impl BackendHandler {
    async fn new(mem_size: u64) -> Self {
        // None of the explicit options seem terribly important but check again in the future
        let instance = wgpu::Instance::new(&Default::default());
        let adapter = instance.request_adapter(&Default::default()).await.unwrap();
        let (device, queue) = adapter.request_device(&Default::default()).await.unwrap();

        let matrix_add_kernel =
            device.create_shader_module(wgpu::include_wgsl!("kernels/matrix_accumulate.wgsl"));

        // Create the layout for every kernel
        // - Slot 1 contains the uniform buffer, which contains the page table and other metadata such as how this kernel should access the slab buffer
        // - Slot 2 contains the slab buffer itself
        let kernel_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Generic kernel layout").map(Into::into),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Create the pipelines for each kernel
        let matrix_add_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Matrix addition"),
                layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
                    label: Some("Matrix addition pipeline layout").map(Into::into),
                    bind_group_layouts: &[&kernel_layout],
                    immediate_size: 0,
                })),
                module: &matrix_add_kernel,
                entry_point: None,
                compilation_options: Default::default(),
                cache: Default::default(),
            });

        let matrix_add_compute_context = KernelContext {
            pipeline: matrix_add_pipeline,
            bind_group_layout: kernel_layout,
        };

        let slab_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Slab buffer").map(Into::into),
            size: mem_size,
            usage: wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let uniform_buffers = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Uniform Buffer Container").map(Into::into),
            size: 64000, // TODO: Come up with a good way to determine this number
            usage: wgpu::BufferUsages::COPY_DST // TODO: Check if these are the correct access
            // modifiers we want
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let compute_context = GpuComputeContext { device, queue };
        let kernel_contexts =
            HashMap::from([(KernelType::MatrixAddition, matrix_add_compute_context)]);

        BackendHandler {
            uniform_buffers,
            slab_buffer,
            compute_context,
            kernel_contexts,
        }
    }

    // Writes data into the specified offset in the slab buffer
    fn write_slab_mem(
        &self,
        data: &[u8],
        phys_id: &PhysicalId,
        page_table: &Vec<Option<MemoryAddr>>,
    ) -> Result<(), BackendError> {
        let offset = page_table
            .get(phys_id.0 as usize)
            .ok_or(BackendError::PhysicalDNE)?
            .ok_or(BackendError::PageDNE)?;
        // Consider looking into write_buffer_width which uses a staging buffer... more
        // efficient?
        self.compute_context
            .queue
            .write_buffer(&self.slab_buffer, offset.0, data);
        // See if this can be potentially optimized in the future - e.g. not writing after
        // *every* write but maybe batching with others
        self.compute_context.queue.submit([]);

        Ok(())
    }

    // Writes data into the specified offset in the uniform buffer
    fn write_uniform_mem(&self, uniform_buffer_data: UniformBuffer, offset: u64) {
        match uniform_buffer_data {
            UniformBuffer::ModifyBackend(uniform_data) => {
                // TODO: How will the scheduler know the offset?
                self.compute_context.queue.write_buffer(
                    &self.uniform_buffers,
                    offset,
                    bytemuck::cast_slice(&[uniform_data]),
                );
            }
        }
    }

    fn create_bind_group(
        &self,
        uniform_buffer_data: UniformBuffer,
        kernel_type: &KernelType,
    ) -> Result<wgpu::BindGroup, BackendError> {
        // Fetch this first to exit early if invalid kernel
        let layout = &self
            .kernel_contexts
            .get(kernel_type)
            .ok_or(BackendError::KernelDNE)?
            .bind_group_layout;

        // TOOD: Figure out how to construct the bind group layout based on offsets
        let mut bind_group_entries = Vec::with_capacity(buffers.len() + 1);
        bind_group_entries.push(wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform_buf.as_entire_binding(),
        });

        let bind_group_entries = [
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: self.slab_buffer,
            },
        ];

        Ok(self
            .compute_context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout,
                entries: &bind_group_entries,
            }))
    }

    fn create_step_encoder(&self) -> wgpu::CommandEncoder {
        self.compute_context
            .device
            .create_command_encoder(&Default::default())
    }

    fn commit_actions(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        kernel_contexts: Vec<KernelType>,
        // Sent as separate vectors to indicate they could perhaps be decoupled in the future
        bind_groups: Vec<wgpu::BindGroup>,
        workgroup_dims: Vec<(u32, u32, u32)>,
    ) -> Result<(), BackendError> {
        assert_eq!(kernel_contexts.len(), bind_groups.len());
        assert_eq!(bind_groups.len(), workgroup_dims.len());
        assert_eq!(workgroup_dims.len(), kernel_contexts.len());

        // probably allow for pass names in future
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            ..Default::default()
        });

        for i in 0..kernel_contexts.len() {
            let context = self
                .kernel_contexts
                .get(&kernel_contexts[i])
                .ok_or(BackendError::KernelDNE)?;
            compute_pass.set_pipeline(&context.pipeline);
            // TODO: What should actually happen here is the offset should be specified into one
            // slab buffer
            compute_pass.set_bind_group(0, &bind_groups[i], &[]);
            compute_pass.dispatch_workgroups(
                workgroup_dims[i].0,
                workgroup_dims[i].1,
                workgroup_dims[i].2,
            );
        }

        Ok(())
    }
}

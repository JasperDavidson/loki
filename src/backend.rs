use std::{
    collections::HashMap,
    num::NonZero,
    sync::mpsc::{RecvError, channel},
};
use thiserror::Error;
use wgpu::{BindGroupEntry, BufferBinding, BufferUses, PipelineLayoutDescriptor};

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

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BinaryOpUniformBuffer {
    left_read_id: u32,
    left_read_offset: u32,
    right_read_id: u32,
    right_read_offset: u32,
    write_id: u32,
    write_offset: u32,
    // TOOD: Surely there's some way around this padding?
    _padding: [u32; 2],
    dim: [u32; 4],
}

impl BinaryOpUniformBuffer {
    pub fn new(
        left_read_id: u32,
        left_read_offset: u32,
        right_read_id: u32,
        right_read_offset: u32,
        write_id: u32,
        write_offset: u32,
        dim: [u32; 4],
    ) -> Self {
        Self {
            left_read_id,
            left_read_offset,
            right_read_id,
            right_read_offset,
            write_id,
            write_offset,
            _padding: [0, 0],
            dim,
        }
    }
}

pub enum UniformBuffer {
    ModifyBackend(ModifyBackendUniformBuffer),
    BinaryOp(BinaryOpUniformBuffer),
}

#[derive(PartialEq, Eq, Hash)]
pub enum KernelType {
    MatrixAccumulate,
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
    #[error("Failed to receive mapped data")]
    RecvError(#[from] RecvError),
    #[error("")]
    BufferAsnycError(#[from] wgpu::BufferAsyncError),
    #[error("Failed to poll from ")]
    PollError(#[from] wgpu::PollError),
}

pub struct JobDescription {
    buffer_offsets: Vec<u64>,
}

pub struct BackendHandler {
    uniform_buffers: wgpu::Buffer,
    slab_buffer: wgpu::Buffer,
    staging_buffer: wgpu::Buffer,
    compute_context: GpuComputeContext,
    kernel_contexts: HashMap<KernelType, KernelContext>,
}

impl BackendHandler {
    pub async fn new(mem_size: u64) -> Self {
        // None of the explicit options seem terribly important but check again in the future
        let instance = wgpu::Instance::new(&Default::default());
        let adapter = instance.request_adapter(&Default::default()).await.unwrap();
        let (device, queue) = adapter.request_device(&Default::default()).await.unwrap();

        let matrix_accumulate_kernel =
            device.create_shader_module(wgpu::include_wgsl!("kernels/matrix_accumulate.wgsl"));
        let matrix_add_kernel =
            device.create_shader_module(wgpu::include_wgsl!("kernels/matrix_add.wgsl"));

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
        let matrix_accumulate_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Matrix accumulation"),
                layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
                    label: Some("Matrix accumulation pipeline layout").map(Into::into),
                    bind_group_layouts: &[&kernel_layout],
                    immediate_size: 0,
                })),
                module: &matrix_accumulate_kernel,
                entry_point: None,
                compilation_options: Default::default(),
                cache: Default::default(),
            });

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

        let matrix_accumulate_compute_context = KernelContext {
            pipeline: matrix_accumulate_pipeline,
            bind_group_layout: kernel_layout.clone(), // Check if this clone is ok
        };

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
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        });
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer").map(Into::into),
            size: 500000, // TODO: Come up with a good way to determine this number, should be
            // as large as the output portion of the slab buffer might be
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let compute_context = GpuComputeContext { device, queue };
        let kernel_contexts = HashMap::from([
            (
                KernelType::MatrixAccumulate,
                matrix_accumulate_compute_context,
            ),
            (KernelType::MatrixAddition, matrix_add_compute_context),
        ]);

        BackendHandler {
            uniform_buffers,
            slab_buffer,
            staging_buffer,
            compute_context,
            kernel_contexts,
        }
    }

    // Writes data into the specified offset in the slab buffer
    pub fn write_slab_mem(
        &self,
        data: &[u8],
        phys_id: &PhysicalId,
        page_table: &[Option<MemoryAddr>],
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

    pub fn write_page_table(&self, page_table: &[Option<MemoryAddr>]) {
        let mut gpu_page_table = vec![0.0f32; page_table.len()];
        for (i, addr) in page_table.iter().enumerate() {
            if let Some(page_addr) = addr {
                gpu_page_table[i] = (page_addr.0 as f32) / 4.0;
            }
        }

        self.compute_context.queue.write_buffer(
            &self.slab_buffer,
            0,
            &bytemuck::cast_slice(&gpu_page_table),
        );
    }

    // Writes data into the specified offset in the uniform buffer
    pub fn write_uniform_mem(&self, uniform_buffer_data: UniformBuffer, offset: u64) {
        // DEFINITELY a better way to structure this -> literally maps to the same outcome
        // each time
        match uniform_buffer_data {
            UniformBuffer::ModifyBackend(uniform_data) => {
                // TODO: How will the scheduler know the offset?
                self.compute_context.queue.write_buffer(
                    &self.uniform_buffers,
                    offset,
                    bytemuck::cast_slice(&[uniform_data]),
                );
            }
            UniformBuffer::BinaryOp(uniform_data) => {
                // TODO: How will the scheduler know the offset?
                self.compute_context.queue.write_buffer(
                    &self.uniform_buffers,
                    offset,
                    bytemuck::cast_slice(&[uniform_data]),
                );
            }
        }
    }

    // TODO: Make it so that this wgpu code is isolated
    pub fn create_bind_group(
        &self,
        label: &str,
        uniform_offset: u64,
        uniform_size: usize,
        kernel_type: &KernelType,
    ) -> Result<wgpu::BindGroup, BackendError> {
        // Fetch this first to exit early if invalid kernel
        let layout = &self
            .kernel_contexts
            .get(kernel_type)
            .ok_or(BackendError::KernelDNE)?
            .bind_group_layout;

        let bind_group_entries = [
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(BufferBinding {
                    buffer: &self.uniform_buffers,
                    offset: uniform_offset,
                    size: NonZero::new(uniform_size as u64),
                }),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: self.slab_buffer.as_entire_binding(),
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

    // TODO: Abstract away inner wgpu workings
    pub fn create_step_encoder(&self) -> wgpu::CommandEncoder {
        self.compute_context
            .device
            .create_command_encoder(&Default::default())
    }

    pub fn commit_actions(
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
            // slab buffer - or already accounted for by the kernels?
            compute_pass.set_bind_group(0, &bind_groups[i], &[]);
            compute_pass.dispatch_workgroups(
                workgroup_dims[i].0,
                workgroup_dims[i].1,
                workgroup_dims[i].2,
            );
        }

        Ok(())
    }

    pub fn fetch_result(
        &self,
        mut encoder: wgpu::CommandEncoder,
        output_id: &PhysicalId,
        output_size: u64,
        page_table: &[Option<MemoryAddr>],
    ) -> Result<&[u8], BackendError> {
        let out_addr = page_table
            .get(output_id.0 as usize)
            .ok_or(BackendError::PhysicalDNE)?
            .ok_or(BackendError::PageDNE)?;

        encoder.copy_buffer_to_buffer(
            &self.slab_buffer,
            out_addr.0,
            &self.staging_buffer,
            0, // Will there be multiple outputs in a single model's output buffer? Likely not
            output_size,
        );

        self.compute_context.queue.submit(Some(encoder.finish()));

        {
            let (tx, rx) = std::sync::mpsc::channel();

            self.staging_buffer.map_async(
                wgpu::MapMode::Read,
                .., /* check what bounds should be */
                move |result| {
                    tx.send(result).unwrap();
                },
            );
            self.compute_context
                .device
                .poll(wgpu::PollType::wait_indefinitely())?;
            rx.recv()??;

            let output_data = self.staging_buffer.get_mapped_range(0..output_size);
            let num_data: &[f32] = bytemuck::cast_slice(&output_data);
            println!("GPU Result: {:?}", num_data);
        }

        Ok(&[5])
    }
}

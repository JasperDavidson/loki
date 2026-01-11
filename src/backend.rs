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

pub trait UniformBuffer: bytemuck::Pod + bytemuck::Zeroable {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
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

impl UniformBuffer for ModifyBackendUniformBuffer {}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BinaryOpUniformBuffer {
    left_read_offset: u32,
    right_read_offset: u32,
    write_offset: u32,
    // TOOD: Surely there's some way around this padding?
    _padding: u32,
    dim: [u32; 4],
}

impl UniformBuffer for BinaryOpUniformBuffer {}

impl BinaryOpUniformBuffer {
    pub fn new(
        left_read_offset: u32,
        right_read_offset: u32,
        write_offset: u32,
        dim: [u32; 4],
    ) -> Self {
        Self {
            left_read_offset,
            right_read_offset,
            write_offset,
            _padding: 0,
            dim,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelType {
    MatrixAccumulate,
    MatrixAddition,
}

// Exists so that the Scheduler can interact with the backend without interfacing with WGPU
pub struct DispatchDescription<T: UniformBuffer> {
    label: String,
    kernel_type: KernelType,
    workgroup_dim: (u32, u32, u32),
    // TODO: How to store uniform buffers + offset?
    uniform_data: T,
    uniform_offset: u64,
    cache_addresses: Option<Vec<MemoryAddr>>, // Optional cache addresses if kernel requires
                                              // (e.g. attention)
}

impl<T: UniformBuffer> DispatchDescription<T> {
    pub fn new(
        label: &str,
        kernel_type: KernelType,
        workgroup_dim: (u32, u32, u32),
        uniform_data: T,
        uniform_offset: u64,
        cache_addresses: Option<Vec<MemoryAddr>>,
    ) -> Self {
        Self {
            label: label.to_string(),
            kernel_type,
            workgroup_dim,
            uniform_data,
            uniform_offset,
            cache_addresses,
        }
    }
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
            size: (mem_size as f32 * 0.9) as u64,
            usage: wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let uniform_buffers = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Uniform Buffer Container").map(Into::into),
            size: (mem_size as f32 * 0.05) as u64, // TODO: Come up with a good way to determine this number
            usage: wgpu::BufferUsages::COPY_DST // TODO: Check if these are the correct access
            // modifiers we want
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        });
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer").map(Into::into),
            size: (mem_size as f32 * 0.05) as u64, // TODO: Come up with a good way to determine this number, should be
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

    // Maybe optimize by moving the values?
    pub fn process_descriptions<T: UniformBuffer>(
        &self,
        dispatch_descriptions: &[DispatchDescription<T>],
    ) -> Result<(), BackendError> {
        let kernel_types = dispatch_descriptions
            .iter()
            .map(|desc| desc.kernel_type)
            .collect();
        let workgroup_dims = dispatch_descriptions
            .iter()
            .map(|desc| desc.workgroup_dim)
            .collect();
        let mut bind_groups = Vec::with_capacity(dispatch_descriptions.len());

        for desc in dispatch_descriptions {
            self.write_uniform_mem(desc.uniform_data, desc.uniform_offset);
            bind_groups.push(self.create_bind_group(
                &desc.label,
                desc.uniform_offset,
                desc.uniform_data.as_bytes().len(),
                &desc.kernel_type,
            )?);
        }
        let encoder = self
            .compute_context
            .device
            .create_command_encoder(&Default::default());

        self.commit_actions(encoder, kernel_types, bind_groups, workgroup_dims)?;

        Ok(())
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

    // Writes data into the specified offset in the uniform buffer
    pub fn write_uniform_mem<T: UniformBuffer>(&self, uniform_buffer_data: T, offset: u64) {
        // Note: Submitted on next command buffer submission to queue
        // Hardware should automatically handle streaming for us between the transfer and
        // compute queue
        self.compute_context.queue.write_buffer(
            &self.uniform_buffers,
            offset,
            uniform_buffer_data.as_bytes(),
        );
    }

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

    pub fn commit_actions(
        &self,
        mut encoder: wgpu::CommandEncoder,
        kernel_types: Vec<KernelType>,
        // Sent as separate vectors to indicate they could perhaps be decoupled in the future
        bind_groups: Vec<wgpu::BindGroup>,
        workgroup_dims: Vec<(u32, u32, u32)>,
    ) -> Result<(), BackendError> {
        // Change these asserts in the future
        assert_eq!(kernel_types.len(), bind_groups.len());
        assert_eq!(bind_groups.len(), workgroup_dims.len());
        assert_eq!(workgroup_dims.len(), kernel_types.len());

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                ..Default::default()
            });

            for i in 0..kernel_types.len() {
                let context = self
                    .kernel_contexts
                    .get(&kernel_types[i])
                    .ok_or(BackendError::KernelDNE)?;
                compute_pass.set_pipeline(&context.pipeline);
                compute_pass.set_bind_group(0, &bind_groups[i], &[]);
                compute_pass.dispatch_workgroups(
                    workgroup_dims[i].0,
                    workgroup_dims[i].1,
                    workgroup_dims[i].2,
                );
            }
        }

        self.compute_context.queue.submit(Some(encoder.finish()));

        Ok(())
    }

    pub fn fetch_result(
        &self,
        output_id: &PhysicalId,
        output_size: u64,
        page_table: &[Option<MemoryAddr>],
    ) -> Result<Vec<f32>, BackendError> {
        let out_addr = page_table
            .get(output_id.0 as usize)
            .ok_or(BackendError::PhysicalDNE)?
            .ok_or(BackendError::PageDNE)?;

        let mut encoder = self
            .compute_context
            .device
            .create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(
            &self.slab_buffer,
            out_addr.0,
            &self.staging_buffer,
            0, // Will there be multiple outputs in a single model's output buffer? Likely not
            output_size,
        );

        self.compute_context.queue.submit(Some(encoder.finish()));
        let recv_data: Vec<f32>;

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
            recv_data = bytemuck::cast_slice(&output_data).to_vec();
        }

        println!("{:?}", recv_data);
        Ok(recv_data)
    }
}

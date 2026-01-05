use std::collections::HashMap;
use thiserror::Error;
use wgpu::{BindGroupEntry, BufferUses, PipelineLayoutDescriptor};

struct GpuComputeContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

struct KernelContext {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

#[derive(PartialEq, Eq, Hash)]
pub enum KernelType {
    MatrixAddition,
}

#[derive(Debug, Error)]
pub enum BackendError {
    #[error("Kernel type not suppported")]
    KernelDNE,
}

pub struct BackendHandler {
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
            device.create_shader_module(wgpu::include_wgsl!("kernels/matrix_add.wgsl"));

        // Create the generic bind layouts for reuse across similar kernel layouts
        let binary_operation_bind_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                // .into() converts &'static str to Cow<'static, str>
                label: Some("Binary operation layout").map(Into::into),
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
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
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

        let unary_inplace_operation_bind_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("unary inplace op layout").map(Into::into),
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

        let unary_move_operation_bind_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("unary move op layout").map(Into::into),
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
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
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
                    bind_group_layouts: &[&unary_move_operation_bind_layout],
                    immediate_size: 0,
                })),
                module: &matrix_add_kernel,
                entry_point: None,
                compilation_options: Default::default(),
                cache: Default::default(),
            });

        let matrix_add_compute_context = KernelContext {
            pipeline: matrix_add_pipeline,
            bind_group_layout: unary_move_operation_bind_layout,
        };

        let slab_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Slab buffer").map(Into::into),
            size: mem_size,
            usage: wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let compute_context = GpuComputeContext { device, queue };
        let kernel_contexts =
            HashMap::from([(KernelType::MatrixAddition, matrix_add_compute_context)]);

        BackendHandler {
            slab_buffer,
            compute_context,
            kernel_contexts,
        }
    }

    // Note that when passing buffers here, ensure that the buffers match the bind group layout
    // associated with the desired kernel
    // The uniform buffer should always be bound to index 0
    fn create_bind_group(
        &self,
        label: &str,
        kernel_type: &KernelType,
        uniform_buf: &wgpu::Buffer,
        buffers: &[wgpu::Buffer],
    ) -> Result<wgpu::BindGroup, BackendError> {
        // Fetch this first to exit early if invalid kernel
        let layout = &self
            .kernel_contexts
            .get(kernel_type)
            .ok_or(BackendError::KernelDNE)?
            .bind_group_layout;

        let mut bind_group_entries = Vec::with_capacity(buffers.len() + 1);
        bind_group_entries.push(wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform_buf.as_entire_binding(),
        });

        for (i, buffer) in buffers.iter().enumerate() {
            bind_group_entries.push(BindGroupEntry {
                binding: (i + 1) as u32,
                resource: buffer.as_entire_binding(),
            });
        }

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

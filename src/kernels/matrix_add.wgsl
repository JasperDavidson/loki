struct Uniforms {
  left_read_offset: u32,
  right_read_offset: u32,
  write_offset: u32,
  _padding: u32,
  dim: vec4<u32>
};

// Uniform buffer
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
// Slab buffer
@group(0) @binding(1) var<storage, read_write> slab_buffer: array<f32>;

fn out_of_bounds(coord: vec3<u32>, bounds: vec3<u32>) -> bool {
  return coord.x > bounds.x || coord.y > bounds.y || coord.z > bounds.z;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  if(out_of_bounds(global_id, uniforms.dim.xyz)) {
    return;
  }

  let thread_idx = global_id.y * uniforms.dim.x + global_id.x;
  let left_idx = uniforms.left_read_offset + thread_idx;
  let right_idx = uniforms.right_read_offset + thread_idx;
  let scratchpad_idx = uniforms.write_offset + thread_idx;

  slab_buffer[scratchpad_idx] = slab_buffer[left_idx] + slab_buffer[right_idx];
}

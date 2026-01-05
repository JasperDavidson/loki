struct Uniforms {
  dim: vec4<u32>
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> additive_buffer: array<f32>;
@group(0) @binding(2) var<storage, read_write> write_buffer: array<f32>;

fn in_bounds(coord: vec3<u32>, bounds: vec3<u32>) -> bool {
  return coord.x <= bounds.x && coord.y <= bounds.y && coord.z <= bounds.z;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  if(in_bounds(global_id, uniforms.dim.xyz)) {
    let idx = global_id.y * uniforms.dim.x + global_id.x;

    write_buffer[idx] += additive_buffer[idx];
  }
}

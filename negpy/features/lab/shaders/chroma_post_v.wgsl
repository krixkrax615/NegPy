// Chroma Denoise: vertical half of the final Gaussian on the guided a*/b*. Taps are gaussian_kernel_1d(post_sigma)
// from logic.py (kernel_w); gp[2] is their half-width.
@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var out_tex: texture_storage_2d<rgba32float, write>;
// gp: [0] box radius (px), [1] eps (L*^2), [2] post-Gaussian half-width (px); built from chroma_guide_params in logic.py.
@group(0) @binding(3) var<storage, read> gp: array<f32>;
@group(0) @binding(4) var<storage, read> kernel_w: array<f32>;

// cv2's default border mode (BORDER_REFLECT_101) -- mirrors the CPU box and Gaussian filters.
fn reflect_101(c: i32, n: i32) -> i32 {
    var v = c;
    if (v < 0) { v = -v; }
    if (v >= n) { v = 2 * (n - 1) - v; }
    return clamp(v, 0, n - 1);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(input_tex);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    let coords = vec2<i32>(i32(gid.x), i32(gid.y));
    let r = i32(gp[2]);
    var acc = vec2<f32>(0.0);
    for (var j = -r; j <= r; j++) {
        let sy = reflect_101(coords.y + j, i32(dims.y));
        acc += textureLoad(input_tex, vec2<i32>(coords.x, sy), 0).xy * kernel_w[u32(j + r)];
    }
    textureStore(out_tex, coords, vec4<f32>(acc, 0.0, 0.0));
}

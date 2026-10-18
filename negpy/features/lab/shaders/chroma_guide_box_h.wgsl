// Chroma Denoise guided filter, step 1: horizontal box mean of (g, a, b, g*g) -> out_a and (g*a, g*b) -> out_b,
// g = L*. Mirrors the first _box_mean calls of guided_filter_ab in logic.py.
@group(0) @binding(0) var lab_tex: texture_2d<f32>;
@group(0) @binding(1) var out_a: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var out_b: texture_storage_2d<rgba32float, write>;
// gp: [0] box radius (px), [1] eps (L*^2), [2] post-Gaussian half-width (px); built from chroma_guide_params in logic.py.
@group(0) @binding(3) var<storage, read> gp: array<f32>;

// cv2's default border mode (BORDER_REFLECT_101) -- mirrors the CPU box and Gaussian filters.
fn reflect_101(c: i32, n: i32) -> i32 {
    var v = c;
    if (v < 0) { v = -v; }
    if (v >= n) { v = 2 * (n - 1) - v; }
    return clamp(v, 0, n - 1);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(lab_tex);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    let coords = vec2<i32>(i32(gid.x), i32(gid.y));
    let r = i32(gp[0]);
    var acc_a = vec4<f32>(0.0);
    var acc_b = vec2<f32>(0.0);
    for (var i = -r; i <= r; i++) {
        let sx = reflect_101(coords.x + i, i32(dims.x));
        let t = textureLoad(lab_tex, vec2<i32>(sx, coords.y), 0).xyz;
        acc_a += vec4<f32>(t.x, t.y, t.z, t.x * t.x);
        acc_b += vec2<f32>(t.x * t.y, t.x * t.z);
    }
    let inv = 1.0 / f32(2 * r + 1);
    textureStore(out_a, coords, acc_a * inv);
    textureStore(out_b, coords, vec4<f32>(acc_b * inv, 0.0, 0.0));
}

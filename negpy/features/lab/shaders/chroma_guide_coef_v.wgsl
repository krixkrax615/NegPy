// Chroma Denoise guided filter, step 2: vertical box mean finishes the window means, then the per-pixel linear
// fit: slope = cov(g, ab) / (max(var(g), 0) + eps), offset = mean(ab) - slope * mean(g). Writes (sa, sb, oa, ob).
@group(0) @binding(0) var stats_a: texture_2d<f32>;
@group(0) @binding(1) var stats_b: texture_2d<f32>;
@group(0) @binding(2) var out_tex: texture_storage_2d<rgba32float, write>;
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
    let dims = textureDimensions(stats_a);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    let coords = vec2<i32>(i32(gid.x), i32(gid.y));
    let r = i32(gp[0]);
    var acc_a = vec4<f32>(0.0);
    var acc_b = vec2<f32>(0.0);
    for (var j = -r; j <= r; j++) {
        let sy = reflect_101(coords.y + j, i32(dims.y));
        let p = vec2<i32>(coords.x, sy);
        acc_a += textureLoad(stats_a, p, 0);
        acc_b += textureLoad(stats_b, p, 0).xy;
    }
    let inv = 1.0 / f32(2 * r + 1);
    let m = acc_a * inv;
    let mg = acc_b * inv;
    let variance = max(m.w - m.x * m.x, 0.0);
    let cov = vec2<f32>(mg.x - m.x * m.y, mg.y - m.x * m.z);
    let slope = cov / (variance + gp[1]);
    let offset = m.yz - slope * m.x;
    textureStore(out_tex, coords, vec4<f32>(slope, offset));
}

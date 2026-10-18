// Luma Denoise self-guided filter, step 2: vertical box mean finishes the window mean and mean of squares, then
// slope = var / (var + eps), offset = (1 - slope) * mean. Writes (slope, offset, 0, 0).
@group(0) @binding(0) var stats_tex: texture_2d<f32>;
@group(0) @binding(2) var out_tex: texture_storage_2d<rgba32float, write>;
// lp: [0] box radius (px), [1] eps (L*^2); built from luma_guide_params in logic.py.
@group(0) @binding(3) var<storage, read> lp: array<f32>;

// cv2's default border mode (BORDER_REFLECT_101) -- mirrors the CPU box filter.
fn reflect_101(c: i32, n: i32) -> i32 {
    var v = c;
    if (v < 0) { v = -v; }
    if (v >= n) { v = 2 * (n - 1) - v; }
    return clamp(v, 0, n - 1);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(stats_tex);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    let coords = vec2<i32>(i32(gid.x), i32(gid.y));
    let r = i32(lp[0]);
    var acc = vec2<f32>(0.0);
    for (var j = -r; j <= r; j++) {
        let sy = reflect_101(coords.y + j, i32(dims.y));
        acc += textureLoad(stats_tex, vec2<i32>(coords.x, sy), 0).xy;
    }
    let m = acc / f32(2 * r + 1);
    let variance = max(m.y - m.x * m.x, 0.0);
    let slope = variance / (variance + lp[1]);
    textureStore(out_tex, coords, vec4<f32>(slope, (1.0 - slope) * m.x, 0.0, 0.0));
}

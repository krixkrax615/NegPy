// Luma Denoise self-guided filter, step 1: horizontal box mean of (L*, L*^2). Input is the (L*, a*, b*, 0)
// texture of chroma_guide_lab.wgsl. Mirrors the first _box_mean calls of self_guided_filter in logic.py.
@group(0) @binding(0) var lab_tex: texture_2d<f32>;
@group(0) @binding(1) var out_tex: texture_storage_2d<rgba32float, write>;
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
    let dims = textureDimensions(lab_tex);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    let coords = vec2<i32>(i32(gid.x), i32(gid.y));
    let r = i32(lp[0]);
    var acc = vec2<f32>(0.0);
    for (var i = -r; i <= r; i++) {
        let sx = reflect_101(coords.x + i, i32(dims.x));
        let l = textureLoad(lab_tex, vec2<i32>(sx, coords.y), 0).x;
        acc += vec2<f32>(l, l * l);
    }
    textureStore(out_tex, coords, vec4<f32>(acc / f32(2 * r + 1), 0.0, 0.0));
}

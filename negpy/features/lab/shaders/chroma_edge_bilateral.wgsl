// Chroma Denoise, colour-edge band: a bilateral filter on a*/b* (cv2.bilateralFilter on (a*, b*, 0), see
// _edge_band_chroma in logic.py). Reads (L*, a*, b*) from chroma_guide_lab.wgsl, writes (a*, b*, 0, 0).
// cv2 weights: space exp(-r^2 / (2 sigma_s^2)) inside the circular window r <= radius, colour exp(-d^2 / (2 sigma_c^2))
// with d the L1 distance over the three channels (the third is 0). BORDER_REFLECT_101 like the CPU.
@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var out_tex: texture_storage_2d<rgba32float, write>;
// gp: [4] window radius (px; 0 passes a*/b* through), [5] spatial sigma (px), [6] colour (range) sigma (a*/b* units).
@group(0) @binding(3) var<storage, read> gp: array<f32>;

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
    let centre = textureLoad(input_tex, coords, 0).yz;
    let radius = i32(gp[4]);
    if (radius <= 0) {
        textureStore(out_tex, coords, vec4<f32>(centre, 0.0, 0.0));
        return;
    }
    let space_k = -0.5 / (gp[5] * gp[5]);
    let range_k = -0.5 / (gp[6] * gp[6]);
    let r2_max = f32(radius * radius);
    var acc = vec2<f32>(0.0);
    var wsum = 0.0;
    for (var j = -radius; j <= radius; j++) {
        let sy = reflect_101(coords.y + j, i32(dims.y));
        for (var i = -radius; i <= radius; i++) {
            let d2 = f32(i * i + j * j);
            if (d2 > r2_max) { continue; }
            let sx = reflect_101(coords.x + i, i32(dims.x));
            let ab = textureLoad(input_tex, vec2<i32>(sx, sy), 0).yz;
            let dc = abs(ab.x - centre.x) + abs(ab.y - centre.y);
            let w = exp(d2 * space_k) * exp(dc * dc * range_k);
            acc += ab * w;
            wsum += w;
        }
    }
    textureStore(out_tex, coords, vec4<f32>(acc / wsum, 0.0, 0.0));
}

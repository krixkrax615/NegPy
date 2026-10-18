// Chroma Denoise guided filter, step 0: linear working RGB -> (L*, a*, b*, 0) so the box passes convert once.
@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var output_tex: texture_storage_2d<rgba32float, write>;

// Working-space TRC (Adobe RGB 1998: pure 563/256 gamma) -- mirrors oetf_decode in lab.wgsl.
fn oetf_decode(c: vec3<f32>) -> vec3<f32> {
    let e = max(c, vec3<f32>(0.0));
    return pow(e, vec3<f32>(2.19921875));
}

// CIELAB (L*, a*, b*) of linear Adobe RGB (D65) -- rgb_to_lab in lab.wgsl.
fn lab_of(rgb: vec3<f32>) -> vec3<f32> {
    let r = max(rgb.r, 0.0);
    let g = max(rgb.g, 0.0);
    let b = max(rgb.b, 0.0);
    var x = (r * 0.5767309 + g * 0.1855540 + b * 0.1881852) / 0.95047;
    var y = r * 0.2973769 + g * 0.6273491 + b * 0.0752741;
    var z = (r * 0.0270343 + g * 0.0706872 + b * 0.9911085) / 1.08883;
    if (x > 0.008856) { x = pow(x, 1.0/3.0); } else { x = (7.787 * x) + (16.0 / 116.0); }
    if (y > 0.008856) { y = pow(y, 1.0/3.0); } else { y = (7.787 * y) + (16.0 / 116.0); }
    if (z > 0.008856) { z = pow(z, 1.0/3.0); } else { z = (7.787 * z) + (16.0 / 116.0); }
    return vec3<f32>(116.0 * y - 16.0, 500.0 * (x - y), 200.0 * (y - z));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(input_tex);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    let coords = vec2<i32>(i32(gid.x), i32(gid.y));
    let lab = lab_of(oetf_decode(textureLoad(input_tex, coords, 0).rgb));
    textureStore(output_tex, coords, vec4<f32>(lab, 0.0));
}

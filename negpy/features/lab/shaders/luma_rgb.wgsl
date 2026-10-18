// Luma Denoise, sharpen source: the working RGB with L* replaced by the denoised L* (a*/b* kept), so the sharpen
// passes see the same L* the CPU does (it sharpens after luma denoise). Only run when Sharpening is on.
@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var luma_tex: texture_2d<f32>;
@group(0) @binding(2) var output_tex: texture_storage_2d<rgba32float, write>;

fn oetf_encode(c: vec3<f32>) -> vec3<f32> {
    let x = clamp(c, vec3<f32>(0.0), vec3<f32>(1.0));
    return pow(x, vec3<f32>(0.45470693));
}

fn oetf_decode(c: vec3<f32>) -> vec3<f32> {
    let e = max(c, vec3<f32>(0.0));
    return pow(e, vec3<f32>(2.19921875));
}

fn rgb_to_lab(rgb: vec3<f32>) -> vec3<f32> {
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

fn lab_to_rgb(lab: vec3<f32>) -> vec3<f32> {
    var y = (lab.x + 16.0) / 116.0;
    var x = lab.y / 500.0 + y;
    var z = y - lab.z / 200.0;
    if (pow(x, 3.0) > 0.008856) { x = pow(x, 3.0); } else { x = (x - 16.0 / 116.0) / 7.787; }
    if (pow(y, 3.0) > 0.008856) { y = pow(y, 3.0); } else { y = (y - 16.0 / 116.0) / 7.787; }
    if (pow(z, 3.0) > 0.008856) { z = pow(z, 3.0); } else { z = (z - 16.0 / 116.0) / 7.787; }
    x = x * 0.95047;
    z = z * 1.08883;
    let r = x * 2.0413690 + y * -0.5649464 + z * -0.3446944;
    let g = x * -0.9692660 + y * 1.8760108 + z * 0.0415560;
    let b = x * 0.0134474 + y * -0.1183897 + z * 1.0154096;
    return max(vec3<f32>(r, g, b), vec3<f32>(0.0));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(input_tex);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    let coords = vec2<i32>(i32(gid.x), i32(gid.y));
    let src = textureLoad(input_tex, coords, 0);
    let lab = rgb_to_lab(oetf_decode(src.rgb));
    let l = textureLoad(luma_tex, coords, 0).x;
    textureStore(output_tex, coords, vec4<f32>(oetf_encode(lab_to_rgb(vec3<f32>(l, lab.y, lab.z))), src.a));
}

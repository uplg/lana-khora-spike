// Faithful port of lana-avatar's `point.wgsl` to standalone WGSL (no
// bevy_pbr imports). The model transform is identity (like lana-avatar's
// CloudRoot), so world == model space. The ONLY intentional difference
// vs lana-avatar: KhoraEngine has no HDR/bloom pipeline, so the
// `EMISSIVE_K` scaling clips instead of blooming — the glow is flatter.
// Everything else (flow drift, jitter, dissolve, back-cull, two-ring
// eyes, scan sweep, mouth-cluster lip-sync, breath, hue drift) matches.

struct Globals {
    view_proj: mat4x4<f32>,
    // xyz: camera world position · w: time (s)
    cam_time: vec4<f32>,
    // p: x openness · y emissive K · z back-cull · w mouth-band centre Y
    p: vec4<f32>,
    // q: x mouth half-height · y mouth amp · z jitter · w mouth X half-width
    q: vec4<f32>,
    // r: x eye-centre Y · y eye-centre |X| · z outer-ring r · w inner-ring r
    r: vec4<f32>,
    // s: x eye-centre Z · y flow amp · z dissolve · w life
    s: vec4<f32>,
};
@group(0) @binding(0) var<uniform> g: Globals;

const SPAN: f32 = 1.7;

fn hash13(p: vec3<f32>) -> f32 {
    return fract(sin(dot(p, vec3<f32>(12.9898, 78.233, 37.719))) * 43758.5453);
}

// Smooth, spatially-coherent vector field: nearby points drift together.
fn flow(p: vec3<f32>, t: f32) -> vec3<f32> {
    return vec3<f32>(
        sin(p.y * 2.3 + t * 0.6) + cos(p.z * 1.9 - t * 0.4),
        sin(p.z * 2.1 + t * 0.5) + cos(p.x * 2.4 + t * 0.35),
        sin(p.x * 2.0 - t * 0.45) + cos(p.y * 1.7 + t * 0.55),
    );
}

// Rotate an RGB colour about the grey axis (hue shift, luminance kept).
fn hue_rot(c: vec3<f32>, a: f32) -> vec3<f32> {
    let k = vec3<f32>(0.57735, 0.57735, 0.57735);
    let ct = cos(a);
    return c * ct + cross(k, c) * sin(a) + k * dot(k, c) * (1.0 - ct);
}

struct VIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) wnormal: vec3<f32>,
    @location(1) wpos: vec3<f32>,
    @location(2) ny: f32,
    @location(3) h: f32,
    @location(4) nx: f32,
    @location(5) nz: f32,
};

@vertex
fn vs_main(v: VIn) -> VOut {
    let time = g.cam_time.w;
    var pos = v.position;
    let h = hash13(v.position);
    let ph = h * 6.2831853;

    // Coherent flow drift + a touch of fine per-point jitter.
    pos = pos + flow(v.position, time) * g.s.y;
    let j = g.q.z;
    pos.x = pos.x + sin(time * 1.7 + ph) * j;
    pos.y = pos.y + cos(time * 1.5 + ph) * j;
    pos.z = pos.z + sin(time * 1.3 + ph) * j;

    // Lip-sync: thin band just below the mouth line drops by openness,
    // gated in X to the mouth cluster.
    let below = clamp((g.p.w - pos.y) / g.q.x, 0.0, 1.0);
    let drop = below * (1.0 - below) * 4.0;
    let xw = 1.0 - smoothstep(0.0, g.q.w, abs(v.position.x));
    pos.y = pos.y - drop * xw * g.p.x * g.q.y;

    var out: VOut;
    out.clip = g.view_proj * vec4<f32>(pos, 1.0);
    out.wpos = pos;
    out.wnormal = normalize(v.normal);
    out.ny = v.position.y;
    out.nx = v.position.x;
    out.nz = v.position.z;
    out.h = h;
    return out;
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {
    let time = g.cam_time.w;
    let life = g.s.w;

    // Dissolve / recompose: zones materialise and fade (a hologram).
    if (fract(in.h * 7.0 + time * 0.08) < g.s.z) {
        discard;
    }

    let n = normalize(in.wnormal);
    let viewdir = normalize(g.cam_time.xyz - in.wpos);
    if (dot(n, viewdir) < g.p.z) {
        discard;
    }

    var col = n * 0.5 + 0.5;

    // Synthetic geometric eye: two concentric glowing particle rings.
    let gz = life * 0.010;
    let gaze = vec2<f32>(sin(time * 0.37) * gz,
                         sin(time * 0.53 + 1.3) * gz * 0.6);
    let eye_c = vec3<f32>(g.r.y + gaze.x, g.r.x + gaze.y, g.s.x);
    let d_eye = distance(vec3<f32>(abs(in.nx), in.ny, in.nz), eye_c);
    let r_out = g.r.z;
    if (d_eye < r_out) {
        let w = r_out * 0.16;
        let on = max(
            1.0 - smoothstep(0.0, w, abs(d_eye - r_out * 0.92)),
            1.0 - smoothstep(0.0, w, abs(d_eye - g.r.w)),
        );
        if (on < 0.2) {
            discard;
        }
        col = vec3<f32>(0.30, 0.95, 1.20) * (0.7 + on * 0.8);
    }

    // Braindance scan sweep.
    let scan_y = (sin(time * 0.8) * 0.5 + 0.5) * SPAN;
    let scan = clamp(1.0 - abs(in.ny - scan_y) / (SPAN * 0.05), 0.0, 1.0);

    // Lower-mouth cluster glows a touch while speaking.
    let mlo = clamp((g.p.w - in.ny) / g.q.x, 0.0, 1.0);
    let mxw = 1.0 - smoothstep(0.0, g.q.w, abs(in.nx));
    let speak = mlo * (1.0 - mlo) * 4.0 * mxw * g.p.x;

    // Life: slow breath in brightness + a gentle hue drift.
    let breath = 1.0 + sin(time * 0.9) * 0.05 * life;
    col = hue_rot(col, sin(time * 0.05) * 0.5 * life);

    let bright = (0.8 + scan * 0.5 + speak * 0.45) * breath;
    return vec4<f32>(col * g.p.y * bright, 1.0);
}

//! Lana avatar viz, on KhoraEngine (throwaway spike).
//!
//! Goal: prove Lana's point-cloud avatar can render through Khora instead
//! of Bevy, from a standalone app, **without editing KhoraEngine**.
//!
//! Mechanism (the de-risk): the engine's `AgentId` enum is closed, so an
//! external app cannot register a custom render `Agent`. Instead we drive
//! a custom `Lane` directly from the `EngineApp::before_agents` hook —
//! which (per the winit run-loop) fires *after* `begin_render_frame`
//! populates the `FrameContext`/`ColorTarget` and *before* `submit_passes`
//! drains the `SharedFrameGraph`. We record our point-cloud pass into a
//! command encoder and push it onto the frame graph exactly the way the
//! engine's own `UiAgent` pushes its UI pass. No `Agent`, no engine edits.

mod glb;
mod points;

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use khora_sdk::khora_core::lane::{ColorTarget, Lane, LaneContext, Slot};
use khora_sdk::khora_core::platform::KhoraWindow;
use khora_sdk::khora_core::renderer::GraphicsDevice;
use khora_sdk::khora_core::renderer::api::core::FrameContext;
use khora_sdk::khora_core::renderer::traits::CommandEncoder;
use khora_sdk::prelude::math::{Mat4, Vec3};
use khora_sdk::prelude::{InputEvent, MouseButton};
use khora_sdk::run_winit;
use khora_sdk::winit_adapters::WinitWindowProvider;
use khora_sdk::{
    AgentProvider, DccService, EngineApp, GameWorld, PhaseProvider, RenderSystem, Runtime,
    WgpuRenderSystem, WindowConfig,
};

use points::{Globals, PointCloudLane};

const WIN_W: u32 = 1280;
const WIN_H: u32 = 800;
/// Max points kept (one draw) — matches lana-avatar's `TARGET_PTS`.
const TARGET_POINTS: usize = 120_000;
/// All clouds normalised to this height, feet at y=0 — lana-avatar's `TARGET_H`.
const TARGET_H: f32 = 1.7;

/// Orbit camera — identical model to lana-avatar's `OrbitCam`: it orbits
/// the vertical axis at `target_y` (face height), looking at
/// `(0, target_y, 0)`.
struct Orbit {
    target_y: f32,
    yaw: f32,
    pitch: f32,
    dist: f32,
    dragging: bool,
    last: (f32, f32),
}

impl Orbit {
    /// Eye position — same formula as lana-avatar `OrbitCam::eye`.
    fn eye(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        Vec3::new(
            self.dist * cp * sy,
            self.target_y + self.dist * sp,
            self.dist * cp * cy,
        )
    }

    fn view_proj(&self, aspect: f32) -> (Mat4, Vec3) {
        let eye = self.eye();
        let view = Mat4::look_at_rh(eye, Vec3::new(0.0, self.target_y, 0.0), Vec3::Y)
            .unwrap_or(Mat4::IDENTITY);
        // Bevy's Camera3d default perspective fov is FRAC_PI_4 (45°).
        let proj = Mat4::perspective_rh_zo(std::f32::consts::FRAC_PI_4, aspect, 0.05, 100.0);
        (proj * view, eye)
    }
}

/// Loads model vertices, normalises them **exactly like lana-avatar**
/// (centre X/Z, drop feet to y=0, scale so height = `TARGET_H`), and
/// returns interleaved `[pos(3), normal(3)]` floats, the max normalised Y
/// (camera `target_y` derives from it) and the auto-detected eye centroid
/// `(|x|, y, z)` (for the two-ring eye). Falls back to a procedural sphere.
fn load_verts() -> (Vec<f32>, f32, [f32; 3]) {
    let path = std::env::var_os("LANA_KHORA_MODEL")
        .map(PathBuf::from)
        .or_else(|| {
            std::fs::read_dir(".").ok().and_then(|rd| {
                rd.filter_map(Result::ok)
                    .map(|e| e.path())
                    .find(|p| {
                        matches!(
                            p.extension().and_then(|e| e.to_str()),
                            Some("glb" | "vrm" | "pcd")
                        )
                    })
            })
        });

    let pts = path
        .as_deref()
        .and_then(|p| {
            log::info!("loading model: {}", p.display());
            glb::sample_points(p, TARGET_POINTS)
        })
        .unwrap_or_else(|| {
            log::warn!("no model found — procedural sphere fallback");
            procedural_sphere(20_000)
        });

    // lana-avatar `normalize`: centre X/Z, drop feet to y=0, scale the
    // model's height to TARGET_H.
    let mut lo = [f32::MAX; 3];
    let mut hi = [f32::MIN; 3];
    for (p, _) in &pts {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    let s = TARGET_H / (hi[1] - lo[1]).max(1e-3);
    let cx = (lo[0] + hi[0]) * 0.5;
    let cz = (lo[2] + hi[2]) * 0.5;

    let mut out = Vec::with_capacity(pts.len() * 6);
    let mut max_y = 0.0_f32;
    for (p, n) in pts {
        let y = (p[1] - lo[1]) * s;
        max_y = max_y.max(y);
        out.extend_from_slice(&[
            (p[0] - cx) * s,
            y,
            (p[2] - cz) * s,
            n[0],
            n[1],
            n[2],
        ]);
    }

    // lana-avatar seeds the eye search at TARGET_H*0.92 ± 0.11.
    let eye_y0 = TARGET_H * 0.92;
    let eye = detect_eyes(&out, eye_y0, 0.11).unwrap_or_else(|| {
        log::warn!("eye cluster not found — fixed fallback (0.07, {eye_y0}, 0.4)");
        [0.07, eye_y0, 0.4]
    });
    log::info!("eye centroid (|x|,y,z) = {eye:?}");
    (out, max_y, eye)
}

/// Auto-detect the eye centroid — port of lana-avatar `detect_eyes`. Eyes
/// are forward-facing points (`z>0`, `n.z>0.2`) in a Y band around eye
/// level, split L/R in X; returns the mean `(|x|, y, z)` of both eyes.
fn detect_eyes(verts: &[f32], y0: f32, band: f32) -> Option<[f32; 3]> {
    let (mut lx, mut ly, mut lz, mut ln) = (0.0_f32, 0.0, 0.0, 0.0);
    let (mut rx, mut ry, mut rz, mut rn) = (0.0_f32, 0.0, 0.0, 0.0);
    for c in verts.chunks_exact(6) {
        let (px, py, pz, nz) = (c[0], c[1], c[2], c[5]);
        if (py - y0).abs() >= band || pz <= 0.0 || nz <= 0.2 {
            continue;
        }
        if px < 0.0 {
            lx += px;
            ly += py;
            lz += pz;
            ln += 1.0;
        } else {
            rx += px;
            ry += py;
            rz += pz;
            rn += 1.0;
        }
    }
    if ln <= 0.0 || rn <= 0.0 {
        return None;
    }
    Some([
        ((lx / ln).abs() + rx / rn) * 0.5,
        (ly / ln + ry / rn) * 0.5,
        (lz / ln + rz / rn) * 0.5,
    ])
}

/// Fibonacci-sphere fallback so the window is never empty.
fn procedural_sphere(n: usize) -> Vec<glb::Point> {
    let golden = std::f32::consts::PI * (3.0 - 5.0_f32.sqrt());
    (0..n)
        .map(|i| {
            let y = 1.0 - (i as f32 / (n as f32 - 1.0)) * 2.0;
            let r = (1.0 - y * y).max(0.0).sqrt();
            let th = golden * i as f32;
            let v = [r * th.cos(), y, r * th.sin()];
            (v, v)
        })
        .collect()
}

struct LanaKhora {
    verts: Vec<f32>,
    lane: Option<PointCloudLane>,
    inited: bool,
    cam: Orbit,
    start: Instant,
    frames: u64,
    /// Current window aspect (w/h), refreshed every frame in `before_frame`
    /// so a resize doesn't squash the avatar.
    aspect: f32,
    /// Constant material params (lana-avatar's PointMaterial p/q/r/s).
    p: [f32; 4],
    q: [f32; 4],
    r: [f32; 4],
    s: [f32; 4],
}

impl EngineApp for LanaKhora {
    fn window_config() -> WindowConfig {
        WindowConfig {
            title: "Lana — Khora spike".to_owned(),
            width: WIN_W,
            height: WIN_H,
            icon: None,
        }
    }

    fn new() -> Self {
        let (verts, max_y, eye) = load_verts();

        // lana-avatar PointMaterial constants (cloud.rs `setup`).
        const EMISSIVE_K: f32 = 2.6;
        const CULL: f32 = 0.05;
        let mouth_y = TARGET_H * 0.89; // MOUTH_CY
        let p = [0.0, EMISSIVE_K, CULL, mouth_y];
        let q = [0.010, 0.008, 0.006, 0.03]; // mouth_h, amp, jitter, mouth_w
        let r = [eye[1], eye[0], 0.018, 0.008]; // eye_y, eye_x, eye_r, pupil_r
        let s = [eye[2], 0.010, 0.15, 1.0]; // eye_z, flow, dissolve, life

        // Pinned defaults identical to lana-avatar (overridable via the
        // same env vars so a pose pinned there transfers verbatim).
        let cam_dist = std::env::var("LANA_AVATAR_CAM_DIST")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .filter(|d| *d > 0.05)
            .unwrap_or(0.535);
        let cam_y_frac = std::env::var("LANA_AVATAR_CAM_Y")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .filter(|f| (0.0..=1.5).contains(f))
            .unwrap_or(0.92);

        Self {
            verts,
            lane: None,
            inited: false,
            cam: Orbit {
                target_y: max_y * cam_y_frac,
                yaw: 0.0,
                pitch: 0.08,
                dist: cam_dist,
                dragging: false,
                last: (0.0, 0.0),
            },
            start: Instant::now(),
            frames: 0,
            aspect: WIN_W as f32 / WIN_H as f32,
            p,
            q,
            r,
            s,
        }
    }

    fn setup(&mut self, _world: &mut GameWorld, _runtime: &Runtime) {}

    // Refresh the aspect from the live window every frame so a resize
    // rescales the projection instead of squashing the avatar.
    fn before_frame(
        &mut self,
        _world: &mut GameWorld,
        _runtime: &Runtime,
        window: &dyn KhoraWindow,
    ) {
        let (w, h) = window.inner_size();
        if w > 0 && h > 0 {
            self.aspect = w as f32 / h as f32;
        }
    }

    fn update(&mut self, _world: &mut GameWorld, inputs: &[InputEvent]) {
        // No auto-spin: the camera stays put (face-framed) unless the user
        // left-drags. Same orbit sensitivities as lana-avatar.
        for ev in inputs {
            match ev {
                InputEvent::MouseButtonPressed {
                    button: MouseButton::Left,
                } => self.cam.dragging = true,
                InputEvent::MouseButtonReleased {
                    button: MouseButton::Left,
                } => self.cam.dragging = false,
                InputEvent::MouseMoved { x, y } => {
                    if self.cam.dragging {
                        self.cam.yaw -= (x - self.cam.last.0) * 0.006;
                        self.cam.pitch = (self.cam.pitch
                            - (y - self.cam.last.1) * 0.006)
                            .clamp(-1.4, 1.4);
                    }
                    self.cam.last = (*x, *y);
                }
                _ => {}
            }
        }
    }

    // We inject from `after_agents` (NOT `before_agents`): the winit loop
    // runs before_agents → run_scheduler (RenderAgent adds its ScenePass) →
    // submit_passes → after_agents → present_frame. Khora's `sorted_passes`
    // orders frame-graph passes writer→reader with insertion order as the
    // tie-break, so a pass added in `before_agents` lands BEFORE
    // RenderAgent's clear-only ScenePass and gets wiped. From `after_agents`
    // we bypass the frame graph entirely and submit our command buffer
    // straight to the device queue — it executes last, on top, just before
    // `present_frame`.
    fn after_agents(&mut self, _world: &mut GameWorld, runtime: &Runtime) {
        self.frames += 1;
        let dbg = self.frames <= 3;
        macro_rules! d {
            ($($a:tt)*) => { if dbg { log::info!($($a)*); } };
        }

        let Some(device) = runtime.backends.get::<Arc<dyn GraphicsDevice>>().cloned() else {
            d!("after_agents f{}: NO GraphicsDevice in backends", self.frames);
            return;
        };
        d!("after_agents f{}: device OK", self.frames);

        if self.lane.is_none() {
            self.lane = Some(PointCloudLane::new(std::mem::take(&mut self.verts)));
        }
        let Some(lane) = self.lane.as_ref() else {
            return;
        };

        if !self.inited {
            let mut ictx = LaneContext::new();
            ictx.insert(device.clone());
            match lane.on_initialize(&mut ictx) {
                Ok(()) => {
                    self.inited = true;
                    log::info!("before_agents: PointCloudLane on_initialize OK");
                }
                Err(e) => {
                    log::error!("PointCloudLane init failed: {e:?}");
                    return;
                }
            }
        }

        // Per-frame uniform: orbit camera + time. Material params (p/q/r/s)
        // are constant; openness (p.x) stays 0 — idle, like lana-avatar
        // when not speaking (no audio in this viz).
        let (vp, eye) = self.cam.view_proj(self.aspect);
        let t = self.start.elapsed().as_secs_f32();
        let globals = Globals {
            view_proj: vp.to_cols_array_2d(),
            cam_time: [eye.x, eye.y, eye.z, t],
            p: self.p,
            q: self.q,
            r: self.r,
            s: self.s,
        };

        let Some(fctx) = runtime.resources.get::<Arc<FrameContext>>() else {
            d!("after_agents f{}: NO FrameContext", self.frames);
            return;
        };
        let Some(color_target) = fctx.get::<ColorTarget>().map(|a| *a) else {
            d!("after_agents f{}: NO ColorTarget (occluded frame)", self.frames);
            return;
        };
        d!("after_agents f{}: fctx+colortarget OK", self.frames);

        let mut encoder = device.create_command_encoder(Some("Lana Points Encoder"));
        {
            let mut ctx = LaneContext::new();
            ctx.insert(device.clone());
            ctx.insert(color_target);
            ctx.insert(globals);
            // SAFETY: identity transmute — launders the encoder borrow's
            // lifetime so the `Slot` satisfies `LaneContext::insert`'s
            // `'static` bound. `ctx` is dropped before `encoder.finish()`.
            // This mirrors the engine's own `UiAgent::execute`.
            let slot = Slot::new(encoder.as_mut());
            ctx.insert(unsafe {
                std::mem::transmute::<Slot<dyn CommandEncoder>, Slot<dyn CommandEncoder>>(slot)
            });
            if let Err(e) = lane.execute(&mut ctx) {
                log::error!("PointCloudLane execute failed: {e:?}");
            }
        }

        let Some(cmd_buf) = encoder.finish() else {
            log::error!("encoder.finish() returned None");
            return;
        };
        // Direct queue submit — NOT the frame graph. RenderAgent's
        // clear-only ScenePass was already submitted by `submit_passes`;
        // ours executes after it, on the same swapchain target, just
        // before `present_frame`.
        device.submit_command_buffer(cmd_buf);
        d!("after_agents f{}: cmd buffer submitted directly", self.frames);
    }
}

impl AgentProvider for LanaKhora {
    fn register_agents(&self, _dcc: &DccService, _runtime: &mut Runtime) {
        // Intentionally empty — the whole point of the spike is that the
        // custom render path needs NO custom Agent (AgentId is closed).
    }
}

impl PhaseProvider for LanaKhora {}

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info"),
    )
    .filter_module("wgpu_hal::vulkan::instance", log::LevelFilter::Off)
    // Khora logs an ERROR every frame the window isn't frontmost
    // ("begin_frame failed: ... Occluded"). Harmless surface-acquire
    // skip; silenced so the console stays readable.
    .filter_module("khora_sdk::engine", log::LevelFilter::Off)
    .init();

    run_winit::<WinitWindowProvider, LanaKhora>(|window, runtime, _event_loop| {
        let mut rs = WgpuRenderSystem::new();
        rs.init(window).expect("renderer init failed");
        runtime.backends.insert(rs.graphics_device());
        let rs: Box<dyn RenderSystem> = Box::new(rs);
        runtime.backends.insert(Arc::new(Mutex::new(rs)));
    })?;
    Ok(())
}

//! `PointCloudLane` — a custom `khora_core::lane::Lane` that renders the
//! avatar's vertices as a `PrimitiveTopology::PointList` draw through a
//! hand-written WGSL pipeline.
//!
//! This is the spike's core proof: a custom GPU pipeline created via the
//! public `GraphicsDevice` trait and recorded into the frame, with **no
//! `Agent`** (the `AgentId` enum is closed, so an external app can't add
//! one) and **no edits to KhoraEngine**. The owning `EngineApp` drives
//! this lane from its `before_agents` hook, mirroring how the engine's
//! own `UiAgent` drives `UiRenderLane`.

use std::borrow::Cow;
use std::num::NonZeroU64;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, OnceLock};

use khora_sdk::khora_core::lane::{ColorTarget, Lane, LaneContext, LaneError, LaneKind, Slot};
use khora_sdk::khora_core::renderer::GraphicsDevice;
use khora_sdk::khora_core::renderer::api::command::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindGroupLayoutId, BindGroupId, BindingType, BufferBindingType, LoadOp, Operations,
    RenderPassColorAttachment, RenderPassDescriptor, StoreOp,
};
use khora_sdk::khora_core::renderer::api::core::{ShaderModuleDescriptor, ShaderSourceData};
use khora_sdk::khora_core::renderer::api::pipeline::enums::{
    PrimitiveTopology, VertexFormat, VertexStepMode,
};
use khora_sdk::khora_core::renderer::api::pipeline::state::ColorWrites;
use khora_sdk::khora_core::renderer::api::pipeline::{
    ColorTargetStateDescriptor, MultisampleStateDescriptor, PipelineLayoutDescriptor,
    PrimitiveStateDescriptor, RenderPipelineDescriptor, RenderPipelineId,
    VertexAttributeDescriptor, VertexBufferLayoutDescriptor,
};
use khora_sdk::khora_core::renderer::api::resource::{BufferDescriptor, BufferId, BufferUsage};
use khora_sdk::khora_core::renderer::api::util::{SampleCount, ShaderStageFlags, TextureFormat};
use khora_sdk::khora_core::renderer::traits::CommandEncoder;

/// Per-frame uniform block. Mirrors `struct Globals` in `point.wgsl`.
/// The owning app inserts a fresh one into the `LaneContext` each frame.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Globals {
    /// Column-major view·projection matrix.
    pub view_proj: [[f32; 4]; 4],
    /// xyz: camera world position · w: time (s)
    pub cam_time: [f32; 4],
    /// x openness · y emissive K · z back-cull dot · w mouth-band centre Y
    pub p: [f32; 4],
    /// x mouth half-height · y mouth amp · z jitter · w mouth X half-width
    pub q: [f32; 4],
    /// x eye-centre Y · y eye-centre |X| · z outer-ring r · w inner-ring r
    pub r: [f32; 4],
    /// x eye-centre Z · y flow amp · z dissolve · w life
    pub s: [f32; 4],
}

const GLOBALS_SIZE: u64 = std::mem::size_of::<Globals>() as u64;

/// A lane that draws `n_points` vertices (interleaved pos+normal) as a
/// glowing point cloud. All GPU handles are created once in
/// `on_initialize` and read lock-free thereafter.
pub struct PointCloudLane {
    /// Interleaved vertex data: `[px,py,pz, nx,ny,nz]` per point.
    verts: Vec<f32>,
    /// Number of points (`verts.len() / 6`).
    n_points: u32,
    /// Near-black clear colour for the avatar scene.
    clear: khora_sdk::khora_core::math::LinearRgba,

    pipeline: OnceLock<RenderPipelineId>,
    bind_layout: OnceLock<BindGroupLayoutId>,
    bind_group: OnceLock<BindGroupId>,
    ubuf: OnceLock<BufferId>,
    vbuf: OnceLock<BufferId>,
    calls: AtomicU32,
}

impl PointCloudLane {
    /// Builds the lane from interleaved `[pos(3), normal(3)]` floats.
    pub fn new(verts: Vec<f32>) -> Self {
        let n_points = u32::try_from(verts.len() / 6).unwrap_or(0);
        Self {
            verts,
            n_points,
            clear: khora_sdk::khora_core::math::LinearRgba::new(0.01, 0.01, 0.02, 1.0),
            pipeline: OnceLock::new(),
            bind_layout: OnceLock::new(),
            bind_group: OnceLock::new(),
            ubuf: OnceLock::new(),
            vbuf: OnceLock::new(),
            calls: AtomicU32::new(0),
        }
    }

    fn gpu_init(&self, device: &dyn GraphicsDevice) -> Result<(), LaneError> {
        let init_err = |what: &str| {
            LaneError::InitializationFailed(Box::new(std::io::Error::other(what.to_owned())))
        };

        // Bind group layout: one uniform buffer at binding 0.
        let bind_layout = device
            .create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("lana_points_bgl"),
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStageFlags::VERTEX | ShaderStageFlags::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(GLOBALS_SIZE),
                    },
                }],
            })
            .map_err(|_| init_err("create_bind_group_layout"))?;
        let _ = self.bind_layout.set(bind_layout);

        // Shader module (WGSL embedded at compile time).
        let shader = device
            .create_shader_module(&ShaderModuleDescriptor {
                label: Some("lana_points_shader"),
                source: ShaderSourceData::Wgsl(Cow::Borrowed(include_str!("point.wgsl"))),
            })
            .map_err(|_| init_err("create_shader_module"))?;

        // Vertex layout: pos vec3 @0, normal vec3 @1 (stride 24).
        let vertex_layout = VertexBufferLayoutDescriptor {
            array_stride: 24,
            step_mode: VertexStepMode::Vertex,
            attributes: Cow::Owned(vec![
                VertexAttributeDescriptor {
                    format: VertexFormat::Float32x3,
                    offset: 0,
                    shader_location: 0,
                },
                VertexAttributeDescriptor {
                    format: VertexFormat::Float32x3,
                    offset: 12,
                    shader_location: 1,
                },
            ]),
        };

        let layout_ids = [bind_layout];
        let pipeline_layout = device
            .create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some(Cow::Borrowed("lana_points_pl")),
                bind_group_layouts: &layout_ids,
            })
            .map_err(|_| init_err("create_pipeline_layout"))?;

        let pipeline = device
            .create_render_pipeline(&RenderPipelineDescriptor {
                label: Some(Cow::Borrowed("lana_points_pipeline")),
                vertex_shader_module: shader,
                vertex_entry_point: Cow::Borrowed("vs_main"),
                fragment_shader_module: Some(shader),
                fragment_entry_point: Some(Cow::Borrowed("fs_main")),
                vertex_buffers_layout: Cow::Owned(vec![vertex_layout]),
                layout: Some(pipeline_layout),
                primitive_state: PrimitiveStateDescriptor {
                    topology: PrimitiveTopology::PointList,
                    ..Default::default()
                },
                depth_stencil_state: None,
                color_target_states: Cow::Owned(vec![ColorTargetStateDescriptor {
                    format: device
                        .get_surface_format()
                        .unwrap_or(TextureFormat::Rgba8UnormSrgb),
                    blend: None,
                    write_mask: ColorWrites::ALL,
                }]),
                multisample_state: MultisampleStateDescriptor {
                    count: SampleCount::X1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
            })
            .map_err(|_| init_err("create_render_pipeline"))?;
        let _ = self.pipeline.set(pipeline);

        // Uniform buffer + its bind group.
        let ubuf = device
            .create_buffer(&BufferDescriptor {
                label: Some(Cow::Borrowed("lana_points_ubo")),
                size: GLOBALS_SIZE,
                usage: BufferUsage::UNIFORM | BufferUsage::COPY_DST,
                mapped_at_creation: false,
            })
            .map_err(|_| init_err("create_buffer(ubo)"))?;
        let _ = self.ubuf.set(ubuf);

        let bind_group = device
            .create_bind_group(&BindGroupDescriptor {
                label: Some("lana_points_bg"),
                layout: bind_layout,
                entries: &[BindGroupEntry::buffer(0, ubuf, 0, NonZeroU64::new(GLOBALS_SIZE))],
            })
            .map_err(|_| init_err("create_bind_group"))?;
        let _ = self.bind_group.set(bind_group);

        // Vertex buffer (static; uploaded once).
        //
        // WORKAROUND for a KhoraEngine bug: `WgpuDevice::create_buffer`
        // maps usage via `wgpu::BufferUsages::from_bits_truncate(usage
        // .bits())` (a raw bit copy) instead of the correct, already-
        // existing `IntoWgpu<wgpu::BufferUsages>` impl. Khora's
        // `BufferUsage` has VERTEX=1<<4 / INDEX=1<<5 while wgpu has
        // INDEX=1<<4 / VERTEX=1<<5, so a VERTEX buffer silently becomes
        // an INDEX buffer (wgpu rejects `set_vertex_buffer`). Requesting
        // VERTEX|INDEX makes the raw copy yield wgpu INDEX|VERTEX, so the
        // VERTEX flag is present either way. Remove the INDEX bit once the
        // engine's `create_buffer` uses `descriptor.usage.into_wgpu()`.
        let vbytes: &[u8] = bytemuck::cast_slice(&self.verts);
        let vbuf = device
            .create_buffer(&BufferDescriptor {
                label: Some(Cow::Borrowed("lana_points_vbo")),
                size: vbytes.len() as u64,
                usage: BufferUsage::VERTEX | BufferUsage::INDEX | BufferUsage::COPY_DST,
                mapped_at_creation: false,
            })
            .map_err(|_| init_err("create_buffer(vbo)"))?;
        device
            .write_buffer(vbuf, 0, vbytes)
            .map_err(|_| init_err("write_buffer(vbo)"))?;
        let _ = self.vbuf.set(vbuf);

        log::info!("PointCloudLane: GPU init OK ({} points)", self.n_points);
        Ok(())
    }
}

impl Lane for PointCloudLane {
    fn strategy_name(&self) -> &'static str {
        "LanaPointCloud"
    }

    fn lane_kind(&self) -> LaneKind {
        LaneKind::Render
    }

    fn on_initialize(&self, ctx: &mut LaneContext) -> Result<(), LaneError> {
        let device = ctx
            .get::<Arc<dyn GraphicsDevice>>()
            .ok_or_else(|| LaneError::missing("Arc<dyn GraphicsDevice>"))?
            .clone();
        self.gpu_init(device.as_ref())
    }

    fn execute(&self, ctx: &mut LaneContext) -> Result<(), LaneError> {
        let globals = *ctx
            .get::<Globals>()
            .ok_or_else(|| LaneError::missing("Globals"))?;
        let device = ctx
            .get::<Arc<dyn GraphicsDevice>>()
            .ok_or_else(|| LaneError::missing("Arc<dyn GraphicsDevice>"))?
            .clone();
        let color_target = ctx
            .get::<ColorTarget>()
            .ok_or_else(|| LaneError::missing("ColorTarget"))?
            .0;
        let encoder = ctx
            .get::<Slot<dyn CommandEncoder>>()
            .ok_or_else(|| LaneError::missing("Slot<dyn CommandEncoder>"))?
            .get();

        let (Some(&pipeline), Some(&bind_group), Some(&ubuf), Some(&vbuf)) = (
            self.pipeline.get(),
            self.bind_group.get(),
            self.ubuf.get(),
            self.vbuf.get(),
        ) else {
            return Err(LaneError::missing("PointCloudLane GPU handles"));
        };

        device
            .write_buffer(ubuf, 0, bytemuck::bytes_of(&globals))
            .map_err(|_| LaneError::missing("write_buffer(ubo)"))?;

        let color_attachment = RenderPassColorAttachment {
            view: &color_target,
            resolve_target: None,
            ops: Operations {
                load: LoadOp::Clear(self.clear),
                store: StoreOp::Store,
            },
            base_array_layer: 0,
        };
        let desc = RenderPassDescriptor {
            label: Some("Lana Point Cloud Pass"),
            color_attachments: &[color_attachment],
            depth_stencil_attachment: None,
        };

        let mut pass = encoder.begin_render_pass(&desc);
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_vertex_buffer(0, &vbuf, 0);
        pass.draw(0..self.n_points, 0..1);

        if self.calls.fetch_add(1, Ordering::Relaxed) < 3 {
            log::info!(
                "PointCloudLane::execute: drew {} points; vp[0]={:?}",
                self.n_points,
                &globals.view_proj[0]
            );
        }
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

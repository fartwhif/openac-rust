//! The world renderer without a window: the regular world renderer draws
//! into a texture this module owns instead of the window's swapchain, and
//! every frame is read back to CPU memory as RGBA8.
//!
//! No window, no display and no swapchain are involved: the device runs
//! on whatever adapter the platform enumerates (a GPU, or a software
//! Vulkan implementation such as llvmpipe on a headless box), and what
//! comes back as pixels is exactly what a window would have presented —
//! the terrain, the buildings, the actors, the sky — for the caller to
//! encode however it likes (the terminal front end turns it into braille
//! cells).

use std::sync::Arc;
use std::time::Instant;

use crate::camera::FlyInput;
use crate::device_health::DeviceHealth;
use crate::renderer::{WorldRenderer, WorldSettings, WorldStats};
use crate::scene::{Scene, SceneSource};
use crate::surface::{block_on, choose_adapter, device_descriptor, supported_sample_counts};
use crate::{GraphicsOptions, RenderError};

/// The offscreen target's format: 8-bit, non-sRGB, like the window's frame
/// buffer (the screenshot readback uses the same formats).
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8Unorm;

/// The clear a frame starts with when the world's own clear is not known
/// yet (a day-sky blue, as the window spec's default).
const CLEAR_FALLBACK: wgpu::Color = wgpu::Color {
    r: 0.25,
    g: 0.35,
    b: 0.5,
    a: 1.0,
};

/// A world renderer drawing into its own texture, with the last completed
/// frame read back as RGBA8.
/// The world renderer is not debug-formattable, and neither is this,
/// once it holds one.
#[allow(missing_debug_implementations)]
pub struct Offscreen {
    device: wgpu::Device,
    queue: wgpu::Queue,
    health: Arc<DeviceHealth>,
    world: WorldRenderer,
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    width: u32,
    height: u32,
    scene: Option<Box<SceneSource>>,
    notices: Vec<String>,
    /// The previous frame's copy, while it is in flight or mapped.
    readback: wgpu::Buffer,
    readback_pending: bool,
    /// The last completed frame: RGBA8, `width * height * 4` bytes.
    pixels: Vec<u8>,
    has_frame: bool,
    /// The adapter's name (the tests' unknown one empty).
    adapter_name: String,
    /// The adapter's backend, as `{:?}`.
    adapter_backend: String,
}

/// A target texture for a frame.
fn target(device: &wgpu::Device, width: u32, height: u32) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("offscreen frame"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

/// The readback buffer for a `width` x `height` frame (rows padded to the
/// copy alignment).
fn readback_buffer(device: &wgpu::Device, width: u32, height: u32) -> wgpu::Buffer {
    let row = width * 4;
    let padded_row =
        row.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("offscreen readback"),
        size: u64::from(padded_row) * u64::from(height),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    })
}

impl Offscreen {
    /// Opens the offscreen target at `width` x `height` pixels: picks the
    /// adapter (`graphics` carries the `OPENAC_GPU_ADAPTER` override and the
    /// capability checks), makes the device, and builds the world renderer
    /// from `world_settings` (the dats) and `display` (the quality preset).
    pub fn new(
        width: u32,
        height: u32,
        world_settings: &WorldSettings,
        display: &ac_host::settings::DisplaySettings,
        graphics: &GraphicsOptions,
    ) -> Result<Self, RenderError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let (adapter, _summary) = choose_adapter(&instance, None, graphics)?;
        let mut features = wgpu::Features::empty();
        // Multisample counts beyond the WebGPU guarantee (4) need the
        // adapter's own format features.
        if adapter
            .features()
            .contains(wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES)
        {
            features |= wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES;
        }
        // Block-compressed dat textures stay compressed on the GPU when it
        // can sample them; without it they are decoded to RGBA on the loader.
        if !graphics.no_block_compression
            && adapter
                .features()
                .contains(wgpu::Features::TEXTURE_COMPRESSION_BC)
        {
            features |= wgpu::Features::TEXTURE_COMPRESSION_BC;
        }
        let (device, queue) =
            block_on(adapter.request_device(&device_descriptor("ac-render offscreen", features)))?;
        Ok(Self::on(
            Some(&adapter),
            &device,
            &queue,
            width,
            height,
            world_settings,
            display,
        ))
    }

    /// On an existing device (the tests): `adapter` supplies the format
    /// capabilities, `None` when unknown (no multisampling beyond 1).
    pub(crate) fn on(
        adapter: Option<&wgpu::Adapter>,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        world_settings: &WorldSettings,
        display: &ac_host::settings::DisplaySettings,
    ) -> Self {
        let width = width.max(1);
        let height = height.max(1);
        let health = DeviceHealth::watch(device);
        let texture = target(device, width, height);
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let msaa_supported = adapter
            .map(|a| supported_sample_counts(a, device, FORMAT))
            .unwrap_or_else(|| vec![1]);
        let adapter_name = adapter.map(|a| a.get_info().name).unwrap_or_default();
        let adapter_backend = adapter
            .map(|a| format!("{:?}", a.get_info().backend))
            .unwrap_or_default();
        let world = WorldRenderer::new(
            device,
            queue,
            FORMAT,
            width,
            height,
            world_settings,
            display,
            &msaa_supported,
        );
        Self {
            device: device.clone(),
            queue: queue.clone(),
            health,
            world,
            texture,
            view,
            width,
            height,
            scene: None,
            notices: Vec::new(),
            readback: readback_buffer(device, width, height),
            readback_pending: false,
            pixels: Vec::new(),
            has_frame: false,
            adapter_name,
            adapter_backend,
        }
    }

    /// The session's scene this target follows: the receiver the session
    /// sends its presentations on (and the command channel it answers
    /// through, when it was built with one).
    pub fn attach_scene(&mut self, source: Box<SceneSource>) {
        self.scene = Some(source);
    }

    /// The scene so far (the 2-D interface reads its facts from here).
    #[must_use]
    pub fn scene(&self) -> Option<&Scene> {
        self.scene.as_deref().map(SceneSource::scene)
    }

    /// The scene, to take its effects (chat, status, interface events).
    #[must_use]
    pub fn scene_mut(&mut self) -> Option<&mut Scene> {
        self.scene.as_mut().map(|source| source.scene_mut())
    }

    /// The session that feeds this target ended.
    #[must_use]
    pub fn session_gone(&self) -> bool {
        self.scene
            .as_ref()
            .is_some_and(|source| source.session_gone())
    }

    /// The world's notices since the last call.
    #[must_use]
    pub fn take_notices(&mut self) -> Vec<String> {
        std::mem::take(&mut self.notices)
    }

    /// The last frame's statistics (landblocks, drawn objects).
    #[must_use]
    pub fn stats(&self) -> WorldStats {
        self.world.stats()
    }

    /// The frame's size in pixels.
    #[must_use]
    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }
    /// The adapter's name, drawing on it (the startup line).
    #[must_use]
    pub fn adapter_name(&self) -> &str {
        &self.adapter_name
    }

    /// The adapter's backend, as `{:?}` (llvmpipe is the Vulkan one).
    #[must_use]
    pub fn adapter_backend(&self) -> &str {
        &self.adapter_backend
    }

    /// Follows a new target size (a terminal resize): the texture, the
    /// readback and the world renderer's frame follow.
    pub fn resize(&mut self, width: u32, height: u32) {
        let width = width.max(1);
        let height = height.max(1);
        if (width, height) == (self.width, self.height) {
            return;
        }
        self.width = width;
        self.height = height;
        let texture = target(&self.device, width, height);
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.texture = texture;
        self.view = view;
        self.readback = readback_buffer(&self.device, width, height);
        self.readback_pending = false;
        self.has_frame = false;
        self.world.resize(&self.device, width, height);
    }

    /// Advances the scene and draws a frame into the target; the copy of
    /// this frame starts its readback. `true` when a completed frame is
    /// available in [`Self::frame`] (the previous one, once the first
    /// frame has finished on the GPU).
    pub fn render(&mut self, fly: FlyInput) -> Result<bool, RenderError> {
        if let Some(lost) = self.health.lost() {
            return Err(RenderError::Gpu(lost));
        }
        let mut ready = false;
        if self.readback_pending {
            // Everything submitted so far is finished by the poll: the
            // previous frame's copy maps and the loader's uploads of the
            // last prepare are done.
            let (tx, rx) = std::sync::mpsc::channel();
            self.readback
                .map_async(wgpu::MapMode::Read, .., move |result| {
                    let _ = tx.send(result);
                });
            self.health
                .poll(&self.device, wgpu::PollType::wait_indefinitely())
                .map_err(RenderError::Gpu)?;
            rx.recv()
                .map_err(|_| RenderError::Screenshot("the readback was dropped".into()))?
                .map_err(|e| RenderError::Screenshot(format!("mapping the readback: {e}")))?;
            let row = (self.width * 4) as usize;
            let padded_row = (row.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize))
                * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize;
            let mut rgba = Vec::with_capacity(row * self.height as usize);
            {
                let mapped =
                    self.readback.slice(..).get_mapped_range().map_err(|e| {
                        RenderError::Screenshot(format!("reading the readback: {e}"))
                    })?;
                for line in mapped.chunks(padded_row).take(self.height as usize) {
                    for px in line[..row].chunks_exact(4) {
                        // The target is BGRA: out as RGBA, alpha opaque.
                        rgba.extend_from_slice(&[px[2], px[1], px[0], 255]);
                    }
                }
            }
            self.readback.unmap();
            self.pixels = rgba;
            self.has_frame = true;
            self.readback_pending = false;
            ready = true;
        }

        let now = Instant::now();
        if let Some(source) = self.scene.as_mut() {
            source.pump();
            source.scene_mut().present_at(now);
        }
        let motions = self
            .scene
            .as_mut()
            .map(|source| source.scene_mut().take_motions())
            .unwrap_or_default();
        let world = &mut self.world;
        world.queue_motions(motions);
        world.prepare(
            &self.device,
            &self.queue,
            self.scene.as_deref().map(SceneSource::scene),
            fly,
        );
        self.notices.extend(world.take_notices());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("offscreen frame"),
            });
        let clear = world.clear_color(CLEAR_FALLBACK);
        world.draw(&mut encoder, &self.view, clear, None);
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(
                        (self.width * 4).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
                            * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT,
                    ),
                    rows_per_image: Some(self.height),
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit([encoder.finish()]);
        self.readback_pending = true;
        Ok(ready)
    }

    /// The last completed frame (RGBA8), `None` until the first one has
    /// finished on the GPU.
    #[must_use]
    pub fn frame(&self) -> Option<&[u8]> {
        self.has_frame.then_some(&self.pixels)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// The settings a frame draws with when the dats are not there: the
    /// world has nothing to draw and the pass only clears.
    fn no_dat_settings() -> WorldSettings {
        WorldSettings {
            content: Arc::new(crate::content::ContentSlot::new(PathBuf::from(
                "no-dats-for-this-test",
            ))),
            quality_overrides: crate::quality::QualityOverrides {
                near_radius: Some(0),
                far_radius: Some(0),
                msaa_samples: Some(0),
                ..Default::default()
            },
            residency: Default::default(),
            orbit: Default::default(),
            reveal_radius: None,
            sky: Default::default(),
            audio: Default::default(),
        }
    }

    /// A surfaceless frame: with no dats the world only clears, and the
    /// readback is that clear colour, every pixel of it.
    #[test]
    fn a_frame_renders_to_the_offscreen_target_and_reads_back() {
        let Some((device, queue)) = crate::test_gpu::device() else {
            return;
        };
        let display = ac_host::settings::DisplaySettings::default();
        let mut off = Offscreen::on(None, &device, &queue, 8, 4, &no_dat_settings(), &display);
        // The first render starts the readback; the second finishes it.
        assert!(!off.render(FlyInput::default()).unwrap());
        assert!(off.render(FlyInput::default()).unwrap());
        let frame = off.frame().expect("the frame has read back");
        assert_eq!(frame.len(), 8 * 4 * 4);
        let (first, rest) = frame.split_at(4);
        for px in rest.chunks_exact(4) {
            assert_eq!(px, first, "a cleared frame is one colour");
        }
        assert_eq!(frame[3], 255, "the readback's alpha is opaque");

        // A resize follows the world renderer and the readback; the new
        // size's readback starts on the first render and completes on the
        // second, as at the start.
        off.resize(4, 2);
        assert_eq!(off.size(), (4, 2));
        assert!(!off.render(FlyInput::default()).unwrap());
        assert!(off.render(FlyInput::default()).unwrap());
        assert_eq!(
            off.frame().expect("the resized frame reads back").len(),
            4 * 2 * 4
        );
    }
}

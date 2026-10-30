//! The `--ui terminal` front end: the 3-D world and the game's 2-D GUI,
//! software-rendered to a plain terminal as braille dots with 24-bit colour.
//!
//! There is no GPU and no image protocol on the target, so the frame is a grid
//! of sub-pixels: each terminal cell holds a 2-wide by 4-tall braille dot
//! matrix (square in a typical 1:2 cell) and a foreground/background colour.
//! The world is projected with the game's own chase camera and light with the
//! game's sun, ambient and fog; the 2-D GUI is the real ImGui `DrawList`.

use std::collections::HashMap;
use std::io::{self, Write};
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::{Arc, mpsc};
use std::time::{Duration, Instant};

use ac_core::terrain::{SurfaceInfo, TerrainBlendingContext};
use ac_core::world::LiveWorldOrigin;
use ac_render::assets::{self, GfxObjData, LandblockData};
use ac_render::camera::{self, CameraController, Lens, OrbitCamera};
use ac_render::content::Content;
use ac_render::environment::{SkyOptions, WindowEnvironment};
use ac_render::gameplay_input::{DeviceState, GameplayInput, GameplayOutput};
use ac_render::scene::{Scene, SceneSource};
use ac_render::streaming::StreamingRegion;
use ac_render::texture_detail::WorldTextureDetail;
use ac_render::ui::{UiPixels, UiTextureSource, WindowUi};
use ac_render::world_frame::{self, LandblockCoord};
use ac_runtime::session::{
    LifecycleCommand, ObjectSummary, PresentationEvent, SessionCommand, SessionEvent,
    SessionHandle, SessionSummary,
};
use ac_ui::draw::{Color as UiColor, DrawCmd, DrawList, Quad, TextureRef};
use ac_ui::input::bindings::KeyBindings;
use ac_ui::{Key, Modifiers};
use crossterm::event::{
    self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind, KeyModifiers,
    MouseEventKind,
};
use glam::{Mat4, Quat, Vec3, Vec4};

use crate::options::{LogLevel, Options};
use crate::stdio::errln;

const REGION_ID: u32 = 0x1300_0000;
/// How many landblocks the drawing frame keeps centered on the player.
const ORIGIN_KEEP_RADIUS: u32 = 2;
/// A frame's target period (the session self-paces; this only caps the UI).
const FRAME_PERIOD: Duration = Duration::from_millis(66); // ~15 fps
/// How many server objects are drawn (nearest first) each frame.
const MAX_DRAWN_OBJECTS: usize = 64;
/// The height (px) the game's GUI is authored for; the width is derived from
/// the terminal's aspect so the whole canvas scales uniformly.
const UI_DESIGN_HEIGHT: u32 = 720;
/// The top-down minimap's side, in sub-pixels (8x8 terminal cells).
const MINIMAP_SIDE: usize = 16;
/// The world span the minimap shows, in meters (player centered).
const MINIMAP_SPAN: f32 = 120.0;

// ---------------------------------------------------------------------------
// Session logging: stderr, so it never disturbs the rendered frame.
// ---------------------------------------------------------------------------

/// A [`super::SessionLog`] that writes progress and problems to standard
/// error, leaving standard output for the rendered world.
pub struct RenderLog {
    level: LogLevel,
    started: Instant,
}

impl RenderLog {
    #[must_use]
    pub fn new(level: LogLevel) -> Self {
        Self {
            level,
            started: Instant::now(),
        }
    }

    fn stamp(&self) -> String {
        format!("[{:>8.3}]", self.started.elapsed().as_secs_f64())
    }
}

impl super::SessionLog for RenderLog {
    fn info(&self, text: &str) {
        if self.level >= LogLevel::Info {
            errln!("{} {text}", self.stamp());
        }
    }

    fn warn(&self, text: &str) {
        if self.level >= LogLevel::Warn {
            errln!("{} warn: {text}", self.stamp());
        }
    }

    fn error(&self, text: &str) {
        errln!("{} error: {text}", self.stamp());
    }

    fn session_event(&self, _event: &SessionEvent) {
        // The 2-D HUD reads the scene; no text feed here.
    }

    fn objects(&self, _objects: &[ObjectSummary]) {
        // No text object list in the terminal view.
    }

    fn feed(&self, text: &str) {
        if self.level >= LogLevel::Debug {
            errln!("{text}");
        }
    }

    fn summary(&self, summary: &SessionSummary) {
        if let Some(error) = &summary.error {
            errln!("{} ended with an error: {error}", self.stamp());
        }
    }
}

// ---------------------------------------------------------------------------
// Frame buffer: sub-pixel colour + depth at braille resolution.
// ---------------------------------------------------------------------------

/// One sub-pixel's colour, display sRGB in 0..1.
type Color = [f32; 3];

/// A rendered frame: one colour and a depth per sub-pixel. The buffer is
/// `2 * cols` wide by `4 * rows` tall, so each terminal cell covers a 2 x 4
/// block of sub-pixels.
struct Frame {
    width: usize,
    height: usize,
    color: Vec<Color>,
    depth: Vec<f32>,
}

impl Frame {
    fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            color: vec![[0.0; 3]; width * height],
            depth: vec![1.0; width * height],
        }
    }
}

/// A terminal cell after encoding: the braille character and its colours.
struct Cell {
    ch: char,
    fg: [u8; 3],
    bg: [u8; 3],
}

const BRIT_BITS: [[u8; 2]; 4] = [[0, 3], [1, 4], [2, 5], [6, 7]];

fn luminance(c: Color) -> f32 {
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}

fn to_u8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// Encode the sub-pixel buffer into a grid of braille cells.
fn encode(frame: &Frame, cols: usize, rows: usize) -> Vec<Vec<Cell>> {
    let mut grid = Vec::with_capacity(rows);
    for cy in 0..rows {
        let mut row = Vec::with_capacity(cols);
        for cx in 0..cols {
            let mut colors = [[0.0f32; 3]; 8];
            let mut lums = [0.0f32; 8];
            for dy in 0..4 {
                for dx in 0..2 {
                    let i = (cy * 4 + dy) * frame.width + (cx * 2 + dx);
                    let c = frame.color[i];
                    let n = dy * 2 + dx;
                    colors[n] = c;
                    lums[n] = luminance(c);
                }
            }
            let min_lum = lums.iter().cloned().fold(f32::INFINITY, f32::min);
            let max_lum = lums.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let (fg, bg, bits): ([u8; 3], [u8; 3], u16) = if max_lum - min_lum < 0.02 {
                // A flat region: a solid block of the average colour.
                let avg: Color = [
                    (colors.iter().map(|c| c[0]).sum::<f32>() / 8.0),
                    (colors.iter().map(|c| c[1]).sum::<f32>() / 8.0),
                    (colors.iter().map(|c| c[2]).sum::<f32>() / 8.0),
                ];
                let fg = [to_u8(avg[0]), to_u8(avg[1]), to_u8(avg[2])];
                (fg, fg, 0xFF)
            } else {
                // Two-colour relief: the darkest sub-pixel fills the cell,
                // the brighter ones light the dots.
                let mid = (min_lum + max_lum) * 0.5;
                let mut min_i = 0usize;
                let mut max_i = 0usize;
                for i in 0..8 {
                    if lums[i] < lums[min_i] {
                        min_i = i;
                    }
                    if lums[i] > lums[max_i] {
                        max_i = i;
                    }
                }
                let mut bits = 0u16;
                for i in 0..8 {
                    if lums[i] >= mid {
                        bits |= 1 << BRIT_BITS[i / 2][i % 2];
                    }
                }
                let fg = [
                    to_u8(colors[max_i][0]),
                    to_u8(colors[max_i][1]),
                    to_u8(colors[max_i][2]),
                ];
                let bg = [
                    to_u8(colors[min_i][0]),
                    to_u8(colors[min_i][1]),
                    to_u8(colors[min_i][2]),
                ];
                (fg, bg, bits)
            };
            row.push(Cell {
                ch: char::from_u32(0x2800 | (bits as u32)).unwrap_or(' '),
                fg,
                bg,
            });
        }
        grid.push(row);
    }
    grid
}

// ---------------------------------------------------------------------------
// Software rasterizer: project + z-buffer + sun/ambient/fog shading + sky.
// ---------------------------------------------------------------------------

/// The light and fog a frame is drawn with.
struct Light {
    eye: Vec3,
    sun_dir: Vec3,
    sun_color: Color,
    ambient: Color,
    fog_color: Color,
    fog_start: f32,
    fog_end: f32,
}

/// A world-space triangle gathered for a frame, ready to rasterize.
struct Tri {
    a: Vec3,
    b: Vec3,
    c: Vec3,
    color: Color,
}

struct Raster<'a> {
    frame: &'a mut Frame,
    w: f32,
    h: f32,
    view: Mat4,
    proj: Mat4,
    inv_view_proj: Mat4,
    light: &'a Light,
}

impl<'a> Raster<'a> {
    fn new(frame: &'a mut Frame, view: Mat4, proj: Mat4, light: &'a Light) -> Self {
        Self {
            w: frame.width as f32,
            h: frame.height as f32,
            inv_view_proj: (proj * view).inverse(),
            view,
            proj,
            frame,
            light,
        }
    }

    /// Project a drawing-frame point to sub-pixels; `None` behind the camera.
    fn project(&self, p: Vec3) -> Option<[f32; 3]> {
        let clip = self.proj * self.view * Vec4::new(p.x, p.y, p.z, 1.0);
        if clip.w <= 0.0 {
            return None;
        }
        let x = (clip.x / clip.w) * 0.5 + 0.5;
        let y = 0.5 - (clip.y / clip.w) * 0.5;
        let z = clip.z / clip.w;
        Some([x * self.w, y * self.h, z])
    }

    /// Fill the buffer with the sky (the background behind the world).
    fn clear_sky(&mut self) {
        let (horizon, zenith) = sky_colors(self.light);
        let eye = self.light.eye;
        let inv = self.inv_view_proj;
        for y in 0..self.frame.height {
            let ny = 1.0 - 2.0 * (y as f32 + 0.5) / self.h;
            for x in 0..self.frame.width {
                let nx = 2.0 * (x as f32 + 0.5) / self.w - 1.0;
                let near = inv * Vec4::new(nx, ny, 0.0, 1.0);
                let ray = (Vec3::new(near.x, near.y, near.z) - eye).normalize_or(Vec3::Z);
                let up = ray.z.clamp(0.0, 1.0);
                let mut c = [
                    horizon[0] + (zenith[0] - horizon[0]) * up,
                    horizon[1] + (zenith[1] - horizon[1]) * up,
                    horizon[2] + (zenith[2] - horizon[2]) * up,
                ];
                if ray.z < 0.0 {
                    c = horizon;
                }
                let to_sun = ray.dot(self.light.sun_dir);
                if to_sun > 0.0 {
                    let glow = to_sun.powf(24.0) * 1.5 + (to_sun - 0.9995).max(0.0) * 40.0;
                    c[0] += self.light.sun_color[0] * glow;
                    c[1] += self.light.sun_color[1] * glow;
                    c[2] += self.light.sun_color[2] * glow;
                }
                let i = y * self.frame.width + x;
                self.frame.color[i] = [
                    c[0].clamp(0.0, 1.0),
                    c[1].clamp(0.0, 1.0),
                    c[2].clamp(0.0, 1.0),
                ];
            }
        }
        for d in self.frame.depth.iter_mut() {
            *d = 1.0;
        }
    }

    /// Draw one world-space triangle with flat lighting and per-vertex fog.
    fn draw_tri(&mut self, a: Vec3, b: Vec3, c: Vec3, albedo: Color) {
        let face = (b - a).cross(c - a);
        let area2 = face.length_squared();
        if area2 <= 1e-12 {
            return;
        }
        let normal = face / face.length();
        let centroid = (a + b + c) * (1.0 / 3.0);
        // Back-face cull.
        if normal.dot(self.light.eye - centroid) < 0.0 {
            return;
        }
        let Some(sa) = self.project(a) else { return };
        let Some(sb) = self.project(b) else { return };
        let Some(sc) = self.project(c) else { return };

        // Flat diffuse: light from the sun toward the surface.
        let ndl = normal.dot(self.light.sun_dir).max(0.0);
        let col: Color = [
            albedo[0] * (self.light.sun_color[0] * ndl + self.light.ambient[0]),
            albedo[1] * (self.light.sun_color[1] * ndl + self.light.ambient[1]),
            albedo[2] * (self.light.sun_color[2] * ndl + self.light.ambient[2]),
        ];
        // Per-vertex fog factor, blended into the colour per fragment.
        let fog = |p: Vec3| -> f32 {
            let d = p.distance(self.light.eye);
            ((d - self.light.fog_start) / (self.light.fog_end - self.light.fog_start))
                .clamp(0.0, 1.0)
        };
        let fa = fog(a);
        let fb = fog(b);
        let fc = fog(c);

        let x0 = sa[0].floor() as i32;
        let y0 = sa[1].floor() as i32;
        let x1 = sc[0].ceil() as i32;
        let y1 = sc[1].ceil() as i32;
        let xmin = x0.min(sb[0].floor() as i32).max(0);
        let xmax = x1.max(sb[0].ceil() as i32).min(self.w as i32 - 1);
        let ymin = y0.min(sb[1].floor() as i32).max(0);
        let ymax = y1.max(sb[1].ceil() as i32).min(self.h as i32 - 1);
        if xmin > xmax || ymin > ymax {
            return;
        }

        let edge = |a: [f32; 3], b: [f32; 3], p: [f32; 2]| {
            (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        };
        let areaw = (sb[0] - sa[0]) * (sc[1] - sa[1]) - (sb[1] - sa[1]) * (sc[0] - sa[0]);
        if areaw.abs() < 1e-6 {
            return;
        }
        let w_neg = areaw < 0.0;

        for py in ymin..=ymax {
            for px in xmin..=xmax {
                let p = [px as f32 + 0.5, py as f32 + 0.5];
                let wa = edge(sb, sc, p);
                let wb = edge(sc, sa, p);
                let wc = edge(sa, sb, p);
                let inside = if w_neg {
                    wa <= 0.0 && wb <= 0.0 && wc <= 0.0
                } else {
                    wa >= 0.0 && wb >= 0.0 && wc >= 0.0
                };
                if !inside {
                    continue;
                }
                let u0 = wa / areaw;
                let u1 = wb / areaw;
                let u2 = wc / areaw;
                let z = u0 * sa[2] + u1 * sb[2] + u2 * sc[2];
                let i = py as usize * self.frame.width + px as usize;
                if z >= self.frame.depth[i] {
                    continue;
                }
                let f = u0 * fa + u1 * fb + u2 * fc;
                self.frame.depth[i] = z;
                self.frame.color[i] = [
                    col[0] * (1.0 - f) + self.light.fog_color[0] * f,
                    col[1] * (1.0 - f) + self.light.fog_color[1] * f,
                    col[2] * (1.0 - f) + self.light.fog_color[2] * f,
                ];
            }
        }
    }
}

/// The sky's horizon and zenith colours from the light (tuned by eye).
fn sky_colors(light: &Light) -> (Color, Color) {
    let daylight = light.sun_dir.z.clamp(0.0, 1.0);
    let horizon = light.fog_color;
    let zenith: Color = [
        light.ambient[0] * 0.4 + 0.02 + 0.05 * daylight,
        light.ambient[1] * 0.4 + 0.06 + 0.1 * daylight,
        light.ambient[2] * 0.4 + 0.12 + 0.2 * daylight,
    ];
    (horizon, zenith)
}

// ---------------------------------------------------------------------------
// The renderer: content, camera, environment, streaming and the frame.
// ---------------------------------------------------------------------------

struct Renderer {
    content: Content,
    height_table: [f32; 256],
    blending: TerrainBlendingContext,
    surface_cache: HashMap<u32, SurfaceInfo>,
    cameras: CameraController,
    lens: Lens,
    env: Option<WindowEnvironment>,
    streaming: StreamingRegion,
    origin: LiveWorldOrigin,
    landblocks: HashMap<LandblockCoord, LandblockData>,
    gfx: HashMap<u32, GfxObjData>,
    gameplay: GameplayInput,
    ui: Option<WindowUi>,
    ui_size: (u32, u32),
    frame: Frame,
}

impl Renderer {
    fn new(
        dat_dir: Option<PathBuf>,
        landblock_radius: Option<u32>,
        cols: usize,
        rows: usize,
    ) -> Result<Self, String> {
        let near = landblock_radius.unwrap_or(2);
        let far = near + 3;
        let dir = dat_dir.or_else(ac_dat::default_dat_dir).ok_or_else(|| {
            "no dat directory found (set OPENAC_DAT_DIR or --dat-dir)".to_string()
        })?;
        let content = Content::open_for_area(&dir, near, far)
            .map_err(|e| format!("could not open the dats at {}: {e}", dir.display()))?;
        let region = content
            .dats
            .get::<ac_dat::objects::Region>(REGION_ID)
            .ok()
            .flatten()
            .ok_or_else(|| "the region record is missing from the dats".to_string())?;
        let height_table = region.land_defs.land_height_table;
        let atlas = assets::build_terrain_atlas(&content, &region, WorldTextureDetail::FULL);
        let sky = assets::region_sky(&content, &region);
        let env = sky.as_ref().and_then(|s| {
            WindowEnvironment::new(
                s.clone(),
                region.game_time.zero_time_of_year,
                SkyOptions::default(),
            )
        });
        let gameplay = GameplayInput::new(KeyBindings::retail_defaults());
        // The 2-D game GUI, laid out at a size whose aspect matches the
        // terminal so the canvas scales uniformly. Best-effort: if the UI dats
        // are unavailable, the 3-D world still renders.
        let ui_h = UI_DESIGN_HEIGHT;
        let ui_w = (ui_h as f32 * (2 * cols) as f32 / (4 * rows).max(1) as f32)
            .round()
            .max(64.0) as u32;
        let ui_size = (ui_w, ui_h);
        let ui = ac_dat::DatCollection::open(&dir)
            .ok()
            .map(Arc::new)
            .and_then(|dats| WindowUi::build(dats, ui_size).ok());
        Ok(Self {
            content,
            height_table,
            blending: atlas.blending,
            surface_cache: HashMap::new(),
            cameras: CameraController::new(OrbitCamera::default()),
            lens: Lens::default(),
            env,
            streaming: StreamingRegion::new(0, 0, near, far),
            origin: LiveWorldOrigin::default(),
            landblocks: HashMap::new(),
            gfx: HashMap::new(),
            gameplay,
            ui,
            ui_size,
            frame: Frame::new(2 * cols, 4 * rows),
        })
    }

    fn ensure_landblock(&mut self, coord: LandblockCoord) {
        if self.landblocks.contains_key(&coord) {
            return;
        }
        let Some(lb) = assets::build_landblock(
            &self.content,
            coord,
            &self.height_table,
            &self.blending,
            &mut self.surface_cache,
        ) else {
            return;
        };
        self.landblocks.insert(coord, lb);
    }

    /// Render one frame from the scene, returning the grid of cells.
    fn render_frame(
        &mut self,
        scene: &Scene,
        dt: f32,
        now: Instant,
        cols: usize,
        rows: usize,
    ) -> Vec<Vec<Cell>> {
        if 2 * cols != self.frame.width || 4 * rows != self.frame.height {
            self.frame = Frame::new(2 * cols, 4 * rows);
        }
        self.lens
            .set_aspect((2 * cols) as f32 / (4 * rows).max(1) as f32);

        let player_pos = scene.player().and_then(|p| p.position);
        let player_world: Option<Vec3> = player_pos
            .as_ref()
            .map(|p| world_frame::to_frame(&self.origin, p.cell_id, vec3(&p.origin)));

        // Keep the drawing frame and the streamed landblocks centered on the player.
        if let Some(pos) = &player_pos {
            let coord = LandblockCoord::of_cell(pos.cell_id);
            self.origin =
                match world_frame::follow_player(&mut self.origin, coord, ORIGIN_KEEP_RADIUS) {
                    world_frame::OriginMove::Recentered { shift } => {
                        self.cameras.translate(shift);
                        self.origin
                    }
                    other => {
                        let _ = other;
                        self.origin
                    }
                };
            self.streaming
                .recenter_to(self.origin.center().0, self.origin.center().1);
        }

        // Chase camera: follow the player, else keep orbiting the current center.
        if let Some(pos) = &player_pos {
            let feet = world_frame::to_frame(&self.origin, pos.cell_id, vec3(&pos.origin));
            let subject =
                camera::ChaseSubject::standing(feet, camera::yaw_of(quat(&pos.orientation)));
            self.cameras.chase.update(&subject, dt, None);
            self.cameras.follow_player(camera::PlayerModeFacts {
                live_in_world: scene.in_world(),
                player_present: true,
                controller_ready: true,
                world_ready: scene.is_ready(),
            });
        }

        let (view_mat, eye) = self.cameras.view_and_eye();
        let light = self.light(scene, now, eye);
        let proj = self.lens.projection();

        // Build the resident landblock set, dropping any that left it.
        let resident: Vec<LandblockCoord> = self.streaming.resident().map(|(c, _)| c).collect();
        for coord in &resident {
            self.ensure_landblock(*coord);
        }
        self.landblocks.retain(|c, _| resident.contains(c));

        // Gather every world-space triangle (terrain and objects) before touching
        // the frame: the rasterizer holds `&mut self.frame`, so all the `&mut self`
        // cache work must be done first.
        let mut tris: Vec<Tri> = Vec::new();
        for (coord, lb) in self.landblocks.iter() {
            let offset = world_frame::landblock_offset(&self.origin, *coord);
            for t in lb.vertices.chunks_exact(3) {
                tris.push(Tri {
                    a: offset + v3(t[0].position),
                    b: offset + v3(t[1].position),
                    c: offset + v3(t[2].position),
                    color: [0.30, 0.42, 0.19], // grass; atlas texturing is a later pass
                });
            }
        }

        // The server objects, nearest first (the player is one of them).
        let player_guid = scene.player_guid();
        let mut draws: Vec<(f32, &ac_runtime::world::ObjectView, bool)> = scene
            .objects()
            .filter_map(|o| match &o.position {
                Some(pos) => {
                    let world = world_frame::to_frame(&self.origin, pos.cell_id, vec3(&pos.origin));
                    Some((world.distance(eye), o, player_guid == Some(o.guid)))
                }
                None => None,
            })
            .collect();
        draws.sort_by(|x, y| x.0.total_cmp(&y.0));
        draws.truncate(MAX_DRAWN_OBJECTS);
        for (_dist, obj, is_player) in draws {
            let Some(pos) = &obj.position else {
                continue;
            };
            let albedo: Color = if is_player {
                [0.92, 0.32, 0.28]
            } else {
                [0.72, 0.66, 0.5]
            };
            self.draw_object(pos, obj, albedo, &mut tris);
        }

        {
            let mut raster = Raster::new(&mut self.frame, view_mat, proj, &light);
            raster.clear_sky();
            for t in &tris {
                raster.draw_tri(t.a, t.b, t.c, t.color);
            }
        }
        // The 2-D GUI, then the top-down minimap, painted over the world.
        let ui_list = self.ui.as_mut().map(|ui| ui.frame(now, self.ui_size));
        if let Some(list) = ui_list {
            self.draw_ui_list(&list);
        }
        self.draw_minimap(player_world);
        encode(&self.frame, cols, rows)
    }

    /// Append an object's model triangles, transformed into the drawing frame.
    fn draw_object(
        &mut self,
        pos: &ac_proto::Position,
        obj: &ac_runtime::world::ObjectView,
        albedo: Color,
        out: &mut Vec<Tri>,
    ) {
        let Some(setup_id) = obj.setup_id else {
            return;
        };
        let Some(parts) = assets::model_parts(&self.content, setup_id) else {
            return;
        };
        let t = world_frame::to_frame(&self.origin, pos.cell_id, vec3(&pos.origin));
        let r = quat(&pos.orientation);
        let s = obj.scale.unwrap_or(1.0);
        let base = Mat4::from_scale_rotation_translation(Vec3::splat(s), r, t);
        for part in &parts {
            let Some(gfx) = self.gfx_entry(part.gfx_obj_id) else {
                continue;
            };
            let m = base * part.transform;
            for sm in &gfx.sub_meshes {
                let first = sm.first_index as usize;
                let idx = &gfx.indices[first..first + sm.index_count as usize];
                for tri in idx.chunks_exact(3) {
                    let bv = sm.base_vertex as usize;
                    let w = |i: u32| {
                        let v = &gfx.vertices[i as usize + bv];
                        m * Vec4::new(v.position[0], v.position[1], v.position[2], 1.0)
                    };
                    out.push(Tri {
                        a: vec3v(&w(tri[0])),
                        b: vec3v(&w(tri[1])),
                        c: vec3v(&w(tri[2])),
                        color: albedo,
                    });
                }
            }
        }
    }

    /// Feed the game's 2-D GUI with the scene (vitals, chat, player facts).
    fn feed_ui(&mut self, scene: &mut Scene) {
        if let Some(ui) = self.ui.as_mut() {
            ui.feed(scene);
        }
    }

    /// The session commands the GUI requested since the last call (a control
    /// clicked with the mouse), routed back to the session.
    fn drain_ui(&mut self) -> Vec<SessionCommand> {
        self.ui.as_mut().map_or_else(Vec::new, WindowUi::drain)
    }

    /// Rasterize the GUI's [`DrawList`] into the frame's sub-pixel buffer, over
    /// the 3-D world.
    fn draw_ui_list(&mut self, list: &DrawList) {
        let width = self.frame.width;
        let height = self.frame.height;
        let (cx, cy) = list.canvas_scale;
        let sx = (width as f32 / self.ui_size.0 as f32) * cx;
        let sy = (height as f32 / self.ui_size.1 as f32) * cy;
        let Some(ui) = self.ui.as_ref() else {
            return;
        };
        let mut surfaces = ui.surfaces();
        // Decode every texture the list uses, once per frame.
        let mut tex: HashMap<TextureRef, UiPixels> = HashMap::new();
        for cmd in list.commands() {
            for t in cmd_textures(cmd).into_iter().flatten() {
                if matches!(t, TextureRef::Surface(0)) || tex.contains_key(&t) {
                    continue;
                }
                if let Some(pixels) = surfaces.decode(t) {
                    tex.insert(t, pixels);
                }
            }
        }
        let color = &mut self.frame.color;
        for cmd in list.commands() {
            rasterize_cmd(&tex, cmd, color, width, height, sx, sy);
        }
    }

    /// A top-down, height-shaded map of the terrain around the player, in the
    /// top-right corner of the frame, with the player marked at the centre.
    fn draw_minimap(&mut self, player: Option<Vec3>) {
        let width = self.frame.width;
        let height = self.frame.height;
        let player = match player {
            Some(p) => p,
            None => return,
        };
        let side = MINIMAP_SIDE.min(width / 3).min(height / 3);
        if side < 4 {
            return;
        }
        let x0 = width - side - 2;
        let y0 = 2;
        // A dark backing so the map reads as a panel.
        fill_rect(
            &mut self.frame.color,
            width,
            height,
            x0 as f32,
            y0 as f32,
            (x0 + side) as f32,
            (y0 + side) as f32,
            [0.03, 0.05, 0.03],
            1.0,
        );
        let span = MINIMAP_SPAN;
        let scale = span / side as f32;
        let clip = (x0, y0, x0 + side, y0 + side);
        let color = &mut self.frame.color;
        for (coord, lb) in self.landblocks.iter() {
            let offset = world_frame::landblock_offset(&self.origin, *coord);
            for t in lb.vertices.chunks_exact(3) {
                let a = offset + v3(t[0].position);
                let b = offset + v3(t[1].position);
                let c = offset + v3(t[2].position);
                // Skip triangles wholly outside the map's world span.
                if a.x.max(b.x).max(c.x) < player.x - span / 2.0
                    || a.x.min(b.x).min(c.x) > player.x + span / 2.0
                    || a.y.max(b.y).max(c.y) < player.y - span / 2.0
                    || a.y.min(b.y).min(c.y) > player.y + span / 2.0
                {
                    continue;
                }
                // Height-shade relative to the player so local relief reads:
                // 25 m below the player is darkest, 25 m above is lightest.
                let t = (((a.z + b.z + c.z) / 3.0 - player.z + 25.0) / 50.0).clamp(0.0, 1.0);
                let shade = [0.10 + 0.55 * t, 0.16 + 0.5 * t, 0.08 + 0.2 * t];
                let to_map = |p: Vec3| {
                    (
                        x0 as f32 + (p.x - player.x + span / 2.0) / scale,
                        y0 as f32 + (p.y - player.y + span / 2.0) / scale,
                    )
                };
                let (ax, ay) = to_map(a);
                let (bx, by) = to_map(b);
                let (cxi, cyi) = to_map(c);
                fill_triangle(color, width, ax, ay, bx, by, cxi, cyi, shade, 1.0, clip);
            }
        }
        // The player: a bright marker at the map's centre.
        let mx = x0 + side / 2;
        let my = y0 + side / 2;
        blend_subpix(color, width, mx, my, [1.0, 0.85, 0.2], 1.0);
        blend_subpix(color, width, mx + 1, my, [1.0, 0.85, 0.2], 1.0);
        blend_subpix(color, width, mx, my + 1, [1.0, 0.85, 0.2], 1.0);
    }

    fn gfx_entry(&mut self, id: u32) -> Option<&GfxObjData> {
        if !self.gfx.contains_key(&id) {
            if let Some(g) = assets::build_gfx_obj(&self.content, id) {
                self.gfx.insert(id, g);
            } else {
                self.gfx.insert(
                    id,
                    GfxObjData {
                        id,
                        vertices: Vec::new(),
                        indices: Vec::new(),
                        sub_meshes: Vec::new(),
                        degrades: Vec::new(),
                        sort_center: Vec3::ZERO,
                        drawing_sphere: None,
                    },
                );
            }
        }
        self.gfx.get(&id)
    }

    /// The light and fog for this frame, from the environment (or defaults).
    fn light(&mut self, scene: &Scene, now: Instant, eye: Vec3) -> Light {
        match &mut self.env {
            Some(env) => {
                let ef = env.frame(scene.synced_ticks(), scene.game_ticks(now), now, false);
                let kf = &ef.keyframe;
                let sun_dir = ac_core::sky::sun_direction(kf);
                let sun = kf.sun_color();
                let amb = kf.ambient_color();
                Light {
                    eye,
                    sun_dir,
                    sun_color: [sun.x, sun.y, sun.z],
                    ambient: [amb.x, amb.y, amb.z],
                    fog_color: [kf.fog_color.x, kf.fog_color.y, kf.fog_color.z],
                    fog_start: ef.atmosphere.fog_start.max(1.0),
                    fog_end: ef.atmosphere.fog_end.max(2.0),
                }
            }
            None => Light {
                eye,
                sun_dir: Vec3::new(0.3, 0.2, 0.9).normalize(),
                sun_color: [1.0, 0.95, 0.85],
                ambient: [0.35, 0.38, 0.42],
                fog_color: [0.6, 0.68, 0.78],
                fog_start: 80.0,
                fog_end: 400.0,
            },
        }
    }

    // ------------------------------------------------------------------
    // Gameplay input: the keys and mouse go to the retail input dispatcher;
    // what fires is routed to the session, and the held movement and the
    // held camera keys are polled each frame. The cameras and the dispatcher
    // are disjoint fields, so they are borrowed separately.
    // ------------------------------------------------------------------

    fn press_key(
        &mut self,
        key: Key,
        pressed: bool,
        now: f32,
        state: &DeviceState<'_>,
    ) -> Vec<GameplayOutput> {
        let mut out = Vec::new();
        let cameras = &mut self.cameras;
        self.gameplay
            .key(key, pressed, state, Some(cameras), now, &mut out);
        out
    }

    fn mouse_button(
        &mut self,
        button: u8,
        pressed: bool,
        now_ms: u64,
        now: f32,
        state: &DeviceState<'_>,
    ) -> Vec<GameplayOutput> {
        let mut out = Vec::new();
        let cameras = &mut self.cameras;
        self.gameplay
            .button(button, pressed, now_ms, state, Some(cameras), now, &mut out);
        out
    }

    fn mouse_wheel(
        &mut self,
        delta: f32,
        now: f32,
        state: &DeviceState<'_>,
    ) -> Vec<GameplayOutput> {
        let mut out = Vec::new();
        let cameras = &mut self.cameras;
        self.gameplay
            .wheel(delta, state, Some(cameras), now, &mut out);
        out
    }

    fn mouse_moved(&mut self, dx: f32, dy: f32, now: f32, state: &DeviceState<'_>) {
        let cameras = &mut self.cameras;
        self.gameplay
            .pointer_moved(dx, dy, state, Some(cameras), now);
    }

    /// One frame of held movement and held camera keys; returns what fired.
    fn tick_gameplay(&mut self, dt: f32, now: f32, state: &DeviceState<'_>) -> Vec<GameplayOutput> {
        let mut out = Vec::new();
        let cameras = &mut self.cameras;
        let _fly = self.gameplay.tick(dt, now, state, Some(cameras), &mut out);
        out
    }
}

// ---------------------------------------------------------------------------
// 2-D GUI and minimap rasterization, painted over the sub-pixel buffer.
// ---------------------------------------------------------------------------

/// The texture named by a draw command, if any.
fn cmd_textures(cmd: &DrawCmd) -> [Option<TextureRef>; 1] {
    match cmd {
        DrawCmd::Fill { .. } => [None],
        DrawCmd::Sprite { texture, .. } => [Some(*texture)],
        DrawCmd::Glyphs { texture, .. } => [Some(*texture)],
        DrawCmd::Polygon { texture, .. } => [*texture],
    }
}

/// Rasterize one GUI draw command into the sub-pixel buffer.
fn rasterize_cmd(
    tex: &HashMap<TextureRef, UiPixels>,
    cmd: &DrawCmd,
    buf: &mut [Color],
    width: usize,
    height: usize,
    sx: f32,
    sy: f32,
) {
    match cmd {
        DrawCmd::Fill { x, y, w, h, color } => {
            fill_rect(
                buf,
                width,
                height,
                x * sx,
                y * sy,
                (x + w) * sx,
                (y + h) * sy,
                [color.r, color.g, color.b],
                color.a,
            );
        }
        DrawCmd::Sprite {
            texture,
            quad,
            tint,
            ..
        } => {
            if let Some(pixels) = tex.get(texture) {
                blit_quad(buf, width, height, pixels, quad, tint, sx, sy);
            }
        }
        DrawCmd::Glyphs {
            texture,
            tint,
            quads,
        } => {
            if let Some(pixels) = tex.get(texture) {
                for quad in quads {
                    blit_quad(buf, width, height, pixels, quad, tint, sx, sy);
                }
            }
        }
        DrawCmd::Polygon {
            vertices, color, ..
        } => {
            // A fan from the first corner. Textured polygons are approximated
            // as flat (they are rare in the in-game HUD).
            if vertices.len() >= 3 {
                let (x0, y0) = (vertices[0].x * sx, vertices[0].y * sy);
                for i in 1..vertices.len() - 1 {
                    let (ax, ay) = (vertices[i].x * sx, vertices[i].y * sy);
                    let (bx, by) = (vertices[i + 1].x * sx, vertices[i + 1].y * sy);
                    fill_triangle(
                        buf,
                        width,
                        x0,
                        y0,
                        ax,
                        ay,
                        bx,
                        by,
                        [color.r, color.g, color.b],
                        color.a,
                        (0, 0, width, height),
                    );
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
/// Blit one textured quad (a sprite or a glyph), tinted, over the buffer.
fn blit_quad(
    buf: &mut [Color],
    width: usize,
    height: usize,
    pixels: &UiPixels,
    quad: &Quad,
    tint: &UiColor,
    sx: f32,
    sy: f32,
) {
    let px0 = quad.x * sx;
    let py0 = quad.y * sy;
    let px1 = (quad.x + quad.w) * sx;
    let py1 = (quad.y + quad.h) * sy;
    let (lo_x, hi_x) = (px0.floor() as i32, px1.ceil() as i32);
    let (lo_y, hi_y) = (py0.floor() as i32, py1.ceil() as i32);
    let span_x = (px1 - px0).max(1e-3);
    let span_y = (py1 - py0).max(1e-3);
    for py in lo_y.max(0)..hi_y.min(height as i32) {
        for px in lo_x.max(0)..hi_x.min(width as i32) {
            let u = quad.u0 + ((px as f32 + 0.5) - px0) / span_x * (quad.u1 - quad.u0);
            let v = quad.v0 + ((py as f32 + 0.5) - py0) / span_y * (quad.v1 - quad.v0);
            let [r, g, b, a] = sample_texel(pixels, u, v);
            blend_subpix(
                buf,
                width,
                px as usize,
                py as usize,
                [r * tint.r, g * tint.g, b * tint.b],
                a * tint.a,
            );
        }
    }
}

/// Sample a decoded texture at (u, v) in 0..1 (tiled), returning straight RGBA
/// in 0..1.
fn sample_texel(pixels: &UiPixels, u: f32, v: f32) -> [f32; 4] {
    let u = (u - u.floor()).clamp(0.0, 1.0 - 1e-4);
    let v = (v - v.floor()).clamp(0.0, 1.0 - 1e-4);
    match pixels {
        UiPixels::Rgba(t) => rgba_at(t.width, t.height, &t.rgba, u, v),
        UiPixels::Plugin(levels) => match levels.first() {
            Some(t) => rgba_at(t.width, t.height, &t.rgba, u, v),
            None => [0.0, 0.0, 0.0, 0.0],
        },
        UiPixels::Coverage {
            width,
            height,
            alpha,
        } => {
            let x = (u * *width as f32) as usize;
            let y = (v * *height as f32) as usize;
            [1.0, 1.0, 1.0, alpha[y * *width as usize + x] as f32 / 255.0]
        }
    }
}

/// Nearest-sampled RGBA texel of a decoded image at (u, v) in 0..1.
fn rgba_at(width: u32, height: u32, rgba: &[u8], u: f32, v: f32) -> [f32; 4] {
    let x = (u * width as f32) as usize;
    let y = (v * height as f32) as usize;
    let i = (y * width as usize + x) * 4;
    [
        rgba[i] as f32 / 255.0,
        rgba[i + 1] as f32 / 255.0,
        rgba[i + 2] as f32 / 255.0,
        rgba[i + 3] as f32 / 255.0,
    ]
}

#[allow(clippy::too_many_arguments)]
/// Blend an opaque-or-translucent rectangle (in sub-pixels) over the buffer.
fn fill_rect(
    buf: &mut [Color],
    width: usize,
    height: usize,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    color: Color,
    alpha: f32,
) {
    for py in (y0.floor() as i32).max(0)..(y1.ceil() as i32).min(height as i32) {
        for px in (x0.floor() as i32).max(0)..(x1.ceil() as i32).min(width as i32) {
            blend_subpix(buf, width, px as usize, py as usize, color, alpha);
        }
    }
}

#[allow(clippy::too_many_arguments)]
/// Scanline-fill a triangle (in sub-pixels) over the buffer, clipped to `clip`
/// (an exclusive `x1`, `y1`).
fn fill_triangle(
    buf: &mut [Color],
    width: usize,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    color: Color,
    alpha: f32,
    clip: (usize, usize, usize, usize),
) {
    let (cx0, cy0, cx1, cy1) = clip;
    let min_y = y0.min(y1).min(y2);
    let max_y = y0.max(y1).max(y2);
    for row in (min_y.floor() as i32).max(cy0 as i32)..(max_y.ceil() as i32).min(cy1 as i32) {
        let y = row as f32 + 0.5;
        let mut xs = [0.0f32; 3];
        let mut n = 0;
        for (ax, ay, bx, by) in [(x0, y0, x1, y1), (x1, y1, x2, y2), (x2, y2, x0, y0)] {
            let dy = by - ay;
            if dy.abs() < 1e-9 {
                continue;
            }
            let t = (y - ay) / dy;
            if !(0.0..=1.0).contains(&t) {
                continue;
            }
            xs[n] = ax + t * (bx - ax);
            n += 1;
        }
        if n < 2 {
            continue;
        }
        xs[0..n].sort_by(|a, b| a.total_cmp(b));
        let lo = xs[0].max(cx0 as f32);
        let hi = xs[n - 1].min(cx1 as f32);
        for px in lo.ceil() as i32..hi.floor() as i32 + 1 {
            blend_subpix(buf, width, px as usize, row as usize, color, alpha);
        }
    }
}

/// Blend one sub-pixel (skipping out-of-bounds and fully transparent).
fn blend_subpix(buf: &mut [Color], width: usize, x: usize, y: usize, color: Color, alpha: f32) {
    if x >= width || y >= buf.len() / width {
        return;
    }
    let a = alpha.clamp(0.0, 1.0);
    if a <= 0.0 {
        return;
    }
    let i = y * width + x;
    let inv = 1.0 - a;
    buf[i] = [
        color[0] * a + buf[i][0] * inv,
        color[1] * a + buf[i][1] * inv,
        color[2] * a + buf[i][2] * inv,
    ];
}

// Small glam conversions from the wire geometry types.
fn vec3(v: &ac_proto::Vector3) -> Vec3 {
    Vec3::new(v.x, v.y, v.z)
}

fn v3(a: [f32; 3]) -> Vec3 {
    Vec3::new(a[0], a[1], a[2])
}

fn quat(q: &ac_proto::Quaternion) -> Quat {
    Quat::from_xyzw(q.x, q.y, q.z, q.w)
}

fn vec3v(v: &Vec4) -> Vec3 {
    Vec3::new(v.x, v.y, v.z)
}

// ---------------------------------------------------------------------------
// The front end: terminal, attach, session monitor and the render loop.
// ---------------------------------------------------------------------------

/// Runs the `--ui terminal` session to its end. The 3-D world and 2-D GUI are
/// rendered on the calling (main) thread; a monitor thread drives the session
/// and reports its exit code.
pub fn run(handle: SessionHandle, options: Options) -> ExitCode {
    if enter_terminal().is_err() {
        errln!("openac: could not set the terminal into raw mode");
        return ExitCode::from(super::EXIT_SESSION_FAILED);
    }
    let code = run_render_loop(handle, options);
    leave_terminal();
    code
}

fn run_render_loop(handle: SessionHandle, options: Options) -> ExitCode {
    let commands = handle.commands();

    let (to_window, from_session) = mpsc::channel::<PresentationEvent>();
    let _ = commands.send(SessionCommand::PresentsWorld(true));
    let _ = commands.send(SessionCommand::Attach(to_window));
    let mut source = SceneSource::new(from_session).with_commands(commands.clone());

    let (tx_code, rx_code) = mpsc::channel::<ExitCode>();
    let monitor_opts = options.clone();
    let tx_monitor = tx_code.clone();
    let monitor = std::thread::Builder::new()
        .name("session-monitor".into())
        .spawn(move || {
            let logger = RenderLog::new(monitor_opts.log_level);
            let code = super::run_session_loop(handle, &logger, &monitor_opts);
            let _ = tx_monitor.send(code);
        });

    let size = crossterm::terminal::size().unwrap_or((80, 24));
    let (cols, rows) = (size.0 as usize, size.1 as usize);
    let mut renderer = match Renderer::new(
        options.dat_dir.clone(),
        options.landblock_radius,
        cols,
        rows,
    ) {
        Ok(r) => r,
        Err(message) => {
            errln!("openac: {message}");
            let _ = tx_code.send(ExitCode::from(super::EXIT_SESSION_FAILED));
            drop(monitor);
            return ExitCode::from(super::EXIT_SESSION_FAILED);
        }
    };

    let mut last = Instant::now();
    let input_start = Instant::now();
    let mut stdout = io::stdout();
    let mut modifiers = Modifiers::NONE;
    let mut buttons: u32 = 0;
    let mut last_mouse: Option<(u16, u16)> = None;
    let mut quit = false;
    let mut session_code: Option<ExitCode> = None;
    loop {
        let now = Instant::now();
        let dt = now.saturating_duration_since(last).as_secs_f32().min(0.1);
        last = now;
        let gameplay_now = input_start.elapsed().as_secs_f32();
        let now_ms = u64::try_from(input_start.elapsed().as_millis()).unwrap_or(u64::MAX);

        // Drain pending input (non-blocking); Esc quits.
        while !quit && event::poll(Duration::ZERO).unwrap_or(false) {
            let Ok(evt) = event::read() else {
                break;
            };
            match evt {
                Event::Key(k) => {
                    if k.code == KeyCode::Esc {
                        quit = true;
                        break;
                    }
                    modifiers = modifiers_from_crossterm(k.modifiers);
                    let Some(key) = key_from_crossterm(k.code) else {
                        continue;
                    };
                    let pressed = match k.kind {
                        KeyEventKind::Press => true,
                        KeyEventKind::Release => false,
                        _ => continue,
                    };
                    let state = input_state(buttons, modifiers);
                    let outs = renderer.press_key(key, pressed, gameplay_now, &state);
                    route_outputs(outs, &commands);
                }
                Event::Mouse(m) => {
                    let state = input_state(buttons, modifiers);
                    match m.kind {
                        MouseEventKind::Down(b) => {
                            let i = mouse_button_index(b);
                            buttons |= 1u32 << i;
                            last_mouse = Some((m.column, m.row));
                            let outs = renderer.mouse_button(i, true, now_ms, gameplay_now, &state);
                            route_outputs(outs, &commands);
                        }
                        MouseEventKind::Up(b) => {
                            let i = mouse_button_index(b);
                            buttons &= !(1u32 << i);
                            let outs =
                                renderer.mouse_button(i, false, now_ms, gameplay_now, &state);
                            route_outputs(outs, &commands);
                        }
                        MouseEventKind::Moved | MouseEventKind::Drag(_) => {
                            let (col, row) = (m.column, m.row);
                            if let Some((px, py)) = last_mouse {
                                let dx = col as i32 - px as i32;
                                let dy = row as i32 - py as i32;
                                if dx != 0 || dy != 0 {
                                    // A cell is 2x4 sub-pixels; feed the motion in sub-pixels.
                                    renderer.mouse_moved(
                                        dx as f32 * 2.0,
                                        dy as f32 * 4.0,
                                        gameplay_now,
                                        &state,
                                    );
                                }
                            }
                            last_mouse = Some((col, row));
                        }
                        MouseEventKind::ScrollDown => {
                            let outs = renderer.mouse_wheel(1.0, gameplay_now, &state);
                            route_outputs(outs, &commands);
                        }
                        MouseEventKind::ScrollUp => {
                            let outs = renderer.mouse_wheel(-1.0, gameplay_now, &state);
                            route_outputs(outs, &commands);
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }

        // Held movement and held camera keys, polled each frame.
        let state = input_state(buttons, modifiers);
        let outs = renderer.tick_gameplay(dt, gameplay_now, &state);
        route_outputs(outs, &commands);

        source.pump();
        let scene = source.scene_mut();
        renderer.feed_ui(scene);
        let grid = renderer.render_frame(scene, dt, now, cols, rows);
        for c in renderer.drain_ui() {
            let _ = commands.send(c);
        }
        if let Err(e) = draw_grid(&mut stdout, &grid) {
            errln!("openac: terminal write failed: {e}");
            break;
        }

        if quit {
            break;
        }
        if let Ok(code) = rx_code.try_recv() {
            session_code = Some(code);
            break;
        }
        if source.session_gone() {
            break;
        }
        std::thread::sleep(FRAME_PERIOD.saturating_sub(now.elapsed()));
    }

    let code = match session_code {
        Some(code) => code,
        None => {
            // The session is still running (Esc, or the terminal broke):
            // ask it to end, then wait for its code.
            let _ = commands.send(SessionCommand::Shutdown);
            rx_code
                .recv_timeout(Duration::from_secs(3))
                .unwrap_or_else(|_| ExitCode::from(super::EXIT_SESSION_FAILED))
        }
    };
    let _ = monitor;
    code
}

// ---------------------------------------------------------------------------
// Input: crossterm events -> the UI's keys and the session's commands.
// ---------------------------------------------------------------------------

/// The gameplay input's device state for the terminal (the world has the
/// keyboard and the cursor; the dispatcher keeps its own held-key set).
fn input_state(buttons: u32, modifiers: Modifiers) -> DeviceState<'static> {
    DeviceState {
        keys: None,
        buttons,
        modifiers,
        want_mouse: false,
        want_keyboard: false,
    }
}

/// A crossterm key as the UI names it (`None` for keys the UI has no name for).
fn key_from_crossterm(code: KeyCode) -> Option<Key> {
    use Key as K;
    use KeyCode as C;
    Some(match code {
        C::Char('a' | 'A') => K::A,
        C::Char('b' | 'B') => K::B,
        C::Char('c' | 'C') => K::C,
        C::Char('d' | 'D') => K::D,
        C::Char('e' | 'E') => K::E,
        C::Char('f' | 'F') => K::F,
        C::Char('g' | 'G') => K::G,
        C::Char('h' | 'H') => K::H,
        C::Char('i' | 'I') => K::I,
        C::Char('j' | 'J') => K::J,
        C::Char('k' | 'K') => K::K,
        C::Char('l' | 'L') => K::L,
        C::Char('m' | 'M') => K::M,
        C::Char('n' | 'N') => K::N,
        C::Char('o' | 'O') => K::O,
        C::Char('p' | 'P') => K::P,
        C::Char('q' | 'Q') => K::Q,
        C::Char('r' | 'R') => K::R,
        C::Char('s' | 'S') => K::S,
        C::Char('t' | 'T') => K::T,
        C::Char('u' | 'U') => K::U,
        C::Char('v' | 'V') => K::V,
        C::Char('w' | 'W') => K::W,
        C::Char('x' | 'X') => K::X,
        C::Char('y' | 'Y') => K::Y,
        C::Char('z' | 'Z') => K::Z,
        C::Char('0') => K::Number0,
        C::Char('1') => K::Number1,
        C::Char('2') => K::Number2,
        C::Char('3') => K::Number3,
        C::Char('4') => K::Number4,
        C::Char('5') => K::Number5,
        C::Char('6') => K::Number6,
        C::Char('7') => K::Number7,
        C::Char('8') => K::Number8,
        C::Char('9') => K::Number9,
        C::Char(' ') => K::Space,
        C::Backspace => K::Backspace,
        C::Enter => K::Enter,
        C::Left => K::Left,
        C::Right => K::Right,
        C::Up => K::Up,
        C::Down => K::Down,
        C::Home => K::Home,
        C::End => K::End,
        C::PageUp => K::PageUp,
        C::PageDown => K::PageDown,
        C::Tab => K::Tab,
        C::Delete => K::Delete,
        C::Insert => K::Insert,
        C::F(n) => match n {
            1 => K::F1,
            2 => K::F2,
            3 => K::F3,
            4 => K::F4,
            5 => K::F5,
            6 => K::F6,
            7 => K::F7,
            8 => K::F8,
            9 => K::F9,
            10 => K::F10,
            11 => K::F11,
            12 => K::F12,
            _ => return None,
        },
        _ => return None,
    })
}

fn modifiers_from_crossterm(m: KeyModifiers) -> Modifiers {
    let mut r = Modifiers::NONE;
    if m.contains(KeyModifiers::SHIFT) {
        r = r | Modifiers::SHIFT;
    }
    if m.contains(KeyModifiers::CONTROL) {
        r = r | Modifiers::CTRL;
    }
    if m.contains(KeyModifiers::ALT) {
        r = r | Modifiers::ALT;
    }
    if m.contains(KeyModifiers::SUPER) {
        r = r | Modifiers::WIN;
    }
    r
}

fn mouse_button_index(button: crossterm::event::MouseButton) -> u8 {
    match button {
        crossterm::event::MouseButton::Left => 0,
        crossterm::event::MouseButton::Right => 1,
        crossterm::event::MouseButton::Middle => 2,
    }
}

/// Route what the gameplay input fired to the session (notices to stderr).
fn route_outputs(outputs: Vec<GameplayOutput>, commands: &mpsc::Sender<SessionCommand>) {
    for output in outputs {
        match output {
            GameplayOutput::Movement(command) => {
                let _ = commands.send(SessionCommand::Movement(command));
            }
            GameplayOutput::Gameplay(command) => {
                let _ = commands.send(SessionCommand::Gameplay(command));
            }
            GameplayOutput::Item(command) => {
                let _ = commands.send(SessionCommand::Item(command));
            }
            GameplayOutput::LogOut => {
                let _ = commands.send(SessionCommand::Lifecycle(
                    LifecycleCommand::EndCharacterSession,
                ));
            }
            GameplayOutput::Notice(text) => {
                errln!("openac: {text}");
            }
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Terminal control.
// ---------------------------------------------------------------------------

fn enter_terminal() -> io::Result<()> {
    crossterm::terminal::enable_raw_mode()?;
    let mut out = io::stdout();
    use crossterm::queue;
    queue!(
        out,
        crossterm::terminal::EnterAlternateScreen,
        crossterm::cursor::Hide,
        EnableMouseCapture
    )?;
    out.flush()?;
    Ok(())
}

fn leave_terminal() {
    let mut out = io::stdout();
    let _ = crossterm::execute!(
        out,
        crossterm::cursor::Show,
        DisableMouseCapture,
        crossterm::terminal::LeaveAlternateScreen
    );
    let _ = crossterm::terminal::disable_raw_mode();
}

/// Write a grid of cells to the terminal with 24-bit colour, emitting a SGR
/// sequence only when a colour changes.
fn draw_grid(out: &mut impl Write, grid: &[Vec<Cell>]) -> io::Result<()> {
    use std::fmt::Write as _;
    let mut buf = String::with_capacity(grid.len() * grid[0].len() * 8);
    let _ = write!(buf, "\x1b[H");
    let mut last_fg: Option<[u8; 3]> = None;
    let mut last_bg: Option<[u8; 3]> = None;
    for (r, row) in grid.iter().enumerate() {
        for cell in row {
            if last_fg != Some(cell.fg) {
                let _ = write!(
                    buf,
                    "\x1b[38;2;{};{};{}m",
                    cell.fg[0], cell.fg[1], cell.fg[2]
                );
                last_fg = Some(cell.fg);
            }
            if last_bg != Some(cell.bg) {
                let _ = write!(
                    buf,
                    "\x1b[48;2;{};{};{}m",
                    cell.bg[0], cell.bg[1], cell.bg[2]
                );
                last_bg = Some(cell.bg);
            }
            let _ = write!(buf, "{}", cell.ch);
        }
        if r + 1 < grid.len() {
            let _ = write!(buf, "\x1b[0m\r\n");
        }
    }
    let _ = write!(buf, "\x1b[0m");
    out.write_all(buf.as_bytes())?;
    out.flush()
}

// ---------------------------------------------------------------------------
// Tests: the rasterizer and braille encoding, content-free.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn light() -> Light {
        Light {
            eye: Vec3::new(0.0, -10.0, 6.0),
            sun_dir: Vec3::new(0.3, 0.2, 0.9).normalize(),
            sun_color: [1.0, 0.95, 0.85],
            ambient: [0.3, 0.32, 0.35],
            fog_color: [0.6, 0.68, 0.78],
            fog_start: 80.0,
            fog_end: 500.0,
        }
    }

    #[test]
    fn a_floor_and_wall_produce_lit_braille_cells() {
        let (cols, rows) = (40, 20);
        let mut frame = Frame::new(2 * cols, 4 * rows);
        let eye = Vec3::new(0.0, -8.0, 5.0);
        let view = glam::camera::rh::view::look_at_mat4(eye, Vec3::new(0.0, 0.0, 2.0), Vec3::Z);
        let mut lens = Lens::default();
        lens.set_aspect((2 * cols) as f32 / (4 * rows) as f32);
        let proj = lens.projection();
        let light = light();
        let mut r = Raster::new(&mut frame, view, proj, &light);
        r.clear_sky();
        // A flat ground quad at z = 0, split into two triangles.
        let quad = [
            Vec3::new(-20.0, -20.0, 0.0),
            Vec3::new(20.0, -20.0, 0.0),
            Vec3::new(20.0, 20.0, 0.0),
            Vec3::new(-20.0, 20.0, 0.0),
        ];
        for (a, b, c) in [(0, 1, 2), (0, 2, 3)] {
            r.draw_tri(quad[a], quad[b], quad[c], [0.3, 0.45, 0.2]);
        }
        let grid = encode(&frame, cols, rows);
        // The lower part of the screen (ground) should have some lit dots; the
        // upper part (sky) should be a solid colour block.
        let lit: u32 = grid
            .iter()
            .flat_map(|row| row.iter())
            .map(|c| (c.ch as u32) & 0xFF)
            .sum();
        assert!(lit > 0, "expected at least one lit sub-pixel, got none");
        // The sky (top row) should be a solid block (all dots lit).
        assert_eq!(
            grid[0][0].ch,
            char::from_u32(0x2800 | 0xFF).unwrap(),
            "top cell should be solid sky"
        );
    }

    #[test]
    fn flat_region_encodes_as_a_solid_block() {
        let mut frame = Frame::new(2, 4);
        for c in frame.color.iter_mut() {
            *c = [0.5, 0.2, 0.1];
        }
        let grid = encode(&frame, 1, 1);
        assert_eq!(grid[0][0].ch, char::from_u32(0x2800 | 0xFF).unwrap());
        assert_eq!(grid[0][0].fg, grid[0][0].bg);
    }

    /// With real dats (`OPENAC_DAT_DIR`), a Holtburg player must produce visible
    /// terrain relief in the rendered frame; a flat sky alone has none. Skipped
    /// when no dat directory is set (the content-free tests still run).
    #[test]
    fn holtburg_scene_renders_terrain() {
        let Some(dir) = std::env::var_os("OPENAC_DAT_DIR") else {
            return;
        };
        use ac_proto::{ModelData, Position, Quaternion, Vector3};
        use ac_runtime::world::{EffectFacts, ObjectKey, ObjectView, RadarFacts, WorldSnapshot};
        let player = ObjectView {
            key: ObjectKey {
                local_id: 1,
                incarnation: 1,
            },
            guid: 0x5000_0001,
            name: None,
            setup_id: Some(0x0200_0001),
            position: Some(Position {
                cell_id: 0xA9B4_0019,
                origin: Vector3 {
                    x: 84.0,
                    y: 7.1,
                    z: 94.005,
                },
                orientation: Quaternion {
                    w: 1.0,
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                },
            }),
            parent: None,
            placement: None,
            rest_placement: 0,
            scale: None,
            translucency: None,
            state: 0,
            velocity: None,
            model: ModelData::default(),
            motion_table_id: None,
            sound_table_id: None,
            radar: RadarFacts::default(),
            motion: None,
            effects: EffectFacts::default(),
            step_blend: None,
        };
        let (cols, rows) = (80, 24);
        let mut renderer =
            Renderer::new(Some(PathBuf::from(dir)), Some(2), cols, rows).expect("open the dats");
        let (tx, rx) = mpsc::channel();
        let mut source = SceneSource::new(rx);
        tx.send(PresentationEvent::Snapshot(WorldSnapshot {
            player_guid: Some(0x5000_0001),
            objects: vec![player],
        }))
        .unwrap();
        source.pump();
        let scene = source.scene_mut();
        assert!(
            scene.is_ready(),
            "the scene should be ready after the snapshot"
        );
        assert_eq!(scene.player_guid(), Some(0x5000_0001));
        // A couple of frames so the chase camera settles and the landblocks build.
        let now = Instant::now();
        renderer.feed_ui(scene);
        let grid = renderer.render_frame(scene, 0.033, now, cols, rows);
        let _ = renderer.render_frame(scene, 0.033, now, cols, rows);
        // Pure sky is a flat solid block per cell; terrain produces relief (lit dots
        // on a background). A visible landscape must be a substantial share of cells.
        let total = grid.iter().flatten().count();
        let relief = grid
            .iter()
            .flatten()
            .filter(|c| (c.ch as u32) & 0xFF != 0xFF)
            .count();
        let ratio = relief as f64 / total as f64;
        assert!(
            ratio > 0.05,
            "expected visible terrain relief, but only {relief}/{total} ({ratio:.1}%) of cells were non-solid"
        );
        // The game's GUI is built from the dats and, fed with the scene, emits
        // drawing commands (vitals, name and the other windows).
        let list = renderer
            .ui
            .as_mut()
            .expect("the game GUI should build from the dats")
            .frame(now, renderer.ui_size);
        assert!(
            list.commands().count() > 0,
            "the GUI should emit drawing commands for this scene"
        );
        // The top-down minimap paints the top-right corner, distinct from the sky.
        let sky = grid[0][0].fg;
        let differs = |r: usize, c: usize| {
            let cell = &grid[r][c];
            [cell.fg, cell.bg].iter().any(|col| {
                (i32::abs(col[0] as i32 - sky[0] as i32)) > 40
                    || (i32::abs(col[1] as i32 - sky[1] as i32)) > 40
                    || (i32::abs(col[2] as i32 - sky[2] as i32)) > 40
            })
        };
        let marked = (0..5)
            .flat_map(|r| (cols - 9..cols).map(move |c| (r, c)))
            .filter(|(r, c)| differs(*r, *c))
            .count();
        assert!(
            marked >= 5,
            "the minimap should paint the top-right corner, but only {marked} of 45 cells differ from the sky"
        );
    }

    /// Holding W (the retail forward key) drives a forward `Movement` command
    /// through the exact path the render loop uses: key -> dispatcher -> tick,
    /// and releasing it clears it again.
    #[test]
    fn holding_forward_key_sends_movement() {
        use ac_runtime::movement::MovementCommand;
        let mut gameplay = GameplayInput::new(KeyBindings::retail_defaults());
        let state = input_state(0, Modifiers::NONE);
        let mut cameras = CameraController::new(OrbitCamera::default());
        let mut out = Vec::new();
        gameplay.key(Key::W, true, &state, Some(&mut cameras), 0.0, &mut out);
        let _ = gameplay.tick(0.016, 0.016, &state, Some(&mut cameras), &mut out);
        assert!(
            out.iter().any(|o| {
                matches!(o, GameplayOutput::Movement(MovementCommand::Keys(k)) if k.forward)
            }),
            "holding W should send forward movement, got {out:?}"
        );
        gameplay.key(Key::W, false, &state, Some(&mut cameras), 0.02, &mut out);
        let _ = gameplay.tick(0.016, 0.02, &state, Some(&mut cameras), &mut out);
        assert!(
            out.iter().any(|o| {
                matches!(o, GameplayOutput::Movement(MovementCommand::Keys(k)) if !k.forward)
            }),
            "releasing W should clear the forward movement, got {out:?}"
        );
    }
    /// The 2-D HUD primitives, straight onto a sub-pixel buffer: solid fills,
    /// triangle fills, coverage (font) sampling with tiling, a full textured
    /// sprite blit, and alpha blending.
    #[test]
    fn hud_primitives_rasterize() {
        use ac_ui::draw::Filter;
        let (w, h) = (16, 8);
        let mut buf = vec![[0.0, 0.0, 0.0]; w * h];

        // A solid fill over the whole left half.
        fill_rect(&mut buf, w, h, 0.0, 0.0, 8.0, 8.0, [1.0, 0.0, 0.0], 1.0);
        assert_eq!(buf[0], [1.0, 0.0, 0.0]); // (0,0)
        assert_eq!(buf[7], [1.0, 0.0, 0.0]); // (7,0), last of the left half
        assert_eq!(buf[8], [0.0, 0.0, 0.0]); // (8,0), right half untouched
        assert_eq!(buf[w * 7], [1.0, 0.0, 0.0]); // (0,7), bottom-left

        // A triangle, apex up, base down, over the right half.
        fill_triangle(
            &mut buf,
            w,
            12.0,
            1.0, // apex
            8.0,
            7.0, // bottom-left
            16.0,
            7.0, // bottom-right
            [0.0, 1.0, 0.0],
            1.0,
            (0, 0, w, h),
        );
        assert_eq!(buf[w * 6 + 12], [0.0, 1.0, 0.0]); // (12,6), inside the base
        assert_eq!(buf[12], [0.0, 0.0, 0.0]); // (12,0), above the apex

        // A coverage (font-style) texture, sampled and tiled.
        let coverage = UiPixels::Coverage {
            width: 2,
            height: 2,
            alpha: vec![255, 0, 0, 255],
        };
        assert_eq!(sample_texel(&coverage, 0.25, 0.25), [1.0, 1.0, 1.0, 1.0]);
        assert_eq!(sample_texel(&coverage, 0.75, 0.25), [1.0, 1.0, 1.0, 0.0]);
        assert_eq!(sample_texel(&coverage, 1.25, 1.25), [1.0, 1.0, 1.0, 1.0]); // tiled
        assert_eq!(sample_texel(&coverage, 1.75, 1.25), [1.0, 1.0, 1.0, 0.0]); // tiled

        // A full sprite blit: a 4x4 sub-pixel quad over the coverage texture,
        // tinted green, through the shared command path.
        let mut buf2 = vec![[0.0, 0.0, 0.0]; w * h];
        let tex = {
            let mut m = HashMap::new();
            m.insert(TextureRef::Surface(0x0600_0001), coverage);
            m
        };
        let cmd = DrawCmd::Sprite {
            texture: TextureRef::Surface(0x0600_0001),
            filter: Filter::Nearest,
            quad: Quad {
                x: 0.0,
                y: 0.0,
                w: 4.0,
                h: 4.0,
                u0: 0.0,
                v0: 0.0,
                u1: 1.0,
                v1: 1.0,
            },
            tint: UiColor::rgba(0.0, 1.0, 0.0, 1.0),
        };
        rasterize_cmd(&tex, &cmd, &mut buf2, w, h, 1.0, 1.0);
        assert_eq!(buf2[0], [0.0, 1.0, 0.0]); // samples a fully-coverage texel
        assert_eq!(buf2[3], [0.0, 0.0, 0.0]); // samples a transparent texel
        assert_eq!(buf2[w + 6], [0.0, 0.0, 0.0]); // outside the quad

        // Alpha blending: a half-opacity blue over red gives a mid tone.
        let mut buf3 = vec![[1.0, 0.0, 0.0]; w * h];
        blend_subpix(&mut buf3, w, 0, 0, [0.0, 0.0, 1.0], 0.5);
        assert!((buf3[0][0] - 0.5).abs() < 1e-6);
        assert!((buf3[0][2] - 0.5).abs() < 1e-6);
    }
}

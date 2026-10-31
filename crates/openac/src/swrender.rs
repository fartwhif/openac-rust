//! The `--ui terminal` front end: the 3-D world, software-rendered to a plain
//! terminal as braille dots with 24-bit colour, plus a 2-D interface drawn from
//! the scene's facts in a primitive cell layer (box-drawing, Nerd Font glyphs,
//! UTF-8 text) — never the game's widget tree.
//!
//! There is no GPU and no image protocol on the target, so the frame is a grid
//! of sub-pixels: each terminal cell holds a 2-wide by 4-tall braille dot
//! matrix (square in a typical 1:2 cell) and a foreground/background colour.
//! The world is projected with the game's own chase camera and light with the
//! game's sun, ambient and fog.

use std::collections::HashMap;
use std::io::{self, Write};
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::mpsc;
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
use ac_render::world_frame::{self, LandblockCoord};
use ac_runtime::session::{
    LifecycleCommand, ObjectSummary, PresentationEvent, SessionCommand, SessionEvent,
    SessionHandle, SessionSummary,
};
use ac_ui::input::bindings::KeyBindings;
use ac_ui::{Key, Modifiers};
use crossterm::event::{
    self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind, KeyModifiers,
    MouseEventKind,
};
use glam::{Mat4, Quat, Vec3, Vec4};

use crate::options::{LogLevel, Options};
use crate::stdio::errln;
use crate::termui;

const REGION_ID: u32 = 0x1300_0000;
/// How many landblocks the drawing frame keeps centered on the player.
const ORIGIN_KEEP_RADIUS: u32 = 2;
/// A frame's target period (the session self-paces; this only caps the UI).
const FRAME_PERIOD: Duration = Duration::from_millis(66); // ~15 fps
/// How many server objects are drawn (nearest first) each frame.
const MAX_DRAWN_OBJECTS: usize = 64;
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
/// Shared with the 2-D primitive interface, which paints the HUD and panels.
#[derive(Clone, Copy)]
pub(crate) struct Cell {
    pub(crate) ch: char,
    pub(crate) fg: [u8; 3],
    pub(crate) bg: [u8; 3],
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

    /// A top-down height grid of the terrain around the player: an `n`×`n`
    /// array of heights relative to the player (`f32::MIN` where no terrain
    /// covers the cell), for the 2-D interface's shade-character minimap.
    fn minimap_heights(&self, scene: &Scene, n: usize) -> Option<Vec<Vec<f32>>> {
        let n = n.max(2);
        let player = scene
            .player()
            .and_then(|p| p.position)
            .map(|pos| world_frame::to_frame(&self.origin, pos.cell_id, vec3(&pos.origin)))?;
        let span = MINIMAP_SPAN;
        let scale = span / n as f32;
        const NO_TERRAIN: f32 = f32::MIN;
        let mut buf: Vec<Color> = vec![[NO_TERRAIN, 0.0, 0.0]; n * n];
        let clip = (0, 0, n, n);
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
                let rel = ((a.z + b.z + c.z) / 3.0) - player.z;
                let to_map = |p: Vec3| {
                    (
                        (p.x - player.x + span / 2.0) / scale,
                        (p.y - player.y + span / 2.0) / scale,
                    )
                };
                let (ax, ay) = to_map(a);
                let (bx, by) = to_map(b);
                let (cx, cy) = to_map(c);
                fill_triangle(
                    &mut buf,
                    n,
                    ax,
                    ay,
                    bx,
                    by,
                    cx,
                    cy,
                    [rel, 0.0, 0.0],
                    1.0,
                    clip,
                );
            }
        }
        let mut out = vec![vec![NO_TERRAIN; n]; n];
        for y in 0..n {
            for x in 0..n {
                out[y][x] = buf[y * n + x][0];
            }
        }
        Some(out)
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
// Sub-pixel fill primitives: the minimap's height buffer is painted with them.
// ---------------------------------------------------------------------------

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
    let mut model = termui::Model::new();
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
        model.update(scene, now);
        let layout = termui::layout(cols, rows, model.panel_open(now));
        let view_grid = renderer.render_frame(scene, dt, now, layout.viewport.w, layout.viewport.h);
        let minimap = renderer.minimap_heights(scene, termui::MINIMAP_N);
        let mut grid = termui::Grid::new(cols, rows);
        grid.blit(layout.viewport.x, layout.viewport.y, &view_grid);
        termui::draw(&mut grid, &model, &layout, now, minimap.as_ref());
        if let Err(e) = draw_grid(&mut stdout, grid.rows()) {
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
        // The top-down height grid is populated around the Holtburg player.
        let heights = renderer
            .minimap_heights(scene, termui::MINIMAP_N)
            .expect("a player gives a height grid");
        let painted = heights.iter().flatten().filter(|h| **h > f32::MIN).count();
        assert!(
            painted > termui::MINIMAP_N * termui::MINIMAP_N / 4,
            "the map should cover terrain, but only {painted} cells were painted"
        );
        // The composed terminal frame paints the HUD (the shade-char minimap) over
        // the viewport.
        let mut model = termui::Model::new();
        model.update(scene, now);
        let layout = termui::layout(cols, rows, false);
        let view = renderer.render_frame(scene, 0.033, now, layout.viewport.w, layout.viewport.h);
        let mut full = termui::Grid::new(cols, rows);
        full.blit(layout.viewport.x, layout.viewport.y, &view);
        let mm = renderer.minimap_heights(scene, termui::MINIMAP_N);
        termui::draw(&mut full, &model, &layout, now, mm.as_ref());
        let has_shade = full
            .rows()
            .iter()
            .flatten()
            .any(|c| (c.ch as u32) >= 0x2581 && (c.ch as u32) <= 0x2588);
        assert!(
            has_shade,
            "the HUD should paint the minimap's shade characters"
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
    /// The sub-pixel triangle fill and alpha blending are the building blocks
    /// the minimap's height buffer is painted with.
    #[test]
    fn fill_primitives_rasterize() {
        let (w, h) = (16, 8);
        let mut buf = vec![[0.0, 0.0, 0.0]; w * h];

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

        // Alpha blending: a half-opacity blue over red gives a mid tone.
        let mut buf3 = vec![[1.0, 0.0, 0.0]; w * h];
        blend_subpix(&mut buf3, w, 0, 0, [0.0, 0.0, 1.0], 0.5);
        assert!((buf3[0][0] - 0.5).abs() < 1e-6);
        assert!((buf3[0][2] - 0.5).abs() < 1e-6);
    }
}

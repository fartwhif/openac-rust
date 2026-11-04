//! The terminal's 2-D interface: a primitive layer drawn straight into the
//! cell grid with UTF-8 text — the vitals, the target, the chat feed and
//! its entry box. It reads the scene's semantic facts and never renders the
//! game's widget tree.
//!
//! The 3-D world — the offscreen render of the real world renderer — fills
//! the whole grid; the HUD paints over its edges.

use std::collections::VecDeque;

use ac_render::scene::Scene;
use ac_runtime::communication::ChatLine;
use ac_runtime::session::interface::InterfaceEvent;
use ac_runtime::world::{PlayerView, VitalReading};
use ac_ui::radar::{BlipShape, CompassPoint, RadarSnapshot};

// -- The cell ---------------------------------------------------------------

/// A terminal cell after encoding: the braille character and its colours.
/// The encoder (the terminal front end) and this layer share it.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct Cell {
    pub(crate) ch: char,
    pub(crate) fg: [u8; 3],
    pub(crate) bg: [u8; 3],
}

// -- Palette ---------------------------------------------------------------

type Rgb = [u8; 3];
const BG: Rgb = [13, 15, 21];
const TEXT: Rgb = [212, 216, 226];
const DIM: Rgb = [138, 146, 158];
const TITLE: Rgb = [240, 238, 208];
const GOLD: Rgb = [228, 188, 96];
const HP: Rgb = [206, 66, 58];
const STAMINA: Rgb = [92, 176, 100];
const MANA: Rgb = [96, 130, 220];
const TARGET: Rgb = [232, 120, 120];
/// The player's marker at the radar's centre (bright green, as the retail's).
const PLAYER: Rgb = [64, 224, 96];
/// The radar disc's body, darker than the world so the blips read.
const RADAR_BODY: Rgb = [15, 19, 27];
/// The radar disc's dim edge.
const RADAR_EDGE: Rgb = [62, 70, 84];

// -- Geometry ---------------------------------------------------------------

/// An inclusive-corner rectangle in cells.
#[derive(Clone, Copy, Debug)]
pub struct Rect {
    pub x: usize,
    pub y: usize,
    pub w: usize,
    pub h: usize,
}

/// Where the 3-D viewport and the HUD regions sit this frame.
pub struct Layout {
    /// The 3-D world's area (the whole grid).
    pub viewport: Rect,
    /// The vitals block, over the viewport's top-left.
    pub vitals: Rect,
    /// The chat feed, over the viewport's bottom (its foot row is the entry
    /// box).
    pub chat: Rect,
    /// The radar disc, over the viewport's top-right (its foot row holds the
    /// coordinates line).
    pub radar: Rect,
}

/// Compute the frame's regions: the world fills the whole grid, the vitals
/// sit over its top-left, and the chat over its bottom. The vitals and the
/// radar take the retail windows' share of the screen (the retail's 1280 x
/// 720 window: the vitals are 160 pixels wide, the radar disc 120).
pub fn layout(cols: usize, rows: usize) -> Layout {
    let cols = cols.max(30);
    let rows = rows.max(12);
    // The vitals window (160 pixels wide in the game data) spans one eighth
    // of the width; the floor keeps the meters' readouts legible on small
    // terminals.
    let vitals_w = (cols / 8).max(16).min(cols);
    // The chat area is the transcript rows, the status line and the entry
    // box's row at its foot.
    let chat_h = 5.min(rows);

    let viewport = Rect {
        x: 0,
        y: 0,
        w: cols,
        h: rows,
    };
    let vitals = Rect {
        x: 1,
        y: 1,
        w: vitals_w,
        h: 7,
    };
    let chat = Rect {
        x: 1,
        y: rows.saturating_sub(chat_h + 1),
        w: cols - 2,
        h: chat_h,
    };
    // The radar disc, over the viewport's top-right: three thirty-seconds of
    // the width and a sixth of the height (120 pixels on the retail's
    // window; a sub-pixel is square, so it reads as a circle at the
    // retail's window aspect). Its foot row, below the disc, holds the
    // coordinates line. Three columns off the screen's right edge: the
    // coordinates line runs past the disc's right side and must clear the
    // edge.
    let radar_w = (3 * cols / 32).max(7).min(cols - 2);
    let radar_h = (rows / 6).max(4);
    let radar = Rect {
        x: cols.saturating_sub(radar_w + 3),
        y: 1.min(rows.saturating_sub(radar_h)),
        w: radar_w,
        h: radar_h,
    };

    Layout {
        viewport,
        vitals,
        chat,
        radar,
    }
}

// -- The cell grid and its painter -----------------------------------------

/// The full terminal's cell grid, row-major, with bounds-checked primitives.
pub struct Grid {
    w: usize,
    h: usize,
    rows: Vec<Vec<Cell>>,
}

impl Grid {
    /// An empty grid of `w`×`h` cells on the panel background.
    pub fn new(w: usize, h: usize) -> Self {
        let w = w.max(1);
        let h = h.max(1);
        let blank = Cell {
            ch: ' ',
            fg: TEXT,
            bg: BG,
        };
        Self {
            w,
            h,
            rows: vec![vec![blank; w]; h],
        }
    }

    /// The grid as rows of cells, for the encoder.
    pub fn rows(&self) -> &[Vec<Cell>] {
        &self.rows
    }

    fn put(&mut self, x: usize, y: usize, ch: char, fg: Rgb, bg: Rgb) {
        if x < self.w && y < self.h {
            // A cell is printed verbatim by the encoder; a control character
            // (a stray newline in chat text, a tab) would move the real
            // terminal's cursor and displace every later cell in its run.
            let ch = if ch.is_control() { ' ' } else { ch };
            self.rows[y][x] = Cell { ch, fg, bg };
        }
    }

    /// Fill `r` with a solid character (use `' '` for a plain background).
    pub fn fill(&mut self, r: Rect, ch: char, fg: Rgb, bg: Rgb) {
        for y in r.y..(r.y + r.h).min(self.h) {
            for x in r.x..(r.x + r.w).min(self.w) {
                self.put(x, y, ch, fg, bg);
            }
        }
    }

    /// Copy the 3-D viewport's grid over this one at `(ox, oy)`.
    pub fn blit(&mut self, ox: usize, oy: usize, grid: &[Vec<Cell>]) {
        for (y, row) in grid.iter().enumerate() {
            for (x, c) in row.iter().enumerate() {
                let tx = ox + x;
                let ty = oy + y;
                if tx < self.w && ty < self.h {
                    self.rows[ty][tx] = *c;
                }
            }
        }
    }

    /// A string left-to-right at `(x, y)`, clipped to the grid.
    pub fn put_str(&mut self, x: usize, y: usize, s: &str, fg: Rgb, bg: Rgb) {
        for (i, c) in s.chars().enumerate() {
            self.put(x + i, y, c, fg, bg);
        }
    }

    /// A labelled meter: a label, a filled bar, and a readout (e.g.
    /// `1500/2500`) over the bar's right end, as the retail's meter draws
    /// its numbers. A non-finite `frac` reads as empty.
    fn bar(&mut self, r: Rect, label: &str, frac: f32, full: Rgb, readout: &str) {
        let (x, y, w) = (r.x, r.y, r.w);
        let mut frac = frac.clamp(0.0, 1.0);
        if !frac.is_finite() {
            frac = 0.0;
        }
        let lw = label.chars().count().max(2);
        let bar_w = w.saturating_sub(lw + 1);
        if bar_w == 0 {
            return;
        }
        self.put_str(x, y, label, TEXT, BG);
        let bx = x + lw + 1;
        let filled = (frac * bar_w as f32).round() as usize;
        let readout: Vec<char> = readout.chars().collect();
        let rw = readout.len();
        for i in 0..bar_w {
            if rw > 0 && i + rw >= bar_w {
                // The readout over the bar's end: the number over the fill.
                // A readout wider than the bar keeps its rightmost part.
                let ch = readout[i + rw - bar_w];
                let bg = if i < filled { full } else { BG };
                self.put(bx + i, y, ch, TEXT, bg);
            } else if i < filled {
                self.put(bx + i, y, '█', full, BG);
            } else {
                self.put(bx + i, y, '░', DIM, BG);
            }
        }
    }
}

// -- Text helpers ----------------------------------------------------------

/// Truncate `s` to at most `n` characters.
fn truncate(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

/// The last `n` characters of `s`.
fn tail(s: &str, n: usize) -> String {
    let count = s.chars().count();
    s.chars().skip(count.saturating_sub(n)).collect()
}

/// `s` as its visual rows at most `n` characters wide (its terminal
/// wrapping): a row breaks at the last space within the width when one
/// exists, else hard at the width, and a break's spaces never carry onto
/// the next row.
fn wrap(s: &str, n: usize) -> Vec<String> {
    let n = n.max(1);
    let mut rows: Vec<String> = Vec::new();
    let mut rest = s;
    while !rest.is_empty() {
        if rest.chars().count() <= n {
            rows.push(rest.to_string());
            break;
        }
        let break_at = rest.char_indices().nth(n).map(|(b, _)| b).unwrap_or(rest.len());
        let cut = rest[..break_at]
            .rfind(' ')
            .filter(|p| *p > 0)
            .unwrap_or(break_at);
        rows.push(rest[..cut].to_string());
        let next = &rest[cut..];
        rest = next.trim_start_matches(' ');
    }
    rows
}

// -- The semantic model ----------------------------------------------------

/// The chat entry box at the foot of the chat area: the line the player is
/// composing, and whether the keys are going to it (text input mode) or to
/// the game. The text stays in the box while the player is out of the mode,
/// so a line left on a Tab can be read, edited or sent later.
#[derive(Default)]
pub struct ChatBox {
    /// The line the box holds.
    pub text: String,
    /// The keys type into the box (text input mode is on).
    pub active: bool,
}

/// The most characters a line of the entry box holds (the game's chat line
/// limit).
pub const CHAT_LINE_LIMIT: usize = 256;

impl ChatBox {
    /// Text input mode is on: the keys are the box's, not the game's.
    pub fn activate(&mut self) {
        self.active = true;
    }

    /// Text input mode leaves: the game gets the keys back, the text stays
    /// in the box.
    pub fn deactivate(&mut self) {
        self.active = false;
    }

    /// A typed character: taken while the line is below the limit; control
    /// characters (a Ctrl combo the terminal delivers as one) are refused.
    pub fn push_char(&mut self, c: char) {
        if c.is_control() {
            return;
        }
        if self.text.chars().count() >= CHAT_LINE_LIMIT {
            return;
        }
        self.text.push(c);
    }

    /// A backspace: the line loses its last character.
    pub fn backspace(&mut self) {
        self.text.pop();
    }

    /// Enter: the line is sent — the box empties and text input mode
    /// leaves. An empty line is not sent and the mode stays.
    pub fn enter(&mut self) -> Option<String> {
        if self.text.is_empty() {
            return None;
        }
        self.active = false;
        Some(std::mem::take(&mut self.text))
    }
}

/// The 2-D interface's copy of the scene's facts, kept in step each frame.
#[derive(Default)]
pub struct Model {
    vitals: Option<PlayerView>,
    char_name: Option<String>,
    target_guid: Option<u32>,
    target_name: Option<String>,
    target_hp: Option<f32>,
    chat: VecDeque<String>,
    status: VecDeque<String>,
    /// The entry box at the chat area's foot (its line and mode are driven
    /// by the terminal's keys, not the scene).
    pub chat_box: ChatBox,
    player_guid: u32,
    /// The circular radar: the player's bearing, the nearby blips, and the
    /// coordinates line, folded from the scene each frame.
    radar: RadarSnapshot,
}

impl Model {
    pub fn new() -> Self {
        Self::default()
    }

    /// The radar blip under terminal cell `(col, row)`, if any (its object
    /// id), for a terminal this `cols` wide and `rows` tall.
    #[must_use]
    pub fn radar_blip_at(&self, cols: usize, rows: usize, col: u16, row: u16) -> Option<u32> {
        radar_blip_at(&layout(cols, rows), &self.radar, col, row)
    }

    /// Fold the scene into the model: the vitals and target, the drained
    /// chat and status, and the interface events (the target's health and
    /// selection).
    pub fn update(&mut self, scene: &mut Scene) {
        self.player_guid = scene.player_guid().unwrap_or(0);
        // The radar: the player's bearing and every placed object, projected
        // to the disc (the scene's range test leaves out the far ones).
        let view = scene.player_view();
        self.radar = ac_ui::panels::radar::build_snapshot(&ac_render::ui::window::radar_scene(
            scene, view, None,
        ));
        self.char_name = scene.player().and_then(|p| p.name.clone());
        if let Some(pv) = scene.player_view() {
            self.vitals = Some(*pv);
            self.target_guid = pv.selected;
            self.target_name = pv
                .selected
                .and_then(|g| scene.object_by_guid(g).map(|o| o.name.clone()))
                .unwrap_or_else(|| self.target_name.clone());
        }
        for line in scene.take_chat() {
            self.push_chat(&line);
        }
        for s in scene.take_status_text() {
            self.status.push_back(s);
            while self.status.len() > 4 {
                self.status.pop_front();
            }
        }
        for ev in scene.take_interface() {
            self.apply(ev);
        }
    }

    fn push_chat(&mut self, line: &ChatLine) {
        let text = if line.text.is_empty() {
            line.words.clone()
        } else {
            line.text.clone()
        };
        if text.is_empty() {
            return;
        }
        self.chat.push_back(text);
        while self.chat.len() > 60 {
            self.chat.pop_front();
        }
    }

    fn apply(&mut self, ev: InterfaceEvent) {
        use InterfaceEvent as E;
        match ev {
            // The selected object's health (its `State` events).
            E::State(s) => {
                self.target_hp = s
                    .selected_health
                    .filter(|h| self.target_guid.is_some_and(|g| g == h.object_id))
                    .and_then(|h| h.percent);
            }
            E::Selection(sel) => {
                if let Some(g) = sel {
                    // A new selection: drop the old target's vitals.
                    if self.target_guid != Some(g) {
                        self.target_guid = Some(g);
                        self.target_name = None;
                        self.target_hp = None;
                    }
                } else {
                    self.target_guid = None;
                    self.target_name = None;
                    self.target_hp = None;
                }
            }
            _ => {}
        }
    }
}

// -- The always-on HUD -----------------------------------------------------

/// The vitals block and the chat over the world.
pub fn draw(g: &mut Grid, m: &Model, l: &Layout) {
    // The vitals + target, top-left: plain text over the world — no panel or
    // frame. Each line's background is only as wide as the line: the name
    // spans just the name, each meter spans its own label-and-readout width.
    // The names are never truncated to the block's width: they take the
    // cells they need and run over the world.
    let r = l.vitals;
    let cx = r.x + 1;
    let cw = r.w.saturating_sub(2);
    if let Some(name) = &m.char_name {
        g.put_str(cx, r.y + 1, name, TITLE, BG);
    }
    if let Some(pv) = &m.vitals {
        g.bar(
            Rect {
                x: cx,
                y: r.y + 2,
                w: cw,
                h: 1,
            },
            "HP",
            pv.health.percent.unwrap_or(0.0),
            HP,
            &vital_readout(&pv.health),
        );
        g.bar(
            Rect {
                x: cx,
                y: r.y + 3,
                w: cw,
                h: 1,
            },
            "SP",
            pv.stamina.percent.unwrap_or(0.0),
            STAMINA,
            &vital_readout(&pv.stamina),
        );
        g.bar(
            Rect {
                x: cx,
                y: r.y + 4,
                w: cw,
                h: 1,
            },
            "MP",
            pv.mana.percent.unwrap_or(0.0),
            MANA,
            &vital_readout(&pv.mana),
        );
        if let Some(guid) = m.target_guid {
            let name = m
                .target_name
                .clone()
                .unwrap_or_else(|| format!("#{guid:x}"));
            let hp = m
                .target_hp
                .map(|p| format!(" {:.0}%", p * 100.0))
                .unwrap_or_default();
            g.put_str(
                cx,
                r.y + 5,
                &format!("→ {name}{hp}"),
                TARGET,
                BG,
            );
        }
    }

    // The radar disc, top-right.
    draw_radar(g, &m.radar, l.radar);
    // The chat feed, bottom.
    draw_chat(g, m, l.chat);
}

/// A vital's readout: `current/max`, or blank until both are known.
fn vital_readout(v: &VitalReading) -> String {
    match (v.current, v.max) {
        (Some(c), Some(m)) => format!("{c}/{m}"),
        _ => String::new(),
    }
}

/// The radar disc and its blips, over the viewport's top-right. The player is
/// at the centre, facing up; the disc turns with the player's heading, the
/// compass letters mark the world directions, and the coordinates line sits
/// under the disc. The blips come pre-projected on the retail's 120-pixel disc
/// (centre 60, radius 50); each axis is rescaled to the cell disc, the vertical
/// halved for the cell's 2:1 aspect so the ellipse reads as a circle.
fn draw_radar(g: &mut Grid, radar: &RadarSnapshot, d: Rect) {
    let (cx, cy) = (d.x as f32 + d.w as f32 / 2.0, d.y as f32 + d.h as f32 / 2.0);
    let (rx, ry) = (d.w as f32 / 2.0, d.h as f32 / 2.0);

    // The disc's body: a dark ellipse so the blips read over the world.
    for y in d.y..(d.y + d.h) {
        let dy = (y as f32 - cy) / ry;
        let dy2 = dy * dy;
        if dy2 >= 1.0 {
            continue;
        }
        let half = rx * (1.0 - dy2).sqrt();
        let x0 = (cx - half).ceil() as isize;
        let x1 = (cx + half).floor() as isize;
        for x in x0..=x1 {
            if x >= 0 {
                g.put(x as usize, y, ' ', RADAR_BODY, RADAR_BODY);
            }
        }
    }
    // The disc's dim edge.
    for i in 0..=256 {
        let t = i as f32 * (std::f32::consts::TAU / 256.0);
        g.put(
            (cx + t.cos() * rx).round() as usize,
            (cy + t.sin() * ry).round() as usize,
            ' ',
            RADAR_EDGE,
            RADAR_EDGE,
        );
    }
    // The compass letters, turned with the player's heading (N ahead when the
    // player faces north).
    for (point, letter) in [
        (CompassPoint::North, 'N'),
        (CompassPoint::East, 'E'),
        (CompassPoint::South, 'S'),
        (CompassPoint::West, 'W'),
    ] {
        let t = radar.player_heading_degrees.to_radians() + compass_angle(point);
        g.put(
            (cx + t.sin() * rx * 0.8).round() as usize,
            (cy + t.cos() * ry * 0.8).round() as usize,
            letter,
            DIM,
            RADAR_BODY,
        );
    }
    // The blips, rescaled from the retail disc (centre 60, radius 50).
    if !radar.blank_blips {
        for b in &radar.blips {
            let x = (cx + (b.pixel_x - 60.0) * (rx / 50.0)).round() as usize;
            let y = (cy + (b.pixel_y - 60.0) * (ry / 50.0)).round() as usize;
            g.put(x, y, blip_char(b.shape), blip_rgb(b.color), RADAR_BODY);
        }
    }
    // The player, at the centre.
    g.put(
        cx.round() as usize,
        cy.round() as usize,
        '●',
        PLAYER,
        RADAR_BODY,
    );
    // The coordinates, under the disc, centred on it like the blips.
    if let Some(coords) = &radar.coordinates_text {
        let w = coords.chars().count();
        let x = (d.x + d.w / 2).saturating_sub(w / 2);
        g.put_str(x, d.y + d.h, coords, DIM, BG);
    }
}

/// The radar blip under terminal cell `(col, row)`, if any: its object id.
/// A blip is a single cell, so a click lands within a cell of it; the nearest
/// wins where blips crowd. Hidden blips (`blank_blips`) are not clickable.
/// This inverts `draw_radar`'s projection.
#[must_use]
pub fn radar_blip_at(layout: &Layout, radar: &RadarSnapshot, col: u16, row: u16) -> Option<u32> {
    if radar.blank_blips {
        return None;
    }
    let d = layout.radar;
    let (cx, cy) = (d.x as f32 + d.w as f32 / 2.0, d.y as f32 + d.h as f32 / 2.0);
    let (rx, ry) = (d.w as f32 / 2.0, d.h as f32 / 2.0);
    let (mx, my) = (col as f32, row as f32);
    let mut best: Option<(f32, u32)> = None;
    for b in &radar.blips {
        let x = (cx + (b.pixel_x - 60.0) * (rx / 50.0)).round();
        let y = (cy + (b.pixel_y - 60.0) * (ry / 50.0)).round();
        let dx = (x - mx).abs();
        let dy = (y - my).abs();
        if dx <= 1.0 && dy <= 1.0 {
            let dist = dx + dy;
            let take = match best {
                None => true,
                Some((d0, _)) => dist < d0,
            };
            if take {
                best = Some((dist, b.object_id));
            }
        }
    }
    best.map(|(_, id)| id)
}

/// A compass point's bearing offset from the player's heading (radians).
fn compass_angle(point: CompassPoint) -> f32 {
    use std::f32::consts::{FRAC_PI_2, PI};
    match point {
        CompassPoint::North => PI,
        CompassPoint::East => FRAC_PI_2,
        CompassPoint::South => 0.0,
        CompassPoint::West => 3.0 * FRAC_PI_2,
    }
}

/// A blip's terminal glyph: the shape reads at a glance, the colour carries
/// the relationship (the retail's shape and colour, one cell each).
fn blip_char(shape: BlipShape) -> char {
    match shape {
        BlipShape::Circle => '·',
        BlipShape::Box => '□',
        BlipShape::X => 'x',
        BlipShape::Plus => '+',
        BlipShape::Triangle => '▲',
        BlipShape::InvertedTriangle => '▼',
        BlipShape::XBox => '⊠',
        BlipShape::Undef => '·',
    }
}

/// A radar colour as terminal RGB.
fn blip_rgb(c: ac_ui::draw::Color) -> [u8; 3] {
    [
        (c.r * 255.0) as u8,
        (c.g * 255.0) as u8,
        (c.b * 255.0) as u8,
    ]
}

fn draw_chat(g: &mut Grid, m: &Model, r: Rect) {
    if r.w < 4 || r.h < 3 {
        return;
    }
    let inner_w = r.w - 2;
    // The last row is the entry box, the one above it the status line, the
    // rest the transcript.
    let line_rows = r.h - 2;
    // A chat line may carry embedded newlines (a server multi-line
    // message), and a line wider than the area wraps over its visual
    // rows — the terminal's size sets the wrap, and a line never loses
    // its tail. The last `line_rows` *rows* are shown, and a newline
    // never lands in a cell.
    let mut rows: Vec<String> = Vec::new();
    for line in m.chat.iter() {
        for segment in line.split('\n') {
            let mut visual = wrap(segment, inner_w);
            if visual.is_empty() {
                // An empty segment is still a blank row.
                visual.push(String::new());
            }
            rows.extend(visual);
        }
    }
    let start = rows.len().saturating_sub(line_rows);
    for (i, row) in rows.iter().skip(start).enumerate() {
        g.put_str(r.x + 1, r.y + i, row, TEXT, BG);
    }
    if let Some(status) = m.status.back() {
        g.put_str(r.x + 1, r.y + r.h - 2, &truncate(status, inner_w), DIM, BG);
    }
    draw_chat_box(g, &m.chat_box, r);
}

/// The entry box on the chat area's bottom row: a `>` prompt, the line's
/// end (the caret sits at its end) and, in text input mode, a block caret.
/// Out of the mode the line stays on show, dimmed; an empty box leaves the
/// row clear.
fn draw_chat_box(g: &mut Grid, entry: &ChatBox, r: Rect) {
    if !entry.active && entry.text.is_empty() {
        return;
    }
    let y = r.y + r.h - 1;
    let row = Rect {
        x: r.x,
        y,
        w: r.w,
        h: 1,
    };
    let ink = if entry.active { TEXT } else { DIM };
    g.fill(row, ' ', ink, BG);
    g.put(r.x + 1, y, '>', ink, BG);
    let caret = entry.active;
    // The row holds `> `, the line's end and the caret.
    let width = r.w - 2 - 2 - usize::from(caret);
    let visible = tail(&entry.text, width);
    g.put_str(r.x + 3, y, &visible, ink, BG);
    if caret {
        g.put(r.x + 3 + visible.chars().count(), y, '█', GOLD, BG);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A chat line may carry embedded newlines (a server multi-line message,
    /// e.g. the login banner). The lines must expand into their visual rows,
    /// and no control character may ever land in a cell — a cell is printed
    /// verbatim, and a raw newline in the middle of a row would move the
    /// terminal's cursor and displace the rest of the frame.
    #[test]
    fn a_multiline_chat_line_occupies_its_visual_rows() {
        let mut m = Model::new();
        m.chat
            .push_back("Welcome to Asheron's Call\n  powered by ACEmulator".into());
        let mut g = Grid::new(40, 8);
        draw_chat(
            &mut g,
            &m,
            Rect {
                x: 1,
                y: 3,
                w: 38,
                h: 4,
            },
        );
        // First visual row on y=3, the continuation on y=4.
        assert_eq!(g.rows()[3][2].ch, 'W', "first line starts at (2, 3)");
        assert_eq!(g.rows()[4][4].ch, 'p', "continuation row is '  powered...'");
        for row in g.rows() {
            for cell in row {
                assert!(
                    !cell.ch.is_control(),
                    "a control char in the grid: {:?}",
                    cell.ch
                );
            }
        }
    }

    /// A chat line wider than the area wraps over its visual rows instead
    /// of losing its tail: it breaks at the last space within the width
    /// when one exists, hard at the width otherwise, and the transcript
    /// still shows its newest `line_rows` rows.
    #[test]
    fn a_long_chat_line_wraps_to_the_area_width() {
        let mut m = Model::new();
        // 43 characters, with spaces; 26 without.
        m.chat.push_back("the quick brown fox jumps over the lazy dog".into());
        m.chat.push_back("abcdefghijklmnopqrstuvwxyz".into());
        let mut g = Grid::new(20, 10);
        // inner width 14, six transcript rows — all six visual rows show.
        draw_chat(
            &mut g,
            &m,
            Rect {
                x: 1,
                y: 1,
                w: 16,
                h: 8,
            },
        );
        let row = |y: usize| -> String { g.rows()[y].iter().map(|c| c.ch).collect() };
        // The spaced line breaks at the last space within 14.
        assert_eq!(&row(1)[2..11], "the quick", "first row breaks at a space");
        assert_eq!(&row(2)[2..11], "brown fox", "second row breaks at a space");
        assert_eq!(&row(3)[2..12], "jumps over", "third row breaks at a space");
        // The tail survives: the last words are whole.
        assert_eq!(&row(4)[2..14], "the lazy dog", "the line's tail is whole");
        // The spaceless token breaks hard at the width.
        assert_eq!(&row(5)[2..16], "abcdefghijklmn", "hard break at the width");
        assert_eq!(&row(6)[2..14], "opqrstuvwxyz", "the token's tail is whole");
    }

    /// The encoder's choke point: any text source (chat, status, a player
    /// name) may carry control characters; a cell never keeps one.
    #[test]
    fn a_cell_never_keeps_a_control_character() {
        let mut g = Grid::new(5, 1);
        g.put_str(0, 0, "a\tb\nc", TEXT, BG);
        assert_eq!(
            g.rows()[0].iter().map(|c| c.ch).collect::<String>(),
            "a b c",
            "control characters become spaces"
        );
    }

    /// The entry box's state: typed characters build the line (up to the
    /// limit, no control characters), backspace shortens it, Enter sends
    /// the line (emptying the box and leaving the mode), an empty Enter
    /// sends nothing and keeps the mode, and leaving the mode (a Tab)
    /// keeps the line.
    #[test]
    fn the_entry_box_keeps_its_line_and_mode() {
        let mut entry = ChatBox::default();
        assert!(!entry.active);

        entry.activate();
        for c in "hello".chars() {
            entry.push_char(c);
        }
        entry.backspace();
        assert_eq!(entry.text, "hell");

        assert_eq!(entry.enter(), Some("hell".into()));
        assert!(!entry.active, "sending leaves text input mode");
        assert!(entry.text.is_empty(), "the sent line leaves the box");

        entry.activate();
        assert_eq!(entry.enter(), None, "an empty line is not sent");
        assert!(entry.active, "an empty Enter keeps the mode");

        for c in "again".chars() {
            entry.push_char(c);
        }
        entry.deactivate();
        assert!(!entry.active);
        assert_eq!(entry.text, "again", "leaving the mode keeps the line");

        entry.push_char('\u{0003}');
        assert_eq!(entry.text, "again", "a control character is refused");
        for _ in 0..CHAT_LINE_LIMIT + 4 {
            entry.push_char('a');
        }
        assert_eq!(
            entry.text.chars().count(),
            CHAT_LINE_LIMIT,
            "the line holds at the limit"
        );
    }

    /// The entry box's row: in text input mode a prompt, the line's end and
    /// a block caret; out of it the line stays on show (no caret); an empty
    /// box leaves the row clear.
    #[test]
    fn the_entry_box_row_shows_the_line_and_its_caret() {
        let mut m = Model::new();
        let r = Rect {
            x: 1,
            y: 1,
            w: 18,
            h: 1,
        };

        m.chat_box.text = "hello".into();
        m.chat_box.active = true;
        let mut g = Grid::new(20, 2);
        draw_chat_box(&mut g, &m.chat_box, r);
        let row: String = g.rows()[1].iter().map(|c| c.ch).collect();
        assert_eq!(row, "  > hello█".to_owned() + &" ".repeat(10));
        assert_eq!(g.rows()[1][9].fg, GOLD, "the caret is the accent colour");

        m.chat_box.active = false;
        let mut g = Grid::new(20, 2);
        draw_chat_box(&mut g, &m.chat_box, r);
        let row: String = g.rows()[1].iter().map(|c| c.ch).collect();
        assert_eq!(row, "  > hello".to_owned() + &" ".repeat(11));

        m.chat_box.text.clear();
        let mut g = Grid::new(20, 2);
        draw_chat_box(&mut g, &m.chat_box, r);
        let row: String = g.rows()[1].iter().map(|c| c.ch).collect();
        assert_eq!(row, " ".repeat(20), "an empty box leaves the row clear");
    }

    /// A line longer than the row shows its end (where the caret is), not
    /// its start.
    #[test]
    fn the_entry_box_row_keeps_the_caret_in_view() {
        let mut m = Model::new();
        m.chat_box.text = "abcdefghij".into();
        m.chat_box.active = true;
        let r = Rect {
            x: 0,
            y: 0,
            w: 8,
            h: 1,
        };
        let mut g = Grid::new(8, 1);
        draw_chat_box(&mut g, &m.chat_box, r);
        let row: String = g.rows()[0].iter().map(|c| c.ch).collect();
        assert_eq!(row, " > hij█ ", "prompt, the line's end, the caret");
    }

    /// The chat area's rows: the transcript on top, the status line above
    /// the entry box's row at the foot.
    #[test]
    fn the_entry_box_takes_the_chat_area_bottom_row() {
        let mut m = Model::new();
        m.chat.push_back("a line".into());
        m.status.push_back("status".into());
        m.chat_box.text = "draft".into();
        let r = Rect {
            x: 1,
            y: 5,
            w: 28,
            h: 5,
        };
        let mut g = Grid::new(30, 10);
        draw_chat(&mut g, &m, r);
        let line = |y: usize| -> String { g.rows()[y].iter().map(|c| c.ch).collect() };
        assert!(
            line(5).starts_with("  a line"),
            "the transcript: {}",
            line(5)
        );
        assert!(line(8).starts_with("  status"), "the status: {}", line(8));
        assert!(line(9).starts_with("  > draft"), "the box: {}", line(9));
    }
    /// The vitals: plain text over the world — no panel or frame. The name's
    /// background spans only the name (the world shows just past it), the
    /// meters keep their labels, and the current/max readout sits over the
    /// bar's right end (the retail's meter) instead of beside it.
    #[test]
    fn the_vitals_sit_as_plain_text_over_the_world() {
        let l = layout(80, 24);
        let r = l.vitals;
        let cx = r.x + 1;
        let cw = r.w - 2;
        let mut g = Grid::new(80, 24);
        // A distinctive world behind the vitals.
        g.fill(l.viewport, '⣿', [255, 0, 0], [0, 0, 0]);
        let mut m = Model::new();
        let name = "+TermTest2";
        m.char_name = Some(name.into());
        m.vitals = Some(PlayerView {
            health: VitalReading {
                current: Some(1500),
                max: Some(2500),
                percent: Some(0.6),
            },
            stamina: VitalReading {
                current: Some(800),
                max: Some(1000),
                percent: Some(0.8),
            },
            mana: VitalReading {
                current: Some(400),
                max: Some(1000),
                percent: Some(0.4),
            },
            ..PlayerView::default()
        });
        draw(&mut g, &m, &l);

        let row_text = |y: usize| -> String { g.rows()[y].iter().map(|c| c.ch).collect() };

        // No frame: the vitals' top row is world, not a border.
        assert_eq!(g.rows()[r.y][r.x].ch, '⣿', "no top frame");

        // The name, with its background only as wide as the name: the cell
        // just past it is still world.
        assert!(
            row_text(r.y + 1).contains(name),
            "the name: {:?}",
            row_text(r.y + 1)
        );
        let past = g.rows()[r.y + 1][r.x + 1 + name.chars().count()].ch;
        assert_eq!(past, '⣿', "the name's background runs past the name");

        // The meters keep their labels, and the readout over the bar's
        // right end: `HP`, the gap, the fill the readout does not cover,
        // the number. On this 80-column terminal the vitals are 16 cells:
        // the bar spans 11 and the 9-character readout covers its right 9.
        let hp: String = row_text(r.y + 2).chars().skip(cx).take(cw).collect();
        assert_eq!(hp, "HP⣿██1500/2500", "the HP row");
        // The readout over the fill keeps the fill as its background, and
        // past the fill the panel's.
        assert_eq!(
            g.rows()[r.y + 2][cx + 3 + 4].bg,
            HP,
            "the number over the fill"
        );
        assert_eq!(
            g.rows()[r.y + 2][cx + 3 + 8].bg,
            BG,
            "the number past the fill"
        );
        // The row spans the vitals' width and no further: just past it is
        // still world.
        assert_eq!(g.rows()[r.y + 2][cx + cw].ch, '⣿', "the row's width");
        assert!(
            row_text(r.y + 3).contains("800/1000"),
            "SP: {:?}",
            row_text(r.y + 3)
        );
        assert!(
            row_text(r.y + 4).contains("400/1000"),
            "MP: {:?}",
            row_text(r.y + 4)
        );
    }

    /// The HUD takes the retail windows' share of the screen (the retail's
    /// 1280 x 720 window: the vitals are 160 pixels wide, the radar disc
    /// 120), with floors so a small terminal keeps a legible meter and a
    /// disc with its compass.
    #[test]
    fn the_hud_takes_the_retail_share_of_the_screen() {
        // The shares (no floor in play): one eighth, three thirty-seconds,
        // a sixth.
        let l = layout(128, 32);
        assert_eq!(l.vitals.w, 16, "the vitals are an eighth of the width");
        assert_eq!(l.radar.w, 12, "the disc is 120 pixels of 1280");
        assert_eq!(l.radar.h, 5, "the disc is 120 pixels of 720");

        // The floors on a small terminal.
        let l = layout(80, 24);
        assert_eq!(l.vitals.w, 16, "the vitals' floor");
        assert_eq!(l.radar.w, 7, "the disc's floor width");
        assert_eq!(l.radar.h, 4, "the disc's floor height");

        // A wide terminal keeps the shares.
        let l = layout(200, 50);
        assert_eq!(l.vitals.w, 25, "the vitals keep their eighth");
        assert_eq!(l.radar.w, 18, "the disc keeps its share");
        assert_eq!(l.radar.h, 8, "the disc keeps its share");
        // The radar never reaches the vitals.
        assert!(
            l.radar.x > l.vitals.x + l.vitals.w,
            "the radar is clear of the vitals"
        );

        // The radar keeps three columns from the screen's right edge, so
        // the coordinates line under the disc clears the edge — at every
        // size, floor and share alike.
        for (cols, rows) in [(128, 32), (80, 24), (200, 50)] {
            let l = layout(cols, rows);
            assert_eq!(
                l.radar.x + l.radar.w + 3,
                cols,
                "the radar's right margin ({cols} x {rows})"
            );
        }
    }

    /// The character's name and the target's line are never truncated to
    /// the vitals' width: a name longer than the block runs over the world
    /// whole (the grid's right edge is the only bound).
    #[test]
    fn the_names_run_over_the_world_untruncated() {
        let l = layout(80, 24);
        let r = l.vitals;
        let mut g = Grid::new(80, 24);
        // A distinctive world to run over.
        g.fill(l.viewport, '⣿', [255, 0, 0], [0, 0, 0]);
        let mut m = Model::new();
        // Longer than the block's width (16 here) and past its right edge.
        let name = "a-very-long-character-name";
        m.char_name = Some(name.into());
        m.vitals = Some(PlayerView::default());
        m.target_guid = Some(0x1234);
        m.target_name = Some("an-equally-long-target-name".into());
        draw(&mut g, &m, &l);

        let row_text = |y: usize| -> String { g.rows()[y].iter().map(|c| c.ch).collect() };
        // The name, whole, starting in the block and ending past it.
        let name_row = row_text(r.y + 1);
        assert!(name_row.contains(name), "the name is whole: {name_row:?}");
        let end = name_row.find(name).unwrap() + name.chars().count();
        assert!(
            end > r.x + r.w,
            "the name runs past the block's right edge: {name_row:?}"
        );
        // The target's line, whole, name and all.
        let target_row = row_text(r.y + 5);
        assert!(
            target_row.contains("→ an-equally-long-target-name"),
            "the target's line is whole: {target_row:?}"
        );
    }

    /// The radar: the player marks the disc's centre, a blip rescales from the
    /// retail 120-pixel disc (centre 60, radius 50) into the cell disc, the
    /// compass sits at the top when facing north, and the coordinates line sits
    /// under the disc.
    #[test]
    fn the_radar_marks_the_player_and_projects_blips() {
        use ac_ui::draw::Color;
        use ac_ui::radar::RadarBlip;
        let d = Rect {
            x: 100,
            y: 2,
            w: 24,
            h: 12,
        };
        let mut g = Grid::new(140, 30);
        let snapshot = RadarSnapshot {
            player_heading_degrees: 0.0,
            blips: vec![RadarBlip {
                object_id: 1,
                name: "east".into(),
                pixel_x: 90.0, // 30 east of the retail centre (0.6 of the radius)
                pixel_y: 60.0,
                color: Color::rgba(1.0, 0.5, 0.0, 1.0),
                shape: BlipShape::Plus,
                selected: false,
            }],
            coordinates_text: Some("42.1N,33.9E".into()),
            blank_blips: false,
            ui_locked: false,
        };
        draw_radar(&mut g, &snapshot, d);
        // The player, at the disc's centre.
        assert_eq!(g.rows()[8][112].ch, '●', "player marker");
        assert_eq!(g.rows()[8][112].fg, PLAYER);
        // The blip: 30 retail pixels east lands in the disc, east of centre.
        let b = &g.rows()[8][119];
        assert_eq!(b.ch, '+', "blip glyph");
        assert_eq!(b.fg, [255, 127, 0], "blip colour");
        // Facing north, the compass's N is at the top, off the centre.
        assert_eq!(
            g.rows()[3].iter().position(|c| c.ch == 'N'),
            Some(112),
            "N at the top"
        );
        // The coordinates line, under the disc.
        let coords = g.rows()[14].iter().map(|c| c.ch).collect::<String>();
        assert!(coords.contains("42.1N,33.9E"), "coordinates: {coords:?}");
    }

    /// A click on a blip's cell selects that blip's object; a click far from
    /// any blip, or with the blips hidden, selects nothing. This inverts
    /// `draw_radar`'s projection.
    #[test]
    fn a_radar_click_selects_the_blip_under_the_cell() {
        use ac_ui::draw::Color;
        use ac_ui::radar::RadarBlip;
        let d = Rect {
            x: 100,
            y: 2,
            w: 24,
            h: 12,
        };
        let layout = Layout {
            viewport: d,
            vitals: d,
            chat: d,
            radar: d,
        };
        let snapshot = RadarSnapshot {
            player_heading_degrees: 0.0,
            blips: vec![RadarBlip {
                object_id: 1,
                name: "east".into(),
                pixel_x: 90.0,
                pixel_y: 60.0,
                color: Color::rgba(1.0, 0.5, 0.0, 1.0),
                shape: BlipShape::Plus,
                selected: false,
            }],
            coordinates_text: None,
            blank_blips: false,
            ui_locked: false,
        };
        // The blip's cell (the draw test lands it at row 8, column 119).
        assert_eq!(
            radar_blip_at(&layout, &snapshot, 119, 8),
            Some(1),
            "on the blip"
        );
        // A cell far from the blip and the disc hits nothing.
        assert_eq!(
            radar_blip_at(&layout, &snapshot, 0, 0),
            None,
            "off the blip"
        );
        // Hidden blips are not clickable.
        let hidden = RadarSnapshot {
            blank_blips: true,
            ..snapshot.clone()
        };
        assert_eq!(
            radar_blip_at(&layout, &hidden, 119, 8),
            None,
            "hidden blips"
        );
    }
}

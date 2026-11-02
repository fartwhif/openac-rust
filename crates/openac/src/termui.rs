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
use ac_runtime::world::PlayerView;

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
}

/// Compute the frame's regions: the world fills the whole grid, the vitals
/// sit over its top-left, and the chat over its bottom.
pub fn layout(cols: usize, rows: usize) -> Layout {
    let cols = cols.max(30);
    let rows = rows.max(12);
    let vitals_w = 26.min(cols);
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
        h: 5,
    };
    let chat = Rect {
        x: 1,
        y: rows.saturating_sub(chat_h + 1),
        w: cols - 2,
        h: chat_h,
    };

    Layout {
        viewport,
        vitals,
        chat,
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

    /// A labelled meter: a short label, a filled bar, and a percent, all on
    /// one row `w` wide.
    fn bar(&mut self, x: usize, y: usize, w: usize, label: &str, frac: f32, full: Rgb) {
        let frac = frac.clamp(0.0, 1.0);
        let lw = 3usize;
        let pw = 4usize;
        if w < lw + pw + 2 {
            return;
        }
        let bar_w = w - lw - pw - 1;
        self.put_str(x, y, &format!("{label:<lw$}"), TEXT, BG);
        let bx = x + lw + 1;
        let filled = (frac * bar_w as f32).round() as usize;
        for i in 0..bar_w {
            if i < filled {
                self.put(bx + i, y, '█', full, BG);
            } else {
                self.put(bx + i, y, '░', DIM, BG);
            }
        }
        self.put_str(x + w - pw, y, &format!("{:.0}%", frac * 100.0), TEXT, BG);
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
}

impl Model {
    pub fn new() -> Self {
        Self::default()
    }

    /// Fold the scene into the model: the vitals and target, the drained
    /// chat and status, and the interface events (the target's health and
    /// selection).
    pub fn update(&mut self, scene: &mut Scene) {
        self.player_guid = scene.player_guid().unwrap_or(0);
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
    // The vitals + target, top-left.
    let r = l.vitals;
    if let Some(name) = &m.char_name {
        g.put_str(r.x, r.y, &truncate(name, r.w), TITLE, BG);
    }
    if let Some(pv) = &m.vitals {
        g.bar(
            r.x,
            r.y + 1,
            r.w,
            "HP",
            pv.health.percent.unwrap_or(0.0),
            HP,
        );
        g.bar(
            r.x,
            r.y + 2,
            r.w,
            "SP",
            pv.stamina.percent.unwrap_or(0.0),
            STAMINA,
        );
        g.bar(
            r.x,
            r.y + 3,
            r.w,
            "MP",
            pv.mana.percent.unwrap_or(0.0),
            MANA,
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
                r.x,
                r.y + 4,
                &truncate(&format!("→ {name}{hp}"), r.w),
                TARGET,
                BG,
            );
        }
    }

    // The chat feed, bottom.
    draw_chat(g, m, l.chat);
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
    // message); expand each into its visual rows so the last `line_rows`
    // *rows* are shown and a newline never lands in a cell.
    let mut rows: Vec<&str> = Vec::new();
    for line in m.chat.iter() {
        rows.extend(line.split('\n'));
    }
    let start = rows.len().saturating_sub(line_rows);
    for (i, row) in rows.iter().skip(start).enumerate() {
        g.put_str(r.x + 1, r.y + i, &truncate(row, inner_w), TEXT, BG);
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
}

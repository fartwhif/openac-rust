//! The terminal's 2-D interface: a primitive layer drawn straight into the
//! cell grid with UTF-8 text — the vitals, the target, the chat feed and
//! its entry box. It reads the scene's semantic facts and never renders the
//! game's widget tree.
//!
//! The 3-D world — the offscreen render of the real world renderer — fills
//! the whole grid; the HUD paints over its edges.

use std::collections::VecDeque;

use ac_core::items::{ClientObject, ClientObjectTable};
use ac_render::scene::Scene;
use ac_runtime::communication::ChatLine;
use ac_runtime::inventory::{
    AppraisalView, ItemCommand, ObjectRef, TradeView, VendorCommand, VendorView,
};
use ac_runtime::session::interface::{InterfaceEvent, apply_items_update};
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
    /// The item widget (the inventory, a container, the trade, the vendor
    /// or an appraisal), over the world: a quarter of the width and three
    /// quarters of the height, just past the vitals.
    pub widget: Rect,
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
    // The item widget: a quarter of the width and three quarters of the
    // height (the floor keeps its rows legible on a small terminal), just
    // past the vitals (which stay readable) when the terminal is wide
    // enough for both.
    let widget_w = (cols / 4).max(14).min(cols - 2);
    let widget_h = (3 * rows / 4).max(10).min(rows - 2);
    let widget_x = if 1 + vitals_w + 1 + widget_w <= cols {
        vitals_w + 2
    } else {
        1
    };
    let widget = Rect {
        x: widget_x.min(cols - widget_w),
        y: 1,
        w: widget_w,
        h: widget_h,
    };

    Layout {
        viewport,
        vitals,
        chat,
        radar,
        widget,
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

// -- The item widgets ------------------------------------------------------

/// The widget open over the world (at most one at a time): the player's own
/// packs, the contents of an open container, the open secure trade, the open
/// vendor's shop, or an appraisal answer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Widget {
    #[default]
    None,
    /// The player's pack, its side packs and what is worn.
    Inventory,
    /// The contents of an open container (a ground container or a pack).
    Container(u32),
    /// The open secure trade.
    Trade,
    /// The open vendor's shop.
    Vendor,
    /// An appraisal answer.
    Appraise,
}

/// One key that can act on the open widget (the terminal maps its keys to
/// these; the navigation keys are consumed, the rest may send a request).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WidgetKey {
    Up,
    Down,
    PageUp,
    PageDown,
    Home,
    End,
    /// The primary action on the selected row (an inventory's use, a
    /// container's take, a vendor's buy of the listing).
    Enter,
    /// Appraise the selected row's item.
    Appraise,
    /// Drop the selected row's item on the ground.
    Drop,
    /// Add: the container's take-all, the trade's offer, the vendor's +1.
    Add,
    /// Buy everything staged (a vendor).
    BuyAll,
    /// Sell everything staged (a vendor).
    SellAll,
    /// Accept, decline or reset (a trade).
    Accept,
    Decline,
    Reset,
    /// Take the selected staged row off its list (a vendor).
    Back,
}

/// What a widget row is: the widget's keys act on it by its kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowKind {
    /// An item the player carries (a pack, an equipment, a container, a
    /// trade or a vendor row).
    Item,
    /// A side pack (its use opens it).
    Pack,
    /// A listed item at the open vendor.
    Shop,
    /// A row on the vendor's buy or sell list.
    Buy,
    Sell,
    /// A plain line (a section header, a count, the purse, a property).
    Plain,
}

/// One row of the open widget: its text and what its keys act on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WidgetRow {
    /// The row's text.
    pub text: String,
    /// The item its keys act on (`None` on a plain line).
    pub guid: Option<u32>,
    /// What the widget's keys do to it.
    pub kind: RowKind,
}

/// A plain row (a section header, a count, the purse, a property).
fn plain(text: impl Into<String>) -> WidgetRow {
    WidgetRow {
        text: text.into(),
        guid: None,
        kind: RowKind::Plain,
    }
}

/// One item's row text: its name (plural for a stack) and the stack's count.
fn item_text(item: &ClientObject) -> String {
    let name = item.appropriate_name().into_owned();
    if item.stack_size > 1 {
        format!("{name} x{}", item.stack_size)
    } else {
        name
    }
}

/// A vendor list row's text: its name, count and price.
fn staged_text(name: &str, quantity: i32, price: i32) -> String {
    format!("{name} x{quantity}  {price}p")
}

/// A yes/no word (the trade's acceptances).
fn word(yes: bool) -> &'static str {
    if yes { "yes" } else { "no" }
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
    /// The item table (the session's copy; the widgets read it).
    items: ClientObjectTable,
    /// The widget open, if any (at most one at a time).
    widget: Widget,
    /// The widget's selected row (into [`Self::widget_rows`]).
    widget_sel: usize,
    /// The widget's first visible row (the wheel's scroll).
    widget_scroll: usize,
    /// The widget the appraise opened over (its close returns to it).
    appraise_under: Option<Widget>,
    /// The open secure trade, as the session shows it.
    trade: TradeView,
    /// The open vendor's shop, if any.
    vendor: Option<VendorView>,
    /// The latest appraisal answer (the appraise widget's content).
    appraisal: Option<AppraisalView>,
    /// The outside container the player is looking into (0: none).
    external_container: u32,
    /// The pack the inventory window shows open (0: none).
    open_pack: u32,
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

    pub(crate) fn apply(&mut self, ev: InterfaceEvent) {
        use InterfaceEvent as E;
        match ev {
            // The item table's copy: the widgets' data. A cleared table
            // (a teleport) empties an open container's contents.
            E::Items(update) => {
                if update.reset {
                    self.open_pack = 0;
                    if matches!(self.widget, Widget::Container(_)) {
                        self.widget = Widget::None;
                    }
                }
                apply_items_update(&mut self.items, &update);
            }
            E::State(s) => {
                // The selected object's health (its `State` events).
                self.target_hp = s
                    .selected_health
                    .filter(|h| self.target_guid.is_some_and(|g| g == h.object_id))
                    .and_then(|h| h.percent);
                // The outside container the player is looking into.
                if s.external_container != self.external_container {
                    self.external_container = s.external_container;
                    if s.external_container != 0 {
                        self.open_widget(Widget::Container(s.external_container));
                    } else if let Widget::Container(g) = self.widget
                        && g != self.open_pack
                    {
                        self.widget = Widget::None;
                    }
                }
            }
            // A pack the player carries was used: the widget opens on it.
            E::OpenContainedContainer(g) => {
                self.open_pack = g;
                self.open_widget(Widget::Container(g));
            }
            E::Trade(t) => {
                let open = t.open;
                self.trade = *t;
                if open {
                    self.open_widget(Widget::Trade);
                } else if self.widget == Widget::Trade {
                    self.widget = Widget::None;
                }
            }
            E::Vendor(v) => match v {
                Some(v) => {
                    self.vendor = Some(*v);
                    self.open_widget(Widget::Vendor);
                }
                None => {
                    self.vendor = None;
                    if self.widget == Widget::Vendor {
                        self.widget = Widget::None;
                    }
                }
            },
            E::Appraisal(a) => {
                let opens = a.opens_window;
                self.appraisal = Some(*a);
                if opens {
                    if self.widget != Widget::Appraise {
                        self.appraise_under = Some(self.widget);
                    }
                    self.widget = Widget::Appraise;
                    self.widget_sel = 0;
                    self.widget_scroll = 0;
                }
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

    /// Open `w` (its selection and scroll start at the top).
    fn open_widget(&mut self, w: Widget) {
        if self.widget != w {
            self.widget = w;
            self.widget_sel = 0;
            self.widget_scroll = 0;
        }
    }

    // -- The item widgets ----------------------------------------------

    /// A widget is open.
    #[must_use]
    pub fn widget_open(&self) -> bool {
        self.widget != Widget::None
    }

    /// The open widget's title line.
    #[must_use]
    pub fn widget_title(&self) -> String {
        match self.widget {
            Widget::None => String::new(),
            Widget::Inventory => "backpack".into(),
            Widget::Container(g) => self
                .items
                .get(g)
                .map(|o| o.appropriate_name().into_owned())
                .unwrap_or_else(|| "container".into()),
            Widget::Trade => match &self.trade.partner_name {
                Some(name) => format!("trade: {name}"),
                None => format!("trade: 0x{:08X}", self.trade.partner_guid),
            },
            Widget::Vendor => match self.vendor.as_ref().and_then(|v| v.name.clone()) {
                Some(name) => format!("vendor: {name}"),
                None => "vendor".into(),
            },
            Widget::Appraise => self
                .appraisal
                .as_ref()
                .and_then(|a| a.name.clone())
                .unwrap_or_else(|| "appraise".into()),
        }
    }

    /// The open widget's keys' line.
    #[must_use]
    pub fn widget_footer(&self) -> &'static str {
        match self.widget {
            Widget::None => "",
            Widget::Inventory => "enter use · e appraise",
            Widget::Container(_) => "enter take · a take all",
            Widget::Trade => "a offer · y accept · esc",
            Widget::Vendor => "a add · b buy · s sell",
            Widget::Appraise => "esc close",
        }
    }

    /// The open widget's rows, whole (the draw and the keys window them).
    #[must_use]
    pub fn widget_rows(&self, width: usize) -> Vec<WidgetRow> {
        match self.widget {
            Widget::None => Vec::new(),
            Widget::Inventory => self.inventory_rows(),
            Widget::Container(g) => self.container_rows(g),
            Widget::Trade => self.trade_rows(),
            Widget::Vendor => self.vendor_rows(),
            Widget::Appraise => self.appraise_rows(width.saturating_sub(2).max(8)),
        }
    }

    fn inventory_rows(&self) -> Vec<WidgetRow> {
        let mut rows = Vec::new();
        let pack: Vec<u32> = self.items.get_contents(self.player_guid).to_vec();
        if pack.is_empty() {
            rows.push(plain("(empty)"));
        }
        for &g in &pack {
            let Some(item) = self.items.get(g) else {
                continue;
            };
            if item.container_type_hint == 1 || item.items_capacity > 0 {
                let n = self.items.get_contents(g).len();
                rows.push(WidgetRow {
                    text: format!("{} ({n} items)", item.appropriate_name()),
                    guid: Some(g),
                    kind: RowKind::Pack,
                });
            } else {
                rows.push(WidgetRow {
                    text: item_text(item),
                    guid: Some(g),
                    kind: RowKind::Item,
                });
            }
        }
        let equipped = self.items.get_equipped_by(self.player_guid);
        if !equipped.is_empty() {
            rows.push(plain("equipped:"));
            for o in equipped {
                rows.push(WidgetRow {
                    text: item_text(o),
                    guid: Some(o.object_id()),
                    kind: RowKind::Item,
                });
            }
        }
        rows
    }

    fn container_rows(&self, guid: u32) -> Vec<WidgetRow> {
        let mut rows = Vec::new();
        for &g in self.items.get_contents(guid) {
            if let Some(item) = self.items.get(g) {
                rows.push(WidgetRow {
                    text: item_text(item),
                    guid: Some(g),
                    kind: RowKind::Item,
                });
            }
        }
        if rows.is_empty() {
            rows.push(plain("(empty)"));
        }
        rows
    }

    fn trade_rows(&self) -> Vec<WidgetRow> {
        let t = &self.trade;
        let name = |m: &Model, g: u32| {
            t.item_names
                .get(&g)
                .cloned()
                .or_else(|| m.items.get(g).map(item_text))
                .unwrap_or_else(|| format!("item 0x{g:08X}"))
        };
        let mut rows = vec![plain("you:")];
        for &g in &t.self_items {
            rows.push(WidgetRow {
                text: name(self, g),
                guid: Some(g),
                kind: RowKind::Item,
            });
        }
        rows.push(plain("them:"));
        for &g in &t.partner_items {
            rows.push(WidgetRow {
                text: name(self, g),
                guid: None,
                kind: RowKind::Plain,
            });
        }
        // The player's not-yet-offered items (an offer's source).
        let offered: std::collections::HashSet<u32> = t.self_items.iter().copied().collect();
        let pack = self
            .items
            .get_contents(self.player_guid)
            .iter()
            .copied()
            .filter(|g| !offered.contains(g))
            .collect::<Vec<_>>();
        if !pack.is_empty() {
            rows.push(plain("pack:"));
            for g in pack {
                if let Some(item) = self.items.get(g) {
                    rows.push(WidgetRow {
                        text: item_text(item),
                        guid: Some(g),
                        kind: RowKind::Item,
                    });
                }
            }
        }
        if t.self_accepted || t.partner_accepted {
            rows.push(plain(format!(
                "accepted: you {}, them {}",
                word(t.self_accepted),
                word(t.partner_accepted)
            )));
        }
        rows
    }

    fn vendor_rows(&self) -> Vec<WidgetRow> {
        let Some(v) = self.vendor.as_ref() else {
            return vec![plain("(no shop)")];
        };
        let mut rows = vec![plain("shop:")];
        for item in &v.items {
            let mut text = item
                .name
                .clone()
                .unwrap_or_else(|| format!("item 0x{:08X}", item.item_guid));
            if item.stack_size > 0 {
                text.push_str(&format!(" x{}", item.stack_size));
            }
            if let Some(price) = v.price(item.item_guid, 1) {
                text.push_str(&format!("  {price}p"));
            }
            rows.push(WidgetRow {
                text,
                guid: Some(item.item_guid),
                kind: RowKind::Shop,
            });
        }
        if !v.lists.buying.is_empty() {
            rows.push(plain("buy:"));
            for r in &v.lists.buying {
                rows.push(WidgetRow {
                    text: staged_text(&r.name, r.quantity, r.price),
                    guid: Some(r.item_guid),
                    kind: RowKind::Buy,
                });
            }
        }
        if !v.lists.selling.is_empty() {
            rows.push(plain("sell:"));
            for r in &v.lists.selling {
                rows.push(WidgetRow {
                    text: staged_text(&r.name, r.quantity, r.price),
                    guid: Some(r.item_guid),
                    kind: RowKind::Sell,
                });
            }
        }
        // The player's items (the sell list's source).
        let mut own: Vec<u32> = self.items.get_contents(self.player_guid).to_vec();
        for o in self.items.get_equipped_by(self.player_guid) {
            own.push(o.object_id());
        }
        if !own.is_empty() {
            rows.push(plain("you:"));
            for g in own {
                if let Some(item) = self.items.get(g) {
                    rows.push(WidgetRow {
                        text: item_text(item),
                        guid: Some(g),
                        kind: RowKind::Item,
                    });
                }
            }
        }
        rows.push(WidgetRow {
            text: v.lists.purse_text.clone(),
            guid: None,
            kind: RowKind::Plain,
        });
        rows
    }

    fn appraise_rows(&self, width: usize) -> Vec<WidgetRow> {
        let Some(a) = self.appraisal.as_ref() else {
            return Vec::new();
        };
        let mut rows = Vec::new();
        if let Some(name) = &a.name {
            rows.push(plain(name.clone()));
        }
        for (name, value) in a.property_lines() {
            // A long line wraps over the widget's width.
            for line in wrap(&format!("{name}: {value}"), width) {
                rows.push(plain(line));
            }
        }
        rows
    }

    /// A click in the widget's region at cell `(col, row)` of a terminal
    /// this `cols` × `rows`: the left button selects the row under the
    /// pointer, the right appraises its item. `None` on the title and keys'
    /// lines and on rows with no item.
    #[must_use]
    pub fn widget_click(
        &mut self,
        cols: usize,
        rows: usize,
        row: u16,
        left: bool,
    ) -> Option<ItemCommand> {
        let r = layout(cols, rows).widget;
        let visible = r.h.saturating_sub(2);
        let y = row as usize;
        if y < r.y + 1 || y >= r.y + 1 + visible {
            return None;
        }
        let index = (y - r.y - 1) + self.widget_scroll;
        let all = self.widget_rows(r.w);
        let target = all.get(index)?;
        let guid = target.guid?;
        self.widget_sel = index;
        if left {
            None
        } else {
            Some(ItemCommand::Appraise(ObjectRef::Guid(guid)))
        }
    }

    /// A wheel notch (±1) over the widget: it scrolls its rows, clamped.
    pub fn widget_wheel(&mut self, delta: i32, cols: usize, rows: usize) {
        let r = layout(cols, rows).widget;
        let visible = r.h.saturating_sub(2);
        let len = self.widget_rows(r.w).len();
        let max = len.saturating_sub(visible);
        let mut scroll = self.widget_scroll;
        if delta > 0 {
            scroll = scroll.saturating_add(delta as usize).min(max);
        } else {
            scroll = scroll.saturating_sub(delta.unsigned_abs() as usize);
        }
        self.widget_scroll = scroll;
    }

    /// A key on the open widget: the navigation keys move the selection
    /// (kept in view); the others return the requests they send (empty when
    /// the key applies to nothing the selection stands on).
    #[must_use]
    pub fn widget_key(&mut self, key: WidgetKey, cols: usize, rows: usize) -> Vec<ItemCommand> {
        let r = layout(cols, rows).widget;
        let visible = r.h.saturating_sub(2).max(1);
        let all = self.widget_rows(r.w);
        let len = all.len();
        if len == 0 {
            return Vec::new();
        }
        self.widget_sel = self.widget_sel.min(len - 1);
        match key {
            WidgetKey::Up => {
                self.widget_sel = self.widget_sel.saturating_sub(1);
            }
            WidgetKey::Down => {
                self.widget_sel = (self.widget_sel + 1).min(len - 1);
            }
            WidgetKey::PageUp => {
                self.widget_sel = self.widget_sel.saturating_sub(visible);
            }
            WidgetKey::PageDown => {
                self.widget_sel = (self.widget_sel + visible).min(len - 1);
            }
            WidgetKey::Home => {
                self.widget_sel = 0;
            }
            WidgetKey::End => {
                self.widget_sel = len - 1;
            }
            _ => {}
        }
        // Keep the selection in view.
        if self.widget_sel < self.widget_scroll {
            self.widget_scroll = self.widget_sel;
        } else if self.widget_sel >= self.widget_scroll + visible {
            self.widget_scroll = self.widget_sel + 1 - visible;
        }
        let selected = all[self.widget_sel].clone();
        self.widget_command(key, selected.kind, selected.guid)
    }

    /// What `key` does to the selected row (the widget's action table).
    fn widget_command(&self, key: WidgetKey, kind: RowKind, guid: Option<u32>) -> Vec<ItemCommand> {
        use ItemCommand as C;
        use RowKind as R;
        match (self.widget, key) {
            // The trade's and the vendor's whole-window keys.
            (Widget::Trade, WidgetKey::Accept) => vec![C::AcceptTrade],
            (Widget::Trade, WidgetKey::Decline) => vec![C::DeclineTrade],
            (Widget::Trade, WidgetKey::Reset) => vec![C::ResetTrade],
            (Widget::Vendor, WidgetKey::BuyAll) => vec![C::Vendor(VendorCommand::BuyAll)],
            (Widget::Vendor, WidgetKey::SellAll) => vec![C::Vendor(VendorCommand::SellAll)],
            (Widget::Container(c), WidgetKey::Add) => self
                .items
                .get_contents(c)
                .iter()
                .copied()
                .map(|g| self.take_command(c, ObjectRef::Guid(g)))
                .collect(),
            // The keys that act on the selected row's item.
            _ => {
                let Some(guid) = guid else {
                    return Vec::new();
                };
                let reference = ObjectRef::Guid(guid);
                match (self.widget, key) {
                    (Widget::Inventory, WidgetKey::Enter) => vec![C::Use(reference)],
                    (Widget::Inventory, WidgetKey::Appraise) => vec![C::Appraise(reference)],
                    (Widget::Inventory, WidgetKey::Drop) => {
                        vec![C::Drop {
                            item: reference,
                            amount: 0,
                        }]
                    }
                    (Widget::Container(c), WidgetKey::Enter) => {
                        vec![self.take_command(c, reference)]
                    }
                    (Widget::Container(_), WidgetKey::Appraise) => vec![C::Appraise(reference)],
                    (Widget::Trade, WidgetKey::Add) if kind == R::Item => {
                        vec![C::AddToTrade(reference)]
                    }
                    (Widget::Vendor, WidgetKey::Enter) if kind == R::Shop => {
                        vec![C::Buy(vec![(reference, 1)])]
                    }
                    (Widget::Vendor, WidgetKey::Add) if kind == R::Shop || kind == R::Buy => {
                        vec![C::Vendor(VendorCommand::AddToBuyList {
                            item: reference,
                            amount: 1,
                        })]
                    }
                    (Widget::Vendor, WidgetKey::Add) if kind == R::Item || kind == R::Sell => {
                        vec![C::Vendor(VendorCommand::AddToSellList(reference))]
                    }
                    (Widget::Vendor, WidgetKey::Back) if kind == R::Buy => {
                        vec![C::Vendor(VendorCommand::RemoveFromBuyList {
                            item: reference,
                            amount: -1,
                        })]
                    }
                    (Widget::Vendor, WidgetKey::Back) if kind == R::Sell => {
                        vec![C::Vendor(VendorCommand::RemoveFromSellList(reference))]
                    }
                    _ => Vec::new(),
                }
            }
        }
    }

    /// Take one item out of the open container `c`: its contents are
    /// picked up, a pack's are moved to the main pack.
    fn take_command(&self, c: u32, item: ObjectRef) -> ItemCommand {
        if self.is_pack(c) {
            ItemCommand::MoveToContainer {
                item,
                container: ObjectRef::Player,
                slot: -1,
                amount: 0,
                join_stack: true,
            }
        } else {
            ItemCommand::PickUp(item)
        }
    }

    /// `c` is one of the player's packs (not a ground container).
    fn is_pack(&self, c: u32) -> bool {
        self.items
            .get(c)
            .is_some_and(|o| o.container_id == self.player_guid)
    }

    /// The open widget's close (its Esc): the requests closing it sends, if
    /// any. An appraise returns to the widget it opened over.
    #[must_use]
    pub fn widget_close(&mut self) -> Vec<ItemCommand> {
        use ItemCommand as C;
        let commands = match self.widget {
            Widget::None | Widget::Inventory | Widget::Appraise => Vec::new(),
            Widget::Container(g) if self.is_pack(g) => {
                self.open_pack = 0;
                vec![C::SetOpenPack(0)]
            }
            Widget::Container(g) => vec![C::StopViewing(ObjectRef::Guid(g))],
            Widget::Trade => vec![C::CloseTrade],
            Widget::Vendor => vec![C::Vendor(VendorCommand::Close)],
        };
        let under = if self.widget == Widget::Appraise {
            self.appraise_under.take()
        } else {
            None
        };
        self.appraise_under = None;
        self.widget = under.unwrap_or(Widget::None);
        self.widget_sel = 0;
        self.widget_scroll = 0;
        commands
    }

    /// The inventory's key: open the inventory widget, or close it when it
    /// is the one open.
    pub fn widget_toggle_inventory(&mut self) {
        if self.widget == Widget::Inventory {
            self.widget = Widget::None;
        } else {
            self.open_widget(Widget::Inventory);
        }
        self.widget_sel = 0;
        self.widget_scroll = 0;
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
    // The item widget, over the world (it paints last: it is open over the
    // rest).
    if m.widget_open() {
        draw_widget(g, m, l.widget);
    }
}

/// The open widget over the world: its title, its rows (the selection
/// marked, the wheel's window) and its keys' line — plain text, no frame.
fn draw_widget(g: &mut Grid, m: &Model, r: Rect) {
    let title = m.widget_title();
    let rows = m.widget_rows(r.w);
    let visible = r.h.saturating_sub(2);
    let len = rows.len();
    let scroll = m.widget_scroll.min(len.saturating_sub(visible));
    g.put_str(r.x, r.y, &title, TITLE, BG);
    // The wheel's window over a long list.
    if len > visible && visible > 0 {
        let last = (scroll + visible).min(len);
        g.put_str(
            r.x + title.chars().count(),
            r.y,
            &format!("  [{scroll}-{last} of {len}]"),
            DIM,
            BG,
        );
    }
    for (i, row) in rows.iter().skip(scroll).take(visible).enumerate() {
        let selected = i + scroll == m.widget_sel && row.kind != RowKind::Plain;
        let fg = if selected {
            GOLD
        } else if row.kind == RowKind::Plain {
            DIM
        } else {
            TEXT
        };
        let mark = if selected { ">" } else { " " };
        g.put_str(r.x, r.y + 1 + i, &format!("{mark} {}", row.text), fg, BG);
    }
    g.put_str(r.x, r.y + r.h - 1, m.widget_footer(), DIM, BG);
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
            widget: d,
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
// -- The item widgets --------------------------------------------------------

#[cfg(test)]
mod widget_tests {
    use super::*;
    use std::collections::BTreeMap;

    use ac_core::items::{VendorShopItem, VendorShopProfile};
    use ac_proto::appraise::IdentifyObjectResponse;
    use ac_runtime::inventory::{StagedRow, TradeView, VendorCommand, VendorListsView, VendorView};
    use ac_runtime::session::interface::{InterfaceState, ItemsUpdate};

    const PLAYER: u32 = 900;

    /// One item in the player's main pack.
    fn item(id: u32, name: &str) -> ClientObject {
        let mut o = ClientObject::new(id);
        o.name = name.into();
        o.container_id = PLAYER;
        o
    }

    /// The model: `ids` in the main pack, the widget open.
    fn model(ids: &[u32], widget: Widget) -> Model {
        let mut m = Model::new();
        let objects: Vec<ClientObject> =
            ids.iter().map(|i| item(*i, &format!("item {i}"))).collect();
        m.apply(InterfaceEvent::Items(ItemsUpdate {
            reset: true,
            objects,
            removed: Vec::new(),
            contents: vec![(PLAYER, Some(ids.to_vec()))],
        }));
        m.player_guid = PLAYER;
        m.widget = widget;
        m
    }

    /// The widget's rows as (text, kind, guid).
    fn rows_of(m: &Model) -> Vec<(String, RowKind, Option<u32>)> {
        let r = layout(100, 30).widget;
        m.widget_rows(r.w)
            .into_iter()
            .map(|row| (row.text, row.kind, row.guid))
            .collect()
    }

    /// The widget is a quarter of the width and three quarters of the
    /// height (with the floors) and sits just past the vitals when the
    /// terminal is wide enough for both.
    #[test]
    fn the_widget_is_a_quarter_of_the_terminal() {
        // (cols, rows): (widget width, widget height, x)
        for (cols, rows, w, h, x) in [
            (100, 30, 25, 22, 18),
            (80, 24, 20, 18, 18),
            (40, 12, 14, 10, 18),
            (30, 12, 14, 10, 1),
        ] {
            let r = layout(cols, rows).widget;
            assert_eq!((r.w, r.h, r.x), (w, h, x), "the widget at {cols} x {rows}");
        }
    }

    /// The inventory's rows: the pack's items, a pack marked with its
    /// contents' count, and the equipped's section.
    #[test]
    fn the_inventory_lists_the_pack_the_packs_and_the_equipped() {
        let mut m = model(&[1, 2, 3, 4], Widget::Inventory);
        let mut pack = item(3, "a pack");
        pack.container_type_hint = 1;
        let mut stacked = item(2, "potion");
        stacked.plural_name = "potions".into();
        stacked.stack_size = 5;
        m.apply(InterfaceEvent::Items(ItemsUpdate {
            reset: false,
            objects: vec![pack, stacked],
            removed: Vec::new(),
            contents: vec![(PLAYER, Some(vec![1, 2, 3, 4])), (3, Some(vec![7]))],
        }));
        let rows = rows_of(&m);
        assert_eq!(
            rows.iter()
                .map(|(t, k, _)| (t.as_str(), k))
                .collect::<Vec<_>>(),
            vec![
                ("item 1", &RowKind::Item),
                ("potions x5", &RowKind::Item),
                ("a pack (1 items)", &RowKind::Pack),
                ("item 4", &RowKind::Item),
            ],
            "the pack's rows"
        );
        // The equipped's section: a weapon in the main hand.
        let mut sword = item(5, "a sword");
        sword.wielder_id = PLAYER;
        sword.currently_equipped_location = ac_core::items::EquipMask::TWO_HANDED;
        m.apply(InterfaceEvent::Items(ItemsUpdate {
            reset: false,
            objects: vec![sword],
            removed: Vec::new(),
            contents: vec![],
        }));
        let rows = rows_of(&m);
        assert_eq!(
            rows.last().map(|r| (r.0.as_str(), r.1, r.2)),
            Some(("a sword", RowKind::Item, Some(5))),
            "the equipped's section: {rows:?}"
        );
    }

    /// The wheel scrolls the rows, clamped at the top and the bottom.
    #[test]
    fn the_wheel_scrolls_the_widget_clamped() {
        let ids: Vec<u32> = (1..=25).collect();
        let mut m = model(&ids, Widget::Inventory);
        // At 100 x 30 the widget shows 20 rows: the 25 rows scroll 0..=5.
        m.widget_wheel(1, 100, 30);
        assert_eq!(m.widget_scroll, 1, "one notch down");
        m.widget_wheel(100, 100, 30);
        assert_eq!(m.widget_scroll, 5, "clamped at the bottom");
        m.widget_wheel(-100, 100, 30);
        assert_eq!(m.widget_scroll, 0, "clamped at the top");
    }

    /// A click selects the row under the pointer (the left button) and a
    /// right click appraises its item; the title's and keys' rows do
    /// nothing.
    #[test]
    fn a_click_selects_and_appraises_the_row_under_the_pointer() {
        let mut m = model(&[1, 2, 3], Widget::Inventory);
        // The widget at 100 x 30: x 18, y 1, w 25, h 22 — the rows run
        // from y 2.
        assert!(
            m.widget_click(100, 30, 1, true).is_none(),
            "the title's row"
        );
        assert!(
            m.widget_click(100, 30, 1, false).is_none(),
            "a right click on the title's row"
        );
        assert!(m.widget_click(100, 30, 23, true).is_none(), "the keys' row");
        assert!(m.widget_click(100, 30, 5, true).is_none(), "past the rows");
        let _ = m.widget_click(100, 30, 4, true);
        assert_eq!(m.widget_sel, 2, "a left click selects the row");
        assert_eq!(
            m.widget_click(100, 30, 4, false),
            Some(ItemCommand::Appraise(ObjectRef::Guid(3))),
            "a right click appraises"
        );
    }

    /// The keys' table: what each key sends to the session, per widget.
    #[test]
    fn the_widget_keys_send_their_commands() {
        use ItemCommand as C;
        use WidgetKey as K;

        // Inventory: Enter uses, e appraises, d drops.
        let mut m = model(&[1, 2], Widget::Inventory);
        assert_eq!(
            m.widget_key(K::Enter, 100, 30),
            vec![C::Use(ObjectRef::Guid(1))]
        );
        assert_eq!(
            m.widget_key(K::Appraise, 100, 30),
            vec![C::Appraise(ObjectRef::Guid(1))]
        );
        let _ = m.widget_key(K::Down, 100, 30);
        assert_eq!(
            m.widget_key(K::Drop, 100, 30),
            vec![C::Drop {
                item: ObjectRef::Guid(2),
                amount: 0
            }]
        );

        // A ground container: Enter picks its selected item up, a takes
        // everything.
        let mut m = model(&[], Widget::Container(500));
        let mut a = ClientObject::new(501);
        a.name = "a coin".into();
        let mut b = ClientObject::new(502);
        b.name = "a key".into();
        m.apply(InterfaceEvent::Items(ItemsUpdate {
            reset: false,
            objects: vec![a, b],
            removed: Vec::new(),
            contents: vec![(500, Some(vec![501, 502]))],
        }));
        assert_eq!(
            m.widget_key(K::Enter, 100, 30),
            vec![C::PickUp(ObjectRef::Guid(501))]
        );
        assert_eq!(
            m.widget_key(K::Add, 100, 30),
            vec![
                C::PickUp(ObjectRef::Guid(501)),
                C::PickUp(ObjectRef::Guid(502))
            ],
            "a takes everything"
        );

        // A carried pack: Enter moves its item to the main pack.
        let mut pack = ClientObject::new(600);
        pack.name = "a pack".into();
        pack.container_id = PLAYER;
        let mut held = ClientObject::new(601);
        held.name = "a bolt".into();
        let mut m = Model::new();
        m.apply(InterfaceEvent::Items(ItemsUpdate {
            reset: true,
            objects: vec![pack, held],
            removed: Vec::new(),
            contents: vec![(PLAYER, Some(vec![600])), (600, Some(vec![601]))],
        }));
        m.player_guid = PLAYER;
        m.widget = Widget::Container(600);
        assert_eq!(
            m.widget_key(K::Enter, 100, 30),
            vec![C::MoveToContainer {
                item: ObjectRef::Guid(601),
                container: ObjectRef::Player,
                slot: -1,
                amount: 0,
                join_stack: true
            }]
        );

        // Navigation: the selection moves and stays in view.
        let mut m = model(&[1, 2, 3, 4, 5, 6, 7, 8, 9], Widget::Inventory);
        let _ = m.widget_key(K::End, 100, 30);
        assert_eq!(m.widget_sel, 8, "End");
        let _ = m.widget_key(K::Home, 100, 30);
        assert_eq!(m.widget_sel, 0, "Home");
        let _ = m.widget_key(K::Down, 100, 30);
        assert_eq!(m.widget_sel, 1, "Down");
        let _ = m.widget_key(K::Up, 100, 30);
        let _ = m.widget_key(K::Up, 100, 30);
        assert_eq!(m.widget_sel, 0, "Up, clamped");

        // A trade: a offers the selected item, y/n/r decide.
        let mut m = model(&[1, 2, 3], Widget::Trade);
        m.trade = TradeView {
            open: true,
            partner_guid: 77,
            partner_name: Some("the partner".into()),
            self_items: Vec::new(),
            partner_items: vec![30],
            ..TradeView::default()
        };
        let rows = rows_of(&m);
        let pack = rows.iter().position(|(t, _, _)| t == "pack:").unwrap();
        m.widget_sel = pack + 1; // 0x1, from the pack
        assert_eq!(
            m.widget_key(K::Add, 100, 30),
            vec![C::AddToTrade(ObjectRef::Guid(1))]
        );
        assert_eq!(m.widget_key(K::Accept, 100, 30), vec![C::AcceptTrade]);
        assert_eq!(m.widget_key(K::Decline, 100, 30), vec![C::DeclineTrade]);
        assert_eq!(m.widget_key(K::Reset, 100, 30), vec![C::ResetTrade]);
        // A partner's item takes no key: it has no guid.
        let them = rows.iter().position(|(t, _, _)| t == "them:").unwrap();
        m.widget_sel = them + 1;
        assert!(
            m.widget_key(K::Add, 100, 30).is_empty(),
            "the partner's rows"
        );

        // A vendor: Enter buys one, a lists, b/s take the lot, Backspace
        // unlists.
        let mut m = model(&[1], Widget::Vendor);
        m.vendor = Some(VendorView {
            vendor_guid: 88,
            name: Some("the shop".into()),
            profile: VendorShopProfile {
                sell_price: 2.0,
                ..VendorShopProfile::default()
            },
            items: vec![VendorShopItem {
                item_guid: 21,
                name: Some("an apple".into()),
                stack_size: 10,
                value: Some(10),
                desc_stack_size: Some(1),
                ..VendorShopItem::default()
            }],
            lists: VendorListsView {
                buying: vec![StagedRow {
                    item_guid: 21,
                    name: "an apple".into(),
                    quantity: 3,
                    price: 60,
                }],
                purse_text: "You have 500p".into(),
                ..VendorListsView::default()
            },
            reason: ac_runtime::inventory::VendorViewReason::Opened,
        });
        let rows = rows_of(&m);
        let shop_row = rows.iter().position(|r| r.1 == RowKind::Shop).unwrap();
        assert_eq!(
            rows[shop_row].0, "an apple x10  20p",
            "the listing's stock and price"
        );
        m.widget_sel = shop_row;
        assert_eq!(
            m.widget_key(K::Enter, 100, 30),
            vec![C::Buy(vec![(ObjectRef::Guid(21), 1)])]
        );
        assert_eq!(
            m.widget_key(K::Add, 100, 30),
            vec![C::Vendor(VendorCommand::AddToBuyList {
                item: ObjectRef::Guid(21),
                amount: 1
            })]
        );
        assert_eq!(
            m.widget_key(K::BuyAll, 100, 30),
            vec![C::Vendor(VendorCommand::BuyAll)]
        );
        assert_eq!(
            m.widget_key(K::SellAll, 100, 30),
            vec![C::Vendor(VendorCommand::SellAll)]
        );
        let buy_row = rows.iter().position(|r| r.1 == RowKind::Buy).unwrap();
        m.widget_sel = buy_row;
        assert_eq!(
            m.widget_key(K::Back, 100, 30),
            vec![C::Vendor(VendorCommand::RemoveFromBuyList {
                item: ObjectRef::Guid(21),
                amount: -1
            })]
        );
        // Selling: a on one's own item.
        let you_row = rows.iter().position(|r| r.0 == "you:").unwrap() + 1;
        m.widget_sel = you_row;
        assert_eq!(
            m.widget_key(K::Add, 100, 30),
            vec![C::Vendor(VendorCommand::AddToSellList(ObjectRef::Guid(1)))]
        );
    }

    /// The close (its Esc): what each widget sends on the way out.
    #[test]
    fn the_close_sends_its_widget_s_requests() {
        use ItemCommand as C;

        let mut m = model(&[1], Widget::Trade);
        assert_eq!(m.widget_close(), vec![C::CloseTrade]);
        assert!(!m.widget_open());

        // A ground container asks to stop looking into it; a carried pack
        // closes on its own (no request the table needs).
        let mut m = model(&[], Widget::Container(500));
        assert_eq!(m.widget_close(), vec![C::StopViewing(ObjectRef::Guid(500))]);
        let mut m = model(&[600], Widget::Container(600));
        assert_eq!(m.widget_close(), vec![C::SetOpenPack(0)], "a pack's close");

        let mut m = model(&[1], Widget::Vendor);
        assert_eq!(m.widget_close(), vec![C::Vendor(VendorCommand::Close)]);

        // An appraise returns to the widget it opened over, with no
        // request.
        let mut m = model(&[1], Widget::Inventory);
        m.appraise_under = Some(Widget::Inventory);
        m.widget = Widget::Appraise;
        assert!(m.widget_close().is_empty(), "an appraise sends nothing");
        assert_eq!(m.widget, Widget::Inventory, "it returns to the inventory");

        // The inventory and nothing open send nothing.
        let mut m = model(&[1], Widget::Inventory);
        assert!(m.widget_close().is_empty());
        let mut m = Model::new();
        assert!(m.widget_close().is_empty(), "nothing open");
    }

    /// The server's events open and close the widgets: the vendor, the
    /// trade, the containers, the state's outside container, and the
    /// appraisal.
    #[test]
    fn the_servers_events_open_and_close_the_widgets() {
        use InterfaceEvent as E;
        let mut m = model(&[1], Widget::None);

        // The vendor: Some opens it, None closes it.
        let view = || {
            Box::new(VendorView {
                vendor_guid: 88,
                name: Some("the shop".into()),
                profile: VendorShopProfile::default(),
                items: Vec::new(),
                lists: VendorListsView::default(),
                reason: ac_runtime::inventory::VendorViewReason::Opened,
            })
        };
        m.apply(E::Vendor(Some(view())));
        assert!(matches!(m.widget, Widget::Vendor), "the vendor opens");
        m.apply(E::Vendor(None));
        assert!(!m.widget_open(), "the vendor closes");

        // The trade: open opens, closed closes (and only the trade's).
        let trade = |open: bool| {
            Box::new(TradeView {
                open,
                partner_guid: 77,
                ..TradeView::default()
            })
        };
        m.apply(E::Trade(trade(true)));
        assert!(matches!(m.widget, Widget::Trade), "the trade opens");
        m.apply(E::Trade(trade(false)));
        assert!(!m.widget_open(), "the trade closes");

        // A carried pack opens its container; an item reset empties it.
        m.apply(E::OpenContainedContainer(300));
        assert!(matches!(m.widget, Widget::Container(300)), "the pack opens");
        m.apply(E::Items(ItemsUpdate {
            reset: true,
            objects: Vec::new(),
            removed: Vec::new(),
            contents: Vec::new(),
        }));
        assert!(!m.widget_open(), "a reset closes the container");

        // The state's outside container: non-zero opens, zero closes (a
        // pack's container stays).
        let state = |external: u32| {
            Box::new(InterfaceState {
                external_container: external,
                ..InterfaceState::default()
            })
        };
        m.apply(E::State(state(400)));
        assert!(
            matches!(m.widget, Widget::Container(400)),
            "the container opens"
        );
        m.apply(E::OpenContainedContainer(300));
        m.apply(E::State(state(0)));
        assert!(
            matches!(m.widget, Widget::Container(300)),
            "a pack's container is not the outside one"
        );
        m.apply(E::State(state(500)));
        assert!(
            matches!(m.widget, Widget::Container(500)),
            "the pack's closes"
        );

        // The appraisal: opens over the current widget; the close returns.
        m.apply(E::Items(ItemsUpdate {
            reset: true,
            objects: vec![item(1, "an item")],
            removed: Vec::new(),
            contents: vec![(PLAYER, Some(vec![1]))],
        }));
        m.player_guid = PLAYER;
        m.widget = Widget::Inventory;
        let appraisal = Box::new(AppraisalView {
            name: Some("an item".into()),
            response: IdentifyObjectResponse {
                properties: ac_proto::PropertyTables {
                    ints: BTreeMap::from([(1, 5)]),
                    ..ac_proto::PropertyTables::default()
                },
                ..IdentifyObjectResponse::default()
            },
            present_in_ui: true,
            opens_window: true,
        });
        m.apply(E::Appraisal(appraisal));
        assert!(matches!(m.widget, Widget::Appraise), "the appraise opens");
        let rows = rows_of(&m);
        assert!(
            rows.iter().any(|(t, _, _)| t == "an item"),
            "the appraise's name: {rows:?}"
        );
        let _ = m.widget_close();
        assert!(
            matches!(m.widget, Widget::Inventory),
            "it returns to the inventory"
        );
    }
}

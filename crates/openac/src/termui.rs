//! The terminal's 2-D interface: a primitive layer drawn straight into the
//! cell grid with box-drawing edges, Nerd Font glyph item icons and UTF-8
//! text. It reads the scene's semantic facts — vitals, chat, the interface
//! panels — and never renders the game's widget tree through the rasterizer.
//!
//! The 3-D world renders into its own viewport on the left; the always-on HUD
//! (vitals, target, minimap, chat) paints over the viewport's edges; and the
//! open content panel (inventory, vendor, trade, ...) takes a bordered column
//! on the right, the world shrinking to make room.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use ac_core::items::{ClientObjectTable, ItemType};
use ac_render::scene::Scene;
use ac_runtime::communication::{ChatLine, JournalView};
use ac_runtime::gameplay::{BookView, ChessView, HouseView};
use ac_runtime::inventory::{AppraisalView, HousePurchaseView, TradeView, VendorView};
use ac_runtime::session::interface::{
    ContractFacts, InterfaceEvent, SocialFacts, TitleFacts, apply_items_update,
};
use ac_runtime::world::PlayerView;

use crate::swrender::Cell;

// -- Palette ---------------------------------------------------------------

type Rgb = [u8; 3];
const BG: Rgb = [13, 15, 21];
const BORDER: Rgb = [108, 128, 150];
const TEXT: Rgb = [212, 216, 226];
const DIM: Rgb = [138, 146, 158];
const TITLE: Rgb = [240, 238, 208];
const GOLD: Rgb = [228, 188, 96];
const HP: Rgb = [206, 66, 58];
const STAMINA: Rgb = [92, 176, 100];
const MANA: Rgb = [96, 130, 220];
const TARGET: Rgb = [232, 120, 120];

/// A content panel stays in the right column this long after its last update
/// before the world reclaims the space; touching it again refreshes the clock.
const PANEL_TTL: Duration = Duration::from_secs(20);

// -- Geometry ---------------------------------------------------------------

/// An inclusive-corner rectangle in cells.
#[derive(Clone, Copy, Debug)]
pub struct Rect {
    pub x: usize,
    pub y: usize,
    pub w: usize,
    pub h: usize,
}

/// Where the 3-D viewport and the HUD/panel regions sit this frame.
pub struct Layout {
    /// The 3-D world's area (the left of the grid, or all of it when no panel
    /// is open).
    pub viewport: Rect,
    /// The right-hand content column, present only while a panel is open.
    pub panel: Option<Rect>,
    /// The vitals block, over the viewport's top-left.
    pub vitals: Rect,
    /// The minimap, over the viewport's top-right.
    pub minimap: Rect,
    /// The chat feed, over the viewport's bottom.
    pub chat: Rect,
}

/// The minimap's side, in cells (an `n`×`n` interior plus its border).
pub const MINIMAP_N: usize = 8;

/// Compute the frame's regions. When `panel_open`, a column of roughly a third
/// of the width is reserved on the right and the viewport shrinks to fill the
/// rest; otherwise the viewport is the whole grid.
pub fn layout(cols: usize, rows: usize, panel_open: bool) -> Layout {
    let cols = cols.max(30);
    let rows = rows.max(12);
    let (pw, view_w) = if panel_open {
        let pw = (cols / 3).clamp(28, cols / 2);
        (pw, cols - pw)
    } else {
        (0, cols)
    };

    let vitals_w = 26.min(view_w);
    let mm_w = (MINIMAP_N + 2).min(view_w);
    let mm_h = (MINIMAP_N + 2).min(rows);
    let chat_h = 4.min(rows);

    let viewport = Rect {
        x: 0,
        y: 0,
        w: view_w,
        h: rows,
    };
    let panel = (pw > 0).then_some(Rect {
        x: view_w,
        y: 0,
        w: pw,
        h: rows,
    });
    let vitals = Rect {
        x: 1,
        y: 1,
        w: vitals_w,
        h: 5,
    };
    let minimap = Rect {
        x: view_w.saturating_sub(mm_w) + 1,
        y: 1,
        w: mm_w,
        h: mm_h,
    };
    let chat = Rect {
        x: 1,
        y: rows.saturating_sub(chat_h + 1),
        w: view_w - 2,
        h: chat_h,
    };

    Layout {
        viewport,
        panel,
        vitals,
        minimap,
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

    /// A box-drawing border around `r` (`r` includes the border).
    pub fn border(&mut self, r: Rect, fg: Rgb, bg: Rgb) {
        if r.w < 2 || r.h < 2 {
            return;
        }
        let (l, t) = (r.x, r.y);
        let (rr, b) = (r.x + r.w - 1, r.y + r.h - 1);
        self.put(l, t, '┌', fg, bg);
        self.put(rr, t, '┐', fg, bg);
        self.put(l, b, '└', fg, bg);
        self.put(rr, b, '┘', fg, bg);
        for x in (l + 1)..rr {
            self.put(x, t, '─', fg, bg);
            self.put(x, b, '─', fg, bg);
        }
        for y in (t + 1)..b {
            self.put(l, y, '│', fg, bg);
            self.put(rr, y, '│', fg, bg);
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

/// Wrap a line into chunks of at most `n` characters, breaking words at the
/// space before the limit.
fn wrap(s: &str, n: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut line = String::new();
    for word in s.split_whitespace() {
        if line.is_empty() {
            line = truncate(word, n);
        } else if line.chars().count() + 1 + word.chars().count() <= n {
            line.push(' ');
            line.push_str(word);
        } else {
            out.push(std::mem::take(&mut line));
            line = truncate(word, n);
        }
    }
    if !line.is_empty() {
        out.push(line);
    }
    out
}

/// A shade character for a 0..1 height fraction (the minimap's relief).
fn shade_char(frac: f32) -> char {
    const S: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    let i = ((frac.clamp(0.0, 0.999) * 8.0) as usize).min(7);
    S[i]
}

/// A Nerd Font glyph for an item's kind: the closest single-width symbol for
/// the item's type bits.
fn icon_for(t: ItemType) -> char {
    use ItemType as T;
    let raw = t.bits();
    let has = |b: u32| raw & b != 0;
    if has(T::MONEY.bits()) {
        '₡'
    } else if has(T::KEY.bits()) {
        '⚷'
    } else if has(T::WEAPON.bits()) {
        '⚔'
    } else if has(T::LIFE_STONE.bits()) {
        '♥'
    } else if has(T::ARMOR.bits()) {
        '⛨'
    } else if has(T::GEM.bits()) {
        '◆'
    } else if has(T::CLOTHING.bits()) {
        '✦'
    } else if has(T::CONTAINER.bits()) {
        '▣'
    } else if has(T::CASTER.bits()) {
        '✧'
    } else if has(T::FOOD.bits()) {
        '✿'
    } else if has(T::SPELL_COMPONENTS.bits()) {
        '⚗'
    } else if has(T::WRITABLE.bits()) {
        '✎'
    } else {
        '▪'
    }
}

// -- The semantic model ----------------------------------------------------

/// Which content panel the right column shows.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PanelKind {
    Container,
    Vendor,
    Trade,
    Appraisal,
    Book,
    Journal,
    House,
    HousePurchase,
    Salvage,
    Chess,
    Contracts,
    Social,
    Titles,
}

/// The last-updated clock per content panel; the most recently active open
/// panel wins the column.
#[derive(Default)]
struct ActiveState {
    container: Option<Instant>,
    vendor: Option<Instant>,
    trade: Option<Instant>,
    appraisal: Option<Instant>,
    book: Option<Instant>,
    journal: Option<Instant>,
    house: Option<Instant>,
    house_purchase: Option<Instant>,
    salvage: Option<Instant>,
    chess: Option<Instant>,
    contracts: Option<Instant>,
    social: Option<Instant>,
    titles: Option<Instant>,
}

/// The 2-D interface's copy of the scene's facts, kept in step each frame.
#[derive(Default)]
pub struct Model {
    items: ClientObjectTable,
    vitals: Option<PlayerView>,
    char_name: Option<String>,
    target_guid: Option<u32>,
    target_name: Option<String>,
    target_hp: Option<f32>,
    chat: VecDeque<String>,
    status: VecDeque<String>,
    player_guid: u32,
    // The content panels, each `Some` when it has data.
    vendor: Option<VendorView>,
    trade: Option<TradeView>,
    appraisal: Option<AppraisalView>,
    book: Option<BookView>,
    journal: Option<JournalView>,
    house: Option<HouseView>,
    house_purchase: Option<HousePurchaseView>,
    contracts: Option<ContractFacts>,
    social: Option<SocialFacts>,
    titles: Option<TitleFacts>,
    chess: Option<ChessView>,
    inventory_window: Option<u32>,
    external_container: u32,
    salvage_guid: Option<u32>,
    active: ActiveState,
}

impl Model {
    pub fn new() -> Self {
        Self::default()
    }

    /// Fold the scene into the model: the vitals and target, the drained chat
    /// and status, and the interface events (updating the panels).
    pub fn update(&mut self, scene: &mut Scene, now: Instant) {
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
            self.apply(ev, now);
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

    fn apply(&mut self, ev: InterfaceEvent, now: Instant) {
        use InterfaceEvent as E;
        match ev {
            E::Items(u) => {
                apply_items_update(&mut self.items, &u);
            }
            E::Vendor(v) => {
                self.vendor = v.map(|b| *b);
                if self.vendor.is_some() {
                    self.active.vendor = Some(now);
                }
            }
            E::Trade(t) => {
                self.trade = Some(*t);
                if self.trade.as_ref().is_some_and(|t| t.open) {
                    self.active.trade = Some(now);
                }
            }
            E::Appraisal(a) => {
                self.appraisal = Some(*a);
                self.active.appraisal = Some(now);
            }
            E::Book(b) => {
                let b = *b;
                if b.book_guid != 0 {
                    self.active.book = Some(now);
                }
                self.book = Some(b);
            }
            E::Journal(j) => {
                let j = *j;
                if !j.pages.is_empty() {
                    self.active.journal = Some(now);
                }
                self.journal = Some(j);
            }
            E::House(h) => {
                self.house = Some(*h);
                if self
                    .house
                    .as_ref()
                    .is_some_and(|h| h.owns_house || !h.lines.is_empty())
                {
                    self.active.house = Some(now);
                }
            }
            E::HousePurchase(h) => {
                let h = *h;
                if h.is_open() {
                    self.active.house_purchase = Some(now);
                }
                self.house_purchase = Some(h);
            }
            E::OpenSalvage(g) => {
                self.salvage_guid = Some(g);
                self.active.salvage = Some(now);
            }
            E::Chess(c) => {
                let c = *c;
                if c.board_guid != 0 {
                    self.active.chess = Some(now);
                }
                self.chess = Some(c);
            }
            E::OpenContainedContainer(c) => {
                self.inventory_window = Some(c);
                self.active.container = Some(now);
            }
            E::State(s) => {
                self.external_container = s.external_container;
                self.target_hp = s
                    .selected_health
                    .filter(|h| self.target_guid.is_some_and(|g| g == h.object_id))
                    .and_then(|h| h.percent);
                if s.external_container != 0 {
                    self.active.container = Some(now);
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
            E::Contracts(c) => {
                self.contracts = Some(*c);
            }
            E::Social(s) => {
                self.social = Some(*s);
            }
            E::Titles(t) => {
                self.titles = Some(*t);
            }
            E::Player(_) | E::ItemMana(_) | E::PluginUi(_) | E::BeginGame(_) | E::ChatEntry(_) => {
                // The spellbook, item mana, plugin UI, game board and the chat
                // entry: no panel of their own in this front end yet.
            }
        }
    }

    /// A panel's open state.
    fn is_open(&self, k: PanelKind) -> bool {
        match k {
            PanelKind::Container => self.external_container != 0 || self.inventory_window.is_some(),
            PanelKind::Vendor => self.vendor.is_some(),
            PanelKind::Trade => self.trade.as_ref().is_some_and(|t| t.open),
            PanelKind::Appraisal => self.appraisal.is_some(),
            PanelKind::Book => self.book.as_ref().is_some_and(|b| b.book_guid != 0),
            PanelKind::Journal => self.journal.as_ref().is_some_and(|j| !j.pages.is_empty()),
            PanelKind::House => self
                .house
                .as_ref()
                .is_some_and(|h| h.owns_house || !h.lines.is_empty()),
            PanelKind::HousePurchase => self.house_purchase.as_ref().is_some_and(|h| h.is_open()),
            PanelKind::Salvage => self.salvage_guid.is_some(),
            PanelKind::Chess => self.chess.as_ref().is_some_and(|c| c.board_guid != 0),
            PanelKind::Contracts => self
                .contracts
                .as_ref()
                .is_some_and(|c| !c.contracts.is_empty()),
            PanelKind::Social => self.social.as_ref().is_some_and(|s| s.in_world),
            PanelKind::Titles => self
                .titles
                .as_ref()
                .is_some_and(|t| !t.title_ids.is_empty()),
        }
    }

    fn last_active(&self, k: PanelKind) -> Option<Instant> {
        match k {
            PanelKind::Container => self.active.container,
            PanelKind::Vendor => self.active.vendor,
            PanelKind::Trade => self.active.trade,
            PanelKind::Appraisal => self.active.appraisal,
            PanelKind::Book => self.active.book,
            PanelKind::Journal => self.active.journal,
            PanelKind::House => self.active.house,
            PanelKind::HousePurchase => self.active.house_purchase,
            PanelKind::Salvage => self.active.salvage,
            PanelKind::Chess => self.active.chess,
            PanelKind::Contracts => self.active.contracts,
            PanelKind::Social => self.active.social,
            PanelKind::Titles => self.active.titles,
        }
    }

    /// The content panel for the right column, `None` when none is open and
    /// fresh.
    pub fn active_panel(&self, now: Instant) -> Option<PanelKind> {
        const ORDER: [PanelKind; 13] = [
            PanelKind::Container,
            PanelKind::Vendor,
            PanelKind::Trade,
            PanelKind::Appraisal,
            PanelKind::Book,
            PanelKind::Journal,
            PanelKind::House,
            PanelKind::HousePurchase,
            PanelKind::Salvage,
            PanelKind::Chess,
            PanelKind::Contracts,
            PanelKind::Social,
            PanelKind::Titles,
        ];
        let mut best: Option<(Instant, usize, PanelKind)> = None;
        for (i, k) in ORDER.iter().enumerate() {
            if !self.is_open(*k) {
                continue;
            }
            let Some(t) = self.last_active(*k) else {
                continue;
            };
            if now.duration_since(t) >= PANEL_TTL {
                continue;
            }
            let better = match &best {
                Some((bt, bi, _)) => t > *bt || (t == *bt && i < *bi),
                None => true,
            };
            if better {
                best = Some((t, i, *k));
            }
        }
        best.map(|(_, _, k)| k)
    }

    /// Whether the right column is in use (the viewport shrinks).
    pub fn panel_open(&self, now: Instant) -> bool {
        self.active_panel(now).is_some()
    }

    // -- Panel building (each view distilled to its basic rows) ------------

    fn build_panel(&self, k: PanelKind) -> Panel {
        match k {
            PanelKind::Container => self.container_panel(),
            PanelKind::Vendor => self.vendor_panel(),
            PanelKind::Trade => self.trade_panel(),
            PanelKind::Appraisal => self.appraisal_panel(),
            PanelKind::Book => self.book_panel(),
            PanelKind::Journal => self.journal_panel(),
            PanelKind::House => self.house_panel(),
            PanelKind::HousePurchase => self.house_purchase_panel(),
            PanelKind::Salvage => self.salvage_panel(),
            PanelKind::Chess => self.chess_panel(),
            PanelKind::Contracts => self.contracts_panel(),
            PanelKind::Social => self.social_panel(),
            PanelKind::Titles => self.titles_panel(),
        }
    }

    fn item_row(&self, guid: u32) -> Option<Row> {
        let o = self.items.get(guid)?;
        let meta = (o.stack_size_max > 1).then(|| format!("x{}", o.stack_size));
        Some(Row::Item {
            icon: icon_for(o.item_type),
            name: o.name.clone(),
            meta,
        })
    }

    fn container_panel(&self) -> Panel {
        let (container, title) = if self.external_container != 0 {
            (
                self.external_container,
                self.name_or(self.external_container, "Container"),
            )
        } else if let Some(c) = self.inventory_window {
            (
                c,
                if c == self.player_guid {
                    "Inventory".into()
                } else {
                    "Pack".into()
                },
            )
        } else {
            return Panel {
                title: "Container".into(),
                rows: vec![Row::Text("(none)".into())],
            };
        };

        let mut rows = Vec::new();
        let mut value = 0i32;
        let guids = self.items.get_contents(container);
        for &g in guids {
            let Some(o) = self.items.get(g) else {
                continue;
            };
            value = value.saturating_add(o.value);
            rows.push(self.item_row(g).unwrap_or_else(|| Row::Item {
                icon: '▪',
                name: format!("#{g:x}"),
                meta: None,
            }));
        }
        if rows.is_empty() {
            rows.push(Row::Text("(empty)".into()));
        }
        rows.push(Row::Blank);
        rows.push(Row::Text(format!("value {value} pyreals")));
        Panel { title, rows }
    }

    fn name_or(&self, guid: u32, fallback: &str) -> String {
        self.items
            .get(guid)
            .map(|o| o.name.clone())
            .unwrap_or_else(|| fallback.to_string())
    }

    fn vendor_panel(&self) -> Panel {
        let Some(v) = &self.vendor else {
            return Panel {
                title: "Vendor".into(),
                rows: Vec::new(),
            };
        };
        let mut rows = Vec::new();
        rows.push(Row::Section("For sale".into()));
        for it in &v.items {
            let (name, icon) = self
                .items
                .get(it.item_guid)
                .map(|o| (o.name.clone(), icon_for(o.item_type)))
                .unwrap_or_else(|| (format!("#{:x}", it.item_guid), '▪'));
            rows.push(Row::Item {
                icon,
                name,
                meta: Some(format!("x{}", it.stack_size.max(1))),
            });
        }
        if !v.lists.buying.is_empty() {
            rows.push(Row::Section("Buying".into()));
            for r in &v.lists.buying {
                rows.push(Row::Text(format!(
                    "{} x{} = {}p",
                    r.name, r.quantity, r.price
                )));
            }
        }
        if !v.lists.selling.is_empty() {
            rows.push(Row::Section("Selling".into()));
            for r in &v.lists.selling {
                rows.push(Row::Text(format!(
                    "{} x{} = {}p",
                    r.name, r.quantity, r.price
                )));
            }
        }
        if !v.lists.purse_text.is_empty() {
            rows.push(Row::Blank);
            rows.push(Row::Text(v.lists.purse_text.clone()));
        }
        Panel {
            title: format!("Vendor: {}", v.name.as_deref().unwrap_or("shop")),
            rows,
        }
    }

    fn trade_panel(&self) -> Panel {
        let Some(t) = &self.trade else {
            return Panel {
                title: "Trade".into(),
                rows: Vec::new(),
            };
        };
        let name = |g: u32| {
            self.items
                .get(g)
                .map(|o| o.name.clone())
                .or_else(|| t.item_names.get(&g).cloned())
                .unwrap_or_else(|| format!("#{:x}", g))
        };
        let mut rows = Vec::new();
        rows.push(Row::Key {
            label: "with".into(),
            value: t
                .partner_name
                .clone()
                .unwrap_or_else(|| format!("#{:x}", t.partner_guid)),
        });
        rows.push(Row::Blank);
        rows.push(Row::Section("you offer".into()));
        for g in &t.self_items {
            rows.push(Row::Text(name(*g)));
        }
        rows.push(Row::Section("they offer".into()));
        for g in &t.partner_items {
            rows.push(Row::Text(name(*g)));
        }
        rows.push(Row::Blank);
        rows.push(Row::Key {
            label: "you".into(),
            value: if t.self_accepted { "accepted" } else { "…" }.into(),
        });
        rows.push(Row::Key {
            label: "them".into(),
            value: if t.partner_accepted {
                "accepted"
            } else {
                "…"
            }
            .into(),
        });
        Panel {
            title: "Trade".into(),
            rows,
        }
    }

    fn appraisal_panel(&self) -> Panel {
        let Some(a) = &self.appraisal else {
            return Panel {
                title: "Appraisal".into(),
                rows: Vec::new(),
            };
        };
        let mut rows = Vec::new();
        if let Some(n) = &a.name {
            rows.push(Row::Section(n.clone()));
        }
        for (k, v) in a.property_lines() {
            rows.push(Row::Key { label: k, value: v });
        }
        Panel {
            title: "Appraisal".into(),
            rows,
        }
    }

    fn book_panel(&self) -> Panel {
        let Some(b) = &self.book else {
            return Panel {
                title: "Book".into(),
                rows: Vec::new(),
            };
        };
        let mut rows = Vec::new();
        if !b.inscription.is_empty() {
            for line in wrap(&b.inscription, 20) {
                rows.push(Row::Text(line));
            }
        }
        let page = (b.current_page + 1).max(0);
        rows.push(Row::Text(format!("page {page} / {}", b.max_num_pages)));
        if !b.scribe_name.is_empty() {
            rows.push(Row::Key {
                label: "by".into(),
                value: b.scribe_name.clone(),
            });
        }
        Panel {
            title: self.name_or(b.book_guid, "Book"),
            rows,
        }
    }

    fn journal_panel(&self) -> Panel {
        let Some(j) = &self.journal else {
            return Panel {
                title: "Journal".into(),
                rows: Vec::new(),
            };
        };
        let cur = j.current();
        let mut rows = Vec::new();
        if !cur.title.is_empty() {
            rows.push(Row::Section(cur.title.clone()));
        }
        for line in cur.notes.lines() {
            for chunk in wrap(line, 20) {
                rows.push(Row::Text(chunk));
            }
        }
        if rows.is_empty() {
            rows.push(Row::Text("(no notes)".into()));
        }
        rows.push(Row::Blank);
        rows.push(Row::Text(format!(
            "page {} / {}",
            j.current_page.max(1),
            j.pages.len()
        )));
        Panel {
            title: "Journal".into(),
            rows,
        }
    }

    fn house_panel(&self) -> Panel {
        let Some(h) = &self.house else {
            return Panel {
                title: "House".into(),
                rows: Vec::new(),
            };
        };
        let rows: Vec<Row> = h.lines.iter().map(|(t, _)| Row::Text(t.clone())).collect();
        Panel {
            title: "House".into(),
            rows,
        }
    }

    fn house_purchase_panel(&self) -> Panel {
        let Some(h) = &self.house_purchase else {
            return Panel {
                title: "Slumlord".into(),
                rows: Vec::new(),
            };
        };
        let rows = vec![
            Row::Key {
                label: "buy".into(),
                value: format!("{} staged", h.buy.len()),
            },
            Row::Key {
                label: "maintenance".into(),
                value: format!("{} staged", h.rent.len()),
            },
        ];
        Panel {
            title: "Slumlord".into(),
            rows,
        }
    }

    fn salvage_panel(&self) -> Panel {
        let Some(g) = self.salvage_guid else {
            return Panel {
                title: "Salvage".into(),
                rows: Vec::new(),
            };
        };
        let row = self.item_row(g).unwrap_or_else(|| Row::Item {
            icon: '▪',
            name: format!("#{g:x}"),
            meta: None,
        });
        Panel {
            title: "Salvage".into(),
            rows: vec![row],
        }
    }

    fn chess_panel(&self) -> Panel {
        let Some(c) = &self.chess else {
            return Panel {
                title: "Chess".into(),
                rows: Vec::new(),
            };
        };
        let rows = vec![
            Row::Text(format!("board #{:x}", c.board_guid)),
            Row::Text(format!("games ended: {}", c.games_ended)),
        ];
        Panel {
            title: "Chess".into(),
            rows,
        }
    }

    fn contracts_panel(&self) -> Panel {
        let Some(c) = &self.contracts else {
            return Panel {
                title: "Contracts".into(),
                rows: Vec::new(),
            };
        };
        let rows = vec![Row::Text(format!(
            "{} contract(s) tracked",
            c.contracts.len()
        ))];
        Panel {
            title: "Contracts".into(),
            rows,
        }
    }

    fn social_panel(&self) -> Panel {
        let Some(s) = &self.social else {
            return Panel {
                title: "Social".into(),
                rows: Vec::new(),
            };
        };
        let mut rows = Vec::new();
        if let Some(a) = &s.allegiance {
            rows.push(Row::Key {
                label: "allegiance".into(),
                value: a.allegiance_name.clone(),
            });
        }
        if !rows.is_empty() {
            rows.push(Row::Text("friends & fellowship".into()));
        } else {
            rows.push(Row::Text("(no allegiance)".into()));
        }
        Panel {
            title: "Social".into(),
            rows,
        }
    }

    fn titles_panel(&self) -> Panel {
        let Some(t) = &self.titles else {
            return Panel {
                title: "Titles".into(),
                rows: Vec::new(),
            };
        };
        let rows = vec![Row::Text(format!("{} title(s) earned", t.title_ids.len()))];
        Panel {
            title: "Titles".into(),
            rows,
        }
    }
}

// -- The panel model and its renderer --------------------------------------

/// A distilled content panel: a title and a list of rows.
pub struct Panel {
    title: String,
    rows: Vec<Row>,
}

enum Row {
    /// An item: a glyph, its name, and right-aligned meta (a quantity, ...).
    Item {
        icon: char,
        name: String,
        meta: Option<String>,
    },
    /// A plain text line.
    Text(String),
    /// A `label` + `value` pair.
    Key { label: String, value: String },
    /// A sub-heading.
    Section(String),
    /// A blank line.
    Blank,
}

/// Draw a panel (border + centred title + rows, clipped to `r`).
fn draw_panel(g: &mut Grid, p: &Panel, r: Rect) {
    g.fill(r, ' ', TEXT, BG);
    g.border(r, BORDER, BG);
    let inner_w = r.w.saturating_sub(2);
    if inner_w > 0 {
        let title = truncate(&p.title, inner_w);
        let off = inner_w.saturating_sub(title.chars().count()) / 2;
        g.put_str(r.x + 1 + off, r.y, &title, TITLE, BG);
    }
    for (i, row) in p.rows.iter().enumerate() {
        let y = r.y + 2 + i;
        if y >= r.y + r.h.saturating_sub(1) {
            break;
        }
        let x = r.x + 1;
        match row {
            Row::Item { icon, name, meta } => {
                let mw = meta.as_ref().map_or(0, |m| m.chars().count());
                let nw = inner_w.saturating_sub(1 + mw + 1);
                g.put(x, y, *icon, GOLD, BG);
                g.put_str(x + 1, y, &truncate(name, nw), TEXT, BG);
                if let Some(m) = meta {
                    g.put_str(x + inner_w - mw, y, m, DIM, BG);
                }
            }
            Row::Text(s) => {
                g.put_str(x, y, &truncate(s, inner_w), TEXT, BG);
            }
            Row::Key { label, value } => {
                let lw = label.chars().count().min(13);
                g.put_str(x, y, &truncate(label, 13), DIM, BG);
                g.put_str(
                    x + 14,
                    y,
                    &truncate(value, inner_w.saturating_sub(14)),
                    TEXT,
                    BG,
                );
                let _ = lw;
            }
            Row::Section(s) => {
                g.put_str(x, y, &truncate(s, inner_w), TITLE, BG);
            }
            Row::Blank => {}
        }
    }
    if p.rows.is_empty() {
        g.put_str(r.x + 1, r.y + 2, "(empty)", DIM, BG);
    }
}

// -- The always-on HUD -----------------------------------------------------

/// The vitals block, the target, the minimap and the chat over the viewport,
/// and the active content panel in the right column.
pub fn draw(g: &mut Grid, m: &Model, l: &Layout, now: Instant, minimap: Option<&Vec<Vec<f32>>>) {
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

    // The minimap, top-right.
    if let Some(mm) = minimap {
        draw_minimap(g, l.minimap, mm);
    }

    // The chat feed, bottom.
    draw_chat(g, m, l.chat);

    // The active content panel, in the right column.
    if let (Some(kind), Some(pr)) = (m.active_panel(now), l.panel) {
        let panel = m.build_panel(kind);
        draw_panel(g, &panel, pr);
    }
}

fn draw_chat(g: &mut Grid, m: &Model, r: Rect) {
    if r.w < 4 || r.h < 2 {
        return;
    }
    let inner_w = r.w.saturating_sub(2);
    let chat_rows = r.h.saturating_sub(1);
    let start = m.chat.len().saturating_sub(chat_rows);
    for (i, line) in m.chat.iter().skip(start).enumerate() {
        g.put_str(r.x + 1, r.y + i, &truncate(line, inner_w), TEXT, BG);
    }
    if let Some(status) = m.status.back() {
        g.put_str(r.x + 1, r.y + r.h - 1, &truncate(status, inner_w), DIM, BG);
    }
}

/// The minimap: shade characters for the terrain height grid plus the player
/// marker at the centre, in a bordered box.
fn draw_minimap(g: &mut Grid, r: Rect, heights: &Vec<Vec<f32>>) {
    const NO_TERRAIN: f32 = f32::MIN;
    let n = heights.len();
    g.border(r, BORDER, BG);
    let iw = (r.w.saturating_sub(2)).min(n);
    let ih = (r.h.saturating_sub(2)).min(n);
    let mut min_h = f32::INFINITY;
    let mut max_h = f32::NEG_INFINITY;
    for row in heights {
        for &h in row {
            if h > NO_TERRAIN {
                min_h = min_h.min(h);
                max_h = max_h.max(h);
            }
        }
    }
    let span = if max_h > min_h { max_h - min_h } else { 1.0 };
    for gy in 0..ih {
        for gx in 0..iw {
            let h = heights
                .get(gy)
                .and_then(|row| row.get(gx))
                .copied()
                .unwrap_or(NO_TERRAIN);
            let (ch, fg) = if h <= NO_TERRAIN {
                (' ', BG)
            } else {
                let f = ((h - min_h) / span).clamp(0.0, 1.0);
                (
                    shade_char(f),
                    [
                        (90.0 + 160.0 * f) as u8,
                        (120.0 + 130.0 * f) as u8,
                        (70.0 + 70.0 * f) as u8,
                    ],
                )
            };
            g.put(r.x + 1 + gx, r.y + 1 + gy, ch, fg, BG);
        }
    }
    // The player at the centre of the map.
    let px = r.x + 1 + (n / 2).min(iw.saturating_sub(1));
    let py = r.y + 1 + (n / 2).min(ih.saturating_sub(1));
    g.put(px, py, '@', GOLD, BG);
}

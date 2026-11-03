//! The `--ui terminal` front end: the real 3-D world, rendered by the
//! normal world renderer into an offscreen target (a GPU adapter, or a
//! software Vulkan implementation on a headless box — no window, no
//! display server, no image protocol) and read back as a grid of
//! sub-pixels, then encoded to a plain terminal in 24-bit colour as
//! braille dots (or flat full blocks). The 2-D interface is drawn from
//! the scene's facts in a primitive cell layer (box drawing, UTF-8 text):
//! the vitals, the target, the chat feed and its entry box — never the
//! game's widget tree.
//!
//! Each terminal cell is a 2-wide by 4-tall block of sub-pixels:
//! `--render braille` (the default) makes it a 2 x 4 braille dot matrix
//! with a foreground/background colour, and `flat` averages the cell's
//! sub-pixels into one full block. The world is drawn exactly as a window
//! would draw it — the chase camera follows the player, the sun, the sky,
//! the fog — so the view is the game's, not a re-projection.

use std::io::{self, Write};
use std::process::ExitCode;
use std::sync::{Arc, mpsc};
use std::time::{Duration, Instant};

#[cfg(unix)]
use nix::unistd::{dup, dup2_stderr};
#[cfg(unix)]
use std::os::unix::io::{AsFd, OwnedFd};

use ac_render::gameplay_input::{DeviceState, GameplayInput, GameplayOutput};
use ac_render::scene::SceneSource;
use ac_render::world_click::{ClickFacts, WorldClick, WorldClicks};
use ac_render::{GraphicsOptions, Offscreen, WorldSettings};
use ac_runtime::session::{
    LifecycleCommand, ObjectSummary, PresentationEvent, SessionCommand, SessionEvent,
    SessionHandle, SessionSummary,
};
use ac_ui::input::bindings::KeyBindings;
use ac_ui::{Key, Modifiers};
use crossterm::event::{
    self, Event, KeyCode, KeyEventKind, KeyModifiers, MouseButton, MouseEvent, MouseEventKind,
};

use crate::options::{LogLevel, Options, RenderStyle};
use crate::stdio::errln;
use crate::termui::{self, Cell};

/// The frame's target period (the GPU paces itself; this only caps the UI
/// loop when the frames finish early, and llvmpipe usually does not).
const FRAME_PERIOD: Duration = Duration::from_millis(33); // ~30 fps

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
// Keeping standard error off the frame.
// ---------------------------------------------------------------------------

/// The terminal and standard error are the same screen. While the render loop
/// owns the frame, anything a crate writes to stderr — the session log, a
/// `vfx:` notice — would land at the cursor and scrawl across the world and
/// the HUD. `StderrToLog` points descriptor 2 at a log file for as long as it
/// lives and restores it on drop, so every writer is diverted at once. Off
/// Unix it is a no-op (there is no alt screen sharing the console).
#[cfg(unix)]
struct StderrToLog(OwnedFd);
#[cfg(not(unix))]
struct StderrToLog;

impl StderrToLog {
    /// Divert standard error to `openac-terminal-<pid>.log` in the temp dir.
    /// Setup failures are reported before this is taken, so they still reach
    /// the console; only the frame's ongoing diagnostics are moved off screen.
    fn take_stderr() -> io::Result<Self> {
        #[cfg(unix)]
        {
            let path =
                std::env::temp_dir().join(format!("openac-terminal-{}.log", std::process::id()));
            let log = std::fs::File::create(&path)?;
            // Keep a copy of the console's descriptor, then point descriptor
            // 2 at the log; dropping `log` closes its own descriptor while
            // descriptor 2 keeps the file open.
            let saved = dup(io::stderr().as_fd()).map_err(io::Error::other)?;
            dup2_stderr(log.as_fd()).map_err(io::Error::other)?;
            Ok(Self(saved))
        }
        #[cfg(not(unix))]
        {
            Ok(Self)
        }
    }
}

#[cfg(unix)]
impl Drop for StderrToLog {
    fn drop(&mut self) {
        // Point descriptor 2 back at the console; the kept copy is released
        // when it drops, leaving descriptor 2 to hold the console.
        let _ = dup2_stderr(self.0.as_fd());
    }
}

// ---------------------------------------------------------------------------
// Frame encoding: the read-back RGBA pixels to terminal cells.
// ---------------------------------------------------------------------------

/// The braille dot's column and row (in the 2 x 4 matrix) by the bit that
/// lights it.
const BRIT_BITS: [[u8; 2]; 4] = [[0, 3], [1, 4], [2, 5], [6, 7]];

fn luminance(c: [u8; 3]) -> f32 {
    0.2126 * c[0] as f32 + 0.7152 * c[1] as f32 + 0.0722 * c[2] as f32
}

/// Encode an RGBA8 frame (`width` x `height`, one 4-byte RGBA per pixel)
/// into a grid of terminal cells: braille lights a 2 x 4 dot matrix per
/// cell, flat averages the cell's sub-pixels into a block. The frame is
/// `2 * cols` wide by `4 * rows` tall.
fn encode(
    rgba: &[u8],
    width: usize,
    cols: usize,
    rows: usize,
    style: RenderStyle,
) -> Vec<Vec<Cell>> {
    let mut grid = Vec::with_capacity(rows);
    for cy in 0..rows {
        let mut row = Vec::with_capacity(cols);
        for cx in 0..cols {
            let mut colors = [[0u8; 3]; 8];
            let mut lums = [0.0f32; 8];
            for dy in 0..4 {
                for dx in 0..2 {
                    let i = ((cy * 4 + dy) * width + (cx * 2 + dx)) * 4;
                    let c = [rgba[i], rgba[i + 1], rgba[i + 2]];
                    let n = dy * 2 + dx;
                    colors[n] = c;
                    lums[n] = luminance(c);
                }
            }
            if style == RenderStyle::Flat {
                // One sample per cell: a full block in the average colour of
                // the cell's sub-pixels.
                let avg: [f32; 3] = [
                    colors.iter().map(|c| c[0] as f32).sum::<f32>() / 8.0,
                    colors.iter().map(|c| c[1] as f32).sum::<f32>() / 8.0,
                    colors.iter().map(|c| c[2] as f32).sum::<f32>() / 8.0,
                ];
                let c = [
                    avg[0].round() as u8,
                    avg[1].round() as u8,
                    avg[2].round() as u8,
                ];
                row.push(Cell {
                    ch: '\u{2588}',
                    fg: c,
                    bg: c,
                });
                continue;
            }
            let min_lum = lums.iter().cloned().fold(f32::INFINITY, f32::min);
            let max_lum = lums.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let (fg, bg, bits): ([u8; 3], [u8; 3], u16) = if max_lum - min_lum < 0.02 {
                // A flat region: a solid block of the average colour.
                let avg: [f32; 3] = [
                    colors.iter().map(|c| c[0] as f32).sum::<f32>() / 8.0,
                    colors.iter().map(|c| c[1] as f32).sum::<f32>() / 8.0,
                    colors.iter().map(|c| c[2] as f32).sum::<f32>() / 8.0,
                ];
                let fg = [
                    avg[0].round() as u8,
                    avg[1].round() as u8,
                    avg[2].round() as u8,
                ];
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
                (colors[max_i], colors[min_i], bits)
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
// The run: session, offscreen world, input, and the cell layer.
// ---------------------------------------------------------------------------

/// Runs the terminal front end to its end and reports its exit code.
/// `paths` is the process's path set (the player's display settings).
pub fn run(handle: SessionHandle, options: Options, paths: Option<&ac_host::PathSet>) -> ExitCode {
    if enter_terminal().is_err() {
        errln!("openac: could not set the terminal into raw mode");
        return ExitCode::from(super::EXIT_SESSION_FAILED);
    }
    let code = run_render_loop(handle, options, paths);
    leave_terminal();
    code
}

fn run_render_loop(
    handle: SessionHandle,
    options: Options,
    paths: Option<&ac_host::PathSet>,
) -> ExitCode {
    let commands = handle.commands();

    let (to_window, from_session) = mpsc::channel::<PresentationEvent>();
    let _ = commands.send(SessionCommand::PresentsWorld(true));
    let _ = commands.send(SessionCommand::Attach(to_window));
    let source = SceneSource::new(from_session).with_commands(commands.clone());

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

    let fail = |message: &str| {
        errln!("openac: {message}");
        let _ = tx_code.send(ExitCode::from(super::EXIT_SESSION_FAILED));
        drop(monitor);
        ExitCode::from(super::EXIT_SESSION_FAILED)
    };

    // A pty without a winsize reports (0, 0); keep the default instead of
    // rendering a 1x1 frame.
    let (mut cols, mut rows) = match crossterm::terminal::size() {
        Ok((c, r)) if c > 0 && r > 0 => (c as usize, r as usize),
        _ => (80, 24),
    };

    let dat_dir = match options.dat_dir.clone().or_else(ac_dat::default_dat_dir) {
        Some(dir) => dir,
        None => return fail("no dat directory found (set OPENAC_DAT_DIR or --dat-dir)"),
    };
    let graphics = match crate::graphics::GraphicsEnv::parse(|name| std::env::var(name).ok()) {
        Ok(graphics) => graphics,
        Err(message) => return fail(&message.to_string()),
    };
    let prefs = crate::graphics::DisplayPrefs::load(paths);
    let near = options
        .landblock_radius
        .or(graphics.landblock_radius)
        .unwrap_or(2);
    let far = graphics.far_radius.unwrap_or(near + 3);
    let world_settings = WorldSettings {
        content: Arc::new(ac_render::content::ContentSlot::for_area(
            dat_dir, near, far,
        )),
        quality_overrides: ac_render::quality::QualityOverrides {
            near_radius: Some(near),
            far_radius: Some(far),
            ..graphics.quality
        },
        residency: graphics.residency,
        orbit: graphics.orbit,
        reveal_radius: graphics.reveal_radius,
        sky: graphics.sky,
        // The terminal runs over SSH: nothing to hear through it.
        audio: ac_render::audio::AudioOptions {
            launch: ac_render::audio::AudioLaunchOptions {
                no_audio: true,
                ..Default::default()
            },
            ..Default::default()
        },
    };
    let display = ac_render::quality::effective_display(&prefs.settings);
    let mut offscreen = match Offscreen::new(
        (2 * cols) as u32,
        (4 * rows) as u32,
        &world_settings,
        &display,
        &GraphicsOptions {
            gpu_adapter: graphics.gpu_adapter.clone(),
            gpu_force_unsupported: graphics.gpu_force_unsupported.clone(),
            no_block_compression: graphics.no_block_compression,
            ..Default::default()
        },
    ) {
        Ok(offscreen) => {
            errln!(
                "openac: terminal view on {} ({})",
                offscreen.adapter_name(),
                offscreen.adapter_backend()
            );
            offscreen
        }
        Err(error) => {
            return fail(&format!(
                "{error} (the terminal view needs a Vulkan-capable adapter; on a headless \
                 box install mesa-vulkan-drivers for llvmpipe)"
            ));
        }
    };
    offscreen.attach_scene(Box::new(source));

    let mut gameplay = GameplayInput::new(KeyBindings::retail_defaults());
    let mut last = Instant::now();
    let input_start = Instant::now();
    let mut stdout = io::stdout();
    let mut input = TerminalInput {
        modifiers: Modifiers::NONE,
        size: (cols as u16, rows as u16),
        taps: KeyTaps::new(),
        cursor: (cols as u16 / 2, rows as u16 / 2),
        mouse_buttons: 0,
        world_clicks: WorldClicks::default(),
    };
    let mut quit = false;
    let mut session_code: Option<ExitCode> = None;
    let mut model = termui::Model::new();
    let mut last_drawn: Option<Vec<Vec<Cell>>> = None;
    // Divert standard error to a log file for the rest of this run so the
    // session log and `vfx:` notices never scrawl over the frame; the guard
    // restores the console when `run_render_loop` returns. Failing to divert
    // only re-exposes the frame to stderr, so carry on.
    let _stderr = StderrToLog::take_stderr().map_err(|e| {
        errln!("openac: could not divert standard error: {e}");
    });
    loop {
        let now = Instant::now();
        let dt = now.saturating_duration_since(last).as_secs_f32().min(0.1);
        last = now;
        let gameplay_now = input_start.elapsed().as_secs_f32();
        let now_ms = u64::try_from(input_start.elapsed().as_millis()).unwrap_or(u64::MAX);

        // Drain input that arrived since the last frame (non-blocking); Esc quits.
        if drain_input(
            Duration::ZERO,
            &mut gameplay,
            &commands,
            gameplay_now,
            now_ms,
            &mut input,
            &mut model,
            &mut offscreen,
        ) {
            quit = true;
        }

        // Held movement, polled each frame.
        let state = input_state(input.modifiers);
        let mut outs = Vec::new();
        let fly = gameplay.tick(dt, gameplay_now, &state, None, &mut outs);
        route_outputs(outs, &commands);

        // The terminal may never report a key coming up: the held keys are
        // let go now, after the held poll has seen them, once their
        // autorepeat has gone quiet for the grace — a held key stays down
        // across its stream instead of walking and stopping at the
        // autorepeat's cadence.
        for (key, mods) in input.taps.take_releases(Instant::now()) {
            let s = input_state(mods);
            let mut outs = Vec::new();
            gameplay.key(key, false, &s, None, gameplay_now, &mut outs);
            route_outputs(outs, &commands);
        }

        // Draw the frame (the GPU paces this); the previous frame reads back.
        let ready = match offscreen.render(fly) {
            Ok(ready) => ready,
            Err(error) => {
                errln!("openac: {error}");
                break;
            }
        };
        for notice in offscreen.take_notices() {
            errln!("openac: {notice}");
        }

        if let Some(scene) = offscreen.scene_mut() {
            model.update(scene);
        }

        // The terminal was resized: adopt the new size. The terminal has
        // reflowed the screen content, so the last-drawn diff no longer
        // matches what is displayed; the next frame repaints everything.
        if (cols, rows) != (input.size.0 as usize, input.size.1 as usize) {
            cols = input.size.0 as usize;
            rows = input.size.1 as usize;
            offscreen.resize((2 * cols) as u32, (4 * rows) as u32);
            last_drawn = None;
        }
        let layout = termui::layout(cols, rows);
        let mut grid = termui::Grid::new(cols, rows);
        if ready && let Some(rgba) = offscreen.frame() {
            let width = offscreen.size().0;
            let view = encode(rgba, width as usize, cols, rows, options.render_style);
            grid.blit(layout.viewport.x, layout.viewport.y, &view);
        }
        termui::draw(&mut grid, &model, &layout);
        if let Err(e) = draw_grid(&mut stdout, grid.rows(), last_drawn.as_deref()) {
            errln!("openac: terminal write failed: {e}");
            break;
        }
        last_drawn = Some(grid.rows().to_vec());

        if quit {
            break;
        }
        if let Ok(code) = rx_code.try_recv() {
            session_code = Some(code);
            break;
        }
        if offscreen.session_gone() {
            break;
        }
        // Wait out the frame, handling input as it arrives so the movement
        // stays responsive instead of waiting a full frame to be seen.
        let remaining = FRAME_PERIOD.saturating_sub(now.elapsed());
        if !quit
            && remaining > Duration::ZERO
            && drain_input(
                remaining,
                &mut gameplay,
                &commands,
                gameplay_now,
                now_ms,
                &mut input,
                &mut model,
                &mut offscreen,
            )
        {
            quit = true;
        }
    }

    let code = match session_code {
        Some(code) => code,
        None => {
            // The session is still running (Esc, or the terminal broke):
            // ask it to end, then wait for its code. A clean log off takes a
            // few seconds — the session asks the server and waits for the
            // confirmation — so give it room before reporting a failure.
            let _ = commands.send(SessionCommand::Shutdown);
            rx_code
                .recv_timeout(Duration::from_secs(10))
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
/// keyboard; the dispatcher keeps its own held-key set).
fn input_state(modifiers: Modifiers) -> DeviceState<'static> {
    DeviceState {
        keys: None,
        buttons: 0,
        modifiers,
        want_mouse: false,
        want_keyboard: false,
    }
}

/// The gameplay input's device state for the terminal's mouse: the held
/// buttons (the orbit reads them) and the modifiers. `want_mouse` stays off —
/// the terminal has no cursor to capture, and `pointer_moved` orbits only
/// while the mouse is not captured.
fn mouse_state(modifiers: Modifiers, buttons: u32) -> DeviceState<'static> {
    DeviceState {
        keys: None,
        buttons,
        modifiers,
        want_mouse: false,
        want_keyboard: false,
    }
}

/// A crossterm mouse button as the gameplay's button number (left 0, right 1,
/// middle 2).
fn mouse_button(button: MouseButton) -> u8 {
    match button {
        MouseButton::Left => 0,
        MouseButton::Right => 1,
        MouseButton::Middle => 2,
    }
}

/// A wheel event as the gameplay's vertical notches (positive away from the
/// user): up one, down one, sideways none (the terminal has no window to
/// scroll sideways in).
fn scroll_notches(kind: MouseEventKind) -> f32 {
    match kind {
        MouseEventKind::ScrollUp => 1.0,
        MouseEventKind::ScrollDown => -1.0,
        _ => 0.0,
    }
}

/// A pointer cell's centre in offscreen pixels: a cell is two sub-pixels wide
/// and four tall (the 2:1 cell aspect the world is drawn at).
fn cursor_pixel(cursor: (u16, u16)) -> (f32, f32) {
    (cursor.0 as f32 * 2.0 + 1.0, cursor.1 as f32 * 4.0 + 2.0)
}

/// A plain terminal often never reports a key coming up: it sends the press
/// (and the autorepeat re-presses) and nothing else. Without a let-go every
/// key the player presses stays held and the character keeps turning or
/// walking with no input at all; but letting a press go after one frame
/// makes a held key walk-stop-walk-stop at the autorepeat's cadence (the
/// movement "pauses every small iteration"). A key is therefore held across
/// its autorepeat stream and let go only once the stream goes quiet for the
/// grace: a repeat press (faster than the grace) keeps it down, a real
/// let-go (no more repeats) is felt after the grace. A release that does
/// arrive (an enhanced keyboard protocol) lets the key go at once.
///
/// The grace is longer than the terminal's autorepeat cadence so a held
/// key's stream keeps it down; it is the let-go's latency when the terminal
/// reports no releases. Tunable with `OPENAC_KEY_HOLD_MS`.
fn key_hold_grace() -> Duration {
    std::env::var("OPENAC_KEY_HOLD_MS")
        .ok()
        .and_then(|raw| raw.trim().parse::<u64>().ok())
        .map(Duration::from_millis)
        .unwrap_or(Duration::from_millis(300))
}

#[derive(Debug)]
struct KeyTaps {
    /// The keys still held: (key, modifiers, last press).
    held: Vec<(Key, Modifiers, Instant)>,
    /// The quiet period before a held key is let go.
    grace: Duration,
}

impl Default for KeyTaps {
    fn default() -> Self {
        Self {
            held: Vec::new(),
            grace: key_hold_grace(),
        }
    }
}

impl KeyTaps {
    /// No key is held.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// A key event: a press (a repeat refreshes the hold, it does not stack)
    /// is kept down; a release lets it go at once.
    pub fn key(&mut self, key: Key, pressed: bool, modifiers: Modifiers) {
        if pressed {
            let now = Instant::now();
            match self.held.iter_mut().find(|(held, _, _)| *held == key) {
                Some(entry) => {
                    entry.1 = modifiers;
                    entry.2 = now;
                }
                None => self.held.push((key, modifiers, now)),
            }
        } else {
            self.held.retain(|(held, _, _)| *held != key);
        }
    }

    /// Let go of the held keys whose autorepeat has gone quiet for the grace
    /// (no press since), in press order; the rest stay down.
    pub fn take_releases(&mut self, now: Instant) -> Vec<(Key, Modifiers)> {
        let mut out = Vec::new();
        self.held.retain(|(key, mods, last)| {
            let gone = now.duration_since(*last) >= self.grace;
            if gone {
                out.push((*key, *mods));
            }
            !gone
        });
        out
    }
}

/// The mutable input state the terminal feeds to the gameplay dispatcher:
/// the held modifiers, the terminal's size, the presses waiting their let-go
/// (the terminal's missing key releases), and the pointer (its last cell, the
/// buttons held, and the click memory a target mode's claim needs).
struct TerminalInput {
    modifiers: Modifiers,
    /// The terminal's current size (the last `Resize` event).
    size: (u16, u16),
    taps: KeyTaps,
    /// The last cell the pointer was in (for a click's pick and a drag's travel).
    cursor: (u16, u16),
    /// The mouse buttons held, one bit per button number.
    mouse_buttons: u32,
    /// The click memory a target mode's claim needs.
    world_clicks: WorldClicks,
}

/// Where one terminal key goes: into the entry box (with the line the box
/// submitted, when an Enter sent one) or to the game.
enum KeyRoute {
    /// The entry box took the key; `send` is the line to send to the
    /// session (an Enter on a non-empty line).
    Box { send: Option<String> },
    /// The key is the game's, as before.
    Game,
}

/// Route one terminal key between the entry box and the game. In text input
/// mode every key is the box's — a release included, so a key let go while
/// typing never reaches the game — and a Ctrl combo types nothing. Out of
/// the mode, an Enter or Tab press hands the keys to the box (the key that
/// did it types nothing anywhere), and every other key is the game's.
fn route_key(
    code: KeyCode,
    pressed: bool,
    modifiers: KeyModifiers,
    entry: &mut termui::ChatBox,
) -> KeyRoute {
    if entry.active {
        let send = if pressed {
            match code {
                KeyCode::Char(c) if !modifiers.contains(KeyModifiers::CONTROL) => {
                    entry.push_char(c);
                    None
                }
                KeyCode::Backspace => {
                    entry.backspace();
                    None
                }
                KeyCode::Enter => entry.enter(),
                KeyCode::Tab => {
                    entry.deactivate();
                    None
                }
                _ => None,
            }
        } else {
            None
        };
        return KeyRoute::Box { send };
    }
    if pressed && matches!(code, KeyCode::Enter | KeyCode::Tab) {
        entry.activate();
        return KeyRoute::Box { send: None };
    }
    KeyRoute::Game
}

/// Handle one terminal event, routing it to the entry box or the gameplay
/// input. Returns `true` when the user asks to quit (Esc).
fn handle_event(
    evt: Event,
    gameplay: &mut GameplayInput,
    commands: &mpsc::Sender<SessionCommand>,
    gameplay_now: f32,
    input: &mut TerminalInput,
    model: &mut termui::Model,
) -> bool {
    match evt {
        Event::Key(k) => {
            if k.code == KeyCode::Esc {
                return true;
            }
            input.modifiers = modifiers_from_crossterm(k.modifiers);
            let pressed = match k.kind {
                KeyEventKind::Press => true,
                KeyEventKind::Release => false,
                _ => return false,
            };
            // The entry box takes the keys in text input mode (and an
            // Enter or Tab press outside it); the game never sees a key
            // the box takes.
            match route_key(k.code, pressed, k.modifiers, &mut model.chat_box) {
                KeyRoute::Box { send } => {
                    if let Some(line) = send {
                        let _ = commands.send(SessionCommand::Chat(line));
                    }
                    return false;
                }
                KeyRoute::Game => {}
            }
            let Some(key) = key_from_crossterm(k.code) else {
                return false;
            };
            input.taps.key(key, pressed, input.modifiers);
            let state = input_state(input.modifiers);
            let mut outs = Vec::new();
            gameplay.key(key, pressed, &state, None, gameplay_now, &mut outs);
            route_outputs(outs, commands);
        }
        // The new size takes effect on the next frame (the loop adopts it);
        // zero-size reports (some terminals mid-resize) are noise.
        Event::Resize(cols, rows) if cols > 0 && rows > 0 => {
            input.size = (cols, rows);
        }
        _ => {}
    }
    false
}

/// Handle one terminal mouse event: the buttons and wheel go to the gameplay
/// input (a click's select/examine/use comes back as a `WorldClick`), and the
/// pointer's cell is remembered for the click's pick and a drag's travel.
#[allow(clippy::too_many_arguments)]
fn handle_mouse(
    m: MouseEvent,
    gameplay: &mut GameplayInput,
    commands: &mpsc::Sender<SessionCommand>,
    gameplay_now: f32,
    now_ms: u64,
    input: &mut TerminalInput,
    offscreen: &mut Offscreen,
    model: &termui::Model,
) {
    let (column, row) = (m.column, m.row);
    match m.kind {
        MouseEventKind::Down(button) | MouseEventKind::Up(button) => {
            let pressed = matches!(m.kind, MouseEventKind::Down(_));
            let button = mouse_button(button);
            if pressed {
                input.mouse_buttons |= 1u32 << button;
            } else {
                input.mouse_buttons &= !(1u32 << button);
            }
            input.cursor = (column, row);
            let state = mouse_state(input.modifiers, input.mouse_buttons);
            let mut outs = Vec::new();
            gameplay.button(
                button,
                pressed,
                now_ms,
                &state,
                Some(offscreen.cameras_mut()),
                gameplay_now,
                &mut outs,
            );
            // A radar blip under the pointer, if any: it is what the click
            // acts on, in place of the world's pick.
            let radar_pick =
                model.radar_blip_at(input.size.0 as usize, input.size.1 as usize, column, row);
            for action in take_world_clicks(outs, commands) {
                resolve_world_click(
                    offscreen,
                    commands,
                    &mut input.world_clicks,
                    input.cursor,
                    action,
                    now_ms,
                    radar_pick,
                );
            }
        }
        // A held button's travel: the chase camera orbits while the right
        // button is held (the window's orbit; the terminal has no orbit
        // camera for the left button).
        MouseEventKind::Drag(_) => {
            let (last_column, last_row) = input.cursor;
            let dx = i32::from(column) - i32::from(last_column);
            let dy = i32::from(row) - i32::from(last_row);
            input.cursor = (column, row);
            let state = mouse_state(input.modifiers, input.mouse_buttons);
            gameplay.pointer_moved(
                dx as f32 * 2.0,
                dy as f32 * 4.0,
                &state,
                Some(offscreen.cameras_mut()),
                gameplay_now,
            );
        }
        // No button held: the cell is only the next click's pick point.
        MouseEventKind::Moved => {
            input.cursor = (column, row);
        }
        MouseEventKind::ScrollUp
        | MouseEventKind::ScrollDown
        | MouseEventKind::ScrollLeft
        | MouseEventKind::ScrollRight => {
            let notches = scroll_notches(m.kind);
            if notches != 0.0 {
                let state = mouse_state(input.modifiers, input.mouse_buttons);
                let mut outs = Vec::new();
                gameplay.wheel(
                    notches,
                    &state,
                    Some(offscreen.cameras_mut()),
                    gameplay_now,
                    &mut outs,
                );
                route_outputs(outs, commands);
            }
        }
    }
}

/// Route the non-click outputs to the session and return the clicks (which
/// need the frame's pick, done by the caller).
fn take_world_clicks(
    outs: Vec<GameplayOutput>,
    commands: &mpsc::Sender<SessionCommand>,
) -> Vec<WorldClick> {
    let mut clicks = Vec::new();
    for out in outs {
        match out {
            GameplayOutput::WorldClick(click) => clicks.push(click),
            other => route_outputs(vec![other], commands),
        }
    }
    clicks
}

/// A click on the 3-D view: what the frame drew under the pointer is
/// selected, examined or used (the window's `WorldClick` handling), and
/// flashes. The terminal has no windows, so nothing is in a target mode and
/// nothing is wielded.
fn resolve_world_click(
    offscreen: &mut Offscreen,
    commands: &mpsc::Sender<SessionCommand>,
    memory: &mut WorldClicks,
    cursor: (u16, u16),
    action: WorldClick,
    now_ms: u64,
    radar_pick: Option<u32>,
) {
    let (x, y) = cursor_pixel(cursor);
    let picked = offscreen.pick(x, y);
    // A radar blip under the pointer, if hit, is what the click acts on in
    // place of the world's pick.
    let guid = radar_pick.or(picked.map(|(g, _)| g));
    let facts = ClickFacts::default();
    let outcome = memory.click(action, guid, facts, now_ms);
    // The selection flash needs a local render, which a radar blip has no
    // claim to (the object may be off the frame), so only a world pick flashes.
    if radar_pick.is_none()
        && let Some((picked_guid, local_id)) = picked
        && outcome.pulse == Some(picked_guid)
    {
        offscreen.begin_selection_flash(picked_guid, local_id);
    }
    for command in outcome.commands {
        let _ = commands.send(command);
    }
}

/// Wait up to `timeout` for input, then drain everything that has arrived, so
/// input is handled as it comes rather than once per rendered frame. With a
/// zero timeout it drains whatever is already pending; otherwise it blocks
/// until the deadline (keeping the render on its own cadence). Returns `true`
/// when the user asks to quit.
#[allow(clippy::too_many_arguments)]
fn drain_input(
    timeout: Duration,
    gameplay: &mut GameplayInput,
    commands: &mpsc::Sender<SessionCommand>,
    gameplay_now: f32,
    now_ms: u64,
    input: &mut TerminalInput,
    model: &mut termui::Model,
    offscreen: &mut Offscreen,
) -> bool {
    let deadline = Instant::now() + timeout;
    let mut quit = false;
    loop {
        let wait = if timeout == Duration::ZERO {
            Duration::ZERO
        } else {
            deadline.saturating_duration_since(Instant::now())
        };
        if !event::poll(wait).unwrap_or(false) {
            break;
        }
        let Ok(evt) = event::read() else {
            break;
        };
        match evt {
            Event::Mouse(m) => {
                handle_mouse(
                    m,
                    gameplay,
                    commands,
                    gameplay_now,
                    now_ms,
                    input,
                    offscreen,
                    model,
                );
            }
            other => {
                quit = handle_event(other, gameplay, commands, gameplay_now, input, model);
            }
        }
        if quit {
            break;
        }
    }
    quit
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
        crossterm::event::EnableMouseCapture,
        crossterm::cursor::Hide
    )?;
    out.flush()?;
    Ok(())
}

fn leave_terminal() {
    let mut out = io::stdout();
    let _ = crossterm::execute!(
        out,
        crossterm::event::DisableMouseCapture,
        crossterm::cursor::Show,
        crossterm::terminal::LeaveAlternateScreen
    );
    let _ = crossterm::terminal::disable_raw_mode();
}

/// Write the cells that differ from the previous frame to the terminal, with
/// 24-bit colour. Only changed cells are emitted — the cursor is positioned
/// with CUP at the start of each changed run and an SGR sequence is emitted
/// only when a colour changes — so an idle scene writes almost nothing. That
/// is what keeps the terminal from flooding and flickering over SSH.
fn draw_grid(
    out: &mut impl Write,
    grid: &[Vec<Cell>],
    prev: Option<&[Vec<Cell>]>,
) -> io::Result<()> {
    use std::fmt::Write as _;
    let rows = grid.len();
    let cols = grid.first().map_or(0, |row| row.len());
    // A different shape (a terminal resize) is a full repaint.
    let aligned =
        prev.is_some_and(|p| p.len() == rows && p.first().is_some_and(|row| row.len() == cols));
    let changed =
        |r: usize, c: usize| -> bool { !prev.is_some_and(|p| aligned && p[r][c] == grid[r][c]) };
    let mut buf = String::with_capacity(rows * cols / 2);
    let mut last_fg: Option<[u8; 3]> = None;
    let mut last_bg: Option<[u8; 3]> = None;
    for (r, row) in grid.iter().enumerate() {
        let mut c = 0;
        while c < cols {
            if !changed(r, c) {
                c += 1;
                continue;
            }
            let _ = write!(buf, "\x1b[{};{}H", r + 1, c + 1);
            while c < cols && changed(r, c) {
                let cell = row[c];
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
                c += 1;
            }
        }
    }
    let _ = write!(buf, "\x1b[0m");
    out.write_all(buf.as_bytes())?;
    out.flush()
}

// ---------------------------------------------------------------------------
// Tests: the encoding, the diff redraw and the input, content-free.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::termui::ChatBox;

    /// A solid RGBA frame (one colour, `cols` x `rows` cells worth).
    fn solid_rgba(cols: usize, rows: usize, color: [u8; 3]) -> (Vec<u8>, usize) {
        let (width, height) = (2 * cols, 4 * rows);
        let mut rgba = Vec::with_capacity(width * height * 4);
        for _ in 0..width * height {
            rgba.extend_from_slice(&[color[0], color[1], color[2], 255]);
        }
        (rgba, width)
    }

    /// The mouse buttons map to the gameplay's numbers (left 0, right 1,
    /// middle 2).
    #[test]
    fn a_mouse_button_maps_to_its_gameplay_number() {
        assert_eq!(mouse_button(MouseButton::Left), 0);
        assert_eq!(mouse_button(MouseButton::Right), 1);
        assert_eq!(mouse_button(MouseButton::Middle), 2);
    }

    /// The wheel scrolls away from the user by one notch, and a sideways step
    /// does nothing (there is no window to scroll sideways in).
    #[test]
    fn a_wheel_step_is_one_vertical_notch() {
        assert_eq!(scroll_notches(MouseEventKind::ScrollUp), 1.0);
        assert_eq!(scroll_notches(MouseEventKind::ScrollDown), -1.0);
        assert_eq!(scroll_notches(MouseEventKind::ScrollLeft), 0.0);
        assert_eq!(scroll_notches(MouseEventKind::ScrollRight), 0.0);
    }

    /// A cell maps to the centre of its two-wide by four-tall sub-pixel block.
    #[test]
    fn a_cell_maps_to_the_centre_of_its_sub_pixel_block() {
        assert_eq!(cursor_pixel((0, 0)), (1.0, 2.0));
        assert_eq!(cursor_pixel((1, 1)), (3.0, 6.0));
        assert_eq!(cursor_pixel((19, 9)), (39.0, 38.0));
    }

    #[test]
    fn flat_region_encodes_as_a_solid_block() {
        let (rgba, width) = solid_rgba(1, 1, [128, 51, 26]);
        let grid = encode(&rgba, width, 1, 1, RenderStyle::Braille);
        assert_eq!(grid[0][0].ch, char::from_u32(0x2800 | 0xFF).unwrap());
        assert_eq!(grid[0][0].fg, grid[0][0].bg);
    }

    #[test]
    fn flat_style_encodes_a_cell_as_a_full_block() {
        // A two-tone cell (dark top, light bottom): braille lights the lower
        // dots; flat averages the two halves into one solid block.
        let (mut rgba, width) = solid_rgba(1, 1, [255, 255, 255]);
        for px in rgba.chunks_exact_mut(4).take(width * 2) {
            px.iter_mut().for_each(|b| *b = 0);
        }
        let flat = encode(&rgba, width, 1, 1, RenderStyle::Flat);
        assert_eq!(flat[0][0].ch, '\u{2588}');
        assert_eq!(flat[0][0].fg, [128, 128, 128]);
        assert_eq!(flat[0][0].fg, flat[0][0].bg);
        let braille = encode(&rgba, width, 1, 1, RenderStyle::Braille);
        assert_eq!(braille[0][0].ch, char::from_u32(0x2800 | 0xE4).unwrap());
    }

    #[test]
    fn draw_grid_only_writes_changed_cells() {
        fn solid(n: usize, ch: char, color: [u8; 3]) -> Vec<Vec<Cell>> {
            let row = vec![
                Cell {
                    ch,
                    fg: color,
                    bg: color
                };
                n
            ];
            (0..n).map(|_| row.clone()).collect()
        }

        // First frame: no previous grid, so every cell is painted.
        let base = solid(10, ' ', [10, 20, 30]);
        let mut out = Vec::new();
        draw_grid(&mut out, &base, None).unwrap();
        let full = out.len();
        assert!(
            full > 100,
            "a full repaint should be non-trivial, got {full} bytes"
        );

        // Second frame: identical, so nothing changed -> only the trailing reset.
        let mut out = Vec::new();
        draw_grid(&mut out, &base, Some(&base)).unwrap();
        assert_eq!(
            String::from_utf8(out).unwrap(),
            "\x1b[0m",
            "an unchanged frame must not redraw any cell"
        );

        // Third frame: one cell changes -> a CUP to that cell and just its char.
        let mut changed = base.clone();
        changed[1][1] = Cell {
            ch: 'X',
            fg: [1, 2, 3],
            bg: [4, 5, 6],
        };
        let mut out = Vec::new();
        draw_grid(&mut out, &changed, Some(&base)).unwrap();
        let s = String::from_utf8(out).unwrap();
        assert!(
            s.contains("\x1b[2;2H"),
            "must position to the changed cell: {s:?}"
        );
        assert_eq!(
            s.matches('X').count(),
            1,
            "only the changed char is written: {s:?}"
        );
        assert!(
            s.len() < full / 2,
            "a single-cell change must be far smaller than a full repaint: {} vs {full}",
            s.len()
        );
    }

    /// A resize changes the grid's shape between frames. The content may be
    /// identical cell for cell, so a per-cell diff would write nothing and
    /// leave the reflowed screen stale; a shape change must be a full repaint
    /// in either direction (A->B and B->A).
    #[test]
    fn a_shape_change_is_a_full_repaint() {
        fn solid(n: usize, ch: char, color: [u8; 3]) -> Vec<Vec<Cell>> {
            let row = vec![
                Cell {
                    ch,
                    fg: color,
                    bg: color
                };
                n
            ];
            (0..n).map(|_| row.clone()).collect()
        }

        let base = solid(10, ' ', [10, 20, 30]);
        let mut out = Vec::new();
        draw_grid(&mut out, &base, None).unwrap();
        let full = out.len();

        // Shrink: same solid content, smaller grid -> still a full repaint.
        let smaller = solid(8, ' ', [10, 20, 30]);
        let mut out = Vec::new();
        draw_grid(&mut out, &smaller, Some(&base)).unwrap();
        assert!(
            out.len() > full / 2,
            "a shrink must repaint the whole grid, got {} bytes (full was {full})",
            out.len()
        );

        // Grow back: the old shape matches base again, but the terminal
        // reflowed in between -> still a full repaint.
        let mut out = Vec::new();
        draw_grid(&mut out, &base, Some(&smaller)).unwrap();
        assert!(
            out.len() > full / 2,
            "growing back to the old shape must repaint too, got {} bytes (full was {full})",
            out.len()
        );
    }

    /// A `Resize` event must update the terminal size the loop adopts; a
    /// zero-size report (some terminals mid-resize) must be ignored.
    #[test]
    fn a_resize_event_updates_the_terminal_size() {
        let mut gameplay = GameplayInput::new(KeyBindings::retail_defaults());
        let (tx, _rx) = mpsc::channel();
        let mut input = TerminalInput {
            modifiers: Modifiers::NONE,
            size: (80, 24),
            taps: KeyTaps::new(),
            cursor: (0, 0),
            mouse_buttons: 0,
            world_clicks: WorldClicks::default(),
        };
        let mut model = termui::Model::new();
        assert!(!handle_event(
            Event::Resize(60, 30),
            &mut gameplay,
            &tx,
            0.0,
            &mut input,
            &mut model
        ));
        assert_eq!(
            input.size,
            (60, 30),
            "a resize must update the terminal size"
        );
        handle_event(
            Event::Resize(0, 0),
            &mut gameplay,
            &tx,
            0.0,
            &mut input,
            &mut model,
        );
        assert_eq!(
            input.size,
            (60, 30),
            "a zero-size report must not clobber the size"
        );
    }

    /// Out of text input mode, an Enter or Tab press hands the keys to the
    /// entry box (and types nothing); every other key — and a release — is
    /// the game's.
    #[test]
    fn enter_or_tab_takes_the_keys_over_to_the_entry_box() {
        let mut entry = ChatBox::default();
        for code in [KeyCode::Enter, KeyCode::Tab] {
            assert!(
                matches!(
                    route_key(code, true, KeyModifiers::NONE, &mut entry),
                    KeyRoute::Box { send: None }
                ),
                "{code:?} must open text input mode"
            );
            assert!(entry.active, "{code:?} must open text input mode");
            entry.deactivate();
        }
        for code in [KeyCode::Char('a'), KeyCode::Backspace, KeyCode::Left] {
            assert!(
                matches!(
                    route_key(code, true, KeyModifiers::NONE, &mut entry),
                    KeyRoute::Game
                ),
                "{code:?} is the game's out of the mode"
            );
            assert!(!entry.active);
        }
        assert!(
            matches!(
                route_key(KeyCode::Enter, false, KeyModifiers::NONE, &mut entry),
                KeyRoute::Game
            ),
            "a release does not open the box"
        );
    }

    /// In text input mode every key is the box's — a release included — and
    /// the line the box holds follows the keys: characters type, backspace
    /// shortens, Enter sends (and leaves the mode), Tab keeps the line and
    /// leaves the mode, a Ctrl combo types nothing.
    #[test]
    fn the_entry_box_takes_every_key_in_text_input_mode() {
        let mut entry = ChatBox::default();
        entry.activate();

        for c in ['h', 'i'] {
            assert!(
                matches!(
                    route_key(KeyCode::Char(c), true, KeyModifiers::NONE, &mut entry),
                    KeyRoute::Box { send: None }
                ),
                "a typed character is the box's"
            );
        }
        assert_eq!(entry.text, "hi");

        assert!(
            matches!(
                route_key(KeyCode::Char('c'), true, KeyModifiers::CONTROL, &mut entry),
                KeyRoute::Box { send: None }
            ),
            "a Ctrl combo is the box's"
        );
        assert_eq!(entry.text, "hi", "a Ctrl combo types nothing");

        assert!(matches!(
            route_key(KeyCode::Backspace, true, KeyModifiers::NONE, &mut entry),
            KeyRoute::Box { send: None }
        ));
        assert_eq!(entry.text, "h");

        assert!(
            matches!(
                route_key(KeyCode::Enter, true, KeyModifiers::NONE, &mut entry),
                KeyRoute::Box {
                    send: Some(line)
                } if line == "h"
            ),
            "Enter sends the line"
        );
        assert!(!entry.active, "sending leaves text input mode");
        assert!(entry.text.is_empty(), "the sent line leaves the box");

        // An empty Enter sends nothing and keeps the mode; a release in the
        // mode is the box's and changes nothing; a Tab keeps the line and
        // leaves the mode, after which the keys are the game's again.
        entry.activate();
        assert!(
            matches!(
                route_key(KeyCode::Enter, true, KeyModifiers::NONE, &mut entry),
                KeyRoute::Box { send: None }
            ),
            "an empty Enter sends nothing"
        );
        assert!(entry.active, "an empty Enter keeps the mode");

        for c in ['k', 'e', 'e', 'p'] {
            let _ = route_key(KeyCode::Char(c), true, KeyModifiers::NONE, &mut entry);
        }
        assert!(matches!(
            route_key(KeyCode::Char('z'), false, KeyModifiers::NONE, &mut entry),
            KeyRoute::Box { send: None }
        ));
        assert!(entry.active, "a release keeps the mode");
        assert_eq!(entry.text, "keep", "a release in the mode changes nothing");

        assert!(matches!(
            route_key(KeyCode::Tab, true, KeyModifiers::NONE, &mut entry),
            KeyRoute::Box { send: None }
        ));
        assert!(!entry.active, "Tab leaves text input mode");
        assert_eq!(entry.text, "keep", "Tab keeps the line in the box");

        assert!(
            matches!(
                route_key(KeyCode::Char('z'), false, KeyModifiers::NONE, &mut entry),
                KeyRoute::Game
            ),
            "after Tab the keys are the game's again"
        );
        assert_eq!(entry.text, "keep");
    }

    /// Pressing W and polling fires forward movement; releasing it and
    /// polling again clears it — with no camera of the terminal's own.
    #[test]
    fn holding_forward_key_sends_movement() {
        use ac_runtime::movement::MovementCommand;
        let mut gameplay = GameplayInput::new(KeyBindings::retail_defaults());
        let state = input_state(Modifiers::NONE);
        let mut out = Vec::new();
        gameplay.key(Key::W, true, &state, None, 0.0, &mut out);
        let _ = gameplay.tick(0.016, 0.016, &state, None, &mut out);
        assert!(
            out.iter().any(|o| {
                matches!(o, GameplayOutput::Movement(MovementCommand::Keys(k)) if k.forward)
            }),
            "holding W should send forward movement, got {out:?}"
        );
        gameplay.key(Key::W, false, &state, None, 0.02, &mut out);
        let _ = gameplay.tick(0.016, 0.02, &state, None, &mut out);
        assert!(
            out.iter().any(|o| {
                matches!(o, GameplayOutput::Movement(MovementCommand::Keys(k)) if !k.forward)
            }),
            "releasing W should clear the forward movement, got {out:?}"
        );
    }

    /// A tap of A (the retail turn-left key) over the terminal is one
    /// frame of turn-left: the held poll sees the press, the terminal's
    /// missing release is let go after it, and with no input at all the
    /// turn is cleared — the character does not keep spinning.
    #[test]
    fn a_terminal_turn_key_tap_stops_when_the_key_stops() {
        use ac_runtime::movement::MovementCommand;
        let mut gameplay = GameplayInput::new(KeyBindings::retail_defaults());
        let mut taps = KeyTaps::new();
        let state = input_state(Modifiers::NONE);
        let mut out = Vec::new();
        // The 'a' byte: a press, and nothing else from the terminal.
        gameplay.key(Key::A, true, &state, None, 0.0, &mut out);
        taps.key(Key::A, true, Modifiers::NONE);
        // The frame's held poll sees the key down.
        let _ = gameplay.tick(0.016, 0.016, &state, None, &mut out);
        assert!(
            out.iter().any(|o| {
                matches!(o, GameplayOutput::Movement(MovementCommand::Keys(k)) if k.turn_left)
            }),
            "the press should turn left, got {out:?}"
        );
        // The terminal never sends a release: the key is let go once its
        // autorepeat has gone quiet for the grace.
        let quiet = Instant::now() + key_hold_grace() + Duration::from_millis(1);
        for (key, mods) in taps.take_releases(quiet) {
            let s = input_state(mods);
            gameplay.key(key, false, &s, None, 0.02, &mut out);
        }
        let _ = gameplay.tick(0.016, 0.032, &state, None, &mut out);
        assert!(
            out.iter().any(|o| {
                matches!(o, GameplayOutput::Movement(MovementCommand::Keys(k)) if !k.turn_left)
            }),
            "the let-go should clear the turn, got {out:?}"
        );
        // No input at all: nothing fires, the turn stays cleared.
        out.clear();
        let _ = gameplay.tick(0.016, 0.048, &state, None, &mut out);
        let _ = gameplay.tick(0.016, 0.064, &state, None, &mut out);
        assert!(
            out.iter()
                .all(|o| { !matches!(o, GameplayOutput::Movement(MovementCommand::Keys(_))) }),
            "with no input there is no movement, got {out:?}"
        );
    }

    /// The hold bookkeeping: a fresh press is still within the grace (not
    /// let go); once the autorepeat has gone quiet for the grace the key is
    /// let go (a repeat did not stack it); a release that does arrive cancels
    /// the hold at once; let-gos come back in press order.
    #[test]
    fn key_taps_hold_across_the_autorepeat_stream() {
        let grace = key_hold_grace();
        let mut taps = KeyTaps::new();
        taps.key(Key::A, true, Modifiers::NONE);
        taps.key(Key::A, true, Modifiers::NONE); // autorepeat
        // A fresh press is still within the grace: nothing is let go yet.
        assert!(taps.take_releases(Instant::now()).is_empty());
        // Once the stream has gone quiet for the grace, the key is let go.
        let quiet = Instant::now() + grace + Duration::from_millis(1);
        assert_eq!(taps.take_releases(quiet), vec![(Key::A, Modifiers::NONE)]);
        assert!(KeyTaps::new().take_releases(quiet).is_empty());
        taps.key(Key::W, true, Modifiers::SHIFT);
        taps.key(Key::W, false, Modifiers::SHIFT); // a real release
        assert!(taps.take_releases(quiet).is_empty());
        taps.key(Key::A, true, Modifiers::NONE);
        taps.key(Key::D, true, Modifiers::NONE);
        assert_eq!(
            taps.take_releases(quiet),
            vec![(Key::A, Modifiers::NONE), (Key::D, Modifiers::NONE)]
        );
    }
}

//! The ratatui rendering mode (`--ui ratatui`): the session's status, the
//! event and chat feed, nearby objects, and a command line, drawn as terminal
//! widgets. The line console is unchanged; this is an alternative front end
//! for the same session. The session loop (in `main`) reports through
//! [`ModelLogger`]; the terminal is driven here.

use std::collections::VecDeque;
use std::io::{IsTerminal, Stdout};
use std::net::SocketAddr;
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use ac_runtime::session::{
    ObjectSummary, SessionCommand, SessionEvent, SessionHandle, SessionSummary,
};
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{self, EnterAlternateScreen, LeaveAlternateScreen};
use ratatui::backend::{Backend, CrosstermBackend};
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, Paragraph};
use ratatui::{Frame, Terminal};

use crate::options::{LogLevel, Options};

/// The largest feed kept (the scrollback; the terminal shows the tail).
const MAX_FEED: usize = 1000;

/// A feed line's kind, for colouring.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FeedKind {
    /// A progress line (a stamped `info`/`debug`).
    Info,
    /// A chat line (no stamp).
    Chat,
    /// A warning or a notice the console says about itself.
    Notice,
    /// An error.
    Error,
}

/// One line of the scrolling feed.
#[derive(Debug, Clone)]
struct FeedLine {
    text: String,
    kind: FeedKind,
}

/// The TUI's view of the session, shared between the session loop (writer)
/// and the render loop (reader).
pub struct TuiModel {
    /// The scrolling event and chat feed (bounded to [`MAX_FEED`]).
    feed: VecDeque<FeedLine>,
    /// The character's name, once entered.
    character: Option<String>,
    /// The login server.
    server: SocketAddr,
    /// A short state word (connecting, in world, logged off, ...).
    state: String,
    /// Nearby objects, nearest first (the reply to `/objects`).
    objects: Vec<String>,
    /// The log verbosity (gates detail lines).
    level: LogLevel,
}

impl TuiModel {
    /// A fresh model for a session on `server`.
    #[must_use]
    pub fn new(level: LogLevel, server: SocketAddr) -> Arc<Mutex<TuiModel>> {
        Arc::new(Mutex::new(Self {
            feed: VecDeque::new(),
            character: None,
            server,
            state: "starting".to_owned(),
            objects: Vec::new(),
            level,
        }))
    }

    /// Append a feed line, dropping the oldest past the bound.
    fn push(&mut self, kind: FeedKind, text: impl Into<String>) {
        self.feed.push_back(FeedLine {
            text: text.into(),
            kind,
        });
        if self.feed.len() > MAX_FEED {
            self.feed.pop_front();
        }
    }

    /// The feed's lines as styled items (for the render).
    fn feed_items(&self) -> Vec<ListItem<'_>> {
        self.feed
            .iter()
            .map(|l| {
                let style = match l.kind {
                    FeedKind::Error => Style::default().fg(Color::Red),
                    FeedKind::Notice => Style::default().fg(Color::Yellow),
                    FeedKind::Chat => Style::default(),
                    FeedKind::Info => Style::default().fg(Color::Gray),
                };
                ListItem::new(Line::from(Span::styled(l.text.clone(), style)))
            })
            .collect()
    }

    /// The objects pane items.
    fn object_items(&self) -> Vec<ListItem<'_>> {
        self.objects
            .iter()
            .map(|l| ListItem::new(Line::from(l.as_str())))
            .collect()
    }
}

/// Reports the session to a [`TuiModel`], the ratatui half of the
/// `SessionLog` the console's `Logger` also implements.
pub struct ModelLogger {
    /// The model to report into.
    pub model: Arc<Mutex<TuiModel>>,
    /// When the run started (for the feed's stamps).
    pub started: Instant,
}

impl ModelLogger {
    /// A `[ 12.345]` stamp, as the console's `Logger` prints.
    fn stamp(&self, text: &str) -> String {
        format!("[{:>8.3}] {text}", self.started.elapsed().as_secs_f64())
    }
}

impl super::SessionLog for ModelLogger {
    fn info(&self, text: &str) {
        let mut m = self.model.lock().unwrap();
        if m.level < LogLevel::Info {
            return;
        }
        m.push(FeedKind::Info, self.stamp(text));
    }

    fn warn(&self, text: &str) {
        let mut m = self.model.lock().unwrap();
        if m.level < LogLevel::Warn {
            return;
        }
        m.push(FeedKind::Notice, self.stamp(&format!("warn: {text}")));
    }

    fn error(&self, text: &str) {
        let mut m = self.model.lock().unwrap();
        m.push(FeedKind::Error, self.stamp(&format!("error: {text}")));
    }

    fn session_event(&self, event: &SessionEvent) {
        let mut m = self.model.lock().unwrap();
        m.state = match event {
            SessionEvent::EnteredWorld(_) => "in world".to_owned(),
            SessionEvent::LoggedOff { .. } => "logging off".to_owned(),
            SessionEvent::ReturnedToCharacterSelect(_) => "character select".to_owned(),
            SessionEvent::CharacterList(_) => "character select".to_owned(),
            SessionEvent::ServerName(_) => "connected".to_owned(),
            SessionEvent::Failed(_) => "failed".to_owned(),
            _ => m.state.clone(),
        };
        if let SessionEvent::EnteredWorld(character) = event {
            m.character = Some(character.name.clone());
        }
        for (kind, text) in session_event_lines(event) {
            m.push(kind, text);
        }
    }

    fn objects(&self, objects: &[ObjectSummary]) {
        let mut m = self.model.lock().unwrap();
        m.push(FeedKind::Info, format!("{} objects known:", objects.len()));
        m.objects = objects.iter().map(object_line).collect();
    }

    fn feed(&self, text: &str) {
        let mut m = self.model.lock().unwrap();
        m.push(FeedKind::Chat, text);
    }

    fn summary(&self, summary: &SessionSummary) {
        let mut m = self.model.lock().unwrap();
        m.state = "ended".to_owned();
        match &summary.error {
            Some(error) => m.push(FeedKind::Error, format!("session ended: {error}")),
            None => m.push(FeedKind::Info, "session ended cleanly".to_owned()),
        }
        m.push(
            FeedKind::Info,
            format!(
                "world: {} objects, player description {}",
                summary.world_objects,
                if summary.player_described {
                    "received"
                } else {
                    "missing"
                }
            ),
        );
    }
}

/// The feed lines a session event produces, with their kinds.
fn session_event_lines(event: &SessionEvent) -> Vec<(FeedKind, String)> {
    use SessionEvent::*;
    match event {
        PhaseChanged(phase) => vec![(FeedKind::Info, format!("phase: {phase:?}"))],
        CharacterList(list) => {
            let names: Vec<&str> = list.characters.iter().map(|c| c.name.as_str()).collect();
            vec![(
                FeedKind::Info,
                format!(
                    "account {} has characters: {}",
                    list.account_name,
                    names.join(", ")
                ),
            )]
        }
        ServerName(name) => vec![(
            FeedKind::Info,
            format!(
                "world {} ({} of {} connections)",
                name.world_name, name.current_connections, name.max_connections
            ),
        )],
        EnteredWorld(character) => vec![(
            FeedKind::Info,
            format!(
                "entered the world as {} (0x{:08X})",
                character.name, character.id
            ),
        )],
        LoginCompleteSent => vec![(FeedKind::Info, "login complete sent".to_owned())],
        LoggedOff { confirmed: true } => vec![(FeedKind::Info, "log off confirmed".to_owned())],
        LoggedOff { confirmed: false } => vec![(
            FeedKind::Notice,
            "log off not confirmed in time; disconnecting anyway".to_owned(),
        )],
        ReturnedToCharacterSelect(generation) => vec![(
            FeedKind::Info,
            format!("back at character selection (generation {generation})"),
        )],
        CharacterCreated { guid, name } => vec![(
            FeedKind::Info,
            format!("created character {name} (0x{guid:08X})"),
        )],
        CreationFailed { code, reason, name } => vec![(
            FeedKind::Notice,
            format!("the server refused to create {name}: {reason} ({code})"),
        )],
        Failed(error) => vec![(FeedKind::Error, error.to_string())],
    }
}

/// One nearby object as a pane line (the console's `Logger::objects` wording).
fn object_line(o: &ObjectSummary) -> String {
    let place = match (o.cell_id, o.origin, o.parent) {
        (Some(cell), Some([x, y, z]), _) => {
            format!("cell 0x{cell:08X} ({x:.1}, {y:.1}, {z:.1})")
        }
        (_, _, Some(parent)) => format!("held by 0x{parent:08X}"),
        _ => "not in the world".to_owned(),
    };
    let distance = o
        .distance
        .map_or_else(String::new, |d| format!(" {d:.1} m"));
    format!(
        "0x{:08X} #{:<5} {:<28} {place}{distance}",
        o.guid,
        o.local_id,
        o.name.as_deref().unwrap_or("?")
    )
}

/// Whether standard output is a terminal (the TUI needs one).
#[must_use]
pub fn is_terminal() -> bool {
    std::io::stdout().is_terminal()
}

/// The terminal in raw mode on the alternate screen; restoring both is
/// guaranteed on drop, so an early return never leaves the terminal broken.
struct Term {
    inner: Terminal<CrosstermBackend<Stdout>>,
}

impl Term {
    fn new() -> std::io::Result<Self> {
        let mut stdout = std::io::stdout();
        execute!(stdout, EnterAlternateScreen)?;
        terminal::enable_raw_mode()?;
        let backend = CrosstermBackend::new(stdout);
        let mut inner = Terminal::new(backend)?;
        inner.backend_mut().clear()?;
        Ok(Self { inner })
    }
}

impl Drop for Term {
    fn drop(&mut self) {
        let _ = terminal::disable_raw_mode();
        let _ = execute!(self.inner.backend_mut(), LeaveAlternateScreen);
        let _ = self.inner.show_cursor();
    }
}

/// The ratatui front end: run the session loop beside the terminal and draw
/// the model until the session ends. Returns the session's exit code.
pub fn run(
    model: Arc<Mutex<TuiModel>>,
    handle: SessionHandle,
    commands: Sender<SessionCommand>,
    options: Options,
) -> std::process::ExitCode {
    let mut term = match Term::new() {
        Ok(term) => term,
        Err(error) => {
            eprintln!("openac: could not start the terminal: {error}");
            return std::process::ExitCode::from(super::EXIT_SESSION_FAILED);
        }
    };
    let logger = ModelLogger {
        model: model.clone(),
        started: Instant::now(),
    };
    let session_thread = match std::thread::Builder::new()
        .name("session".into())
        .spawn(move || super::run_session_loop(handle, &logger, &options))
    {
        Ok(thread) => thread,
        Err(error) => {
            eprintln!("openac: could not start the session loop: {error}");
            return std::process::ExitCode::from(super::EXIT_SESSION_FAILED);
        }
    };
    let mut input = String::new();
    let mut prefs = crate::graphics::DisplayPrefs::load(None);
    let code;
    loop {
        {
            let model = model.lock().unwrap();
            let _ = term.inner.draw(|f| draw(f, &model, &input));
        }
        while event::poll(Duration::from_millis(16)).unwrap_or(false) {
            match event::read() {
                Ok(Event::Key(key)) => handle_key(key, &mut input, &model, &mut prefs, &commands),
                Ok(_) => {}
                Err(_) => break,
            }
        }
        if session_thread.is_finished() {
            code = session_thread
                .join()
                .unwrap_or_else(|_| std::process::ExitCode::from(super::EXIT_SESSION_FAILED));
            break;
        }
    }
    code
}

/// One keystroke: build the command line, or log off on Ctrl+C.
fn handle_key(
    key: KeyEvent,
    input: &mut String,
    model: &Arc<Mutex<TuiModel>>,
    prefs: &mut crate::graphics::DisplayPrefs,
    commands: &Sender<SessionCommand>,
) {
    match key.code {
        KeyCode::Enter => {
            let line = std::mem::take(input);
            send_line(&line, model, prefs, commands);
        }
        KeyCode::Backspace => {
            input.pop();
        }
        KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            let _ = commands.send(SessionCommand::Shutdown);
        }
        KeyCode::Char(c) if key.modifiers.is_empty() => {
            input.push(c);
        }
        _ => {}
    }
}

/// Send one typed line: echo it, route it the way the console does, and show
/// any notice it produces.
fn send_line(
    line: &str,
    model: &Arc<Mutex<TuiModel>>,
    prefs: &mut crate::graphics::DisplayPrefs,
    commands: &Sender<SessionCommand>,
) {
    {
        let mut m = model.lock().unwrap();
        m.push(FeedKind::Chat, format!("you> {line}"));
    }
    let routed = super::route_console_line(line, None, prefs, commands);
    if let Some(notice) = routed.notice {
        let mut m = model.lock().unwrap();
        m.push(FeedKind::Notice, notice);
    }
    if let Some(command) = routed.command {
        let _ = commands.send(command);
    }
}

/// The whole screen: a status header, the feed beside the objects, and the
/// command line with a hint.
fn draw(f: &mut Frame, m: &TuiModel, input: &str) {
    let area = f.area();
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(1),
            Constraint::Length(2),
        ])
        .split(area);
    draw_header(f, rows[0], m);
    draw_main(f, rows[1], m);
    draw_footer(f, rows[2], input);
}

/// The status line: character, server, and state.
fn draw_header(f: &mut Frame, area: Rect, m: &TuiModel) {
    let who = m.character.as_deref().unwrap_or("no character");
    let spans = vec![
        Span::styled(
            " openAC ".to_owned(),
            Style::default().fg(Color::White).bg(Color::Blue),
        ),
        Span::raw(format!("  {who}  ")),
        Span::styled(
            format!("@ {}", m.server),
            Style::default().fg(Color::DarkGray),
        ),
        Span::raw("  "),
        Span::styled(
            m.state.clone(),
            Style::default().add_modifier(Modifier::BOLD),
        ),
    ];
    let para = Paragraph::new(Line::from(spans))
        .block(Block::default().borders(Borders::ALL).title(" status "));
    f.render_widget(para, area);
}

/// The feed (left) beside the nearby objects (right).
fn draw_main(f: &mut Frame, area: Rect, m: &TuiModel) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(68), Constraint::Percentage(32)])
        .split(area);
    let feed =
        List::new(m.feed_items()).block(Block::default().borders(Borders::ALL).title(" feed "));
    f.render_widget(feed, cols[0]);
    let objects = List::new(m.object_items())
        .block(Block::default().borders(Borders::ALL).title(" objects "));
    f.render_widget(objects, cols[1]);
}

/// The command line (with a block cursor) and a hint.
fn draw_footer(f: &mut Frame, area: Rect, input: &str) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Length(1)])
        .split(area);
    let line = Paragraph::new(format!("> {input}\u{2588}"))
        .style(Style::default().add_modifier(Modifier::REVERSED));
    f.render_widget(line, rows[0]);
    let hint =
        Paragraph::new("Enter send · /objects /time /contracts /journal /quit · Ctrl+C log off")
            .style(Style::default().fg(Color::DarkGray));
    f.render_widget(hint, rows[1]);
}

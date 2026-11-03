//! Plugin icons: a PNG that passed the install rules (`validate_icon`),
//! decoded to 8-bit RGBA for the window. The rules and the decoder are the
//! plugin host's (`ac_plugin::icon`), which the client's plugin dock
//! uses for the same file.

pub use ac_plugin::icon::{PluginIcon, decode_icon, decode_png};

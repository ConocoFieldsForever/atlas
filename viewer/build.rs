//! Build script: on Windows, embed the app icon (`resources/atlas.ico`) into the .exe so it shows in
//! Explorer, the taskbar, and a desktop shortcut. A no-op on every other target.
fn main() {
    println!("cargo:rerun-if-changed=resources/atlas.ico");
    #[cfg(windows)]
    {
        let mut res = winresource::WindowsResource::new();
        res.set_icon("resources/atlas.ico");
        if let Err(e) = res.compile() {
            // Never fail the whole build over the icon: if the resource compiler is unavailable the
            // exe simply ships without an embedded icon (the runtime window icon still applies).
            println!("cargo:warning=atlas icon embed skipped: {e}");
        }
    }
}

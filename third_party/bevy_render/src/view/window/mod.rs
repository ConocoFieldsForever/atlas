use crate::renderer::WgpuWrapper;
use crate::{
    render_resource::{SurfaceTexture, TextureView},
    renderer::{RenderAdapter, RenderDevice, RenderInstance},
    Extract, ExtractSchedule, Render, RenderApp, RenderSystems,
};
use bevy_app::{App, Plugin};
use bevy_ecs::{entity::EntityHashMap, prelude::*};
use bevy_platform::collections::HashSet;
use bevy_utils::default;
use bevy_window::{
    CompositeAlphaMode, PresentMode, PrimaryWindow, RawHandleWrapper, Window, WindowClosing,
};
use core::{
    num::NonZero,
    ops::{Deref, DerefMut},
};
use tracing::{debug, warn};
use wgpu::{
    SurfaceConfiguration, SurfaceTargetUnsafe, TextureFormat, TextureUsages, TextureViewDescriptor,
};

pub mod screenshot;

use screenshot::ScreenshotPlugin;

pub struct WindowRenderPlugin;

impl Plugin for WindowRenderPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ScreenshotPlugin);

        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .init_resource::<ExtractedWindows>()
                .init_resource::<WindowSurfaces>()
                .add_systems(ExtractSchedule, extract_windows)
                .add_systems(
                    Render,
                    create_surfaces
                        .run_if(need_surface_configuration)
                        .before(prepare_windows),
                )
                .add_systems(Render, prepare_windows.in_set(RenderSystems::ManageViews));
        }
    }
}

pub struct ExtractedWindow {
    /// An entity that contains the components in [`Window`].
    pub entity: Entity,
    pub handle: RawHandleWrapper,
    pub physical_width: u32,
    pub physical_height: u32,
    pub present_mode: PresentMode,
    pub desired_maximum_frame_latency: Option<NonZero<u32>>,
    /// Note: this will not always be the swap chain texture view. When taking a screenshot,
    /// this will point to an alternative texture instead to allow for copying the render result
    /// to CPU memory.
    pub swap_chain_texture_view: Option<TextureView>,
    pub swap_chain_texture: Option<SurfaceTexture>,
    pub swap_chain_texture_format: Option<TextureFormat>,
    pub size_changed: bool,
    pub present_mode_changed: bool,
    pub alpha_mode: CompositeAlphaMode,
}

impl ExtractedWindow {
    fn set_swapchain_texture(&mut self, frame: wgpu::SurfaceTexture) {
        let texture_view_descriptor = TextureViewDescriptor {
            format: Some(frame.texture.format().add_srgb_suffix()),
            ..default()
        };
        self.swap_chain_texture_view = Some(TextureView::from(
            frame.texture.create_view(&texture_view_descriptor),
        ));
        self.swap_chain_texture = Some(SurfaceTexture::from(frame));
    }
}

#[derive(Default, Resource)]
pub struct ExtractedWindows {
    pub primary: Option<Entity>,
    pub windows: EntityHashMap<ExtractedWindow>,
}

impl Deref for ExtractedWindows {
    type Target = EntityHashMap<ExtractedWindow>;

    fn deref(&self) -> &Self::Target {
        &self.windows
    }
}

impl DerefMut for ExtractedWindows {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.windows
    }
}

fn extract_windows(
    mut extracted_windows: ResMut<ExtractedWindows>,
    mut closing: Extract<MessageReader<WindowClosing>>,
    windows: Extract<Query<(Entity, &Window, &RawHandleWrapper, Option<&PrimaryWindow>)>>,
    mut removed: Extract<RemovedComponents<RawHandleWrapper>>,
    mut window_surfaces: ResMut<WindowSurfaces>,
) {
    for (entity, window, handle, primary) in windows.iter() {
        if primary.is_some() {
            extracted_windows.primary = Some(entity);
        }

        let (new_width, new_height) = (
            window.resolution.physical_width().max(1),
            window.resolution.physical_height().max(1),
        );

        let extracted_window = extracted_windows.entry(entity).or_insert(ExtractedWindow {
            entity,
            handle: handle.clone(),
            physical_width: new_width,
            physical_height: new_height,
            present_mode: window.present_mode,
            desired_maximum_frame_latency: window.desired_maximum_frame_latency,
            swap_chain_texture: None,
            swap_chain_texture_view: None,
            size_changed: false,
            swap_chain_texture_format: None,
            present_mode_changed: false,
            alpha_mode: window.composite_alpha_mode,
        });

        // NOTE: Drop the swap chain frame here
        extracted_window.swap_chain_texture_view = None;
        extracted_window.size_changed = new_width != extracted_window.physical_width
            || new_height != extracted_window.physical_height;
        extracted_window.present_mode_changed =
            window.present_mode != extracted_window.present_mode;

        if extracted_window.size_changed {
            debug!(
                "Window size changed from {}x{} to {}x{}",
                extracted_window.physical_width,
                extracted_window.physical_height,
                new_width,
                new_height
            );
            extracted_window.physical_width = new_width;
            extracted_window.physical_height = new_height;
        }

        if extracted_window.present_mode_changed {
            debug!(
                "Window Present Mode changed from {:?} to {:?}",
                extracted_window.present_mode, window.present_mode
            );
            extracted_window.present_mode = window.present_mode;
        }
    }

    for closing_window in closing.read() {
        extracted_windows.remove(&closing_window.window);
        window_surfaces.remove(&closing_window.window);
    }
    for removed_window in removed.read() {
        extracted_windows.remove(&removed_window);
        window_surfaces.remove(&removed_window);
    }
}

struct SurfaceData {
    // TODO: what lifetime should this be?
    surface: WgpuWrapper<wgpu::Surface<'static>>,
    configuration: SurfaceConfiguration,
}

#[derive(Resource, Default)]
pub struct WindowSurfaces {
    surfaces: EntityHashMap<SurfaceData>,
    /// List of windows that we have already called the initial `configure_surface` for
    configured_windows: HashSet<Entity>,
}

impl WindowSurfaces {
    fn remove(&mut self, window: &Entity) {
        self.surfaces.remove(window);
        self.configured_windows.remove(window);
    }
}

/// (re)configures window surfaces, and obtains a swapchain texture for rendering.
///
/// NOTE: `get_current_texture` in `prepare_windows` can take a long time if the GPU workload is
/// the performance bottleneck. This can be seen in profiles as multiple prepare-set systems all
/// taking an unusually long time to complete, and all finishing at about the same time as the
/// `prepare_windows` system. Improvements in bevy are planned to avoid this happening when it
/// should not but it will still happen as it is easy for a user to create a large GPU workload
/// relative to the GPU performance and/or CPU workload.
/// This can be caused by many reasons, but several of them are:
/// - GPU workload is more than your current GPU can manage
/// - Error / performance bug in your custom shaders
/// - wgpu was unable to detect a proper GPU hardware-accelerated device given the chosen
///   [`Backends`](crate::settings::Backends), [`WgpuLimits`](crate::settings::WgpuLimits),
///   and/or [`WgpuFeatures`](crate::settings::WgpuFeatures). For example, on Windows currently
///   `DirectX 11` is not supported by wgpu 0.12 and so if your GPU/drivers do not support Vulkan,
///   it may be that a software renderer called "Microsoft Basic Render Driver" using `DirectX 12`
///   will be chosen and performance will be very poor. This is visible in a log message that is
///   output during renderer initialization.
///   Another alternative is to try to use [`ANGLE`](https://github.com/gfx-rs/wgpu#angle) and
///   [`Backends::GL`](crate::settings::Backends::GL) with the `gles` feature enabled if your
///   GPU/drivers support `OpenGL 4.3` / `OpenGL ES 3.0` or later.
pub fn prepare_windows(
    mut windows: ResMut<ExtractedWindows>,
    mut window_surfaces: ResMut<WindowSurfaces>,
    render_device: Res<RenderDevice>,
) {
    // ATLAS PATCH (vendored 0.17.3): upstream tolerates SurfaceError::Timeout only on Linux
    // mesa (AMD/Intel allowlist) and panics on every other acquire error. On Windows a GPU-bound
    // machine that enters the Win32 modal move/size loop stalls presentation until the 1 s
    // acquire budget expires (Timeout), and wgpu-hal's DXGI-FIFO pacing fence can surface the
    // same stall as a spurious device loss (`Other`, wgpu#9029) — under panic=abort either one
    // was a silent process death (field report: RX 6800, crash on window move/resize; upstream
    // bevy#13150/#21753 show the same panic strings). A missed acquire is a skippable frame,
    // exactly as the Linux arm and the post-reconfigure arm below already treat it, so treat
    // Timeout/Other that way on every platform. Lost joins Outdated in the reconfigure arm
    // (wgpu documents Lost as "recreate the swap chain"). OutOfMemory keeps the panic — that
    // one is a genuine unrecoverable state.
    use core::sync::atomic::{AtomicU32, Ordering};
    static SKIPPED_FRAMES: AtomicU32 = AtomicU32::new(0);

    for window in windows.windows.values_mut() {
        let window_surfaces = window_surfaces.deref_mut();
        let Some(surface_data) = window_surfaces.surfaces.get(&window.entity) else {
            continue;
        };

        let surface = &surface_data.surface;
        match surface.get_current_texture() {
            Ok(frame) => {
                window.set_swapchain_texture(frame);
            }
            Err(wgpu::SurfaceError::Outdated | wgpu::SurfaceError::Lost) => {
                render_device.configure_surface(surface, &surface_data.configuration);
                let frame = match surface.get_current_texture() {
                    Ok(frame) => frame,
                    Err(err) => {
                        // This is a common occurrence on X11 and Xwayland with NVIDIA drivers
                        // when opening and resizing the window.
                        warn!("Couldn't get swap chain texture after configuring. Cause: '{err}'");
                        continue;
                    }
                };
                window.set_swapchain_texture(frame);
            }
            Err(err @ (wgpu::SurfaceError::Timeout | wgpu::SurfaceError::Other)) => {
                // Rate-limited: during a sustained stall (window drag on a saturated GPU)
                // this fires every frame; log the first few and then one line per 128.
                let n = SKIPPED_FRAMES.fetch_add(1, Ordering::Relaxed);
                if n < 8 || n % 128 == 0 {
                    warn!(
                        "Couldn't get swap chain texture (skipping frame #{n}): {err}. \
                         Expected during window moves/resizes under GPU load; if it persists \
                         every frame the device may be lost — see the log for wgpu errors."
                    );
                }
            }
            Err(err @ wgpu::SurfaceError::OutOfMemory) => {
                panic!("Couldn't get swap chain texture, operation unrecoverable: {err}");
            }
        }
        window.swap_chain_texture_format = Some(surface_data.configuration.format);
    }
}

pub fn need_surface_configuration(
    windows: Res<ExtractedWindows>,
    window_surfaces: Res<WindowSurfaces>,
) -> bool {
    for window in windows.windows.values() {
        if !window_surfaces.configured_windows.contains(&window.entity)
            || window.size_changed
            || window.present_mode_changed
        {
            return true;
        }
    }
    false
}

// 2 is wgpu's default/what we've been using so far.
// 1 is the minimum, but may cause lower framerates due to the cpu waiting for the gpu to finish
// all work for the previous frame before starting work on the next frame, which then means the gpu
// has to wait for the cpu to finish to start on the next frame.
const DEFAULT_DESIRED_MAXIMUM_FRAME_LATENCY: u32 = 2;

/// Creates window surfaces.
pub fn create_surfaces(
    // By accessing a NonSend resource, we tell the scheduler to put this system on the main thread,
    // which is necessary for some OS's
    #[cfg(any(target_os = "macos", target_os = "ios"))] _marker: bevy_ecs::system::NonSendMarker,
    windows: Res<ExtractedWindows>,
    mut window_surfaces: ResMut<WindowSurfaces>,
    render_instance: Res<RenderInstance>,
    render_adapter: Res<RenderAdapter>,
    render_device: Res<RenderDevice>,
) {
    for window in windows.windows.values() {
        let data = window_surfaces
            .surfaces
            .entry(window.entity)
            .or_insert_with(|| {
                let surface_target = SurfaceTargetUnsafe::RawHandle {
                    raw_display_handle: window.handle.get_display_handle(),
                    raw_window_handle: window.handle.get_window_handle(),
                };
                // SAFETY: The window handles in ExtractedWindows will always be valid objects to create surfaces on
                let surface = unsafe {
                    // NOTE: On some OSes this MUST be called from the main thread.
                    // As of wgpu 0.15, only fallible if the given window is a HTML canvas and obtaining a WebGPU or WebGL2 context fails.
                    render_instance
                        .create_surface_unsafe(surface_target)
                        .expect("Failed to create wgpu surface")
                };
                let caps = surface.get_capabilities(&render_adapter);
                let formats = caps.formats;
                // For future HDR output support, we'll need to request a format that supports HDR,
                // but as of wgpu 0.15 that is not yet supported.
                // Prefer sRGB formats for surfaces, but fall back to first available format if no sRGB formats are available.
                let mut format = *formats.first().expect("No supported formats for surface");
                for available_format in formats {
                    // Rgba8UnormSrgb and Bgra8UnormSrgb and the only sRGB formats wgpu exposes that we can use for surfaces.
                    if available_format == TextureFormat::Rgba8UnormSrgb
                        || available_format == TextureFormat::Bgra8UnormSrgb
                    {
                        format = available_format;
                        break;
                    }
                }

                let configuration = SurfaceConfiguration {
                    format,
                    width: window.physical_width,
                    height: window.physical_height,
                    usage: TextureUsages::RENDER_ATTACHMENT,
                    present_mode: match window.present_mode {
                        PresentMode::Fifo => wgpu::PresentMode::Fifo,
                        PresentMode::FifoRelaxed => wgpu::PresentMode::FifoRelaxed,
                        PresentMode::Mailbox => wgpu::PresentMode::Mailbox,
                        PresentMode::Immediate => wgpu::PresentMode::Immediate,
                        PresentMode::AutoVsync => wgpu::PresentMode::AutoVsync,
                        PresentMode::AutoNoVsync => wgpu::PresentMode::AutoNoVsync,
                    },
                    desired_maximum_frame_latency: window
                        .desired_maximum_frame_latency
                        .map(NonZero::<u32>::get)
                        .unwrap_or(DEFAULT_DESIRED_MAXIMUM_FRAME_LATENCY),
                    // ATLAS PATCH: downgrade an unsupported alpha mode instead of aborting.
                    //
                    // `configure_surface` does NOT negotiate this. wgpu-core's only fallback list
                    // is for `Auto` and reads `[Opaque, Inherit]`, so `Auto` can never select a
                    // blending mode, and any explicitly-requested mode the surface does not
                    // advertise becomes `UnsupportedAlphaMode` -> `Validation` -> `panic!`. With
                    // `panic = "abort"` that is process death at startup: measured as 0xC0000409
                    // on `PostMultiplied` (NVIDIA) and on `PreMultiplied` (Intel). An
                    // `on_uncaptured_error` handler does not rescue it either, because `configure`
                    // only attaches the surface error sink on success.
                    //
                    // Which matters because the supported set is per-adapter and not knowable in
                    // advance: measured `OPAQUE | PRE_MULTIPLIED` on an RTX 5090 but
                    // `OPAQUE | INHERIT` on an Intel iGPU, and `bevy_window`'s own documentation
                    // recommends `PostMultiplied`, which neither of them supports. So the choice
                    // is between a degraded (opaque) window and a machine that cannot start.
                    alpha_mode: {
                        let want = match window.alpha_mode {
                            CompositeAlphaMode::Auto => wgpu::CompositeAlphaMode::Auto,
                            CompositeAlphaMode::Opaque => wgpu::CompositeAlphaMode::Opaque,
                            CompositeAlphaMode::PreMultiplied => {
                                wgpu::CompositeAlphaMode::PreMultiplied
                            }
                            CompositeAlphaMode::PostMultiplied => {
                                wgpu::CompositeAlphaMode::PostMultiplied
                            }
                            CompositeAlphaMode::Inherit => wgpu::CompositeAlphaMode::Inherit,
                        };
                        // `Auto` is always accepted, so it must not be measured against the list.
                        if want == wgpu::CompositeAlphaMode::Auto
                            || caps.alpha_modes.contains(&want)
                        {
                            want
                        } else {
                            // A blending request degrades to another BLENDING mode first: the
                            // caller asked for transparency, and the advertised set is just
                            // vendor dialect (measured: NVIDIA offers PreMultiplied, Intel offers
                            // Inherit, and the two composite identically). Only when the surface
                            // has no blending mode at all does this fall to Opaque.
                            let fallback = [
                                wgpu::CompositeAlphaMode::PreMultiplied,
                                wgpu::CompositeAlphaMode::PostMultiplied,
                                wgpu::CompositeAlphaMode::Inherit,
                            ]
                            .into_iter()
                            .find(|m| caps.alpha_modes.contains(m))
                            .unwrap_or(wgpu::CompositeAlphaMode::Opaque);
                            warn!(
                                "surface does not support composite alpha mode {:?} (supported: \
                                 {:?}) - using {:?} instead.{}",
                                want,
                                caps.alpha_modes,
                                fallback,
                                if fallback == wgpu::CompositeAlphaMode::Opaque {
                                    " A transparent window will be solid; everything else is \
                                     unaffected."
                                } else {
                                    ""
                                }
                            );
                            fallback
                        }
                    },
                    view_formats: if !format.is_srgb() {
                        vec![format.add_srgb_suffix()]
                    } else {
                        vec![]
                    },
                };

                render_device.configure_surface(&surface, &configuration);

                SurfaceData {
                    surface: WgpuWrapper::new(surface),
                    configuration,
                }
            });

        if window.size_changed || window.present_mode_changed {
            data.configuration.width = window.physical_width;
            data.configuration.height = window.physical_height;
            data.configuration.present_mode = match window.present_mode {
                PresentMode::Fifo => wgpu::PresentMode::Fifo,
                PresentMode::FifoRelaxed => wgpu::PresentMode::FifoRelaxed,
                PresentMode::Mailbox => wgpu::PresentMode::Mailbox,
                PresentMode::Immediate => wgpu::PresentMode::Immediate,
                PresentMode::AutoVsync => wgpu::PresentMode::AutoVsync,
                PresentMode::AutoNoVsync => wgpu::PresentMode::AutoNoVsync,
            };
            render_device.configure_surface(&data.surface, &data.configuration);
        }

        window_surfaces.configured_windows.insert(window.entity);
    }
}

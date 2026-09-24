// Local, offline image generation by shelling out to a `stable-diffusion.cpp` ("sd") binary — the
// "llama.cpp of images". No cloud API and no companion GUI (ComfyUI/A1111): LexiChat invokes the
// located `sd` CLI with a GGUF/safetensors diffusion model and reads back a PNG, which then renders
// inline via the same tool-image path as matplotlib charts and Mapbox static maps.
//
// The binary and model are NOT bundled (they're large + platform-specific); they're located from,
// in order: the user's explicit config path, the app data dir (`<data>/lexichat/image-gen/…`), or
// PATH (binary only). When neither is found, generate() returns a clear setup message the model
// relays to the user rather than a cryptic failure.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Per-install image-generation settings (held in AppState, pushed from the frontend). All fields
/// optional/defaulted so an empty config is valid and simply triggers auto-detection.
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct ImageGenConfig {
    /// Path to the `sd` (stable-diffusion.cpp) executable. Empty → auto-detect (data dir, then PATH).
    #[serde(default)] pub binary_path: String,
    /// Path to the diffusion model (.gguf / .safetensors / .ckpt). Empty → first model file found
    /// in `<data>/lexichat/image-gen/models/`.
    #[serde(default)] pub model_path: String,
    /// Default sampling steps. Turbo models want ~4; classic SD ~20. 0 → 20.
    #[serde(default)] pub steps: u32,
    /// Default square image size in px (512 / 768 / 1024). 0 → 512.
    #[serde(default)] pub size: u32,
    /// CFG (classifier-free guidance) scale. Turbo models want ~1.0; classic ~7. 0 → 7.
    #[serde(default)] pub cfg_scale: f32,
    /// Extra raw CLI args for advanced users (space-split), e.g. "--clip_l path --t5xxl path".
    #[serde(default)] pub extra_args: String,
    /// Optional separate VAE (`--vae`). Needed by models shipped as components rather than one
    /// checkpoint (Qwen-Image, FLUX, SD3): their VAE is a distinct file and is NOT interchangeable
    /// between model families.
    #[serde(default)] pub vae_path: String,
    /// Optional separate text encoder (`--llm`) — e.g. Qwen2.5-VL for Qwen-Image.
    #[serde(default)] pub text_encoder_path: String,
    /// Optional vision projector (`--llm_vision`, an `mmproj-*` file), required only for
    /// reference-image editing with a GGUF text encoder.
    #[serde(default)] pub vision_path: String,
    /// Hard limit on one generation, in seconds. 0 → automatic (see `generate_timeout`).
    #[serde(default)] pub timeout_secs: u64,
}

fn image_gen_dir() -> Option<PathBuf> {
    dirs::data_dir().map(|d| d.join("lexichat").join("image-gen"))
}

/// Where downloaded models live (auto-detected by resolve_model).
pub fn models_dir() -> Option<PathBuf> {
    image_gen_dir().map(|d| d.join("models"))
}

/// A sensible, well-known default model to pre-fill the download field. SD-Turbo is compact and
/// fast (~1-4 steps). It's editable in the UI — some models require sign-in/licence acceptance, in
/// which case the user pastes a direct URL to an ungated .safetensors/.gguf instead.
pub const DEFAULT_MODEL_URL: &str =
    "https://huggingface.co/stabilityai/sd-turbo/resolve/main/sd_turbo.safetensors";

/// Stream-download a model file into the models dir, reporting progress via `on_progress(received,
/// total)` and honoring `cancel`. Writes to a `.part` file and renames on success (so a cancelled
/// or failed download never looks like a complete model). Returns the final path.
pub async fn download_model(
    url: &str,
    filename: Option<&str>,
    cancel: &std::sync::atomic::AtomicBool,
    mut on_progress: impl FnMut(u64, Option<u64>),
) -> Result<PathBuf, String> {
    use std::io::Write;
    use std::sync::atomic::Ordering;

    let dir = models_dir().ok_or("Cannot resolve the app data directory.")?;
    std::fs::create_dir_all(&dir).map_err(|e| format!("Cannot create models dir: {e}"))?;

    // Derive a safe filename from the arg or the URL's last path segment.
    let raw = filename
        .map(str::to_string)
        .filter(|s| !s.trim().is_empty())
        .or_else(|| url.split('/').next_back().map(|s| s.split('?').next().unwrap_or(s).to_string()))
        .filter(|s| !s.is_empty())
        .ok_or("Cannot determine a filename from the URL — set one explicitly.")?;
    let name = std::path::Path::new(&raw)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("model.safetensors")
        .to_string();
    let dest = dir.join(&name);
    let part = dir.join(format!("{name}.part"));

    let client = reqwest::Client::builder()
        .build()
        .map_err(|e| format!("HTTP client error: {e}"))?;

    // Already have a complete copy? Skip the (multi-GB) transfer — HEAD the URL and, if the local
    // file matches the server's size, just report done. Makes "Download & use" idempotent so
    // re-selecting an installed model is instant instead of re-downloading it.
    if dest.is_file() {
        if let Ok(head) = client.head(url).send().await {
            if let (Some(remote), Ok(meta)) = (head.content_length(), std::fs::metadata(&dest)) {
                if remote > 0 && meta.len() == remote {
                    on_progress(remote, Some(remote));
                    return Ok(dest);
                }
            }
        }
    }

    let mut resp = client
        .get(url)
        .send()
        .await
        .map_err(|e| format!("Download failed: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!(
            "Download failed: HTTP {}. The model URL may be wrong, or the model requires \
             sign-in/licence acceptance — paste a direct link to an ungated .safetensors/.gguf.",
            resp.status()
        ));
    }
    let total = resp.content_length();
    let mut file = std::fs::File::create(&part).map_err(|e| format!("Cannot write file: {e}"))?;
    let mut received: u64 = 0;
    on_progress(0, total);
    loop {
        if cancel.load(Ordering::SeqCst) {
            drop(file);
            let _ = std::fs::remove_file(&part);
            return Err("Download cancelled.".to_string());
        }
        match resp.chunk().await.map_err(|e| format!("Download interrupted: {e}"))? {
            Some(chunk) => {
                file.write_all(&chunk).map_err(|e| format!("Write error: {e}"))?;
                received += chunk.len() as u64;
                on_progress(received, total);
            }
            None => break,
        }
    }
    file.flush().ok();
    drop(file);
    // Guard against a truncated download (connection closed early): if the server told us the size,
    // require we got all of it before promoting the .part file — a short file yields a corrupt model
    // that the engine rejects cryptically.
    if let Some(total) = total {
        if received < total {
            let _ = std::fs::remove_file(&part);
            return Err(format!("Download incomplete: received {received} of {total} bytes — please try again."));
        }
    }
    std::fs::rename(&part, &dest).map_err(|e| format!("Cannot finalize the model file: {e}"))?;
    Ok(dest)
}

/// Find an executable by name on the PATH (no extra crate). Returns the first hit.
fn find_on_path(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let candidate = dir.join(name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

/// Resolve the sd binary: explicit config → app data dir (where the bundled engine is provisioned
/// on first launch, or the user dropped it) → PATH.
fn resolve_binary(cfg: &ImageGenConfig) -> Option<PathBuf> {
    let explicit = cfg.binary_path.trim();
    if !explicit.is_empty() {
        // An explicit path is authoritative: use it, or fail (surfacing the config error) rather
        // than silently auto-detecting something else. Matches resolve_model.
        let p = PathBuf::from(explicit);
        return p.is_file().then_some(p);
    }
    if let Some(dir) = image_gen_dir() {
        for name in ["sd", "sd.exe", "sd-cli", "sd-cli.exe", "stable-diffusion", "stable-diffusion.exe"] {
            let p = dir.join(name);
            if p.is_file() {
                return Some(p);
            }
        }
    }
    find_on_path("sd")
        .or_else(|| find_on_path("sd-cli"))
        .or_else(|| find_on_path("stable-diffusion"))
}

/// Filename fragments that mark a file as a *component* of a model rather than the diffusion
/// model itself. Without this, auto-detection happily picks `Qwen_Image-VAE.safetensors` as the
/// model (it sorts first) and every generation fails with a confusing tensor error.
const COMPONENT_HINTS: [&str; 8] =
    ["vae", "text_encoder", "mmproj", "clip_l", "clip_g", "t5xxl", "_vl_", "vl-"];

fn looks_like_component(name: &str) -> bool {
    let low = name.to_ascii_lowercase();
    COMPONENT_HINTS.iter().any(|h| low.contains(h))
}

/// Build the model-selection arguments.
///
/// stable-diffusion.cpp distinguishes a *full checkpoint* (`-m`, everything in one file — SD1.5,
/// SDXL) from a bare *diffusion model* (`--diffusion-model`) whose VAE and text encoder are
/// supplied separately. Passing a component-style model with `-m` fails, so the presence of any
/// component path decides which form we use.
fn model_args(cfg: &ImageGenConfig, model: &Path) -> Vec<String> {
    let vae = cfg.vae_path.trim();
    let llm = cfg.text_encoder_path.trim();
    let vision = cfg.vision_path.trim();
    let multi = !vae.is_empty() || !llm.is_empty() || !vision.is_empty();

    let mut args: Vec<String> = Vec::new();
    args.push(if multi { "--diffusion-model".into() } else { "-m".into() });
    args.push(model.to_string_lossy().into_owned());
    if !vae.is_empty() { args.push("--vae".into()); args.push(vae.to_string()); }
    if !llm.is_empty() { args.push("--llm".into()); args.push(llm.to_string()); }
    if !vision.is_empty() { args.push("--llm_vision".into()); args.push(vision.to_string()); }
    args
}

/// Resolve the model: explicit config → first model file in `<data>/…/image-gen/models/`.
fn resolve_model(cfg: &ImageGenConfig) -> Option<PathBuf> {
    let explicit = cfg.model_path.trim();
    if !explicit.is_empty() {
        let p = PathBuf::from(explicit);
        return p.is_file().then_some(p);
    }
    let models = image_gen_dir()?.join("models");
    let mut entries: Vec<PathBuf> = std::fs::read_dir(&models)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .and_then(|x| x.to_str())
                .map(|x| matches!(x.to_ascii_lowercase().as_str(), "gguf" | "safetensors" | "ckpt"))
                .unwrap_or(false)
        })
        // A models dir holding a multi-file set also contains its VAE and text encoder; those are
        // never the diffusion model.
        .filter(|p| !p.file_name().and_then(|n| n.to_str()).map(looks_like_component).unwrap_or(false))
        .collect();
    entries.sort();
    entries.into_iter().next()
}

/// Pinned stable-diffusion.cpp release the engine is fetched from on first use.
// Bumped from master-820 for Qwen-Image / Qwen-Image-2.1 support (added upstream 2026-09-20).
// Newer builds renamed the CLI `sd` -> `sd-cli`; extract_engine normalizes either to `sd`, so
// existing installs keep working.
const ENGINE_RELEASE_TAG: &str = "master-908-88411ef";
const ENGINE_REPO: &str = "leejet/stable-diffusion.cpp";

/// Whether an sd engine is installed in the app data dir (drives the Images-tab status).
pub fn image_gen_dir_has_engine() -> bool {
    image_gen_dir()
        .map(|d| d.join(if cfg!(windows) { "sd.exe" } else { "sd" }).is_file())
        .unwrap_or(false)
}

/// Is a prebuilt engine available for this platform? (macOS-arm64/Metal, Windows-x64/Vulkan,
/// Linux-x64/Vulkan.) Other targets have no upstream prebuilt.
pub fn engine_supported() -> bool {
    use std::env::consts::{ARCH, OS};
    matches!((OS, ARCH), ("macos", "aarch64") | ("windows", "x86_64") | ("linux", "x86_64"))
}

/// Whether a release-asset name is the right engine build for this platform. Returns a priority
/// (lower = preferred: GPU build before CPU) or None if it doesn't match.
fn engine_asset_priority(name: &str) -> Option<u8> {
    use std::env::consts::{ARCH, OS};
    let n = name.to_ascii_lowercase();
    match (OS, ARCH) {
        ("macos", "aarch64") if n.contains("darwin") && n.contains("arm64") => Some(0),
        ("windows", "x86_64") if n.contains("win") && n.contains("vulkan") && n.contains("x64") => Some(0),
        ("windows", "x86_64") if n.contains("win") && n.contains("cpu") && n.contains("x64") => Some(1),
        ("linux", "x86_64") if n.contains("linux") && n.contains("vulkan") && n.contains("x86_64") => Some(0),
        ("linux", "x86_64") if n.contains("linux") && n.contains("x86_64") && !n.contains("rocm") && !n.contains("cuda") => Some(1),
        _ => None,
    }
}

/// Download and install the offline image engine (stable-diffusion.cpp) for this platform into the
/// app data dir on first use — the app ships without it (bundling unsigned binaries breaks macOS
/// notarization). Small (~50 MB), one-time. Reports progress and honors `cancel`.
pub async fn download_engine(
    cancel: &std::sync::atomic::AtomicBool,
    mut on_progress: impl FnMut(u64, Option<u64>),
) -> Result<PathBuf, String> {
    use std::io::Write;
    use std::sync::atomic::Ordering;

    if !engine_supported() {
        return Err("The offline image engine isn't available for this platform yet.".to_string());
    }
    let dest = image_gen_dir().ok_or("Cannot resolve the app data directory.")?;
    std::fs::create_dir_all(&dest).map_err(|e| format!("Cannot create image-gen dir: {e}"))?;

    let client = reqwest::Client::builder().build().map_err(|e| format!("HTTP client error: {e}"))?;

    // 1. Find the matching asset in the pinned release.
    let rel_url = format!("https://api.github.com/repos/{ENGINE_REPO}/releases/tags/{ENGINE_RELEASE_TAG}");
    let rel: serde_json::Value = client.get(&rel_url).header("User-Agent", "lexichat")
        .send().await.map_err(|e| format!("Cannot reach the release server: {e}"))?
        .json().await.map_err(|e| format!("Unexpected release data: {e}"))?;
    let mut best: Option<(u8, String)> = None;
    for a in rel.get("assets").and_then(|a| a.as_array()).map(|v| v.as_slice()).unwrap_or(&[]) {
        let name = a.get("name").and_then(|n| n.as_str()).unwrap_or("");
        if let Some(p) = engine_asset_priority(name) {
            if best.as_ref().map(|(bp, _)| p < *bp).unwrap_or(true) {
                let url = a.get("browser_download_url").and_then(|u| u.as_str()).unwrap_or("").to_string();
                best = Some((p, url));
            }
        }
    }
    let (_, url) = best.ok_or("No matching engine build found in the release.")?;

    // 2. Download the zip to a temp file (progress + cancel + truncation guard).
    let mut resp = client.get(&url).header("User-Agent", "lexichat").send().await
        .map_err(|e| format!("Engine download failed: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("Engine download failed: HTTP {}", resp.status()));
    }
    let total = resp.content_length();
    let tmp_zip = dest.join("engine-download.zip.part");
    let mut file = std::fs::File::create(&tmp_zip).map_err(|e| format!("Cannot write engine file: {e}"))?;
    let mut received: u64 = 0;
    on_progress(0, total);
    loop {
        if cancel.load(Ordering::SeqCst) {
            drop(file); let _ = std::fs::remove_file(&tmp_zip);
            return Err("Engine download cancelled.".to_string());
        }
        match resp.chunk().await.map_err(|e| format!("Engine download interrupted: {e}"))? {
            Some(chunk) => { file.write_all(&chunk).map_err(|e| format!("Write error: {e}"))?; received += chunk.len() as u64; on_progress(received, total); }
            None => break,
        }
    }
    file.flush().ok(); drop(file);
    if let Some(t) = total { if received < t {
        let _ = std::fs::remove_file(&tmp_zip);
        return Err(format!("Engine download incomplete ({received}/{t} bytes) — please retry."));
    }}

    // 3. Extract the sd executable + its shared libraries into the image-gen dir.
    let sd = extract_engine(&tmp_zip, &dest);
    let _ = std::fs::remove_file(&tmp_zip);
    let sd = sd?;

    // 4. Executable bit + de-quarantine (so a downloaded binary can run on macOS).
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if let Ok(meta) = std::fs::metadata(&sd) { let mut p = meta.permissions(); p.set_mode(0o755); let _ = std::fs::set_permissions(&sd, p); }
    }
    #[cfg(target_os = "macos")]
    { let _ = std::process::Command::new("xattr").arg("-dr").arg("com.apple.quarantine").arg(&dest).status(); }

    Ok(sd)
}

/// Extract the sd executable (normalized to `sd`/`sd.exe`) and its shared libraries from the release
/// zip into `dest`. Returns the path to the installed executable.
fn extract_engine(zip_path: &Path, dest: &Path) -> Result<PathBuf, String> {
    let f = std::fs::File::open(zip_path).map_err(|e| format!("Cannot open engine archive: {e}"))?;
    let mut archive = zip::ZipArchive::new(f).map_err(|e| format!("Bad engine archive: {e}"))?;
    let dest_exe = dest.join(if cfg!(windows) { "sd.exe" } else { "sd" });
    let mut found_exe = false;
    for i in 0..archive.len() {
        let mut entry = archive.by_index(i).map_err(|e| e.to_string())?;
        if !entry.is_file() { continue; }
        let raw = entry.name().to_string();
        let base = raw.rsplit(['/', '\\']).next().unwrap_or(&raw).to_string();
        let low = base.to_ascii_lowercase();
        let is_exe = matches!(low.as_str(), "sd-cli" | "sd" | "sd-cli.exe" | "sd.exe");
        let is_lib = low.ends_with(".dylib") || low.ends_with(".dll") || low.ends_with(".so") || low.contains(".so.");
        if !is_exe && !is_lib { continue; }
        let out = if is_exe { dest_exe.clone() } else { dest.join(&base) };
        let mut w = std::fs::File::create(&out).map_err(|e| format!("Cannot write {base}: {e}"))?;
        std::io::copy(&mut entry, &mut w).map_err(|e| format!("Extract error: {e}"))?;
        if is_exe { found_exe = true; }
    }
    if !found_exe { return Err("Engine archive did not contain the sd executable.".to_string()); }
    Ok(dest_exe)
}

const SETUP_HELP: &str = "Image generation isn't set up yet. Open Settings → the Images tab and \
    click \"Install image engine\" (a small one-time download), then pick and download a model. \
    Everything runs offline on your machine after that.";

const MODEL_HELP: &str = "No image model found. Put a diffusion model (.gguf or .safetensors — e.g. \
    an SD/SDXL-Turbo model) in the app's image-gen/models folder, or set its path in Settings → \
    Image Generation.";

/// How long a single generation may run before it's killed (a stuck/huge job shouldn't hang a turn).
/// Sized for single-file checkpoints (SD/SDXL-Turbo), which finish in seconds.
const GENERATE_TIMEOUT_SECS: u64 = 300;

/// Component models are far heavier: measured on an M-series Mac, Qwen-Image-2.1 Q4_K takes ~16.5s
/// per step at 768px after loading ~9 GB of weights, so the recommended 20 steps at 1024px runs
/// well past the single-file limit. Killing those at 300s made them look broken.
const COMPONENT_TIMEOUT_SECS: u64 = 2700;

/// Timeout for one generation: an explicit `timeout_secs` wins; otherwise component models
/// (separate VAE/text encoder) get the long budget and everything else the short one.
fn generate_timeout(cfg: &ImageGenConfig) -> u64 {
    if cfg.timeout_secs > 0 { return cfg.timeout_secs; }
    let component = !cfg.vae_path.trim().is_empty()
        || !cfg.text_encoder_path.trim().is_empty()
        || !cfg.vision_path.trim().is_empty();
    if component { COMPONENT_TIMEOUT_SECS } else { GENERATE_TIMEOUT_SECS }
}

/// Generate one image. Returns PNG bytes on success, or a human-readable error the model relays.
/// `size` overrides the config default; 0 falls back to config → 512.
/// Scale (w,h) so the long edge is at most `max_edge`, preserving aspect ratio, then round each
/// side to a multiple of 64 (Stable Diffusion requires that) with a floor of 64. Used for img2img
/// so an edited photo keeps its shape instead of being squashed into a square.
fn fit_dims(w: u32, h: u32, max_edge: u32) -> (u32, u32) {
    if w == 0 || h == 0 { return (512, 512); }
    let max_edge = if max_edge == 0 { 1024 } else { max_edge };
    let long = w.max(h);
    let scale = if long > max_edge { max_edge as f64 / long as f64 } else { 1.0 };
    let round64 = |v: f64| -> u32 { (((v / 64.0).round() as u32).max(1)) * 64 };
    (round64(w as f64 * scale), round64(h as f64 * scale))
}

/// Rasterize a mask from a tiny normalized-shape DSL (used for model-driven region edits). Shapes
/// are separated by ';', each either `rect x y w h` (top-left + size) or `ellipse cx cy rx ry`
/// (centre + radii), all as fractions 0..1 of the image. White (255) marks the region to edit; the
/// rest is black (kept). Returns a grayscale PNG at `w`×`h`, or None if no valid shape parsed.
pub fn rasterize_mask(dsl: &str, w: u32, h: u32) -> Option<Vec<u8>> {
    use image::{GrayImage, Luma};
    if w == 0 || h == 0 { return None; }
    let mut img = GrayImage::from_pixel(w, h, Luma([0u8]));
    let mut any = false;
    for shape in dsl.split(';').map(str::trim).filter(|s| !s.is_empty()) {
        let mut it = shape.split_whitespace();
        let kind = it.next().unwrap_or("");
        let nums: Vec<f32> = it.filter_map(|t| t.parse::<f32>().ok()).collect();
        if nums.len() < 4 { continue; }
        let (a, b, c, d) = (nums[0], nums[1], nums[2], nums[3]);
        match kind {
            "rect" => {
                let x0 = (a.clamp(0.0, 1.0) * w as f32) as i64;
                let y0 = (b.clamp(0.0, 1.0) * h as f32) as i64;
                let x1 = ((a + c).clamp(0.0, 1.0) * w as f32) as i64;
                let y1 = ((b + d).clamp(0.0, 1.0) * h as f32) as i64;
                for y in y0.max(0)..y1.min(h as i64) {
                    for x in x0.max(0)..x1.min(w as i64) {
                        img.put_pixel(x as u32, y as u32, Luma([255]));
                    }
                }
                any = true;
            }
            "ellipse" => {
                let (cx, cy, rx, ry) = (a * w as f32, b * h as f32, (c * w as f32).max(1.0), (d * h as f32).max(1.0));
                for y in 0..h {
                    for x in 0..w {
                        let dx = (x as f32 - cx) / rx;
                        let dy = (y as f32 - cy) / ry;
                        if dx * dx + dy * dy <= 1.0 { img.put_pixel(x, y, Luma([255])); }
                    }
                }
                any = true;
            }
            _ => {}
        }
    }
    if !any { return None; }
    // Feather the edges so the inpaint blends instead of showing a hard seam.
    let sigma = (w.min(h) as f32 * 0.012).max(2.0);
    let blurred = image::imageops::blur(&img, sigma);
    let mut buf = std::io::Cursor::new(Vec::new());
    image::DynamicImage::ImageLuma8(blurred).write_to(&mut buf, image::ImageFormat::Png).ok()?;
    Some(buf.into_inner())
}

/// Generate an image from `prompt` (text→image), or — when `init_image` is a real path to an
/// existing image — EDIT that image (image→image): the result keeps the source's composition and
/// only changes it as far as `strength` allows (0 = identical, 1 = ignore the source). This is what
/// lets "recolour the building in this photo pink" edit the actual photo rather than reimagining it.
///
/// When `mask_png` is supplied (with an init image) the edit is INPAINTED: only the white area of
/// the mask is regenerated, and the result is composited back onto the pixel-exact original so
/// everything outside the mask is unchanged — "update just this part" instead of reimagining.
pub async fn generate(
    cfg: &ImageGenConfig,
    prompt: &str,
    negative: Option<&str>,
    size: u32,
    steps: u32,
    seed: Option<i64>,
    init_image: Option<&str>,
    strength: Option<f32>,
    mask_png: Option<&[u8]>,
) -> Result<Vec<u8>, String> {
    let bin = resolve_binary(cfg).ok_or_else(|| SETUP_HELP.to_string())?;
    let model = resolve_model(cfg).ok_or_else(|| MODEL_HELP.to_string())?;

    let dim = if size > 0 { size } else if cfg.size > 0 { cfg.size } else { 512 };
    let steps = if steps > 0 { steps } else if cfg.steps > 0 { cfg.steps } else { 20 };
    let cfg_scale = if cfg.cfg_scale > 0.0 { cfg.cfg_scale } else { 7.0 };
    // For img2img, derive the output size from the source image (preserve aspect, cap the long edge
    // — a full-res photo would be far too slow/large for the CPU/GPU sd build). `size`, if given,
    // caps the long edge; otherwise 1024. For txt2img keep the square `dim`.
    let (out_w, out_h) = match init_image {
        Some(p) => match imagesize::size(p) {
            Ok(d) => fit_dims(d.width as u32, d.height as u32, if size > 0 { size } else { 1024 }),
            Err(e) => return Err(format!("Could not read the source image '{p}': {e}")),
        },
        None => (dim, dim),
    };

    let out_dir = std::env::temp_dir().join("lexichat-imagegen");
    std::fs::create_dir_all(&out_dir).map_err(|e| format!("Cannot create output dir: {e}"))?;
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let out = out_dir.join(format!("img-{nanos}.png"));

    // Inpaint masking only applies when we have an init image to edit. Decode the mask, resize it to
    // the working resolution, and write it next to the output for sd to consume via --mask.
    let mask_file: Option<std::path::PathBuf> = match (init_image, mask_png) {
        (Some(_), Some(m)) => {
            let decoded = image::load_from_memory(m)
                .map_err(|e| format!("Could not read the edit mask: {e}"))?
                .to_luma8();
            let resized = image::imageops::resize(&decoded, out_w, out_h, image::imageops::FilterType::Triangle);
            let mf = out_dir.join(format!("mask-{nanos}.png"));
            resized.save(&mf).map_err(|e| format!("Could not write the edit mask: {e}"))?;
            Some(mf)
        }
        _ => None,
    };

    let mut cmd = tokio::process::Command::new(&bin);
    cmd.args(model_args(cfg, &model));
    cmd.arg("-p").arg(prompt)
        .arg("-o").arg(&out)
        .arg("-W").arg(out_w.to_string())
        .arg("-H").arg(out_h.to_string())
        .arg("--steps").arg(steps.to_string())
        .arg("--cfg-scale").arg(format!("{cfg_scale}"));
    if let Some(init) = init_image {
        // img2img: in sd.cpp, supplying an init image in the default `img_gen` mode denoises FROM
        // that image (no separate mode flag exists in this build), keeping the source and changing
        // it only up to `strength` (default 0.6 — enough to repaint an object while preserving the
        // overall photo; lower stays closer to the original).
        // With a mask (inpaint) we want a strong change WITHIN the region — the surrounding photo is
        // protected by the composite step regardless — so default higher there; plain img2img
        // defaults gentler to avoid reimagining the whole frame.
        let default_strength = if mask_file.is_some() { 0.85 } else { 0.6 };
        let strength = strength.unwrap_or(default_strength).clamp(0.0, 1.0);
        cmd.arg("-i").arg(init)
            .arg("--strength").arg(format!("{strength}"));
        if let Some(mf) = &mask_file {
            cmd.arg("--mask").arg(mf);
        }
    }
    if let Some(n) = negative.map(str::trim).filter(|n| !n.is_empty()) {
        cmd.arg("-n").arg(n);
    }
    if let Some(s) = seed {
        cmd.arg("-s").arg(s.to_string());
    }
    for a in cfg.extra_args.split_whitespace() {
        cmd.arg(a);
    }

    // Make the dynamic loader find sd's runtime library, which sits in the same dir as the binary
    // (both the data-dir install and the provisioned-from-resources engine put them together).
    if let Some(dir) = bin.parent() {
        #[cfg(target_os = "macos")]
        cmd.env("DYLD_FALLBACK_LIBRARY_PATH", dir);
        #[cfg(target_os = "linux")]
        cmd.env("LD_LIBRARY_PATH", dir);
        #[cfg(target_os = "windows")]
        {
            // Windows resolves DLLs from PATH — prepend the binary's dir.
            let mut all = vec![dir.to_path_buf()];
            if let Some(p) = std::env::var_os("PATH") { all.extend(std::env::split_paths(&p)); }
            if let Ok(joined) = std::env::join_paths(all) { cmd.env("PATH", joined); }
        }
    }

    cmd.kill_on_drop(true);

    let run = cmd.output();
    let limit = generate_timeout(cfg);
    let output = match tokio::time::timeout(std::time::Duration::from_secs(limit), run).await {
        Ok(Ok(o)) => o,
        Ok(Err(e)) => return Err(format!("Failed to launch image generator ({}): {e}", bin.display())),
        Err(_) => return Err(format!("Image generation timed out after {limit}s — try fewer steps or a \
            smaller size, or raise the timeout in Settings → Images → Advanced.")),
    };
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let last = stderr.lines().rev().find(|l| !l.trim().is_empty()).unwrap_or("unknown error");
        return Err(format!("Image generation failed: {last}"));
    }

    let bytes = std::fs::read(&out)
        .map_err(|e| format!("Image generator ran but produced no output file: {e}"))?;
    let _ = std::fs::remove_file(&out);
    if let Some(mf) = &mask_file { let _ = std::fs::remove_file(mf); }
    if bytes.is_empty() {
        return Err("Image generator produced an empty file.".to_string());
    }

    // Inpaint composite: paste ONLY the masked region of the sd output back onto the pixel-exact
    // original, at the original's full resolution. Everything outside the mask is byte-for-byte the
    // source photo, so the edit is local ("update just this part") rather than a whole-frame redo.
    if let (Some(init), Some(mask)) = (init_image, mask_png) {
        if let Ok(edited) = composite_inpaint(init, &bytes, mask) {
            return Ok(edited);
        }
        // If compositing fails for any reason, fall back to the raw sd output rather than erroring.
    }
    Ok(bytes)
}

/// Composite the sd inpaint `edited_png` onto the original at `init_path`, blending by `mask_png`
/// (white = take the edit, black = keep the original), at the ORIGINAL image's resolution.
fn composite_inpaint(init_path: &str, edited_png: &[u8], mask_png: &[u8]) -> Result<Vec<u8>, String> {
    use image::imageops::FilterType;
    let orig = image::open(init_path).map_err(|e| format!("open original: {e}"))?.to_rgba8();
    let (w, h) = orig.dimensions();
    let edited = image::load_from_memory(edited_png).map_err(|e| format!("decode edited: {e}"))?;
    let edited = image::imageops::resize(&edited.to_rgba8(), w, h, FilterType::Lanczos3);
    let mask = image::load_from_memory(mask_png).map_err(|e| format!("decode mask: {e}"))?;
    let mask = image::imageops::resize(&mask.to_luma8(), w, h, FilterType::Triangle);

    let mut out = orig.clone();
    for (x, y, px) in out.enumerate_pixels_mut() {
        let a = mask.get_pixel(x, y)[0] as u32; // 0..255 blend weight
        if a == 0 { continue; }
        let e = edited.get_pixel(x, y);
        let o = *px;
        for c in 0..3 {
            px[c] = ((e[c] as u32 * a + o[c] as u32 * (255 - a)) / 255) as u8;
        }
    }
    let mut buf = std::io::Cursor::new(Vec::new());
    image::DynamicImage::ImageRgba8(out).write_to(&mut buf, image::ImageFormat::Png)
        .map_err(|e| format!("encode composite: {e}"))?;
    Ok(buf.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_binary_gives_setup_help() {
        // An empty config with no sd binary anywhere resolves to None → SETUP_HELP.
        let cfg = ImageGenConfig { binary_path: "/definitely/not/here/sd".into(), ..Default::default() };
        assert!(resolve_binary(&cfg).is_none());
    }

    #[test]
    fn fit_dims_preserves_aspect_caps_and_rounds_to_64() {
        // Landscape photo capped to 1024 long edge, sides rounded to /64.
        let (w, h) = fit_dims(4000, 3000, 1024);
        assert!(w <= 1024 && h <= 1024);
        assert_eq!(w % 64, 0);
        assert_eq!(h % 64, 0);
        assert!(w > h); // landscape stays landscape
        // Small image is not upscaled, just rounded to /64.
        let (w2, h2) = fit_dims(500, 500, 1024);
        assert_eq!((w2, h2), (512, 512));
        // Degenerate input falls back to a safe square.
        assert_eq!(fit_dims(0, 0, 1024), (512, 512));
    }

    #[test]
    fn rasterize_mask_marks_region_and_ignores_junk() {
        // A rect over the left half → left column white, right column black; feathered so exact
        // centre values are near the extremes rather than precisely 0/255.
        let png = rasterize_mask("rect 0 0 0.5 1.0", 128, 128).expect("some mask");
        let m = image::load_from_memory(&png).unwrap().to_luma8();
        assert!(m.get_pixel(10, 64)[0] > 200, "inside region should be near-white");
        assert!(m.get_pixel(118, 64)[0] < 55, "outside region should be near-black");
        // No parseable shape → None (caller then treats it as a whole-image img2img).
        assert!(rasterize_mask("garbage; nonsense", 64, 64).is_none());
        assert!(rasterize_mask("", 64, 64).is_none());
    }

    #[test]
    fn single_file_model_uses_dash_m() {
        let cfg = ImageGenConfig::default();
        let args = model_args(&cfg, Path::new("/m/sdxl.safetensors"));
        assert_eq!(args, vec!["-m".to_string(), "/m/sdxl.safetensors".to_string()]);
    }

    #[test]
    fn component_paths_switch_to_diffusion_model_form() {
        // Any one component is enough to make it a component-style model.
        let cfg = ImageGenConfig {
            vae_path: "/m/qwen_vae.safetensors".into(),
            text_encoder_path: "/m/qwen2.5vl-Q4_K_M.gguf".into(),
            ..Default::default()
        };
        let args = model_args(&cfg, Path::new("/m/qwen-Q4_K_M.gguf"));
        assert_eq!(args[0], "--diffusion-model", "component sets must not use -m: {args:?}");
        assert_eq!(args[1], "/m/qwen-Q4_K_M.gguf");
        assert!(args.windows(2).any(|w| w[0] == "--vae" && w[1] == "/m/qwen_vae.safetensors"), "{args:?}");
        assert!(args.windows(2).any(|w| w[0] == "--llm" && w[1] == "/m/qwen2.5vl-Q4_K_M.gguf"), "{args:?}");
        assert!(!args.iter().any(|a| a == "--llm_vision"), "vision is opt-in: {args:?}");

        // Vision projector only appears when set (editing with a GGUF text encoder).
        let cfg2 = ImageGenConfig { vision_path: "/m/mmproj-F16.gguf".into(), ..Default::default() };
        let args2 = model_args(&cfg2, Path::new("/m/qwen.gguf"));
        assert!(args2.windows(2).any(|w| w[0] == "--llm_vision" && w[1] == "/m/mmproj-F16.gguf"), "{args2:?}");
    }

    #[test]
    fn component_files_are_not_mistaken_for_the_model() {
        for name in ["Qwen_Image-VAE.safetensors", "qwen_2.5_vl_7b.safetensors",
                     "mmproj-Qwen3VL-8B-Instruct-F16.gguf", "clip_l.safetensors", "t5xxl_fp16.safetensors"] {
            assert!(looks_like_component(name), "{name} should be treated as a component");
        }
        for name in ["sdxl_turbo.safetensors", "qwen_image_2.1-Q4_K.gguf", "sd15.safetensors"] {
            assert!(!looks_like_component(name), "{name} is a diffusion model, not a component");
        }
    }

    #[test]
    fn timeout_scales_for_component_models_and_honours_override() {
        let single = ImageGenConfig::default();
        assert_eq!(generate_timeout(&single), GENERATE_TIMEOUT_SECS);
        // One component path is enough: these models load GBs of weights before sampling.
        let component = ImageGenConfig { text_encoder_path: "/m/llm.gguf".into(), ..Default::default() };
        assert_eq!(generate_timeout(&component), COMPONENT_TIMEOUT_SECS);
        // An explicit setting always wins, in both directions.
        let pinned = ImageGenConfig { timeout_secs: 60, ..component.clone() };
        assert_eq!(generate_timeout(&pinned), 60);
        let raised = ImageGenConfig { timeout_secs: 5000, ..Default::default() };
        assert_eq!(generate_timeout(&raised), 5000);
    }

    #[test]
    fn explicit_missing_model_is_none() {
        let cfg = ImageGenConfig { model_path: "/nope/model.gguf".into(), ..Default::default() };
        assert!(resolve_model(&cfg).is_none());
    }

    // Real end-to-end inpaint check — needs the installed engine+model, so ignored in CI.
    // Run locally with: cargo test masked_inpaint_keeps_outside_region -- --ignored --nocapture
    #[tokio::test]
    #[ignore]
    async fn masked_inpaint_keeps_outside_region() {
        use image::{RgbImage, Rgb};
        let dir = std::env::temp_dir().join("lexichat-inpaint-test");
        std::fs::create_dir_all(&dir).unwrap();
        let init = dir.join("init.png");
        // Solid grey source so any change is obvious.
        RgbImage::from_pixel(512, 512, Rgb([120, 120, 120])).save(&init).unwrap();
        let mask = rasterize_mask("rect 0.35 0.35 0.3 0.3", 512, 512).expect("mask");

        let cfg = ImageGenConfig::default(); // auto-detects the installed engine + model
        let png = generate(&cfg, "a bright red square", None, 512, 4, Some(0), init.to_str(), None, Some(&mask))
            .await.expect("generate should succeed with an installed engine");
        let out = image::load_from_memory(&png).unwrap().to_rgb8();
        assert_eq!(out.dimensions(), (512, 512), "output is at the original resolution");
        // A far corner is outside the (feathered) mask → must be the untouched original grey.
        assert_eq!(out.get_pixel(5, 5), &Rgb([120, 120, 120]), "outside the mask must be pixel-identical");
        // The centre of the mask should have changed.
        assert_ne!(out.get_pixel(256, 256), &Rgb([120, 120, 120]), "inside the mask must change");
    }

    #[tokio::test]
    async fn generate_reports_setup_when_unconfigured() {
        let cfg = ImageGenConfig { binary_path: "/no/sd".into(), model_path: "/no/m.gguf".into(), ..Default::default() };
        let err = generate(&cfg, "a cat", None, 512, 4, None, None, None, None).await.unwrap_err();
        assert!(err.contains("Image generation isn't set up") || err.contains("No image model"));
    }

    #[test]
    fn engine_asset_priority_matches_this_platform_or_none() {
        // Sanity: on a supported platform at least one representative asset name matches; on an
        // unsupported one, nothing does. (Names mirror the stable-diffusion.cpp release assets.)
        let samples = [
            "sd-master-bin-Darwin-macOS-arm64.zip",
            "sd-master-bin-win-vulkan-x64.zip",
            "sd-master-bin-Linux-Ubuntu-x86_64-vulkan.zip",
        ];
        let any = samples.iter().any(|s| engine_asset_priority(s).is_some());
        assert_eq!(any, engine_supported());
    }

    /// Real end-to-end for a COMPONENT model (diffusion model + separate VAE + text encoder):
    /// proves the `--diffusion-model`/`--vae`/`--llm` wiring actually drives the engine. Needs the
    /// Qwen-Image-2.1 set installed, so ignored by default. Run with:
    ///   cargo test qwen_component_model_generates -- --ignored --nocapture
    #[tokio::test]
    #[ignore]
    async fn qwen_component_model_generates() {
        let models = models_dir().expect("models dir");
        let cfg = ImageGenConfig {
            model_path: models.join("qwen_image_2.1-Q4_K.gguf").to_string_lossy().into_owned(),
            vae_path: models.join("qwen_image_2.1_vae_bf16.safetensors").to_string_lossy().into_owned(),
            text_encoder_path: models.join("Qwen3VL-8B-Instruct-Q4_K_M.gguf").to_string_lossy().into_owned(),
            ..Default::default()
        };
        // Text rendering is this family's party trick, and an obvious way to see it worked.
        let png = generate(&cfg, "a cafe chalkboard sign reading 'LexiChat', warm lighting, \
                            bright and well lit, detailed chalk lettering",
                           None, 1024, 20, Some(42), None, None, None)
            .await.expect("component model should generate");
        let img = image::load_from_memory(&png).expect("valid png");
        assert!(img.width() >= 512 && img.height() >= 512, "got {}x{}", img.width(), img.height());
        let out = std::env::temp_dir().join("qwen-2.1-test.png");
        std::fs::write(&out, &png).unwrap();
        println!("wrote {} ({} KB)", out.display(), png.len() / 1024);
    }

    // Real end-to-end: fetch + extract the engine from GitHub and confirm the binary runs. Hits the
    // network and installs into the real app-data dir, so it's #[ignore]d — run explicitly with
    // `cargo test download_engine_real -- --ignored`.
    #[tokio::test]
    #[ignore]
    async fn download_engine_real() {
        let cancel = std::sync::atomic::AtomicBool::new(false);
        let sd = download_engine(&cancel, |_, _| {}).await.expect("engine download+extract");
        assert!(sd.is_file(), "sd should be installed");
        let out = std::process::Command::new(&sd).arg("--help").output().expect("sd should run");
        let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
        assert!(text.contains("model") || text.contains("prompt"), "sd --help should print usage; got: {}", &text[..text.len().min(200)]);
    }
}

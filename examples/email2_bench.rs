#![allow(clippy::unwrap_used)]
//! Mixed-locale inbox benchmark for brokkr integration.
//!
//! Simulates a multilingual inbox matching dev-seed's "mixed" locale mode at
//! high thread counts. ~70% Latin, ~30% CJK/Arabic/Hindi/Korean - each message
//! draws from large character pools to maximize distinct glyph count.
//!
//! Target: 500-2000+ distinct glyphs, 50k+ glyph instances across 200+ buffers.
//! This stresses cold-path glyph processing, atlas growth, and GPU rendering
//! at realistic international workloads.
//!
//! Run via brokkr:  brokkr sluggrs hotpath --target email2
//! Run standalone:  cargo run --release --example email2-bench --features hotpath

use std::time::Instant;

use cosmic_text::{Attrs, Buffer, Family, FontSystem, Metrics, Shaping, Weight};
use sluggrs::{
    Cache, ColorMode, Resolution, SwashCache, TextArea, TextAtlas, TextBounds, TextRenderer,
    Viewport,
};

// As of hotpath 0.17, CountingAllocator is a plain generic type the consumer
// must declare as #[global_allocator] themselves - the crate no longer wires it
// up internally under hotpath-alloc. Without this, --alloc builds fall back to
// the system allocator and track_alloc/track_dealloc never fire, so every
// measured function silently reports 0 bytes.
#[cfg(feature = "hotpath-alloc")]
#[global_allocator]
static ALLOC: hotpath::CountingAllocator = hotpath::CountingAllocator::new();

const WIDTH: u32 = 1920;
const HEIGHT: u32 = 1080;

// ---------------------------------------------------------------------------
// Character pools - large enough to produce hundreds of distinct glyphs per
// script when sampled into messages. These approximate the character frequency
// you'd see in a real multilingual email inbox.
// ---------------------------------------------------------------------------

/// Common CJK characters (simplified Chinese + shared kanji). ~500 chars.
const CJK_POOL: &str = "\
的一是不了人我在有他这中大来上个国到说们为子和你地出会也时要就能\
对着事过好天没那里年还可多自后能去道得法都知用方以对学想所工个还\
看当小前开已将两面已从本行些现分实它要只何发什合金体如外成下力新\
电长风气声水火地山海森林花草树木河湖江溪泉瀑雨雪霜冰云雾虹霞晴\
明暗光影色形状态度感情爱恨喜怒哀乐悲欢离合生死存亡始终进退来去\
东西南北左右上下前后内外远近高低快慢大小多少长短轻重厚薄深浅宽窄\
强弱贫富贵贱美丑善恶真假是非对错好坏优劣胜负得失成败利害安危吉凶\
春夏秋冬朝暮晨昏日月星辰年岁时刻分秒今古昔未来将来现在过去\
父母兄弟姐妹夫妻子女孙祖家族亲戚朋友同事邻居老师学生医生护士\
工程师设计律师会计教授研究员科学家作技术经济政治文化历史地理数物\
化英语文数计算机网络系统程序软件硬件数据库服务器客户端接口协议\
安全测试部署运维监控日志错误警告信息调试优化性能内存处理线程进程";

/// Japanese hiragana + katakana + common kanji not in CJK_POOL above.
const JA_POOL: &str = "\
あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほ\
まみむめもやゆよらりるれろわをんがぎぐげござじずぜぞだぢづでど\
ばびぶべぼぱぴぷぺぽ\
アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホ\
マミムメモヤユヨラリルレロワヲンガギグゲゴザジズゼゾダヂヅデド\
バビブベボパピプペポ\
実行結果報告確認完了問題解決方法提案検討評価改善修正変更追加削除\
作成編集保存読込送信受信返信転送添付資料画像動画音声文書表図\
会議予定日程場所参加者議題決定事項連絡注意重要緊急至急";

/// Korean syllables - common ones used in business/tech email.
const KO_POOL: &str = "\
가나다라마바사아자차카타파하거너더러머버서어저처커터퍼허\
고노도로모보소오조초코토포호구누두루무부수우주추쿠투푸후\
회의결과보고서작성완료확인요청승인검토수정변경추가삭제\
프로젝트일정관리개발테스트배포운영모니터링로그분석\
시스템서버클라이언트네트워크데이터베이스보안성능최적화\
이메일메시지알림설정환경구성파일디렉토리경로\
안녕하세요감사합니다죄송합니다부탁드립니다말씀해주세요\
진행상황업데이트공유논의협의조율피드백리뷰\
화요일수요일목요일금요일월요일토요일일요일\
오전오후시간분초지금내일어제모레글피다음이번저번";

/// Arabic characters and common word fragments.
const AR_POOL: &str = "\
ابتثجحخدذرزسشصضطظعغفقكلمنهوي\
ءآأؤإئ\
الذيهذهمنعلىفيإلىأنمعكانلملهاقدعنبينكلحتىبعدقبلثممنذ\
تحليلأداءمحركعرضالنصوصمعالجةرسومياتتحدياتفريدةاتصالحروف\
أشكالسياقيةمختلفةمنحنىتشكيلطبقةإضافيةتعقيداتجاهاليمينلليسار\
معالجةخاصةتخطيطنتائجمسارتوصيةتحسينحجمأطلسنصوصثنائية\
اجتماعمشروعتقريرمراجعةتطويراختبارنشرصيانةمراقبةتنبيه\
رسالةإشعارتحديثجديدمهمعاجلمرفقملفصورةرابطمستند";

/// Hindi / Devanagari characters and common conjuncts.
const HI_POOL: &str = "\
अआइईउऊऋएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह\
ंःँ\
क्षत्रज्ञश्रद्वक्रप्रस्तभ्रन्दस्थम्पर्वत्वज्वद्भच्छ\
प्रदर्शनविश्लेषणपरिणामसाझाचुनौतियाँविशेषताएँसंयुक्त\
अक्षरशिरोरेखावर्णोंमात्राएँचिह्नअतिरिक्तबेंचमार्क\
परियोजनाविकासपरीक्षणतैनातीसंचालननिगरानीचेतावनीसूचना\
संदेशसमीक्षाअनुमोदनप्रतिक्रियाअद्यतनमहत्वपूर्णतत्काल\
बैठकप्रगतिस्थितिसमयसारणीकार्यसूचीप्राथमिकता";

// ---------------------------------------------------------------------------
// Simple RNG - deterministic, no external dependency. Xorshift64.
// ---------------------------------------------------------------------------

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_add(1)) // avoid zero state
    }

    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn range(&mut self, max: usize) -> usize {
        (self.next() as usize) % max
    }

    /// Pick `count` characters from a char pool, building a String.
    fn sample_chars(&mut self, pool: &[char], count: usize) -> String {
        let mut s = String::with_capacity(count * 3);
        for _ in 0..count {
            s.push(pool[self.range(pool.len())]);
        }
        s
    }
}

// ---------------------------------------------------------------------------
// Message generation
// ---------------------------------------------------------------------------

enum Script {
    Latin,
    Japanese,
    Chinese,
    Korean,
    Arabic,
    Hindi,
}

/// Latin subjects - enough variety to cover the ASCII+accented range.
const LATIN_SUBJECTS: &[&str] = &[
    "Re: Performance review of the Slug GPU text renderer",
    "Shipping estimate for Q3 iced integration milestone",
    "CI pipeline status: nightly breakage on wgpu 28.1",
    "Re: Memory usage spike on large font collections",
    "Thread pool sizing for parallel glyph processing",
    "Quarterly OKR alignment: text rendering team goals",
    "Re: Font rendering comparison across variable fonts",
    "GPU shader optimization: branch divergence analysis",
    "Texture growth stall on mixed-script content loads",
    "RFC: Retained text cache invalidation strategy",
    "Re: Subpixel positioning and stem darkening tuning",
    "Benchmark infrastructure: headless GPU profiling CI",
    "Re: cosmic_text 0.18 shaping regression with ligatures",
    "Atlas memory budget: per-workload capacity heuristics",
    "Code review: prepare_outline refactor for zero-copy",
    "Re: Noto CJK font loading latency in FontSystem::new()",
    "Meeting notes: shader optimization review (April 3)",
    "Ärger mit Übersetzungen - i18n text rendering Prüfung",
    "Résultats des tests: comparaison cryoglyph vs sluggrs",
    "Þórdís: Nordic glyph coverage and diacritic rendering",
];

const LATIN_BODIES: &[&str] = &[
    "Hi everyone, I've finished the initial performance review. The cold prepare path \
     is dominated by per-glyph allocation in build_bands(). Key findings: BandScratch \
     reuse saves ~100µs, prepare_outline() clone is waste, and batching write_buffer \
     from 92 calls to 1 saves ~20-90µs. GPU time is already good at 11µs headless.",
    "Quick update on the integration timeline. The sluggrs branch passes basic rendering \
     tests. Remaining: emoji fallback, trim() invalidation, ColorMode, stride alignment. \
     The Noto font family has 2,000+ glyphs per weight - mixed content easily hits 500+ \
     distinct glyphs on first render.",
    "The nightly CI broke overnight. Root cause: wgpu 28.1 changed TextureDescriptor \
     validation. Fix is trivial but exposed a deeper issue - no integration test exercises \
     the full prepare→render pipeline. Proposed: headless GPU test via llvmpipe.",
    "I investigated the RSS spike. FontSystem::new() loads all system fonts (~200 fonts, \
     ~40MB). cosmic_text doesn't lazy-load. Options: new_with_fonts() for specific fonts, \
     contribute lazy loading upstream, or accept the one-time cost.",
    "Thinking ahead to rayon for parallel cold-glyph processing. 4 threads: 3.6× speedup. \
     8 threads: 6.0× speedup. 16 threads: 6.5× (diminishing). Thread pool creation \
     overhead dominates below ~50 glyphs. For CJK (500+), 8 threads should scale linearly.",
    "Here's the OKR draft: O1 Ship sluggrs as default in iced. KR1 Pass all visual \
     regression tests. KR2 Cold prepare < 500µs for 100 Latin glyphs. KR3 GPU render \
     < 50µs for 1000 instances at 1080p. O2 International text without degradation.",
    "Notes from the shader review: pack i16 pairs into i32, precompute a,b from unshifted \
     coords, rejected compute shader rewrite. The fragment shader spends 60% in curve \
     evaluation loops. Storage buffer fetch is efficient on discrete GPUs but may be \
     bandwidth-bound on integrated.",
    "Font comparison results at 1920×1080: Noto Sans 10px has stem darkening + MSAA 4×. \
     14px clean, 0.5px dilation. 20px crisp. 48px+ could disable dilation. Inter variable \
     weight 400: 34 curves/glyph avg, 8 bands. All render correctly with half-pixel model.",
];

/// Build a message string for the given script, using character pools for
/// non-Latin scripts to ensure high distinct glyph count.
fn build_message(rng: &mut Rng, script: &Script, pools: &Pools) -> String {
    match script {
        Script::Latin => {
            let subj = LATIN_SUBJECTS[rng.range(LATIN_SUBJECTS.len())];
            let body = LATIN_BODIES[rng.range(LATIN_BODIES.len())];
            format!("{subj}\n\n{body}")
        }
        Script::Japanese => {
            let sn = 15 + rng.range(20);
            let bn = 80 + rng.range(120);
            let subj = rng.sample_chars(&pools.ja, sn);
            let body = rng.sample_chars(&pools.ja, bn);
            format!("{subj}\n\n{body}")
        }
        Script::Chinese => {
            let sn = 10 + rng.range(15);
            let bn = 60 + rng.range(100);
            let subj = rng.sample_chars(&pools.cjk, sn);
            let body = rng.sample_chars(&pools.cjk, bn);
            format!("{subj}\n\n{body}")
        }
        Script::Korean => {
            let sn = 12 + rng.range(18);
            let bn = 70 + rng.range(100);
            let subj = rng.sample_chars(&pools.ko, sn);
            let body = rng.sample_chars(&pools.ko, bn);
            format!("{subj}\n\n{body}")
        }
        Script::Arabic => {
            let sn = 15 + rng.range(20);
            let bn = 80 + rng.range(120);
            let subj = rng.sample_chars(&pools.ar, sn);
            let body = rng.sample_chars(&pools.ar, bn);
            format!("{subj}\n\n{body}")
        }
        Script::Hindi => {
            let sn = 12 + rng.range(18);
            let bn = 70 + rng.range(100);
            let subj = rng.sample_chars(&pools.hi, sn);
            let body = rng.sample_chars(&pools.hi, bn);
            format!("{subj}\n\n{body}")
        }
    }
}

/// Pre-computed char vectors from the const &str pools.
struct Pools {
    cjk: Vec<char>,
    ja: Vec<char>,
    ko: Vec<char>,
    ar: Vec<char>,
    hi: Vec<char>,
}

impl Pools {
    fn new() -> Self {
        // Merge CJK + JA pools for Japanese (kanji + kana)
        let cjk: Vec<char> = CJK_POOL.chars().filter(|c| !c.is_whitespace()).collect();
        let ja: Vec<char> = JA_POOL
            .chars()
            .chain(CJK_POOL.chars())
            .filter(|c| !c.is_whitespace())
            .collect();
        let ko: Vec<char> = KO_POOL.chars().filter(|c| !c.is_whitespace()).collect();
        let ar: Vec<char> = AR_POOL.chars().filter(|c| !c.is_whitespace()).collect();
        let hi: Vec<char> = HI_POOL.chars().filter(|c| !c.is_whitespace()).collect();
        Self {
            cjk,
            ja,
            ko,
            ar,
            hi,
        }
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let _guard = hotpath::HotpathGuardBuilder::new("sluggrs::email2_bench")
        .percentiles(&[50.0, 95.0, 99.0])
        .functions_limit(0)
        .build();

    let (device, queue) = create_device();
    let mut harness = RenderHarness::new(&device, &queue);

    let pools = Pools::new();
    let mut rng = Rng::new(42);

    let total_messages = 200;
    let intl_scripts = [
        Script::Japanese,
        Script::Chinese,
        Script::Korean,
        Script::Arabic,
        Script::Hindi,
    ];

    // Generate messages: ~70% Latin, ~30% non-Latin
    let mut messages: Vec<String> = Vec::with_capacity(total_messages);
    for i in 0..total_messages {
        let script = if rng.range(100) < 70 {
            Script::Latin
        } else {
            // Cycle through non-Latin scripts
            let idx = i % intl_scripts.len();
            match idx {
                0 => Script::Japanese,
                1 => Script::Chinese,
                2 => Script::Korean,
                3 => Script::Arabic,
                _ => Script::Hindi,
            }
        };
        messages.push(build_message(&mut rng, &script, &pools));
    }

    // Build cosmic_text buffers
    let mut buffers = build_buffers(&mut harness.font_system, &messages);
    let text_areas = layout_text_areas(&buffers);

    // Self-reported wall for brokkr --bench (mandatory elapsed_ms KV on the
    // kv-raw harness path). Excludes device/font init and buffer building.
    let measured_start = Instant::now();

    // -- Cold prepare: all caches empty --
    let cold_start = Instant::now();
    harness
        .prepare_areas(&text_areas)
        .expect("Cold prepare failed");
    let cold_us = cold_start.elapsed().as_micros();

    // Clear redraw flags for warm path
    for buf in &mut buffers {
        buf.set_redraw(false);
    }
    let text_areas = layout_text_areas(&buffers);

    let distinct_glyphs = harness.atlas.glyph_count();
    let total_instances: usize = text_areas
        .iter()
        .map(|a| {
            a.buffer
                .layout_runs()
                .flat_map(|run| run.glyphs.iter())
                .count()
        })
        .sum();

    // -- Warm prepare --
    let warm_iterations = 20u32;
    let warm_start = Instant::now();
    for _ in 0..warm_iterations {
        harness
            .prepare_areas(&text_areas)
            .expect("Warm prepare failed");
    }
    let warm_avg_us = warm_start.elapsed().as_micros() / warm_iterations as u128;

    // -- GPU render --
    for _ in 0..5 {
        harness.render_gpu();
    }
    let mut gpu_times: Vec<f64> = Vec::new();
    for _ in 0..20 {
        if let Some(ms) = harness.render_gpu() {
            gpu_times.push(ms);
        }
    }
    gpu_times.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let final_glyphs = harness.atlas.glyph_count();
    let buffer_elements = harness.atlas.buffer_elements_used();

    // -- Emit KV pairs for brokkr --
    eprintln!(
        "elapsed_ms={:.3}",
        measured_start.elapsed().as_secs_f64() * 1000.0
    );
    eprintln!("distinct_glyphs={distinct_glyphs}");
    eprintln!("final_glyphs={final_glyphs}");
    eprintln!("total_glyph_instances={total_instances}");
    eprintln!("text_areas={}", text_areas.len());
    eprintln!("buffer_elements={buffer_elements}");
    eprintln!("buffer_bytes={}", buffer_elements as u64 * 8);
    eprintln!("cold_prepare_us={cold_us}");
    eprintln!("warm_prepare_avg_us={warm_avg_us}");
    eprintln!("warm_iterations={warm_iterations}");

    if let Some(median) = gpu_times.get(gpu_times.len() / 2) {
        eprintln!("gpu_text_render_us={}", (*median * 1000.0) as u64);
    }
}

// ---------------------------------------------------------------------------
// Buffer construction
// ---------------------------------------------------------------------------

fn build_buffers(font_system: &mut FontSystem, messages: &[String]) -> Vec<Buffer> {
    messages
        .iter()
        .map(|text| {
            let mut buffer = Buffer::new(font_system, Metrics::new(14.0, 20.0));
            buffer.set_size(Some(WIDTH as f32 - 40.0), None);

            // Split on first double-newline for subject vs body styling
            let (subject, body) = text.split_once("\n\n").unwrap_or((text, ""));
            let spans: Vec<(&str, Attrs)> = vec![
                (
                    subject,
                    Attrs::new().family(Family::SansSerif).weight(Weight::BOLD),
                ),
                ("\n\n", Attrs::new()),
                (body, Attrs::new().family(Family::SansSerif)),
            ];

            buffer.set_rich_text(
                spans,
                &Attrs::new().family(Family::SansSerif),
                Shaping::Advanced,
                None,
            );
            buffer.shape_until_scroll(font_system, false);
            buffer
        })
        .collect()
}

fn layout_text_areas(buffers: &[Buffer]) -> Vec<TextArea<'_>> {
    let mut top = 20.0f32;

    buffers
        .iter()
        .map(|buffer| {
            let line_count = buffer.layout_runs().count();
            let height = (line_count as f32) * 20.0;

            let area = TextArea {
                buffer,
                left: 20.0,
                top,
                scale: 1.0,
                bounds: TextBounds {
                    left: 0,
                    top: 0,
                    right: WIDTH as i32,
                    bottom: HEIGHT as i32,
                },
                default_color: cosmic_text::Color::rgb(230, 230, 230),
            };
            top += height + 30.0;
            area
        })
        .collect()
}

// ---------------------------------------------------------------------------
// GPU infrastructure (shared pattern with hotpath.rs / email_bench.rs)
// ---------------------------------------------------------------------------

fn create_device() -> (wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
                force_fallback_adapter: false,
        apply_limit_buckets: false,
    }))
    .expect("No suitable GPU adapter found");

    let mut features = wgpu::Features::empty();
    if adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
        features |= wgpu::Features::TIMESTAMP_QUERY;
    }
    if adapter
        .features()
        .contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES)
    {
        features |= wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES;
    }

    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("sluggrs email2-bench"),
        required_features: features,
        ..Default::default()
    }))
    .expect("Failed to create device")
}

struct RenderHarness {
    renderer: TextRenderer,
    atlas: TextAtlas,
    viewport: Viewport,
    font_system: FontSystem,
    swash_cache: SwashCache,
    device: wgpu::Device,
    queue: wgpu::Queue,
    _render_target: wgpu::Texture,
    render_view: wgpu::TextureView,
    gpu_profiler: Option<wgpu_profiler::GpuProfiler>,
}

impl RenderHarness {
    fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let cache = Cache::new(device);
        let format = wgpu::TextureFormat::Bgra8UnormSrgb;
        let mut atlas =
            TextAtlas::with_color_mode(device, queue, &cache, format, ColorMode::Accurate);
        let renderer =
            TextRenderer::new(&mut atlas, device, wgpu::MultisampleState::default(), None);
        let mut viewport = Viewport::new(device, &cache);
        viewport.update(
            queue,
            Resolution {
                width: WIDTH,
                height: HEIGHT,
            },
        );

        let render_target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("offscreen render target"),
            size: wgpu::Extent3d {
                width: WIDTH,
                height: HEIGHT,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let render_view = render_target.create_view(&wgpu::TextureViewDescriptor::default());

        let gpu_profiler = if device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            Some(
                wgpu_profiler::GpuProfiler::new(
                    device,
                    wgpu_profiler::GpuProfilerSettings {
                        enable_timer_queries: true,
                        enable_debug_groups: false,
                        max_num_pending_frames: 8,
                    },
                )
                .expect("Failed to create GPU profiler"),
            )
        } else {
            None
        };

        Self {
            renderer,
            atlas,
            viewport,
            font_system: FontSystem::new(),
            swash_cache: SwashCache::new(),
            device: device.clone(),
            queue: queue.clone(),
            _render_target: render_target,
            render_view,
            gpu_profiler,
        }
    }

    fn prepare_areas(&mut self, areas: &[TextArea]) -> Result<(), sluggrs::PrepareError> {
        let encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        self.renderer.prepare(
            &self.device,
            &self.queue,
            &mut self.font_system,
            &mut self.atlas,
            &self.viewport,
            areas.iter().copied()
        )
    }

    fn render_gpu(&mut self) -> Option<f64> {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("gpu profiling pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.render_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                ..Default::default()
            });

            let query = self
                .gpu_profiler
                .as_ref()
                .map(|p| p.begin_query("text_render", &mut pass));

            self.renderer
                .render(&self.atlas, &self.viewport, &mut pass)
                .unwrap();

            if let (Some(profiler), Some(query)) = (&self.gpu_profiler, query) {
                profiler.end_query(&mut pass, query);
            }
        }

        if let Some(profiler) = &mut self.gpu_profiler {
            profiler.resolve_queries(&mut encoder);
        }

        self.queue.submit(std::iter::once(encoder.finish()));

        if let Some(profiler) = &mut self.gpu_profiler {
            let _ = profiler.end_frame();
            if let Some(results) =
                profiler.process_finished_frame(self.queue.get_timestamp_period())
            {
                for r in &results {
                    if let Some(time) = &r.time {
                        return Some((time.end - time.start) * 1000.0);
                    }
                }
            }
        }
        None
    }
}

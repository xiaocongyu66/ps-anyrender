use std::sync::Arc;

use anyrender::{Filter, NormalizedCoord, Paint, PaintRef, PaintScene, RenderContext};
use glifo::FontEmbolden;
use kurbo::{Affine, BezPath, Diagonal2, Rect, Shape, Stroke};
use peniko::{BlendMode, Color, Fill, FontData, ImageBrush, StyleRef};
use vello_cpu::{ImageSource, PaintType, Pixmap};

const DEFAULT_TOLERANCE: f64 = 0.1;

fn anyrender_paint_to_vello_cpu_paint<'a>(paint: PaintRef<'a>) -> PaintType {
    match paint {
        Paint::Solid(alpha_color) => PaintType::Solid(alpha_color),
        Paint::Gradient(gradient) => PaintType::Gradient(gradient.clone()),
        Paint::Image(image) => PaintType::Image(ImageBrush {
            #[cfg(not(feature = "experimental_image_cache"))]
            image: ImageSource::from_peniko_image_data(image.image),
            #[cfg(feature = "experimental_image_cache")]
            image: convert_image_cached(image.image),
            sampler: image.sampler,
        }),
        // TODO: custom paint
        Paint::Resource(_) => PaintType::Solid(peniko::color::palette::css::TRANSPARENT),
        Paint::Custom(_) => PaintType::Solid(peniko::color::palette::css::TRANSPARENT),
    }
}

#[cfg(feature = "experimental_image_cache")]
fn convert_image_cached(image: &peniko::ImageData) -> ImageSource {
    use std::collections::HashMap;
    use std::sync::{LazyLock, Mutex};
    static CACHE: LazyLock<Mutex<HashMap<u64, ImageSource>>> =
        LazyLock::new(|| Mutex::new(HashMap::new()));

    let mut map = CACHE.lock().unwrap();
    let id = image.data.id();
    map.entry(id)
        .or_insert_with(|| ImageSource::from_peniko_image_data(image))
        .clone()
}

/// One layer open on `render_ctx`, kept so it can be closed and reopened.
///
/// Everything `push_layer` was given, because reopening has to reproduce it
/// exactly: a clip that came back a fraction different would move the seam
/// between the two halves of the layer's content.
#[derive(Clone)]
struct OpenLayer {
    transform: Affine,
    clip: Option<BezPath>,
    blend: Option<BlendMode>,
    alpha: Option<f32>,
    #[cfg(feature = "filters")]
    filter: Option<vello_common::filter_effects::Filter>,
}

pub struct VelloCpuScenePainter {
    pub render_ctx: vello_cpu::RenderContext,
    pub resources: vello_cpu::Resources,
    /// The layers open on `render_ctx` right now, outermost first.
    ///
    /// A backdrop snapshot is only legal at depth zero: `render_to_pixmap`
    /// asserts `!wide.has_layers()`, and asking for one while a layer is open
    /// aborts the process rather than dropping an effect. Blitz-style painters
    /// reach `push_layer` several frames deep, so on a real page that depth is
    /// never zero and this backend used to decline every backdrop it was asked
    /// for. Recording the stack is what lets it unwind to zero, take the
    /// snapshot, and put the stack back. See `paint_filtered_backdrop`.
    layers: Vec<OpenLayer>,
}

impl VelloCpuScenePainter {
    /// A painter over a fresh `width` by `height` context.
    ///
    /// A constructor rather than a struct literal because the layer stack is
    /// private: it is an invariant of this type that it mirrors what is open on
    /// `render_ctx`, and a caller that could set it could break the snapshot.
    pub fn new(width: u16, height: u16) -> Self {
        Self {
            render_ctx: vello_cpu::RenderContext::new(width, height),
            resources: vello_cpu::Resources::new(),
            layers: Vec::new(),
        }
    }

    /// How many layers are open. Was a public field; kept as a reader because
    /// it is genuinely useful to a caller and costs nothing.
    pub fn open_layers(&self) -> usize {
        self.layers.len()
    }

    /// Push a layer onto the context and remember it.
    fn open(&mut self, layer: OpenLayer) {
        self.render_ctx.set_transform(layer.transform);
        #[cfg(feature = "filters")]
        let filter = layer.filter.clone();
        #[cfg(not(feature = "filters"))]
        let filter = None;
        self.render_ctx
            .push_layer(layer.clip.as_ref(), layer.blend, layer.alpha, None, filter);
        self.layers.push(layer);
    }

    /// Close every open layer, leaving the context at depth zero.
    ///
    /// The stack itself is kept: this is half of a round trip, and the caller
    /// puts it back with [`Self::rewind`].
    #[cfg(feature = "filters")]
    fn unwind(&mut self) {
        for _ in 0..self.layers.len() {
            self.render_ctx.pop_layer();
        }
    }

    /// Reopen everything [`Self::unwind`] closed, in the order it was pushed.
    #[cfg(feature = "filters")]
    fn rewind(&mut self) {
        for layer in std::mem::take(&mut self.layers) {
            self.open(layer);
        }
    }
    pub fn finish(mut self) -> Pixmap {
        let mut pixmap = Pixmap::new(self.render_ctx.width(), self.render_ctx.height());
        // `render_to_pixmap(resources, target)` became `render(target, resources)`
        // in vello_cpu 0.2: the same two operands, in the opposite order.
        self.render_ctx.render(&mut pixmap, &mut self.resources);
        pixmap
    }

    /// Draw the scene so far back over itself, filtered, inside `clip`.
    ///
    /// This is `backdrop-filter`. A layer filter runs over the layer's *own*
    /// content, so it can never express "blur what is behind me" — the input
    /// for that is `FilterSource::BackgroundImage`, which vello declares and
    /// implements nowhere. Rather than wait for it, take the backdrop the only
    /// way it is actually available: render what has been drawn up to this
    /// point, then draw that image back through a filtered layer clipped to the
    /// same shape. The element's own content is then drawn on top by the
    /// ordinary layer the caller pushes next.
    ///
    /// `render_to_pixmap` takes `&self`, which is what makes this possible
    /// mid-scene: the context is snapshotted, not consumed, and keeps
    /// accumulating afterwards.
    ///
    /// It costs a full render of the frame per backdrop-filtered layer, so it
    /// is priced for a handful of glass panels and not for a page of them.
    #[cfg(feature = "filters")]
    fn paint_filtered_backdrop(
        &mut self,
        backdrop_filter: Option<Arc<Filter>>,
        transform: Affine,
        clip: &impl Shape,
    ) {
        let Some(backdrop_filter) = backdrop_filter else {
            return;
        };
        // Same restriction as the layer filter path above: vello_cpu declines
        // to apply filters while multithreaded.
        if cfg!(feature = "multithreading") {
            return;
        }
        let Some(filter) = crate::filters::convert_filter(backdrop_filter) else {
            return;
        };
        let (width, height) = (self.render_ctx.width(), self.render_ctx.height());
        if width == 0 || height == 0 {
            return;
        }

        // `render_to_pixmap` asserts `!wide.has_layers()`, so the snapshot has
        // to be taken at depth zero. Blitz-style painters reach `push_layer`
        // several frames deep, so on a real page that depth is never zero: this
        // used to abort the process with "some layers haven't been popped yet",
        // and then, once that was caught, to decline every backdrop it was
        // asked for. Neither is an answer.
        //
        // Closing the stack and reopening it is the answer, and it is also the
        // *right* backdrop: what a filtered element sits over is the frame as
        // composited so far, which is exactly what closing every layer
        // produces. `anyrender_vello` reaches the same place by cutting the
        // frame into segments; it has to, because a GPU render consumes the
        // scene. Here the context is snapshotted rather than consumed, so the
        // round trip is the whole mechanism.
        //
        // The one thing it costs: a layer carrying alpha or a blend mode is
        // composited twice, once for the content before this point and once for
        // the content after, and those two differ from one composite of both
        // wherever they overlap. Clips, which is nearly all of what a page
        // pushes, are exact. `anyrender_vello` makes the same trade for the
        // same reason.
        self.unwind();
        let mut backdrop = Pixmap::new(width, height);
        self.render_ctx.render(&mut backdrop, &mut self.resources);
        self.rewind();

        let canvas = Rect::new(0.0, 0.0, f64::from(width), f64::from(height));
        self.render_ctx.set_transform(transform);
        self.render_ctx.push_layer(
            Some(&clip.into_path(DEFAULT_TOLERANCE)),
            None,
            None,
            None,
            Some(filter),
        );
        // The snapshot is in device pixels, so it is drawn untransformed
        // whatever transform the layer itself carries.
        self.render_ctx.set_transform(Affine::IDENTITY);
        self.render_ctx.set_paint(PaintType::Image(ImageBrush {
            image: ImageSource::Pixmap(Arc::new(backdrop)),
            sampler: peniko::ImageSampler::default(),
        }));
        self.render_ctx.fill_rect(&canvas);
        self.render_ctx.pop_layer();
    }
}

impl RenderContext for VelloCpuScenePainter {}
impl PaintScene for VelloCpuScenePainter {
    fn reset(&mut self) {
        self.render_ctx.reset();
        self.layers.clear();
    }

    fn push_layer(
        &mut self,
        blend: impl Into<BlendMode>,
        alpha: f32,
        transform: Affine,
        clip: &impl Shape,
        filter: Option<Arc<Filter>>,
        backdrop_filter: Option<Arc<Filter>>,
    ) {
        #[cfg(feature = "filters")]
        let filter = filter
            .and_then(crate::filters::convert_filter)
            .filter(|_| cfg!(not(feature = "multithreading")));

        // Without the feature there is nothing to convert a filter into, and
        // `OpenLayer` does not carry one, so the argument is simply dropped.
        #[cfg(not(feature = "filters"))]
        let _ = filter;

        #[cfg(feature = "filters")]
        self.paint_filtered_backdrop(backdrop_filter, transform, clip);
        #[cfg(not(feature = "filters"))]
        let _ = backdrop_filter;

        let entry = OpenLayer {
            transform,
            clip: Some(clip.into_path(DEFAULT_TOLERANCE)),
            blend: Some(blend.into()),
            alpha: Some(alpha),
            #[cfg(feature = "filters")]
            filter,
        };
        self.open(entry);
    }

    fn push_clip_layer(&mut self, transform: Affine, clip: &impl Shape) {
        // A clip is a layer like any other here, recorded the same way, because
        // the snapshot has to be able to close and reopen it. `push_clip_layer`
        // is only the cheaper spelling.
        self.open(OpenLayer {
            transform,
            clip: Some(clip.into_path(DEFAULT_TOLERANCE)),
            blend: None,
            alpha: None,
            #[cfg(feature = "filters")]
            filter: None,
        });
    }

    fn pop_layer(&mut self) {
        self.render_ctx.pop_layer();
        // A caller that pops more than it pushed is already in trouble; going
        // negative here would leave the stack disagreeing with the context,
        // and the snapshot unwinds by that stack.
        self.layers.pop();
    }

    fn stroke<'a>(
        &mut self,
        style: &Stroke,
        transform: Affine,
        paint: impl Into<PaintRef<'a>>,
        brush_transform: Option<Affine>,
        shape: &impl Shape,
    ) {
        self.render_ctx.set_transform(transform);
        self.render_ctx.set_stroke(style.clone());
        self.render_ctx
            .set_paint(anyrender_paint_to_vello_cpu_paint(paint.into()));
        self.render_ctx
            .set_paint_transform(brush_transform.unwrap_or(Affine::IDENTITY));
        self.render_ctx
            .stroke_path(&shape.into_path(DEFAULT_TOLERANCE));
    }

    fn fill<'a>(
        &mut self,
        style: Fill,
        transform: Affine,
        paint: impl Into<PaintRef<'a>>,
        brush_transform: Option<Affine>,
        shape: &impl Shape,
    ) {
        self.render_ctx.set_transform(transform);
        self.render_ctx.set_fill_rule(style);
        self.render_ctx
            .set_paint(anyrender_paint_to_vello_cpu_paint(paint.into()));
        self.render_ctx
            .set_paint_transform(brush_transform.unwrap_or(Affine::IDENTITY));
        self.render_ctx
            .fill_path(&shape.into_path(DEFAULT_TOLERANCE));
    }

    fn draw_glyphs<'a, 's: 'a>(
        &'a mut self,
        font: &'a FontData,
        font_size: f32,
        hint: bool,
        normalized_coords: &'a [NormalizedCoord],
        embolden: kurbo::Vec2,
        style: impl Into<StyleRef<'a>>,
        paint: impl Into<PaintRef<'a>>,
        _brush_alpha: f32,
        transform: Affine,
        glyph_transform: Option<Affine>,
        glyphs: impl Iterator<Item = anyrender::Glyph> + Clone,
    ) {
        self.render_ctx.set_transform(transform);
        self.render_ctx
            .set_paint(anyrender_paint_to_vello_cpu_paint(paint.into()));

        let style: StyleRef<'a> = style.into();
        match style {
            StyleRef::Fill(fill) => {
                self.render_ctx.set_fill_rule(fill);
                self.render_ctx
                    .glyph_run(&mut self.resources, font)
                    .font_size(font_size)
                    .hint(hint)
                    .normalized_coords(normalized_coords)
                    .font_embolden(FontEmbolden::new(Diagonal2::new(embolden.x, embolden.y)))
                    .glyph_transform(glyph_transform.unwrap_or_default())
                    .fill_glyphs(glyphs.map(|g| vello_cpu::Glyph {
                        id: g.id,
                        x: g.x,
                        y: g.y,
                    }));
            }
            StyleRef::Stroke(stroke) => {
                self.render_ctx.set_stroke(stroke.clone());
                self.render_ctx
                    .glyph_run(&mut self.resources, font)
                    .font_size(font_size)
                    .hint(hint)
                    .normalized_coords(normalized_coords)
                    .glyph_transform(glyph_transform.unwrap_or_default())
                    .stroke_glyphs(glyphs.map(|g| vello_cpu::Glyph {
                        id: g.id,
                        x: g.x,
                        y: g.y,
                    }));
            }
        }
    }
    fn draw_box_shadow(
        &mut self,
        transform: Affine,
        rect: Rect,
        color: Color,
        radius: f64,
        std_dev: f64,
    ) {
        // CSS spec: blur radius B means a Gaussian with σ = B/2 (the value
        // handed down from ps-blitz-paint is the raw CSS blur radius). Passing
        // it through as σ makes small shadows (the common 1px hairline) render
        // ~1.9× stronger than Chrome/Skia.
        let std_dev = std_dev * 0.5;
        self.render_ctx.set_transform(transform);
        self.render_ctx.set_paint(PaintType::Solid(color));
        // vello_cpu 0.2 added `invert`, which fills outside the rect instead of
        // inside so the same primitive can draw an inset shadow. This trait
        // method is the outer-shadow path, so `false` is the existing behaviour.
        self.render_ctx
            .fill_blurred_rounded_rect(&rect, radius as f32, std_dev as f32, false);
    }
}

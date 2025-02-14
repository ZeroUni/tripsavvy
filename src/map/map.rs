use egui::epaint::{emath::lerp, vec2, Color32, Pos2, Rect, Shape, Stroke};
use egui::text::LayoutJob;
use egui::{pos2, Galley, Rangef, Response, Sense, Ui, Vec2, Widget, WidgetInfo, WidgetType};
use futures::future::join_all;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use serde::{Deserialize, Serialize};
use lru::LruCache;
use rstar::{RTree, RTreeObject, AABB};

use super::map_tile::{Coordinate, MapTile, PixelBounds, PixelCoordinate};
use super::vector_tile::{FeatureValue, VectorFeature, VectorTile};

#[derive(Default, Clone, Serialize, Deserialize)]
pub struct MapState {
    center: PixelCoordinate,
    zoom: f32,
    dragging: bool,
    drag_start: Option<Pos2>,
    drag_delta: Vec2,
    cached_view: Vec<(u32, u32, u32, Rect, Rect)>,
}

impl MapState {
    pub fn load(ctx: &egui::Context, id: egui::Id) -> Self {
        ctx.data_mut(|d| d.get_persisted::<Self>(id).unwrap_or_default())
    }

    pub fn store(self, ctx: &egui::Context, id: egui::Id) {
        ctx.data_mut(|d| d.insert_persisted(id, self));
    }
}

#[derive(Debug, Clone)]
struct LabelPoint {
    position: Pos2,
    text: String,
    layout: Arc<Galley>,
    dimensions: Vec2,  
    importance: f32,   
}

impl LabelPoint {
    fn overlaps(&self, other: &LabelPoint) -> bool {
        // Create bounding boxes for both labels
        let self_rect = Rect::from_min_size(
            self.position,
            self.dimensions
        );
        let other_rect = Rect::from_min_size(
            other.position,
            other.dimensions
        );
        
        self_rect.intersects(other_rect)
    }

    fn adjust_position(&mut self, other: &LabelPoint, minimum_distance: f32) {
        // Push away based on overlap vector
        let delta = (self.position - other.position).normalized();
        let push_strength = minimum_distance * 0.5;
        
        // Calculate importance ratio, ensuring equal movement for equal importance
        let self_move_ratio = if self.importance == other.importance {
            0.5 // Equal movement for equal importance
        } else {
            other.importance / (self.importance + other.importance)
        };

        // Move self away from other
        self.position += delta * push_strength * self_move_ratio;
    }
}

impl PartialEq for LabelPoint {
    fn eq(&self, other: &Self) -> bool {
        self.position == other.position
    }
}

impl rstar::RTreeObject for LabelPoint {
    type Envelope = AABB<[f32; 2]>;

    fn envelope(&self) -> Self::Envelope {
        // Create an AABB that encompasses the entire label rectangle
        let min_x = self.position.x;
        let min_y = self.position.y;
        let max_x = min_x + self.dimensions.x;
        let max_y = min_y + self.dimensions.y;
        
        AABB::from_corners([min_x, min_y], [max_x, max_y])
    }
}

#[derive(Debug, Clone)]
struct LabelManager {
    tree: RTree<LabelPoint>,
    minimum_distance: f32,
}

impl LabelManager {
    pub fn new(minimum_distance: f32) -> Self {
        Self {
            tree: RTree::new(),
            minimum_distance,
        }
    }

    pub fn add_label(&mut self, mut label: LabelPoint) {
        // If a label with the same name already exists, skip
        if self.tree.iter().any(|l| l.text == label.text) {
            return;
        }
        // Check nearby labels within search radius
        let label_envelope = label.envelope();
        let search_area = AABB::from_corners(
            [
                label_envelope.lower()[0] - self.minimum_distance,
                label_envelope.lower()[1] - self.minimum_distance
            ],
            [
                label_envelope.upper()[0] + self.minimum_distance,
                label_envelope.upper()[1] + self.minimum_distance
            ]
        );
        
        let nearby: Vec<_> = self.tree.locate_in_envelope(&search_area).collect();
        
        let mut needs_adjustment = true;
        let max_iterations = 5;
        let mut iterations = 0;
        
        while needs_adjustment && iterations < max_iterations {
            needs_adjustment = false;
            
            for other in &nearby {
                if label.overlaps(other) {
                    label.adjust_position(other, self.minimum_distance);
                    needs_adjustment = true;
                }
            }
            
            iterations += 1;
        }
        
        // Only add if we found a good position
        if !needs_adjustment || iterations < max_iterations {
            self.tree.insert(label);
        }

    }
}

pub struct Map<'a> {
    id: egui::Id,
    tile_cache: &'a mut LruCache<(u32, u32, u32), MapTile>,
    vector_tile_cache: &'a mut LruCache<(u32, u32, u32), VectorTile>,
    viewport_size: Vec2,
    missing_tiles: &'a mut Vec<(u32, u32, u32)>,
    runtime: Arc<tokio::runtime::Runtime>,
}

impl<'a> Widget for Map<'a> {
    fn ui(mut self, ui: &mut Ui) -> Response {
        let mut state = MapState::load(ui.ctx(), self.id);

        let (rect, response) = ui.allocate_exact_size(
            self.viewport_size,
            Sense::click_and_drag()
        );

        ui.painter().rect(rect, 0.0, Color32::from_rgb(100, 0, 100), Stroke::new(1.0, Color32::WHITE));

        let map_painter = ui.painter().with_clip_rect(rect);
        let mut modified = false;
        // Handle interactions
        if response.dragged() {
            if !state.dragging {
                state.drag_start = response.hover_pos();
                state.dragging = true;
            }
            if let Some(current_pos) = response.hover_pos() {
                if let Some(start_pos) = state.drag_start {
                    let delta = current_pos - start_pos;
                    // longitude and latitude are not directly modifiable, so we create a Vec2 delta to add to the center
                    state.drag_delta = Vec2::new(-delta.x, -delta.y);
                    state.center = state.center.add_delta(state.drag_delta, state.zoom);

                    state.drag_start = Some(current_pos);
                }
            }
            modified = true;
        } else if state.dragging {
            state.dragging = false;
            state.drag_start = None;
        }

        let mut zoomed = false;
        // Handle zoom for pinch / touch
        let zoom_delta = ui.input(|i| i.zoom_delta()) - 1.0;
        if zoom_delta.abs() > f32::EPSILON {
            let zoom_new = lerp(Rangef::new(0.0, 1.0), zoom_delta.abs()) * zoom_delta.signum();
            state.zoom = (state.zoom + zoom_new).clamp(0.0, 20.0);
            zoomed = true;
            modified = true;
        }

        // Handle zoom for scroll
        let mut scroll = ui.input(|i| i.smooth_scroll_delta).y;
        if (scroll - 0.0).abs() > f32::EPSILON && !zoomed {
            // Normalize scroll further using tanh
            scroll = (scroll / 10.0).tanh();
            state.zoom = (state.zoom + scroll).clamp(0.0, 20.0);
            modified = true;
        }

        // If modified, update the view
        if modified || state.cached_view.is_empty() {
            state.cached_view = self.calculate_visible_tiles(rect, &state);
        }

        for (z, x, y, tile_map, uv) in &state.cached_view {
            let tile_rect = Rect::from_min_max(
                Pos2 {
                    x: rect.min.x + tile_map.min.x * rect.width(),
                    y: rect.min.y + tile_map.min.y * rect.height(),
                },
                Pos2 {
                    x: rect.min.x + tile_map.max.x * rect.width(), 
                    y: rect.min.y + tile_map.max.y * rect.height(),
                }
            );
            if let Some(tile) = self.tile_cache.get_mut(&(*z, *x, *y)) {
                
                map_painter.image(
                    tile.texture().clone(),
                    tile_rect,
                    *uv,
                    Color32::WHITE
                );
            } else {
                self.missing_tiles.push((*z, *x, *y));
                // Try to fetch the tile "above" it
                if let Some(shape) = self.fetch_parent_tile(*z, *x, *y, 3, ui, tile_rect, uv) {
                    map_painter.add(shape);
                } else {
                    map_painter.rect_filled(tile_rect, 0.0, Color32::GRAY);
                }
            }
        }

        // Render overlay
        let view = PixelBounds::from_center(state.center.clone(), state.zoom);
        let ref_painter = Arc::new(map_painter);
        let rendered_overlays = self.render_overlay(ui, rect, &view, ref_painter);
        //self.paint_overlay(ui, rect, rendered_overlays);

        // for (z, x, y, _, _) in &state.cached_view {
        //     if let Some((tile, coords)) = self.get_vector_tile(*z, *x, *y) {
        //         let (z, x, y) = coords;
        //         let boundaries = PixelBounds::from_x_y_zoom(x, y, z);
        //         for layer in tile.iter_layers() {
        //             let lowest_rank = layer.get_lowest_rank();
        //             if layer.name.eq("place") {
        //                 for feature in &layer.features {
        //                     match feature.geometry_type {
        //                         crate::map::vector_tile::GeometryType::Point => {
        //                             // Convert the relative coordinates to pixel coordinates
        //                             let coordinates = feature.coordinates.iter().map(|c| {
        //                                 let temp_coord = c.first().unwrap();
        //                                 let pixel_x = boundaries.left() + (boundaries.width()) * temp_coord.0 as f64;
        //                                 let pixel_y = boundaries.top() + (boundaries.height()) * temp_coord.1 as f64;
        //                                 PixelCoordinate::new(pixel_x, pixel_y)                                
        //                             }).collect::<Vec<PixelCoordinate>>();
        //                             // Draw the first point with its name:en 
        //                             if let Some(projected_point) = view.project_point(coordinates.first().unwrap()) {
        //                                 if let Some(name) = feature.properties.get("name:en") {
        //                                     if let Some(class) = feature.properties.get("class") {
        //                                         match class.as_string() {
        //                                             Some(s) => {
        //                                                 match s.as_str() {
        //                                                     _ => {
        //                                                         if feature.get_rank() > lowest_rank {
        //                                                             continue;
        //                                                         }
        //                                                         let paint_point = pos2(
        //                                                             rect.min.x + projected_point.x() as f32 * rect.width(),
        //                                                             rect.min.y + projected_point.y() as f32 * rect.height()
        //                                                             );
        //                                                         // Calculate font size based on zoom level, rank, and text length
        //                                                         let dynamic_font = egui::FontId::proportional(
        //                                                             17.0 - (feature.get_name().len() as f32 / 16.0).clamp(1.0, 12.0)
        //                                                         );
        //                                                         let layout = map_painter.layout(
        //                                                             feature.get_name_lang("en").to_string(),
        //                                                             dynamic_font,
        //                                                             Color32::WHITE,
        //                                                             f32::MAX
        //                                                         );
        //                                                         let size = layout.size();
        //                                                         let label = LabelPoint {
        //                                                             position: paint_point - size / 2.0,
        //                                                             text: feature.get_name_lang("en").to_string(),
        //                                                             layout,
        //                                                             dimensions: size,
        //                                                             importance: 1.0,
        //                                                         };                                    
        //                                                         label_manager.add_label(label);
        //                                                     }
        //                                                 }
        //                                             }
        //                                             None => {}
        //                                         }
        //                                     }
        //                                 }    
        //                             }
        //                         }
        //                         crate::map::vector_tile::GeometryType::Line => {
        //                         }
        //                         crate::map::vector_tile::GeometryType::Polygon => {
        //                         }
        //                     }
        //                 }    
        //             }
        //         }
        //     }
        // }
        // // Render all non-overlapping labels
        

        // Store updated state
        state.store(ui.ctx(), self.id);

        response
    }
}

impl<'a> Map<'a> {
    pub fn new(id_source: impl std::hash::Hash, tile_cache: &'a mut LruCache<(u32, u32, u32), MapTile>, vector_tile_cache: &'a mut LruCache<(u32, u32, u32), VectorTile>, missing_tiles: &'a mut Vec<(u32, u32, u32)>, runtime: Arc<tokio::runtime::Runtime>) -> Self {
        Self {
            id: egui::Id::new(id_source),
            tile_cache,
            vector_tile_cache,
            viewport_size: Vec2::new(1024.0, 1024.0),
            missing_tiles,
            runtime,
        }
    }

    pub fn with_viewport(id_source: impl std::hash::Hash, tile_cache: &'a mut LruCache<(u32, u32, u32), MapTile>, vector_tile_cache: &'a mut LruCache<(u32, u32, u32), VectorTile>, missing_tiles: &'a mut Vec<(u32, u32, u32)>, runtime: Arc<tokio::runtime::Runtime>, viewport_size: Vec2) -> Self {
        Self {
            id: egui::Id::new(id_source),
            tile_cache,
            vector_tile_cache,
            viewport_size,
            missing_tiles,
            runtime,
        }
    }

    pub fn viewport_size(mut self, size: Vec2) -> Self {
        self.viewport_size = size;
        self
    }

    fn calculate_visible_tiles(&self, viewport: Rect, state: &MapState) -> Vec<(u32, u32, u32, Rect, Rect)> {
        //let mut visible_tiles = Vec::new();
        
        // Clamp zoom to valid range
        let fidelity_zoom = state.zoom + 1.0;
        let z = fidelity_zoom.floor().max(0.0) as u32;

        // Get the geobounds of the viewport
        let bounds = PixelBounds::from_center(state.center.clone(), state.zoom);
    
        // Get the tile coordinates of the viewport
        let tiles = bounds.all_x_y_zoom(fidelity_zoom); // +1 to increase image fidelity
        // println!("Bounds: {:?}, Tiles: {:?}, Zoom: {}", bounds, tiles, state.zoom);

        let mut visible_tiles = Vec::new();

        for (x, y) in tiles {
            let tile_bounds = PixelBounds::from_x_y_zoom(x, y, z);
            
            for (tile_rect, uv_rect) in bounds.uv_map(&tile_bounds) {
                visible_tiles.push((z, x, y, tile_rect, uv_rect));
            }
        }

        // println!("Visible tiles: {:?}", visible_tiles);
        
        // Return nothing for now
        return visible_tiles;
    }

    fn fetch_parent_tile(&mut self, z: u32, x: u32, y: u32, depth: u32, ui: &Ui, tile_rect: Rect, uv: &Rect) -> Option<Shape> {
        if z <= 1 || depth == 0 {
            return None;
        }

        let parent_z = z - 1;
        let parent_x = x / 2;
        let parent_y = y / 2;

        if let Some(tile) = self.tile_cache.get(&(parent_z, parent_x, parent_y)) {
            let uv_x = (x % 2) as f32 / 2.0;
            let uv_y = (y % 2) as f32 / 2.0;
            
            let uv = Rect::from_min_max(
                Pos2::new(uv.min.x / 2.0 + uv_x, uv.min.y / 2.0 + uv_y),
                Pos2::new(uv.max.x / 2.0 + uv_x, uv.max.y / 2.0 + uv_y),
            );

            Some(Shape::image(
                tile.texture().clone(),
                tile_rect,
                uv,
                Color32::WHITE
            ))
        } else {
            // Try the next parent level recursively
            self.fetch_parent_tile(parent_z, parent_x, parent_y, depth - 1, ui, tile_rect, uv)
        }
    }

    fn get_vector_tile(&mut self, z: u32, x: u32, y: u32) -> Option<(&VectorTile, (u32, u32, u32))> {
        // Attempt 3 iterations to fetch the tile or the tile above it. Always clamp z to 14
        let z = z as i32;
        let diff = (z - 14).max(0);
        for i in 0..3 {
            let shift = i + diff;
            // Recalculate z, x, y (z = z-i, x = x/2^i, y = y/2^i)
            let z = z - shift;
            if z < 0 {
                return None;
            }
            let z = z as u32;
            let x = x >> shift;
            let y = y >> shift;
            if let Some(_) = self.vector_tile_cache.peek(&(z, x, y)) {
                return self.vector_tile_cache.get(&(z, x, y)).map(|tile| (tile, (z, x, y)));
            }
        }
        None
    }

    fn render_overlay(&mut self, ui: &egui::Ui, rect: Rect, view: &PixelBounds, map_painter: Arc<egui::Painter>) -> Vec<RenderedOverlay> {
        let mut overlay_layers = HashSet::new();
        for layer_name in vec!["place", "boundary", "water_name"] {
            overlay_layers.insert(layer_name);
        }

        let mut join_handles: Vec<JoinHandle<RenderedOverlay>> = Vec::new();
        let wrapped_rect = Arc::new(rect);
        let wrapped_view = Arc::new(view.clone());

        for (z, x, y, _, _) in &MapState::load(ui.ctx(), self.id).cached_view {
            if let Some((tile, coords)) = self.get_vector_tile(*z, *x, *y) {
                let boundaries = Arc::new(PixelBounds::from_x_y_zoom(coords.1, coords.2, coords.0));
                for layer in tile.layers.clone().iter() { // Clone to avoid borrowing issues, using Arc for performance
                    // Not a fan of this, but it works for now and shouldnt have a huge performance impact
                    if !overlay_layers.contains(&layer.name.as_str()) {
                        continue;
                    }
                    let min_rank = layer.get_lowest_rank();
                    for feature in layer.features.iter() {
                        // self.render_feature(&mut label_manager, rect, view, boundaries, feature, layer.get_lowest_rank(), &mut rendered_overlays, map_painter);
                        let wrapped_rect = wrapped_rect.clone();
                        let wrapped_view = wrapped_view.clone();
                        let wrapped_painter = map_painter.clone();
                        let wrapped_bounds = boundaries.clone();
                        let wrapped_feature = feature.clone();
                        let handle = self.runtime.spawn(async move {
                            // Render the feature
                            render_feature(
                                wrapped_rect,
                                wrapped_view,
                                wrapped_bounds,
                                wrapped_feature,
                                min_rank,
                                wrapped_painter,
                            )
                        });
                        join_handles.push(handle);
                    }
                }
            }
        }

        let results = self.runtime.block_on(async {
            join_all(join_handles).await
        });

        results.into_iter().filter_map(|r| r.ok()).collect()
    }
}

enum RenderedOverlay {
    Point(LabelPoint),
    Line(Shape),
    Polygon(Shape),
    None(),
}

fn render_feature(
    viewport: Arc<Rect>,
    view: Arc<PixelBounds>,
    boundaries: Arc<PixelBounds>,
    feature: Arc<VectorFeature>,
    lowest_rank: i32,
    map_painter: Arc<egui::Painter>,
) -> RenderedOverlay {
    match feature.geometry_type {
        crate::map::vector_tile::GeometryType::Point => {
            render_point(viewport, view, boundaries, feature, lowest_rank, map_painter)
        }
        crate::map::vector_tile::GeometryType::Line => {
            //self.render_line(viewport, view, boundaries, feature, rendered_overlays, map_painter)
            RenderedOverlay::None()
        }
        crate::map::vector_tile::GeometryType::Polygon => {
            //self.render_polygon(viewport, view, boundaries, feature, rendered_overlays, map_painter)
            // Ignore polygons until we've figured out lines
            RenderedOverlay::None()
        }
    }
}

fn render_point(
    viewport: Arc<Rect>,
    view: Arc<PixelBounds>,
    boundaries: Arc<PixelBounds>,
    feature: Arc<VectorFeature>,
    lowest_rank: i32,
    map_painter: Arc<egui::Painter>,
) -> RenderedOverlay {
    let coordinates = feature.coordinates.iter().map(|c| {
        let temp_coord = c.first().unwrap();
        let pixel_x = boundaries.left() + boundaries.width() * temp_coord.0 as f64;
        let pixel_y = boundaries.top() + boundaries.height() * temp_coord.1 as f64;
        crate::map::map_tile::PixelCoordinate::new(pixel_x, pixel_y)
    }).collect::<Vec<_>>();

    if let Some(projected_point) = view.project_point(coordinates.first().unwrap()) {
        if feature.get_rank() > lowest_rank {
            return RenderedOverlay::None();
        }

        let paint_point = pos2(
            viewport.min.x + projected_point.x() as f32 * viewport.width(),
            viewport.min.y + projected_point.y() as f32 * viewport.height(),
        );

        let (font_size, color) = get_point_style(feature.get_class(), feature.get_rank());
        let dynamic_font = egui::FontId::proportional(font_size);
        // Layout the text here, before creating the LabelPoint
        let layout = map_painter.layout(feature.get_name_lang("en").to_string(), dynamic_font, color, f32::MAX);
        
        let size = layout.size();

        let label = LabelPoint {
            position: paint_point - size / 2.0,
            text: feature.get_name_lang("en").to_string(),
            layout,
            dimensions: size,
            importance: 1.0,
        };

        return RenderedOverlay::Point(label);
    }
    RenderedOverlay::None()
}

fn render_line(
    viewport: Arc<Rect>,
    view: Arc<PixelBounds>,
    boundaries: Arc<PixelBounds>,
    feature: Arc<VectorFeature>,
    map_painter: Arc<egui::Painter>,
) -> RenderedOverlay {
    let (stroke, color) = get_line_style(feature.get_class(), feature.get_rank());
    let points: Vec<egui::Pos2> = feature.coordinates.iter().flatten().map(|(x, y)| {
        let pixel_x = boundaries.left() + boundaries.width() * (*x as f64);
        let pixel_y = boundaries.top() + boundaries.height() * (*y as f64);
        pos2(viewport.min.x + pixel_x as f32 * viewport.width(), viewport.min.y + pixel_y as f32 * viewport.height())
    }).collect();

    if points.len() >= 2 {
        let line = egui::Shape::line(points, Stroke::new(stroke, color));
        return RenderedOverlay::Line(line);
    }
    RenderedOverlay::None()
}

fn render_polygon(
    viewport: Arc<Rect>,
    view: Arc<PixelBounds>,
    boundaries: Arc<PixelBounds>,
    feature: Arc<VectorFeature>,
    map_painter: Arc<egui::Painter>,
) -> RenderedOverlay {
    let (fill, stroke_color) = get_polygon_style(feature.get_class(), feature.get_rank());
    let points: Vec<egui::Pos2> = feature.coordinates.iter().flatten().map(|(x, y)| {
        let pixel_x = boundaries.left() + boundaries.width() * (*x as f64);
        let pixel_y = boundaries.top() + boundaries.height() * (*y as f64);
        pos2(viewport.min.x + pixel_x as f32 * viewport.width(), viewport.min.y + pixel_y as f32 * viewport.height())
    }).collect();

    if !points.is_empty() {
        let polygon = egui::Shape::convex_polygon(points, fill, Stroke::new(1.0, stroke_color));
        return RenderedOverlay::Polygon(polygon);
    }
    RenderedOverlay::None()
}

fn get_point_style(class: &str, rank: i32) -> (f32, Color32) {
    if class == "city" && rank < 5 {
        (22.0, Color32::YELLOW)
    } else {
        (17.0, Color32::WHITE)
    }
}

fn get_line_style(class: &str, rank: i32) -> (f32, Color32) {
    if class == "road" && rank < 3 {
        (2.0, Color32::LIGHT_BLUE)
    } else {
        (1.0, Color32::GRAY)
    }
}

fn get_polygon_style(class: &str, rank: i32) -> (Color32, Color32) {
    if class == "park" {
        (Color32::GREEN, Color32::DARK_GREEN)
    } else {
        (Color32::from_gray(128), Color32::from_gray(60))
    }
}

fn paint_overlay(ui: &mut egui::Ui, rect: Rect, rendered_overlays: Vec<RenderedOverlay>) {
    let map_painter = ui.painter().with_clip_rect(rect);
    for overlay in rendered_overlays {
        match overlay {
            RenderedOverlay::Point(label) => {
                map_painter.galley(label.position, label.layout.clone(), Color32::WHITE);
            }
            RenderedOverlay::Line(line) => {
                map_painter.add(line);
            }
            RenderedOverlay::Polygon(polygon) => {
                map_painter.add(polygon);
            }
            RenderedOverlay::None() => {}
        }
    }
}
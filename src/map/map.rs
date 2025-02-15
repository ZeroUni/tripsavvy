use egui::epaint::{emath::lerp, vec2, Color32, Pos2, Rect, Shape, Stroke};
use egui::text::LayoutJob;
use egui::{pos2, Galley, Rangef, Response, Sense, Ui, Vec2, Widget, WidgetInfo, WidgetType};
use futures::future::join_all;
use rayon::prelude::*;
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
    #[serde(skip)]
    cached_overlay: Vec<RenderedOverlay>,
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
    font_size: f32,
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
    labels: HashSet<String>, // Store label names to avoid duplicates
}

impl LabelManager {
    pub fn new(minimum_distance: f32) -> Self {
        Self {
            tree: RTree::new(),
            minimum_distance,
            labels: HashSet::new(),
        }
    }

    pub fn add_label(&mut self, mut label: LabelPoint) {
        // If a label with the same name already exists, skip
        if self.labels.contains(&label.text) {
            return;
        } else {
            self.labels.insert(label.text.clone());
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

        let center = state.center.clone();
        let zoom = state.zoom.clone();

        if modified || state.cached_overlay.is_empty() {
            let view = PixelBounds::from_center(center, zoom);
            let ref_painter = Arc::new(map_painter);
            state.cached_overlay = self.render_overlay(ui, rect, &view, ref_painter, &zoom);
        }

        // Render overlay
        paint_overlay(ui, rect, state.cached_overlay.clone());

        // Store updated state
        state.store(ui.ctx(), self.id);

        response
    }
}

impl<'a> Map<'a> {
    pub fn new(id_source: impl std::hash::Hash, tile_cache: &'a mut LruCache<(u32, u32, u32), MapTile>, vector_tile_cache: &'a mut LruCache<(u32, u32, u32), VectorTile>, missing_tiles: &'a mut Vec<(u32, u32, u32)>) -> Self {
        Self {
            id: egui::Id::new(id_source),
            tile_cache,
            vector_tile_cache,
            viewport_size: Vec2::new(1024.0, 1024.0),
            missing_tiles,
        }
    }

    pub fn with_viewport(id_source: impl std::hash::Hash, tile_cache: &'a mut LruCache<(u32, u32, u32), MapTile>, vector_tile_cache: &'a mut LruCache<(u32, u32, u32), VectorTile>, missing_tiles: &'a mut Vec<(u32, u32, u32)>, viewport_size: Vec2) -> Self {
        Self {
            id: egui::Id::new(id_source),
            tile_cache,
            vector_tile_cache,
            viewport_size,
            missing_tiles,
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

    fn render_overlay(&mut self, ui: &egui::Ui, rect: Rect, view: &PixelBounds, map_painter: Arc<egui::Painter>, zoom: &f32) -> Vec<RenderedOverlay> {
        let mut overlay_layers = HashSet::new();
        for layer_name in vec!["place", "boundary", "water_name", "waterway", "park"] {
            overlay_layers.insert(layer_name);
        }
    
        let wrapped_rect = Arc::new(rect);
        let wrapped_view = Arc::new(view.clone());
    
        // First collect all tiles and their boundaries
        let tiles_and_boundaries: Vec<_> = MapState::load(ui.ctx(), self.id)
            .cached_view
            .iter()
            .filter_map(|(z, x, y, _, _)| {
                self.get_vector_tile(*z, *x, *y).map(|(tile, coords)| {
                    let boundaries = Arc::new(PixelBounds::from_x_y_zoom(coords.1, coords.2, coords.0));
                    (tile.clone(), boundaries)
                })
            })
            .collect();
    
        // Then process features in parallel
        let features: Vec<_> = MapState::load(ui.ctx(), self.id)
            .cached_view
            .iter()
            .filter_map(|(z, x, y, _, _)| {
                self.get_vector_tile(*z, *x, *y).map(|(tile, coords)| {
                    let boundaries = Arc::new(PixelBounds::from_x_y_zoom(coords.1, coords.2, coords.0));
                    // Pre-collect features with their boundaries
                    tile.layers
                        .iter()
                        .filter(|layer| overlay_layers.contains(&layer.name.as_str()))
                        .flat_map(|layer| {
                            let min_rank = layer.get_lowest_rank();
                            let boundaries = Arc::clone(&boundaries);
                            layer.features.iter().map(move |feature| {
                                (Arc::clone(&boundaries), Arc::clone(feature), min_rank)
                            })
                        })
                        .collect::<Vec<_>>()
                })
            })
            .flatten()
            .collect();
    
        // Finally process all features in parallel
        let results: Vec<RenderedOverlay> = features
            .par_iter()
            .map(|(boundaries, feature, min_rank)| {
                render_feature(
                    Arc::clone(&wrapped_rect),
                    Arc::clone(&wrapped_view),
                    Arc::clone(boundaries),
                    Arc::clone(feature),
                    *min_rank,
                    Arc::clone(&map_painter),
                    *zoom,
                )
            })
            .collect();
    
        results
    }
}

#[derive(Debug, Clone)]
enum RenderedOverlay {
    Point(LabelPoint),
    Line(Vec<Shape>),
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
    zoom: f32,
) -> RenderedOverlay {
    match feature.geometry_type {
        crate::map::vector_tile::GeometryType::Point => {
            render_point(viewport, view, boundaries, feature, lowest_rank, map_painter, &zoom)
        }
        crate::map::vector_tile::GeometryType::Line => {
            render_line(viewport, view, boundaries, feature, map_painter, &zoom)
        }
        crate::map::vector_tile::GeometryType::Polygon => {
            //self.render_polygon(viewport, view, boundaries, feature, rendered_overlays, map_painter, &zoom)
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
    zoom: &f32,
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

        // Use new styling with zoom parameter
        if let Some((font_size, color)) = get_point_style(feature.get_class(), feature.get_rank(), zoom) {
            let dynamic_font = egui::FontId::proportional(font_size);
            let layout = map_painter.layout(
                feature.get_name_lang("en").to_string(),
                dynamic_font,
                color,
                f32::MAX
            );
            
            let size = layout.size();

            let label = LabelPoint {
                position: paint_point - size / 2.0,
                text: feature.get_name_lang("en").to_string(),
                layout,
                dimensions: size,
                font_size,
                importance: calculate_importance(feature.get_class(), feature.get_rank(), zoom),
            };

            return RenderedOverlay::Point(label);
        }
    }
    RenderedOverlay::None()
}

fn render_line(
    viewport: Arc<Rect>,
    view: Arc<PixelBounds>,
    boundaries: Arc<PixelBounds>,
    feature: Arc<VectorFeature>,
    map_painter: Arc<egui::Painter>,
    zoom: &f32,
) -> RenderedOverlay {
    if feature.get_maritime() { // Don't render maritime borders
        return RenderedOverlay::None();
    }
    let (stroke, color) = get_line_style(&feature.get_admin_level(), zoom);
    
    // Process coordinate groups in parallel using rayon
    let zoom = *zoom;
    let all_lines: Vec<Shape> = feature.coordinates
        .par_iter() // Use parallel iterator
        .filter_map(|coords| {
            // Process this coordinate group in parallel
            let optimized_coords = optimize_line_points(coords.clone(), 0.01 / zoom.max(1.0));
            
            let coordinates: Vec<PixelCoordinate> = optimized_coords.iter().map(|(x, y)| {
                let pixel_x = boundaries.left() + boundaries.width() * (*x as f64);
                let pixel_y = boundaries.top() + boundaries.height() * (*y as f64);
                PixelCoordinate::new(pixel_x, pixel_y)
            }).collect();
            
            let line_segments = view.project_line(&coordinates);
            
            Some(line_segments.into_iter()
                .map(|segment| {
                    segment.into_iter()
                        .map(|coord| pos2(
                            viewport.min.x + coord.x() as f32 * viewport.width(),
                            viewport.min.y + coord.y() as f32 * viewport.height()
                        ))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>())
        })
        .flatten() // Flatten the Vec<Vec<Points>> into Vec<Points>
        .map(|points| egui::Shape::line(points, Stroke::new(stroke, color)))
        .collect();

    if !all_lines.is_empty() {
        return RenderedOverlay::Line(all_lines);
    }
    RenderedOverlay::None()
}

fn render_polygon(
    viewport: Arc<Rect>,
    view: Arc<PixelBounds>,
    boundaries: Arc<PixelBounds>,
    feature: Arc<VectorFeature>,
    map_painter: Arc<egui::Painter>,
    zoom: &f32,
) -> RenderedOverlay {
    let (fill, stroke_color) = get_polygon_style(feature.get_class(), feature.get_rank());
    let points: Vec<egui::Pos2> = feature.coordinates.iter().map(|coords| {
        let temp_coord = coords.first().unwrap();
        let pixel_x = boundaries.left() + boundaries.width() * temp_coord.0 as f64;
        let pixel_y = boundaries.top() + boundaries.height() * temp_coord.1 as f64;
        crate::map::map_tile::PixelCoordinate::new(pixel_x, pixel_y)
    }).filter_map(|coord| {
        view.project_point(&coord).map(|projected| {
            pos2(
                viewport.min.x + projected.x() as f32 * viewport.width(),
                viewport.min.y + projected.y() as f32 * viewport.height(),
            )
        })
    }).collect();

    if !points.is_empty() {
        let polygon = egui::Shape::convex_polygon(points, fill, Stroke::new(1.0, stroke_color));
        return RenderedOverlay::Polygon(polygon);
    }
    RenderedOverlay::None()
}

#[derive(Debug)]
struct PointStyle {
    font_size: f32,
    color: Color32,
    min_zoom: f32,
    max_zoom: f32,
    zoom_scaling: f32,
}

fn get_point_style(class: &str, rank: i32, zoom: &f32) -> Option<(f32, Color32)> {
    // Base styles for different classes
    let style = match class {
        "continent" => PointStyle {
            font_size: 32.0,
            color: Color32::from_rgb(244, 0, 34), 
            min_zoom: 0.0,
            max_zoom: 6.0,
            zoom_scaling: 0.8,
        },
        "country" => PointStyle {
            font_size: 28.0,
            color: Color32::from_rgb(255, 255, 180), 
            min_zoom: 2.5,
            max_zoom: 8.0,
            zoom_scaling: 0.9,
        },
        "state" => PointStyle {
            font_size: 24.0,
            color: Color32::from_rgb(255, 255, 200),
            min_zoom: 3.5,
            max_zoom: 10.0,
            zoom_scaling: 1.0,
        },
        "city" => {
            match rank {
                0..=2 => PointStyle { // Capital cities
                    font_size: 22.0,
                    color: Color32::from_rgb(255, 255, 255), // White
                    min_zoom: 4.0,
                    max_zoom: 20.0,
                    zoom_scaling: 1.1,
                },
                3..=5 => PointStyle { // Major cities
                    font_size: 20.0,
                    color: Color32::from_rgb(255, 255, 220), // Off-white
                    min_zoom: 6.0,
                    max_zoom: 20.0,
                    zoom_scaling: 1.0,
                },
                _ => PointStyle { // Other cities
                    font_size: 17.0,
                    color: Color32::from_rgb(235, 235, 235), // Light gray
                    min_zoom: 8.0,
                    max_zoom: 20.0,
                    zoom_scaling: 0.9,
                },
            }
        },
        "ocean" | "sea" => PointStyle {
            font_size: 26.0,
            color: Color32::from_rgb(158, 189, 255), // Light blue
            min_zoom: 2.0,
            max_zoom: 7.0,
            zoom_scaling: 0.8,
        },
        "lake" => PointStyle {
            font_size: 18.0,
            color: Color32::from_rgb(158, 189, 255), // Light blue
            min_zoom: 6.0,
            max_zoom: 15.0,
            zoom_scaling: 0.9,
        },
        "bay" | "gulf" => PointStyle {
            font_size: 16.0,
            color: Color32::from_rgb(158, 189, 255), // Light blue
            min_zoom: 7.0,
            max_zoom: 15.0,
            zoom_scaling: 0.9,
        },
        "river" => PointStyle {
            font_size: 16.0,
            color: Color32::from_rgb(158, 189, 255), // Light blue
            min_zoom: 8.0,
            max_zoom: 15.0,
            zoom_scaling: 0.9,
        },
        "national_forest" => PointStyle {
            font_size: 20.0,
            color: Color32::from_rgb(144, 238, 144), // Light green
            min_zoom: 7.0,
            max_zoom: 15.0,
            zoom_scaling: 0.9,
        },
        "national_marine_sanctuary" => PointStyle {
            font_size: 20.0,
            color: Color32::from_rgb(135, 206, 235), // Sky blue
            min_zoom: 7.0,
            max_zoom: 15.0,
            zoom_scaling: 0.9,
        },
        "state_forest" => PointStyle {
            font_size: 18.0,
            color: Color32::from_rgb(154, 205, 50), // Yellow green
            min_zoom: 8.0,
            max_zoom: 15.0,
            zoom_scaling: 0.9,
        },
        "nature_reserve" => PointStyle {
            font_size: 18.0,
            color: Color32::from_rgb(124, 252, 0), // Lawn green
            min_zoom: 8.0,
            max_zoom: 15.0,
            zoom_scaling: 0.9,
        },
        "national_wildlife_refuge" => PointStyle {
            font_size: 19.0,
            color: Color32::from_rgb(173, 255, 47), // Green yellow
            min_zoom: 7.0,
            max_zoom: 15.0,
            zoom_scaling: 0.9,
        },
        "state_park" => PointStyle {
            font_size: 17.0,
            color: Color32::from_rgb(152, 251, 152), // Pale green
            min_zoom: 9.0,
            max_zoom: 15.0,
            zoom_scaling: 0.9,
        },
        "town" => PointStyle {
            font_size: 15.0,
            color: Color32::from_rgb(220, 220, 220), // Light gray
            min_zoom: 8.0,
            max_zoom: 20.0,
            zoom_scaling: 1.2,
        },
        "village" => PointStyle {
            font_size: 14.0,
            color: Color32::from_rgb(200, 200, 200), // Slightly darker gray
            min_zoom: 11.0,
            max_zoom: 20.0,
            zoom_scaling: 0.8,
        },
        _ => PointStyle {
            font_size: 16.0,
            color: Color32::from_rgb(200, 200, 200), // Default gray
            min_zoom: 10.0,
            max_zoom: 20.0,
            zoom_scaling: 1.0,
        },
    };

    // Check zoom level bounds
    if *zoom < style.min_zoom || *zoom > style.max_zoom {
        return None;
    }

    // Calculate zoom-based scaling
    let zoom_factor = match *zoom {
        z if z <= style.min_zoom => 0.8,
        z if z >= style.max_zoom => 1.2,
        z => {
            let range = style.max_zoom - style.min_zoom;
            let position = (z - style.min_zoom) / range;
            0.8 + position * 0.4 * style.zoom_scaling
        }
    };

    // Apply zoom scaling to font size
    let scaled_size = style.font_size * zoom_factor;
    
    Some((scaled_size, style.color))
}

fn calculate_importance(class: &str, rank: i32, zoom: &f32) -> f32 {
    // Base importance for different classes
    let importance = match class {
        "continent" => 1.0,
        "country" => 0.9,
        "state" => 0.8,
        "city" => {
            match rank {
                0..=2 => 0.7,
                3..=5 => 0.6,
                _ => 0.5,
            }
        },
        "ocean" | "sea" => 0.9,
        "lake" => 0.8,
        "bay" => 0.7,
        _ => 0.5,
    };

    // Calculate zoom-based scaling
    let zoom_factor = match *zoom {
        z if z <= 5.0 => 0.8,
        z if z >= 10.0 => 1.2,
        z => {
            let range = 10.0 - 5.0;
            let position = (z - 5.0) / range;
            0.8 + position * 0.4
        }
    };

    // Apply zoom scaling to importance
    importance * zoom_factor
}

fn get_line_style(level: &i32, zoom: &f32) -> (f32, Color32) {
    /// Based on the admin level (2 - 10), return a stroke width and color
    // More important levels have a higher base stroke, and different levels have varying border colors
    let zoom_factor = 1.0 + zoom.fract();
    match level {
        2 => (1.8 * zoom_factor, Color32::from_rgb(208, 214, 179)),
        3 => (1.5 * zoom_factor, Color32::from_rgb(41, 23, 32)),
        4 => (1.0 * zoom_factor, Color32::from_rgb(41, 23, 32)),
        5 => (0.8 * zoom_factor, Color32::from_rgb(30, 50, 49)),
        6 => (0.6 * zoom_factor, Color32::from_rgb(30, 50, 49)),
        7 => (0.4 * zoom_factor, Color32::from_rgb(30, 50, 49)),
        8 => (0.2 * zoom_factor, Color32::from_rgb(152, 108, 106)),
        9 => (0.1 * zoom_factor, Color32::from_rgb(152, 108, 106)),
        10 => (0.05 * zoom_factor, Color32::from_rgb(75, 136, 162)),
        _ => (0.05 * zoom_factor, Color32::from_rgb(255, 0, 0)),
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
    let mut label_manager = LabelManager::new(20.0);
    for overlay in rendered_overlays {
        match overlay {
            RenderedOverlay::Point(label) => {
                label_manager.add_label(label);
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
    for label in label_manager.tree.iter() {
        map_painter.text(
            egui::Pos2::new(label.position.x - 1.2, label.position.y - 1.2),
            egui::Align2::LEFT_TOP,
            label.text.clone(),
            egui::FontId::proportional(label.font_size),
            Color32::BLACK,
        );
        map_painter.text(
            egui::Pos2::new(label.position.x + 1.2, label.position.y + 1.2),
            egui::Align2::LEFT_TOP,
            label.text.clone(),
            egui::FontId::proportional(label.font_size),
            Color32::BLACK,
        );
        map_painter.galley(
            label.position,
            label.layout.clone(),
            Color32::WHITE,
        );
    }
}

/// Optimizes line points using a distance-based decimation algorithm
fn optimize_line_points(coords: Arc<[(f32, f32)]>, tolerance: f32) -> Vec<(f32, f32)> {
    if coords.len() <= 2 {
        return coords.to_vec();
    }

    let mut result = Vec::with_capacity(coords.len() / 2);
    result.push(coords[0]); // Always keep first point
    
    let mut last_kept = coords[0];
    
    // Simple distance-based filtering
    for i in 1..coords.len() - 1 {
        let current = coords[i];
        
        // Calculate squared distance (avoid sqrt for performance)
        let dx = current.0 - last_kept.0;
        let dy = current.1 - last_kept.1;
        let dist_sq = dx * dx + dy * dy;
        
        // Only keep points that are far enough from the last kept point
        if dist_sq > tolerance * tolerance {
            result.push(current);
            last_kept = current;
        }
    }
    
    // Always keep last point
    result.push(*coords.last().unwrap());
    result
}
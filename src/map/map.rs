use egui::epaint::{emath::lerp, vec2, Color32, Pos2, Rect, Shape, Stroke};
use egui::{pos2, Rangef, Response, Sense, Ui, Vec2, Widget, WidgetInfo, WidgetType};
use std::collections::{HashMap, HashSet};
use serde::{Deserialize, Serialize};
use lru::LruCache;

use super::map_tile::{Coordinate, MapTile, PixelBounds, PixelCoordinate};

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

pub struct Map<'a> {
    id: egui::Id,
    tile_cache: &'a mut LruCache<(u32, u32, u32), MapTile>,
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

        // Store updated state
        state.store(ui.ctx(), self.id);

        response
    }
}

impl<'a> Map<'a> {
    pub fn new(id_source: impl std::hash::Hash, tile_cache: &'a mut LruCache<(u32, u32, u32), MapTile>, missing_tiles: &'a mut Vec<(u32, u32, u32)>) -> Self {
        Self {
            id: egui::Id::new(id_source),
            tile_cache,
            viewport_size: Vec2::new(1024.0, 1024.0),
            missing_tiles: missing_tiles,
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
}
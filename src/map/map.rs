use egui::epaint::{emath::lerp, vec2, Color32, Pos2, Rect, Shape, Stroke};
use egui::{Response, Sense, Ui, Widget, WidgetInfo, WidgetType, Vec2, pos2};
use std::collections::{HashMap, HashSet};
use serde::{Deserialize, Serialize};

use super::map_tile::{Coordinate, GeoBounds, MapTile};

#[derive(Default, Clone, Serialize, Deserialize)]
pub struct MapState {
    center: Coordinate,
    zoom: f32,
    dragging: bool,
    drag_start: Option<Pos2>,
    drag_delta: Vec2,
}

impl MapState {
    pub fn load(ctx: &egui::Context, id: egui::Id) -> Self {
        ctx.data_mut(|d| d.get_persisted::<Self>(id).unwrap_or_default())
    }

    pub fn store(self, ctx: &egui::Context, id: egui::Id) {
        ctx.data_mut(|d| d.insert_persisted(id, self));
    }

    fn update_center(&mut self, new_center: &Coordinate) {
        // Clamp latitude to valid range (-85.05112878 to 85.05112878 are the limits
        // of Web Mercator projection)
        let lat = new_center.latitude()
            .clamp(-85.05112878, 85.05112878);
            
        // Normalize longitude to -180 to 180 range
        let mut lon = new_center.longitude();
        while lon > 180.0 {
            lon -= 360.0;
        }
        while lon < -180.0 {
            lon += 360.0;
        }

        self.center = Coordinate::new(lat, lon);
    }
}

pub struct Map<'a> {
    id: egui::Id,
    tile_cache: &'a mut HashMap<(u32, u32, u32), MapTile>,
    viewport_size: Vec2,
    missing_tiles: &'a mut Vec<(u32, u32, u32)>,
}

impl<'a> Widget for Map<'a> {
    fn ui(self, ui: &mut Ui) -> Response {
        let mut state = MapState::load(ui.ctx(), self.id);

        let (rect, response) = ui.allocate_exact_size(
            self.viewport_size,
            Sense::click_and_drag()
        );

        ui.painter().rect(rect, 0.0, Color32::from_rgb(100, 0, 100), Stroke::new(1.0, Color32::WHITE));

        let map_painter = ui.painter().with_clip_rect(rect);

        // Handle interactions
        if response.dragged() {
            if !state.dragging {
                state.drag_start = response.hover_pos();
                state.dragging = true;
            }
            if let Some(current_pos) = response.hover_pos() {
                if let Some(start_pos) = state.drag_start {
                    let delta = current_pos - start_pos;
                    
                    let zoom_factor = 2.0f32.powi(state.zoom.floor() as i32);
                    let degrees_per_pixel = 360.0 / (zoom_factor * 512.0);
                    // longitude and latitude are not directly modifiable, so we create a Vec2 delta to add to the center
                    state.drag_delta = Vec2::new(-delta.x * degrees_per_pixel, delta.y * degrees_per_pixel);
                    let new_center = state.center.clone() + state.drag_delta;
                    state.update_center(&new_center);

                    state.drag_start = Some(current_pos);
                }
            }
        } else if state.dragging {
            state.dragging = false;
            state.drag_start = None;
        }

        // Handle zoom for pinch / touch
        let zoom_delta = ui.input(|i| i.zoom_delta());
        if (zoom_delta - 1.0).abs() > f32::EPSILON {
            state.zoom = (state.zoom * zoom_delta).clamp(1.0, 20.0);
        }

        // Handle zoom for scroll
        let mut scroll = ui.input(|i| i.smooth_scroll_delta).y;
        if (scroll - 0.0).abs() > f32::EPSILON {
            // Normalize scroll further using tanh
            scroll = (scroll / 10.0).tanh();
            state.zoom = (state.zoom + scroll).clamp(0.0, 20.0);
        }

        // Calculate and paint visible tiles
        let visible_tiles = self.calculate_visible_tiles(rect, &state);
        
        for (z, x, y, tile_map, uv) in visible_tiles {
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
            if let Some(tile) = self.tile_cache.get_mut(&(z, x, y)) {
                let texture = tile.texture(ui.ctx());
                
                map_painter.image(
                    texture.id(),
                    tile_rect,
                    uv,
                    Color32::WHITE
                );
            } else {
                self.missing_tiles.push((z, x, y));
                map_painter.rect_filled(tile_rect, 0.0, Color32::GRAY);
            }
        }

        // If we detect the user input ctrl + 'c', we reset the map state
        if ui.input(|i| i.modifiers.ctrl && i.key_pressed(egui::Key::C)) {
            state = MapState::default();
        }

        // Store updated state
        state.store(ui.ctx(), self.id);

        response
    }
}

impl<'a> Map<'a> {
    pub fn new(id_source: impl std::hash::Hash, tile_cache: &'a mut HashMap<(u32, u32, u32), MapTile>, missing_tiles: &'a mut Vec<(u32, u32, u32)>) -> Self {
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
        let z = state.zoom.floor().max(0.0) as u32;
        let scale = 2.0f32.powf(state.zoom - z as f32);
        
        // For zoom 0, only one tile exists
        if z == 0 {
            let uv_rect = Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0));
            return vec![(0, 0, 0, uv_rect, uv_rect)];
        }
    
        // Get the geobounds of the viewport
        let bounds = GeoBounds::from_center(state.center.clone(), state.zoom, 512.0);

        // Get the tile coordinates of the viewport
        let tiles = bounds.all_x_y_zoom(state.zoom);
        //println!("Bounds: {:?}, Tiles: {:?}, Zoom: {}", bounds, tiles, state.zoom);

        let mut visible_tiles = Vec::new();

        for (x, y) in tiles {
            let tile_bounds = GeoBounds::from_x_y_zoom(x, y, z);
            let (tile_rect, uv_rect) = bounds.uv_map(&tile_bounds);

            visible_tiles.push((z, y, x, tile_rect, uv_rect));
        }

        // println!("Visible tiles: {:?}", visible_tiles);
        
        // Return nothing for now
        return visible_tiles;
    }
}
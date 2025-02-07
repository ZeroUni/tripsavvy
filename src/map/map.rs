use egui::epaint::{emath::lerp, vec2, Color32, Pos2, Rect, Shape, Stroke};
use egui::{pos2, Rangef, Response, Sense, Ui, Vec2, Widget, WidgetInfo, WidgetType};
use std::collections::{HashMap, HashSet};
use serde::{Deserialize, Serialize};
use lru::LruCache;

use super::map_tile::{Coordinate, GeoBounds, MapTile, PixelBounds, PixelCoordinate};

#[derive(Default, Clone, Serialize, Deserialize)]
pub struct MapState {
    center: PixelCoordinate,
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
}

pub struct Map<'a> {
    id: egui::Id,
    tile_cache: &'a mut LruCache<(u32, u32, u32), MapTile>,
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
                    state.drag_delta = Vec2::new(-delta.x * degrees_per_pixel, -delta.y * degrees_per_pixel);
                    state.center = state.center.add_delta(state.drag_delta, state.zoom);

                    state.drag_start = Some(current_pos);
                }
            }
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
        }

        // Handle zoom for scroll
        let mut scroll = ui.input(|i| i.smooth_scroll_delta).y;
        if (scroll - 0.0).abs() > f32::EPSILON && !zoomed {
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
        let z = state.zoom.floor().max(0.0) as u32;

        // Get the geobounds of the viewport
        let bounds = PixelBounds::from_center(state.center.clone(), state.zoom);

        // For zoom 0, only one tile exists
        if z == 0 {
            // Only one tile to compare
            // let tile_bounds = PixelBounds::from_x_y_zoom(0, 0, z);
            // println!("Tile bounds: {:?}, Viewport bounds: {:?}", tile_bounds, bounds);
            // let mut visible_tiles = Vec::new();

            // let mapping_vec = bounds.uv_map(&tile_bounds);
            // //println!("Mapping vec: {:?}", mapping_vec); 
            
            // for (tile_rect, uv_rect) in mapping_vec {
            //     visible_tiles.push((0, 0, 0, tile_rect.clone(), uv_rect.clone()));
            // }
            // println!("Visible tiles: {:?}", visible_tiles);

            // return visible_tiles;

            // For clarity, lets use the zoom level 1 tiles (increased dpi)
            let tiles = bounds.all_x_y_zoom(1.0_f32);
            // println!("Bounds: {:?}, Tiles: {:?}, Zoom: {}", bounds, tiles, 1);

            let mut visible_tiles = Vec::new();

            for (x, y) in tiles {
                let tile_bounds = PixelBounds::from_x_y_zoom(x, y, 1);
                let mapping_vec = bounds.uv_map(&tile_bounds);

                // println!("Tile {:?}, Bounds: {:?}, Mapping: {:?}", (x, y), tile_bounds, mapping_vec);
                
                for (tile_rect, uv_rect) in mapping_vec{
                    visible_tiles.push((1, x, y, tile_rect, uv_rect));
                }
            }

            // println!("Visible tiles: {:?}", visible_tiles);

            return visible_tiles;
        }
    
        // Get the tile coordinates of the viewport
        let tiles = bounds.all_x_y_zoom(state.zoom);
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
}
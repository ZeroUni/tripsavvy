use serde::{Deserialize, Serialize};
use image;

#[derive(Debug, Deserialize, Serialize)]
pub struct GeoBounds {
    south: f64, // minimum latitude
    west: f64,  // minimum longitude
    north: f64, // maximum latitude
    east: f64,  // maximum longitude
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Coordinate {
    latitude: f64,
    longitude: f64,
}

#[derive(Deserialize, Serialize)]
pub struct MapTile {
    pub x: u32,
    pub y: u32,
    pub zoom: u32,
    pub image_size: egui::Vec2,  // In pixels
    pub geo_bounds: GeoBounds,   // Geographical bounds
    image_data: Vec<u8>,         // Raw image data
    #[serde(skip)]
    texture: Option<egui::TextureHandle>, // Has to be an option so it can be loaded lazily, without needing the app context
}

impl MapTile {
    pub fn new(x: u32, y: u32, zoom: u32, image_size: egui::Vec2, geo_bounds: GeoBounds, image_data: Vec<u8>) -> Self {
        Self {
            x,
            y,
            zoom,
            image_size,
            geo_bounds,
            image_data,
            texture: None,
        }
    }

    pub fn texture(&mut self, ctx: &egui::Context) -> &egui::TextureHandle {
        if self.texture.is_none() {
            let color_image = egui::ColorImage::from_rgba_unmultiplied(
                [self.image_size.x as usize, self.image_size.y as usize],
                &self.image_data,
            );
            let texture = ctx.load_texture(
                format!("tile_{}_{}_zoom{}", self.x, self.y, self.zoom),
                color_image,
                egui::TextureOptions::default()
            );
            self.texture = Some(texture);
        }
        self.texture.as_ref().unwrap()
    }
}

/// Convert a latitude and longitude to tile x, y coordinates for a given zoom.
/// Uses the Web Mercator projection.
pub fn latlng_to_tile_coords(lat: f64, lon: f64, zoom: u32) -> (u32, u32) {
    let lat_rad = lat.to_radians();
    let n = 2.0_f64.powi(zoom as i32);
    let x_tile = ((lon + 180.0) / 360.0 * n).floor() as u32;
    let y_tile = ((1.0 - (lat_rad.tan() + 1.0 / lat_rad.cos()).ln() / std::f64::consts::PI) / 2.0 * n).floor() as u32;
    (x_tile, y_tile)
}

/// Convert tile x, y, zoom into geographical bounds (GeoBounds)
pub fn tile_coords_to_geo_bounds(x: u32, y: u32, zoom: u32) -> GeoBounds {
    let n = 2.0_f64.powi(zoom as i32);
    
    // Longitudes:
    let west = x as f64 / n * 360.0 - 180.0;
    let east = (x as f64 + 1.0) / n * 360.0 - 180.0;
    
    // For latitudes, we use the inverse of the Mercator projection.
    let lat_rad_north = ((std::f64::consts::PI * (1.0 - 2.0 * y as f64 / n)).sinh()).atan();
    let lat_rad_south = ((std::f64::consts::PI * (1.0 - 2.0 * (y as f64 + 1.0) / n)).sinh()).atan();
    
    let north = lat_rad_north.to_degrees();
    let south = lat_rad_south.to_degrees();
    
    GeoBounds {
        south,
        west,
        north,
        east,
    }
}
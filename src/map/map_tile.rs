use std::{net::Incoming, ops::Add};

use serde::{Deserialize, Serialize};
use image;

#[derive(Debug, Deserialize, Serialize)]
pub struct GeoBounds {
    south: f64, // minimum latitude
    west: f64,  // minimum longitude
    north: f64, // maximum latitude
    east: f64,  // maximum longitude
}

impl GeoBounds {
    pub fn south(&self) -> f64 {
        self.south
    }

    pub fn west(&self) -> f64 {
        self.west
    }

    pub fn north(&self) -> f64 {
        self.north
    }

    pub fn east(&self) -> f64 {
        self.east
    }

    pub fn size(&self) -> (f64, f64) {
        (self.north - self.south, self.east - self.west)
    }

    pub fn center(&self) -> Coordinate {
        Coordinate {
            latitude: (self.south + self.north) / 2.0,
            longitude: (self.west + self.east) / 2.0,
        }
    }

    pub fn contains(&self, other: &GeoBounds) -> bool {
        self.south <= other.south && self.west <= other.west && self.north >= other.north && self.east >= other.east
    }

    pub fn intersects(&self, other: &GeoBounds) -> bool {
        self.south <= other.north && self.north >= other.south && self.west <= other.east && self.east >= other.west
    }

    pub fn intersect(&self, other: &GeoBounds) -> Option<GeoBounds> {
        if !self.intersects(other) {
            return None;
        }
        Some(GeoBounds {
            south: self.south.max(other.south),
            west: self.west.max(other.west),
            north: self.north.min(other.north),
            east: self.east.min(other.east),
        })
    }

    pub fn all_x_y_zoom(&self, zoom: f32) -> Vec<(u32, u32)> {
        let mut tiles = Vec::new();
        let zoom_floor = zoom.floor() as u32;
        
        // Get tile coordinates for all corners
        let (nw_x, nw_y) = latlng_to_tile_coords(self.north, self.west, zoom_floor);
        let (ne_x, ne_y) = latlng_to_tile_coords(self.north, self.east, zoom_floor);
        let (sw_x, sw_y) = latlng_to_tile_coords(self.south, self.west, zoom_floor);
        let (se_x, se_y) = latlng_to_tile_coords(self.south, self.east, zoom_floor);
        
        // Find the min/max x and y coordinates
        let min_x = nw_x.min(sw_x);
        let max_x = ne_x.max(se_x);
        let min_y = ne_y.min(nw_y);
        let max_y = se_y.max(sw_y);
        
        // Handle wrap-around at 180/-180 degrees longitude
        if self.west > self.east {
            // Bounds cross the 180/-180 meridian
            let n = 2u32.pow(zoom_floor);
            for x in min_x..n {
                for y in min_y..=max_y {
                    tiles.push((x, y));
                }
            }
            for x in 0..=max_x {
                for y in min_y..=max_y {
                    tiles.push((x, y));
                }
            }
        } else {
            // Normal case
            for x in min_x..=max_x {
                for y in min_y..=max_y {
                    tiles.push((x, y));
                }
            }
        }
        
        tiles
    }

    pub fn from_x_y_zoom(x: u32, y: u32, zoom: u32) -> Self {
        let bounds = tile_coords_to_geo_bounds(x, y, zoom);
        GeoBounds {
            south: bounds.south,
            west: bounds.west,
            north: bounds.north,
            east: bounds.east,
        }
    }

    pub fn from_center(center: Coordinate, zoom: f32, tile_size: impl Into<f64>) -> Self {
        let tile_size = tile_size.into();
        let zoom_floor = zoom.floor();
        let zoom_fract = zoom - zoom_floor;
        
        // Calculate spans at both zoom levels
        let n_floor = 2.0_f64.powf(zoom_floor as f64);
        let n_ceil = 2.0_f64.powf((zoom_floor + 1.0) as f64);
        
        // Calculate spans at both levels
        let lng_span_floor = 360.0 / n_floor;
        let lng_span_ceil = 360.0 / n_ceil;
        
        // Lerp between the two spans
        let lng_span = lng_span_floor * (1.0 - zoom_fract as f64) + lng_span_ceil * zoom_fract as f64;
        
        // Adjust for mercator projection at the given latitude
        let lat_span = lng_span * (1.0 / center.latitude.to_radians().cos());
        
        // Scale by tile size
        let scale = tile_size / 512.0;
        let half_lng = (lng_span * scale) / 2.0;
        let half_lat = (lat_span * scale) / 2.0;
    
        GeoBounds {
            west: center.longitude - half_lng,
            east: center.longitude + half_lng,
            north: center.latitude + half_lat,
            south: center.latitude - half_lat,
        }
    }

    pub fn uv_map(&self, other: &GeoBounds) -> (egui::Rect, egui::Rect) {
        // Returns the mapping from the other bounds on to this bound. 
        // The first one is relative to this bounds, the second one is relative to the other bounds.
        // i.e. if other covers the bottom left quadrant of this, the first rect is egui::Rect::from_min_max(0.0, 0.0, 0.5, 0.5)
        // The second one has the same aspect ratio, but may be egui::Rect::from_min_max(0.5, 0.5, 1.0, 1.0) or could be egui::Rect::from_min_max(0.8, 0.8, 1.0, 1.0) if the other bounds is larger than this bounds.
        // Assumes that all bounds are square in mercator projection, and returns Rect::NONE if the bounds do not intersect.
        if !self.intersects(other) {
            return (egui::Rect::NOTHING, egui::Rect::NOTHING);
        }
    
        // Convert to mercator y coordinates
        let merc_y = |lat: f64| (std::f64::consts::PI / 4.0 + lat.to_radians() / 2.0).tan().ln();
        
        // Calculate mercator coordinates for both bounds
        let self_north_y = merc_y(self.north);
        let self_south_y = merc_y(self.south);
        let other_north_y = merc_y(other.north);
        let other_south_y = merc_y(other.south);
    
        // Calculate relative positions in mercator space for first rect
        let x1 = (other.west - self.west) / (self.east - self.west);
        let x2 = (other.east - self.west) / (self.east - self.west);
        let y1 = (other_north_y - self_north_y) / (self_south_y - self_north_y);
        let y2 = (other_south_y - self_north_y) / (self_south_y - self_north_y);
    
        // Calculate relative positions in mercator space for second rect
        let u1 = (self.west - other.west) / (other.east - other.west);
        let u2 = (self.east - other.west) / (other.east - other.west);
        let v1 = (self_north_y - other_north_y) / (other_south_y - other_north_y);
        let v2 = (self_south_y - other_north_y) / (other_south_y - other_north_y);
    
        // Clamp all values to [0,1]
        let rect1 = egui::Rect::from_min_max(
            egui::pos2(x1.clamp(0.0, 1.0) as f32, y1.clamp(0.0, 1.0) as f32),
            egui::pos2(x2.clamp(0.0, 1.0) as f32, y2.clamp(0.0, 1.0) as f32)
        );
        
        let rect2 = egui::Rect::from_min_max(
            egui::pos2(u1.clamp(0.0, 1.0) as f32, v1.clamp(0.0, 1.0) as f32),
            egui::pos2(u2.clamp(0.0, 1.0) as f32, v2.clamp(0.0, 1.0) as f32)
        );
    
        (rect1, rect2)
    
    }
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Coordinate {
    latitude: f64,
    longitude: f64,
}

impl Default for Coordinate {
    fn default() -> Self {
        Self {
            latitude: 0.0,
            longitude: 0.0,
        }
    }
}

impl Add<Coordinate> for Coordinate {
    type Output = Coordinate;

    fn add(self, other: Coordinate) -> Coordinate {
        Coordinate {
            latitude: self.latitude + other.latitude,
            longitude: self.longitude + other.longitude,
        }
    }
}

impl Add<egui::Vec2> for Coordinate {
    type Output = Coordinate;

    fn add(self, other: egui::Vec2) -> Coordinate {
        Coordinate {
            latitude: self.latitude + other.y as f64,
            longitude: self.longitude + other.x as f64,
        }
    }
}

impl Coordinate {
    pub fn new(latitude: f64, longitude: f64) -> Self {
        Self {
            latitude,
            longitude,
        }
    }
    pub fn latitude(&self) -> f64 {
        self.latitude
    }

    pub fn longitude(&self) -> f64 {
        self.longitude
    }
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
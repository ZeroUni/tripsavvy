use std::{collections::HashSet, net::Incoming, ops::{Add, Mul}};

use egui::pos2;
use serde::{Deserialize, Serialize};
use image::{self, Pixel};

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
        // Assuming a web mercator tile projection, invert it to get the center
        let lat_rad = ((self.south + self.north) / 2.0).to_radians();
        let lon_rad = ((self.west + self.east) / 2.0).to_radians();
        Coordinate {
            latitude: lat_rad.to_degrees(),
            longitude: lon_rad.to_degrees(),
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

    pub fn quadrant(&self, n: u8) -> GeoBounds {
        let center = self.center();
        let (lat_size, lon_size) = self.size();
        let lat_size = lat_size / 2.0;
        let lon_size = lon_size / 2.0;
        match n {
            0 => GeoBounds {
                south: center.latitude - lat_size,
                west: center.longitude - lon_size,
                north: center.latitude,
                east: center.longitude,
            },
            1 => GeoBounds {
                south: center.latitude - lat_size,
                west: center.longitude,
                north: center.latitude,
                east: center.longitude + lon_size,
            },
            2 => GeoBounds {
                south: center.latitude,
                west: center.longitude - lon_size,
                north: center.latitude + lat_size,
                east: center.longitude,
            },
            3 => GeoBounds {
                south: center.latitude,
                west: center.longitude,
                north: center.latitude + lat_size,
                east: center.longitude + lon_size,
            },
            _ => panic!("Invalid quadrant number"),
        }
    }

    pub fn all_x_y_zoom(&self, zoom: f32) -> Vec<(u32, u32)> {
        let mut tiles = Vec::new();
        let zoom_floor = zoom.floor() as u32;
        //let modulus = 2u32.pow(2*zoom_floor);
        
        // Get tile coordinates for all corners
        let (nw_x, nw_y) = lat_lon_to_tile_coords(self.north, self.west, zoom_floor);
        let (ne_x, ne_y) = lat_lon_to_tile_coords(self.north, self.east, zoom_floor);
        let (sw_x, sw_y) = lat_lon_to_tile_coords(self.south, self.west, zoom_floor);
        let (se_x, se_y) = lat_lon_to_tile_coords(self.south, self.east, zoom_floor);
        
        // Add all tiles to the list
        for x in nw_x..=ne_x {
            for y in nw_y..=sw_y {
                tiles.push((x, y));
            }
        }
        
        tiles
    }

    pub fn from_x_y_zoom(x: u32, y: u32, zoom: u32) -> Self {
        if (x == y && x == 0) || zoom == 0 {
            return GeoBounds {
                south: -85.05112877980659,
                west: -180.0,
                north: 85.05112877980659,
                east: 180.0,
            };
        }
        let bounds = tile_coords_to_geo_bounds(x, y, zoom);
        GeoBounds {
            south: bounds.south,
            west: bounds.west,
            north: bounds.north,
            east: bounds.east,
        }
    }

    pub fn from_center(center: Coordinate, zoom: f32) -> Self {
        // Base size at zoom 0 (one tile covers world)
        const BASE_LON_SIZE: f64 = 360.0;
        const BASE_LAT_SIZE: f64 = 170.1022; // ~85.0511 degrees north/south

        // Calculate size at target zoom
        // let size_factor = 2.0f64.powf(-zoom as f64);
        // Lerp our size factor linearly between -zoom and -(zoom + 1)
        let floor = zoom.floor() as i32;
        let size_factor = egui::lerp(egui::Rangef::new(2.0_f32.powi(-(floor + 1)), 2.0_f32.powi(-floor)), zoom % zoom.floor()) as f64;
        let lon_size = BASE_LON_SIZE * size_factor;
        let lat_size = BASE_LAT_SIZE * size_factor;

        // Calculate bounds
        let half_lon = lon_size / 2.0;
        let half_lat = lat_size / 2.0;

        let mut west = center.longitude() - half_lon;
        let mut east = center.longitude() + half_lon;
        let mut north = center.latitude() + half_lat;
        let mut south = center.latitude() - half_lat;

        // Wrap longitude
        west = ((west + 180.0) % 360.0) - 180.0;
        east = ((east + 180.0) % 360.0) - 180.0;

        // Clamp latitude
        north = north.min(85.0511);
        south = south.max(-85.0511);

        GeoBounds {
            south,
            west,
            north,
            east,
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

#[derive(Debug, Deserialize, Serialize, Clone, Copy)]
// Geobounds are nice, but we need to think about wrapping. So, we can use pixel bounds corresponding to actual map tiles, and allow them to wrap.
/// Pixel bounds are used to represent the bounds of a map tile in pixel coordinates. A normalized pixel coordinate is in the range [0, 1], however, the bounds can wrap around the edges of the map.
/// This means that [.6, 1.2, 0.6, 1.2] would represent a tile that goes from the middle-ish of the map to to top right, then contines to the bottom left and back to the 0.2.
/// It is also equivalent to [0.6, 0.2, 0.6, 0.2] (modulo 1).
pub struct PixelBounds {
    left: f64,
    right: f64,
    top: f64,
    bottom: f64,
}

impl Default for PixelBounds {
    fn default() -> Self {
        Self {
            left: 0.0,
            right: 1.0,
            top: 0.0,
            bottom: 1.0,
        }
    }
}

impl PixelBounds {
    pub fn new(left: f64, right: f64, top: f64, bottom: f64) -> Self {
        /// Create a new PixelBounds object with the given coordinates. The coordinates are normalized to the range [0, 1], wrapping.
        // Apply modulo 1.0 to values greater than 1.0, excluding 1.0
        let normalize = |x: f64| {
            if x > 1.0 {
                x - x.floor()
            } else if x < 0.0 {
                x - x.ceil() + 1.0
            } else {
                x
            }
        };
        
        Self {
            left: normalize(left),
            right: normalize(right),
            top: normalize(top),
            bottom: normalize(bottom),
        }
    }

    pub fn left(&self) -> f64 {
        /// Get the left coordinate of the bounds
        self.left
    }

    pub fn right(&self) -> f64 {
        /// Get the right coordinate of the bounds
        self.right
    }

    pub fn top(&self) -> f64 {
        /// Get the top coordinate of the bounds
        self.top
    }

    pub fn bottom(&self) -> f64 {
        /// Get the bottom coordinate of the bounds
        self.bottom
    }

    pub fn width(&self) -> f64 {
        /// Calculate the width of the bounds
        let diff = (self.right - self.left);
        if diff.abs() < f64::EPSILON {
            1.0
        } else if diff < 0.0 {
            diff + 1.0
        }
        else {
            diff
        }
    }

    pub fn height(&self) -> f64 {
        /// Calculate the height of the bounds
        let diff = (self.bottom - self.top);
        if diff.abs() < f64::EPSILON {
            1.0
        } else if diff < 0.0 {
            diff + 1.0
        }
        else {
            diff
        }
    }

    pub fn from_x_y_zoom(x: u32, y: u32, zoom: u32) -> Self {
        /// Create a PixelBounds object from tile coordinates and zoom level
        let n = 2_u32.pow(zoom);
        
        let len = 1.0 / n as f64;
        let left = x as f64 * len;
        let right = (x + 1) as f64 * len;
        let top = y as f64 * len;
        let bottom = (y + 1) as f64 * len;

        Self::new(left, right, top, bottom)
    }

    pub fn all_x_y_zoom(&self, zoom: f32) -> HashSet<(u32, u32)> {
        let mut tiles = HashSet::new();
        let zoom_floor = zoom.floor() as u32;
        let n = 2_u32.pow(zoom_floor); // Number of total tiles in each row / column

        // Add all tiles to the list, accounting for wrapping
        match self.wrapping() {
            (false, false) => {
                // Normal case
                let x1 = (self.left() * n as f64).floor() as u32;
                let x2 = (self.right() * n as f64).floor() as u32;
                let y1 = (self.top() * n as f64).floor() as u32;
                let y2 = (self.bottom() * n as f64).floor() as u32;

                for x in x1..=x2 {
                    for y in y1..=y2 {
                        tiles.insert((x % n, y % n));
                    }
                }
            },
            (true, false) => {
                // Horizontal wrapping
                let x1 = (self.left() * n as f64).floor() as u32;
                let x2 = (self.right() * n as f64).floor() as u32;
                let y1 = (self.top() * n as f64).floor() as u32;
                let y2 = (self.bottom() * n as f64).floor() as u32;

                for x in x1..=n + x2 {
                    for y in y1..=y2 {
                        tiles.insert((x % n, y % n));
                    }
                }
            },
            (false, true) => {
                // Vertical wrapping
                let x1 = (self.left() * n as f64).floor() as u32;
                let x2 = (self.right() * n as f64).floor() as u32;
                let y1 = (self.top() * n as f64).floor() as u32;
                let y2 = (self.bottom() * n as f64).floor() as u32;

                for x in x1..=x2 {
                    for y in y1..=n + y2 {
                        tiles.insert((x % n, y % n));
                    }
                }
            },
            (true, true) => {
                // Both wrapping
                let x1 = (self.left() * n as f64).floor() as u32;
                let x2 = (self.right() * n as f64).floor() as u32;
                let y1 = (self.top() * n as f64).floor() as u32;
                let y2 = (self.bottom() * n as f64).floor() as u32;

                for x in x1..=n + x2 {
                    for y in y1..=n + y2 {
                        tiles.insert((x % n, y % n));
                    }
                }
            },
        }
        
        tiles
    }

    pub fn contains(&self, other: &PixelBounds) -> bool {
        /// Check if the other bounds are contained within this bounds, taking wrapping into account
        (if self.left <= self.right {
            // Normal case
            other.left >= self.left && other.right <= self.right
        } else {
            // Wrap-around case
            (other.left >= self.left && other.left <= 1.0) || 
            (other.right >= 0.0 && other.right <= self.right)
        }
        && 
        if self.top <= self.bottom {
            // Normal case
            other.top >= self.top && other.bottom <= self.bottom
        } else {
            // Wrap-around case
            (other.top >= self.top && other.top <= 1.0) ||
            (other.bottom >= 0.0 && other.bottom <= self.bottom)
        })
    }

    pub fn intersects(&self, other: &PixelBounds) -> bool {
        /// Check if the other bounds intersect with this bounds, taking wrapping into account
        match self.wrapping() {
            (false, false) => {
                // Normal case
                other.left <= self.right && other.right >= self.left && other.top <= self.bottom && other.bottom >= self.top
            },
            (true, false) => {
                // Horizontal wrapping
                (other.left <= self.right || other.right >= self.left) && other.top <= self.bottom && other.bottom >= self.top
            },
            (false, true) => {
                // Vertical wrapping
                other.left <= self.right && other.right >= self.left && (other.top <= self.bottom || other.bottom >= self.top)
            },
            (true, true) => {
                // Both wrapping
                (other.left <= self.right || other.right >= self.left) && (other.top <= self.bottom || other.bottom >= self.top)
            },
        }
    }

    pub fn intersect(&self, other: &PixelBounds) -> Option<PixelBounds> {
        /// Calculate the intersection of this bounds with another bounds, taking wrapping into account
        let dim_intersect = |a_start: f64, a_end: f64, b_start: f64, b_end: f64| -> Option<(f64, f64)> {
            if a_start <= a_end && b_start <= b_end {
                // No wrap case
                let left = a_start.max(b_start);
                let right = a_end.min(b_end);
                if left <= right {
                    Some((left, right))
                } else {
                    None
                }
            } else if a_start > a_end && b_start > b_end {
                // Both wrap
                let left = a_start.max(b_start);
                let right = a_end.min(b_end);
                Some((left, right))
            } else if a_start > a_end {
                // First wraps
                if b_start <= a_end || b_end >= a_start {
                    Some((b_start.max(a_start), b_end.min(a_end)))
                } else {
                    None
                }
            } else {
                // Second wraps
                if a_start <= b_end || a_end >= b_start {
                    Some((a_start.max(b_start), a_end.min(b_end)))
                } else {
                    None
                }
            }
        };

        // Get horizontal and vertical intersections
        let x_intersect = dim_intersect(self.left, self.right, other.left, other.right)?;
        let y_intersect = dim_intersect(self.top, self.bottom, other.top, other.bottom)?;

        Some(PixelBounds::new(
            x_intersect.0,
            x_intersect.1,
            y_intersect.0,
            y_intersect.1
        ))
    }

    pub fn from_center(center: PixelCoordinate, zoom: f32) -> Self {
        /// Create a PixelBounds object from a center coordinate, zoom level, and scale factor
        let remainder = zoom - zoom.floor();
        let z_int = zoom.floor() as i32;
        let len = egui::lerp(egui::Rangef::new(2.0_f32.powi(-(z_int + 1)), 2.0_f32.powi(-z_int)), 1.0 - remainder);
        let half_len = (len / 2.0) as f64;
        let left = center.x() - half_len;
        let right = center.x() + half_len;
        let top = center.y() - half_len;
        let bottom = center.y() + half_len;

        PixelBounds::new(left, right, top, bottom)
    }

    pub fn uv_map(&self, other: &PixelBounds) -> Vec<(egui::Rect, egui::Rect)> {
        /// Returns the mapping from the other bounds on to this bound. 
        /// Assumes that self may be wrapping, but other is not.
        // The first one is relative to this bounds, the second one is relative to the other bounds.
        // i.e. if other covers the bottom left quadrant of this, the first rect is egui::Rect::from_min_max(0.0, 0.0, 0.5, 0.5)
        // The second one has the same aspect ratio, but may be egui::Rect::from_min_max(0.5, 0.5, 1.0, 1.0) or could be egui::Rect::from_min_max(0.8, 0.8, 1.0, 1.0) if the other bounds is larger than this bounds.
        // Assumes that all bounds are square in mercator projection, and returns Rect::NOTHING if the bounds do not intersect.
        if !self.intersects(other) {
            return Vec::new();
        }

        let mut result = Vec::new();

        match self.wrapping() {
            (false, false) => {
                // Normal case, clamp -> shift -> expand
                let x1 = other.left.max(self.left);
                let x2 = other.right.min(self.right);
                let y1 = other.top.max(self.top);
                let y2 = other.bottom.min(self.bottom);

                let rect1 = egui::Rect::from_min_max(
                    pos2(((x1 - self.left) / self.width()) as f32, ((y1 - self.top) / self.height()) as f32),
                    pos2(((x2 - self.left) / self.width()) as f32, ((y2 - self.top) / self.height()) as f32),
                );

                let rect2 = egui::Rect::from_min_max(
                    pos2(((x1 - other.left) / other.width()) as f32, ((y1 - other.top) / other.height()) as f32),
                    pos2(((x2 - other.left) / other.width()) as f32, ((y2 - other.top) / other.height()) as f32),
                );

                result.push((rect1, rect2));
            },
            (true, false) => {
                // Horizontal wrapping, two sections to identify
                // Each section needs to be checked for intersection
                // Wrapping section follows procedure clamp -> join -> shift -> expand
                // Non-wrapping section follows procedure clamp -> shift -> expand
                let cont_section = (self.left(), 1.0_f64);
                // Do a regular intersection check for the non-wrapping section, then check the rect size.
                {
                    let x1 = other.left.max(cont_section.0);
                    let x2 = other.right.min(cont_section.1);
                    let y1 = other.top.max(self.top);
                    let y2 = other.bottom.min(self.bottom);

                    if x2 > x1 && y2 > y1 {
                        // There is a meaningful intersection
                        let rect1 = egui::Rect::from_min_max(
                            pos2(((x1 - cont_section.0) / self.width()) as f32, ((y1 - self.top) / self.height()) as f32),
                            pos2(((x2 - cont_section.0) / self.width()) as f32, ((y2 - self.top) / self.height()) as f32),
                        );

                        let rect2 = egui::Rect::from_min_max(
                            pos2(((x1 - other.left) / other.width()) as f32, ((y1 - other.top) / other.height()) as f32),
                            pos2(((x2 - other.left) / other.width()) as f32, ((y2 - other.top) / other.height()) as f32),
                        );
                        result.push((rect1, rect2));
                    }
                }
                // Do the wrapping section
                let segmented_segment = (0.0_f64, self.right());

                {
                    let x1 = other.left.max(segmented_segment.0);
                    let x2 = other.right.min(segmented_segment.1);
                    let y1 = other.top.max(self.top);
                    let y2 = other.bottom.min(self.bottom);

                    if x2 > x1 && y2 > y1 {
                        // There is a meaningful intersection
                        let rect1 = egui::Rect::from_min_max(
                            pos2(((x1 + 1.0 - self.left) / self.width()) as f32, ((y1 - self.top) / self.height()) as f32),
                            pos2(((x2 + 1.0 - self.left) / self.width()) as f32, ((y2 - self.top) / self.height()) as f32),
                        );

                        let rect2 = egui::Rect::from_min_max(
                            pos2(((x1 - other.left) / other.width()) as f32, ((y1 - other.top) / other.height()) as f32),
                            pos2(((x2 - other.left) / other.width()) as f32, ((y2 - other.top) / other.height()) as f32),
                        );
                        result.push((rect1, rect2));
                    }
                }

            },
            (false, true) => {
                // Vertical wrapping, two sections to identify
                // Each section needs to be checked for intersection
                // Wrapping section follows procedure clamp -> join -> shift -> expand
                // Non-wrapping section follows procedure clamp -> shift -> expand
                let cont_section = (self.top(), 1.0_f64);

                // Do a regular intersection check for the non-wrapping section, then check the rect size.
                {
                    let x1 = other.left.max(self.left);
                    let x2 = other.right.min(self.right);
                    let y1 = other.top.max(cont_section.0);
                    let y2 = other.bottom.min(cont_section.1);

                    if x2 > x1 && y2 > y1 {
                        // There is a meaningful intersection
                        let rect1 = egui::Rect::from_min_max(
                            pos2(((x1 - self.left) / self.width()) as f32, ((y1 - cont_section.0) / self.height()) as f32),
                            pos2(((x2 - self.left) / self.width()) as f32, ((y2 - cont_section.0) / self.height()) as f32),
                        );

                        let rect2 = egui::Rect::from_min_max(
                            pos2(((x1 - other.left) / other.width()) as f32, ((y1 - other.top) / other.height()) as f32),
                            pos2(((x2 - other.left) / other.width()) as f32, ((y2 - other.top) / other.height()) as f32),
                        );
                        result.push((rect1, rect2));
                    }
                }

                // Do the wrapping section
                let segmented_segment = (0.0_f64, self.bottom());

                {
                    let x1 = other.left.max(self.left);
                    let x2 = other.right.min(self.right);
                    let y1 = other.top.max(segmented_segment.0);
                    let y2 = other.bottom.min(segmented_segment.1);

                    if x2 > x1 && y2 > y1 {
                        // There is a meaningful intersection
                        let rect1 = egui::Rect::from_min_max(
                            pos2(((x1 - self.left) / self.width()) as f32, ((y1 + 1.0 - self.top) / self.height()) as f32),
                            pos2(((x2 - self.left) / self.width()) as f32, ((y2 + 1.0 - self.top) / self.height()) as f32),
                        );

                        let rect2 = egui::Rect::from_min_max(
                            pos2(((x1 - other.left) / other.width()) as f32, ((y1 - other.top) / other.height()) as f32),
                            pos2(((x2 - other.left) / other.width()) as f32, ((y2 - other.top) / other.height()) as f32),
                        );
                        result.push((rect1, rect2));
                    }
                }
            },
            (true, true) => {
                // Both wrapping, four sections to identify
                // Each section needs to be checked for intersection
                // Wrapping section follows procedure clamp -> join -> shift -> expand
                // Non-wrapping section follows procedure clamp -> shift -> expand
                let cont_section = (self.left(), 1.0_f64, self.top(), 1.0_f64);

                // Do a regular intersection check for the non-wrapping section, then check the rect size.
                {
                    let x1 = other.left.max(cont_section.0);
                    let x2 = other.right.min(cont_section.1);
                    let y1 = other.top.max(cont_section.2);
                    let y2 = other.bottom.min(cont_section.3);

                    if x2 > x1 && y2 > y1 {
                        // There is a meaningful intersection
                        let rect1 = egui::Rect::from_min_max(
                            pos2(((x1 - self.left) / self.width()) as f32, ((y1 - self.top) / self.height()) as f32),
                            pos2(((x2 - self.left) / self.width()) as f32, ((y2 - self.top) / self.height()) as f32),
                        );

                        let rect2 = egui::Rect::from_min_max(
                            pos2(((x1 - other.left) / other.width()) as f32, ((y1 - other.top) / other.height()) as f32),
                            pos2(((x2 - other.left) / other.width()) as f32, ((y2 - other.top) / other.height()) as f32),
                        );
                        result.push((rect1, rect2));
                    }
                }

                // Do the x only wrapping section
                let x_segment = (0.0_f64, self.right());
                {
                    let x1 = other.left.max(x_segment.0);
                    let x2 = other.right.min(x_segment.1);
                    let y1 = other.top.max(self.top());
                    let y2 = other.bottom.min(1.0_f64);

                    if x2 > x1 && y2 > y1 {
                        // There is a meaningful intersection
                        let rect1 = egui::Rect::from_min_max(
                            pos2(((x1 + 1.0 - self.left) / self.width()) as f32, ((y1 - self.top) / self.height()) as f32),
                            pos2(((x2 + 1.0 - self.left) / self.width()) as f32, ((y2 - self.top) / self.height()) as f32),
                        );

                        let rect2 = egui::Rect::from_min_max(
                            pos2(((x1 - other.left) / other.width()) as f32, ((y1 - other.top) / other.height()) as f32),
                            pos2(((x2 - other.left) / other.width()) as f32, ((y2 - other.top) / other.height()) as f32),
                        );
                        result.push((rect1, rect2));
                    }
                }

                // Do the y only wrapping section
                let y_segment = (0.0_f64, self.bottom());
                {
                    let x1 = other.left.max(self.left());
                    let x2 = other.right.min(1.0_f64);
                    let y1 = other.top.max(y_segment.0);
                    let y2 = other.bottom.min(y_segment.1);

                    if x2 > x1 && y2 > y1 {
                        // There is a meaningful intersection
                        let rect1 = egui::Rect::from_min_max(
                            pos2(((x1 - self.left) / self.width()) as f32, ((y1 + 1.0 - self.top) / self.height()) as f32),
                            pos2(((x2 - self.left) / self.width()) as f32, ((y2 + 1.0 - self.top) / self.height()) as f32),
                        );

                        let rect2 = egui::Rect::from_min_max(
                            pos2(((x1 - other.left) / other.width()) as f32, ((y1 - other.top) / other.height()) as f32),
                            pos2(((x2 - other.left) / other.width()) as f32, ((y2 - other.top) / other.height()) as f32),
                        );
                        result.push((rect1, rect2));
                    }
                }

                // Do the x and y wrapping section
                {
                    let x1 = other.left.max(0.0);
                    let x2 = other.right.min(self.right());
                    let y1 = other.top.max(0.0);
                    let y2 = other.bottom.min(self.bottom());

                    if x2 > x1 && y2 > y1 {
                        // There is a meaningful intersection
                        let rect1 = egui::Rect::from_min_max(
                            pos2(((x1 + 1.0 - self.left) / self.width()) as f32, ((y1 + 1.0 - self.top) / self.height()) as f32),
                            pos2(((x2 + 1.0 - self.left) / self.width()) as f32, ((y2 + 1.0 - self.top) / self.height()) as f32),
                        );

                        let rect2 = egui::Rect::from_min_max(
                            pos2(((x1 - other.left) / other.width()) as f32, ((y1 - other.top) / other.height()) as f32),
                            pos2(((x2 - other.left) / other.width()) as f32, ((y2 - other.top) / other.height()) as f32),
                        );
                        result.push((rect1, rect2));
                    }
                }
            },
        }
        
    
        result
    }

    pub fn wrapping(&self) -> (bool, bool) {
        ((self.right - self.left) < f64::EPSILON, (self.bottom - self.top) < f64::EPSILON)
    }
}

#[derive(Debug, Deserialize, Serialize, Clone)]
/// Purely functional wrapper for f64 coordinates, which represent a point between 0 and 1 that is a fraction of the image size.
pub struct PixelCoordinate {
    x: f64,
    y: f64,
}

impl Default for PixelCoordinate {
    fn default() -> Self {
        Self {
            x: 0.2,
            y: 0.6,
        }
    }
}

impl PixelCoordinate {
    pub fn new(x: f64, y: f64) -> Self {
        // Return the values modulo 1.0 to ensure they are in the range [0, 1]
        Self {
            x: x.rem_euclid(1.0),
            y: y.rem_euclid(1.0),
        }
    }

    pub fn from_lon_lat(lon: f64, lat: f64) -> Self {
        // Clamp latitude to valid mercator bounds
        const MAX_LAT: f64 = 85.05112878;
        let lat = lat.clamp(-MAX_LAT, MAX_LAT);
        
        // Convert longitude to x coordinate (0 to 1)
        let x = (lon + 180.0) / 360.0;
        
        // Convert latitude to y coordinate using mercator projection
        let lat_rad = lat.to_radians();
        let y = (1.0 - (lat_rad.tan() + (1.0 / lat_rad.cos())).ln() / std::f64::consts::PI) / 2.0;
        
        Self { x, y }
    }

    pub fn to_lon_lat(&self) -> (f64, f64) {
        // Convert x back to longitude
        let lon = self.x * 360.0 - 180.0;
        
        // Convert y back to latitude
        let lat_rad = std::f64::consts::PI * (1.0 - 2.0 * self.y);
        let lat = lat_rad.to_degrees().atan() * 2.0;
        
        (lon, lat)
    }

    pub fn x(&self) -> f64 {
        self.x
    }

    pub fn y(&self) -> f64 {
        self.y
    }

    pub fn add(&self, other: PixelCoordinate) -> PixelCoordinate {
        PixelCoordinate {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }

    pub fn add_vec(&self, other: egui::Vec2) -> PixelCoordinate {
        PixelCoordinate {
            x: self.x + other.x as f64,
            y: self.y + other.y as f64,
        }
    }

    pub fn add_delta(&self, delta: egui::Vec2, zoom: f32) -> PixelCoordinate {
        /// Add a delta in pixels to the coordinate, taking zoom into account
        /// Wraps around the edges of the image if the coordinate goes outside the bounds
        const BASE_SENSITIVITY: f64 = 0.005;
        
        // Calculate zoom-based scale factor
        // At zoom 0: scale = 1.0
        // At zoom 10: scale ≈ 0.001
        let zoom_scale = 2.0f64.powf(-zoom as f64);
        
        // Apply sensitivity and zoom scaling
        let scaled_delta_x = delta.x as f64 * BASE_SENSITIVITY * zoom_scale;
        let scaled_delta_y = delta.y as f64 * BASE_SENSITIVITY * zoom_scale;
        
        // Calculate new coordinates with wrapping
        let new_x = (self.x + scaled_delta_x).rem_euclid(1.0);
        let new_y = (self.y + scaled_delta_y).rem_euclid(1.0);
        
        PixelCoordinate::new(new_x, new_y)
    }
}

#[derive(Deserialize, Serialize)]
pub struct MapTile {
    pub x: u32,
    pub y: u32,
    pub zoom: u32,
    pub image_size: egui::Vec2,  // In pixels
    pub bounds: PixelBounds,   // Geographical bounds
    #[serde(skip)]
    texture: Option<egui::TextureHandle>, // literally dies if not an Option
}

impl MapTile {
    pub fn new(x: u32, y: u32, zoom: u32, image_size: egui::Vec2, bounds: PixelBounds, texture: &egui::TextureHandle) -> Self {
        Self {
            x,
            y,
            zoom,
            image_size,
            bounds,
            texture: Some(texture.to_owned()),
        }
    }

    pub fn texture(&self) -> egui::TextureId {
        self.texture.as_ref().unwrap().id()
    }
}

/// Convert a latitude and longitude to tile x, y coordinates for a given zoom.
/// Uses the Web Mercator projection.
fn lat_lon_to_tile_coords(lat: f64, lon: f64, zoom: u32) -> (u32, u32) {
    // Clip latitude to valid range (-85.0511 to 85.0511)
    let lat = lat.max(-85.0511).min(85.0511);
    
    // Convert to radians
    let lat_rad = lat.to_radians();
    let lon_rad = lon.to_radians();
    
    // Get number of tiles for zoom level
    let n = 2_u32.pow(zoom) as f64;
    
    // Calculate tile coordinates using Web Mercator formulas
    let x_tile = ((lon_rad + std::f64::consts::PI) / (2.0 * std::f64::consts::PI) * n).floor() as u32;
    let y_tile = ((1.0 - (lat_rad.tan().asinh() / std::f64::consts::PI)) / 2.0 * n).floor() as u32;
    
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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_pixel_bounds_new() {
        // Test normal case within [0,1]
        let bounds = PixelBounds::new(0.2, 0.4, 0.3, 0.5);
        assert_relative_eq!(bounds.left(), 0.2);
        assert_relative_eq!(bounds.right(), 0.4);
        assert_relative_eq!(bounds.top(), 0.3);
        assert_relative_eq!(bounds.bottom(), 0.5);

        // Test wrapping > 1.0
        let bounds = PixelBounds::new(1.2, 1.4, 1.3, 1.5);
        assert_relative_eq!(bounds.left(), 0.2);
        assert_relative_eq!(bounds.right(), 0.4);
        assert_relative_eq!(bounds.top(), 0.3);
        assert_relative_eq!(bounds.bottom(), 0.5);

        // Test wrapping < 0.0
        let bounds = PixelBounds::new(-0.8, -0.6, -0.7, -0.5);
        assert_relative_eq!(bounds.left(), 0.2);
        assert_relative_eq!(bounds.right(), 0.4);
        assert_relative_eq!(bounds.top(), 0.3);
        assert_relative_eq!(bounds.bottom(), 0.5);
    }

    #[test]
    fn test_pixel_bounds_width_height() {
        // Normal case
        let bounds = PixelBounds::new(0.2, 0.4, 0.3, 0.5);
        assert_relative_eq!(bounds.width(), 0.2);
        assert_relative_eq!(bounds.height(), 0.2);

        // Wrapping case
        let bounds = PixelBounds::new(0.8, 0.2, 0.8, 0.2);
        assert_relative_eq!(bounds.width(), 0.4);
        assert_relative_eq!(bounds.height(), 0.4);

        // Zero width/height case
        let bounds = PixelBounds::new(0.5, 0.5, 0.5, 0.5);
        assert_relative_eq!(bounds.width(), 1.0);
        assert_relative_eq!(bounds.height(), 1.0);
    }

    #[test]
    fn test_pixel_bounds_from_x_y_zoom() {
        // Test zoom level 0 (one tile)
        let bounds = PixelBounds::from_x_y_zoom(0, 0, 0);
        assert_relative_eq!(bounds.left(), 0.0);
        assert_relative_eq!(bounds.right(), 1.0);
        assert_relative_eq!(bounds.top(), 0.0);
        assert_relative_eq!(bounds.bottom(), 1.0);

        // Test zoom level 1 (four tiles)
        let bounds = PixelBounds::from_x_y_zoom(1, 1, 1);
        assert_relative_eq!(bounds.left(), 0.5);
        assert_relative_eq!(bounds.right(), 1.0);
        assert_relative_eq!(bounds.top(), 0.5);
        assert_relative_eq!(bounds.bottom(), 1.0);
    }

    #[test]
    fn test_pixel_bounds_wrapping() {
        // No wrapping
        let bounds = PixelBounds::new(0.2, 0.4, 0.3, 0.5);
        assert_eq!(bounds.wrapping(), (false, false));

        // Horizontal wrapping
        let bounds = PixelBounds::new(0.8, 0.2, 0.3, 0.5);
        assert_eq!(bounds.wrapping(), (true, false));

        // Vertical wrapping
        let bounds = PixelBounds::new(0.2, 0.4, 0.8, 0.2);
        assert_eq!(bounds.wrapping(), (false, true));

        // Both wrapping
        let bounds = PixelBounds::new(0.8, 0.2, 0.8, 0.2);
        assert_eq!(bounds.wrapping(), (true, true));
    }

    #[test]
    fn test_pixel_bounds_contains() {
        // Normal case
        let bounds = PixelBounds::new(0.2, 0.4, 0.3, 0.5);
        let inner = PixelBounds::new(0.25, 0.35, 0.35, 0.45);
        assert!(bounds.contains(&inner));

        // Wrapping case
        let bounds = PixelBounds::new(0.8, 0.2, 0.8, 0.2);
        let inner = PixelBounds::new(0.9, 0.1, 0.9, 0.1);
        assert!(bounds.contains(&inner));

        // Non-containing case
        let bounds = PixelBounds::new(0.2, 0.4, 0.3, 0.5);
        let outer = PixelBounds::new(0.1, 0.5, 0.2, 0.6);
        assert!(!bounds.contains(&outer));
    }

    #[test]
    fn test_pixel_bounds_intersects() {
        // Normal intersection
        let bounds1 = PixelBounds::new(0.2, 0.4, 0.3, 0.5);
        let bounds2 = PixelBounds::new(0.3, 0.5, 0.4, 0.6);
        assert!(bounds1.intersects(&bounds2));

        // Wrapping intersection
        let bounds1 = PixelBounds::new(0.8, 0.2, 0.8, 0.2);
        let bounds2 = PixelBounds::new(0.9, 0.1, 0.9, 0.1);
        assert!(bounds1.intersects(&bounds2));

        // No intersection
        let bounds1 = PixelBounds::new(0.2, 0.4, 0.3, 0.5);
        let bounds2 = PixelBounds::new(0.5, 0.7, 0.6, 0.8);
        assert!(!bounds1.intersects(&bounds2));
    }

    #[test]
    fn test_pixel_bounds_intersect() {
        // Normal intersection
        let bounds1 = PixelBounds::new(0.2, 0.4, 0.3, 0.5);
        let bounds2 = PixelBounds::new(0.3, 0.5, 0.4, 0.6);
        let intersection = bounds1.intersect(&bounds2).unwrap();
        assert_eq!(intersection.left(), 0.3);
        assert_eq!(intersection.right(), 0.4);
        assert_eq!(intersection.top(), 0.4);
        assert_eq!(intersection.bottom(), 0.5);

        // No intersection
        let bounds1 = PixelBounds::new(0.2, 0.4, 0.3, 0.5);
        let bounds2 = PixelBounds::new(0.5, 0.7, 0.6, 0.8);
        assert!(bounds1.intersect(&bounds2).is_none());
    }

    #[test]
    fn test_pixel_bounds_all_x_y_zoom() {
        // Test zoom level 0
        let bounds = PixelBounds::new(0.0, 1.0, 0.0, 1.0);
        let tiles = bounds.all_x_y_zoom(0.0);
        assert_eq!(tiles.len(), 1);
        assert!(tiles.contains(&(0, 0)));

        // Test zoom level 1 with partial coverage
        let bounds = PixelBounds::new(0.0, 0.5, 0.0, 0.5);
        let tiles = bounds.all_x_y_zoom(1.0);
        // assert_eq!(tiles.len(), 1);
        assert!(tiles.contains(&(0, 0)));

        // Test wrapping case
        let bounds = PixelBounds::new(0.8, 0.2, 0.8, 0.2);
        let tiles = bounds.all_x_y_zoom(1.0);
        assert_eq!(tiles.len(), 4);  // Should get all 4 corner tiles
    }

    #[test]
    fn test_pixel_bounds_uv_map() {
        // Normal case
        let bounds1 = PixelBounds::new(0.2, 0.4, 0.3, 0.5);
        let bounds2 = PixelBounds::new(0.3, 0.5, 0.4, 0.6);
        let mapping = bounds1.uv_map(&bounds2);
        assert_eq!(mapping.len(), 1);
        let (rect1, rect2) = mapping[0];
        assert_relative_eq!(rect1.min.x, 0.5);
        assert_relative_eq!(rect1.min.y, 0.5);
        assert_relative_eq!(rect1.max.x, 1.0);
        assert_relative_eq!(rect1.max.y, 1.0);
        assert_relative_eq!(rect2.min.x, 0.0);
        assert_relative_eq!(rect2.min.y, 0.0);
        assert_relative_eq!(rect2.max.x, 0.5);
        assert_relative_eq!(rect2.max.y, 0.5);

        // Wrapping case
        let bounds1 = PixelBounds::new(0.8, 0.2, 0.8, 0.2);
        let bounds2 = PixelBounds::new(0.7, 0.9, 0.7, 0.9);
        let mapping = bounds1.uv_map(&bounds2);
        assert_eq!(mapping.len(), 1);
        let (rect1, rect2) = mapping[0];
        assert_relative_eq!(rect1.min.x, 0.0);
        assert_relative_eq!(rect1.min.y, 0.0);
        assert_relative_eq!(rect1.max.x, 0.25);
        assert_relative_eq!(rect1.max.y, 0.25);
        assert_relative_eq!(rect2.min.x, 0.5);
        assert_relative_eq!(rect2.min.y, 0.5);
        assert_relative_eq!(rect2.max.x, 1.0);
        assert_relative_eq!(rect2.max.y, 1.0);

        // Wrapping case
        let bounds1 = PixelBounds::new(0.7, 0.7, 0.1, 0.1);
        let bounds2 = PixelBounds::new(0.0, 0.5, 0.5, 1.0);
        let mapping = bounds1.uv_map(&bounds2);
        assert_eq!(mapping.len(), 1); 
        let (rect1, rect2) = mapping[0];
        

    }
}
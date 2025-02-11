use std::collections::HashMap;
use std::error::Error;
use std::io::Cursor;
use geozero::mvt::tile::{Feature, GeomType, Layer, Value};
use geozero::mvt::{Message, Tile};
use reqwest;
use egui;
use image::{self, codecs::webp};
use serde::{Deserialize, Serialize};
use geozero::{mvt, FeatureProcessor, GeozeroGeometry};
use flate2::read::GzDecoder;
use std::io::Read;

use crate::map::map_tile::MapTile;
use crate::map::vector_tile::{GeometryType, VectorFeature, VectorLayer, VectorTile};

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct TileRetriever {
    #[serde(skip)]
    client: reqwest::Client,
    access_token: String,
    tile_size: u32,
    #[serde(skip)]
    ctx: egui::Context,
}

pub enum TileType {
    Raster(MapTile),
    Vector(VectorTile)
}

impl TileRetriever {
    pub fn new(access_token: String, tile_size: u32, ctx: egui::Context) -> Self {
        Self {
            client: reqwest::Client::new(),
            access_token,
            tile_size,
            ctx,
        }
    }

    /// Asynchronously fetches a tile and converts it into a MapTile.
    pub async fn fetch_tile(
        &self,
        zoom: u32,
        x: u32,
        y: u32,
    ) -> Result<TileType, Box<dyn Error>> {
        // Construct the URL using Mapbox's Raster Tiles API
        let url = format!(
            "https://api.mapbox.com/v4/mapbox.satellite/{}/{}/{}@2x.webp?access_token={}",
            zoom, x, y, self.access_token
        );
        println!("Fetching tile from {}", url);

        // Fetch the image data
        let response = self.client.get(&url).send().await?;

        // If the response is not successful, return an error
        if !response.status().is_success() {
            return Err(format!("Failed to fetch tile: {}", response.status()).into());
        }

        let bytes = response.bytes().await?;

        // Decode the image using the image crate
        let image = image::load_from_memory(&bytes)?;
        let image_buffer = image.to_rgba8();
        let (width, height) = image_buffer.dimensions();

        let texture = self.ctx.load_texture(
            format!("{}-{}-{}_tile", zoom, x, y),
            egui::ColorImage::from_rgba_unmultiplied(
                [width as usize, height as usize],
                &image_buffer.into_raw(),
            ),
            egui::TextureOptions::default(),
        );

        // Compute geographical bounds from the tile coordinates
        let bounds = crate::map::map_tile::PixelBounds::from_x_y_zoom(x, y, zoom);

        // Construct and return the MapTile
        Ok(TileType::Raster(MapTile::new(
            x,
            y,
            zoom,
            egui::vec2(512.0, 512.0),
            bounds,
            &texture,
        )))
    }

    pub async fn fetch_vector_tile(
        &self,
        zoom: u32,
        x: u32,
        y: u32,
    ) -> Result<TileType, Box<dyn Error>> {
        // Fetches pbf geo information from OSM using OpenFreeMap api, and converts them to a VectorTile
        let url = format!(
            "https://tiles.openfreemap.org/planet/stable/{}/{}/{}.pbf",
            zoom, x, y
        );
        
        let response = self.client.get(&url).send().await?;
        if !response.status().is_success() {
            return Err(format!("Failed to fetch vector tile: {}", response.status()).into());
        }
        

        println!("Processing vector tile from {}", url);
        let decompressed_bytes: Vec<u8> = { // Decompression block
            if response.headers().get("Content-Encoding").map_or(false, |header| header == "gzip") {
                let bytes = response.bytes().await?; // Read response bytes
                println!("Response is gzip encoded, decompressing...");
                decompress_gzip(&bytes)? // Decompress if gzip encoded
            } else {
                let bytes = response.bytes().await?; // Read response bytes
                println!("Response is not gzip encoded.");
                bytes.to_vec() // Use bytes directly if not gzip
            }
        };

        parse_vector_tile(decompressed_bytes) // Pass decompressed bytes to parser
    }
}

fn parse_vector_tile(data: Vec<u8>) -> Result<TileType, Box<dyn Error>> {
    let cursor = Cursor::new(data);
    let tile = Tile::decode(cursor).map_err(|e| format!("Geozero decode error: {}", e))?;

    let mut vector_tile = VectorTile::new(); // Create your custom VectorTile

    for layer in &tile.layers { // Iterate over geozero layers
        let vector_layer = parse_layer(layer)?; // Parse each layer
        vector_tile.layers.push(vector_layer); // Add parsed layer to VectorTile
    }

    Ok(TileType::Vector(vector_tile))
}

fn parse_layer(layer: &Layer) -> Result<VectorLayer, Box<dyn Error>> {
    let mut vector_layer = VectorLayer {
        name: layer.name.clone(),
        features: Vec::new(),
    };

    for feature in &layer.features { // Iterate over geozero features in the layer
        let vector_feature = parse_feature(feature, layer.extent(), layer)?; // Parse each feature, passing layer extent
        vector_layer.features.push(vector_feature); // Add parsed feature to VectorLayer
    }

    Ok(vector_layer)
}


fn parse_feature(feature: &Feature, extent: u32, layer: &Layer) -> Result<VectorFeature, Box<dyn Error>> {
    let geometry_type = match feature.r#type() { // Convert geozero GeomType to your GeometryType enum
        GeomType::Point => GeometryType::Point,
        GeomType::Linestring => GeometryType::Line,
        GeomType::Polygon => GeometryType::Polygon,
        _ => return Err(format!("Unsupported geometry type: {:?}", feature.r#type()).into()), // Handle other types if needed
    };

    let mut coordinates: Vec<Vec<(f32, f32)>> = Vec::new();
    let mut current_path: Vec<(f32, f32)> = Vec::new();
    
    // Create a geometry processor
    struct GeomProcessor {
        coordinates: Vec<Vec<(f32, f32)>>,
        current_path: Vec<(f32, f32)>,
        extent: f32,
    }

    impl geozero::GeomProcessor for GeomProcessor {
        fn xy(&mut self, x: f64, y: f64, _idx: usize) -> Result<(), geozero::error::GeozeroError> {
            let (nx, ny) = normalize_coords(x as i64, y as i64, self.extent);
            self.current_path.push((nx, ny));
            Ok(())
        }

        fn point_begin(&mut self, _idx: usize) -> Result<(), geozero::error::GeozeroError> {
            self.current_path.clear();
            Ok(())
        }

        fn point_end(&mut self, _idx: usize) -> Result<(), geozero::error::GeozeroError> {
            self.coordinates.push(self.current_path.clone());
            Ok(())
        }

        fn linestring_begin(&mut self, tagged: bool, _idx: usize, _size: usize) -> Result<(), geozero::error::GeozeroError> {
            self.current_path.clear();
            Ok(())
        }
    
        fn linestring_end(&mut self, tagged: bool, _idx: usize) -> Result<(), geozero::error::GeozeroError> {
            self.coordinates.push(self.current_path.clone());
            self.current_path.clear();
            Ok(())
        }
    
        fn polygon_begin(&mut self, tagged: bool, _idx: usize, _size: usize) -> Result<(), geozero::error::GeozeroError> {
            Ok(())
        }
    
        fn polygon_end(&mut self, tagged: bool, _idx: usize) -> Result<(), geozero::error::GeozeroError> {
            Ok(())
        }
    }

    let mut processor = GeomProcessor {
        coordinates: Vec::new(),
        current_path: Vec::new(),
        extent: extent as f32,
    };

    // Process the geometry
    feature.process_geom(&mut processor)?;
    coordinates = processor.coordinates;

    // Extract properties
    let mut properties = HashMap::new();
    let tags = feature.tags.chunks(2); // Tags come in pairs of [key_idx, value_idx]
    
    for chunk in tags {
        if let [key_idx, value_idx] = chunk {
            if let Some(key) = layer.keys.get(*key_idx as usize) {
                if let Some(value) = layer.values.get(*value_idx as usize) {
                    properties.insert(key.to_string(), value.string_value().to_string());
                }
            }
        }
    }

    Ok(VectorFeature {
        geometry_type,
        coordinates,
        properties,
    })
}


fn normalize_coords(x: i64, y: i64, extent: f32) -> (f32, f32) {
    // Normalizes tile coordinates to 0.0-1.0 range based on tile extent
    (
        x as f32 / extent,
        y as f32 / extent,
    )
}

fn decompress_gzip(compressed_bytes: &[u8]) -> Result<Vec<u8>, std::io::Error> {
    let mut decoder = GzDecoder::new(compressed_bytes);
    let mut decompressed_bytes = Vec::new();
    decoder.read_to_end(&mut decompressed_bytes)?;
    Ok(decompressed_bytes)
}
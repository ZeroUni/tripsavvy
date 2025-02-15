use std::collections::HashMap;
use std::error::Error;
use std::io::Cursor;
use std::sync::Arc;
use geozero::mvt::tile::{Feature, GeomType, Layer, Value};
use geozero::mvt::{Message, Tile, TileValue};
use reqwest;
use egui;
use image::{self, codecs::webp};
use serde::{Deserialize, Serialize};
use geozero::{mvt, FeatureProcessor, GeozeroGeometry};
use flate2::read::GzDecoder;
use std::io::Read;

use crate::map::map_tile::MapTile;
use crate::map::vector_tile::{FeatureValue, GeometryType, VectorFeature, VectorLayer, VectorTile};

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
                decompress_gzip(&bytes)? // Decompress if gzip encoded
            } else {
                let bytes = response.bytes().await?; // Read response bytes
                bytes.to_vec() // Use bytes directly if not gzip
            }
        };

        parse_vector_tile(decompressed_bytes) // Pass decompressed bytes to parser
    }
}

fn parse_vector_tile(data: Vec<u8>) -> Result<TileType, Box<dyn Error>> {
    let cursor = Cursor::new(data);
    let tile = Tile::decode(cursor).map_err(|e| format!("Geozero decode error: {}", e))?;

    let mut vector_tile_layers = Vec::new();

    for layer in &tile.layers { // Iterate over geozero layers
        let vector_layer = parse_layer(layer)?; // Parse each layer
        vector_tile_layers.push(vector_layer); // Add parsed layer to VectorTile
    }

    // For debugging purposes, write the vector tile as debug to a local file debug.txt
    // std::fs::write("debug/debug.txt", format!("{:#?}", vector_tile)).expect("Unable to write file");

    Ok(TileType::Vector(VectorTile::new(vector_tile_layers)))
}

fn parse_layer(layer: &Layer) -> Result<VectorLayer, Box<dyn Error>> {
    let mut vector_layer = VectorLayer {
        name: layer.name.clone(),
        features: Vec::new(),
    };

    for feature in &layer.features { // Iterate over geozero features in the layer
        let vector_feature = Arc::new(parse_feature(feature, layer.extent(), layer)?); // Parse each feature, passing layer extent
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

    // let mut coordinates: Vec<Arc<Vec<(f32, f32)>>> = Vec::new();
    
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

        fn empty_point(&mut self, idx: usize) -> geozero::error::Result<()> {
            self.current_path.push((0.0, 0.0));
            Ok(())
        }

        fn multipoint_begin(&mut self, size: usize, idx: usize) -> geozero::error::Result<()> {
            self.current_path.clear();
            Ok(())
        }

        fn multipoint_end(&mut self, idx: usize) -> geozero::error::Result<()> {
            self.coordinates.push(self.current_path.clone());
            Ok(())
        }

        fn coordinate(
                &mut self,
                x: f64,
                y: f64,
                z: Option<f64>,
                m: Option<f64>,
                t: Option<f64>,
                tm: Option<u64>,
                idx: usize,
            ) -> geozero::error::Result<()> {
                let (nx, ny) = normalize_coords(x as i64, y as i64, self.extent);
                self.current_path.push((nx, ny));
                Ok(())
        }

        fn multicurve_begin(&mut self, size: usize, idx: usize) -> geozero::error::Result<()> {
            self.current_path.clear();
            Ok(())
        }

        fn multicurve_end(&mut self, idx: usize) -> geozero::error::Result<()> {
            self.coordinates.push(self.current_path.clone());
            Ok(())
        }

        fn curvepolygon_begin(&mut self, size: usize, idx: usize) -> geozero::error::Result<()> {
            self.current_path.clear();
            Ok(())
        }

        fn curvepolygon_end(&mut self, idx: usize) -> geozero::error::Result<()> {
            self.coordinates.push(self.current_path.clone());
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
    // coordinates = processor.coordinates;
    // Convert coordinates from Vec<Vec<(f32, f32)>> to Arc<[Arc<[(f32, f32)]>]>
    let coordinates: Arc<[Arc<[(f32, f32)]>]> = Arc::from(processor.coordinates.into_iter().map(|v| Arc::from(v.into_boxed_slice())).collect::<Vec<_>>().into_boxed_slice());


    // Extract properties
    let mut properties = HashMap::new();
    let tags = feature.tags.chunks(2); // Tags come in pairs of [key_idx, value_idx]
    
    for chunk in tags {
        if let [key_idx, value_idx] = chunk {
            if let Some(key) = layer.keys.get(*key_idx as usize) {
                if let Some(value) = layer.values.get(*value_idx as usize) {
                    properties.insert(key.to_string(), convert_value(value));
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

fn convert_value(value: &Value) -> FeatureValue {
    // Convert geozero Value to FeatureValue using a match matrix
    match (
        value.string_value.as_ref(),
        value.float_value,
        value.double_value,
        value.int_value,
        value.uint_value,
        value.sint_value,
        value.bool_value,
    ) {
        (Some(s), _, _, _, _, _, _) => FeatureValue::String(s.to_string()),
        (_, Some(f), _, _, _, _, _) => FeatureValue::Float(f),
        (_, _, Some(d), _, _, _, _) => FeatureValue::Double(d),
        (_, _, _, Some(i), _, _, _) => FeatureValue::Int(i as i32),
        (_, _, _, _, Some(u), _, _) => FeatureValue::Uint(u as u32),
        (_, _, _, _, _, Some(s), _) => FeatureValue::Sint(s as i32),
        (_, _, _, _, _, _, Some(b)) => FeatureValue::Bool(b),
        _ => FeatureValue::String("".to_string()), // Default case
    }
}


// Tests
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_coords() {
        assert_eq!(normalize_coords(0, 0, 4096.0), (0.0, 0.0));
        assert_eq!(normalize_coords(2048, 2048, 4096.0), (0.5, 0.5));
        assert_eq!(normalize_coords(4096, 4096, 4096.0), (1.0, 1.0));
    }

    #[test]
    fn test_convert_value() {
        let value = Value {
            string_value: Some("string".to_string()),
            float_value: Some(1.0),
            double_value: Some(2.0),
            int_value: Some(3),
            uint_value: Some(4),
            sint_value: Some(5),
            bool_value: Some(true),
        };

        assert_eq!(
            convert_value(&value),
            FeatureValue::String("string".to_string())
        );
    }

    #[tokio::test]
    async fn test_fetch_vector_tile() {
        let ctx = egui::Context::default();
        let tile_retriever = TileRetriever::new("your_test_token".to_string(), 512, ctx);
        
        let result = tile_retriever.fetch_vector_tile(9, 137, 184).await;
        assert!(result.is_ok(), "Failed to fetch vector tile: {:?}", result.err());
        
        if let Ok(TileType::Vector(vector_tile)) = result {
            assert!(!vector_tile.layers.is_empty(), "Vector tile should contain layers");
            let layer_names = vector_tile.get_all_layer_names();
            println!("Layer names: {:?}", layer_names);

            // Write the debug info of the vector tile to target/debug/debug.txt
            std::fs::write("target/debug/debug.txt", format!("{:#?}", vector_tile)).expect("Unable to write file");
        } else {
            panic!("Expected Vector tile type");
        }
    }
}
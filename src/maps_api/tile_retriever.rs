use std::error::Error;
use reqwest;
use egui;
use image::{self, codecs::webp};
use serde::{Deserialize, Serialize};

use crate::map::map_tile::MapTile;

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct TileRetriever {
    #[serde(skip)]
    client: reqwest::Client,
    access_token: String,
    tile_size: u32,
}

impl TileRetriever {
    pub fn new(access_token: String, tile_size: u32) -> Self {
        Self {
            client: reqwest::Client::new(),
            access_token,
            tile_size,
        }
    }

    /// Asynchronously fetches a tile and converts it into a MapTile.
    pub async fn fetch_tile(
        &self,
        zoom: u32,
        x: u32,
        y: u32,
    ) -> Result<MapTile, Box<dyn Error>> {
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
        // let image = image::load_from_memory_with_format(&bytes, image::ImageFormat::WebP)?;
        // let image_buffer = image.to_rgba8();
        // let (width, height) = image_buffer.dimensions();

        // Compute geographical bounds from the tile coordinates
        let bounds = crate::map::map_tile::PixelBounds::from_x_y_zoom(x, y, zoom);

        // Construct and return the MapTile
        Ok(MapTile::new(
            x,
            y,
            zoom,
            egui::vec2(512.0, 512.0),
            bounds,
            bytes.to_vec(),
        ))
    }
}

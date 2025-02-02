use std::error::Error;
use reqwest;
use egui;
use image;
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
        let bytes = response.bytes().await?;

        // Decode the image using the image crate
        let image = image::load_from_memory(&bytes)?;
        let image_buffer = image.to_rgba8();
        let (width, height) = image_buffer.dimensions();

        // Compute geographical bounds from the tile coordinates
        let geo_bounds = crate::map::map_tile::tile_coords_to_geo_bounds(x, y, zoom);

        // Construct and return the MapTile
        Ok(MapTile::new(
            x,
            y,
            zoom,
            egui::vec2(width as f32, height as f32),
            geo_bounds,
            image_buffer.into_raw()
        ))
    }
}

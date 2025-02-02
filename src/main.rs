#![warn(clippy::all, rust_2018_idioms)]
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] 

mod ui;
mod map;
mod maps_api;

use dotenv;

#[cfg(not(target_arch = "wasm32"))]
fn main() -> eframe::Result<()> {
    use maps_api::tile_retriever;

    env_logger::init(); 

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size(egui::vec2(1920.0, 1080.0))
            .with_min_inner_size(egui::vec2(400.0, 300.0))
            .with_title("TripSavvy")
            .with_resizable(true)
            .with_decorations(true)
            .with_drag_and_drop(true),
        ..Default::default()
    };

    let key = dotenv::var("MAP_BOX_API_TOKEN").expect("MAP_BOX_API_TOKEN must be set");

    let tile_retriever = tile_retriever::TileRetriever::new(
        key.clone(),
        512,
    );

    eframe::run_native(
        "TripSavvy",
        native_options,
        Box::new(|cc| {
            Ok(Box::new(ui::my_app::MyApp::new(cc, tile_retriever)))
        }),
    )
}
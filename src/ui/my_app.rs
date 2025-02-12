use eframe::egui;
use egui::emath::Numeric;
use egui::{frame, Color32, Margin, Style};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use std::collections::{HashMap, HashSet};
use std::num::NonZeroU16;
use std::{env, mem};
use std::error::Error;
use lru::LruCache;
use crate::map;
use crate::map::map_tile::{Coordinate, MapTile};
use crate::map::map::Map;
use crate::map::vector_tile::VectorTile;
use crate::maps_api::tile_retriever::{TileRetriever, TileType};

#[derive(Deserialize, Serialize)]
#[serde(default)]

pub struct MyApp {
    #[serde(skip)]
    memory: LruCache<(u32, u32, u32), MapTile>,
    #[serde(skip)]
    memory_overlay: LruCache<(u32, u32, u32), VectorTile>,
    tile_retriever: TileRetriever,
    #[serde(skip)]
    pending_tiles: HashSet<(u32, u32, u32)>,
    #[serde(skip)]
    receiver: mpsc::UnboundedReceiver<(u32, u32, u32, Result<TileType, Box<dyn Error + Send + Sync>>)>,
    #[serde(skip)]
    sender: mpsc::UnboundedSender<(u32, u32, u32, Result<TileType, Box<dyn Error + Send + Sync>>)>,
    #[serde(skip)]
    runtime: tokio::runtime::Runtime,
}

impl Default for MyApp {
    fn default() -> Self {
        Self {
            memory: LruCache::new(NonZeroU16::new(512).unwrap_or(NonZeroU16::MAX).into()),
            memory_overlay: LruCache::new(NonZeroU16::new(512).unwrap_or(NonZeroU16::MAX).into()),
            tile_retriever: TileRetriever::new("".to_string(), 512, egui::Context::default()),
            pending_tiles: HashSet::new(),
            receiver: mpsc::unbounded_channel().1,
            sender: mpsc::unbounded_channel().0,
            runtime: tokio::runtime::Builder::new_multi_thread().enable_all().build().expect("Unable to create runtime")
        }
    }
}

impl eframe::App for MyApp {
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, eframe::APP_KEY, self);
    }
    
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {

        // Test for f11 key, to toggle fullscreen
        if let Some(new_fullscreen) = ctx.input(|i| {
            if i.key_pressed(egui::Key::F11) { Some(!i.viewport().fullscreen.unwrap_or(false)) }
            else                       { None                                            }
        }) {
            ctx.send_viewport_cmd(egui::ViewportCommand::Fullscreen(  new_fullscreen));
            ctx.send_viewport_cmd(egui::ViewportCommand::Decorations(!new_fullscreen));
            ctx.send_viewport_cmd(egui::ViewportCommand::Maximized(  !new_fullscreen));
            ctx.send_viewport_cmd(egui::ViewportCommand::Focus);
        }

        // TEMP FOR NOW, display a single image that is 0 0 0 tile, cached
        let frame = egui::Frame {
            fill: egui::Color32::TRANSPARENT,
            stroke: egui::Stroke::new(1.0, egui::Color32::WHITE),
            inner_margin: Margin::same(1.0),
            outer_margin: Margin::same(0.0),
            ..Default::default()
        };

        // egui::Window::new("Map") // TODO: Replace window with map widget
        //         .frame(frame)
        //         .fixed_size(egui::vec2(1024.0, 1024.0))
        //         .scroll([false, false])
        //         .title_bar(false)
        //         .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
        //         .show(ctx, |ui| {
        //             ui.set_height(1024.0);
        //             ui.set_width(1024.0);
        //             let coords = (self.zoom, 0, 0);
        //             if let Some(tile) = self.memory.get_mut(&coords) {
        //                 ui.painter().image(tile.texture(ctx).id(), ui.max_rect(), egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), Color32::WHITE);
        //             } else {
        //                 ui.painter().rect_filled(ui.max_rect(), 0.0, Color32::from_gray(128));
        //                 // Check if we need to fetch the tile, or are waiting for it
        //                 if !self.pending_tiles.contains(&coords) && !self.memory.contains_key(&coords) {
        //                     let sender = self.sender.clone();
        //                     let tile_retriever = self.tile_retriever.clone();
        //                     let (x, y, z) = coords;
                
        //                     self.runtime.spawn(async move {
        //                         println!("Fetching tile ({}, {}, {})", x, y, z);
        //                         let result = tile_retriever.fetch_tile(x, y, z).await
        //                             .map_err(|e| -> Box<dyn Error + Send + Sync> {
        //                                 Box::new(std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))
        //                             });
        //                         sender.send((x, y, z, result)).unwrap();
        //                     });
                
        //                     self.pending_tiles.insert(coords);
        //                 }
        //             }
        //             let response = ui.response();
        //             if let (true, 0.0) = (
        //                 response.contains_pointer, 
        //                 ui.input(|i| i.time_since_last_scroll()), 
        //             ) { 
        //                 // Get any scroll delta that may have occurred
        //                 let scroll_delta = ui.input(|i| i.raw_scroll_delta.y);
        //                 // If the scroll delta is not zero, update the zoom level (-1 or 1 steps based on direction)
        //                 if scroll_delta != 0.0 {
        //                     let change = if scroll_delta > 0.0 { 1 } else { -1 };
        //                     self.zoom = (self.zoom as i32 + change).clamp(0, 20) as u32;
        //                 }
        //             }
        //         });
        let center = ctx.screen_rect().center();
        egui::Window::new("Map") 
                .frame(frame)
                .fixed_size(egui::vec2(1024.0, 1024.0))
                .scroll([false, false])
                .title_bar(false)
                .anchor(egui::Align2::CENTER_CENTER, [-480.0, 0.0])
                .show(ctx, |ui| {
                    // Hide debug info
                    ui.style_mut().debug.debug_on_hover = false;

                    let mut missing_tiles = Vec::new();
                    let map = Map::new("interactible_map", &mut self.memory, &mut self.memory_overlay, &mut missing_tiles).viewport_size(egui::vec2(780.0, 780.0));
                    
                    ui.add(map);

                    
                    for (z, x, y) in missing_tiles {
                        // Check if we need to fetch the tile, or are waiting for it
                        if !self.pending_tiles.contains(&(z, x, y)) && self.memory.peek(&(z, x, y)).is_none() {
                            let sender = self.sender.clone();
                            let tile_retriever = self.tile_retriever.clone();
                            let requester = ctx.clone(); // Uses ARC so can be cloned to a new thread cheaply

                            // TEMP add a secondary fetcher for vector tiles
                            let vector_retriever = tile_retriever.clone();
                            let vector_sender = self.sender.clone();
                            
                            self.runtime.spawn(async move {
                                println!("Fetching tile ({}, {}, {})", x, y, z);
                                let result = tile_retriever.fetch_tile(z, x, y).await
                                    .map_err(|e| -> Box<dyn Error + Send + Sync> {
                                        Box::new(std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))
                                    });
                                sender.send((x, y, z, result)).unwrap();
                                requester.request_repaint();
                            });

                            self.runtime.spawn( async move {
                                let result = vector_retriever.fetch_vector_tile(z, x, y).await
                                    .map_err(|e| -> Box<dyn Error + Send + Sync> {
                                        Box::new(std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))
                                    });

                                vector_sender.send((x, y, z, result)).unwrap();
                            });
                            
                            self.pending_tiles.insert((z, x, y));
                        }
                    }
                });

        // Check if the future is ready
        // Process completed requests
        while let Ok((x, y, z, result)) = self.receiver.try_recv() {
            match result {
                Ok(tile) => {
                    match tile {
                        TileType::Raster(texture) => {
                            self.memory.put((z, x, y), texture);
                            self.pending_tiles.remove(&(z, x, y));
                        }
                        TileType::Vector(vector_tile) => {
                            self.memory_overlay.put((z, x, y), vector_tile);
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Error fetching tile ({}, {}, {}): {}", x, y, z, e);
                }
            }
        }
        
    }
}

impl MyApp {
    pub fn new(cc: &eframe::CreationContext<'_>, tile_retriever: TileRetriever) -> Self {
        cc.egui_ctx.set_style(Self::get_dark_theme_style(&cc.egui_ctx));
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(8) // Set max number of worker threads
            .thread_name("tile-fetcher")
            .thread_stack_size(3 * 1024 * 1024) // 3MB stack size
            .enable_all()
            .build()
            .expect("Unable to create runtime");
        let (sender, receiver) = mpsc::unbounded_channel();
        Self {
            memory: LruCache::new(NonZeroU16::new(512).unwrap_or(NonZeroU16::MAX).into()),
            memory_overlay: LruCache::new(NonZeroU16::new(512).unwrap_or(NonZeroU16::MAX).into()),
            tile_retriever,
            pending_tiles: HashSet::new(),
            receiver,
            sender,
            runtime,
        }
    }

    pub fn from_storage(storage: &mut dyn eframe::Storage) -> Self {
        eframe::get_value(storage, eframe::APP_KEY).unwrap_or_default()
    }

    pub fn get_dark_theme_style(ctx: &egui::Context) -> Style {
        use egui::{
            style::{Selection, Visuals, Widgets},
            Color32, FontFamily, FontId, Rounding, Stroke, TextStyle,
        };
    
        let mut style = (*ctx.style()).clone();
    
        // Set text styles
        style.text_styles = [
            (TextStyle::Heading, FontId::new(22.0, FontFamily::Proportional)),
            (TextStyle::Body, FontId::new(18.0, FontFamily::Proportional)),
            (TextStyle::Monospace, FontId::new(16.0, FontFamily::Monospace)),
            (TextStyle::Button, FontId::new(18.0, FontFamily::Proportional)),
            (TextStyle::Small, FontId::new(14.0, FontFamily::Proportional)),
        ]
        .into();
    
        // Primary background color
        let primary_bg_color = Color32::from_rgb(32, 33, 36);
    
        // Configure visuals
        style.visuals = Visuals::dark();
        style.visuals.override_text_color = Some(Color32::LIGHT_GRAY);
        style.visuals.widgets = Widgets {
            noninteractive: egui::style::WidgetVisuals {
                bg_fill: primary_bg_color,
                bg_stroke: Stroke::new(1.0, Color32::from_gray(60)),
                fg_stroke: Stroke::new(1.0, Color32::LIGHT_GRAY),
                rounding: Rounding::same(4.0),
                weak_bg_fill: Color32::from_gray(32),
                expansion: 0.0,
            },
            inactive: egui::style::WidgetVisuals {
                bg_fill: primary_bg_color,
                bg_stroke: Stroke::new(1.0, Color32::from_gray(75)),
                fg_stroke: Stroke::new(1.0, Color32::LIGHT_GRAY),
                rounding: Rounding::same(4.0),
                weak_bg_fill: Color32::from_gray(32),
                expansion: 0.0,
            },
            hovered: egui::style::WidgetVisuals {
                bg_fill: Color32::from_rgb(50, 50, 50),
                bg_stroke: Stroke::new(1.0, Color32::WHITE),
                fg_stroke: Stroke::new(1.0, Color32::WHITE),
                rounding: Rounding::same(4.0),
                weak_bg_fill: Color32::from_gray(32),
                expansion: 0.5,
            },
            active: egui::style::WidgetVisuals {
                bg_fill: Color32::from_rgb(60, 60, 60),
                bg_stroke: Stroke::new(1.0, Color32::WHITE),
                fg_stroke: Stroke::new(1.0, Color32::WHITE),
                rounding: Rounding::same(4.0),
                weak_bg_fill: Color32::from_gray(32),
                expansion: 2.0,
            },
            open: egui::style::WidgetVisuals {
                bg_fill: Color32::from_rgb(40, 40, 40),
                bg_stroke: Stroke::new(1.0, Color32::WHITE),
                fg_stroke: Stroke::new(1.0, Color32::WHITE),
                rounding: Rounding::same(4.0),
                weak_bg_fill: Color32::from_gray(32),
                expansion: 0.0,
            },
        };
    
        // Selection colors
        style.visuals.selection = Selection {
            bg_fill: Color32::from_rgb(75, 75, 75),
            stroke: Stroke::new(1.0, Color32::WHITE),
        };
    
        // Window settings
        style.visuals.window_rounding = Rounding::same(6.0);
        style.visuals.window_shadow = egui::Shadow {
            offset: egui::vec2(0.0, 1.0),
            blur: 3.0,
            spread: 0.0,
            color: Color32::from_black_alpha(128),
        };
        style.visuals.window_fill = primary_bg_color;
        style.visuals.window_stroke = Stroke::new(1.0, Color32::from_gray(60));
        style.visuals.panel_fill = primary_bg_color;
    
        // Spacing settings
        //style.spacing.item_spacing = egui::vec2(8.0, 6.0);
        style.spacing.window_margin = egui::Margin::same(4.0);
        style.spacing.button_padding = egui::vec2(2.0, 2.0);
    
        style
    }
}
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub enum GeometryType {
    Point,
    Line,
    Polygon,
}

#[derive(Debug, Clone)]
pub struct VectorFeature {
    pub geometry_type: GeometryType,
    pub coordinates: Vec<Vec<(f32, f32)>>,
    pub properties: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct VectorLayer {
    pub name: String,
    pub features: Vec<VectorFeature>,
}

#[derive(Debug, Clone)]
pub struct VectorTile {
    pub layers: Vec<VectorLayer>,
}

impl VectorTile {
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
        }
    }
}
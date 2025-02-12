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
    pub properties: HashMap<String, FeatureValue>,
}

impl VectorFeature {
    pub fn get_class(&self) -> &str {
        match self.properties.get("class") {
            Some(FeatureValue::String(class)) => class,
            _ => "",
        }
    }

    pub fn get_name(&self) -> &str {
        match self.properties.get("name") {
            Some(FeatureValue::String(name)) => name,
            _ => "",
        }
    }

    pub fn get_rank(&self) -> i32 {
        match self.properties.get("rank") {
            Some(FeatureValue::Sint(rank)) => *rank,
            _ => 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct VectorLayer {
    pub name: String,
    pub features: Vec<VectorFeature>,
}

impl VectorLayer {
    pub fn get_lowest_rank(&self) -> i32 {
        let mut lowest_rank = i32::MAX;
        for feature in &self.features {
            let rank = feature.get_rank();
            if rank < lowest_rank {
                lowest_rank = rank;
            }
        }
        lowest_rank
    }
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

#[derive(Debug, Clone)]
pub enum FeatureValue {
    String(String),
    Float(f32),
    Double(f64),
    Int(i32),
    Uint(u32),
    Sint(i32),
    Bool(bool),
}

// Use eq. based on enum inner value
impl PartialEq for FeatureValue {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (FeatureValue::String(a), FeatureValue::String(b)) => a == b,
            (FeatureValue::Float(a), FeatureValue::Float(b)) => a == b,
            (FeatureValue::Double(a), FeatureValue::Double(b)) => a == b,
            (FeatureValue::Int(a), FeatureValue::Int(b)) => a == b,
            (FeatureValue::Uint(a), FeatureValue::Uint(b)) => a == b,
            (FeatureValue::Sint(a), FeatureValue::Sint(b)) => a == b,
            (FeatureValue::Bool(a), FeatureValue::Bool(b)) => a == b,
            _ => false,
        }
    }
}

impl FeatureValue {
    pub fn inner(&self) -> &dyn std::any::Any {
        match self {
            FeatureValue::String(v) => v,
            FeatureValue::Float(v) => v,
            FeatureValue::Double(v) => v,
            FeatureValue::Int(v) => v,
            FeatureValue::Uint(v) => v,
            FeatureValue::Sint(v) => v,
            FeatureValue::Bool(v) => v,
        }
    }

    pub fn as_string(&self) -> Option<&String> {
        match self {
            FeatureValue::String(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_float(&self) -> Option<&f32> {
        match self {
            FeatureValue::Float(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_double(&self) -> Option<&f64> {
        match self {
            FeatureValue::Double(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_int(&self) -> Option<&i32> {
        match self {
            FeatureValue::Int(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_uint(&self) -> Option<&u32> {
        match self {
            FeatureValue::Uint(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_sint(&self) -> Option<&i32> {
        match self {
            FeatureValue::Sint(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<&bool> {
        match self {
            FeatureValue::Bool(v) => Some(v),
            _ => None,
        }
    }
    
}
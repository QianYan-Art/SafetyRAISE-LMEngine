//! Read-only-model Q8 sidecar cache.
//!
//! The cache stores derived Q8 linear weights outside the source model
//! directory. Source safetensors files are only inspected for metadata.

use std::collections::HashMap;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use serde::{Deserialize, Serialize};

use crate::error::{Result, RsinferError};
use crate::tensor::Q8LinearWeight;

const MANIFEST_FILE: &str = "manifest.json";
const MAGIC: &[u8; 8] = b"RSQ8W001";

#[derive(Debug)]
pub struct Q8SidecarCache {
    dir: PathBuf,
    manifest: Q8SidecarManifest,
    pub hits: usize,
    pub writes: usize,
    pub fallbacks: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Q8SidecarManifest {
    format: String,
    model_fingerprint: String,
    entries: HashMap<String, Q8SidecarEntry>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Q8SidecarEntry {
    file: String,
    out_features: usize,
    in_features: usize,
}

#[derive(Clone, Debug)]
pub struct Q8SidecarReport {
    pub dir: PathBuf,
    pub hits: usize,
    pub writes: usize,
    pub fallbacks: Vec<String>,
}

impl Q8SidecarCache {
    pub fn open(model_dir: &Path, override_dir: Option<&Path>) -> Result<Self> {
        let dir = match override_dir {
            Some(path) => path.to_path_buf(),
            None => default_sidecar_dir(model_dir)?,
        };
        fs::create_dir_all(&dir)?;

        let fingerprint = fingerprint_model_dir(model_dir)?;
        let manifest_path = dir.join(MANIFEST_FILE);
        let manifest = match fs::read_to_string(&manifest_path) {
            Ok(content) => match serde_json::from_str::<Q8SidecarManifest>(&content) {
                Ok(manifest) if manifest.model_fingerprint == fingerprint => manifest,
                _ => Q8SidecarManifest::new(&fingerprint),
            },
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                Q8SidecarManifest::new(&fingerprint)
            }
            Err(err) => return Err(err.into()),
        };

        Ok(Self {
            dir,
            manifest,
            hits: 0,
            writes: 0,
            fallbacks: Vec::new(),
        })
    }

    pub fn load_or_create<F>(
        &mut self,
        name: &str,
        out_features: usize,
        in_features: usize,
        create: F,
    ) -> Result<Q8LinearWeight>
    where
        F: FnOnce() -> Result<Q8LinearWeight>,
    {
        if let Some(weight) = self.try_load(name, out_features, in_features) {
            self.hits += 1;
            return Ok(weight);
        }

        let weight = create()?;
        self.store(name, &weight)?;
        self.writes += 1;
        Ok(weight)
    }

    pub fn record_fallback(&mut self, name: &str, err: impl Into<String>) {
        self.fallbacks.push(format!("{name}: {}", err.into()));
    }

    pub fn report(&self) -> Q8SidecarReport {
        Q8SidecarReport {
            dir: self.dir.clone(),
            hits: self.hits,
            writes: self.writes,
            fallbacks: self.fallbacks.clone(),
        }
    }

    fn try_load(
        &mut self,
        name: &str,
        out_features: usize,
        in_features: usize,
    ) -> Option<Q8LinearWeight> {
        let entry = self.manifest.entries.get(name)?;
        if entry.out_features != out_features || entry.in_features != in_features {
            return None;
        }
        let path = self.dir.join(&entry.file);
        match read_q8_weight(&path, out_features, in_features) {
            Ok(weight) => Some(weight),
            Err(err) => {
                self.record_fallback(name, format!("sidecar read failed: {err}"));
                None
            }
        }
    }

    fn store(&mut self, name: &str, weight: &Q8LinearWeight) -> Result<()> {
        let file = format!("{:016x}.q8bin", fnv1a64(name.as_bytes()));
        let path = self.dir.join(&file);
        write_q8_weight_atomic(&path, weight)?;
        self.manifest.entries.insert(
            name.to_string(),
            Q8SidecarEntry {
                file,
                out_features: weight.out_features,
                in_features: weight.in_features,
            },
        );
        self.write_manifest()
    }

    fn write_manifest(&self) -> Result<()> {
        let manifest_path = self.dir.join(MANIFEST_FILE);
        let tmp_path = self.dir.join("manifest.tmp");
        let json = serde_json::to_vec_pretty(&self.manifest)?;
        fs::write(&tmp_path, json)?;
        fs::rename(tmp_path, manifest_path)?;
        Ok(())
    }
}

impl Q8SidecarManifest {
    fn new(fingerprint: &str) -> Self {
        Self {
            format: "rsinfer-q8-sidecar-v1".to_string(),
            model_fingerprint: fingerprint.to_string(),
            entries: HashMap::new(),
        }
    }
}

fn default_sidecar_dir(model_dir: &Path) -> Result<PathBuf> {
    let parent = model_dir.parent().ok_or_else(|| {
        RsinferError::WeightError(format!(
            "Cannot derive sidecar parent for model dir {}",
            model_dir.display()
        ))
    })?;
    let name = model_dir
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| {
            RsinferError::WeightError(format!(
                "Cannot derive sidecar name for model dir {}",
                model_dir.display()
            ))
        })?;
    Ok(parent.join(format!("{name}.rsinfer-q8")))
}

fn fingerprint_model_dir(model_dir: &Path) -> Result<String> {
    let mut parts = Vec::new();
    for entry in fs::read_dir(model_dir)? {
        let entry = entry?;
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
            continue;
        };
        let is_model_file = name == "config.json"
            || name == "model.safetensors"
            || name == "model.safetensors.index.json"
            || (name.starts_with("model-") && name.ends_with(".safetensors"));
        if !is_model_file {
            continue;
        }
        let metadata = entry.metadata()?;
        let modified = metadata
            .modified()?
            .duration_since(UNIX_EPOCH)
            .map_err(|err| RsinferError::WeightError(format!("Invalid mtime for {name}: {err}")))?;
        parts.push(format!(
            "{}:{}:{}:{}",
            name,
            metadata.len(),
            modified.as_secs(),
            modified.subsec_nanos()
        ));
    }
    parts.sort();
    Ok(format!("{:016x}", fnv1a64(parts.join("|").as_bytes())))
}

fn write_q8_weight_atomic(path: &Path, weight: &Q8LinearWeight) -> Result<()> {
    let tmp_path = path.with_extension("tmp");
    {
        let mut file = fs::File::create(&tmp_path)?;
        file.write_all(MAGIC)?;
        file.write_all(&(weight.out_features as u64).to_le_bytes())?;
        file.write_all(&(weight.in_features as u64).to_le_bytes())?;
        for &scale in &weight.scales {
            file.write_all(&scale.to_le_bytes())?;
        }
        let qbytes: Vec<u8> = weight.qweight.iter().map(|&value| value as u8).collect();
        file.write_all(&qbytes)?;
        file.sync_all()?;
    }
    fs::rename(tmp_path, path)?;
    Ok(())
}

fn read_q8_weight(path: &Path, out_features: usize, in_features: usize) -> Result<Q8LinearWeight> {
    let mut file = fs::File::open(path)?;
    let mut magic = [0u8; 8];
    file.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(RsinferError::WeightError(format!(
            "Invalid Q8 sidecar magic in {}",
            path.display()
        )));
    }

    let stored_out = read_u64(&mut file)? as usize;
    let stored_in = read_u64(&mut file)? as usize;
    if stored_out != out_features || stored_in != in_features {
        return Err(RsinferError::ShapeMismatch {
            expected: vec![out_features, in_features],
            actual: vec![stored_out, stored_in],
        });
    }

    let mut scale_bytes = vec![0u8; out_features * std::mem::size_of::<f32>()];
    file.read_exact(&mut scale_bytes)?;
    let scales = scale_bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    let expected = out_features * in_features;
    let mut qbytes = vec![0u8; expected];
    file.read_exact(&mut qbytes)?;
    let qweight = qbytes.into_iter().map(|value| value as i8).collect();

    Ok(Q8LinearWeight {
        qweight,
        scales,
        out_features,
        in_features,
    })
}

fn read_u64(file: &mut fs::File) -> Result<u64> {
    let mut bytes = [0u8; 8];
    file.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for &byte in bytes {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    #[test]
    fn q8_sidecar_roundtrip_preserves_weight() {
        let dir =
            std::env::temp_dir().join(format!("rsinfer-q8-sidecar-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("weight.q8bin");
        let source = [
            f16::from_f32(0.5),
            f16::from_f32(-1.0),
            f16::from_f32(1.5),
            f16::from_f32(0.25),
        ];
        let weight = Q8LinearWeight::from_f16(&source, 2, 2).unwrap();

        write_q8_weight_atomic(&path, &weight).unwrap();
        let loaded = read_q8_weight(&path, 2, 2).unwrap();
        assert_eq!(loaded.qweight, weight.qweight);
        assert_eq!(loaded.scales, weight.scales);

        let _ = fs::remove_dir_all(&dir);
    }
}

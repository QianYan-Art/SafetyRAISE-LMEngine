//! CLI 与 HTTP 共用的模型装载、模板、媒体交接和输出协议。

use crate::artifact::WeightStore;
use crate::backend::{Backend, NativeBackend, WgpuQ8Backend};
use crate::engine::scheduler::{Engine, EngineConfig, GenerationHandle, GenerationRequest, Prompt};
use crate::frontend::{
    ChatFrontend, ChatMessage, ChatRequest, ChatRole, ContentPart, ImageInput, MessageContent,
    PreparedPrompt, SamplingParams, Tool, ToolCall,
};
use crate::model::{native::NativeModel, spec::ModelSpec};
use crate::vision::{VisionLimits, VisionTower, VisualEmbedding};
use crate::{Result, RsinferError};
use base64::Engine as _;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

fn bad(message: impl Into<String>) -> RsinferError {
    RsinferError::ConfigError(message.into())
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ServiceConfig {
    pub model_path: PathBuf,
    pub tokenizer_path: Option<PathBuf>,
    pub chat_template_file: Option<PathBuf>,
    #[serde(default = "default_name")]
    pub model_name: String,
    #[serde(default)]
    pub engine: EngineConfig,
    /// cpu、native 或 wgpu-q8；插件路径独立配置，不从请求接收。
    #[serde(default = "default_backend")]
    pub backend: String,
    pub native_library: Option<PathBuf>,
    #[serde(default = "default_gpu_budget")]
    pub gpu_memory_bytes: usize,
    #[serde(default = "default_body")]
    pub max_body_bytes: usize,
    #[serde(default = "default_output")]
    pub max_output_bytes: usize,
    #[serde(default = "default_vision_tokens")]
    pub vision_max_tokens: usize,
    /// 包括 JSON/模板/图像准备的并发，不与模型活动槽位混用。
    #[serde(default = "default_preparing")]
    pub max_preparing: usize,
    /// 所有并发准备请求的保守临时内存预算，与模型工作区分开计算。
    #[serde(default = "default_preparation_memory")]
    pub preparation_memory_bytes: usize,
}
fn default_name() -> String {
    "safetyraise-local".into()
}
fn default_backend() -> String {
    "cpu".into()
}
fn default_gpu_budget() -> usize {
    1024 * 1024 * 1024
}
fn default_body() -> usize {
    8 * 1024 * 1024
}
fn default_output() -> usize {
    1024 * 1024
}
fn default_vision_tokens() -> usize {
    256
}
fn default_preparing() -> usize {
    1
}
fn default_preparation_memory() -> usize {
    512 * 1024 * 1024
}

pub struct Service {
    pub engine: Engine,
    pub frontend: ChatFrontend,
    pub vision: Option<VisionTower>,
    pub config: ServiceConfig,
}

pub fn open_weights(path: &Path) -> Result<(WeightStore, ModelSpec)> {
    let store = WeightStore::open(path)?;
    if let Some(config) = store.hf_config() {
        let spec = ModelSpec::from_hf(config)?;
        Ok((store, spec))
    } else {
        crate::model::gguf::normalize(&store)
    }
}

impl Service {
    pub fn load(config: ServiceConfig) -> Result<Self> {
        if config.model_name.is_empty()
            || config.model_name.len() > 128
            || config.max_body_bytes == 0
            || config.max_output_bytes == 0
            || config.vision_max_tokens == 0
            || config.vision_max_tokens > 4096
            || config.max_preparing == 0
            || config.max_preparing > 16
            || config.max_body_bytes > 64 * 1024 * 1024
            || config
                .max_body_bytes
                .checked_mul(8)
                .and_then(|n| n.checked_mul(config.max_preparing))
                .is_none_or(|n| n > config.preparation_memory_bytes)
        {
            return Err(bad("服务名称、字节上限或准备内存预算非法"));
        }
        let backend = match config.backend.as_str() {
            "cpu" => Backend::Cpu,
            "native" => Backend::Native(Arc::new(NativeBackend::load(
                config
                    .native_library
                    .as_deref()
                    .ok_or_else(|| bad("native 后端需要 native_library"))?,
            )?)),
            "wgpu-q8" => Backend::WgpuQ8(Arc::new(WgpuQ8Backend::new(config.gpu_memory_bytes)?)),
            _ => return Err(bad("backend 仅支持 cpu/native/wgpu-q8")),
        };
        let (weights, spec) = open_weights(&config.model_path)?;
        let chat_template = if let Some(path) = &config.chat_template_file {
            if std::fs::metadata(path)?.len() > 1024 * 1024 {
                return Err(bad("外部 chat_template 超过 1MiB"));
            }
            Some(std::fs::read_to_string(path)?)
        } else {
            None
        };
        let frontend = ChatFrontend::load_with_options(
            config
                .tokenizer_path
                .as_deref()
                .unwrap_or(&config.model_path),
            crate::frontend::FrontendLoadOptions {
                allow_plain_fallback: false,
                chat_template,
            },
        )?;
        let vision = spec
            .vision
            .as_ref()
            .map(|v| VisionTower::from_weights(&weights, v, backend.clone()))
            .transpose()?;
        let model = Arc::new(NativeModel::from_weights(&weights, spec, backend)?);
        let mtp = crate::model::mtp::MtpModel::from_weights(&weights, &model)?.map(Arc::new);
        let engine = Engine::with_mtp(model, config.engine.clone(), mtp)?;
        Ok(Self {
            engine,
            frontend,
            vision,
            config,
        })
    }
    pub fn prepare(&self, input: &ApiRequest) -> Result<PreparedGeneration> {
        if input.model != self.config.model_name {
            return Err(bad("请求 model 与已加载模型不匹配"));
        }
        if input.n != 1 {
            return Err(bad("当前每个请求只支持 n=1"));
        }
        if input.messages.is_empty() || input.messages.len() > 4096 || input.tools.len() > 128 {
            return Err(bad("消息或工具数量超出范围"));
        }
        let (request, images) = input.to_frontend()?;
        let prepared = self.frontend.prepare(&request)?;
        if prepared
            .tokens
            .len()
            .checked_add(prepared.max_tokens)
            .is_none_or(|n| n > self.config.engine.max_context)
        {
            return Err(bad("输入和输出预算超过上下文窗口"));
        }
        let prompt = if images.is_empty() {
            Prompt::text(prepared.tokens.clone())
        } else {
            let tower = self
                .vision
                .as_ref()
                .ok_or_else(|| bad("当前模型没有已加载的视觉塔"))?;
            let limits = self.media_limits(images.len(), prepared.tokens.len())?;
            let visual = images
                .iter()
                .map(|bytes| tower.encode_bytes(bytes, &limits))
                .collect::<Result<Vec<_>>>()?;
            merge_images(
                self.engine.model(),
                &prepared.tokens,
                &visual,
                self.config.engine.max_context - prepared.max_tokens,
            )?
        };
        let generation = GenerationRequest {
            prompt,
            max_new_tokens: prepared.max_tokens,
            sampling: request.sampling,
            deadline: input.timeout_ms.map(Duration::from_millis),
            cache_prefix: input.cache_prefix,
            thinking: Some(prepared.thinking.clone()),
        };
        Ok(PreparedGeneration {
            prepared,
            generation,
            tool_names: request.tools.into_iter().map(|t| t.name).collect(),
        })
    }
    fn media_limits(&self, image_count: usize, text_tokens: usize) -> Result<VisionLimits> {
        let spec = &self.engine.model().spec;
        let vision = spec.vision.as_ref().ok_or_else(|| bad("缺少视觉配置"))?;
        let overflow = || bad("媒体准备预算溢出");
        let patches = self
            .config
            .vision_max_tokens
            .checked_mul(vision.spatial_merge_size)
            .and_then(|n| n.checked_mul(vision.spatial_merge_size))
            .ok_or_else(overflow)?
            .min((128_000_000usize / vision.num_heads).isqrt());
        let pixels = 4 * 1024 * 1024usize;
        let tokens = text_tokens
            .checked_add(
                image_count
                    .checked_mul(self.config.vision_max_tokens)
                    .ok_or_else(overflow)?,
            )
            .ok_or_else(overflow)?;
        let prompt_bytes = tokens
            .checked_mul(
                spec.hidden_size
                    .checked_mul(4)
                    .and_then(|n| n.checked_add(32))
                    .ok_or_else(overflow)?,
            )
            .ok_or_else(overflow)?;
        if prompt_bytes > self.config.engine.max_input_bytes {
            return Err(RsinferError::ResourceExhausted(
                "视觉嵌入的预留大小超过 max_input_bytes".into(),
            ));
        }
        let patch_width = vision
            .patch_size
            .checked_mul(vision.patch_size)
            .and_then(|n| n.checked_mul(vision.temporal_patch_size))
            .and_then(|n| n.checked_mul(3))
            .ok_or_else(overflow)?;
        let width = vision
            .hidden_size
            .checked_mul(12)
            .and_then(|n| n.checked_add(vision.intermediate_size.checked_mul(3)?))
            .and_then(|n| n.checked_add(patch_width))
            .ok_or_else(overflow)?;
        let estimate = patches
            .checked_mul(width)
            .and_then(|n| n.checked_mul(4))
            .and_then(|n| n.checked_add(pixels.checked_mul(24)?))
            .and_then(|n| n.checked_add(prompt_bytes.checked_mul(2)?))
            .and_then(|n| n.checked_add(self.config.max_body_bytes.checked_mul(8)?))
            .ok_or_else(overflow)?;
        if estimate > self.config.preparation_memory_bytes / self.config.max_preparing {
            return Err(RsinferError::ResourceExhausted(format!(
                "单请求媒体准备估算 {estimate} 字节超过准备预算"
            )));
        }
        Ok(VisionLimits {
            max_pixels: pixels,
            max_patches: patches,
            max_tokens: self.config.vision_max_tokens,
            max_bytes: self.config.max_body_bytes,
        })
    }
    pub fn submit(
        &self,
        prepared: PreparedGeneration,
    ) -> Result<(PreparedPrompt, GenerationHandle, Vec<String>)> {
        let handle = self.engine.submit(prepared.generation)?;
        Ok((prepared.prepared, handle, prepared.tool_names))
    }
    pub fn capabilities(&self) -> Value {
        json!({"model":self.config.model_name,"family":self.engine.model().spec.family,"backend":self.engine.model().backend.name(),
            "max_context":self.config.engine.max_context,"image":self.vision.is_some(),"video":false,"mtp_weights":self.engine.has_mtp(),"mtp_max_draft":self.config.engine.max_draft_tokens,
            "templates":self.frontend.template_names().collect::<Vec<_>>(),"scheduling":"chunked_prefill_round_robin_decode","constrained_json_decoding":false})
    }
}

pub struct PreparedGeneration {
    pub prepared: PreparedPrompt,
    pub generation: GenerationRequest,
    pub tool_names: Vec<String>,
}

/// 只接受这一明确子集；未实现字段不会被静默忽略。
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ApiRequest {
    pub model: String,
    pub messages: Vec<ApiMessage>,
    #[serde(default)]
    pub tools: Vec<ApiTool>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default = "one")]
    pub n: usize,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    pub repetition_penalty: Option<f32>,
    pub seed: Option<u64>,
    pub stop: Option<StopStrings>,
    pub enable_thinking: Option<bool>,
    pub thinking_budget: Option<usize>,
    pub template_name: Option<String>,
    pub timeout_ms: Option<u64>,
    #[serde(default)]
    pub cache_prefix: bool,
}
fn one() -> usize {
    1
}
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum StopStrings {
    One(String),
    Many(Vec<String>),
}
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ApiMessage {
    pub role: ChatRole,
    #[serde(default)]
    pub content: Value,
    pub name: Option<String>,
    pub tool_call_id: Option<String>,
    #[serde(default)]
    pub tool_calls: Vec<ApiToolCall>,
}
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ApiTool {
    #[serde(rename = "type")]
    pub kind: String,
    pub function: Tool,
}
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ApiToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub function: ApiFunctionCall,
}
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ApiFunctionCall {
    pub name: String,
    pub arguments: String,
}

impl ApiRequest {
    pub fn to_frontend(&self) -> Result<(ChatRequest, Vec<Vec<u8>>)> {
        let mut images = vec![];
        let mut messages = vec![];
        for message in &self.messages {
            let content = match &message.content {
                Value::String(s) => MessageContent::Text(s.clone()),
                Value::Null if message.role == ChatRole::Assistant => {
                    MessageContent::Text(String::new())
                }
                Value::Array(parts) => {
                    let mut content = vec![];
                    for part in parts {
                        match part.get("type").and_then(Value::as_str) {
                            Some("text") => content.push(ContentPart::Text(
                                part.get("text")
                                    .and_then(Value::as_str)
                                    .ok_or_else(|| bad("text 段缺少字符串 text"))?
                                    .to_string(),
                            )),
                            Some("image_url") => {
                                if images.len() >= 4 {
                                    return Err(bad("每个请求最多 4 张图像"));
                                }
                                let url = part
                                    .pointer("/image_url/url")
                                    .and_then(Value::as_str)
                                    .ok_or_else(|| bad("image_url 段缺少 url"))?;
                                let (kind, data) = url
                                    .split_once(',')
                                    .ok_or_else(|| bad("图像仅接受 PNG/JPEG base64 data URL"))?;
                                let media = match kind {
                                    "data:image/png;base64" => "image/png",
                                    "data:image/jpeg;base64" => "image/jpeg",
                                    _ => return Err(bad("图像仅接受 PNG/JPEG base64 data URL")),
                                };
                                if data.len() > 16 * 1024 * 1024 {
                                    return Err(bad("图像编码字节超过上限"));
                                }
                                let bytes = base64::engine::general_purpose::STANDARD
                                    .decode(data)
                                    .map_err(|_| bad("图像 base64 无效"))?;
                                content.push(ContentPart::Image(ImageInput::bytes(
                                    vec![],
                                    Some(media.into()),
                                )));
                                images.push(bytes);
                            }
                            _ => return Err(bad("消息段仅支持 text/image_url")),
                        }
                    }
                    MessageContent::Parts(content)
                }
                _ => return Err(bad("content 必须是字符串或消息段数组")),
            };
            let mut result = ChatMessage::new(message.role, content);
            result.name = message.name.clone();
            result.tool_call_id = message.tool_call_id.clone();
            result.tool_calls = message
                .tool_calls
                .iter()
                .map(|call| {
                    if call.kind != "function" {
                        return Err(bad("tool_calls 仅支持 function"));
                    }
                    let arguments: Value = serde_json::from_str(&call.function.arguments)?;
                    if !arguments.is_object() {
                        return Err(bad("tool arguments 必须是 JSON 对象"));
                    }
                    Ok(ToolCall {
                        id: Some(call.id.clone()),
                        name: call.function.name.clone(),
                        arguments,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            messages.push(result);
        }
        let tools = self
            .tools
            .iter()
            .map(|tool| {
                if tool.kind == "function" {
                    Ok(tool.function.clone())
                } else {
                    Err(bad("tools 仅支持 function"))
                }
            })
            .collect::<Result<Vec<_>>>()?;
        let mut sampling = SamplingParams::default();
        if let Some(v) = self.temperature {
            sampling.temperature = v;
        }
        if let Some(v) = self.top_p {
            sampling.top_p = v;
        }
        if let Some(v) = self.top_k {
            sampling.top_k = v;
        }
        if let Some(v) = self.repetition_penalty {
            sampling.repetition_penalty = v;
        }
        sampling.seed = self.seed;
        if let Some(n) = self.max_tokens {
            sampling.max_new_tokens = n;
        }
        Ok((
            ChatRequest {
                messages,
                tools,
                template_name: self.template_name.clone(),
                add_generation_prompt: true,
                enable_thinking: self.enable_thinking,
                thinking_budget: self.thinking_budget,
                stop: match &self.stop {
                    None => vec![],
                    Some(StopStrings::One(s)) => vec![s.clone()],
                    Some(StopStrings::Many(s)) => s.clone(),
                },
                max_tokens: self.max_tokens,
                sampling,
            },
            images,
        ))
    }
}

fn merge_images(
    model: &NativeModel,
    tokens: &[u32],
    images: &[VisualEmbedding],
    max_tokens: usize,
) -> Result<Prompt> {
    let image_id = model
        .spec
        .image_token_id
        .ok_or_else(|| bad("模型未声明 image_token_id"))?;
    if tokens.iter().filter(|id| **id == image_id).count() != images.len() {
        return Err(bad("模板图像占位符与媒体数量不匹配"));
    }
    let mut expanded = Vec::new();
    let mut values = Vec::new();
    let mut positions = Vec::new();
    let mut base = 0usize;
    let mut ordinal = 0;
    for token in tokens {
        if *token == image_id {
            let image = &images[ordinal];
            ordinal += 1;
            let [t, h, w] = image.grid;
            let count = t
                .checked_mul(h)
                .and_then(|n| n.checked_mul(w))
                .ok_or_else(|| bad("视觉 grid 溢出"))?;
            if count == 0
                || count.checked_mul(model.spec.hidden_size) != Some(image.values.len())
                || image.hidden_size != model.spec.hidden_size
                || expanded
                    .len()
                    .checked_add(count)
                    .is_none_or(|n| n > max_tokens)
            {
                return Err(bad("视觉交接维度或上下文预算非法"));
            }
            for ti in 0..t {
                for hi in 0..h {
                    for wi in 0..w {
                        positions.push([base + ti, base + hi, base + wi]);
                        expanded.push(image_id);
                    }
                }
            }
            values.extend_from_slice(&image.values);
            base = base
                .checked_add(t.max(h).max(w))
                .ok_or_else(|| bad("视觉位置溢出"))?;
        } else {
            if expanded.len() == max_tokens {
                return Err(bad("多模态输入超过上下文预算"));
            }
            values.extend(model.embed(&[*token])?);
            expanded.push(*token);
            positions.push([base; 3]);
            base += 1;
        }
    }
    Ok(Prompt {
        tokens: expanded,
        embeddings: Some((values, positions)),
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    pub(crate) fn fixture() -> Service {
        let (weights, spec) =
            crate::model::native::fixtures::tiny(crate::model::spec::ModelFamily::Qwen35);
        let model = Arc::new(NativeModel::from_weights(&weights, spec, Backend::Cpu).unwrap());
        let mut config: ServiceConfig =
            serde_json::from_value(json!({"model_path":"fixture","model_name":"tiny"})).unwrap();
        config.engine.max_context = 128;
        config.engine.prefill_chunk = 2;
        config.engine.event_capacity = 1;
        config.engine.max_input_bytes = 1024 * 1024;
        config.engine.working_memory_bytes = 64 * 1024 * 1024;
        let engine = Engine::new(model, config.engine.clone()).unwrap();
        Service {
            engine,
            frontend: crate::frontend::fixtures::frontend(),
            vision: None,
            config,
        }
    }
    #[test]
    fn image_handoff_preserves_merge_order_and_axis_positions() {
        let (weights, mut spec) =
            crate::model::native::fixtures::tiny(crate::model::spec::ModelFamily::Qwen35);
        spec.image_token_id = Some(7);
        let model = NativeModel::from_weights(&weights, spec, Backend::Cpu).unwrap();
        let image = VisualEmbedding {
            values: (0..24).map(|i| i as f32 / 24.0).collect(),
            grid: [1, 2, 3],
            hidden_size: 4,
        };
        let prompt = merge_images(&model, &[1, 7, 2], std::slice::from_ref(&image), 16).unwrap();
        assert_eq!(prompt.tokens, vec![1, 7, 7, 7, 7, 7, 7, 2]);
        let (values, positions) = prompt.embeddings.unwrap();
        assert_eq!(&values[4..28], image.values.as_slice());
        assert_eq!(
            positions,
            vec![
                [0, 0, 0],
                [1, 1, 1],
                [1, 1, 2],
                [1, 1, 3],
                [1, 2, 1],
                [1, 2, 2],
                [1, 2, 3],
                [4, 4, 4]
            ]
        );
        let mut whole = model.create_state(16).unwrap();
        let expected = model
            .forward_embeddings(&values, &positions, &mut whole, true)
            .unwrap();
        let mut state = model.create_state(16).unwrap();
        let mut logits = vec![];
        for (chunk, pos) in values.chunks(12).zip(positions.chunks(3)) {
            logits.extend(
                model
                    .forward_embeddings(chunk, pos, &mut state, true)
                    .unwrap()
                    .logits,
            );
        }
        for (a, b) in logits.iter().zip(expected.logits) {
            assert!((a - b).abs() < 1e-5);
        }
        assert_eq!(state.next_position(), 5);
        assert!(merge_images(&model, &[1, 7, 2], &[image], 7).is_err());
    }

    #[test]
    fn media_budget_rejects_before_encoding_and_bounds_patch_work() {
        let (weights, mut spec) =
            crate::model::native::fixtures::tiny(crate::model::spec::ModelFamily::Qwen35);
        spec.vision = Some(crate::model::spec::VisionSpec {
            depth: 1,
            hidden_size: 4,
            intermediate_size: 8,
            num_heads: 1,
            in_channels: 3,
            patch_size: 2,
            spatial_merge_size: 2,
            temporal_patch_size: 2,
            out_hidden_size: 4,
            num_position_embeddings: 16,
        });
        spec.image_token_id = Some(7);
        let model = Arc::new(NativeModel::from_weights(&weights, spec, Backend::Cpu).unwrap());
        let mut service = fixture();
        service.engine = Engine::new(model, service.config.engine.clone()).unwrap();
        let limits = service.media_limits(4, 2).unwrap();
        assert!(limits.max_patches * limits.max_patches <= 128_000_000);
        service.config.preparation_memory_bytes = 1;
        assert!(matches!(
            service.media_limits(1, 2),
            Err(RsinferError::ResourceExhausted(_))
        ));
        service.config.preparation_memory_bytes = 512 * 1024 * 1024;
        service.config.engine.max_input_bytes = 1;
        assert!(matches!(
            service.media_limits(1, 2),
            Err(RsinferError::ResourceExhausted(_))
        ));
    }

    #[test]
    fn api_tool_contract_rejects_malformed_definitions_and_calls() {
        let service = fixture();
        let empty_tool_name: ApiRequest = serde_json::from_value(json!({
            "model": "tiny",
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [{
                "type": "function",
                "function": {"name": "", "parameters": {}}
            }]
        }))
        .unwrap();
        assert!(service.prepare(&empty_tool_name).is_err());

        let non_object_parameters: ApiRequest = serde_json::from_value(json!({
            "model": "tiny",
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [{
                "type": "function",
                "function": {"name": "lookup", "parameters": []}
            }]
        }))
        .unwrap();
        assert!(service.prepare(&non_object_parameters).is_err());

        let unknown_tool_field: std::result::Result<ApiRequest, _> =
            serde_json::from_value(json!({
                "model": "tiny",
                "messages": [{"role": "user", "content": "hello"}],
                "tools": [{
                    "type": "function",
                    "function": {"name": "lookup", "parameters": {}, "strict": true}
                }]
            }));
        assert!(unknown_tool_field.is_err());

        let empty_tool_call_name: ApiRequest = serde_json::from_value(json!({
            "model": "tiny",
            "messages": [
                {"role": "user", "content": "hello"},
                {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "", "arguments": "{}"}
                    }]
                }
            ]
        }))
        .unwrap();
        assert!(service.prepare(&empty_tool_call_name).is_err());

        let non_object_tool_call_arguments: ApiRequest = serde_json::from_value(json!({
            "model": "tiny",
            "messages": [
                {"role": "user", "content": "hello"},
                {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "[]"}
                    }]
                }
            ]
        }))
        .unwrap();
        assert!(service.prepare(&non_object_tool_call_arguments).is_err());

        let empty_tool_call_id: ApiRequest = serde_json::from_value(json!({
            "model": "tiny",
            "messages": [
                {"role": "user", "content": "hello"},
                {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"}
                    }]
                }
            ]
        }))
        .unwrap();
        assert!(service.prepare(&empty_tool_call_id).is_err());
    }
}

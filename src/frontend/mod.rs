//! Hugging Face tokenizer/chat-template 前端。
//!
//! 本模块只负责消息契约、模板渲染、token 编解码、输出分流和采样参数。模型
//! 执行器负责把 `PreparedPrompt.tokens` 送入模型；本模块不会下载图片、调用
//! HTTP，也不会根据业务含义伪造工具结果。

mod decoder;
mod sampling;

pub use decoder::{DecodedChunk, OutputEvent, OutputMarkers, OutputSplitter, TokenDecoder};
pub use sampling::{Sampler, SamplingParams};

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use minijinja::{Environment, Error as MiniError, ErrorKind as MiniErrorKind, State};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tokenizers::Tokenizer;

use crate::{Result, RsinferError};

const MAX_TOKENIZER_CONFIG_BYTES: usize = 64 * 1024 * 1024;
const MAX_RENDERED_PROMPT_BYTES: usize = 64 * 1024 * 1024;

/// 消息角色；模板会收到对应的小写字符串。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatRole {
    System,
    Developer,
    User,
    Assistant,
    Tool,
}

impl ChatRole {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::Developer => "developer",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
        }
    }
}

/// 图片来源只保留结构化描述；前端不会读取任意 HTTP URL 或本地路径。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ImageSource {
    Bytes(Vec<u8>),
    Path(PathBuf),
    Url(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageInput {
    pub source: ImageSource,
    pub media_type: Option<String>,
    pub alt_text: Option<String>,
}

impl ImageInput {
    pub fn bytes(bytes: Vec<u8>, media_type: Option<String>) -> Self {
        Self {
            source: ImageSource::Bytes(bytes),
            media_type,
            alt_text: None,
        }
    }

    pub fn path(path: PathBuf, media_type: Option<String>) -> Self {
        Self {
            source: ImageSource::Path(path),
            media_type,
            alt_text: None,
        }
    }

    pub fn url(url: String, media_type: Option<String>) -> Self {
        Self {
            source: ImageSource::Url(url),
            media_type,
            alt_text: None,
        }
    }
}

/// 多模态消息段。实际视觉编码由模型前端负责，本模块仅把段传给模板。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text(String),
    Image(ImageInput),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

impl From<String> for MessageContent {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

impl From<&str> for MessageContent {
    fn from(value: &str) -> Self {
        Self::Text(value.to_owned())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: Option<String>,
    pub name: String,
    pub arguments: Value,
}

impl ToolCall {
    pub fn validate(&self) -> Result<()> {
        if self.name.is_empty() {
            return Err(RsinferError::ConfigError("tool call name 不能为空".into()));
        }
        if self.id.as_deref().is_some_and(str::is_empty) {
            return Err(RsinferError::ConfigError("tool call id 不能为空".into()));
        }
        if !self.arguments.is_object() {
            return Err(RsinferError::ConfigError(
                "tool call arguments 必须是 JSON object".into(),
            ));
        }
        Ok(())
    }
}

/// 一条对话消息。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: MessageContent,
    pub name: Option<String>,
    pub tool_call_id: Option<String>,
    pub tool_calls: Vec<ToolCall>,
}

impl ChatMessage {
    pub fn new(role: ChatRole, content: impl Into<MessageContent>) -> Self {
        Self {
            role,
            content: content.into(),
            name: None,
            tool_call_id: None,
            tool_calls: Vec::new(),
        }
    }

    pub fn system(content: impl Into<MessageContent>) -> Self {
        Self::new(ChatRole::System, content)
    }

    pub fn user(content: impl Into<MessageContent>) -> Self {
        Self::new(ChatRole::User, content)
    }

    pub fn assistant(content: impl Into<MessageContent>) -> Self {
        Self::new(ChatRole::Assistant, content)
    }

    pub fn tool(content: impl Into<MessageContent>, tool_call_id: impl Into<String>) -> Self {
        let mut message = Self::new(ChatRole::Tool, content);
        message.tool_call_id = Some(tool_call_id.into());
        message
    }
}

/// 传给 chat template 的 function 工具定义。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tool {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Value,
}

impl Tool {
    pub fn validate(&self) -> Result<()> {
        if self.name.is_empty() {
            return Err(RsinferError::ConfigError("tool name 不能为空".into()));
        }
        if !self.parameters.is_object() {
            return Err(RsinferError::ConfigError(
                "tool parameters 必须是 JSON object".into(),
            ));
        }
        Ok(())
    }

    pub fn function(
        name: impl Into<String>,
        description: Option<String>,
        parameters: Value,
    ) -> Result<Self> {
        let tool = Self {
            name: name.into(),
            description,
            parameters,
        };
        tool.validate()?;
        Ok(tool)
    }
}

/// 一次模板渲染请求。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ChatRequest {
    pub messages: Vec<ChatMessage>,
    pub tools: Vec<Tool>,
    pub template_name: Option<String>,
    pub add_generation_prompt: bool,
    pub enable_thinking: Option<bool>,
    pub thinking_budget: Option<usize>,
    pub stop: Vec<String>,
    /// OpenAI 风格请求的输出 token 上限；未设置时由 sampling.max_new_tokens 控制。
    pub max_tokens: Option<usize>,
    pub sampling: SamplingParams,
}

impl Default for ChatRequest {
    fn default() -> Self {
        Self {
            messages: Vec::new(),
            tools: Vec::new(),
            template_name: None,
            add_generation_prompt: true,
            enable_thinking: None,
            thinking_budget: None,
            stop: Vec::new(),
            max_tokens: None,
            sampling: SamplingParams::default(),
        }
    }
}

impl ChatRequest {
    pub fn validate(&self) -> Result<()> {
        if self.messages.is_empty() {
            return Err(RsinferError::ConfigError(
                "ChatRequest.messages 不能为空".into(),
            ));
        }
        if self.stop.iter().any(String::is_empty) {
            return Err(RsinferError::ConfigError("stop 字符串不能为空".into()));
        }
        if self.max_tokens == Some(0) {
            return Err(RsinferError::ConfigError("max_tokens 必须大于零".into()));
        }
        if self.enable_thinking == Some(false) && self.thinking_budget.is_some() {
            return Err(RsinferError::ConfigError(
                "禁用 thinking 时不能设置 thinking_budget".into(),
            ));
        }
        for tool in &self.tools {
            tool.validate()?;
        }
        for message in &self.messages {
            for call in &message.tool_calls {
                call.validate()?;
            }
        }
        self.sampling.validate()
    }
}

/// 允许用户显式选择无 chat_template 时的纯文本 fallback。
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct FrontendLoadOptions {
    pub allow_plain_fallback: bool,
    /// 外部提供的完整 Jinja 模板；设置后覆盖 tokenizer_config 中的同名字段。
    pub chat_template: Option<String>,
}

/// 输出模板和生成控制信息。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TemplateInfo {
    pub name: Option<String>,
    pub plain_fallback: bool,
    pub output_markers: OutputMarkers,
}

/// thinking 控制器所需的模板确认 token 序列。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ThinkingSpec {
    pub supported: bool,
    pub enabled: bool,
    pub budget: Option<usize>,
    pub start_tokens: Vec<u32>,
    pub end_tokens: Vec<u32>,
    /// 模板已经在 prompt 末尾打开 thinking 时，生成从该状态开始。
    pub initially_in_thinking: bool,
    pub disabled_reason: Option<String>,
}

impl ThinkingSpec {
    fn disabled(reason: impl Into<String>) -> Self {
        Self {
            supported: false,
            enabled: false,
            budget: None,
            start_tokens: Vec::new(),
            end_tokens: Vec::new(),
            initially_in_thinking: false,
            disabled_reason: Some(reason.into()),
        }
    }
}

/// 生成循环每接收一个 token 后调用的 thinking 控制器。
pub struct ThinkingController {
    spec: ThinkingSpec,
    in_thinking: bool,
    used: usize,
    forced: bool,
    recent: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ThinkingAction {
    Disabled,
    Continue,
    Entered,
    Exited,
    ForceClose { end_tokens: Vec<u32> },
}

impl ThinkingController {
    pub fn new(spec: ThinkingSpec) -> Self {
        let in_thinking = spec.supported && spec.enabled && spec.initially_in_thinking;
        Self {
            spec,
            in_thinking,
            used: 0,
            forced: false,
            recent: Vec::new(),
        }
    }

    pub fn spec(&self) -> &ThinkingSpec {
        &self.spec
    }

    pub fn in_thinking(&self) -> bool {
        self.in_thinking
    }

    pub fn used_tokens(&self) -> usize {
        self.used
    }

    /// 在生成下一个 token 前检查 thinking 预算。
    ///
    /// 当预算为零，或上一个 token 已经达到预算但调用方尚未注入结束标记时，
    /// 返回模板声明的 end marker。返回一次后会保持强制关闭状态，避免重复注入。
    pub fn before_step(&mut self) -> ThinkingAction {
        if !self.spec.supported || !self.spec.enabled {
            return ThinkingAction::Disabled;
        }
        if self.in_thinking
            && !self.forced
            && self.spec.budget.is_some_and(|budget| self.used >= budget)
        {
            self.forced = true;
            return ThinkingAction::ForceClose {
                end_tokens: self.spec.end_tokens.clone(),
            };
        }
        ThinkingAction::Continue
    }

    /// 方便生成器在下一步采样前直接取得强制闭合 token。
    pub fn take_force_close(&mut self) -> Option<Vec<u32>> {
        match self.before_step() {
            ThinkingAction::ForceClose { end_tokens } => Some(end_tokens),
            _ => None,
        }
    }

    /// 返回的 ForceClose token 序列直接来自 tokenizer 中声明的 end marker。
    pub fn observe(&mut self, token_id: u32) -> ThinkingAction {
        if !self.spec.supported || !self.spec.enabled {
            return ThinkingAction::Disabled;
        }
        let max_len = self.spec.start_tokens.len().max(self.spec.end_tokens.len());
        self.recent.push(token_id);
        if self.recent.len() > max_len {
            let drop_count = self.recent.len() - max_len;
            self.recent.drain(..drop_count);
        }
        if !self.in_thinking && ends_with(&self.recent, &self.spec.start_tokens) {
            self.in_thinking = true;
            self.used = 0;
            self.forced = false;
            return ThinkingAction::Entered;
        }
        if self.in_thinking && ends_with(&self.recent, &self.spec.end_tokens) {
            self.in_thinking = false;
            return ThinkingAction::Exited;
        }
        if self.in_thinking {
            self.used = self.used.saturating_add(1);
            if !self.forced && self.spec.budget.is_some_and(|budget| self.used >= budget) {
                self.forced = true;
                return ThinkingAction::ForceClose {
                    end_tokens: self.spec.end_tokens.clone(),
                };
            }
        }
        ThinkingAction::Continue
    }
}

fn ends_with(sequence: &[u32], suffix: &[u32]) -> bool {
    !suffix.is_empty()
        && sequence.len() >= suffix.len()
        && sequence[sequence.len() - suffix.len()..] == *suffix
}

/// 一次模板渲染后的模型输入。
#[derive(Debug, Clone)]
pub struct PreparedPrompt {
    pub tokens: Vec<u32>,
    pub rendered: String,
    pub template: TemplateInfo,
    pub thinking: ThinkingSpec,
    pub stop: Vec<String>,
    pub max_tokens: usize,
    pub image_placeholders: Vec<ImagePlaceholder>,
    tokenizer: Arc<Tokenizer>,
}

/// 图片段的有序占位符。视觉执行器可在 `tokens` 中定位 marker 后替换为 image tokens。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImagePlaceholder {
    pub ordinal: usize,
    pub marker: String,
    pub token_ids: Vec<u32>,
    pub media_type: Option<String>,
}

impl PreparedPrompt {
    pub fn token_decoder(&self, max_output_bytes: usize) -> Result<TokenDecoder> {
        TokenDecoder::with_options(
            Arc::clone(&self.tokenizer),
            self.stop.clone(),
            max_output_bytes,
            false,
        )
    }

    pub fn output_splitter(&self) -> Result<OutputSplitter> {
        OutputSplitter::new_with_initial_state(
            self.template.output_markers.clone(),
            self.thinking.initially_in_thinking,
        )
    }

    pub fn thinking_controller(&self) -> ThinkingController {
        ThinkingController::new(self.thinking.clone())
    }
}

/// 已加载 tokenizer 和 chat templates 的前端。
#[derive(Clone)]
pub struct ChatFrontend {
    model_dir: PathBuf,
    tokenizer: Arc<Tokenizer>,
    templates: BTreeMap<String, String>,
    tokenizer_config: Value,
    output_markers: OutputMarkers,
    plain_fallback: bool,
}

impl ChatFrontend {
    /// 严格加载：没有 chat_template 时返回明确错误。
    pub fn load(model_dir: &Path) -> Result<Self> {
        Self::load_with_options(model_dir, FrontendLoadOptions::default())
    }

    pub fn load_with_options(model_dir: &Path, options: FrontendLoadOptions) -> Result<Self> {
        if !model_dir.is_dir() {
            return Err(RsinferError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("模型目录不存在：{}", model_dir.display()),
            )));
        }
        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|error| RsinferError::TokenizerError(error.to_string()))?;
        let config_path = model_dir.join("tokenizer_config.json");
        let mut config = if config_path.is_file() {
            let bytes = fs::read(&config_path)?;
            if bytes.len() > MAX_TOKENIZER_CONFIG_BYTES {
                return Err(RsinferError::ConfigError(
                    "tokenizer_config.json 超过大小限制".into(),
                ));
            }
            serde_json::from_slice::<Value>(&bytes)?
        } else if options.allow_plain_fallback || options.chat_template.is_some() {
            Value::Object(Map::new())
        } else {
            return Err(RsinferError::Unsupported(
                "缺少 tokenizer_config.json；不能确认 chat_template，请显式启用 plain fallback"
                    .into(),
            ));
        };
        if let Some(template) = options.chat_template.as_deref() {
            if template.trim().is_empty() {
                return Err(RsinferError::ConfigError(
                    "外部 chat_template 不能为空".into(),
                ));
            }
            let object = config.as_object_mut().ok_or_else(|| {
                RsinferError::ConfigError("tokenizer_config 顶层必须是 object".into())
            })?;
            object.insert("chat_template".into(), Value::String(template.to_owned()));
        }
        let templates = parse_templates(&config)?;
        if templates.is_empty() && !options.allow_plain_fallback {
            return Err(RsinferError::Unsupported(
                "tokenizer_config.json 未声明 chat_template；请显式启用 plain fallback".into(),
            ));
        }
        let output_markers = resolve_output_markers(&config, &templates, &tokenizer)?;
        output_markers.validate()?;
        Ok(Self {
            model_dir: model_dir.to_path_buf(),
            tokenizer: Arc::new(tokenizer),
            templates,
            tokenizer_config: config,
            output_markers,
            plain_fallback: options.allow_plain_fallback,
        })
    }

    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    pub fn template_names(&self) -> impl Iterator<Item = &str> {
        self.templates.keys().map(String::as_str)
    }

    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(tokens, true)
            .map_err(|error| RsinferError::TokenizerError(error.to_string()))
    }

    pub fn prepare(&self, request: &ChatRequest) -> Result<PreparedPrompt> {
        request.validate()?;
        let (template_name, rendered, plain_fallback) = if self.templates.is_empty() {
            if !self.plain_fallback {
                return Err(RsinferError::Unsupported(
                    "当前前端没有可用 chat_template；请显式启用 plain fallback".into(),
                ));
            }
            (None, render_plain(request), true)
        } else {
            let name = select_template(&self.templates, request.template_name.as_deref())?;
            let context = request_context(request, &self.tokenizer_config);
            let mut environment = Environment::new();
            environment.set_unknown_method_callback(python_string_method);
            let template = self
                .templates
                .get(&name)
                .ok_or_else(|| RsinferError::ConfigError("模板选择状态失效".into()))?;
            // Transformers 的 chat-template 扩展用 generation/endgeneration 标记
            // 生成区域；Minijinja 没有该语义扩展，去掉标记本身即可保留其正文。
            let template = normalize_generation_blocks(template);
            let rendered = environment
                .render_str(&template, context)
                .map_err(|error| RsinferError::TokenizerError(error.to_string()))?;
            (Some(name), rendered, false)
        };
        if rendered.len() > MAX_RENDERED_PROMPT_BYTES {
            return Err(RsinferError::DimensionError(
                "渲染后的 prompt 超过大小限制".into(),
            ));
        }
        let encoding = self
            .tokenizer
            .encode(rendered.clone(), false)
            .map_err(|error| RsinferError::TokenizerError(error.to_string()))?;
        let thinking = self.thinking_spec(request, &rendered)?;
        let image_placeholders = self.image_placeholders(request)?;
        Ok(PreparedPrompt {
            tokens: encoding.get_ids().to_vec(),
            rendered,
            template: TemplateInfo {
                name: template_name,
                plain_fallback,
                output_markers: self.output_markers.clone(),
            },
            thinking,
            stop: request.stop.clone(),
            max_tokens: request
                .max_tokens
                .unwrap_or(request.sampling.max_new_tokens),
            image_placeholders,
            tokenizer: Arc::clone(&self.tokenizer),
        })
    }

    fn thinking_spec(&self, request: &ChatRequest, rendered: &str) -> Result<ThinkingSpec> {
        let Some(start) = self.output_markers.thinking_start.as_deref() else {
            if request.enable_thinking == Some(true) {
                return Err(RsinferError::Unsupported(
                    "请求启用 thinking，但 tokenizer_config 未声明 thinking 标记；thinking 已明确禁用".into(),
                ));
            }
            return Ok(ThinkingSpec::disabled(
                "tokenizer_config 未声明成对 thinking 标记",
            ));
        };
        let Some(end) = self.output_markers.thinking_end.as_deref() else {
            return Ok(ThinkingSpec::disabled("thinking 标记不成对"));
        };
        let start_tokens = encode_marker(&self.tokenizer, start)?;
        let end_tokens = encode_marker(&self.tokenizer, end)?;
        if start_tokens.is_empty() || end_tokens.is_empty() {
            return Ok(ThinkingSpec::disabled(
                "thinking 标记无法由 tokenizer 编码为 token 序列",
            ));
        }
        Ok(ThinkingSpec {
            supported: true,
            enabled: request.enable_thinking.unwrap_or(false),
            budget: request.thinking_budget,
            start_tokens,
            end_tokens,
            initially_in_thinking: request.enable_thinking.unwrap_or(false)
                && last_marker_is_start(rendered, start, end),
            disabled_reason: None,
        })
    }

    fn image_placeholders(&self, request: &ChatRequest) -> Result<Vec<ImagePlaceholder>> {
        let mut placeholders = Vec::new();
        for message in &request.messages {
            let MessageContent::Parts(parts) = &message.content else {
                continue;
            };
            for part in parts {
                let ContentPart::Image(image) = part else {
                    continue;
                };
                let ordinal = placeholders.len();
                let marker = format!("<|rsinfer_image_{ordinal}|>");
                let token_ids = encode_marker(&self.tokenizer, &marker)?;
                placeholders.push(ImagePlaceholder {
                    ordinal,
                    marker,
                    token_ids,
                    media_type: image.media_type.clone(),
                });
            }
        }
        Ok(placeholders)
    }
}

fn encode_marker(tokenizer: &Tokenizer, marker: &str) -> Result<Vec<u32>> {
    tokenizer
        .encode(marker.to_owned(), false)
        .map(|encoding| encoding.get_ids().to_vec())
        .map_err(|error| RsinferError::TokenizerError(error.to_string()))
}

fn parse_templates(config: &Value) -> Result<BTreeMap<String, String>> {
    let Some(value) = config.get("chat_template") else {
        return Ok(BTreeMap::new());
    };
    let mut templates = BTreeMap::new();
    match value {
        Value::String(template) => {
            templates.insert("default".into(), template.clone());
        }
        Value::Object(object) => {
            for (name, template) in object {
                let template = template.as_str().ok_or_else(|| {
                    RsinferError::ConfigError(format!("chat_template.{name} 必须是字符串"))
                })?;
                templates.insert(name.clone(), template.to_owned());
            }
        }
        Value::Array(entries) => {
            for entry in entries {
                let object = entry.as_object().ok_or_else(|| {
                    RsinferError::ConfigError("chat_template 数组项必须是 object".into())
                })?;
                let name = object
                    .get("name")
                    .and_then(Value::as_str)
                    .ok_or_else(|| RsinferError::ConfigError("chat_template 项缺少 name".into()))?;
                let template = object
                    .get("template")
                    .and_then(Value::as_str)
                    .ok_or_else(|| {
                        RsinferError::ConfigError(format!("chat_template.{name} 缺少 template"))
                    })?;
                if templates
                    .insert(name.to_owned(), template.to_owned())
                    .is_some()
                {
                    return Err(RsinferError::ConfigError(format!(
                        "chat_template 重复名称：{name}"
                    )));
                }
            }
        }
        _ => {
            return Err(RsinferError::ConfigError(
                "chat_template 必须是字符串、object 或数组".into(),
            ))
        }
    }
    Ok(templates)
}

fn select_template(
    templates: &BTreeMap<String, String>,
    requested: Option<&str>,
) -> Result<String> {
    if let Some(name) = requested {
        if templates.contains_key(name) {
            return Ok(name.to_owned());
        }
        return Err(RsinferError::ConfigError(format!(
            "请求的 chat_template 不存在：{name}"
        )));
    }
    if templates.len() == 1 {
        return Ok(templates.keys().next().cloned().unwrap());
    }
    if templates.contains_key("default") {
        return Ok("default".into());
    }
    Err(RsinferError::ConfigError(
        "tokenizer_config 包含多个 chat_template，但请求未指定 template_name".into(),
    ))
}

fn parse_output_markers(config: &Value) -> Result<OutputMarkers> {
    let value = config
        .get("output_markers")
        .or_else(|| config.get("generation_markers"));
    let Some(value) = value else {
        return Ok(OutputMarkers::default());
    };
    let object = value
        .as_object()
        .ok_or_else(|| RsinferError::ConfigError("output_markers 必须是 object".into()))?;
    let get = |name: &str| -> Result<Option<String>> {
        object
            .get(name)
            .map(|value| {
                value.as_str().map(str::to_owned).ok_or_else(|| {
                    RsinferError::ConfigError(format!("output_markers.{name} 必须是字符串"))
                })
            })
            .transpose()
    };
    Ok(OutputMarkers {
        thinking_start: get("thinking_start")?,
        thinking_end: get("thinking_end")?,
        tool_start: get("tool_start")?,
        tool_end: get("tool_end")?,
    })
}

/// 解析显式标记并在缺少配置时从真实模板中识别 Qwen 常用标记。
///
/// 显式 `output_markers`/`generation_markers` 始终优先；自动识别只使用模板
/// 源码中实际出现的成对字面量，且每个标记都必须通过 tokenizer 的 encode/decode
/// 往返校验，避免把普通文本误当成生成控制标记。
fn resolve_output_markers(
    config: &Value,
    templates: &BTreeMap<String, String>,
    tokenizer: &Tokenizer,
) -> Result<OutputMarkers> {
    let explicit = config
        .get("output_markers")
        .or_else(|| config.get("generation_markers"))
        .is_some();
    let mut markers = parse_output_markers(config)?;
    if explicit {
        markers.validate()?;
    }
    if !explicit {
        let has_marker = |start: &str, end: &str| {
            templates
                .values()
                .any(|template| template.contains(start) && template.contains(end))
        };
        if has_marker("<think>", "</think>") {
            markers.thinking_start = Some("<think>".into());
            markers.thinking_end = Some("</think>".into());
        }
        if has_marker("<tool_call>", "</tool_call>") {
            markers.tool_start = Some("<tool_call>".into());
            markers.tool_end = Some("</tool_call>".into());
        }
    }

    sanitize_marker_pair(
        tokenizer,
        &mut markers.thinking_start,
        &mut markers.thinking_end,
    );
    sanitize_marker_pair(tokenizer, &mut markers.tool_start, &mut markers.tool_end);
    Ok(markers)
}

fn sanitize_marker_pair(
    tokenizer: &Tokenizer,
    start: &mut Option<String>,
    end: &mut Option<String>,
) {
    let valid = match (start.as_deref(), end.as_deref()) {
        (Some(start), Some(end)) => {
            marker_round_trips(tokenizer, start) && marker_round_trips(tokenizer, end)
        }
        _ => false,
    };
    if !valid {
        *start = None;
        *end = None;
    }
}

fn marker_round_trips(tokenizer: &Tokenizer, marker: &str) -> bool {
    let Ok(ids) = tokenizer.encode(marker.to_owned(), false) else {
        return false;
    };
    !ids.get_ids().is_empty()
        && tokenizer
            .decode(ids.get_ids(), false)
            .map(|decoded| decoded == marker)
            .unwrap_or(false)
}

fn last_marker_is_start(rendered: &str, start: &str, end: &str) -> bool {
    let start_index = rendered.rfind(start);
    let end_index = rendered.rfind(end);
    let trailing_is_open = |index: usize| {
        rendered
            .get(index.saturating_add(start.len())..)
            .is_some_and(|tail| tail.trim().is_empty())
    };
    match (start_index, end_index) {
        (Some(start_index), Some(end_index)) => {
            start_index > end_index && trailing_is_open(start_index)
        }
        (Some(start_index), None) => trailing_is_open(start_index),
        _ => false,
    }
}

fn normalize_generation_blocks(template: &str) -> String {
    [
        "{%- generation %}",
        "{% generation %}",
        "{%- endgeneration %}",
        "{% endgeneration %}",
    ]
    .into_iter()
    .fold(template.to_owned(), |template, marker| {
        template.replace(marker, "")
    })
}

/// Qwen 官方模板依赖的少量 Python 字符串方法兼容层。
fn python_string_method(
    _state: &State,
    value: &minijinja::Value,
    method: &str,
    args: &[minijinja::Value],
) -> std::result::Result<minijinja::Value, MiniError> {
    let Some(subject) = value.as_str() else {
        return Err(MiniError::from(MiniErrorKind::UnknownMethod));
    };
    let argument = |index: usize| {
        args.get(index)
            .and_then(minijinja::Value::as_str)
            .ok_or_else(|| {
                MiniError::new(
                    MiniErrorKind::InvalidOperation,
                    format!("字符串方法 {method} 的参数 {index} 必须是字符串"),
                )
            })
    };
    let no_arguments = || {
        if args.is_empty() {
            Ok(())
        } else {
            Err(MiniError::new(
                MiniErrorKind::InvalidOperation,
                format!("字符串方法 {method} 不接受参数"),
            ))
        }
    };

    match method {
        "startswith" => {
            if args.len() != 1 {
                return Err(MiniError::new(
                    MiniErrorKind::InvalidOperation,
                    "startswith 需要一个字符串参数",
                ));
            }
            Ok(minijinja::Value::from(subject.starts_with(argument(0)?)))
        }
        "endswith" => {
            if args.len() != 1 {
                return Err(MiniError::new(
                    MiniErrorKind::InvalidOperation,
                    "endswith 需要一个字符串参数",
                ));
            }
            Ok(minijinja::Value::from(subject.ends_with(argument(0)?)))
        }
        "strip" | "lstrip" | "rstrip" => {
            if args.len() > 1 {
                return Err(MiniError::new(
                    MiniErrorKind::InvalidOperation,
                    format!("字符串方法 {method} 至多接受一个参数"),
                ));
            }
            let chars = args
                .first()
                .map(|value| {
                    value.as_str().ok_or_else(|| {
                        MiniError::new(
                            MiniErrorKind::InvalidOperation,
                            format!("字符串方法 {method} 的参数 0 必须是字符串"),
                        )
                    })
                })
                .transpose()?;
            let output = match (method, chars) {
                ("strip", None) => subject.trim().to_owned(),
                ("lstrip", None) => subject.trim_start().to_owned(),
                ("rstrip", None) => subject.trim_end().to_owned(),
                ("strip", Some(chars)) => subject
                    .trim_matches(|character| chars.contains(character))
                    .to_owned(),
                ("lstrip", Some(chars)) => subject
                    .trim_start_matches(|character| chars.contains(character))
                    .to_owned(),
                ("rstrip", Some(chars)) => subject
                    .trim_end_matches(|character| chars.contains(character))
                    .to_owned(),
                _ => String::new(),
            };
            Ok(minijinja::Value::from(output))
        }
        "lower" => {
            no_arguments()?;
            Ok(minijinja::Value::from(subject.to_lowercase()))
        }
        "upper" => {
            no_arguments()?;
            Ok(minijinja::Value::from(subject.to_uppercase()))
        }
        "replace" => {
            if !(2..=3).contains(&args.len()) {
                return Err(MiniError::new(
                    MiniErrorKind::InvalidOperation,
                    "replace 需要两个或三个参数",
                ));
            }
            let old = argument(0)?;
            let replacement = argument(1)?;
            let output = if let Some(limit) = args.get(2).and_then(minijinja::Value::as_usize) {
                subject.replacen(old, replacement, limit)
            } else if args.len() == 3 {
                return Err(MiniError::new(
                    MiniErrorKind::InvalidOperation,
                    "replace 的第三个参数必须是非负整数",
                ));
            } else {
                subject.replace(old, replacement)
            };
            Ok(minijinja::Value::from(output))
        }
        "split" => {
            if args.len() > 1 {
                return Err(MiniError::new(
                    MiniErrorKind::InvalidOperation,
                    "split 至多接受一个参数",
                ));
            }
            let pieces = match args.first().and_then(minijinja::Value::as_str) {
                None => subject
                    .split_whitespace()
                    .map(minijinja::Value::from)
                    .collect::<Vec<_>>(),
                Some(separator) if !separator.is_empty() => subject
                    .split(separator)
                    .map(minijinja::Value::from)
                    .collect::<Vec<_>>(),
                Some(_) => {
                    return Err(MiniError::new(
                        MiniErrorKind::InvalidOperation,
                        "split 的分隔符不能为空",
                    ))
                }
            };
            Ok(minijinja::Value::from(pieces))
        }
        _ => Err(MiniError::from(MiniErrorKind::UnknownMethod)),
    }
}

fn request_context(request: &ChatRequest, config: &Value) -> Value {
    let mut context = Map::new();
    let mut image_ordinal = 0usize;
    context.insert(
        "messages".into(),
        Value::Array(
            request
                .messages
                .iter()
                .map(|message| message_value(message, &mut image_ordinal))
                .collect(),
        ),
    );
    context.insert(
        "tools".into(),
        Value::Array(request.tools.iter().map(tool_value).collect()),
    );
    context.insert(
        "add_generation_prompt".into(),
        Value::Bool(request.add_generation_prompt),
    );
    context.insert(
        "enable_thinking".into(),
        Value::Bool(request.enable_thinking.unwrap_or(false)),
    );
    context.insert(
        "thinking_budget".into(),
        request
            .thinking_budget
            .map(|value| Value::from(value as u64))
            .unwrap_or(Value::Null),
    );
    for key in ["bos_token", "eos_token", "pad_token", "unk_token"] {
        if let Some(value) = config.get(key).and_then(token_config_string) {
            context.insert(key.into(), Value::String(value));
        }
    }
    Value::Object(context)
}

fn token_config_string(value: &Value) -> Option<String> {
    value.as_str().map(str::to_owned).or_else(|| {
        value
            .get("content")
            .and_then(Value::as_str)
            .map(str::to_owned)
    })
}

fn message_value(message: &ChatMessage, image_ordinal: &mut usize) -> Value {
    let mut object = Map::new();
    object.insert("role".into(), Value::String(message.role.as_str().into()));
    object.insert(
        "content".into(),
        content_value(&message.content, image_ordinal),
    );
    if let Some(name) = &message.name {
        object.insert("name".into(), Value::String(name.clone()));
    }
    if let Some(tool_call_id) = &message.tool_call_id {
        object.insert("tool_call_id".into(), Value::String(tool_call_id.clone()));
    }
    if !message.tool_calls.is_empty() {
        object.insert(
            "tool_calls".into(),
            Value::Array(message.tool_calls.iter().map(tool_call_value).collect()),
        );
    }
    Value::Object(object)
}

fn content_value(content: &MessageContent, image_ordinal: &mut usize) -> Value {
    match content {
        MessageContent::Text(text) => Value::String(text.clone()),
        MessageContent::Parts(parts) => Value::Array(
            parts
                .iter()
                .map(|part| content_part_value(part, image_ordinal))
                .collect(),
        ),
    }
}

fn content_part_value(part: &ContentPart, image_ordinal: &mut usize) -> Value {
    match part {
        ContentPart::Text(text) => {
            let mut object = Map::new();
            object.insert("type".into(), Value::String("text".into()));
            object.insert("text".into(), Value::String(text.clone()));
            Value::Object(object)
        }
        ContentPart::Image(image) => {
            let ordinal = *image_ordinal;
            *image_ordinal = image_ordinal.saturating_add(1);
            image_value(image, ordinal)
        }
    }
}

fn image_value(image: &ImageInput, ordinal: usize) -> Value {
    let mut object = Map::new();
    object.insert("type".into(), Value::String("image".into()));
    object.insert(
        "placeholder".into(),
        Value::String(format!("<|rsinfer_image_{ordinal}|>")),
    );
    if let Some(media_type) = &image.media_type {
        object.insert("media_type".into(), Value::String(media_type.clone()));
    }
    if let Some(alt_text) = &image.alt_text {
        object.insert("alt_text".into(), Value::String(alt_text.clone()));
    }
    let mut source = Map::new();
    match &image.source {
        ImageSource::Bytes(bytes) => {
            source.insert("kind".into(), Value::String("bytes".into()));
            source.insert("byte_len".into(), Value::from(bytes.len() as u64));
        }
        ImageSource::Path(path) => {
            source.insert("kind".into(), Value::String("path".into()));
            source.insert(
                "path".into(),
                Value::String(path.to_string_lossy().into_owned()),
            );
        }
        ImageSource::Url(url) => {
            source.insert("kind".into(), Value::String("url".into()));
            source.insert("url".into(), Value::String(url.clone()));
        }
    }
    object.insert("source".into(), Value::Object(source));
    Value::Object(object)
}

fn tool_value(tool: &Tool) -> Value {
    let mut function = Map::new();
    function.insert("name".into(), Value::String(tool.name.clone()));
    if let Some(description) = &tool.description {
        function.insert("description".into(), Value::String(description.clone()));
    }
    function.insert("parameters".into(), tool.parameters.clone());
    let mut object = Map::new();
    object.insert("type".into(), Value::String("function".into()));
    object.insert("function".into(), Value::Object(function));
    Value::Object(object)
}

fn tool_call_value(call: &ToolCall) -> Value {
    let mut function = Map::new();
    function.insert("name".into(), Value::String(call.name.clone()));
    function.insert("arguments".into(), call.arguments.clone());
    let mut object = Map::new();
    if let Some(id) = &call.id {
        object.insert("id".into(), Value::String(id.clone()));
    }
    object.insert("type".into(), Value::String("function".into()));
    object.insert("function".into(), Value::Object(function));
    Value::Object(object)
}

fn render_plain(request: &ChatRequest) -> String {
    let mut rendered = String::new();
    for message in &request.messages {
        rendered.push_str(message.role.as_str());
        rendered.push_str(": ");
        append_plain_content(&mut rendered, &message.content);
        rendered.push('\n');
    }
    if request.add_generation_prompt {
        rendered.push_str("assistant: ");
    }
    rendered
}

fn append_plain_content(output: &mut String, content: &MessageContent) {
    match content {
        MessageContent::Text(text) => output.push_str(text),
        MessageContent::Parts(parts) => {
            for part in parts {
                match part {
                    ContentPart::Text(text) => output.push_str(text),
                    ContentPart::Image(_) => output.push_str("[image]"),
                }
            }
        }
    }
}

/// 服务层端到端测试使用的内存 tokenizer；不依赖模型目录或网络。
#[cfg(test)]
pub(crate) mod fixtures {
    use super::*;
    use serde_json::json;
    use tokenizers::{models::wordlevel::WordLevel, pre_tokenizers::whitespace::Whitespace};

    pub(crate) fn frontend() -> ChatFrontend {
        let model: WordLevel = serde_json::from_str(
            r#"{"type":"WordLevel","vocab":{"<unk>":0,"user":1,"assistant":2,"system":3,"hello":4,"world":5,"tool":6,"result":7},"unk_token":"<unk>"}"#,
        )
        .expect("WordLevel fixture must deserialize");
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(Whitespace));
        let template =
            "{% for message in messages %}{{ message.role }} {{ message.content }}{% endfor %}";
        let config = json!({ "chat_template": template });
        ChatFrontend {
            model_dir: PathBuf::from("<frontend-fixture>"),
            tokenizer: Arc::new(tokenizer),
            templates: BTreeMap::from([(String::from("default"), String::from(template))]),
            tokenizer_config: config,
            output_markers: OutputMarkers::default(),
            plain_fallback: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qwen3_frontend() -> ChatFrontend {
        let mut frontend = fixtures::frontend();
        let template = include_str!("templates/qwen3.jinja").to_owned();
        frontend
            .templates
            .insert("default".into(), template.clone());
        frontend.tokenizer_config = serde_json::json!({ "chat_template": template });
        frontend.output_markers = OutputMarkers {
            thinking_start: Some("<think>".into()),
            thinking_end: Some("</think>".into()),
            tool_start: Some("<tool_call>".into()),
            tool_end: Some("</tool_call>".into()),
        };
        frontend
    }

    #[test]
    fn template_selection_requires_explicit_name_when_ambiguous() {
        let templates = BTreeMap::from([
            ("default".into(), "a".into()),
            ("tool_use".into(), "b".into()),
        ]);
        assert_eq!(select_template(&templates, None).unwrap(), "default");
        assert_eq!(
            select_template(&templates, Some("tool_use")).unwrap(),
            "tool_use"
        );
        assert!(select_template(&templates, Some("missing")).is_err());
    }

    #[test]
    fn output_marker_config_never_guesses_defaults() {
        assert_eq!(
            parse_output_markers(&Value::Object(Map::new())).unwrap(),
            OutputMarkers::default()
        );
    }

    #[test]
    fn thinking_controller_uses_exact_marker_tokens() {
        let spec = ThinkingSpec {
            supported: true,
            enabled: true,
            budget: Some(1),
            start_tokens: vec![10],
            end_tokens: vec![11],
            initially_in_thinking: false,
            disabled_reason: None,
        };
        let mut controller = ThinkingController::new(spec);
        assert_eq!(controller.observe(10), ThinkingAction::Entered);
        assert_eq!(
            controller.observe(12),
            ThinkingAction::ForceClose {
                end_tokens: vec![11]
            }
        );
    }

    #[test]
    fn image_value_keeps_source_structured() {
        let value = image_value(
            &ImageInput::url("https://example.test/a.png".into(), None),
            0,
        );
        assert_eq!(value["source"]["kind"], "url");
        assert_eq!(value["source"]["url"], "https://example.test/a.png");
        assert_eq!(value["placeholder"], "<|rsinfer_image_0|>");
    }

    #[test]
    fn bundled_qwen3_template_renders_with_fixture_tokenizer() {
        let frontend = qwen3_frontend();
        let request = ChatRequest {
            messages: vec![ChatMessage::user("hello")],
            enable_thinking: Some(false),
            ..ChatRequest::default()
        };
        let prepared = frontend.prepare(&request).unwrap();
        assert!(!prepared.tokens.is_empty());
        assert!(prepared.rendered.contains("<|im_start|>assistant"));
        assert!(prepared.rendered.contains("</think>"));
    }

    #[test]
    fn bundled_qwen3_template_renders_multiturn_thinking_modes() {
        let frontend = qwen3_frontend();
        let request = ChatRequest {
            messages: vec![
                ChatMessage::system("be precise"),
                ChatMessage::user("hello"),
                ChatMessage::assistant("world"),
                ChatMessage::user("result"),
            ],
            enable_thinking: Some(true),
            ..ChatRequest::default()
        };
        let prepared = frontend.prepare(&request).unwrap();
        assert!(prepared.rendered.contains("<|im_start|>system"));
        assert!(prepared.rendered.contains("<|im_start|>assistant"));
        assert!(prepared.thinking.supported);
        assert!(prepared.thinking.enabled);
        assert!(!prepared.thinking.initially_in_thinking);
    }

    #[test]
    fn thinking_marker_in_user_text_does_not_open_generation_state() {
        let frontend = qwen3_frontend();
        let request = ChatRequest {
            messages: vec![ChatMessage::user("literal <think> marker")],
            enable_thinking: Some(true),
            ..ChatRequest::default()
        };
        let prepared = frontend.prepare(&request).unwrap();
        assert!(!prepared.thinking.initially_in_thinking);
    }

    #[test]
    fn bundled_qwen3_template_renders_tools_and_tool_reply() {
        let frontend = qwen3_frontend();
        let tool = Tool::function(
            "lookup",
            Some("look up a value".into()),
            serde_json::json!({
                "type": "object",
                "properties": { "key": { "type": "string" } }
            }),
        )
        .unwrap();
        let mut assistant = ChatMessage::assistant("");
        assistant.tool_calls.push(ToolCall {
            id: Some("call-1".into()),
            name: "lookup".into(),
            arguments: serde_json::json!({ "key": "x" }),
        });
        let request = ChatRequest {
            messages: vec![
                ChatMessage::user("find x"),
                assistant,
                ChatMessage::tool("value", "call-1"),
            ],
            tools: vec![tool],
            enable_thinking: Some(false),
            ..ChatRequest::default()
        };
        let prepared = frontend.prepare(&request).unwrap();
        assert!(prepared.rendered.contains("<tools>"));
        assert!(prepared.rendered.contains("\"lookup\""));
        assert!(prepared.rendered.contains("<tool_call>"));
        assert!(prepared.rendered.contains("<tool_response>"));
        assert!(prepared.rendered.contains("value"));
        assert!(!prepared.thinking.enabled);
    }

    #[test]
    #[ignore = "需要设置 RSINFER_QWEN3_TOKENIZER_DIR 指向本机真实 TS-Qwen3 tokenizer 目录"]
    fn real_ts_qwen3_tokenizer_renders_external_template_and_decodes_utf8() {
        let model_dir = std::env::var_os("RSINFER_QWEN3_TOKENIZER_DIR")
            .expect("RSINFER_QWEN3_TOKENIZER_DIR must point to a real tokenizer");
        let frontend = ChatFrontend::load_with_options(
            Path::new(&model_dir),
            FrontendLoadOptions {
                allow_plain_fallback: false,
                chat_template: Some(include_str!("templates/qwen3.jinja").to_owned()),
            },
        )
        .unwrap();
        let tool = Tool::function(
            "lookup",
            Some("look up a value".into()),
            serde_json::json!({
                "type": "object",
                "properties": { "key": { "type": "string" } }
            }),
        )
        .unwrap();
        let request = ChatRequest {
            messages: vec![
                ChatMessage::system("You are concise."),
                ChatMessage::user("hello"),
            ],
            tools: vec![tool],
            enable_thinking: Some(false),
            ..ChatRequest::default()
        };
        let prepared = frontend.prepare(&request).unwrap();
        assert!(!prepared.tokens.is_empty());
        assert!(prepared.rendered.contains("<think>"));
        assert!(prepared.thinking.supported);
        assert!(!prepared.thinking.initially_in_thinking);
        assert_eq!(prepared.output_splitter().unwrap().state(), "content");

        let text = "你好，世界";
        let ids = frontend.tokenizer().encode(text, false).unwrap();
        let mut decoder = prepared.token_decoder(1024).unwrap();
        let mut decoded = String::new();
        for token_id in ids.get_ids() {
            decoded.push_str(&decoder.push(*token_id).unwrap().text);
        }
        decoded.push_str(&decoder.finish().unwrap().text);
        assert_eq!(decoded, text);
    }

    #[test]
    #[ignore = "需要设置 RSINFER_QWEN_TOKENIZER_DIR 指向本机真实 tokenizer 目录"]
    fn real_qwen_template_renders_with_python_string_methods() {
        let model_dir = std::env::var_os("RSINFER_QWEN_TOKENIZER_DIR")
            .expect("RSINFER_QWEN_TOKENIZER_DIR must point to a real tokenizer");
        let frontend = ChatFrontend::load(Path::new(&model_dir)).unwrap();
        let tool = Tool::function(
            "lookup",
            Some("look up a value".into()),
            serde_json::json!({
                "type": "object",
                "properties": { "key": { "type": "string" } }
            }),
        )
        .unwrap();
        let request = ChatRequest {
            messages: vec![
                ChatMessage::system("You are concise."),
                ChatMessage::user("hello"),
            ],
            tools: vec![tool],
            enable_thinking: Some(true),
            thinking_budget: Some(0),
            ..ChatRequest::default()
        };
        let prepared = frontend.prepare(&request).unwrap();
        assert!(!prepared.tokens.is_empty());
        assert!(prepared.rendered.contains("<think>"));
        assert!(prepared.thinking.supported);
        assert!(prepared.thinking.initially_in_thinking);
        assert_eq!(prepared.output_splitter().unwrap().state(), "thinking");
    }
}

//! 增量 token 解码、stop 串保护和输出通道拆分。

use std::sync::Arc;

use serde_json::Value;
use tokenizers::Tokenizer;

use crate::{Result, RsinferError};

/// 单次增量解码返回的文本片段。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedChunk {
    pub text: String,
    pub stop: Option<String>,
    pub finished: bool,
}

/// 通过累积 decode 保持跨 token UTF-8 和 stop 前缀安全的解码器。
pub struct TokenDecoder {
    tokenizer: Arc<Tokenizer>,
    token_ids: Vec<u32>,
    decoded: String,
    pending: String,
    stop_strings: Vec<String>,
    max_output_bytes: usize,
    emitted_bytes: usize,
    skip_special_tokens: bool,
    finished: bool,
}

impl TokenDecoder {
    pub fn new(
        tokenizer: Arc<Tokenizer>,
        stop_strings: Vec<String>,
        max_output_bytes: usize,
    ) -> Result<Self> {
        Self::with_options(tokenizer, stop_strings, max_output_bytes, true)
    }

    /// `skip_special_tokens=false` 适用于需要识别模板声明的思考/工具标记。
    pub fn with_options(
        tokenizer: Arc<Tokenizer>,
        stop_strings: Vec<String>,
        max_output_bytes: usize,
        skip_special_tokens: bool,
    ) -> Result<Self> {
        if max_output_bytes == 0 {
            return Err(RsinferError::ConfigError(
                "解码器 max_output_bytes 必须大于零".into(),
            ));
        }
        if stop_strings.iter().any(String::is_empty) {
            return Err(RsinferError::ConfigError("stop 字符串不能为空".into()));
        }
        Ok(Self {
            tokenizer,
            token_ids: Vec::new(),
            decoded: String::new(),
            pending: String::new(),
            stop_strings,
            max_output_bytes,
            emitted_bytes: 0,
            skip_special_tokens,
            finished: false,
        })
    }

    pub fn token_ids(&self) -> &[u32] {
        &self.token_ids
    }

    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// 吸收一个 token；返回可安全发布的文本，stop 本身永远不会泄露。
    pub fn push(&mut self, token_id: u32) -> Result<DecodedChunk> {
        if self.finished {
            return Ok(DecodedChunk {
                text: String::new(),
                stop: None,
                finished: true,
            });
        }
        self.token_ids.push(token_id);
        let full = self
            .tokenizer
            .decode(&self.token_ids, self.skip_special_tokens)
            .map_err(|error| RsinferError::TokenizerError(error.to_string()))?;
        // 字节级 BPE/byte-fallback tokenizer 可能先把不完整的 UTF-8 序列
        // 解为 U+FFFD，并在下一个字节到达后重写这个尾部（例如
        // `\xE4` -> `\xE4\xBD\xA0`：`�` -> `你`）。只有已经发给调用方的
        // 前缀必须保持不变，缓冲尾部允许重新解码。
        let emitted_len = self.decoded.len().saturating_sub(self.pending.len());
        if common_prefix_len(&self.decoded, &full) < emitted_len {
            return Err(RsinferError::TokenizerError(
                "增量 tokenizer decode 不是前缀扩展，无法安全流式输出".into(),
            ));
        }
        self.decoded = full;
        self.pending = self.decoded[emitted_len..].to_owned();
        self.flush_pending(false)
    }

    /// 生成结束时释放尚未确定是否为 stop 前缀的安全尾部。
    pub fn finish(&mut self) -> Result<DecodedChunk> {
        if self.finished {
            return Ok(DecodedChunk {
                text: String::new(),
                stop: None,
                finished: true,
            });
        }
        self.finished = true;
        self.flush_pending(true)
    }

    fn flush_pending(&mut self, final_chunk: bool) -> Result<DecodedChunk> {
        if let Some((index, stop)) = find_stop(&self.pending, &self.stop_strings) {
            let output = self.pending[..index].to_owned();
            self.pending.clear();
            self.finished = true;
            self.account_output(&output)?;
            return Ok(DecodedChunk {
                text: output,
                stop: Some(stop),
                finished: true,
            });
        }
        if final_chunk {
            let output = std::mem::take(&mut self.pending);
            self.account_output(&output)?;
            return Ok(DecodedChunk {
                text: output,
                stop: None,
                finished: true,
            });
        }
        let safe_len = self.pending.len().saturating_sub(
            longest_stop_prefix_suffix(&self.pending, &self.stop_strings)
                .max(unstable_utf8_suffix_len(&self.pending)),
        );
        let split_at = previous_char_boundary(&self.pending, safe_len);
        let output = self.pending[..split_at].to_owned();
        self.pending.drain(..split_at);
        self.account_output(&output)?;
        Ok(DecodedChunk {
            text: output,
            stop: None,
            finished: false,
        })
    }

    fn account_output(&mut self, output: &str) -> Result<()> {
        self.emitted_bytes = self
            .emitted_bytes
            .checked_add(output.len())
            .ok_or_else(|| RsinferError::DimensionError("解码输出长度溢出".into()))?;
        if self.emitted_bytes > self.max_output_bytes {
            return Err(RsinferError::DimensionError(format!(
                "解码输出超过 max_output_bytes={} 限制",
                self.max_output_bytes
            )));
        }
        Ok(())
    }
}

fn common_prefix_len(left: &str, right: &str) -> usize {
    left.as_bytes()
        .iter()
        .zip(right.as_bytes())
        .take_while(|(left, right)| left == right)
        .count()
}

/// `tokenizers` 会把末尾不完整的 UTF-8 字节序列表示为 U+FFFD。
/// 保留这个替代字符，直到下一个 token 或最终收束给解码器机会补全序列。
fn unstable_utf8_suffix_len(text: &str) -> usize {
    if text.ends_with('\u{FFFD}') {
        '\u{FFFD}'.len_utf8()
    } else {
        0
    }
}

fn find_stop(text: &str, stops: &[String]) -> Option<(usize, String)> {
    stops
        .iter()
        .filter_map(|stop| text.find(stop).map(|index| (index, stop)))
        .min_by(|left, right| {
            left.0
                .cmp(&right.0)
                .then_with(|| left.1.len().cmp(&right.1.len()))
        })
        .map(|(index, stop)| (index, stop.clone()))
}

fn longest_stop_prefix_suffix(text: &str, stops: &[String]) -> usize {
    let mut longest = 0usize;
    for stop in stops {
        let limit = text.len().min(stop.len());
        for length in 1..=limit {
            let text_start = text.len() - length;
            let stop_end = length;
            if !text.is_char_boundary(text_start) || !stop.is_char_boundary(stop_end) {
                continue;
            }
            if text[text_start..] == stop[..stop_end] {
                longest = longest.max(length);
            }
        }
    }
    longest
}

fn previous_char_boundary(text: &str, index: usize) -> usize {
    let mut index = index.min(text.len());
    while index > 0 && !text.is_char_boundary(index) {
        index -= 1;
    }
    index
}

/// 模板明确声明的输出标记。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OutputMarkers {
    pub thinking_start: Option<String>,
    pub thinking_end: Option<String>,
    pub tool_start: Option<String>,
    pub tool_end: Option<String>,
}

impl OutputMarkers {
    pub fn validate(&self) -> Result<()> {
        for (name, value) in [
            ("thinking_start", &self.thinking_start),
            ("thinking_end", &self.thinking_end),
            ("tool_start", &self.tool_start),
            ("tool_end", &self.tool_end),
        ] {
            if value.as_deref().is_some_and(str::is_empty) {
                return Err(RsinferError::ConfigError(format!(
                    "输出标记 {name} 不能为空"
                )));
            }
        }
        if self.thinking_start.is_some() != self.thinking_end.is_some() {
            return Err(RsinferError::ConfigError(
                "thinking 输出标记必须成对声明".into(),
            ));
        }
        if self.tool_start.is_some() != self.tool_end.is_some() {
            return Err(RsinferError::ConfigError(
                "tool 输出标记必须成对声明".into(),
            ));
        }
        Ok(())
    }

    pub fn has_thinking(&self) -> bool {
        self.thinking_start.is_some() && self.thinking_end.is_some()
    }

    pub fn has_tools(&self) -> bool {
        self.tool_start.is_some() && self.tool_end.is_some()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum OutputEvent {
    Thinking(String),
    Content(String),
    ToolCall { value: Value, raw: String },
    Finished,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutputState {
    Content,
    Thinking,
    ToolCall,
}

/// 将模型输出拆分为 thinking/content/tool call，标记和半截 stop 不会泄露。
pub struct OutputSplitter {
    markers: OutputMarkers,
    state: OutputState,
    pending: String,
    tool_buffer: String,
}

impl OutputSplitter {
    pub fn new(markers: OutputMarkers) -> Result<Self> {
        Self::new_with_initial_state(markers, false)
    }

    /// 创建输出拆分器，并允许模板在 prompt 中已经打开 thinking。
    pub fn new_with_initial_state(
        markers: OutputMarkers,
        initially_in_thinking: bool,
    ) -> Result<Self> {
        markers.validate()?;
        if initially_in_thinking && !markers.has_thinking() {
            return Err(RsinferError::ConfigError(
                "初始 thinking 状态需要成对 thinking 输出标记".into(),
            ));
        }
        Ok(Self {
            markers,
            state: if initially_in_thinking {
                OutputState::Thinking
            } else {
                OutputState::Content
            },
            pending: String::new(),
            tool_buffer: String::new(),
        })
    }

    pub fn state(&self) -> &'static str {
        match self.state {
            OutputState::Content => "content",
            OutputState::Thinking => "thinking",
            OutputState::ToolCall => "tool_call",
        }
    }

    pub fn push_text(&mut self, text: &str) -> Result<Vec<OutputEvent>> {
        self.pending.push_str(text);
        self.drain(false)
    }

    pub fn finish(&mut self) -> Result<Vec<OutputEvent>> {
        let mut events = self.drain(true)?;
        if self.state == OutputState::ToolCall {
            return Err(RsinferError::TokenizerError(
                "生成结束时 tool call 标记未闭合".into(),
            ));
        }
        events.push(OutputEvent::Finished);
        Ok(events)
    }

    fn drain(&mut self, final_chunk: bool) -> Result<Vec<OutputEvent>> {
        let mut events = Vec::new();
        loop {
            let marker = self.next_marker();
            let Some((index, kind, marker_len)) = marker else {
                if final_chunk {
                    let tail = std::mem::take(&mut self.pending);
                    self.emit_text(&mut events, tail)?;
                } else {
                    let safe_len = self
                        .pending
                        .len()
                        .saturating_sub(self.longest_active_prefix_suffix());
                    let split_at = previous_char_boundary(&self.pending, safe_len);
                    let tail = self.pending[..split_at].to_owned();
                    self.pending.drain(..split_at);
                    self.emit_text(&mut events, tail)?;
                }
                break;
            };

            let before = self.pending[..index].to_owned();
            self.pending.drain(..index + marker_len);
            self.emit_text(&mut events, before)?;
            match kind {
                MarkerKind::ThinkingStart => self.state = OutputState::Thinking,
                MarkerKind::ThinkingEnd => self.state = OutputState::Content,
                MarkerKind::ToolStart => {
                    self.tool_buffer.clear();
                    self.state = OutputState::ToolCall;
                }
                MarkerKind::ToolEnd => {
                    let raw = std::mem::take(&mut self.tool_buffer);
                    let value = serde_json::from_str::<Value>(&raw).map_err(|error| {
                        RsinferError::TokenizerError(format!("tool call JSON 解析失败：{error}"))
                    })?;
                    events.push(OutputEvent::ToolCall { value, raw });
                    self.state = OutputState::Content;
                }
            }
        }
        Ok(events)
    }

    fn emit_text(&mut self, events: &mut Vec<OutputEvent>, text: String) -> Result<()> {
        if text.is_empty() {
            return Ok(());
        }
        match self.state {
            OutputState::Content => events.push(OutputEvent::Content(text)),
            OutputState::Thinking => events.push(OutputEvent::Thinking(text)),
            OutputState::ToolCall => self.tool_buffer.push_str(&text),
        }
        Ok(())
    }

    fn next_marker(&self) -> Option<(usize, MarkerKind, usize)> {
        let candidates: Vec<(&str, MarkerKind)> = match self.state {
            OutputState::Content => [
                self.markers
                    .thinking_start
                    .as_deref()
                    .map(|value| (value, MarkerKind::ThinkingStart)),
                self.markers
                    .tool_start
                    .as_deref()
                    .map(|value| (value, MarkerKind::ToolStart)),
            ]
            .into_iter()
            .flatten()
            .collect(),
            OutputState::Thinking => self
                .markers
                .thinking_end
                .as_deref()
                .map(|value| vec![(value, MarkerKind::ThinkingEnd)])
                .unwrap_or_default(),
            OutputState::ToolCall => self
                .markers
                .tool_end
                .as_deref()
                .map(|value| vec![(value, MarkerKind::ToolEnd)])
                .unwrap_or_default(),
        };
        candidates
            .into_iter()
            .filter_map(|(marker, kind)| {
                self.pending
                    .find(marker)
                    .map(|index| (index, kind, marker.len()))
            })
            .min_by(|left, right| left.0.cmp(&right.0).then_with(|| left.2.cmp(&right.2)))
    }

    fn longest_active_prefix_suffix(&self) -> usize {
        let markers: Vec<&str> = match self.state {
            OutputState::Content => [
                self.markers.thinking_start.as_deref(),
                self.markers.tool_start.as_deref(),
            ]
            .into_iter()
            .flatten()
            .collect(),
            OutputState::Thinking => self.markers.thinking_end.as_deref().into_iter().collect(),
            OutputState::ToolCall => self.markers.tool_end.as_deref().into_iter().collect(),
        };
        longest_stop_prefix_suffix(
            &self.pending,
            &markers
                .iter()
                .map(|value| (*value).to_owned())
                .collect::<Vec<_>>(),
        )
    }
}

#[derive(Debug, Clone, Copy)]
enum MarkerKind {
    ThinkingStart,
    ThinkingEnd,
    ToolStart,
    ToolEnd,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    fn byte_level_tokenizer() -> Arc<Tokenizer> {
        Arc::new(
            Tokenizer::from_str(
                r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":false},"post_processor":null,"decoder":{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true},"model":{"type":"BPE","dropout":null,"unk_token":"<unk>","continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,"vocab":{"<unk>":0,"ä":1,"½":2,"ł":3,"a":4,"b":5},"merges":[]}}"#,
            )
            .expect("byte-level tokenizer fixture must deserialize"),
        )
    }

    #[test]
    fn stop_prefix_is_not_emitted() {
        let tokenizer = Arc::new(Tokenizer::new(tokenizers::models::bpe::BPE::default()));
        let mut decoder = TokenDecoder::new(tokenizer, vec!["<stop>".into()], 32).unwrap();
        decoder.pending = "<st".into();
        let chunk = decoder.flush_pending(false).unwrap();
        assert!(chunk.text.is_empty());
        assert_eq!(decoder.pending, "<st");
    }

    #[test]
    fn byte_level_utf8_is_buffered_until_complete() {
        let tokenizer = byte_level_tokenizer();
        let encoded = tokenizer.encode("你", false).unwrap();
        assert_eq!(encoded.get_ids(), &[1, 2, 3]);
        let mut decoder = TokenDecoder::with_options(tokenizer, vec![], 32, false).unwrap();

        assert_eq!(decoder.push(1).unwrap().text, "");
        assert_eq!(decoder.push(2).unwrap().text, "");
        assert_eq!(decoder.push(3).unwrap().text, "你");
        assert_eq!(decoder.finish().unwrap().text, "");
    }

    #[test]
    fn splitter_holds_marker_and_parses_tool_json() {
        let markers = OutputMarkers {
            thinking_start: Some("<think>".into()),
            thinking_end: Some("</think>".into()),
            tool_start: Some("<tool>".into()),
            tool_end: Some("</tool>".into()),
        };
        let mut splitter = OutputSplitter::new(markers).unwrap();
        assert_eq!(
            splitter.push_text("a<think>b</think>c").unwrap(),
            vec![
                OutputEvent::Content("a".into()),
                OutputEvent::Thinking("b".into()),
                OutputEvent::Content("c".into())
            ]
        );
        let events = splitter.push_text("<tool>{\"name\":\"x\"}</tool>").unwrap();
        assert!(matches!(events.as_slice(), [OutputEvent::ToolCall { .. }]));
    }
}

//! 有限 OpenAI 风格协议：聊天、SSE、模型能力与指标。断开流即取消生成。

use crate::engine::scheduler::{FinishReason, GenerationEvent, GenerationHandle, Usage};
use crate::frontend::{OutputEvent, OutputSplitter, PreparedPrompt, TokenDecoder};
use crate::service::{ApiRequest, Service};
use crate::{Result, RsinferError};
use axum::{
    body::Body,
    extract::{DefaultBodyLimit, FromRequest, State},
    http::{HeaderMap, Request, StatusCode},
    middleware::{self, Next},
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    routing::{get, post},
    Json, Router,
};
use serde_json::{json, Value};
use std::{
    convert::Infallible,
    net::SocketAddr,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::sync::{mpsc, Semaphore};
use tokio_stream::{wrappers::ReceiverStream, StreamExt};

static NEXT_REQUEST: AtomicU64 = AtomicU64::new(1);
#[derive(Clone)]
struct App {
    service: Arc<Service>,
    api_key: Option<Arc<str>>,
    preparing: Arc<Semaphore>,
}

pub fn router(service: Arc<Service>, api_key: Option<String>) -> Router {
    let state = App {
        preparing: Arc::new(Semaphore::new(service.config.max_preparing)),
        api_key: api_key.map(Arc::from),
        service,
    };
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/metrics", get(metrics))
        .route("/v1/chat/completions", post(chat))
        .fallback(not_found)
        .layer(DefaultBodyLimit::max(state.service.config.max_body_bytes))
        .layer(middleware::from_fn_with_state(state.clone(), authenticate))
        .with_state(state)
}
pub async fn serve(
    service: Arc<Service>,
    address: SocketAddr,
    api_key: Option<String>,
) -> Result<()> {
    if !address.ip().is_loopback() && api_key.as_ref().is_none_or(|key| key.is_empty()) {
        return Err(RsinferError::ConfigError(
            "非回环监听需要 SAFETYRAISE_API_KEY".into(),
        ));
    }
    let listener = tokio::net::TcpListener::bind(address).await?;
    eprintln!(
        "服务监听 {}，模型 {}",
        listener.local_addr()?,
        service.config.model_name
    );
    axum::serve(listener, router(service, api_key))
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}
async fn authenticate(
    State(app): State<App>,
    headers: HeaderMap,
    request: Request<Body>,
    next: Next,
) -> Response {
    if let Some(key) = &app.api_key {
        let authorized = headers
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "));
        if !authorized.is_some_and(|value| {
            value.len() == key.len()
                && value
                    .bytes()
                    .zip(key.bytes())
                    .fold(0u8, |x, (a, b)| x | (a ^ b))
                    == 0
        }) {
            return error_response(StatusCode::UNAUTHORIZED, "需要有效 Bearer API key");
        }
    }
    next.run(request).await
}
async fn health(State(app): State<App>) -> Json<Value> {
    Json(json!({"status":"ok","capabilities":app.service.capabilities()}))
}
async fn models(State(app): State<App>) -> Json<Value> {
    Json(
        json!({"object":"list","data":[{"id":app.service.config.model_name,"object":"model","owned_by":"local","capabilities":app.service.capabilities()}]}),
    )
}
async fn metrics(State(app): State<App>) -> Response {
    let backend = match app.service.engine.model().backend() {
        crate::backend::Backend::Cpu => json!({"name":"cpu"}),
        crate::backend::Backend::Native(native) => {
            let stats = native.stats();
            json!({"name":native.name(),"prepared_weights":stats.prepared_weights,"fallback_count":stats.fallback_count})
        }
        crate::backend::Backend::WgpuQ8(wgpu) => match wgpu.stats() {
            Ok(stats) => json!({"name":"wgpu-q8","stats":stats}),
            Err(error) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, error),
        },
    };
    Json(json!({"engine":app.service.engine.metrics(),"backend":backend})).into_response()
}
async fn not_found() -> Response {
    error_response(StatusCode::NOT_FOUND, "接口不存在")
}
fn error_response(status: StatusCode, message: impl ToString) -> Response {
    (status,Json(json!({"error":{"message":message.to_string(),"type":"engine_error","code":status.as_u16()}}))).into_response()
}
fn status(error: &RsinferError) -> StatusCode {
    match error {
        RsinferError::ResourceExhausted(_) => StatusCode::TOO_MANY_REQUESTS,
        RsinferError::Unavailable(_) => StatusCode::SERVICE_UNAVAILABLE,
        RsinferError::WeightError(_) | RsinferError::Io(_) => StatusCode::INTERNAL_SERVER_ERROR,
        _ => StatusCode::BAD_REQUEST,
    }
}

async fn chat(State(app): State<App>, request: Request<Body>) -> Response {
    // 在读完整 JSON/base64 之前入场，避免大量连接同时分配请求体。
    let permit = match app.preparing.clone().try_acquire_owned() {
        Ok(permit) => permit,
        Err(_) => return error_response(StatusCode::TOO_MANY_REQUESTS, "输入处理容量已满"),
    };
    let body = match tokio::time::timeout(
        Duration::from_secs(30),
        Json::<ApiRequest>::from_request(request, &()),
    )
    .await
    {
        Ok(body) => body,
        Err(_) => return error_response(StatusCode::REQUEST_TIMEOUT, "请求体读取超时"),
    };
    let Json(input) = match body {
        Ok(value) => value,
        Err(error) => return error_response(error.status(), error.body_text()),
    };
    if input.model != app.service.config.model_name {
        return error_response(StatusCode::NOT_FOUND, "模型不存在");
    }
    let stream = input.stream;
    let service = app.service.clone();
    let prepared = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        service.prepare(&input)
    })
    .await;
    let prepared = match prepared {
        Ok(Ok(p)) => p,
        Ok(Err(error)) => return error_response(status(&error), error),
        Err(_) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, "输入处理线程失败"),
    };
    let (prompt, handle, tools) = match app.service.submit(prepared) {
        Ok(v) => v,
        Err(error) => return error_response(status(&error), error),
    };
    let id = format!("chatcmpl-{}", NEXT_REQUEST.fetch_add(1, Ordering::Relaxed));
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let model = app.service.config.model_name.clone();
    let max_bytes = app.service.config.max_output_bytes;
    if stream {
        let (sender, receiver) = mpsc::channel::<Value>(8);
        tokio::spawn(async move {
            let result = stream_chat(
                &prompt, handle, &tools, max_bytes, &id, created, &model, &sender,
            )
            .await;
            if let Err(error) = result {
                let _ = send(
                    &sender,
                    json!({"error":{"message":error.to_string(),"type":"generation_error"}}),
                )
                .await;
            }
            let _ = send(&sender, Value::Null).await;
        });
        let events = ReceiverStream::new(receiver).map(|value| {
            Ok::<Event, Infallible>(Event::default().data(if value.is_null() {
                "[DONE]".into()
            } else {
                value.to_string()
            }))
        });
        Sse::new(events)
            .keep_alive(KeepAlive::new().interval(Duration::from_secs(10)))
            .into_response()
    } else {
        match complete(&prompt,handle,&tools,max_bytes).await {
            Ok(result)=>Json(json!({"id":id,"object":"chat.completion","created":created,"model":model,
                "choices":[{"index":0,"message":result.message,"finish_reason":result.finish_reason}],"usage":usage_json(&result.usage)})).into_response(),
            Err(error)=>error_response(status(&error),error),
        }
    }
}

pub struct Completion {
    pub message: Value,
    pub finish_reason: String,
    pub usage: Usage,
}
struct Output {
    decoder: TokenDecoder,
    splitter: OutputSplitter,
    content: String,
    thinking: String,
    tools: Vec<Value>,
    allowed: Vec<String>,
    stopped: bool,
}
impl Output {
    fn new(prompt: &PreparedPrompt, tools: &[String], max_bytes: usize) -> Result<Self> {
        Ok(Self {
            decoder: prompt.token_decoder(max_bytes)?,
            splitter: prompt.output_splitter()?,
            content: String::new(),
            thinking: String::new(),
            tools: vec![],
            allowed: tools.to_vec(),
            stopped: false,
        })
    }
    fn token(&mut self, id: u32) -> Result<Vec<Value>> {
        let chunk = self.decoder.push(id)?;
        self.stopped |= chunk.finished;
        let events = self.splitter.push_text(&chunk.text)?;
        self.events(events)
    }
    fn finish(&mut self) -> Result<Vec<Value>> {
        let tail = self.decoder.finish()?;
        let mut events = self.splitter.push_text(&tail.text)?;
        events.extend(self.splitter.finish()?);
        self.events(events)
    }
    fn events(&mut self, events: Vec<OutputEvent>) -> Result<Vec<Value>> {
        let mut deltas = vec![];
        for event in events {
            match event {
                OutputEvent::Content(text) => {
                    self.content.push_str(&text);
                    deltas.push(json!({"content":text}));
                }
                OutputEvent::Thinking(text) => {
                    self.thinking.push_str(&text);
                    deltas.push(json!({"reasoning_content":text}));
                }
                OutputEvent::ToolCall { value, .. } => {
                    let name = value.get("name").and_then(Value::as_str).ok_or_else(|| {
                        RsinferError::ConfigError("生成的 tool_call 缺少 name".into())
                    })?;
                    if !self.allowed.iter().any(|n| n == name) {
                        return Err(RsinferError::ConfigError("模型返回未声明的工具".into()));
                    }
                    let args = value
                        .get("arguments")
                        .filter(|v| v.is_object())
                        .ok_or_else(|| {
                            RsinferError::ConfigError("工具参数不是 JSON 对象".into())
                        })?;
                    let index = self.tools.len();
                    let tool = json!({"id":format!("call_{index}"),"type":"function","function":{"name":name,"arguments":args.to_string()}});
                    self.tools.push(tool.clone());
                    let mut delta = tool;
                    delta["index"] = json!(index);
                    deltas.push(json!({"tool_calls":[delta]}));
                }
                // 分流器关闭只是正常冲刷，不能把 token 上限误报为 stop。
                OutputEvent::Finished => {}
            }
        }
        Ok(deltas)
    }
    fn reason(&self, reason: FinishReason) -> Result<String> {
        if self.stopped || reason == FinishReason::Stop {
            return Ok(if self.tools.is_empty() {
                "stop"
            } else {
                "tool_calls"
            }
            .into());
        }
        match reason {
            FinishReason::Length => Ok("length".into()),
            _ => Err(RsinferError::Unavailable(format!("生成未完成：{reason:?}"))),
        }
    }
    fn message(&self) -> Value {
        let mut message = json!({"role":"assistant","content":self.content});
        if !self.thinking.is_empty() {
            message["reasoning_content"] = json!(self.thinking);
        }
        if !self.tools.is_empty() {
            message["tool_calls"] = json!(self.tools);
        }
        message
    }
}

pub async fn complete(
    prompt: &PreparedPrompt,
    mut handle: GenerationHandle,
    tools: &[String],
    max_bytes: usize,
) -> Result<Completion> {
    let mut output = Output::new(prompt, tools, max_bytes)?;
    while let Some(event) = handle.receiver.recv().await {
        match event {
            GenerationEvent::Token { id, .. } => {
                if !output.stopped {
                    output.token(id)?;
                    if output.stopped {
                        handle.cancel();
                    }
                }
            }
            GenerationEvent::Error(message) => return Err(RsinferError::Unavailable(message)),
            GenerationEvent::Finished { reason, usage } => {
                output.finish()?;
                return Ok(Completion {
                    message: output.message(),
                    finish_reason: output.reason(reason)?,
                    usage,
                });
            }
        }
    }
    Err(RsinferError::Unavailable(
        "生成通道关闭，未收到完成事件".into(),
    ))
}
fn usage_json(usage: &Usage) -> Value {
    json!({"prompt_tokens":usage.prompt_tokens,"completion_tokens":usage.completion_tokens,
    "total_tokens":usage.prompt_tokens+usage.completion_tokens,"prompt_tokens_details":{"cached_tokens":usage.cached_tokens},"queue_ms":usage.queue_ms,"elapsed_ms":usage.elapsed_ms})
}
fn chunk(id: &str, created: u64, model: &str, delta: Value, reason: Option<&str>) -> Value {
    json!({"id":id,"object":"chat.completion.chunk","created":created,"model":model,"choices":[{"index":0,"delta":delta,"finish_reason":reason}]})
}
async fn send(sender: &mpsc::Sender<Value>, value: Value) -> Result<()> {
    tokio::time::timeout(Duration::from_secs(10), sender.send(value))
        .await
        .map_err(|_| RsinferError::Unavailable("客户端读取超时".into()))?
        .map_err(|_| RsinferError::Unavailable("客户端已断开".into()))
}
#[allow(clippy::too_many_arguments)]
async fn stream_chat(
    prompt: &PreparedPrompt,
    mut handle: GenerationHandle,
    tools: &[String],
    max_bytes: usize,
    id: &str,
    created: u64,
    model: &str,
    sender: &mpsc::Sender<Value>,
) -> Result<()> {
    let mut output = Output::new(prompt, tools, max_bytes)?;
    send(
        sender,
        chunk(id, created, model, json!({"role":"assistant"}), None),
    )
    .await?;
    while let Some(event) = tokio::select! {
        _ = sender.closed() => return Err(RsinferError::Unavailable("流式消费者已断开".into())),
        event = handle.receiver.recv() => event,
    } {
        match event {
            GenerationEvent::Token { id: token, .. } => {
                if !output.stopped {
                    for delta in output.token(token)? {
                        send(sender, chunk(id, created, model, delta, None)).await?;
                    }
                    if output.stopped {
                        handle.cancel();
                    }
                }
            }
            GenerationEvent::Error(message) => return Err(RsinferError::Unavailable(message)),
            GenerationEvent::Finished { reason, usage } => {
                for delta in output.finish()? {
                    send(sender, chunk(id, created, model, delta, None)).await?;
                }
                let reason = output.reason(reason)?;
                let mut final_chunk = chunk(id, created, model, json!({}), Some(&reason));
                final_chunk["usage"] = usage_json(&usage);
                send(sender, final_chunk).await?;
                return Ok(());
            }
        }
    }
    Err(RsinferError::Unavailable("生成通道提前关闭".into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use http_body_util::BodyExt;
    use tower::ServiceExt;
    fn request(stream: bool) -> Value {
        json!({"model":"tiny","messages":[{"role":"user","content":"hello world"}],"temperature":0.0,"max_tokens":4,"stream":stream})
    }
    async fn call(app: Router, value: Value) -> (StatusCode, String) {
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(value.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = response.status();
        let body = tokio::time::timeout(Duration::from_secs(4), response.into_body().collect())
            .await
            .unwrap()
            .unwrap()
            .to_bytes();
        (status, String::from_utf8(body.to_vec()).unwrap())
    }
    #[tokio::test]
    async fn real_fixture_flows_through_http_json_and_sse_with_same_output() {
        let app = router(Arc::new(crate::service::tests::fixture()), None);
        let (code, body) = call(app.clone(), request(false)).await;
        assert_eq!(code, StatusCode::OK, "{body}");
        let full: Value = serde_json::from_str(&body).unwrap();
        let (code, body) = call(app.clone(), request(true)).await;
        assert_eq!(code, StatusCode::OK, "{body}");
        assert!(body.contains("data: [DONE]"));
        let mut content = String::new();
        let mut finished = false;
        for line in body
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .filter(|s| *s != "[DONE]")
        {
            let event: Value = serde_json::from_str(line).unwrap();
            assert!(event.get("error").is_none(), "{event}");
            if let Some(text) = event
                .pointer("/choices/0/delta/content")
                .and_then(Value::as_str)
            {
                content.push_str(text);
            }
            finished |= event
                .pointer("/choices/0/finish_reason")
                .is_some_and(|v| v == "length");
        }
        assert_eq!(
            full.pointer("/choices/0/message/content").unwrap(),
            &json!(content)
        );
        assert!(finished);
        let mut stopped = request(false);
        stopped["max_tokens"] = json!(20);
        if let Some(first) = content.chars().next() {
            stopped["stop"] = json!(first.to_string());
            let (status, body) = call(app, stopped).await;
            assert_eq!(status, StatusCode::OK, "{body}");
        }
    }
    #[tokio::test]
    async fn protocol_rejects_unknown_model_fields_and_missing_auth() {
        let service = Arc::new(crate::service::tests::fixture());
        let app = router(service.clone(), None);
        let mut unknown = request(false);
        unknown["model"] = json!("missing");
        assert_eq!(call(app.clone(), unknown).await.0, StatusCode::NOT_FOUND);
        let mut unsupported = request(false);
        unsupported["logprobs"] = json!(true);
        assert_eq!(
            call(app, unsupported).await.0,
            StatusCode::UNPROCESSABLE_ENTITY
        );
        assert_eq!(
            call(router(service, Some("test-key".into())), request(false))
                .await
                .0,
            StatusCode::UNAUTHORIZED
        );
    }

    #[tokio::test]
    async fn sse_disconnect_cancels_even_before_first_generated_token() {
        let service = crate::service::tests::fixture();
        let mut input = request(false);
        input["max_tokens"] = json!(40);
        let (_, blocker, _) = service
            .submit(
                service
                    .prepare(&serde_json::from_value(input).unwrap())
                    .unwrap(),
            )
            .unwrap();
        tokio::time::timeout(Duration::from_secs(2), async {
            while service.engine.metrics().generated_tokens < 2 {
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        })
        .await
        .unwrap();
        let (prompt, handle, tools) = service
            .submit(
                service
                    .prepare(&serde_json::from_value(request(true)).unwrap())
                    .unwrap(),
            )
            .unwrap();
        let (sender, mut receiver) = mpsc::channel(1);
        let task = tokio::spawn(async move {
            stream_chat(&prompt, handle, &tools, 1024, "test", 0, "tiny", &sender).await
        });
        assert!(receiver.recv().await.is_some());
        drop(receiver);
        let result = tokio::time::timeout(Duration::from_millis(500), task)
            .await
            .unwrap()
            .unwrap();
        assert!(result.is_err());
        drop(blocker);
        tokio::time::timeout(Duration::from_secs(2), async {
            while service.engine.metrics().active != 0 || service.engine.metrics().queued != 0 {
                tokio::time::sleep(Duration::from_millis(2)).await;
            }
        })
        .await
        .unwrap();
    }
}

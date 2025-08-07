# SmolLM3 Web Architecture: Rust 1.88 + Axum 0.8.4 + HTMX + SSE Call Chain

## 🚀 **Complete Call Chain Architecture**

### **System Architecture Overview**
```
Browser (HTMX) → Axum Routes → SmolLM3 Service → SSE Stream → MiniJinja Templates → UI Updates
```

## 📋 **Dependency Stack**

```toml
[dependencies]
# Core Rust 1.88 Web Stack
axum = "0.8.4"
tokio = { version = "1.42", features = ["full"] }
tower = "0.5"
tower-http = { version = "0.6", features = ["fs", "cors", "trace"] }

# SmolLM3 AI Stack (Official Candle)
candle-core = "0.9.1"
candle-nn = "0.9.1"  
candle-transformers = "0.9.1"
tokenizers = "0.21"

# Templating & Web UI
minijinja = { version = "2.3", features = ["json"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

# SSE & Real-time
axum-streams = "0.8"
futures = "0.3"
tokio-stream = "0.1"

# Utilities
uuid = { version = "1.17", features = ["v4"] }
tracing = "0.1"
tracing-subscriber = "0.3"
anyhow = "1.0"
```

## 🔗 **Complete Call Chain Flow**

### **1. Browser → HTMX Form Submission**
```html
<!-- Chat Interface with HTMX -->
<form hx-post="/api/chat/message" 
      hx-target="#chat-container" 
      hx-swap="beforeend"
      hx-trigger="submit"
      hx-indicator="#loading"
      hx-on::after-request="this.reset()">
  
  <input type="hidden" name="session_id" value="{{ session_id }}">
  <input type="hidden" name="reasoning_mode" value="{{ reasoning_mode }}">
  
  <div class="flex gap-2">
    <input type="text" 
           name="message" 
           placeholder="Type your message..."
           class="flex-1 p-2 border rounded"
           required>
    <button type="submit" class="px-4 py-2 bg-blue-500 text-white rounded">
      Send
    </button>
  </div>
  
  <!-- Reasoning Mode Toggle -->
  <label class="flex items-center gap-2 mt-2">
    <input type="checkbox" 
           name="enable_thinking"
           hx-post="/api/chat/toggle-thinking"
           hx-target="#reasoning-indicator"
           hx-swap="outerHTML">
    <span>Enable Thinking Mode</span>
  </label>
</form>

<!-- SSE Connection for Real-time Updates -->
<div hx-ext="sse" 
     sse-connect="/api/chat/stream/{{ session_id }}"
     sse-swap="message">
  <div id="chat-container">
    <!-- Messages appear here -->
  </div>
</div>
```

### **2. Axum Route Handler**
```rust
use axum::{
    extract::{Form, Path, State},
    response::{Html, Response, sse::Event},
    routing::{get, post},
    Router,
};
use minijinja::{Environment, context};
use serde::{Deserialize, Serialize};
use tokio_stream::{wrappers::UnboundedReceiverStream, StreamExt};

// Route setup
pub fn create_chat_routes() -> Router<AppState> {
    Router::new()
        .route("/", get(chat_page))
        .route("/api/chat/message", post(handle_message))
        .route("/api/chat/stream/:session_id", get(chat_stream))
        .route("/api/chat/toggle-thinking", post(toggle_thinking_mode))
        .route("/api/chat/session", post(create_session))
}

// Message form data
#[derive(Deserialize)]
pub struct ChatMessageForm {
    session_id: String,
    message: String,
    reasoning_mode: Option<String>,
    enable_thinking: Option<bool>,
}

// Main message handler
pub async fn handle_message(
    State(app_state): State<AppState>,
    Form(form): Form<ChatMessageForm>,
) -> Result<Html<String>, AppError> {
    tracing::info!("💬 Received message: '{}'", form.message);
    
    // 1. Validate session
    let session = app_state.conversation_service
        .get_session(&form.session_id)
        .await?;
    
    // 2. Apply SmolLM3 chat template
    let formatted_prompt = app_state.template_service
        .format_chat_message(&form, &session)
        .await?;
    
    // 3. Add user message to chat history
    let user_message = ChatMessage {
        id: uuid::Uuid::new_v4().to_string(),
        role: "user".to_string(),
        content: form.message.clone(),
        timestamp: chrono::Utc::now(),
        reasoning_mode: form.reasoning_mode.clone(),
    };
    
    // 4. Render user message immediately
    let user_html = app_state.template_service
        .render_message(&user_message)
        .await?;
    
    // 5. Start async SmolLM3 generation
    tokio::spawn(async move {
        let _ = app_state.smollm3_service
            .generate_with_stream(&form.session_id, &formatted_prompt)
            .await;
    });
    
    Ok(Html(user_html))
}
```

### **3. SmolLM3 Chat Template Formatting**
```rust
use minijinja::{Environment, context};
use candle_transformers::models::quantized_llama::Llama;

pub struct TemplateService {
    env: Environment<'static>,
}

impl TemplateService {
    pub fn new() -> Result<Self> {
        let mut env = Environment::new();
        
        // Load SmolLM3 chat template
        env.add_template("smollm3_chat", include_str!("templates/smollm3_chat.jinja2"))?;
        env.add_template("user_message", include_str!("templates/user_message.html"))?;
        env.add_template("assistant_message", include_str!("templates/assistant_message.html"))?;
        env.add_template("thinking_bubble", include_str!("templates/thinking_bubble.html"))?;
        
        Ok(Self { env })
    }
    
    pub async fn format_chat_message(
        &self,
        form: &ChatMessageForm,
        session: &ConversationSession,
    ) -> Result<String> {
        let template = self.env.get_template("smollm3_chat")?;
        
        // Build messages array for SmolLM3
        let mut messages = session.message_history.clone();
        messages.push(serde_json::json!({
            "role": "user",
            "content": form.message
        }));
        
        // Apply SmolLM3 chat template
        let formatted = template.render(context! {
            messages => messages,
            reasoning_mode => form.reasoning_mode.as_deref().unwrap_or("think"),
            enable_thinking => form.enable_thinking.unwrap_or(true),
            add_generation_prompt => true,
            tools => session.available_tools,
            system_message => session.system_prompt,
        })?;
        
        tracing::debug!("🎯 SmolLM3 formatted prompt: {}", formatted);
        Ok(formatted)
    }
    
    pub async fn render_message(&self, message: &ChatMessage) -> Result<String> {
        let template_name = match message.role.as_str() {
            "user" => "user_message",
            "assistant" => "assistant_message",
            _ => "user_message",
        };
        
        let template = self.env.get_template(template_name)?;
        Ok(template.render(context! {
            message => message,
            show_thinking => message.reasoning_mode == Some("think".to_string()),
        })?)
    }
}
```

### **4. SmolLM3 Service Integration**
```rust
use candle_transformers::models::quantized_llama::{Llama, ModelWeights};
use candle_transformers::generation::LogitsProcessor;
use tokenizers::Tokenizer;

pub struct SmolLM3Service {
    model: Llama,
    tokenizer: Tokenizer,
    logits_processor: LogitsProcessor,
    stream_senders: Arc<Mutex<HashMap<String, UnboundedSender<String>>>>,
}

impl SmolLM3Service {
    pub async fn generate_with_stream(
        &self,
        session_id: &str,
        prompt: &str,
    ) -> Result<()> {
        tracing::info!("🎯 Starting SmolLM3 generation for session: {}", session_id);
        
        // 1. Tokenize with SmolLM3 tokenizer
        let input_ids = self.tokenizer.encode(prompt, false)?
            .get_ids()
            .to_vec();
        
        // 2. Create input tensor
        let input_tensor = Tensor::new(input_ids.as_slice(), &self.device)?
            .unsqueeze(0)?;
        
        // 3. Get stream sender for this session
        let sender = {
            let senders = self.stream_senders.lock().await;
            senders.get(session_id).cloned()
        };
        
        if let Some(sender) = sender {
            // 4. Start generation loop
            let mut tokens = input_ids.clone();
            let mut thinking_mode = false;
            let mut current_text = String::new();
            
            for step in 0..512 {
                // Forward pass through SmolLM3
                let input = if step == 0 {
                    input_tensor.clone()
                } else {
                    Tensor::new(&[tokens[tokens.len()-1]], &self.device)?
                        .unsqueeze(0)?
                };
                
                let logits = self.model.forward(&input, step)?;
                
                // Sample next token
                let next_token = self.logits_processor.sample(&logits)?;
                tokens.push(next_token);
                
                // Decode token to text
                let token_text = self.tokenizer.decode(&[next_token], false)?;
                
                // 5. Handle thinking mode detection
                if token_text.contains("<think>") {
                    thinking_mode = true;
                    let _ = sender.send(json!({
                        "type": "thinking_start",
                        "content": ""
                    }).to_string());
                    continue;
                }
                
                if token_text.contains("</think>") {
                    thinking_mode = false;
                    let _ = sender.send(json!({
                        "type": "thinking_end", 
                        "content": current_text.clone()
                    }).to_string());
                    current_text.clear();
                    continue;
                }
                
                // 6. Stream token to frontend
                current_text.push_str(&token_text);
                
                let message_type = if thinking_mode { "thinking" } else { "response" };
                let _ = sender.send(json!({
                    "type": message_type,
                    "content": token_text,
                    "accumulated": current_text.clone()
                }).to_string());
                
                // Check for stop tokens
                if self.is_stop_token(next_token) {
                    break;
                }
                
                // Small delay for natural typing effect
                tokio::time::sleep(Duration::from_millis(30)).await;
            }
            
            // 7. Signal completion
            let _ = sender.send(json!({
                "type": "complete",
                "final_content": current_text
            }).to_string());
        }
        
        Ok(())
    }
}
```

### **5. Server-Sent Events (SSE) Handler**
```rust
use axum::response::sse::{Event, Sse};
use tokio_stream::wrappers::UnboundedReceiverStream;

pub async fn chat_stream(
    Path(session_id): Path<String>,
    State(app_state): State<AppState>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    tracing::info!("📡 SSE connection established for session: {}", session_id);
    
    // Create channel for this session
    let (sender, receiver) = tokio::sync::mpsc::unbounded_channel();
    
    // Register sender with SmolLM3 service
    {
        let mut senders = app_state.smollm3_service.stream_senders.lock().await;
        senders.insert(session_id.clone(), sender);
    }
    
    // Convert to SSE stream
    let stream = UnboundedReceiverStream::new(receiver)
        .map(|msg| {
            // Parse message and render appropriate template
            let data: serde_json::Value = serde_json::from_str(&msg).unwrap_or_default();
            
            match data["type"].as_str() {
                Some("thinking_start") => {
                    Event::default()
                        .event("thinking_start")
                        .data(r#"<div class="thinking-indicator">🤔 Thinking...</div>"#)
                },
                Some("thinking") => {
                    Event::default()
                        .event("thinking_content")
                        .data(format!(r#"<span class="thinking-text">{}</span>"#, 
                                    data["content"].as_str().unwrap_or("")))
                },
                Some("thinking_end") => {
                    Event::default()
                        .event("thinking_end")
                        .data(r#"<div class="thinking-complete">💡 Reasoning complete</div>"#)
                },
                Some("response") => {
                    Event::default()
                        .event("message")
                        .data(format!(r#"<span class="token">{}</span>"#,
                                    data["content"].as_str().unwrap_or("")))
                },
                Some("complete") => {
                    Event::default()
                        .event("complete")
                        .data(r#"<div class="message-complete"></div>"#)
                },
                _ => Event::default().data("")
            }
        })
        .map(Ok);
    
    Sse::new(stream).keep_alive(
        axum::response::sse::KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text("keep-alive-text")
    )
}
```

### **6. MiniJinja Templates**

#### **SmolLM3 Chat Template (`templates/smollm3_chat.jinja2`)**
```jinja2
{%- set reasoning_mode = reasoning_mode|default("think") -%}

{%- if system_message -%}
<|im_start|>system
{{ system_message }}
<|im_end|>
{%- endif -%}

{%- for message in messages -%}
  {%- if message.role == "user" -%}
<|im_start|>user
{{ message.content }}<|im_end|>
  {%- elif message.role == "assistant" -%}
<|im_start|>assistant
    {%- if reasoning_mode == "think" and enable_thinking -%}
<think>
{{ message.thinking_content|default("") }}
</think>
    {%- endif -%}
{{ message.content }}<|im_end|>
  {%- endif -%}
{%- endfor -%}

{%- if tools -%}
<|im_start|>tools
{%- for tool in tools -%}
<tool><function name="{{ tool.name }}">{{ tool.description }}</function></tool>
{%- endfor -%}
<|im_end|>
{%- endif -%}

{%- if add_generation_prompt -%}
<|im_start|>assistant
  {%- if reasoning_mode == "think" and enable_thinking -%}
<think>
  {%- endif -%}
{%- endif -%}
```

#### **User Message Template (`templates/user_message.html`)**
```html
<div class="message user-message" data-message-id="{{ message.id }}">
  <div class="message-header">
    <span class="role-badge user">You</span>
    <span class="timestamp">{{ message.timestamp|strftime('%H:%M') }}</span>
  </div>
  <div class="message-content">
    {{ message.content }}
  </div>
</div>
```

#### **Assistant Message Template (`templates/assistant_message.html`)**
```html
<div class="message assistant-message" 
     data-message-id="{{ message.id }}"
     hx-ext="sse"
     sse-connect="/api/chat/stream/{{ session_id }}"
     sse-swap="message">
     
  <div class="message-header">
    <span class="role-badge assistant">SmolLM3</span>
    <span class="timestamp">{{ message.timestamp|strftime('%H:%M') }}</span>
    {% if message.reasoning_mode == "think" %}
    <span class="thinking-badge">🧠 Thinking Mode</span>
    {% endif %}
  </div>
  
  {% if show_thinking %}
  <div class="thinking-section">
    <div class="thinking-header">🤔 Reasoning Process</div>
    <div class="thinking-content" id="thinking-{{ message.id }}">
      <!-- Thinking content streams here -->
    </div>
  </div>
  {% endif %}
  
  <div class="message-content" id="response-{{ message.id }}">
    <!-- Response content streams here -->
  </div>
  
  <div class="message-actions">
    <button class="copy-btn" onclick="copyMessage('{{ message.id }}')">📋</button>
    <button class="regenerate-btn" hx-post="/api/chat/regenerate" 
            hx-vals='{"message_id": "{{ message.id }}"}'>🔄</button>
  </div>
</div>
```

### **7. HTMX SSE Integration**
```html
<!-- In main chat page -->
<script>
document.body.addEventListener('htmx:sseMessage', function(e) {
    const data = JSON.parse(e.detail.data);
    const messageId = getCurrentMessageId();
    
    switch(e.detail.type) {
        case 'thinking_start':
            showThinkingIndicator(messageId);
            break;
        case 'thinking_content':
            appendToThinking(messageId, data.content);
            break;
        case 'thinking_end':
            finalizeThinking(messageId);
            break;
        case 'message':
            appendToResponse(messageId, data.content);
            break;
        case 'complete':
            finalizeMessage(messageId);
            break;
    }
});

function showThinkingIndicator(messageId) {
    const thinkingEl = document.getElementById(`thinking-${messageId}`);
    if (thinkingEl) {
        thinkingEl.style.display = 'block';
        thinkingEl.innerHTML = '<div class="thinking-indicator">🤔 Thinking...</div>';
    }
}

function appendToResponse(messageId, content) {
    const responseEl = document.getElementById(`response-${messageId}`);
    if (responseEl) {
        responseEl.innerHTML += content;
        responseEl.scrollIntoView({ behavior: 'smooth', block: 'end' });
    }
}
</script>
```

### **8. Application State & Services**
```rust
#[derive(Clone)]
pub struct AppState {
    pub smollm3_service: Arc<SmolLM3Service>,
    pub template_service: Arc<TemplateService>, 
    pub conversation_service: Arc<ConversationService>,
}

pub async fn create_app() -> Result<Router> {
    // Initialize services
    let smollm3_service = Arc::new(
        SmolLM3Service::new("models/SmolLM3-3B-Q4_K_M.gguf").await?
    );
    
    let template_service = Arc::new(TemplateService::new()?);
    let conversation_service = Arc::new(ConversationService::new());
    
    let app_state = AppState {
        smollm3_service,
        template_service,
        conversation_service,
    };
    
    // Build router
    let app = Router::new()
        .merge(create_chat_routes())
        .nest_service("/static", ServeDir::new("static"))
        .layer(TraceLayer::new_for_http())
        .layer(CorsLayer::permissive())
        .with_state(app_state);
    
    Ok(app)
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::init();
    
    let app = create_app().await?;
    
    let listener = tokio::net::TcpListener::bind("127.0.0.1:3000").await?;
    println!("🚀 SmolLM3 Chat Server running on http://127.0.0.1:3000");
    
    axum::serve(listener, app).await?;
    Ok(())
}
```

## 🎯 **Complete Call Chain Summary**

1. **HTMX Form** → POST `/api/chat/message`
2. **Axum Handler** → Extract form data, validate session
3. **Template Service** → Apply SmolLM3 chat template formatting
4. **SmolLM3 Service** → Official quantized_llama inference
5. **Token Streaming** → Real-time generation with thinking detection
6. **SSE Channel** → Server-sent events to browser
7. **HTMX Updates** → Progressive UI updates with MiniJinja templates
8. **User Experience** → Real-time chat with thinking mode visualization

This architecture provides a complete, production-ready SmolLM3 web chat interface with official Candle integration, real-time streaming, and thinking mode support.
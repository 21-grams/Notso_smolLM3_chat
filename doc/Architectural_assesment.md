# SmolLM3 Official Foundation Migration: Comprehensive File Structure

## 🏗️ **Proposed File Structure for Official Candle Migration**

### **Root Project Structure**
```
stt-chatbot/
├── Cargo.toml
├── src/
│   ├── main.rs                               # Application entry point
│   ├── lib.rs                                # Library exports  
│   ├── error.rs                              # Centralized error handling
│   ├── config.rs                             # Application configuration
│   │
│   ├── handlers/                             # 🎯 SIMPLIFIED HANDLERS
│   │   ├── mod.rs
│   │   ├── chat.rs                          # Single unified chat handler
│   │   └── health.rs                        # Health checks
│   │
│   ├── services/                             # 🔥 CORE SERVICES
│   │   ├── mod.rs
│   │   │
│   │   ├── ml/                              # AI Services
│   │   │   ├── mod.rs
│   │   │   ├── service.rs                   # Unified ML service (streaming built-in)
│   │   │   ├── buffer.rs                    # Token buffer management
│   │   │   ├── events.rs                    # Chat event types
│   │   │   │
│   │   │   ├── official/                    # Official Candle Foundation  
│   │   │   │   ├── mod.rs
│   │   │   │   ├── model.rs                 # quantized_llama wrapper
│   │   │   │   ├── config.rs                # SmolLM3Config
│   │   │   │   ├── loader.rs                # GGUF loading
│   │   │   │   └── device.rs                # Device management
│   │   │   │
│   │   │   └── smollm3/                     # SmolLM3 Extensions
│   │   │       ├── mod.rs  
│   │   │       ├── adapter.rs               # Bridge official → SmolLM3
│   │   │       ├── thinking.rs              # Thinking mode detection
│   │   │       ├── tokenizer.rs             # SmolLM3 tokenizer
│   │   │       └── generation.rs            # Generation with streaming
│   │   │
│   │   ├── session/                         # Session Management
│   │   │   ├── mod.rs
│   │   │   ├── manager.rs                   # Session lifecycle
│   │   │   └── broadcaster.rs               # SSE event broadcasting
│   │   │
│   │   └── template/                        # Template Rendering
│   │       ├── mod.rs
│   │       ├── engine.rs                    # MiniJinja wrapper
│   │       └── chat.rs                      # SmolLM3 chat template
│   │
│   ├── models/                              # Data Types
│   │   ├── mod.rs
│   │   ├── chat.rs                          # Chat types
│   │   ├── session.rs                       # Session types
│   │   └── events.rs                        # Event types
│   │
│   └── utils/                               # Utilities
│       ├── mod.rs
│       ├── device.rs                        # Device detection
│       └── performance.rs                   # Performance monitoring
│
├── templates/                               # MiniJinja Templates
│   ├── base.html                           # Base layout
│   ├── chat.html                           # Single chat page
│   └── smollm3_chat.jinja2                 # SmolLM3 template
│
├── static/                                 # Static assets
├── models/                                 # AI model files
└── tests/                                  # Tests
```

## 📋 **Detailed File Breakdown**

### **1. Official Candle Foundation Layer**

#### **`src/services/ml/official/model.rs`**
```rust
//! Official quantized_llama wrapper with SmolLM3 configuration

use candle_transformers::models::quantized_llama::{Llama, ModelWeights};
use candle_core::{Device, Tensor, Result};
use crate::services::ml::official::config::SmolLM3Config;

/// Wrapper around official Candle Llama model
pub struct OfficialSmolLM3Model {
    model: Llama,
    config: SmolLM3Config,
    device: Device,
}

impl OfficialSmolLM3Model {
    /// Load model using official Candle patterns
    pub async fn load(
        weights: &ModelWeights,
        config: SmolLM3Config,
        device: &Device,
    ) -> Result<Self> {
        let llama_config = config.to_llama_config();
        let model = Llama::load(weights, &llama_config, device)?;
        
        Ok(Self {
            model,
            config,
            device: device.clone(),
        })
    }
    
    /// Forward pass using official implementation
    pub fn forward(&mut self, input_ids: &Tensor, position: usize) -> Result<Tensor> {
        self.model.forward(input_ids, position)
    }
    
    /// Get model configuration
    pub fn config(&self) -> &SmolLM3Config {
        &self.config
    }
}
```

#### **`src/services/ml/official/config.rs`**
```rust
//! SmolLM3 configuration extending official LlamaConfig

use candle_transformers::models::quantized_llama::LlamaConfig;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmolLM3Config {
    /// Base LlamaConfig for official compatibility
    pub base: LlamaConfig,
    
    /// SmolLM3-specific features
    pub nope_layers: Vec<usize>,              // [3,7,11,15,19,23,27,31,35]
    pub thinking_tokens: ThinkingTokens,      // <think>, </think>
    pub reasoning_mode: ReasoningMode,        // think/no_think
    pub tool_calling: ToolCallingConfig,      // Tool execution settings
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinkingTokens {
    pub start: u32,    // <think> = 128002
    pub end: u32,      // </think> = 128003
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReasoningMode {
    Think,
    NoThink,
    Adaptive,
}

impl Default for SmolLM3Config {
    fn default() -> Self {
        Self {
            base: LlamaConfig {
                vocab_size: 128256,
                hidden_size: 2048,
                n_layer: 36,
                n_head: 16,
                n_kv_head: 4,           // GQA 4:1 ratio
                intermediate_size: 11008,
                max_seq_len: 65536,     // Extended context
                rope_theta: 2000000.0,  // 2M theta
                rms_norm_eps: 1e-5,
            },
            nope_layers: vec![3, 7, 11, 15, 19, 23, 27, 31, 35],
            thinking_tokens: ThinkingTokens {
                start: 128002,
                end: 128003,
            },
            reasoning_mode: ReasoningMode::Think,
            tool_calling: ToolCallingConfig::default(),
        }
    }
}

impl SmolLM3Config {
    /// Convert to official LlamaConfig
    pub fn to_llama_config(&self) -> LlamaConfig {
        self.base.clone()
    }
    
    /// Check if layer should skip RoPE (NoPE layer)
    pub fn is_nope_layer(&self, layer_idx: usize) -> bool {
        self.nope_layers.contains(&layer_idx)
    }
}
```

#### **`src/services/ml/official/loader.rs`**
```rust
//! Official GGUF loading using Candle patterns

use candle_transformers::models::quantized_llama::ModelWeights;
use candle_core::quantized::gguf_file;
use candle_core::{Device, Result};
use std::path::Path;

pub struct OfficialLoader;

impl OfficialLoader {
    /// Load GGUF using official Candle patterns
    pub async fn load_gguf<P: AsRef<Path>>(
        path: P,
        device: &Device,
    ) -> Result<ModelWeights> {
        tracing::info!("🚀 Loading GGUF with official Candle patterns");
        
        let mut file = std::fs::File::open(path)?;
        let content = gguf_file::Content::read(&mut file)?;
        
        tracing::info!("📊 GGUF loaded: {} tensors, {} metadata entries",
                      content.tensor_infos.len(),
                      content.metadata.len());
        
        let weights = ModelWeights::from_gguf(content, &mut file, device)?;
        
        tracing::info!("✅ Official model weights loaded successfully");
        Ok(weights)
    }
    
    /// Validate GGUF file before loading
    pub fn validate_gguf<P: AsRef<Path>>(path: P) -> Result<()> {
        let path = path.as_ref();
        
        if !path.exists() {
            candle_core::bail!("GGUF file not found: {:?}", path);
        }
        
        let metadata = std::fs::metadata(path)?;
        let size_gb = metadata.len() as f64 / (1024.0 * 1024.0 * 1024.0);
        
        if size_gb < 0.1 {
            candle_core::bail!("GGUF file too small: {:.2} GB", size_gb);
        }
        
        tracing::info!("✅ GGUF validation passed: {:.2} GB", size_gb);
        Ok(())
    }
}
```

### **2. SmolLM3 Extension Layer**

#### **`src/services/ml/smollm3/adapter.rs`**
```rust
//! Bridge between official Candle and SmolLM3 features

use crate::services::ml::official::{OfficialSmolLM3Model, SmolLM3Config};
use candle_core::{Tensor, Result, Device};
use super::nope_layers::NopeHandler;
use super::thinking::ThinkingDetector;

pub struct SmolLM3Adapter {
    model: OfficialSmolLM3Model,
    nope_handler: NopeHandler,
    thinking_detector: ThinkingDetector,
}

impl SmolLM3Adapter {
    pub fn new(model: OfficialSmolLM3Model) -> Self {
        let config = model.config().clone();
        
        Self {
            nope_handler: NopeHandler::new(config.nope_layers.clone()),
            thinking_detector: ThinkingDetector::new(config.thinking_tokens),
            model,
        }
    }
    
    /// Forward pass with SmolLM3 extensions
    pub fn forward_with_extensions(
        &mut self,
        input_ids: &Tensor,
        position: usize,
    ) -> Result<Tensor> {
        // Check if current layer needs NoPE handling
        let layer_idx = position % self.model.config().base.n_layer;
        
        if self.nope_handler.should_skip_rope(layer_idx) {
            // Custom handling for NoPE layers
            self.forward_nope_layer(input_ids, layer_idx)
        } else {
            // Standard official forward pass
            self.model.forward(input_ids, position)
        }
    }
    
    fn forward_nope_layer(&mut self, input_ids: &Tensor, layer_idx: usize) -> Result<Tensor> {
        // Implementation depends on how official Candle exposes layer-level control
        // This might require extending the official API or using composition
        self.model.forward(input_ids, layer_idx)
    }
}
```

#### **`src/services/ml/smollm3/thinking.rs`**
```rust
//! Thinking mode detection and processing

use crate::models::events::ThinkingEvent;
use tokenizers::Tokenizer;

pub struct ThinkingDetector {
    start_token: u32,
    end_token: u32,
}

impl ThinkingDetector {
    pub fn new(tokens: crate::services::ml::official::config::ThinkingTokens) -> Self {
        Self {
            start_token: tokens.start,
            end_token: tokens.end,
        }
    }
    
    /// Detect thinking mode transitions in token stream
    pub fn process_token(&mut self, token: u32, tokenizer: &Tokenizer) -> Option<ThinkingEvent> {
        if token == self.start_token {
            Some(ThinkingEvent::Start)
        } else if token == self.end_token {
            Some(ThinkingEvent::End)
        } else {
            None
        }
    }
    
    /// Check if text contains thinking markers
    pub fn contains_thinking_markers(&self, text: &str) -> bool {
        text.contains("<think>") || text.contains("</think>")
    }
}
```

#### **`src/services/ml/smollm3/generation.rs`**
```rust
//! SmolLM3 generation pipeline with streaming

use crate::services::ml::official::OfficialSmolLM3Model;
use crate::services::ml::streaming::events::GenerationEvent;
use candle_transformers::generation::LogitsProcessor;
use tokenizers::Tokenizer;
use tokio::sync::mpsc::UnboundedSender;

pub struct SmolLM3Generator {
    model: OfficialSmolLM3Model,
    tokenizer: Tokenizer,
    logits_processor: LogitsProcessor,
}

impl SmolLM3Generator {
    pub fn new(
        model: OfficialSmolLM3Model,
        tokenizer: Tokenizer,
        logits_processor: LogitsProcessor,
    ) -> Self {
        Self {
            model,
            tokenizer,
            logits_processor,
        }
    }
    
    /// Generate with streaming support
    pub async fn generate_stream(
        &mut self,
        prompt: &str,
        sender: UnboundedSender<GenerationEvent>,
        max_tokens: usize,
    ) -> anyhow::Result<String> {
        // 1. Tokenize input
        let input_ids = self.tokenizer.encode(prompt, false)?
            .get_ids()
            .to_vec();
        
        // 2. Generation loop with streaming
        let mut tokens = input_ids.clone();
        let mut thinking_mode = false;
        let mut accumulated_text = String::new();
        
        for step in 0..max_tokens {
            // Forward pass
            let input_tensor = self.create_input_tensor(&tokens, step)?;
            let logits = self.model.forward(&input_tensor, step)?;
            
            // Sample next token
            let next_token = self.logits_processor.sample(&logits)?;
            tokens.push(next_token);
            
            // Decode and process
            let token_text = self.tokenizer.decode(&[next_token], false)?;
            
            // Handle thinking mode
            if let Some(event) = self.process_thinking_transition(&token_text, &mut thinking_mode) {
                let _ = sender.send(event);
                continue;
            }
            
            // Stream token
            accumulated_text.push_str(&token_text);
            let event = if thinking_mode {
                GenerationEvent::ThinkingToken(token_text)
            } else {
                GenerationEvent::ResponseToken(token_text)
            };
            
            let _ = sender.send(event);
            
            // Check stop conditions
            if self.is_stop_token(next_token) {
                break;
            }
        }
        
        let _ = sender.send(GenerationEvent::Complete);
        Ok(accumulated_text)
    }
    
    fn create_input_tensor(&self, tokens: &[u32], step: usize) -> Result<Tensor> {
        use candle_core::{Tensor, Device};
        
        let device = self.model.device();
        if step == 0 {
            // Prefill: entire sequence
            Tensor::new(tokens, device)?.unsqueeze(0)
        } else {
            // Generation: only last token
            Tensor::new(&[tokens[tokens.len()-1]], device)?.unsqueeze(0)
        }
    }
    
    fn process_thinking_transition(
        &self,
        token_text: &str,
        thinking_mode: &mut bool,
    ) -> Option<GenerationEvent> {
        if token_text.contains("<think>") {
            *thinking_mode = true;
            Some(GenerationEvent::ThinkingStart)
        } else if token_text.contains("</think>") {
            *thinking_mode = false;
            Some(GenerationEvent::ThinkingEnd)
        } else {
            None
        }
    }
}
```

### **3. Streaming Layer**

#### **`src/services/ml/streaming/events.rs`**
```rust
//! Event types for real-time streaming

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum GenerationEvent {
    /// Generation started
    Start,
    
    /// Thinking mode started
    ThinkingStart,
    
    /// Token generated during thinking
    ThinkingToken(String),
    
    /// Thinking mode ended
    ThinkingEnd,
    
    /// Regular response token
    ResponseToken(String),
    
    /// Generation completed
    Complete,
    
    /// Error occurred
    Error(String),
}

impl GenerationEvent {
    /// Convert to SSE-compatible format
    pub fn to_sse_event(&self) -> String {
        serde_json::to_string(self).unwrap_or_default()
    }
    
    /// Get event type name for SSE
    pub fn event_name(&self) -> &'static str {
        match self {
            GenerationEvent::Start => "start",
            GenerationEvent::ThinkingStart => "thinking_start",
            GenerationEvent::ThinkingToken(_) => "thinking_token",
            GenerationEvent::ThinkingEnd => "thinking_end",
            GenerationEvent::ResponseToken(_) => "response_token",
            GenerationEvent::Complete => "complete",
            GenerationEvent::Error(_) => "error",
        }
    }
}
```

### **4. High-Level Service Interface**

#### **`src/services/ml/service.rs`**
```rust
//! High-level ML service orchestrating all components

use crate::services::ml::{
    official::{OfficialLoader, SmolLM3Config},
    smollm3::{SmolLM3Adapter, SmolLM3Generator},
    streaming::events::GenerationEvent,
};
use candle_core::Device;
use tokenizers::Tokenizer;
use tokio::sync::mpsc::UnboundedSender;
use std::sync::Arc;

pub struct MLService {
    generator: SmolLM3Generator,
    config: SmolLM3Config,
    device: Device,
}

impl MLService {
    /// Initialize ML service with official foundation
    pub async fn new(model_path: &str, tokenizer_path: &str) -> anyhow::Result<Self> {
        tracing::info!("🚀 Initializing ML service with official Candle foundation");
        
        // 1. Device detection
        let device = crate::utils::device::detect_optimal_device()?;
        tracing::info!("🎮 Using device: {:?}", device);
        
        // 2. Load configuration
        let config = SmolLM3Config::default();
        
        // 3. Official GGUF loading
        OfficialLoader::validate_gguf(model_path)?;
        let weights = OfficialLoader::load_gguf(model_path, &device).await?;
        
        // 4. Create official model
        let official_model = crate::services::ml::official::OfficialSmolLM3Model::load(
            &weights,
            config.clone(),
            &device,
        ).await?;
        
        // 5. Load tokenizer
        let tokenizer = Tokenizer::from_file(tokenizer_path)?;
        
        // 6. Create logits processor
        let logits_processor = candle_transformers::generation::LogitsProcessor::new(
            42,           // seed
            Some(0.7),    // temperature
            Some(0.9),    // top_p
        );
        
        // 7. Create generator
        let generator = SmolLM3Generator::new(official_model, tokenizer, logits_processor);
        
        tracing::info!("✅ ML service initialized successfully");
        
        Ok(Self {
            generator,
            config,
            device,
        })
    }
    
    /// Generate response with streaming
    pub async fn generate_stream(
        &mut self,
        prompt: &str,
        sender: UnboundedSender<GenerationEvent>,
    ) -> anyhow::Result<String> {
        self.generator.generate_stream(prompt, sender, 512).await
    }
    
    /// Get model configuration
    pub fn config(&self) -> &SmolLM3Config {
        &self.config
    }
}
```

## 🎯 **Migration Implementation Plan**

### **Phase 1: Official Foundation (Days 1-3)**
1. **Replace** `src/services/ml/smollm3/model.rs` with `official/model.rs`
2. **Update** dependencies to use official Candle patterns
3. **Test** basic inference with `bin/test_official.rs`
4. **Validate** performance improvements

### **Phase 2: SmolLM3 Extensions (Days 4-6)**
1. **Implement** `smollm3/adapter.rs` for NoPE layers
2. **Add** thinking mode detection in `smollm3/thinking.rs`
3. **Create** generation pipeline in `smollm3/generation.rs`
4. **Test** SmolLM3-specific features

### **Phase 3: Streaming Integration (Days 7-9)**
1. **Implement** SSE event system in `streaming/events.rs`
2. **Add** streaming pipeline in `streaming/pipeline.rs`
3. **Create** connection manager in `streaming/manager.rs`
4. **Test** real-time streaming

### **Phase 4: Web Integration (Days 10-14)**
1. **Update** handlers to use new ML service
2. **Integrate** template system with new events
3. **Test** complete web interface
4. **Performance optimization**

## ✅ **Expected Outcomes**

- **✅ Immediate error resolution** - Official patterns fix shape mismatch
- **✅ 50-100x performance improvement** - Direct quantized operations
- **✅ Maintainable architecture** - Clear separation of concerns
- **✅ Future-proof design** - Built on official Candle foundation
- **✅ Production readiness** - Comprehensive error handling and monitoring

This file structure provides a clear migration path from custom implementation to official Candle foundation while preserving SmolLM3-specific features as focused extensions.

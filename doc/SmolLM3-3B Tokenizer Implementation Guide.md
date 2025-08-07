# SmolLM3-3B Tokenizer Implementation Guide

This guide outlines the requirements and implementation details for creating a tokenizer compatible with the **SmolLM3-3B** model, a 3-billion parameter language model by Hugging Face, optimized for multilingual support (up to 128k tokens with YaRN) and Q4_K_M GGUF quantization. It includes the chat template extracted from the provided images and instructions for integrating it with **MiniJinja2** (a Rust-based Jinja2 templating engine) and **HTMX** (for dynamic HTML updates) in a web application.

## Tokenizer Requirements

- **Model Source**: Based on the `HuggingFaceTB/SmolLM3-3B` repository on Hugging Face.
- **Tokenizer Files**: Requires `tokenizer.json`, `tokenizer_config.json`, and `special_tokens_map.json` from the repository.
  - `tokenizer.json`: Contains the SentencePiece vocabulary (~128k tokens) and token-to-ID mappings, supporting multilingual text (English, French, Spanish, German, Italian, Portuguese, and partially Arabic, Chinese, Russian).
  - `tokenizer_config.json`: Defines special tokens (`<s>`, `</s>`, `<pad>`, `<unk>`), a Jinja2 chat template, and settings for 128k token context length.
  - `special_tokens_map.json`: Maps special tokens to their string representations, including potential tool-calling tokens (`<tool_call>`, `<code>`).
- **GGUF Compatibility**: Must produce token IDs compatible with the Q4_K_M quantization (4-bit with super-blocks of 8 blocks, 32 weights each, scales, and mins quantized with 6 bits).
- **Candle Ecosystem**: Uses `candle-core 0.9.1` and `tokenizers 0.21.0` for Rust-based implementation, with optional CUDA support.
- **Performance**: Optimized for parallel processing and GPU acceleration, supporting batch encoding for efficiency.

## Extracted Chat Template

The following Jinja2-style chat template was extracted from the provided images. It formats messages for the SmolLM3-3B model, supporting system instructions, user inputs, assistant responses, reasoning modes (`think`/`no_think`), tool calling, and metadata.

```jinja2
{%- set defaults %}
  {%- if enable_thinking is not defined %}
    {%- set enable_thinking = true %}
  {%- endif -%}
{%- endset -%}

{%- set reasoning_mode %}
  {%- if enable_thinking %}
    {%- set reasoning_mode = "think" %}
  {%- else %}
    {%- set reasoning_mode = "no_think" %}
  {%- endif -%}
{%- endset -%}

{%- set header (system_message) %}
  {%- if messages|length == 0 and system_message is not defined %}
    {%- set system_message = "" %}
  {%- endif %}
  {%- if system_message is defined and system_message|content %}
    {%- set reasoning_mode = "no_think" %}
    {%- set thinking_in_system = reasoning_mode == "think" %}
    {%- if thinking_in_system %}
      {%- set system_instructions = system_message|replace("no_think", "")|replace("think", "")|strip %}
    {%- else %}
      {%- set system_instructions = system_message|replace("no_think", "")|replace("think", "")|strip %}
    {%- endif %}
    {%- if "system_override" in system_message %}
      {{ "<|im_start|>system\n" ~ system_instructions ~ "\n<|im_end|>" }}
    {%- else %}
      {{ "<|im_start|>system\n" ~ system_instructions ~ "\n<|im_end|>" }}
    {%- endif %}
  {%- endif %}
{%- endset -%}

{%- for message in messages %}
  {%- set content = message.content if message.content is string else "" %}
  {%- if message.role == "user" %}
    {{ "<|im_start|>user\n" ~ content ~ "<|im_end|>" }}
  {%- elif message.role == "assistant" %}
    {%- if reasoning_mode == "think" %}
      {{ "<|im_start|>assistant\n" ~ content|strip("\n") ~ "<|im_end|>" }}
    {%- else %}
      {{ "<|im_start|>assistant\n" ~ "<think>\n" ~ content|strip("\n") ~ "\n</think>\n" ~ content|strip("\n") ~ "<|im_end|>" }}
    {%- endif %}
  {%- endif %}
{%- endfor -%}

{%- if add_generation_prompt %}
  {%- if reasoning_mode == "think" %}
    {{ "<|im_start|>assistant\n" ~ "<think>\n" ~ "<|im_end|>" }}
  {%- else %}
    {{ "<|im_start|>assistant\n" ~ "<|im_end|>" }}
  {%- endif %}
{%- endif -%}

{%- if tools is defined %}
  {%- set ns = namespace(xml_tool_string="", python_tool_string="") %}
  {%- for tool in tools %}
    {%- if tool.type == "function" %}
      {%- set ns.xml_tool_string = ns.xml_tool_string + "<tool><function name=\"" + tool.function.name + "\">" + tool.function.description + "</function></tool>" %}
      {%- set ns.python_tool_string = ns.python_tool_string + "def " + tool.function.name + "(**kwargs):\n    '''" + tool.function.description + "'''\n    pass\n" %}
    {%- endif %}
  {%- endfor %}
  {%- if ns.xml_tool_string %}
    {{ "<|im_start|>tools\n" ~ ns.xml_tool_string ~ "<|im_end|>" }}
  {%- elif ns.python_tool_string %}
    {{ "<|im_start|>tools\n" ~ ns.python_tool_string ~ "<|im_end|>" }}
  {%- else %}
    {{ "" }}
  {%- endif %}
{%- endif -%}

{%- if custom_instructions is defined %}
  {%- set custom_instructions = custom_instructions + "\n" %}
  {{ "<|im_start|>system\n" ~ custom_instructions ~ "\n<|im_end|>" }}
{%- endif -%}

{%- set Metadata %}
  {%- set Knowledge_Cutoff_Date = "June 2025" %}
  {%- set today = strftime("%d %b %Y") %}
  {%- set Today_Date = today + " " + reasoning_mode %}
  {{ "<|im_start|>system\n" ~ "Knowledge Cutoff Date: " ~ Knowledge_Cutoff_Date ~ "\nToday Date: " ~ Today_Date ~ "\n<|im_end|>" }}
{%- endset -%}
```

- **Notes**: 
  - Current date (August 07, 2025, 12:29 PM CST) renders `today` as `07 Aug 2025`, so `Today_Date` would be `07 Aug 2025 think` if `reasoning_mode` is `"think"`.
  - Supports dual-mode reasoning (`think`/`no_think`) and tool calling (XML or Python).

## Working with the Chat Template Using MiniJinja2 and HTMX

### Prerequisites
- **Rust Environment**: Install Rust (1.75+), `cargo`, and dependencies (`candle-core 0.9.1`, `tokenizers 0.21.0`, `minijinja`, `axum`, `htmx`).
- **HTMX**: Include the HTMX library via CDN in your HTML (`<script src="https://unpkg.com/htmx.org@1.9.10"></script>`).

### Implementation Steps

1. **Set Up the Project**
   - Create a new Rust project with `cargo new smollm3-web` and configure `Cargo.toml`:
     ```toml
     [package]
     name = "smollm3-web"
     version = "0.1.0"
     edition = "2021"

     [dependencies]
     candle-core = { version = "0.9.1", features = ["cuda"] }
     tokenizers = { version = "0.21.0", features = ["serde"] }
     hf-hub = { version = "0.3.2" }
     minijinja = { version = "1.0" }
     axum = { version = "0.7" }
     serde = { version = "1.0", features = ["derive"] }
     serde_json = "1.0"
     tokio = { version = "1.0", features = ["full"] }
     ```

2. **Load Tokenizer and Chat Template**
   - Use `hf-hub` to download `tokenizer.json` and `tokenizer_config.json` from `HuggingFaceTB/SmolLM3-3B`.
   - Extract the chat template from `tokenizer_config.json` and load it into MiniJinja2.

3. **Define the Tokenizer Wrapper**
   - Create a struct to handle tokenization and chat template rendering:
     ```rust
     use candle_core::{Device, Tensor};
     use tokenizers::Tokenizer;
     use minijinja::Environment;
     use anyhow::Result;

     struct TokenizerWrapper {
         tokenizer: Tokenizer,
         jinja: Environment<'static>,
     }

     impl TokenizerWrapper {
         fn new(tokenizer_path: &str, template: &str) -> Result<Self> {
             let tokenizer = Tokenizer::from_file(tokenizer_path)?;
             let mut jinja = Environment::new();
             jinja.add_template("chat", template)?;
             Ok(TokenizerWrapper { tokenizer, jinja })
         }

         fn apply_chat_template(&self, messages: Vec<serde_json::Value>) -> Result<String> {
             let ctx = minijinja::context! {
                 messages => messages,
                 enable_thinking => true,
                 add_generation_prompt => true,
             };
             Ok(self.jinja.get_template("chat")?.render(ctx)?)
         }

         fn encode(&self, text: &str) -> Result<Vec<u32>> {
             Ok(self.tokenizer.encode(text, true)?.get_ids().to_vec())
         }
     }
     ```

4. **Set Up Web Server with Axum and HTMX**
   - Create an Axum server to handle HTTP requests and render responses dynamically with HTMX.
   - Example `main.rs`:
     ```rust
     use axum::{
         routing::get,
         Router, response::Html,
     };
     use std::net::SocketAddr;

     #[tokio::main]
     async fn main() {
         let app = Router::new()
             .route("/", get(handler))
             .route("/submit", get(submit_handler));

         let addr = SocketAddr::from(([127.0.0.1], 3000));
         println!("Server running at http://{}", addr);
         axum::Server::bind(&addr)
             .serve(app.into_make_service())
             .await
             .unwrap();
     }

     async fn handler() -> Html<String> {
         Html(include_str!("index.html").to_string())
     }

     async fn submit_handler() -> Html<String> {
         Html("<div hx-swap-oob='innerHTML'>Response from server</div>".to_string())
     }
     ```

5. **Create HTML with HTMX**
   - Use HTMX for real-time updates. Example `index.html`:
     ```html
     <!DOCTYPE html>
     <html>
     <head>
       <script src="https://unpkg.com/htmx.org@1.9.10"></script>
       <title>SmolLM3-3B Chat</title>
     </head>
     <body>
       <div id="chat">
         <div><strong>System:</strong> You are a helpful AI assistant.</div>
       </div>
       <form hx-get="/submit" hx-target="#chat" hx-swap="beforeend">
         <input type="text" name="message" placeholder="Type your message...">
         <button type="submit">Send</button>
       </form>
     </body>
     </html>
     ```

6. **Integrate Chat Template and Tokenization**
   - On form submission, apply the chat template to the message list, tokenize it, and update the UI.
   - Example handler modification:
     ```rust
     use axum::extract::Query;
     use serde::Deserialize;

     #[derive(Deserialize)]
     struct SubmitParams {
         message: String,
     }

     async fn submit_handler(Query(params): Query<SubmitParams>) -> Html<String> {
         let messages = vec![serde_json::json!({
             "role": "user",
             "content": params.message
         })];
         let template = include_str!("chat_template.jinja2"); // Saved separately
         let tokenizer = TokenizerWrapper::new("tokenizer.json", template).unwrap();
         let formatted = tokenizer.apply_chat_template(messages).unwrap();
         let tokens = tokenizer.encode(&formatted).unwrap();
         Html(format!("<div hx-swap-oob='innerHTML'><strong>Assistant:</strong> Tokens: {}</div>", tokens.len()))
     }
     ```

7. **Run the Application**
   - Save the chat template as `chat_template.jinja2`.
   - Compile and run: `cargo run --release`.
   - Access `http://127.0.0.1:3000` to interact with the chat interface. HTMX will update the chat div dynamically.

## Additional Notes
- **MiniJinja2**: Provides a lightweight Jinja2 implementation for Rust, suitable for rendering the chat template with dynamic data.
- **HTMX**: Enables server-side rendering with client-side interactivity, reducing JavaScript overhead.
- **Performance**: Use `RUSTFLAGS="-Ctarget-cpu=native"` for CPU optimization and enable CUDA for GPU support.
- **Testing**: Verify multilingual support and 128k token context handling with diverse inputs.

This guide provides a foundation for building a web-based SmolLM3-3B tokenizer interface. Extend it with model inference using Candle for full functionality.

use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::ffi::{CStr, CString};
use std::fs;
use std::io::{self, BufReader, BufRead, Write, BufWriter};
use std::mem;
use std::os::raw::c_void;
use std::os::unix::net::UnixListener;
use std::path::Path;
use std::ptr;
use std::slice;
use std::str;
use clap::{Command, Arg, ArgMatches, value_parser};
use serde::{Serialize, Deserialize};
use serde_json;
use llama_server_rs::{
    LlamaModel, default_model_params, LlamaContext, default_context_params, LlamaBatch, LlamaToken,
    LlamaTokenData, LlamaTokenDataArray,
};
use llama_server_rs::ffi as ffi;

fn parse_args() -> ArgMatches {
    Command::new("llama-server-rs")
        .arg(Arg::new("model").short('m').long("model"))
        .arg(Arg::new("ctx_size").short('c').long("ctx-size")
            .value_parser(value_parser!(usize))
            .default_value("4096"))
        .get_matches()
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Request<'a> {
    Completion(CompletionRequest<'a>),
    BatchCompletion(BatchCompletionRequest<'a>),
    HiddenStates(HiddenStatesRequest<'a>),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct CompletionRequest<'a> {
    prompt: Cow<'a, str>,
    #[serde(default = "const_usize::<128>")]
    n_predict: usize,
    control_vectors: Option<Cow<'a, [ControlVector<'a>]>>,
}

#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
struct ControlVector<'a> {
    #[serde(alias = "fname")]
    name: Cow<'a, str>,
    strength: f32,
    layer_start: Option<usize>,
    layer_end: Option<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct BatchCompletionRequest<'a> {
    prompt: Cow<'a, str>,
    #[serde(default = "const_usize::<128>")]
    n_predict: usize,
    control_vectors: Option<Cow<'a, [ControlVector<'a>]>>,
    batch_size: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct HiddenStatesRequest<'a> {
    prompts: Cow<'a, [Cow<'a, str>]>,
}

fn const_usize<const N: usize>() -> usize { N }


#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Response<'a> {
    Completion(CompletionResponse<'a>),
    BatchCompletion(BatchCompletionResponse<'a>),
    HiddenStates(HiddenStatesResponse),
    Error(ErrorResponse<'a>),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct CompletionResponse<'a> {
    content: Cow<'a, str>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct BatchCompletionResponse<'a> {
    content: Cow<'a, [String]>,
}

/// Response header giving hidden state vectors for a set of prompts.  The JSON response header is
/// followed by a payload consisting of `n_prompt * n_layer * n_embd` 32-bit floats, with the float
/// values sent in little-endian order.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct HiddenStatesResponse {
    /// Number of prompts for which hidden states are provided.  This is the same as
    /// `prompts.len()` from the request.
    n_prompt: usize,
    /// Number of layers in the model.  For each prompt, the payload contains a vector for each
    /// layer of the model.
    n_layer: usize,
    /// Number of values in each hidden state vector.
    n_embd: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ErrorResponse<'a> {
    msg: Cow<'a, str>,
}

impl<'a> From<CompletionResponse<'a>> for Response<'a> {
    fn from(x: CompletionResponse<'a>) -> Response<'a> {
        Response::Completion(x)
    }
}

impl<'a> From<BatchCompletionResponse<'a>> for Response<'a> {
    fn from(x: BatchCompletionResponse<'a>) -> Response<'a> {
        Response::BatchCompletion(x)
    }
}

impl<'a> From<HiddenStatesResponse> for Response<'a> {
    fn from(x: HiddenStatesResponse) -> Response<'a> {
        Response::HiddenStates(x)
    }
}

impl<'a> From<ErrorResponse<'a>> for Response<'a> {
    fn from(x: ErrorResponse<'a>) -> Response<'a> {
        Response::Error(x)
    }
}


#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum HiddenStatesStreamingResponse<'a> {
    Done,
    Error(ErrorResponse<'a>),
    #[serde(untagged)]
    Chunk(HiddenStatesChunkHeader<'a>),
}

/// Chunk header used when streaming hidden state vectors.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct HiddenStatesChunkHeader<'a> {
    /// The layer whose data is contained in this chunk.
    #[serde(rename = "l")]
    layer: usize,
    /// The prompts whose data is contained in this chunk.
    #[serde(rename = "p")]
    prompts: Cow<'a, [usize]>,
}


fn write_json(socket: &mut impl Write, x: &impl Serialize) -> io::Result<()> {
    serde_json::to_writer(&mut *socket, x)?;
    socket.write_all(b"\n")?;
    socket.flush()?;
    Ok(())
}

/// Try to send `resp` on `socket`.  Returns `Err` if this fails.
fn try_send_message(socket: impl Write, resp: &impl Serialize) -> Result<(), String> {
    let mut socket = BufWriter::new(socket);
    write_json(&mut socket, &resp)
        .map_err(|e| format!("error sending response: {}", e))?;
    Ok(())
}

/// Send `resp` on `socket`.  This first tries to send `resp`; if that fails, it tries to send an
/// error message; and if that also fails, it gives up and returns without  doing anything.
fn send_message(socket: impl Write, resp: &impl Serialize) {
    let mut socket = BufWriter::new(socket);

    let err = match write_json(&mut socket, &resp) {
        Ok(_) => return,
        Err(e) => format!("error sending response: {}", e),
    };
    let err_resp = Response::Error(ErrorResponse {
        msg: (&err).into(),
    });
    match write_json(&mut socket, &err_resp) {
        Ok(_) => return,
        Err(e) => {
            // Give up
            eprintln!("response dropped: {}; {}", err, e);
            return;
        },
    }
}

fn send_response(socket: impl Write, resp: &Response) {
    send_message(socket, resp);
}


struct ServerContext<'a, 'b> {
    model: &'a LlamaModel,
    ctx: &'b mut LlamaContext<'a>,
    /// The set of control vectors that are currently loaded into `ctx`.
    loaded_control_vectors: Vec<ControlVector<'static>>,
}

impl<'a, 'b> ServerContext<'a, 'b> {
    pub fn new(
        model: &'a LlamaModel,
        ctx: &'b mut LlamaContext<'a>,
    ) -> ServerContext<'a, 'b> {
        ServerContext {
            model, ctx,
            loaded_control_vectors: Vec::new(),
        }
    }

    fn handle_request(
        &mut self,
        socket: impl Write,
        req: &Request,
    ) -> Result<(), String> {
        match *req {
            Request::Completion(ref cr) =>
                self.handle_completion_request(socket, cr),
            Request::BatchCompletion(ref bcr) =>
                self.handle_batch_completion_request(socket, bcr),
            Request::HiddenStates(ref hsr) =>
                self.handle_hidden_states_request(socket, hsr),
        }
    }

    fn handle_completion_request(
        &mut self,
        socket: impl Write,
        req: &CompletionRequest,
    ) -> Result<(), String> {
        let start = std::time::Instant::now();

        self.ctx.kv_cache_clear();

        if let Some(ref control_vectors) = req.control_vectors {
            self.update_control_vectors(control_vectors)?;
        }

        // Tokenize and process the prompt.
        let tokens = self.tokenize(&req.prompt)?;
        let batch = batch_with_tokens(&tokens, /* start_pos: */ 0, /* seq_id: */ 0);
        self.ctx.decode(&batch)?;
        drop(batch);

        // Sampling
        let mut token_data = self.empty_token_data();
        let mut token_buf = Vec::with_capacity(64);
        let mut content = String::new();
        let mut batch = LlamaBatch::new(1, 1);
        for offset in 0 .. req.n_predict {
            let pos = tokens.len() + offset;
            let logits_index = if offset == 0 { tokens.len() - 1 } else { 0 };

            let token = self.sample_token(logits_index, &mut token_data);
            let token_str = self.token_to_str(token, &mut token_buf)?;

            eprint!("{}", token_str);
            content.push_str(token_str);

            // TODO: Stop on EOS token

            // Process the newly added token.
            // TODO: Skip this part on the final iteration, since its results aren't used.
            batch.clear();
            batch.push(token, pos, /* seq_id: */ 0, true);
            self.ctx.decode(&batch)?;
        }
        eprintln!();

        let dur = start.elapsed();
        eprintln!("completed {} tokens in {:?}, {:.2} T/s",
            req.n_predict, dur, req.n_predict as f32 / dur.as_secs_f32());

        send_response(socket, &CompletionResponse {
            content: content.into(),
        }.into());

        Ok(())
    }

    fn handle_batch_completion_request(
        &mut self,
        socket: impl Write,
        req: &BatchCompletionRequest,
    ) -> Result<(), String> {
        self.ctx.kv_cache_clear();

        if let Some(ref control_vectors) = req.control_vectors {
            self.update_control_vectors(control_vectors)?;
        }

        // Tokenize and process the prompt.
        let tokens = self.tokenize(&req.prompt)?;
        let batch = batch_with_tokens(&tokens, /* start_pos: */ 0, /* seq_id: */ 0);
        self.ctx.decode(&batch)?;
        drop(batch);

        if req.batch_size > self.ctx.n_seq_max() {
            return Err(format!("batch size of {} is too high; max is {}",
                req.batch_size, self.ctx.n_seq_max()));
        }

        // Populate all sequences in the KV cache with the prompt data.
        for i in 1 .. req.batch_size {
            unsafe {
                // Copy all data from sequence 0 to sequence `i`.
                ffi::llama_kv_cache_seq_cp(
                    self.ctx.as_ptr(),
                    0,
                    i.try_into().unwrap(),
                    -1,
                    -1,
                );
            }
        }


        // Sampling
        let mut token_data = self.empty_token_data();
        let mut token_buf = Vec::with_capacity(64);
        let mut content = (0 .. req.batch_size).map(|_| String::new()).collect::<Vec<_>>();
        let mut batch = LlamaBatch::new(req.batch_size, req.batch_size);
        for offset in 0 .. req.n_predict {
            let pos = tokens.len() + offset;

            batch.clear();

            // Sample tokens and prepare a new batch.
            if offset == 0 {
                // Initially, we sample continuations of the prompt, all using the logits from the
                // last prompt token.
                for (i, s) in content.iter_mut().enumerate() {
                    // TODO: Reuse the same final token_data for each `llama_sample_token` call.
                    let token = self.sample_token(tokens.len() - 1, &mut token_data);
                    let token_str = self.token_to_str(token, &mut token_buf)?;
                    s.push_str(token_str);
                    batch.push(token, pos, /* seq_id: */ i, /* logits: */ true);
                }
            } else {
                // Afterward, we get separate logits for each completion in the batch.
                for (i, s) in content.iter_mut().enumerate() {
                    let token = self.sample_token(i, &mut token_data);
                    let token_str = self.token_to_str(token, &mut token_buf)?;
                    s.push_str(token_str);
                    batch.push(token, pos, /* seq_id: */ i, /* logits: */ true);
                }
            }

            // TODO: Stop on EOS token
            // TODO: Allow some completions in the batch to stop early.

            // Process the newly added tokens.
            self.ctx.decode(&batch)?;
        }

        eprintln!("{} completions:", req.batch_size);
        for s in &content {
            eprintln!("  {}", s);
        }

        send_response(socket, &BatchCompletionResponse {
            content: content.into(),
        }.into());

        Ok(())
    }

    /// Handle a request for hidden states from a set of prompts.
    ///
    /// The response looks like this:
    ///
    /// ```json
    /// {"kind": "hidden_states_response", "n_layer": <n>, ...} \n
    /// {"layer": <i>, "prompts": [<j0>, <j1>, ...]} \n <floating-point data>
    /// {"layer": <i>, "prompts": [<j0>, <j1>, ...]} \n <floating-point data>
    /// {"kind": "done"} \n
    /// ```
    ///
    /// Each JSON object is followed by a newline.  Binary data is *not* followed by a newline.
    ///
    /// The response has three parts:
    ///
    /// 1. A tagged `HiddenStatesResponse` containing the dimensions of the data.
    /// 2. The data itself, transmitted in chunks.  Each chunk is preceded by a
    ///    `HiddenStatesChunkHeader` indicating where it belongs in the output tensor.  The chunk's
    ///    floating-point data consists of `n_embd` floats for prompt `prompts[0]` and layer
    ///    `layer`, then `n_embd` more floats for `prompts[1], layer`, and so on.
    /// 3. A final object containing only `{"kind": "done"}`, indicating that all chunks have been
    ///    transmitted.
    ///
    /// Any of these objects may be replaced with a tagged `ErrorResponse`, after which the client
    /// should close the connection.  (The server might send more data, possibly including
    /// additional chunks or `ErrorResponse`s, but it is not likely to be meaningful.)
    fn handle_hidden_states_request(
        &mut self,
        mut socket: impl Write,
        req: &HiddenStatesRequest,
    ) -> Result<(), String> {
        let n_embd = self.model.n_embd();
        try_send_message(&mut socket, &Response::from(HiddenStatesResponse {
            n_prompt: req.prompts.len(),
            n_layer: self.model.n_layer(),
            n_embd,
        }))?;

        // Eval callback
        #[derive(Clone, Debug, Default)]
        struct CallbackState<W> {
            socket: W,
            error: Option<String>,
            /// Number of tokens we've already seen in each layer.  llama.cpp splits the input
            /// batch of size `n_batch` into pieces of size `n_ubatch` for processing.  Tokens from
            /// previous ubatches are counted here so we know where the next tokens fall within the
            /// overall batch.
            tokens_seen: Box<[usize]>,
            /// Map from token index to prompt index.  For each entry `(token, prompt)` in this
            /// map, the token at position `token` in the batch is the last token of `prompt` (and
            /// thus should have its internal states extracted).
            token_prompt_map: BTreeMap<usize, usize>,
            /// Buffer for hidden states.  Data is copied here before being sent out.
            data_buf: Vec<f32>,
            /// Buffer for collecting prompt indices to be listed in the header.
            prompts_buf: Vec<usize>,
        }
        let cb_state = RefCell::new(CallbackState {
            socket: &mut socket,
            error: None,
            tokens_seen: vec![0; self.model.n_layer()].into_boxed_slice(),
            token_prompt_map: BTreeMap::new(),
            data_buf: Vec::new(),
            prompts_buf: Vec::new(),
        });

        let mut callback = |t: *mut ffi::ggml_tensor, ask| -> bool {
            // Get the tensor name.  The name is stored as a fixed-size array of `c_char` with a
            // null terminator.  We convert it to a slice and drop the terminator and everything
            // after it.
            let name = unsafe { (*t).name.as_slice() };
            let name = unsafe { mem::transmute::<&[i8], &[u8]>(name) };
            let name_len = name.iter().position(|&b| b == 0).unwrap_or(name.len());
            let name = &name[..name_len];
            /*
            eprintln!("running {} callback on {:?}",
                if ask { "ask" } else { "inspect" },
                String::from_utf8_lossy(name));
            */

            if ask {
                // We only want to inspect the contents of the layer output tensors.
                return name.starts_with(b"l_out-");
            }

            // In the non-ask case, we return true in all non-error cases so that inference
            // continues.

            let name_suffix = match name.strip_prefix(b"l_out-") {
                Some(s) => s,
                None => return true,
            };

            let mut state = cb_state.borrow_mut();
            let state = &mut *state;
            if state.error.is_some() {
                // Don't send more chunks if we've already encountered an error.
                return false;
            }

            let layer_idx = str::from_utf8(name_suffix).unwrap()
                .parse::<usize>().unwrap();

            unsafe {
                let ubatch_len = usize::try_from((*t).ne[1]).unwrap();
                let ubatch_start = state.tokens_seen[layer_idx];
                let ubatch_end = ubatch_start + ubatch_len;
                state.tokens_seen[layer_idx] += ubatch_len;

                state.prompts_buf.clear();
                state.prompts_buf.extend(
                    state.token_prompt_map.range(ubatch_start .. ubatch_end)
                        .map(|(_, &prompt)| prompt)
                );

                let r = try_send_message(
                    &mut *state.socket,
                    &HiddenStatesStreamingResponse::Chunk(HiddenStatesChunkHeader {
                        layer: layer_idx,
                        prompts: (&state.prompts_buf).into(),
                    }),
                );
                match r {
                    Ok(()) => {},
                    Err(e) => {
                        state.error = Some(e);
                        return false;
                    },
                }

                let data_buf = &mut state.data_buf;
                data_buf.clear();
                data_buf.reserve(state.prompts_buf.len() * n_embd);

                type Element = f32;

                assert_eq!(usize::try_from((*t).ne[0]).unwrap(), n_embd);
                assert_eq!(usize::try_from((*t).nb[0]).unwrap(), mem::size_of::<Element>());

                let token_prompt_iter =
                    state.token_prompt_map.range(ubatch_start .. ubatch_end)
                        .enumerate();
                for (i, (&token_idx, _)) in token_prompt_iter {
                    let input_idx = token_idx - ubatch_start;
                    let input_byte_offset = usize::try_from((*t).nb[1]).unwrap() * input_idx;
                    debug_assert_eq!(input_byte_offset % mem::size_of::<Element>(), 0);
                    let output_idx = i;
                    let output_byte_offset = n_embd * output_idx * mem::size_of::<Element>();
                    // TODO: It might be faster to use `ggml_backend_tensor_get_async` here, but we
                    // don't have a way to get the necessary `ggml_backend_t` handle.
                    ffi::ggml_backend_tensor_get(
                        t,
                        data_buf.as_mut_ptr().cast::<u8>().add(output_byte_offset)
                            .cast::<c_void>(),
                        input_byte_offset.try_into().unwrap(),
                        (n_embd * mem::size_of::<Element>()).try_into().unwrap(),
                    );

                    data_buf.set_len(n_embd * (i + 1));
                }

                assert_eq!(data_buf.len(), state.prompts_buf.len() * n_embd);

                // Flip float values to little endian if needed.
                #[cfg(target_endian = "big")] {
                    let data_words = slice::from_raw_parts_mut(
                        data_buf.as_mut_ptr().cast::<u32>(),
                        data_buf.len(),
                    );
                    for w in data_words {
                        *w = (*w).swap_bytes();
                    }
                }

                let data_bytes = slice::from_raw_parts(
                    data_buf.as_ptr().cast::<u8>(),
                    data_buf.len() * mem::size_of::<Element>(),
                );
                match state.socket.write_all(data_bytes) {
                    Ok(()) => {},
                    Err(e) => {
                        state.error = Some(format!("failed to send tensor data: {}", e));
                        return false;
                    },
                }
            }

            true
        };

        // Create a new context specifically for this request.
        let mut context_params = default_context_params();
        context_params.n_ctx = self.ctx.n_ctx().try_into().unwrap();
        context_params.n_batch = self.ctx.n_batch().try_into().unwrap();
        context_params.n_seq_max = self.ctx.n_batch().try_into().unwrap();

        let mut ctx = LlamaContext::with_model_and_eval_callback(
            &self.model,
            context_params,
            &mut callback,
        ).ok_or("failed to create context")?;

        // Tokenize each prompt.
        let prompt_tokens =
            req.prompts.iter().map(|s| self.tokenize(s)).collect::<Result<Vec<_>, _>>()?;

        let mut batch = LlamaBatch::new(ctx.n_batch(), ctx.n_seq_max());

        let mut prompt_tokens = prompt_tokens.iter().enumerate().peekable();
        while prompt_tokens.peek().is_some() {
            batch.clear();

            let mut cb_state_access = cb_state.borrow_mut();
            cb_state_access.tokens_seen.fill(0);
            cb_state_access.token_prompt_map.clear();

            ctx.kv_cache_clear();

            // Populate the batch with prompts until we run out of prompts, batch space, or
            // seq_ids.
            let mut seq_ids = 0 .. ctx.n_seq_max();
            loop {
                let (prompt_idx, prompt) = match prompt_tokens.peek() {
                    Some(&p) => p,
                    None => break,
                };
                let seq_id = match seq_ids.next() {
                    Some(i) => i,
                    None => break,
                };
                if prompt.len() > batch.capacity() - batch.len() {
                    break;
                }

                for (pos, &token) in prompt.iter().enumerate() {
                    let logits = pos == prompt.len() - 1;
                    batch.push(token, pos, seq_id, logits);
                }
                cb_state_access.token_prompt_map.insert(batch.len() - 1, prompt_idx);

                prompt_tokens.next();
            }

            eprintln!("run batch with {} tokens, {} seqs", batch.len(), seq_ids.start);

            // Drop the `borrow_mut` so the callback can access the state without a panic.
            drop(cb_state_access);

            ctx.decode(&batch)?;
        }

        // Drop `ctx`, which (indirectly) borrows `socket`, so we can send a final message.
        drop(ctx);

        try_send_message(&mut socket, &HiddenStatesStreamingResponse::Done)?;
        Ok(())
    }

    fn tokenize(&self, s: &str) -> Result<Vec<LlamaToken>, String> {
        let mut tokens = Vec::with_capacity(self.ctx.n_ctx());
        self.model.try_tokenize(s, &mut tokens, /* add_bos: */ true, /* special: */ true)
            .map_err(|n| format!("input has too many tokens: {}", n))?;
        Ok(tokens)
    }

    /// Build an empty array of token data with length equal to `n_vocab`.
    fn empty_token_data(&self) -> Box<[LlamaTokenData]> {
        let n_vocab = self.model.n_vocab();
        let v = vec![LlamaTokenData { id: 0, logit: 0., p: 0. }; n_vocab];
        v.into_boxed_slice()
    }

    /// Sample the next token based on the logits for `token_idx` from the last batch.  Uses `buf`
    /// as scratch space; it must have length equal to `n_vocab`.
    fn sample_token(&self, token_idx: usize, buf: &mut [LlamaTokenData]) -> LlamaToken {
        debug_assert_eq!(buf.len(), self.model.n_vocab());

        let logits = self.ctx.logits_ith(token_idx);
        for (i, (&logit, data)) in logits.iter().zip(buf.iter_mut()).enumerate() {
            // The whole buffer is potentially invalid, so we set all fields.  Even `id` may be
            // modified, since `llama_sample_softmax` sorts the array by logit.
            data.id = i.try_into().unwrap();
            data.logit = logit;
            // Clear `p` so that calling `sample_token` without doing the necessary processing
            // first will fail in an obvious way.
            data.p = 0.;
        }

        let mut candidates = LlamaTokenDataArray::new(buf);

        self.ctx.sample_softmax(&mut candidates);
        let token = self.ctx.sample_token(&mut candidates);
        token
    }

    /// Convert `token` to a string.  Uses `str_buf` as scratch space.
    fn token_to_str<'s>(
        &self,
        token: LlamaToken,
        str_buf: &'s mut Vec<u8>,
    ) -> Result<&'s str, String> {
        str_buf.clear();
        let token_str = self.model.token_to_piece_str(token, str_buf)
            .map_err(|e| format!("bad utf8 in model output: {}", e))?;
        Ok(token_str)
    }

    fn update_control_vectors(
        &mut self,
        control_vectors: &[ControlVector],
    ) -> Result<(), String> {
        if self.loaded_control_vectors == control_vectors {
            return Ok(());
        }

        let n_embd = self.model.n_embd();

        if control_vectors.len() == 0 {
            unsafe {
                let r = ffi::llama_control_vector_apply(
                    self.ctx.as_ptr(),
                    ptr::null_mut(),
                    0,
                    n_embd.try_into().unwrap(),
                    -1,
                    -1,
                );
                assert_eq!(r, 0);
            }
        } else {
            let data = self.load_control_vectors(control_vectors)?;

            // Skip the data for the first layer.  `llama_control_vector_apply` only wants data for
            // layers 1 and up.
            let data = &data[n_embd ..];

            unsafe {
                let r = ffi::llama_control_vector_apply(
                    self.ctx.as_ptr(),
                    data.as_ptr(),
                    data.len().try_into().unwrap(),
                    n_embd.try_into().unwrap(),
                    1,
                    self.model.n_layer().try_into().unwrap(),
                );
                assert_eq!(r, 0);
            }
        }

        self.loaded_control_vectors = control_vectors.iter().map(|cv| {
            let ControlVector { ref name, strength, layer_start, layer_end } = *cv;
            let name = String::from(&**name).into();
            ControlVector { name, strength, layer_start, layer_end }
        }).collect();
        Ok(())
    }

    /// Load control vector data as specified in `control_vectors`.  Returns a vector containing
    /// `n_layer * n_embd` values.
    fn load_control_vectors(
        &self,
        control_vectors: &[ControlVector],
    ) -> Result<Vec<f32>, String> {
        // TODO: Restrict file paths for security against untrusted clients

        let mut data = vec![0.; self.model.n_layer() * self.model.n_embd()];
        let n_layer = self.model.n_layer();
        let n_embd = self.model.n_embd();
        let n_embd_tensor_dim = n_embd.try_into().unwrap();
        for cv in control_vectors {
            unsafe {
                let path_cstr = CString::new(&*cv.name)
                    .map_err(|e| format!("bad path {:?}: {}", cv.name, e))?;
                let mut ggml_ctx = ptr::null_mut();
                let gguf = ffi::gguf_init_from_file(path_cstr.as_ptr(), ffi::gguf_init_params {
                    no_alloc: false,
                    ctx: &mut ggml_ctx,
                });
                if gguf.is_null() {
                    return Err(format!("failed to parse {:?} as gguf", cv.name));
                }

                let n_tensors = ffi::gguf_get_n_tensors(gguf);
                assert!(n_tensors >= 0);

                let layer_start = cv.layer_start.unwrap_or(0);
                let layer_end = cv.layer_end.unwrap_or(n_layer);
                let mut found_tensors = false;
                for layer in layer_start .. layer_end {
                    let tensor_name = format!("direction.{}\0", layer);
                    let tensor_name_cstr =
                        CStr::from_bytes_with_nul(tensor_name.as_bytes()).unwrap();
                    let tensor = ffi::ggml_get_tensor(ggml_ctx, tensor_name_cstr.as_ptr());
                    if tensor.is_null() {
                        continue;
                    }
                    found_tensors = true;

                    // Check tensor format
                    if (*tensor).type_ != ffi::ggml_type_GGML_TYPE_F32 {
                        return Err(format!("bad type {} for tensor {} of {:?}",
                                (*tensor).type_, tensor_name, cv.name));
                    }
                    if (*tensor).ne != [n_embd_tensor_dim, 1, 1, 1] {
                        return Err(format!(
                            "bad dimensions {:?} for tensor {} of {:?} (n_embd = {})",
                            (*tensor).ne, tensor_name, cv.name, n_embd,
                        ));
                    }

                    let tensor_data = slice::from_raw_parts(
                        (*tensor).data.cast::<f32>(),
                        n_embd,
                    );
                    let layer_data_offset = layer * n_embd;
                    let layer_data = &mut data[layer_data_offset .. layer_data_offset + n_embd];
                    for (&src, dest) in tensor_data.iter().zip(layer_data.iter_mut()) {
                        *dest += src * cv.strength;
                    }
                }

                if !found_tensors {
                    return Err(format!("found no tensors in layer range {} .. {} of {:?}",
                        layer_start, layer_end, cv.name));
                }
            }
        }

        Ok(data)
    }
}

/// Make a batch containing the sequence `tokens`.  The first token is placed at `start_pos`, and
/// remaining tokens are placed at sequentially increasing positions, all in sequence `seq_id`.
/// Logits are requested for the last token only.
fn batch_with_tokens(
    tokens: &[LlamaToken],
    start_pos: usize,
    seq_id: usize,
) -> LlamaBatch {
    let mut batch = LlamaBatch::new(tokens.len(), 1);
    for (i, &tok) in tokens.iter().enumerate() {
        let last = i == tokens.len() - 1;
        batch.push(tok, start_pos + i, seq_id, last);
    }
    batch
}


fn main() -> io::Result<()> {
    let args = parse_args();


    // Load model
    let mut model_params = default_model_params();
    model_params.n_gpu_layers = 999;

    let model_path: String = args.get_one::<String>("model").unwrap().clone();
    let model = LlamaModel::load_from_file(model_path, model_params).unwrap();
    eprintln!("model = {:?}", model);

    // Create context
    let mut context_params = default_context_params();
    context_params.n_ctx = args.get_one::<usize>("ctx_size").unwrap().clone().try_into().unwrap();
    let mut ctx = LlamaContext::with_model(&model, context_params).unwrap();
    eprintln!("ctx = {:?}", ctx);


    let mut scx = ServerContext::new(&model, &mut ctx);


    // Accept loop
    let socket_path = Path::new("llama.socket");
    if socket_path.exists() {
        fs::remove_file(socket_path)?;
    }
    let listener = UnixListener::bind(socket_path)?;

    let mut line_buf = String::new();
    for socket in listener.incoming() {
        let mut socket = socket?;
        // NB: We assume the client is non-malicious.  A malicious client can exhaust all memory by
        // sending a large amount of data with no '\n'.
        line_buf.clear();
        BufReader::new(&mut socket).read_line(&mut line_buf)?;

        let req = match serde_json::from_str::<Request>(&line_buf) {
            Ok(x) => x,
            Err(e) => {
                send_response(&mut socket, &ErrorResponse {
                    msg: format!("bad request: {}", e).into(),
                }.into());
                continue;
            },
        };
        eprintln!("request: {:?}", req);

        let result = scx.handle_request(&mut socket, &req);
        match result {
            Ok(()) => {},
            Err(msg) => {
                send_response(&mut socket, &ErrorResponse { msg: msg.into() }.into());
            },
        }
    }

    Ok(())
}

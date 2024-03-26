use std::borrow::Cow;
use std::fs;
use std::io::{self, BufReader, BufRead, Write};
use std::os::unix::net::UnixListener;
use std::path::Path;
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
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct CompletionRequest<'a> {
    prompt: Cow<'a, str>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct BatchCompletionRequest<'a> {
    #[serde(flatten)]
    c: CompletionRequest<'a>,
    batch_size: usize,
}


#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Response<'a> {
    Completion(CompletionResponse<'a>),
    BatchCompletion(BatchCompletionResponse<'a>),
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

impl<'a> From<ErrorResponse<'a>> for Response<'a> {
    fn from(x: ErrorResponse<'a>) -> Response<'a> {
        Response::Error(x)
    }
}


/// Send `resp` on `socket`.  This first tries to send `resp`; if that fails, it tries to send an
/// error message; and if that also fails, it gives up and returns without  doing anything.
fn send_response(mut socket: impl Write, resp: &Response) {
    let resp_json = serde_json::to_string(&resp).unwrap();
    let err = match socket.write_all(resp_json.as_bytes()) {
        Ok(_) => return,
        Err(e) => format!("error sending response: {}", e),
    };
    let err_resp = Response::Error(ErrorResponse {
        msg: (&err).into(),
    });
    let err_resp_json = serde_json::to_string(&err_resp).unwrap();
    match socket.write_all(err_resp_json.as_bytes()) {
        Ok(_) => return,
        Err(e) => {
            // Give up
            eprintln!("response dropped: {}; {}", err, e);
            return;
        },
    }
}


struct ServerContext<'a, 'b> {
    model: &'a LlamaModel,
    ctx: &'b mut LlamaContext<'a>,
}

impl<'a, 'b> ServerContext<'a, 'b> {
    pub fn new(
        model: &'a LlamaModel,
        ctx: &'b mut LlamaContext<'a>,
    ) -> ServerContext<'a, 'b> {
        ServerContext { model, ctx }
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
        }
    }

    fn handle_completion_request(
        &mut self,
        socket: impl Write,
        req: &CompletionRequest,
    ) -> Result<(), String> {
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
        for offset in 0 .. 128 {
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
        // Tokenize and process the prompt.
        let tokens = self.tokenize(&req.c.prompt)?;
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
        for offset in 0 .. 32 {
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

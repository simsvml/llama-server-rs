use std::borrow::Cow;
use std::fs;
use std::io::{self, BufReader, BufRead, Write};
use std::os::unix::net::UnixListener;
use std::path::Path;
use clap::{Command, Arg, ArgMatches, value_parser};
use serde::{Serialize, Deserialize};
use serde_json;
use llama_server_rs::{
    LlamaModel, default_model_params, LlamaContext, default_context_params, LlamaBatch,
    LlamaTokenData, LlamaTokenDataArray,
};
    

fn parse_args() -> ArgMatches {
    Command::new("llama-server-rs")
        .arg(Arg::new("model").short('m').long("model"))
        .arg(Arg::new("ctx_size").short('c').long("ctx-size")
            .value_parser(value_parser!(usize))
            .default_value("4096"))
        .get_matches()
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
enum Request<'a> {
    Completion(CompletionRequest<'a>),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct CompletionRequest<'a> {
    prompt: Cow<'a, str>,
}


#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
enum Response<'a> {
    Completion(CompletionResponse<'a>),
    Error(ErrorResponse<'a>),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct CompletionResponse<'a> {
    content: Cow<'a, str>,
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
            Request::Completion(ref cr) => self.handle_completion_request(socket, cr),
        }
    }

    fn handle_completion_request(
        &mut self,
        socket: impl Write,
        req: &CompletionRequest,
    ) -> Result<(), String> {
        // Tokenize input
        let mut tokens = Vec::with_capacity(self.ctx.n_ctx());
        self.model.try_tokenize(&req.prompt, &mut tokens, /* add_bos: */ true, /* special: */ true)
            .map_err(|n| format!("input has too many tokens: {}", n))?;
        eprintln!("tokens = {:?}", tokens);

        // Prompt processing
        let mut batch = LlamaBatch::new(tokens.len());
        for (i, &tok) in tokens.iter().enumerate() {
            let last = i == tokens.len() - 1;
            batch.push(tok, i, /* seq_id: */ 0, last);
        }

        self.ctx.decode(&batch)?;
        eprintln!("llama_decode ok");


        // Sampling
        let mut pos = tokens.len();
        let end_pos = pos + 128;
        let mut logits_index = pos - 1;
        let mut token_buf = Vec::with_capacity(64);
        let mut content = String::new();
        loop {
            // Sample token
            let logits = self.ctx.logits_ith(logits_index);
            let mut token_data = logits.iter().enumerate().map(|(i, &logit)| {
                LlamaTokenData {
                    id: i.try_into().unwrap(),
                    logit,
                    p: 0.,
                }
            }).collect::<Vec<_>>();

            let mut candidates = LlamaTokenDataArray::new(&mut token_data);

            self.ctx.sample_softmax(&mut candidates);
            let token = self.ctx.sample_token(&mut candidates);

            token_buf.clear();
            let token_str = self.model.token_to_piece_str(token, &mut token_buf)
                .map_err(|e| format!("bad utf8 in model output: {}", e))?;

            eprint!("{}", token_str);
            //eprintln!("sampled token {} = {:?}", token, token_str);
            content.push_str(token_str);

            if pos >= end_pos {
                break;
            }

            // TODO: Stop on EOS token

            // Prepare next batch
            batch.clear();
            batch.push(token, /* pos: */ pos, /* seq_id: */ 0, true);
            pos += 1;
            logits_index = 0;

            self.ctx.decode(&batch)?;
            //eprintln!("llama_decode ok");
        }
        eprintln!();

        send_response(socket, &CompletionResponse {
            content: content.into(),
        }.into());

        Ok(())
    }
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
    let mut listener = UnixListener::bind(socket_path)?;

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

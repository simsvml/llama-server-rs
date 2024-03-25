use clap::{Command, Arg, ArgMatches, value_parser};
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

fn main() {
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

    // Tokenize input
    let input = "[INST] Is Rust a compiled language? [/INST]";
    let mut tokens = Vec::with_capacity(ctx.n_ctx());
    model.try_tokenize(input, &mut tokens, /* add_bos: */ true, /* special: */ true)
        .unwrap_or_else(|n| panic!("input has too many tokens: {}", n));
    eprintln!("tokens = {:?}", tokens);

    // Prompt processing
    let mut batch = LlamaBatch::new(tokens.len());
    for (i, &tok) in tokens.iter().enumerate() {
        let last = i == tokens.len() - 1;
        batch.push(tok, i, /* seq_id: */ 0, last);
    }

    ctx.decode(&batch).unwrap();
    eprintln!("llama_decode ok");


    // Sampling
    let mut pos = tokens.len();
    let mut logits_index = pos - 1;
    let mut token_buf = Vec::with_capacity(64);
    loop {
        // Sample token
        let logits = ctx.logits_ith(logits_index);
        let mut token_data = logits.iter().enumerate().map(|(i, &logit)| {
            LlamaTokenData {
                id: i.try_into().unwrap(),
                logit,
                p: 0.,
            }
        }).collect::<Vec<_>>();

        let mut candidates = LlamaTokenDataArray::new(&mut token_data);

        ctx.sample_softmax(&mut candidates);
        let token = ctx.sample_token(&mut candidates);

        token_buf.clear();
        let token_str = model.try_token_to_piece_str(token, &mut token_buf).unwrap();

        eprint!("{}", token_str);
        //eprintln!("sampled token {} = {:?}", token, token_str);

        if pos > 64 {
            break;
        }

        // Prepare next batch
        batch.clear();
        batch.push(token, /* pos: */ pos, /* seq_id: */ 0, true);
        pos += 1;
        logits_index = 0;

        ctx.decode(&batch).unwrap();
        //eprintln!("llama_decode ok");
    }
    eprintln!();
}

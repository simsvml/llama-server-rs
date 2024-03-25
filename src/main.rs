use std::ffi::{CStr, CString};
use std::mem;
use clap::{Command, Arg, ArgMatches, value_parser};
use libc;
use llama_server_rs::ffi;

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
    let mut model_params = unsafe { ffi::llama_model_default_params() };
    model_params.n_gpu_layers = 999;

    let model_path: String = args.get_one::<String>("model").unwrap().clone();
    let model = unsafe {
        let name = CString::new(model_path).unwrap();
        ffi::llama_load_model_from_file(name.as_ptr(), model_params)
    };
    eprintln!("model = {:?}", model);

    // Create context
    let mut context_params = unsafe { ffi::llama_context_default_params() };
    context_params.n_ctx = args.get_one::<usize>("ctx_size").unwrap().clone().try_into().unwrap();
    let ctx = unsafe { ffi::llama_new_context_with_model(model, context_params) };
    eprintln!("ctx = {:?}", ctx);

    // Tokenize input
    let input = "[INST] Is Rust a compiled language? [/INST]";
    let n_ctx = usize::try_from(unsafe { ffi::llama_n_ctx(ctx) }).unwrap();
    let mut tokens = vec![0; n_ctx];

    unsafe {
        let r = ffi::llama_tokenize(
            model,
            input.as_ptr().cast::<i8>(),
            input.len().try_into().unwrap(),
            tokens.as_mut_ptr(),
            tokens.capacity().try_into().unwrap(),
            true,   // add_bos
            true,   // special
        );
        assert!(r >= 0, "input has too many tokens: {}", r);
        tokens.set_len(r.try_into().unwrap());
    }
    eprintln!("tokens = {:?}", tokens);

    // Prompt processing
    let mut batch = unsafe {
        ffi::llama_batch_init(
            tokens.len().try_into().unwrap(),
            0,  // embd
            1,  // n_seq_max
        )
    };

    for (i, &tok) in tokens.iter().enumerate() {
        unsafe {
            *batch.token.add(i) = tok;
            *batch.pos.add(i) = i.try_into().unwrap();
            *batch.n_seq_id.add(i) = 1;
            *batch.seq_id.add(i) = libc::malloc(mem::size_of::<ffi::llama_seq_id>()
                .try_into().unwrap()).cast::<i32>();
            *(*batch.seq_id.add(i)) = 0;
            *batch.logits.add(i) = (i == tokens.len() - 1) as _;
        }
    }
    batch.n_tokens = tokens.len().try_into().unwrap();

    let r = unsafe { ffi::llama_decode(ctx, batch) };
    match r {
        0 => {},
        1 => panic!("couldn't find kv slot for batch of size {}", batch.n_tokens),
        _ => panic!("llama_decode failed"),
    }
    eprintln!("llama_decode ok");


    // Sampling
    let n_vocab = usize::try_from(unsafe { ffi::llama_n_vocab(model) }).unwrap();
    let mut pos = tokens.len();
    let mut logits_index = pos - 1;
    loop {
        // Sample token
        let logits = unsafe {
            ffi::llama_get_logits_ith(
                ctx,
                logits_index.try_into().unwrap(),
            )
        };
        let mut token_data = Vec::with_capacity(n_vocab);
        for i in 0 .. n_vocab {
            let logit = unsafe { *logits.add(i) };
            token_data.push(ffi::llama_token_data {
                id: i.try_into().unwrap(),
                logit,
                p: 0.,
            });
        }
        let mut candidates = ffi::llama_token_data_array {
            data: token_data.as_mut_ptr(),
            size: token_data.len().try_into().unwrap(),
            sorted: false,
        };

        unsafe { ffi::llama_sample_softmax(ctx, &mut candidates) };
        let token = unsafe { ffi::llama_sample_token(ctx, &mut candidates) };

        let token_str = unsafe {
            let mut buf = Vec::<u8>::with_capacity(64);
            let r = ffi::llama_token_to_piece(
                model,
                token,
                buf.as_mut_ptr().cast::<i8>(),
                buf.capacity().try_into().unwrap(),
            );
            assert!(r >= 0, "sampled token is too long for buffer: {}", r);
            buf.set_len(r.try_into().unwrap());
            String::from_utf8(buf).unwrap()
        };

        eprint!("{}", token_str);
        //eprintln!("sampled token {} = {:?}", token, token_str);

        if pos > 64 {
            break;
        }

        // Prepare next batch
        unsafe {
            *batch.logits.add(logits_index) = 0;

            batch.n_tokens = 1;
            *batch.token = token;
            *batch.pos = pos.try_into().unwrap();
            if *batch.n_seq_id == 0 {
                *batch.n_seq_id = 1;
                *batch.seq_id = libc::malloc(mem::size_of::<ffi::llama_seq_id>()
                    .try_into().unwrap()).cast::<i32>();
            }
            *(*batch.seq_id) = 0;
            *batch.logits = true as _;
        }
        pos += 1;
        logits_index = 0;

        let r = unsafe { ffi::llama_decode(ctx, batch) };
        match r {
            0 => {},
            1 => panic!("couldn't find kv slot for batch of size {}", batch.n_tokens),
            _ => panic!("llama_decode failed"),
        }
        //eprintln!("llama_decode ok");
    }
    eprintln!();
}

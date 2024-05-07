# Building

First, build llama.cpp.  Run `git submodule update --init llama.cpp`, then `cd
llama.cpp` and build according to the normal llama.cpp build instructions.
This should produce `libllama.a`, which is needed to build the server.

Build the server by running `cargo build`.  Requires a Rust toolchain.


# Running

Start the core server like this:

```sh
cargo run -- -m path/to/model.gguf --ngl 999 -c 4096
```

Use `cargo run -- --help` to see what options are supported.  The command-line
parsing library this uses doesn't allow multi-letter options with a single
dash, so llama.cpp's `-ngl` option is instead written `--ngl`.

The core server listens on the Unix socket `./llama.socket` and speaks a custom
protocol.  Some of the tools in this repo such as `train_control_vector.py` use
the custom protocol directly.  For other frontends, like SillyTavern, run
`python3 api_server.py` to start an HTTP server that translates between the
kobold.cpp API and the custom protocol.


# Training control vectors

First, prepare a JSON file containing a list of prompt pairs:

```javascript
[
    "positive prompt 1",
    "negative prompt 1",
    "positive prompt 2",
    "negative prompt 2",
    // etc.
]
```

This can be done with a simple Python script, for example.

Then, train the control vector by running:

```sh
python3 train_control_vector.py prompts.json out.gguf
```

Add `--mode mean-diff` to use the mean difference method instead of PCA.


# Using control vectors

As an extension to the normal kobold.cpp API, text-generation endpoints like
`/api/extra/generate/stream` accept an additional `control_vectors` field,
which contains a list of control vectors to apply.  Example:

```javascript
{
    "prompt": "...",
    // Other options go here...
    "control_vectors": [
        {
            "name": "path/to/cv1.gguf",
            "strength": 1.0,
            "layer_start": 10,
            "layer_end": 20,
        },
        {
            "name": "path/to/cv2.gguf",
            "strength": 0.5,
        }
    ]
}
```

`name` and `strength` are required.  `layer_start` and `layer_end` are optional
and default to 0 and the number of layers in the model respectively.
`layer_end` is an exclusive bound.  Different control vectors in the list can
cover different layer ranges.  You can even apply the same control vector
multiple times; for example, you could list `foo.gguf` with strength 0.2 and
then list `foo.gguf` again at strength 0.3 on layers 10-20 to have it applied
with total strength 0.5 in those 10 layers and with strength 0.2 elsewhere.

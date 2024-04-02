// Copyright 2018 Parity Technologies (UK) Ltd.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.

use anyhow::{Context, Result};
use futures::stream::StreamExt;
use libp2p::{gossipsub, mdns, noise, swarm::NetworkBehaviour, swarm::SwarmEvent, tcp, yamux};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::ggml_time_us;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::AddBos;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::token::data_array::LlamaTokenDataArray;
use std::collections::hash_map::DefaultHasher;
use std::error::Error;
use std::ffi::CString;
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::pin::pin;
use std::str::FromStr;
use std::time::Duration;
use tokio::{io, io::AsyncBufReadExt, select};
use tracing_subscriber::EnvFilter;

const threads: u32 = 16;

fn predict(
    prompt: &str,
    model: &LlamaModel,
    ctx: &mut LlamaContext,
) -> Result<String, Box<dyn Error>> {
    let n_len = 256;
    // tokenize the prompt

    ctx.clear_kv_cache();

    let tokens_list = model
        .str_to_token(&prompt, AddBos::Always)
        .with_context(|| format!("failed to tokenize {prompt}"))?;
    let prompt_len = tokens_list.len() as i32;

    let n_cxt = ctx.n_ctx() as i32;
    let n_kv_req = tokens_list.len() as i32 + (n_len - tokens_list.len() as i32);

    // println!("got prompt {}", prompt);
    // eprintln!("n_len = {n_len}, n_ctx = {n_cxt}, k_kv_req = {n_kv_req}");

    // make sure the KV cache is big enough to hold all the prompt and generated tokens
    // if n_kv_req > n_cxt {
    //     println!(
    //         "n_kv_req > n_ctx, the required kv cache size is not big enough
    //     either reduce n_len or increase n_ctx"
    //     )
    // }

    // if tokens_list.len() >= usize::try_from(n_len)? {
    //     println!("the prompt is too long, it has more tokens than n_len")
    // }

    // print the prompt token-by-token
    // eprintln!();

    // for token in &tokens_list {
    // eprint!("{}", model.token_to_str(*token)?);
    // }

    std::io::stderr().flush()?;

    // create a llama_batch with size 512
    // we use this object to submit token data for decoding
    let mut batch = LlamaBatch::new(512, 1);

    let last_index: i32 = (tokens_list.len() - 1) as i32;
    for (i, token) in (0_i32..).zip(tokens_list.into_iter()) {
        // llama_decode will output logits only for the last token of the prompt
        let is_last = i == last_index;
        batch.add(token, i, &[0], is_last)?;
    }

    ctx.decode(&mut batch)
        .with_context(|| "llama_decode() failed")?;

    // main loop

    let mut n_cur = batch.n_tokens();
    let mut n_decode = 0;

    let t_main_start = ggml_time_us();

    let mut response = String::new();
    response.push_str("Response: ");

    while n_cur <= n_len {
        // sample the next token
        {
            let candidates = ctx.candidates_ith(batch.n_tokens() - 1);

            let candidates_p = LlamaTokenDataArray::from_iter(candidates, false);

            // sample the most likely token
            let new_token_id = ctx.sample_token_greedy(candidates_p);

            // is it an end of stream?
            if new_token_id == model.token_eos() {
                // eprintln!();
                break;
            }

            let token_str = model.token_to_str(new_token_id)?;

            if n_cur > prompt_len {
                response.push_str(&token_str);

                // if new_token_id == model.token_nl() {
                //     break;
                // }
            }

            batch.clear();
            batch.add(new_token_id, n_cur, &[0], true)?;
        }

        n_cur += 1;

        ctx.decode(&mut batch).with_context(|| "failed to eval")?;

        n_decode += 1;
    }

    // eprintln!("\n");

    let t_main_end = ggml_time_us();

    let duration = Duration::from_micros((t_main_end - t_main_start) as u64);

    // eprintln!(
    //     "decoded {} tokens in {:.2} s, speed {:.2} t/s\n",
    //     n_decode,
    //     duration.as_secs_f32(),
    //     n_decode as f32 / duration.as_secs_f32()
    // );

    // println!("{}", ctx.timings());

    // println!("Response: {}", response);

    Ok(response)
}

// We create a custom network behaviour that combines Gossipsub and Mdns.
#[derive(NetworkBehaviour)]
struct MyBehaviour {
    gossipsub: gossipsub::Behaviour,
    mdns: mdns::tokio::Behaviour,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .try_init();

    // init LLM
    let backend = LlamaBackend::init()?;

    // offload all layers to the gpu
    let model_params = {
        #[cfg(feature = "cublas")]
        if !disable_gpu {
            LlamaModelParams::default().with_n_gpu_layers(1000)
        } else {
            LlamaModelParams::default()
        }
        #[cfg(not(feature = "cublas"))]
        LlamaModelParams::default()
    };

    let mut model_params = pin!(model_params);
    // model_params
    //     .as_mut()
    //     .append_kv_override("temperature".as_c_str(), &"0.7");
    // model_params.set_temperature(0.7);
    // model_params.set_repeat_penalty(1.1);
    // model_params.set_n_ctx(2048);

    // for (k, v) in &key_value_overrides {
    //     let k = CString::new(k.as_bytes()).with_context(|| format!("invalid key: {k}"))?;
    //     model_params.as_mut().append_kv_override(k.as_c_str(), *v);
    // }

    let model_path = "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";

    let model = LlamaModel::load_from_file(&backend, model_path, &model_params)
        .with_context(|| "unable to load model")?;

    // initialize the context
    let mut ctx_params = LlamaContextParams::default();
    // .with_n_ctx(2048)
    // .with_seed(1234);

    ctx_params = ctx_params.with_n_threads(threads);
    // if let Some(threads_batch) = batch {
    //   ctx_params = ctx_params.with_n_threads_batch(threads_batch);
    // }

    let mut ctx = model
        .new_context(&backend, ctx_params)
        .with_context(|| "unable to create the llama_context")?;

    // let model_options = ModelOptions::default();

    // let llama = LLama::new(
    //     "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf".into(),
    //     &model_options,
    // )
    // .unwrap();

    let mut swarm = libp2p::SwarmBuilder::with_new_identity()
        .with_tokio()
        .with_tcp(
            tcp::Config::default(),
            noise::Config::new,
            yamux::Config::default,
        )?
        .with_quic()
        .with_behaviour(|key| {
            // To content-address message, we can take the hash of message and use it as an ID.
            let message_id_fn = |message: &gossipsub::Message| {
                let mut s = DefaultHasher::new();
                message.data.hash(&mut s);
                gossipsub::MessageId::from(s.finish().to_string())
            };

            // Set a custom gossipsub configuration
            let gossipsub_config = gossipsub::ConfigBuilder::default()
                .heartbeat_interval(Duration::from_secs(10)) // This is set to aid debugging by not cluttering the log space
                .validation_mode(gossipsub::ValidationMode::Strict) // This sets the kind of message validation. The default is Strict (enforce message signing)
                .message_id_fn(message_id_fn) // content-address messages. No two messages of the same content will be propagated.
                .build()
                .map_err(|msg| io::Error::new(io::ErrorKind::Other, msg))?; // Temporary hack because `build` does not return a proper `std::error::Error`.

            // build a gossipsub network behaviour
            let gossipsub = gossipsub::Behaviour::new(
                gossipsub::MessageAuthenticity::Signed(key.clone()),
                gossipsub_config,
            )?;

            let mdns =
                mdns::tokio::Behaviour::new(mdns::Config::default(), key.public().to_peer_id())?;
            Ok(MyBehaviour { gossipsub, mdns })
        })?
        .with_swarm_config(|c| c.with_idle_connection_timeout(Duration::from_secs(60)))
        .build();

    // Create a Gossipsub topic
    let topic = gossipsub::IdentTopic::new("test-net");
    // subscribes to our topic
    swarm.behaviour_mut().gossipsub.subscribe(&topic)?;

    // Read full lines from stdin
    let mut stdin = io::BufReader::new(io::stdin()).lines();

    // Listen on all interfaces and whatever port the OS assigns
    swarm.listen_on("/ip4/0.0.0.0/udp/0/quic-v1".parse()?)?;
    swarm.listen_on("/ip4/0.0.0.0/tcp/0".parse()?)?;

    println!("Enter messages via STDIN and they will be sent to connected peers using Gossipsub");

    // let query: &str = "hello are you operational?";

    // let prompt = format!(
    //     r#"
    //                 <|im_start|>system:
    //                   A chat between a user and an artificial intelligent assistant. The assistant is dumb. The assistant answers in 69 words or less.
    //                 <|im_end|>

    //                 <|im_start|>user
    //                   {}
    //                 <|im_end|>

    //                 <|im_start|>assistant"#,
    //     query
    // );

    // predict(&prompt, &model, &mut ctx);

    // Kick it off
    loop {
        select! {
            Ok(Some(line)) = stdin.next_line() => {
                if let Err(e) = swarm
                    .behaviour_mut().gossipsub
                    .publish(topic.clone(), line.as_bytes()) {
                    println!("Publish error: {e:?}");
                }
            }
            event = swarm.select_next_some() => match event {
                SwarmEvent::Behaviour(MyBehaviourEvent::Mdns(mdns::Event::Discovered(list))) => {
                    for (peer_id, _multiaddr) in list {
                        println!("mDNS discovered a new peer: {peer_id}");
                        swarm.behaviour_mut().gossipsub.add_explicit_peer(&peer_id);
                    }
                },
                SwarmEvent::Behaviour(MyBehaviourEvent::Mdns(mdns::Event::Expired(list))) => {
                    for (peer_id, _multiaddr) in list {
                        println!("mDNS discover peer has expired: {peer_id}");
                        swarm.behaviour_mut().gossipsub.remove_explicit_peer(&peer_id);
                    }
                },
                SwarmEvent::Behaviour(MyBehaviourEvent::Gossipsub(gossipsub::Event::Message {
                    propagation_source: peer_id,
                    message_id: id,
                    message,
                })) => {

                    let chat = String::from_utf8_lossy(&message.data).to_string();
                    // ./main -ngl 32 -m tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf --color -c 2048 --temp 0.7 --repeat_penalty 1.1 -n -1 -p "<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant"

                    if chat.starts_with("Response:") {
                      println!("Got response {chat} id: {id} from peer: {peer_id}");
                    } else {
                      println!(
                        "Got query {chat} with id: {id} from peer: {peer_id}"
                      );

                      let prompt = format!(
                        r#"
  ### Instruction:
  A chat between a user and an artificial intelligent assistant. The assistant answers in 69 words or less.
  The user has asked:
  {}
  ### Response:"#,
                        chat
                    );

                      let prediction = predict(&prompt, &model, &mut ctx).unwrap();

                      println!("Got prediction: {}", prediction);

                      if let Err(e) = swarm
                        .behaviour_mut().gossipsub
                        .publish(topic.clone(), prediction.as_bytes()) {
                          println!("Publish error: {e:?}");
                        }
                    }
                  }
                SwarmEvent::NewListenAddr { address, .. } => {
                    println!("Local node is listening on {address}");
                }
                _ => {}
            }
        }
    }
}

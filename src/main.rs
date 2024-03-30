use libp2p::{
  identity,
  noise::{self, NoiseConfig, X25519Spec},
  tcp::TokioTcpConfig,
  mplex, swarm::SwarmBuilder,
  Multiaddr, PeerId, Swarm, Transport,
};
use tokio::io::{self, AsyncBufReadExt};
use tokio::select;
use tokio::signal;
use libp2p::futures::StreamExt;

async fn run() -> Result<(), Box<dyn std::error::Error>> {
  // Generate a random key for this node.
  let local_key = identity::Keypair::generate_ed25519();
  let local_peer_id = PeerId::from(local_key.public());

  println!("Local peer id: {:?}", local_peer_id);

// Generate noise keys from the identity keys
let noise_keys = noise::Keypair::<noise::X25519Spec>::new().into_authentic(&local_key)?;

// Set up an encrypted TCP transport over the Mplex protocol
let transport = TokioTcpConfig::new()
    .upgrade(libp2p::core::upgrade::Version::V1)
    .authenticate(NoiseConfig::xx(noise_keys).into_authenticated())
    .multiplex(mplex::MplexConfig::new())
    .boxed();

  // Create a Swarm to manage peers and events.
  let mut swarm = {
      let behaviour = libp2p::ping::Ping::new(libp2p::ping::Config::new().with_keep_alive(true));
      SwarmBuilder::new(transport, behaviour, local_peer_id)
          .executor(Box::new(|fut| {
              tokio::spawn(fut);
          }))
          .build()
  };

  // Listen on all interfaces and a random port.
  swarm.listen_on("/ip4/0.0.0.0/tcp/0".parse()?)?;

  // Use tokio for async runtime.
  let mut stdin = io::BufReader::new(io::stdin()).lines();

  loop {
      select! {
          line = stdin.next_line() => {
              let line = line?.expect("stdin closed");
              if line == "EXIT" {
                  break;
              } else if line.starts_with("/dial") {
                  let addr: Multiaddr = line[6..].parse()?;
                  swarm.dial(addr)?;
                  println!("Dialed {:?}", line);
              }
          }
          event = swarm.select_next_some() => {
              println!("Swarm Event: {:?}", event);
          }
          _ = signal::ctrl_c() => {
              break;
          }
      }
  }

  Ok(())
}

#[tokio::main]
async fn main() {
  if let Err(e) = run().await {
      println!("Error: {}", e);
  }
}

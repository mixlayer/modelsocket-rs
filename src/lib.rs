pub mod protocol;

use std::sync::Once;

// re-export async_trait for toolbox implementation convenience
pub use async_trait::async_trait;

pub use protocol::*;

#[cfg(feature = "client")]
pub mod client;

#[cfg(feature = "client")]
pub use client::*;

#[cfg(feature = "python")]
mod python;

#[cfg(feature = "python")]
pub use python::*;
use rustls::crypto::{self, CryptoProvider};

static INIT_RUSTLS: Once = Once::new();

pub fn ensure_rustls_provider() {
    INIT_RUSTLS.call_once(|| {
        // If something else already installed a provider, just trust it.
        if CryptoProvider::get_default().is_none() {
            // Try to install ring; ignore failure because at worst someone else
            // raced and installed one between get_default() and this call.
            let _ = crypto::ring::default_provider().install_default();
        }
    });
}

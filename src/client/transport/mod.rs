use crate::ModelSocketError;
use crate::protocol::{MSEvent, MSRequest};
use futures::{Sink, Stream};

pub mod ws;

pub trait MSTransport<RX, TX>
where
    RX: Stream<Item = Result<MSEvent, ModelSocketError>>,
    TX: Sink<MSRequest>,
{
    fn split(self) -> (TX, RX);
}

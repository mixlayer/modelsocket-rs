use crate::protocol::MSRequest;
use futures::{stream::SplitStream, Sink, Stream};
use futures_util::{stream::SplitSink, StreamExt};
use http::{Request, Uri};
use std::pin::Pin;

use tokio::net::TcpStream;
use tokio_tungstenite::{
    connect_async,
    tungstenite::{self, protocol::Message},
    MaybeTlsStream, WebSocketStream,
};

use crate::{transport::MSTransport, ModelSocketError};

pub struct WebSocketTransport {
    ws_sink: WsSink,
    ws_stream: WsStream,
}

impl WebSocketTransport {
    pub async fn connect(url: &str, api_key: Option<&str>) -> Result<Self, ModelSocketError> {
        let uri: Uri = url.parse().unwrap();

        let mut request_builder = Request::builder().uri(&uri);

        request_builder = request_builder
            .header(
                "Sec-WebSocket-Key",
                tungstenite::handshake::client::generate_key(),
            )
            .header("host", uri.host().unwrap())
            .header("upgrade", "websocket")
            .header("connection", "upgrade")
            .header("sec-websocket-version", 13);

        if let Some(key) = api_key {
            request_builder = request_builder.header("Authorization", format!("Bearer {key}"));
        }

        let request = request_builder
            .body(())
            .map_err(|e| ModelSocketError::Protocol(e.to_string()))?;

        let (ws_stream, http_resp): (
            WebSocketStream<MaybeTlsStream<TcpStream>>,
            http::Response<Option<Vec<u8>>>,
        ) = connect_async(request).await?;

        if http_resp.status() != 101 {
            return Err(ModelSocketError::Protocol(format!(
                "WebSocket upgrade failed with status: {}",
                http_resp.status()
            )));
        }

        let (ws_sink, ws_stream) = ws_stream.split();

        let ws_sink = WsSink::new(ws_sink);
        let ws_stream = WsStream::new(ws_stream);

        Ok(Self { ws_sink, ws_stream })
    }
}

impl MSTransport<WsStream, WsSink> for WebSocketTransport {
    fn split(self) -> (WsSink, WsStream) {
        (self.ws_sink, self.ws_stream)
    }
}

/// A sink for transmitting ModelSocket requests over a WebSocket
pub struct WsSink {
    inner: Pin<Box<SplitSink<WebSocketStream<MaybeTlsStream<TcpStream>>, Message>>>,
}

impl WsSink {
    pub fn new(inner: SplitSink<WebSocketStream<MaybeTlsStream<TcpStream>>, Message>) -> Self {
        Self {
            inner: Box::pin(inner),
        }
    }
}

impl Sink<MSRequest> for WsSink {
    type Error = ModelSocketError;

    fn poll_ready(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), Self::Error>> {
        self.inner
            .as_mut()
            .poll_ready(cx)
            .map_err(|e| ModelSocketError::WebSocket(e))
    }

    fn start_send(mut self: std::pin::Pin<&mut Self>, item: MSRequest) -> Result<(), Self::Error> {
        let msg_text = serde_json::to_string(&item)
            .map_err(|_e| ModelSocketError::Command("error serializing ws frame".to_string()))?;

        self.inner
            .as_mut()
            .start_send(Message::Text(msg_text))
            .map_err(|e| ModelSocketError::WebSocket(e))
    }

    fn poll_flush(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), Self::Error>> {
        self.inner
            .as_mut()
            .poll_flush(cx)
            .map_err(|e| ModelSocketError::WebSocket(e))
    }

    fn poll_close(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), Self::Error>> {
        self.inner
            .as_mut()
            .poll_close(cx)
            .map_err(|e| ModelSocketError::WebSocket(e))
    }
}

pub struct WsStream {
    inner: Pin<Box<SplitStream<WebSocketStream<MaybeTlsStream<TcpStream>>>>>,
}

impl WsStream {
    pub fn new(inner: SplitStream<WebSocketStream<MaybeTlsStream<TcpStream>>>) -> Self {
        Self {
            inner: Box::pin(inner),
        }
    }
}

impl Stream for WsStream {
    type Item = Result<crate::protocol::MSEvent, ModelSocketError>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        match self.inner.as_mut().poll_next(cx) {
            std::task::Poll::Ready(Some(Ok(msg))) => {
                if let Message::Text(text) = msg {
                    match serde_json::from_str::<crate::protocol::MSEvent>(&text) {
                        Ok(event) => std::task::Poll::Ready(Some(Ok(event))),
                        Err(e) => std::task::Poll::Ready(Some(Err(ModelSocketError::Json(e)))),
                    }
                } else {
                    std::task::Poll::Ready(Some(Err(ModelSocketError::Protocol(
                        "Unexpected binary WebSocket message".to_string(),
                    ))))
                }
            }
            std::task::Poll::Ready(Some(Err(e))) => {
                std::task::Poll::Ready(Some(Err(ModelSocketError::WebSocket(e))))
            }
            std::task::Poll::Ready(None) => std::task::Poll::Ready(None),
            std::task::Poll::Pending => std::task::Poll::Pending,
        }
    }
}

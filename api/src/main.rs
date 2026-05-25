// Rinha de Backend 2026 — fraud detection API
//
// Hand-rolled HTTP/1.1 loop on tokio (current_thread). Two routes:
//   GET  /ready        -> 200 ok
//   POST /fraud-score  -> 200 {"approved":bool,"fraud_score":f}
// Any failure path returns a safe-approve fallback to avoid HTTP 5xx (weight 5).

use std::net::SocketAddr;
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

mod index;
mod parse;
mod vector;

use index::Index;

const READ_BUF: usize = 4096;

#[tokio::main(flavor = "current_thread")]
async fn main() -> std::io::Result<()> {
    let addr: SocketAddr = std::env::var("LISTEN_ADDR")
        .unwrap_or_else(|_| "0.0.0.0:8080".into())
        .parse()
        .expect("LISTEN_ADDR");

    let index_path = std::env::var("INDEX_PATH").unwrap_or_else(|_| "/opt/index.bin".into());
    let idx = Arc::new(Index::open(&index_path).expect("open index"));
    eprintln!("[api] index loaded: {} vectors, dim={}", idx.n, idx.d);

    let listener = TcpListener::bind(addr).await?;
    eprintln!("[api] listening on {}", addr);

    loop {
        let (sock, _) = listener.accept().await?;
        let _ = sock.set_nodelay(true);
        let idx = idx.clone();
        tokio::spawn(async move {
            let _ = handle_conn(sock, idx).await;
        });
    }
}

async fn handle_conn(mut sock: TcpStream, idx: Arc<Index>) -> std::io::Result<()> {
    let mut buf = vec![0u8; READ_BUF];
    let mut len = 0usize;
    loop {
        // 1) Read until we have request headers.
        let headers_end = loop {
            if let Some(pos) = find_double_crlf(&buf[..len]) {
                break pos;
            }
            if len == buf.len() {
                if buf.len() >= 65536 {
                    return Ok(());
                }
                buf.resize(buf.len() * 2, 0);
            }
            let n = sock.read(&mut buf[len..]).await?;
            if n == 0 {
                return Ok(());
            }
            len += n;
        };

        // 2) Parse request line + content-length (copy out small enum to release the borrow).
        let head = {
            let (method, path, content_length, keep_alive) =
                parse_request_head(&buf[..headers_end]);
            let route = classify_route(method, path);
            (route, content_length, keep_alive)
        };
        let (route, content_length, keep_alive) = head;

        // 3) Read body if present.
        let body_start = headers_end + 4;
        let body_end = body_start + content_length;
        while len < body_end {
            if buf.len() < body_end {
                buf.resize(body_end.max(buf.len() * 2), 0);
            }
            let n = sock.read(&mut buf[len..]).await?;
            if n == 0 {
                return Ok(());
            }
            len += n;
        }

        // 4) Route + write response.
        let mut resp_buf: Vec<u8> = Vec::with_capacity(128);
        match route {
            Route::Ready => {
                resp_buf.extend_from_slice(
                    b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: keep-alive\r\n\r\nok",
                );
            }
            Route::FraudScore => {
                let body = &buf[body_start..body_end];
                let (approved, score) = score_request(body, &idx);
                render_response(&mut resp_buf, approved, score, keep_alive);
            }
            Route::Other => {
                resp_buf.extend_from_slice(
                    b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                );
            }
        }
        sock.write_all(&resp_buf).await?;

        // 5) Advance ring buffer.
        buf.copy_within(body_end..len, 0);
        len -= body_end;
        if !keep_alive {
            return Ok(());
        }
    }
}

fn score_request(body: &[u8], idx: &Index) -> (bool, f32) {
    let v = match parse::parse_payload(body) {
        Some(v) => v,
        None => return (true, 0.0),
    };
    let frauds = idx.knn5_count_frauds(&v);
    let score = frauds as f32 / 5.0;
    (score < 0.6, score)
}

fn render_response(out: &mut Vec<u8>, approved: bool, score: f32, keep_alive: bool) {
    let body: &'static [u8] = match (approved, (score * 5.0).round() as i32) {
        (true, 0) => b"{\"approved\":true,\"fraud_score\":0.0}",
        (true, 1) => b"{\"approved\":true,\"fraud_score\":0.2}",
        (true, 2) => b"{\"approved\":true,\"fraud_score\":0.4}",
        (false, 3) => b"{\"approved\":false,\"fraud_score\":0.6}",
        (false, 4) => b"{\"approved\":false,\"fraud_score\":0.8}",
        (false, 5) => b"{\"approved\":false,\"fraud_score\":1.0}",
        _ => b"{\"approved\":true,\"fraud_score\":0.0}",
    };
    let conn: &[u8] = if keep_alive { b"keep-alive" } else { b"close" };
    out.extend_from_slice(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: ");
    let len_buf = itoa_u32(body.len() as u32);
    out.extend_from_slice(&len_buf);
    out.extend_from_slice(b"\r\nConnection: ");
    out.extend_from_slice(conn);
    out.extend_from_slice(b"\r\n\r\n");
    out.extend_from_slice(body);
}

fn itoa_u32(mut n: u32) -> Vec<u8> {
    if n == 0 {
        return vec![b'0'];
    }
    let mut tmp = [0u8; 10];
    let mut i = tmp.len();
    while n > 0 {
        i -= 1;
        tmp[i] = b'0' + (n % 10) as u8;
        n /= 10;
    }
    tmp[i..].to_vec()
}

fn find_double_crlf(buf: &[u8]) -> Option<usize> {
    if buf.len() < 4 {
        return None;
    }
    for i in 0..=buf.len() - 4 {
        if &buf[i..i + 4] == b"\r\n\r\n" {
            return Some(i);
        }
    }
    None
}

#[derive(Clone, Copy)]
enum Route { Ready, FraudScore, Other }

fn classify_route(method: &[u8], path: &[u8]) -> Route {
    match (method, path) {
        (b"GET", b"/ready") => Route::Ready,
        (b"POST", b"/fraud-score") => Route::FraudScore,
        _ => Route::Other,
    }
}

fn parse_request_head(req: &[u8]) -> (&[u8], &[u8], usize, bool) {
    let mut sp1 = 0;
    while sp1 < req.len() && req[sp1] != b' ' {
        sp1 += 1;
    }
    let method = &req[..sp1];
    let mut sp2 = sp1 + 1;
    while sp2 < req.len() && req[sp2] != b' ' {
        sp2 += 1;
    }
    let path = &req[sp1 + 1..sp2];

    let mut content_length = 0usize;
    let mut keep_alive = true;
    let mut i = sp2;
    while i < req.len() {
        while i < req.len() && req[i] != b'\n' {
            i += 1;
        }
        i += 1;
        if i >= req.len() {
            break;
        }
        if header_match(req, i, b"content-length:") {
            let mut j = i + b"content-length:".len();
            while j < req.len() && (req[j] == b' ' || req[j] == b'\t') {
                j += 1;
            }
            let mut n = 0usize;
            while j < req.len() && req[j].is_ascii_digit() {
                n = n * 10 + (req[j] - b'0') as usize;
                j += 1;
            }
            content_length = n;
        } else if header_match(req, i, b"connection:") {
            let mut j = i + b"connection:".len();
            while j < req.len() && (req[j] == b' ' || req[j] == b'\t') {
                j += 1;
            }
            if j + 5 <= req.len() && req[j..j + 5].eq_ignore_ascii_case(b"close") {
                keep_alive = false;
            }
        }
    }
    (method, path, content_length, keep_alive)
}

fn header_match(buf: &[u8], i: usize, needle_lower: &[u8]) -> bool {
    if i + needle_lower.len() > buf.len() {
        return false;
    }
    for k in 0..needle_lower.len() {
        let c = buf[i + k];
        let cl = if c.is_ascii_uppercase() { c + 32 } else { c };
        if cl != needle_lower[k] {
            return false;
        }
    }
    true
}

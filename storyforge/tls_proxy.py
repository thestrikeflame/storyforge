"""A tiny HTTPS front door.

Browsers only allow the microphone on secure pages (https, or localhost). When a certificate
exists in certs/, this listens on the HTTPS port and passes everything through to the normal
site on port 8000, so the same app is reachable both ways with no extra software.
"""
from __future__ import annotations

import asyncio
import ssl

from .config import settings
from .sysinfo import log


async def _pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while True:
            data = await reader.read(65536)
            if not data:
                break
            writer.write(data)
            await writer.drain()
    except (ConnectionError, OSError, ssl.SSLError, asyncio.CancelledError):
        pass
    finally:
        try:
            writer.close()
        except (OSError, RuntimeError):
            pass


async def _handle(client_reader: asyncio.StreamReader, client_writer: asyncio.StreamWriter) -> None:
    try:
        upstream_reader, upstream_writer = await asyncio.open_connection("127.0.0.1", settings.port)
    except OSError:
        client_writer.close()
        return
    await asyncio.gather(
        _pipe(client_reader, upstream_writer),
        _pipe(upstream_reader, client_writer),
        return_exceptions=True,
    )


def certificate_paths():
    return settings.cert_dir / "cert.pem", settings.cert_dir / "key.pem"


def https_ready() -> bool:
    cert, key = certificate_paths()
    return settings.https_port > 0 and cert.is_file() and key.is_file()


async def start_https_proxy() -> asyncio.AbstractServer | None:
    if not https_ready():
        return None
    cert, key = certificate_paths()
    try:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(str(cert), str(key))
        server = await asyncio.start_server(_handle, settings.host, settings.https_port, ssl=context)
    except (OSError, ssl.SSLError) as exc:
        log.warning("https could not start on port %d: %s", settings.https_port, exc)
        return None
    log.info("https (for the microphone) is on port %d", settings.https_port)
    return server

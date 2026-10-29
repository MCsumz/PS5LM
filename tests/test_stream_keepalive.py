"""A streamed reply must leave its keep-alive connection open.

llama-server's chunked_content_provider calls sink.done() and then returns
false on its last pass. cpp-httplib took a false return as a cancelled
response and closed the connection right after one that advertised
Keep-Alive. A client that reuses it at once (Open WebUI sends each tool-call
continuation within a fraction of a second) got a disconnect.
patches/0011 lets a provider that has already called done() end normally.

Each test builds a server with that provider on the vendored cpp-httplib,
either as llama.cpp ships it or with patches/ applied, and sends a second
request on the same connection.
"""

import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
LLAMA = ROOT / 'third_party' / 'llama.cpp'
HTTPLIB = 'vendor/cpp-httplib/httplib.cpp'

SERVER = r'''
#include "httplib.h"
#include <cstdio>
#include <memory>

// /stream ends the way llama-server's chunked_content_provider does: done(),
// then return has_next (false). /cancel returns false without done(), as a
// provider does when the client has gone. Prints the port it listens on.
int main() {
    httplib::Server svr;
    svr.Post("/stream", [](const httplib::Request &, httplib::Response & res) {
        auto left = std::make_shared<int>(3);
        res.set_chunked_content_provider("text/event-stream",
            [left](size_t, httplib::DataSink & sink) -> bool {
                const bool has_next = --*left > 0;
                const std::string chunk = "data: x\n\n";
                if (!sink.write(chunk.data(), chunk.size())) {
                    return false;
                }
                if (!has_next) {
                    sink.done();
                }
                return has_next;
            });
    });
    svr.Post("/cancel", [](const httplib::Request &, httplib::Response & res) {
        res.set_chunked_content_provider("text/event-stream",
            [](size_t, httplib::DataSink & sink) -> bool {
                const std::string chunk = "data: x\n\n";
                sink.write(chunk.data(), chunk.size());
                return false;
            });
    });
    svr.Post("/plain", [](const httplib::Request &, httplib::Response & res) {
        res.set_content("ok", "text/plain");
    });
    const int port = svr.bind_to_any_port("127.0.0.1");
    std::printf("%d\n", port);
    std::fflush(stdout);
    svr.listen_after_bind();
}
'''


def post(path):
    return (f'POST {path} HTTP/1.1\r\nHost: 127.0.0.1\r\n'
            'Content-Length: 0\r\n\r\n').encode()


def read_response(sock):
    """Read one response; a chunked body ends at the zero chunk."""
    data = b''
    while b'\r\n\r\n' not in data:
        chunk = sock.recv(4096)
        if not chunk:
            raise EOFError('connection closed before a response')
        data += chunk
    head, body = data.split(b'\r\n\r\n', 1)
    if b'chunked' in head.lower():
        while not body.endswith(b'0\r\n\r\n'):
            chunk = sock.recv(4096)
            if not chunk:
                raise EOFError('connection closed inside the body')
            body += chunk
    else:
        length = int(re.search(rb'(?i)content-length:\s*(\d+)', head).group(1))
        while len(body) < length:
            body += sock.recv(4096)
    return head


def vendored_httplib(directory, patched):
    """Write httplib.h/.cpp as the submodule commit has them, optionally with
    patches/ applied, whatever the checkout's working tree holds."""
    for name in ('vendor/cpp-httplib/httplib.h', HTTPLIB):
        source = subprocess.run(['git', '-C', str(LLAMA), 'show', f'HEAD:{name}'],
                                check=True, capture_output=True).stdout
        target = Path(directory) / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source)
    if patched:
        for patch in sorted((ROOT / 'patches').glob('*.patch')):
            if HTTPLIB in patch.read_text():
                subprocess.run(['git', 'apply', f'--include={HTTPLIB}', str(patch)],
                               cwd=directory, check=True)
    return Path(directory) / 'vendor' / 'cpp-httplib'


class StreamKeepAlive(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which(os.environ.get('CXX', 'c++'))
        if not compiler or not (LLAMA / HTTPLIB).is_file():
            raise unittest.SkipTest('needs a host C++ compiler and the llama.cpp submodule')
        cls.tmp = tempfile.TemporaryDirectory()
        source = Path(cls.tmp.name) / 'server.cpp'
        source.write_text(SERVER)
        cls.binaries = {}
        for patched in (False, True):
            include = vendored_httplib(Path(cls.tmp.name) / str(patched), patched)
            binary = Path(cls.tmp.name) / f'server-{patched}'
            subprocess.run([compiler, '-std=c++17', '-O1', '-pthread', f'-I{include}', str(source),
                            str(include / 'httplib.cpp'), '-o', str(binary)], check=True)
            cls.binaries[patched] = binary

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def second_request_on_same_connection(self, patched, first):
        process = subprocess.Popen([str(self.binaries[patched])], stdout=subprocess.PIPE, text=True)
        try:
            port = int(process.stdout.readline())
            with socket.create_connection(('127.0.0.1', port), timeout=10) as sock:
                sock.sendall(post(first))
                try:
                    read_response(sock)
                except EOFError:
                    pass  # a cancelled stream ends without its zero chunk
                time.sleep(0.2)  # the server has finished with the first request
                sock.sendall(post('/plain'))
                return read_response(sock).split(b'\r\n', 1)[0]
        finally:
            process.terminate()
            process.wait(timeout=10)
            process.stdout.close()

    def test_unpatched_httplib_closes_after_a_finished_stream(self):
        # Documents the cause: if this starts passing a second request, the
        # vendored cpp-httplib changed and patches/0011 may no longer be needed.
        with self.assertRaises((EOFError, ConnectionResetError, BrokenPipeError)):
            self.second_request_on_same_connection(False, '/stream')

    def test_patched_httplib_keeps_the_connection_after_a_finished_stream(self):
        self.assertEqual(self.second_request_on_same_connection(True, '/stream'), b'HTTP/1.1 200 OK')

    def test_patched_httplib_still_closes_a_cancelled_stream(self):
        with self.assertRaises((EOFError, ConnectionResetError, BrokenPipeError)):
            self.second_request_on_same_connection(True, '/cancel')


class ProviderShape(unittest.TestCase):
    def test_server_http_ends_a_stream_with_done_then_has_next(self):
        # The harness copies this shape; if llama.cpp changes how a stream
        # ends, revisit the harness and patches/0011.
        source = (LLAMA / 'tools/server/server-http.cpp').read_text()
        provider = re.search(r'chunked_content_provider = \[.*?\n        \};', source, re.S)
        self.assertIsNotNone(provider, 'chunked_content_provider not found')
        self.assertRegex(provider.group(0), r'sink\.done\(\);[^}]*\}\s*return has_next;\s*\};$')


if __name__ == '__main__':
    unittest.main()

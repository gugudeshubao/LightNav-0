"""Loopback-only TCP fault relay; forwards real bytes, never model responses."""
import select
import socket
import threading


class FaultRelay:
    def __init__(self, target_port=19882):
        self.target_port = target_port
        self.lock = threading.Lock()
        self.down = False
        self.stopped = False
        self.sockets = set()
        self.workers = []
        self.listener = socket.socket()
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen()
        self.listener.settimeout(0.1)
        self.port = self.listener.getsockname()[1]
        self.thread = threading.Thread(target=self._accept, daemon=True)
        self.thread.start()

    @staticmethod
    def _close(sock):
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        sock.close()

    def set_down(self, down):
        with self.lock:
            self.down = down
            if down:
                for sock in list(self.sockets):
                    self._close(sock)
                self.sockets.clear()

    def _accept(self):
        while not self.stopped:
            try:
                client, _ = self.listener.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            worker = threading.Thread(target=self._forward, args=(client,), daemon=True)
            self.workers.append(worker)
            worker.start()

    def _forward(self, client):
        upstream = None
        try:
            with self.lock:
                if self.down or self.stopped:
                    return
                self.sockets.add(client)
            upstream = socket.create_connection(("127.0.0.1", self.target_port), timeout=1)
            with self.lock:
                if self.down or self.stopped:
                    return
                self.sockets.add(upstream)
            while not self.stopped:
                readable, _, _ = select.select([client, upstream], [], [], 0.1)
                for source in readable:
                    data = source.recv(65536)
                    if not data:
                        return
                    (upstream if source is client else client).sendall(data)
        except (OSError, ValueError):
            pass  # A fault closes the real socket; client observes transport failure.
        finally:
            with self.lock:
                for sock in (client, upstream):
                    if sock is not None:
                        self.sockets.discard(sock)
                        self._close(sock)

    def close(self):
        self.stopped = True
        self.set_down(True)
        self.listener.close()
        self.thread.join(timeout=2)
        for worker in self.workers:
            worker.join(timeout=2)

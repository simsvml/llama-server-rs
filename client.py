import asyncio
from contextlib import asynccontextmanager
import json


def try_index(haystack, needle):
    try:
        return haystack.index(needle)
    except ValueError:
        return None

class Client:
    def __init__(self, socket):
        self.socket = socket
        self.recv_buf = bytearray()

    def send_json(self, x):
        self.socket.send(json.dumps(x).encode('utf-8') + b'\n')

    def recv_json_raw(self):
        '''Receive a newline-terminated JSON value from the socket.'''
        idx = try_index(self.recv_buf, b'\n')
        while idx is None:
            chunk = self.socket.recv(4096)
            if len(chunk) == 0:
                raise ValueError('socket closed')
            idx = try_index(chunk, b'\n')
            if idx is not None:
                idx += len(self.recv_buf)
            self.recv_buf.extend(chunk)
        j = json.loads(self.recv_buf[:idx].decode('utf-8'))
        # Delete the JSON part and the trailing '\n'
        del self.recv_buf[: idx + 1]
        return j

    def recv_json(self):
        '''Receive a newline-terminated JSON value from the socket.  If the
        value is not an object, or the object has `"kind": "error"`, this
        throws an exception.'''
        x = self.recv_json_raw()
        if not isinstance(x, dict):
            raise ValueError('expected dict, but got %s' % (type(x),))
        if x.get('kind') == 'error':
            msg = x.get('msg', 'received error with no msg')
            raise ValueError(msg)
        return x

    def recv_bytes_into(self, dest):
        '''Receive bytes into the buffer object `dest`.  This will keep going
        until the buffer is full.'''
        mv = memoryview(dest).cast('B')

        copy_amount = min(len(self.recv_buf), len(mv))
        if copy_amount > 0:
            mv[:copy_amount] = self.recv_buf[:copy_amount]
            del self.recv_buf[:copy_amount]
            mv = mv[copy_amount:]

        while len(mv) > 0:
            n = self.socket.recv_into(mv)
            if n == 0:
                raise ValueError('socket closed')
            mv = mv[n:]

    def recv_hidden_states(self):
        '''Receive a `hidden_states` request.  Returns a dict mapping layer
        index to an `ndarray` containing the hidden states.'''
        import numpy as np

        resp1 = self.recv_json()
        assert resp1.get('kind') == 'hidden_states', \
                'expected "hidden_states" response, but got %r' % (resp1.get('kind'),)

        n_prompt = resp1['n_prompt']
        n_layer = resp1['n_layer']
        n_embd = resp1['n_embd']

        hidden_states = {i: np.zeros((n_prompt, n_embd), dtype=np.float32)
            for i in range(n_layer)}

        progress = 0
        progress_max = n_layer * n_prompt
        last_reported_progress = 0

        while True:
            resp = self.recv_json()
            if 'kind' not in resp:
                # This is a chunk header, which is sent in abbreviated form.
                layer = resp['l']
                prompt_idxs = resp['p']
                layer_hidden_states = hidden_states[layer]
                for prompt_idx in prompt_idxs:
                    self.recv_bytes_into(layer_hidden_states[prompt_idx])
                progress += len(prompt_idxs)
                if progress >= last_reported_progress + 1000 or progress == progress_max:
                    print('receiving hidden states: %d/%d' % (progress, progress_max))
                    last_reported_progress = progress
            else:
                assert resp.get('kind') == 'done', \
                        'expected "done" response, but got %r' % (resp1.get('kind'),)
                break

        return hidden_states


@asynccontextmanager
async def async_client(path):
    c = AsyncClient()
    try:
        await c.connect(path)
    except Exception as e:
        print('failed to connect to %r: %s' % (path, e))
        # FIXME: Figure out why this exception gets swallowed
        raise
    yield c
    await c.close()

class AsyncClient:
    def __init__(self):
        self.reader = None
        self.writer = None

    async def connect(self, path):
        r, w = await asyncio.open_unix_connection(path)
        self.reader = r
        self.writer = w

    async def close(self):
        w = self.writer
        self.reader = None
        self.writer = None
        w.close()
        await w.wait_closed()

    async def send_json(self, x):
        self.writer.write(json.dumps(x).encode('utf-8'))
        self.writer.write(b'\n')
        await self.writer.drain()

    async def recv_json_raw(self):
        '''Receive a newline-terminated JSON value from the socket.'''
        line = await self.reader.readline()
        if len(line) == 0:
            raise ValueError('socket closed')
        line = line.rstrip(b'\n')
        return json.loads(line.decode('utf-8'))

    async def recv_json(self):
        '''Receive a newline-terminated JSON value from the socket.  If the
        value is not an object, or the object has `"kind": "error"`, this
        throws an exception.'''
        x = await self.recv_json_raw()
        if not isinstance(x, dict):
            raise ValueError('expected dict, but got %s' % (type(x),))
        if x.get('kind') == 'error':
            msg = x.get('msg', 'received error with no msg')
            raise ValueError(msg)
        return x

    async def recv_bytes_into(self, dest):
        '''Receive bytes into the buffer object `dest`.  This will keep going
        until the buffer is full.'''
        mv = memoryview(dest).cast('B')

        while len(mv) > 0:
            limit = min(len(mv), 64 * 1024)
            b = await self.reader.read(limit)
            n = len(b)
            if n == 0:
                raise ValueError('socket closed')
            mv[:n] = b
            mv = mv[n:]

    async def recv_hidden_states(self):
        '''Receive a `hidden_states` request.  Returns a dict mapping layer
        index to an `ndarray` containing the hidden states.'''
        import numpy as np

        resp1 = await self.recv_json()
        assert resp1.get('kind') == 'hidden_states', \
                'expected "hidden_states" response, but got %r' % (resp1.get('kind'),)

        n_prompt = resp1['n_prompt']
        n_layer = resp1['n_layer']
        n_embd = resp1['n_embd']

        hidden_states = {i: np.zeros((n_prompt, n_embd), dtype=np.float32)
            for i in range(n_layer)}

        progress = 0
        progress_max = n_layer * n_prompt
        last_reported_progress = 0

        while True:
            resp = await self.recv_json()
            if 'kind' not in resp:
                # This is a chunk header, which is sent in abbreviated form.
                layer = resp['l']
                prompt_idxs = resp['p']
                layer_hidden_states = hidden_states[layer]
                for prompt_idx in prompt_idxs:
                    await self.recv_bytes_into(layer_hidden_states[prompt_idx])
                progress += len(prompt_idxs)
                if progress >= last_reported_progress + 1000 or progress == progress_max:
                    print('receiving hidden states: %d/%d' % (progress, progress_max))
                    last_reported_progress = progress
            else:
                assert resp.get('kind') == 'done', \
                        'expected "done" response, but got %r' % (resp1.get('kind'),)
                break

        return hidden_states

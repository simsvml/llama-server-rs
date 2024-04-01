from aiohttp import web
from collections import deque
from contextlib import contextmanager, asynccontextmanager
import argparse
import asyncio
import json
import logging

from client import async_client


def parse_args():
    parser = argparse.ArgumentParser(
            description='produce multiple completions of a single prompt')
    parser.add_argument('port', metavar='PORT', nargs='?',
            type=int, default=8080,
            help='port number to listen on')
    parser.add_argument('--host', metavar='HOST',
            default='127.0.0.1',
            help='host IP to bind to')

    return parser.parse_args()


LLAMA_CLIENT_LOCK = asyncio.Lock();

@asynccontextmanager
async def llama_client():
    async with LLAMA_CLIENT_LOCK:
        async with async_client('./llama.socket') as c:
            yield c


class StreamingCompletion:
    def __init__(self):
        self.queue = deque(())
        self.event = asyncio.Event()
        self.done = False

    @staticmethod
    @contextmanager
    def start(backend_req):
        '''Start a streaming completion request.  The request will be cancelled
        automatically upon exit from the context manager if it's still ongoing.
        Yields the `StreamingCompletion` object, which can be iterated over
        with `async for` to obtain the response messages.'''
        sc = StreamingCompletion()
        task = asyncio.create_task(sc.backend_run(backend_req))
        yield sc
        task.cancel()

    async def backend_run(self, backend_req):
        try:
            async with llama_client() as c:
                await c.send_json(backend_req)

                while True:
                    # Get the first response chunk and decide whether to return an error
                    # code.
                    backend_resp = await c.recv_json_raw()
                    self.queue.append(backend_resp)
                    self.event.set()
                    if backend_resp.get('kind') == 'done':
                        break
        finally:
            # Always notify the main task when terminating.
            self.done = True
            self.event.set()

    async def __aiter__(self):
        while True:
            while not self.queue and not self.done:
                await self.event.wait()
            while self.queue:
                yield self.queue.popleft()
            if self.done:
                break
            self.event.clear()


async def make_event_stream_response(request):
    response = web.StreamResponse()
    response.headers['Content-Type'] = 'text/event-stream'
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['Connection'] = 'keep-alive'
    response.enable_chunked_encoding()
    await response.prepare(request)
    return response


UNSET = object()

def pop_field(obj, key, expect_type=None, default=UNSET):
    val = obj.pop(key, UNSET)
    if val is UNSET:
        if default is not UNSET:
            return default
        else:
            raise web.HTTPBadRequest(text='missing %s' % key)
    if expect_type is not None:
        if expect_type is int and isinstance(val, float) and val % 1 == 0:
            val = int(val)
        if expect_type is float and isinstance(val, int):
            val = float(val)
        if not isinstance(val, expect_type):
            raise web.HTTPBadRequest(text='bad value for %s' % key)
    return val

def kobold_convert_request(json_in):
    json_out = {}
    json_out['prompt'] = pop_field(json_in, 'prompt')
    if (val := pop_field(json_in, 'max_length', int, None)) is not None:
        json_out['n_predict'] = val
    if (val := pop_field(json_in, 'control_vectors', list, None)) is not None:
        json_out['control_vectors'] = val
    if (val := pop_field(json_in, 'memory', str, None)) is not None:
        json_out['prompt_prefix'] = val

    sampler_order = pop_field(json_in, 'sampler_order', list, KOBOLD_DEFAULT_SAMPLER_ORDER)
    samplers = []
    for sampler_idx in sampler_order:
        if sampler_idx == 0:
            if (val := pop_field(json_in, 'top_k', int, None)) is not None:
                samplers.append({'top_k': val})
        elif sampler_idx == 1:
            if (val := pop_field(json_in, 'top_a', float, None)) is not None:
                if val != 0:
                    raise web.HTTPBadRequest(text='top_a sampler is not supported')
        elif sampler_idx == 2:
            if (val := pop_field(json_in, 'top_p', float, None)) is not None:
                samplers.append({'top_p': val})
            if (val := pop_field(json_in, 'min_p', float, None)) is not None:
                samplers.append({'min_p': val})
        elif sampler_idx == 3:
            if (val := pop_field(json_in, 'tfs', float, None)) is not None:
                samplers.append({'tail_free': val})
        elif sampler_idx == 4:
            if (val := pop_field(json_in, 'typical', float, None)) is not None:
                samplers.append({'typical': val})
        elif sampler_idx == 5:
            if (val := pop_field(json_in, 'temperature', float, None)) is not None:
                samplers.append({'temp': val})
        elif sampler_idx == 6:
            if (val := pop_field(json_in, 'rep_pen', float, None)) is not None:
                if val != 1:
                    raise web.HTTPBadRequest(text='rep_pen sampler is not supported')
        else:
            raise web.HTTPBadRequest(text='unknown sampler kind %d' % sampler_idx)
    if len(samplers) > 0:
        json_out['samplers'] = samplers

    return json_out

KOBOLD_DEFAULT_SAMPLER_ORDER = (6, 0, 1, 3, 4, 2, 5)
async def kobold_generate_stream(request):
    request_json = await request.json()
    backend_req = kobold_convert_request(request_json)
    backend_req['kind'] = 'streaming_completion'

    async def init_response():
        return await make_event_stream_response(request)

    response = None

    with StreamingCompletion.start(backend_req) as sc:
        async for backend_resp in sc:
            kind = backend_resp.get('kind')
            if kind == 'error':
                print('got error from backend: %r' % (backend_resp,))
                if response is None:
                    raise web.HTTPBadRequest(text=backend_resp.get('msg', 'unknown error'))
                else:
                    # We don't have a good way to report the error.  Just close
                    # the connection and let the client have the partial
                    # results that were already sent.
                    break
            elif kind == 'done':
                break
            elif kind is None and (token := backend_resp.get('t')) is not None:
                if response is None:
                    response = await init_response()
                await response.write(b'event: message\r\ndata: %s\r\n\r\n' %
                    json.dumps({'token': token}).encode('utf-8'))
            else:
                print('bad response from backend: %r' % (backend_resp,))
                if response is None:
                    raise web.HTTPInternalServerError()
                else:
                    break

    if response is None:
        response = await init_response()
    return response

async def kobold_token_count(request):
    request_json = await request.json()
    prompt = pop_field(request_json, 'prompt', str)

    async with llama_client() as c:
        await c.send_json({
            'kind': 'tokenize',
            'prompt': prompt,
            })
        backend_resp = await c.recv_json_raw()

    kind = backend_resp.get('kind')
    if kind == 'error':
        print('got error from backend: %r' % (backend_resp,))
        raise web.HTTPBadRequest(text=backend_resp.get('msg', 'unknown error'))
    elif kind != 'tokenize':
        print('bad response from backend: %r' % (backend_resp,))
        raise web.HTTPInternalServerError()

    tokens = backend_resp['tokens']
    return web.json_response({
        'value': len(tokens),
        'ids': tokens,
        })


async def openai_models(request):
    model_obj = {
            'object': 'model',
            'id': 'dummy/model',
            'created': 1,
            'owned_by': 'dummy',
            'root': 'dummy',
            'permission': [],
            }

    obj = {
            'object': 'list',
            'data': [model_obj],
            }

    return web.json_response(obj)


LLAMA_DEFAULT_SAMPLER_ORDER = ('top_k', 'tfs_z', 'typical_p', 'top_p', 'min_p', 'temperature')

def llama_convert_request(json_in):
    json_out = {}
    json_out['prompt'] = pop_field(json_in, 'prompt')
    if (val := pop_field(json_in, 'n_predict', int, None)) is not None:
        json_out['n_predict'] = val
    if (val := pop_field(json_in, 'control_vectors', list, None)) is not None:
        json_out['control_vectors'] = val

    sampler_order = pop_field(json_in, 'samplers', list, LLAMA_DEFAULT_SAMPLER_ORDER)
    samplers = []
    for sampler_kind in sampler_order:
        if sampler_kind == 'top_k':
            if (val := pop_field(json_in, 'top_k', int, None)) is not None:
                samplers.append({'top_k': val})
        elif sampler_kind == 'top_p':
            if (val := pop_field(json_in, 'top_p', float, None)) is not None:
                samplers.append({'top_p': val})
        elif sampler_kind == 'min_p':
            if (val := pop_field(json_in, 'min_p', float, None)) is not None:
                samplers.append({'min_p': val})
        elif sampler_kind == 'tfs_z':
            if (val := pop_field(json_in, 'tfs_z', float, None)) is not None:
                samplers.append({'tail_free': val})
        elif sampler_kind == 'typical_p':
            if (val := pop_field(json_in, 'typical_p', float, None)) is not None:
                samplers.append({'typical': val})
        elif sampler_kind == 'temperature':
            if (val := pop_field(json_in, 'temperature', float, None)) is not None:
                samplers.append({'temp': val})
        else:
            raise web.HTTPBadRequest(text='unknown sampler kind %r' % (sampler_kind,))
    if len(samplers) > 0:
        json_out['samplers'] = samplers

    return json_out

async def llama_completion(request):
    request_json = await request.json()
    backend_req = llama_convert_request(request_json)
    backend_req['kind'] = 'streaming_completion'
    stream = pop_field(request_json, 'stream', bool, False)

    if not stream:
        raise web.HTTPBadRequest(text='non-streaming completion is not supported')

    async def init_response():
        return await make_event_stream_response(request)

    response = None

    with StreamingCompletion.start(backend_req) as sc:
        prev_token = None
        async for backend_resp in sc:
            kind = backend_resp.get('kind')
            if kind == 'error':
                print('got error from backend: %r' % (backend_resp,))
                if response is None:
                    raise web.HTTPBadRequest(text=backend_resp.get('msg', 'unknown error'))
                else:
                    await response.write(b'error: %s\n\n' %
                        json.dumps({'msg': backend_resp['msg']}).encode('utf-8'))
                    break
            elif kind == 'done':
                break
            elif kind is None and (token := backend_resp.get('t')) is not None:
                if response is None:
                    response = await init_response()
                if prev_token is not None:
                    await response.write(b'data: %s\n\n' %
                        json.dumps({'content': prev_token, 'stop': False}).encode('utf-8'))
                prev_token = token
            else:
                print('bad response from backend: %r' % (backend_resp,))
                if response is None:
                    raise web.HTTPInternalServerError()
                else:
                    await response.write(b'error: %s\n\n' %
                        json.dumps({'msg': 'internal server error'}).encode('utf-8'))
                    break

    if response is None:
        response = await init_response()
    if prev_token is not None:
        await response.write(b'data: %s\n\n' %
            json.dumps({'content': prev_token, 'stop': True}).encode('utf-8'))
    return response


def main():
    logging.basicConfig(level=logging.INFO)

    args = parse_args()

    app = web.Application()
    app.add_routes([
        web.get('/v1/models', openai_models),
        web.post('/completion', llama_completion),
        web.post('/api/extra/generate/stream', kobold_generate_stream),
        web.post('/api/extra/tokencount', kobold_token_count),
        ])

    web.run_app(app, host=args.host, port=args.port)

if __name__ == '__main__':
    main()

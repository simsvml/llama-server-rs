from aiohttp import web
from contextlib import asynccontextmanager
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

    buffer = []
    event = asyncio.Event()
    cancel = False

    async def backend_run():
        try:
            async with llama_client() as c:
                await c.send_json(backend_req)

                while not cancel:
                    # Get the first response chunk and decide whether to return an error
                    # code.
                    backend_resp = await c.recv_json_raw()
                    buffer.append(backend_resp)
                    event.set()
                    if backend_resp.get('kind') == 'done':
                        break
        finally:
            # Always notify the main task when terminating.
            buffer.append(None)
            event.set()

    asyncio.create_task(backend_run())

    async def init_response():
        response = web.StreamResponse()
        response.headers['Content-Type'] = 'text/event-stream'
        response.headers['Cache-Control'] = 'no-cache'
        response.headers['Connection'] = 'keep-alive'
        response.enable_chunked_encoding()
        await response.prepare(request)
        return response

    try:
        response = None
        done = False
        while not done:
            await event.wait()

            buffer_copy = buffer[:]
            buffer.clear()
            event.clear()

            for backend_resp in buffer_copy:
                if backend_resp is None:
                    # Background task has terminated.
                    done = True
                    break

                kind = backend_resp.get('kind')
                if kind == 'error':
                    print('got error from backend: %r' % (backend_resp,))
                    if response is None:
                        raise web.HTTPBadRequest(text=backend_resp.get('msg', 'unknown error'))
                    else:
                        # We don't have a good way to report the error.  Just
                        # close the connection and let the client have the
                        # partial results that were already sent.
                        done = True
                        break
                elif kind == 'done':
                    done = True
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
                        done = True
                        break

    finally:
        cancel = True

    if response is None:
        response = await init_response()
    return response


def main():
    logging.basicConfig(level=logging.INFO)

    args = parse_args()

    app = web.Application()
    app.add_routes([
        web.post('/api/extra/generate/stream', kobold_generate_stream),
        ])

    web.run_app(app, host=args.host, port=args.port)

if __name__ == '__main__':
    main()

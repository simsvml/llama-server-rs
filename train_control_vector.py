import argparse
import gguf
import json
import numpy as np
from sklearn.decomposition import PCA
import socket
import sys

def parse_args():
    parser = argparse.ArgumentParser(
            description='train a control vector from a set of prompts')
    parser.add_argument('prompts', metavar='PROMPTS.TXT',
            help='file to read the prompts from')
    parser.add_argument('control_vector', metavar='CVEC.GGUF',
            help='where to write the control vector')
    return parser.parse_args()


########################################
# The following code is copied from vgel/repeng, under the following license:
#
# MIT License
# 
# Copyright (c) 2024 Theia Vogel
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

def project_onto_direction(H, direction):
    """Project matrix H (n, d_1) onto direction vector (d_2,)"""
    mag = np.linalg.norm(direction)
    assert not np.isinf(mag)
    return (H @ direction) / mag

def read_representations(
    layer_hiddens: dict[int, np.ndarray],
) -> dict[int, np.ndarray]:
    """
    Extract the representations based on the contrast dataset.
    """

    hidden_layers = sorted(layer_hiddens.keys())
    num_inputs = next(iter(layer_hiddens.values())).shape[0] // 2
    print('%d inputs' % num_inputs)

    # get differences between (positive, negative) pairs
    relative_layer_hiddens = {}
    for layer in hidden_layers:
        relative_layer_hiddens[layer] = (
            layer_hiddens[layer][::2] - layer_hiddens[layer][1::2]
        )

    # get directions for each layer using PCA
    directions: dict[int, np.ndarray] = {}
    for layer in hidden_layers:
        assert layer_hiddens[layer].shape[0] == num_inputs * 2
        print('running PCA for layer %d/%d' % (layer + 1, len(hidden_layers)))

        # fit layer directions
        train = np.vstack(
            relative_layer_hiddens[layer]
            - relative_layer_hiddens[layer].mean(axis=0, keepdims=True)
        )
        pca_model = PCA(n_components=1, whiten=False).fit(train)
        # shape (n_features,)
        directions[layer] = pca_model.components_.astype(np.float32).squeeze(axis=0)

        # calculate sign
        projected_hiddens = project_onto_direction(
            layer_hiddens[layer], directions[layer]
        )

        # order is [positive, negative, positive, negative, ...]
        positive_smaller_mean = np.mean(
            [
                projected_hiddens[i] < projected_hiddens[i + 1]
                for i in range(0, num_inputs * 2, 2)
            ]
        )
        positive_larger_mean = np.mean(
            [
                projected_hiddens[i] > projected_hiddens[i + 1]
                for i in range(0, num_inputs * 2, 2)
            ]
        )

        if positive_smaller_mean > positive_larger_mean:  # type: ignore
            directions[layer] *= -1

    return directions

def export_gguf(directions, path: str):
    """
    Export a trained ControlVector to a llama.cpp .gguf file.
    """

    arch = "controlvector"
    writer = gguf.GGUFWriter(path, arch)
    #writer.add_string(f"{arch}.model_hint", model_type)
    #writer.add_uint32(f"{arch}.layer_count", len(directions))
    for layer in directions.keys():
        if layer == 0:
            # For some reason, llama.cpp bails out if it sees a direction.0
            # tensor.
            continue
        writer.add_tensor(f"direction.{layer}", directions[layer])
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

# End of code copied from vgel/repeng
########################################


def try_index(haystack, needle):
    try:
        return haystack.index(needle)
    except ValueError:
        return None

class SocketReceiver:
    def __init__(self, socket):
        self.socket = socket
        self.buf = bytearray()

    def recv_json(self):
        '''Receive a newline-terminated JSON object from the socket.'''
        idx = try_index(self.buf, b'\n')
        while idx is None:
            chunk = self.socket.recv(4096)
            if len(chunk) == 0:
                raise ValueError('socket closed')
            idx = try_index(chunk, b'\n')
            if idx is not None:
                idx += len(self.buf)
            self.buf.extend(chunk)
        j = json.loads(self.buf[:idx].decode('utf-8'))
        # Delete the JSON part and the trailing '\n'
        del self.buf[: idx + 1]
        return j

    def recv_bytes_into(self, dest):
        '''Receive bytes into the buffer object `dest`.  This will keep going
        until the buffer is full.'''
        mv = memoryview(dest).cast('B')

        copy_amount = min(len(self.buf), len(mv))
        if copy_amount > 0:
            mv[:copy_amount] = self.buf[:copy_amount]
            del self.buf[:copy_amount]
            mv = mv[copy_amount:]

        while len(mv) > 0:
            n = self.socket.recv_into(mv)
            if n == 0:
                raise ValueError('socket closed')
            mv = mv[n:]


def main():
    args = parse_args()

    prompts = open(args.prompts).read().splitlines()
    prompts = [p for p in prompts if p != '']

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.connect('./llama.socket')
        msg = json.dumps({
            'kind': 'hidden_states',
            'prompts': prompts,
        })
        s.send(msg.encode('utf-8') + b'\n')

        r = SocketReceiver(s)
        resp1 = r.recv_json()
        assert resp1.get('kind') == 'hidden_states', \
                'unexpected response from server: %r' % (resp1,)

        n_prompt = resp1['n_prompt']
        n_layer = resp1['n_layer']
        n_embd = resp1['n_embd']

        hidden_states = {i: np.empty((n_prompt, n_embd), dtype=np.float32)
            for i in range(n_layer)}

        progress = 0
        progress_max = n_layer * n_prompt
        last_reported_progress = 0

        while True:
            resp = r.recv_json()
            if 'kind' not in resp:
                # This is a chunk header, which is sent in abbreviated form.
                layer = resp['l']
                prompt_idxs = resp['p']
                layer_hidden_states = hidden_states[layer]
                for prompt_idx in prompt_idxs:
                    r.recv_bytes_into(layer_hidden_states[prompt_idx])
                progress += len(prompt_idxs)
                if progress >= last_reported_progress + 1000 or progress == progress_max:
                    print('receiving hidden states: %d/%d' % (progress, progress_max))
                    last_reported_progress = progress
            else:
                assert resp.get('kind') == 'done', \
                        'unexpected response from server: %r' % (resp,)
                break

        layer_vectors = read_representations(hidden_states)

        print('writing gguf to %s' % (args.control_vector,))
        export_gguf(layer_vectors, args.control_vector)

if __name__ == '__main__':
    main()

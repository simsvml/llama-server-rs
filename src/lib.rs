use std::ffi::CString;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::os::raw::c_void;
use std::ptr::{self, NonNull};
use std::slice;
use std::str::{self, Utf8Error};


pub mod sequence_trie;


pub mod ffi {
    #![allow(non_upper_case_globals)]
    #![allow(non_camel_case_types)]
    #![allow(non_snake_case)]
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}


pub type LlamaToken = ffi::llama_token;
pub type LlamaTokenData = ffi::llama_token_data;
pub type LlamaModelParams = ffi::llama_model_params;
pub type LlamaContextParams = ffi::llama_context_params;


#[derive(Debug)]
pub struct LlamaModel(NonNull<ffi::llama_model>);

impl LlamaModel {
    pub fn load_from_file(
        path: impl Into<Vec<u8>>,
        model_params: ffi::llama_model_params,
    ) -> Option<LlamaModel> {
        unsafe {
            let name = CString::new(path).unwrap();
            let ptr = ffi::llama_load_model_from_file(name.as_ptr(), model_params);
            let ptr = NonNull::new(ptr)?;
            Some(LlamaModel(ptr))
        }
    }

    pub fn as_ptr(&self) -> *mut ffi::llama_model {
        self.0.as_ptr()
    }

    pub fn n_vocab(&self) -> usize {
        unsafe {
            ffi::llama_n_vocab(self.as_ptr()).try_into().unwrap()
        }
    }

    pub fn n_embd(&self) -> usize {
        unsafe {
            ffi::llama_n_embd(self.as_ptr()).try_into().unwrap()
        }
    }

    pub fn n_layer(&self) -> usize {
        unsafe {
            ffi::llama_n_layer(self.as_ptr()).try_into().unwrap()
        }
    }

    pub fn token_bos(&self) -> LlamaToken {
        unsafe {
            ffi::llama_token_bos(self.as_ptr())
        }
    }

    pub fn token_eos(&self) -> LlamaToken {
        unsafe {
            ffi::llama_token_eos(self.as_ptr())
        }
    }

    pub fn token_nl(&self) -> LlamaToken {
        unsafe {
            ffi::llama_token_nl(self.as_ptr())
        }
    }

    /// Tokenize `text`, appending the tokens to `tokens`.  This never grows `tokens`; the caller
    /// should ensure it has enough free capacity before the call.  On success, returns `Ok` with
    /// the number of tokens that were added to `tokens`.  On failure (not enough space in
    /// `tokens`), returns `Err(n)` where `n` is the number of tokens found in `text`.  (In the
    /// `Err(n)` case, the caller might want to call `tokens.reserve(n)` and try again.)
    pub fn try_tokenize(
        &self,
        text: &str,
        tokens: &mut Vec<ffi::llama_token>,
        add_bos: bool,
        special: bool,
    ) -> Result<usize, usize> {
        unsafe {
            let space = tokens.spare_capacity_mut();
            let r = ffi::llama_tokenize(
                self.as_ptr(),
                text.as_ptr().cast::<i8>(),
                text.len().try_into().unwrap(),
                space.as_mut_ptr().cast::<ffi::llama_token>(),
                space.len().try_into().unwrap(),
                add_bos,
                special,
            );
            if r < 0 {
                // Not enough space in `tokens`.
                return Err((-r).try_into().unwrap());
            }

            let added = usize::try_from(r).unwrap();
            // Otherwise, `r` tokens were appended to `tokens`.
            debug_assert!(added <= space.len());
            debug_assert_eq!(space.len(), tokens.capacity() - tokens.len());
            // This addition can't overflow because we know the sum doesn't exceed
            // `tokens.capacity()`.
            tokens.set_len(tokens.len() + added);
            Ok(added)
        }
    }

    pub fn try_token_to_piece(
        &self,
        token: ffi::llama_token,
        buf: &mut Vec<u8>,
    ) -> Result<usize, usize> {
        unsafe {
            let space = buf.spare_capacity_mut();
            let r = ffi::llama_token_to_piece(
                self.as_ptr(),
                token,
                space.as_mut_ptr().cast::<i8>(),
                space.len().try_into().unwrap(),
            );
            if r < 0 {
                // Not enough space in `buf`.
                return Err((-r).try_into().unwrap());
            }

            let added = usize::try_from(r).unwrap();
            // Otherwise, `r` bytes were appended to `buf`.
            debug_assert!(added <= space.len());
            debug_assert_eq!(space.len(), buf.capacity() - buf.len());
            // This addition can't overflow because we know the sum doesn't exceed
            // `buf.capacity()`.
            buf.set_len(buf.len() + added);
            Ok(added)
        }
    }

    // TODO: this panics on bad utf8 - do any models produce such output?
    pub fn try_token_to_piece_str<'a>(
        &self,
        token: ffi::llama_token,
        buf: &'a mut Vec<u8>,
    ) -> Result<&'a str, usize> {
        let old_len = buf.len();
        self.try_token_to_piece(token, buf)?;
        let new_len = buf.len();
        let s = str::from_utf8(&buf[old_len..new_len]).unwrap();
        Ok(s)
    }

    /// Convert a token ID to bytes, appending the bytes to `buf`.  Returns a slice of `buf`
    /// containing the new bytes.
    pub fn token_to_piece<'a>(
        &self,
        token: ffi::llama_token,
        buf: &'a mut Vec<u8>,
    ) -> &'a [u8] {
        let old_len = buf.len();
        match self.try_token_to_piece(token, buf) {
            Ok(_) => {},
            Err(need) => {
                buf.reserve(need);
                self.try_token_to_piece(token, buf).unwrap();
            },
        }
        let new_len = buf.len();
        &buf[old_len..new_len]
    }

    pub fn token_to_piece_str<'a>(
        &self,
        token: ffi::llama_token,
        buf: &'a mut Vec<u8>,
    ) -> Result<&'a str, Utf8Error> {
        let old_len = buf.len();
        self.token_to_piece(token, buf);
        let new_len = buf.len();
        let s = str::from_utf8(&buf[old_len..new_len]).unwrap();
        Ok(s)
    }
}

impl Drop for LlamaModel {
    fn drop(&mut self) {
        unsafe {
            ffi::llama_free_model(self.as_ptr());
        }
    }
}

#[derive(Debug)]
pub struct LlamaContext<'a> {
    ptr: NonNull<ffi::llama_context>,
    /// Indicates whether logits are available for a given token index (as used with
    /// `llama_get_logits_ith`).  This is initialized from `batch.logits()` when calling
    /// `self.decode(&batch)`.
    logits_available: Vec<bool>,
    _marker: PhantomData<&'a LlamaModel>,
}

impl<'a> LlamaContext<'a> {
    pub fn with_model(
        model: &'a LlamaModel,
        context_params: ffi::llama_context_params,
    ) -> Option<LlamaContext<'a>> {
        unsafe {
            let ptr = ffi::llama_new_context_with_model(model.as_ptr(), context_params);
            let ptr = NonNull::new(ptr)?;
            Some(LlamaContext {
                ptr,
                logits_available: Vec::new(),
                _marker: PhantomData,
            })
        }
    }

    pub fn with_model_and_eval_callback<F>(
        model: &'a LlamaModel,
        context_params: ffi::llama_context_params,
        callback: &'a mut F,
    ) -> Option<LlamaContext<'a>>
    where F: FnMut(*mut ffi::ggml_tensor, bool) -> bool {
        let mut context_params = context_params;

        unsafe extern "C" fn dispatch<F>(
            t: *mut ffi::ggml_tensor,
            ask: bool,
            user_data: *mut c_void,
        ) -> bool
        where F: FnMut(*mut ffi::ggml_tensor, bool) -> bool {
            let callback = &mut *(user_data.cast::<F>());
            callback(t, ask)
        }
        context_params.cb_eval = Some(dispatch::<F>);
        context_params.cb_eval_user_data = ptr::addr_of_mut!(*callback).cast::<c_void>();

        Self::with_model(model, context_params)
    }


    pub fn as_ptr(&self) -> *mut ffi::llama_context {
        self.ptr.as_ptr()
    }

    pub fn n_ctx(&self) -> usize {
        unsafe {
            ffi::llama_n_ctx(self.as_ptr()).try_into().unwrap()
        }
    }

    pub fn n_batch(&self) -> usize {
        unsafe {
            ffi::llama_n_batch(self.as_ptr()).try_into().unwrap()
        }
    }

    pub fn n_seq_max(&self) -> usize {
        unsafe {
            ffi::llama_n_seq_max(self.as_ptr()).try_into().unwrap()
        }
    }

    /// Process a batch of tokens.
    ///
    /// This takes `&mut`, so it can't be called while a borrow from `logits_ith` is outstanding.
    pub fn decode(&mut self, batch: &LlamaBatch) -> Result<(), String> {
        unsafe {
            self.logits_available = Vec::with_capacity(batch.len());
            let r = ffi::llama_decode(self.as_ptr(), batch.as_raw());
            match r {
                0 => {},
                1 => return Err(format!(
                        "couldn't find kv slot for batch of size {}", batch.len())),
                _ => return Err(format!("llama_decode failed: {}", r)),
            }
            self.logits_available.extend(batch.logits().iter().map(|&x| x != 0));
            Ok(())
        }
    }

    /// Obtain the logits for token `i` from the last batch.
    ///
    /// This borrows from `self`, which prevents `decode()` (which might overwrite the logits
    /// buffer) from being called while the returned slice is still in use.
    pub fn logits_ith<'b>(&'b self, i: usize) -> &'b [f32] {
        unsafe {
            assert!(self.logits_available[i]);

            let model = ffi::llama_get_model(self.as_ptr());
            let n_vocab = ffi::llama_n_vocab(model);
            let ptr = ffi::llama_get_logits_ith(self.as_ptr(), i.try_into().unwrap());
            slice::from_raw_parts(
                ptr,
                n_vocab.try_into().unwrap(),
            )
        }
    }

    /// Clear all data from the KV cache.
    pub fn kv_cache_clear(&self) {
        unsafe {
            ffi::llama_kv_cache_seq_rm(self.as_ptr(), -1, -1, -1);
        }
    }

    pub fn sample_softmax(&self, candidates: &mut LlamaTokenDataArray) {
        unsafe {
            ffi::llama_sample_softmax(self.as_ptr(), candidates.as_mut_ptr())
        }
    }

    pub fn sample_temp(&self, candidates: &mut LlamaTokenDataArray, temp: f32) {
        unsafe {
            ffi::llama_sample_temp(self.as_ptr(), candidates.as_mut_ptr(), temp)
        }
    }

    pub fn sample_token(&self, candidates: &mut LlamaTokenDataArray) -> ffi::llama_token {
        unsafe {
            ffi::llama_sample_token(self.as_ptr(), candidates.as_mut_ptr())
        }
    }
}

impl<'a> Drop for LlamaContext<'a> {
    fn drop(&mut self) {
        unsafe {
            ffi::llama_free(self.as_ptr());
        }
    }
}


pub struct LlamaBatch {
    batch: ffi::llama_batch,
    capacity: usize,
    n_seq_max: usize,
}

impl LlamaBatch {
    pub fn new(n_tokens: usize, n_seq_max: usize) -> LlamaBatch {
        unsafe {
            let batch = ffi::llama_batch_init(
                n_tokens.try_into().unwrap(),
                0,  // embd
                n_seq_max.try_into().unwrap(),
            );
            LlamaBatch {
                batch,
                capacity: n_tokens,
                n_seq_max,
            }
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn len(&self) -> usize {
        // This cast is okay because 0 <= n_tokens <= capacity
        self.batch.n_tokens as usize
    }

    pub fn clear(&mut self) {
        self.batch.n_tokens = 0;
    }

    pub fn push(
        &mut self,
        token: ffi::llama_token,
        pos: usize,
        seq_id: usize,
        logits: bool,
    ) {
        unsafe {
            assert!(self.len() < self.capacity());
            let i = self.len();
            *self.batch.token.add(i) = token;
            *self.batch.pos.add(i) = pos.try_into().unwrap();
            *self.batch.n_seq_id.add(i) = 1;
            *(*self.batch.seq_id.add(i)).add(0) = seq_id.try_into().unwrap();
            *self.batch.logits.add(i) = logits as i8;

            self.batch.n_tokens += 1;
        }
    }

    pub fn add_seq_to_token(&mut self, token_idx: usize, seq_id: usize) {
        let i = token_idx;
        unsafe {
            assert!(i < self.len());
            let n_seq_id_ptr = self.batch.n_seq_id.add(i);
            let seq_id_ptr = *self.batch.seq_id.add(i);

            let n_seq_id = usize::try_from(*n_seq_id_ptr).unwrap();
            assert!(n_seq_id < self.n_seq_max);
            *seq_id_ptr.add(n_seq_id) = seq_id.try_into().unwrap();
            *n_seq_id_ptr = (n_seq_id + 1).try_into().unwrap();
        }
    }

    pub fn logits<'a>(&'a self) -> &'a [i8] {
        unsafe {
            slice::from_raw_parts(
                self.batch.logits,
                self.len(),
            )
        }
    }

    pub fn as_raw(&self) -> ffi::llama_batch {
        self.batch
    }

    pub fn into_inner(self) -> ffi::llama_batch {
        self.batch
    }
}

impl Drop for LlamaBatch {
    fn drop(&mut self) {
        unsafe {
            ffi::llama_batch_free(self.batch);
        }
    }
}


pub struct LlamaTokenDataArray<'a> {
    raw: ffi::llama_token_data_array,
    _marker: PhantomData<&'a mut [ffi::llama_token_data]>,
}

impl<'a> LlamaTokenDataArray<'a> {
    pub fn new(buf: &'a mut [ffi::llama_token_data]) -> LlamaTokenDataArray<'a> {
        LlamaTokenDataArray {
            raw: ffi::llama_token_data_array {
                data: buf.as_mut_ptr(),
                size: buf.len().try_into().unwrap(),
                sorted: false,
            },
            _marker: PhantomData,
        }
    }

    pub fn as_mut_ptr(&mut self) -> *mut ffi::llama_token_data_array {
        ptr::addr_of_mut!(self.raw)
    }
}

impl Deref for LlamaTokenDataArray<'_> {
    type Target = [LlamaTokenData];

    fn deref(&self) -> &[LlamaTokenData] {
        unsafe {
            // This is safe because `data` and `size` are initially set by deconstructing a slice,
            // and the llama.cpp methods we use don't modify `data` or increase `size`.
            slice::from_raw_parts(
                self.raw.data,
                self.raw.size as usize,
            )
        }
    }
}

impl DerefMut for LlamaTokenDataArray<'_> {
    fn deref_mut(&mut self) -> &mut [LlamaTokenData] {
        unsafe {
            // This is safe because `data` and `size` are initially set by deconstructing a slice,
            // and the llama.cpp methods we use don't modify `data` or increase `size`.
            slice::from_raw_parts_mut(
                self.raw.data,
                self.raw.size as usize,
            )
        }
    }
}


pub fn default_model_params() -> ffi::llama_model_params {
    unsafe { ffi::llama_model_default_params() }
}

pub fn default_context_params() -> ffi::llama_context_params {
    unsafe { ffi::llama_context_default_params() }
}


pub fn dump_batch(batch: ffi::llama_batch) {
    unsafe {
        unsafe fn mk_slice<'a, T>(ptr: *const T, len: usize) -> &'a [T] {
            if ptr.is_null() {
                &[]
            } else {
                std::slice::from_raw_parts(ptr, len)
            }
        }

        eprintln!("batch: {} tokens", batch.n_tokens);
        let n = usize::try_from(batch.n_tokens).unwrap();

        let token = mk_slice(batch.token, n);
        let pos = mk_slice(batch.pos, n);
        let n_seq_id = mk_slice(batch.n_seq_id, n);
        let seq_id = mk_slice(batch.seq_id, n);
        let logits = mk_slice(batch.logits, n);

        let iter = token.iter()
            .zip(pos.iter())
            .zip(n_seq_id.iter())
            .zip(seq_id.iter())
            .zip(logits.iter());
        for ((((&token, &pos), &n_seq_id), &seq_id), &logits) in iter {
            let seq_id = mk_slice(seq_id, n_seq_id.try_into().unwrap());
            eprintln!("  token {}, pos {}, n_seq_id {}, seq_id {:?}, logits {}",
                token, pos, n_seq_id, seq_id, logits);
        }
    }
}

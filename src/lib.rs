pub mod arith;
pub mod codec;
pub mod lstm;
pub mod lz;
pub mod model;

#[cfg(target_arch = "wasm32")]
mod wasm {
    use crate::codec;

    /// Allocate a buffer the JS side can write input bytes into.
    #[unsafe(no_mangle)]
    pub extern "C" fn mo_alloc(len: usize) -> *mut u8 {
        let mut buf = Vec::<u8>::with_capacity(len);
        let ptr = buf.as_mut_ptr();
        std::mem::forget(buf);
        ptr
    }

    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn mo_free(ptr: *mut u8, len: usize) {
        unsafe { drop(Vec::from_raw_parts(ptr, 0, len)) };
    }

    /// Compress `len` bytes at `ptr`. Returns a pointer to the result;
    /// the result length is written to `out_len`. Caller frees with mo_free.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn mo_compress(
        ptr: *const u8,
        len: usize,
        ai: u32,
        out_len: *mut usize,
    ) -> *mut u8 {
        let data = unsafe { std::slice::from_raw_parts(ptr, len) };
        let result = codec::compress(data, ai != 0);
        unsafe { *out_len = result.len() };
        let mut boxed = result.into_boxed_slice();
        let out = boxed.as_mut_ptr();
        std::mem::forget(boxed);
        out
    }

    /// Decompress a .mo container. Returns null on invalid input.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn mo_decompress(
        ptr: *const u8,
        len: usize,
        out_len: *mut usize,
    ) -> *mut u8 {
        let data = unsafe { std::slice::from_raw_parts(ptr, len) };
        match codec::decompress(data) {
            Ok(result) => {
                unsafe { *out_len = result.len() };
                let mut boxed = result.into_boxed_slice();
                let out = boxed.as_mut_ptr();
                std::mem::forget(boxed);
                out
            }
            Err(_) => {
                unsafe { *out_len = 0 };
                std::ptr::null_mut()
            }
        }
    }
}

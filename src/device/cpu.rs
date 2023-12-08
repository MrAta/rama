use rayon::prelude::*;
use packed_simd::*;

use super::device::*;
pub struct CPU {}
impl Device for CPU {
    type Err = ();
    fn matmul(o: &mut [f32], a: &[f32], b: &[f32], width: usize, _o_rows: usize, o_cols: usize) -> Result<(), ()> {
        if width % f32x4::lanes() != 0 || o_cols % f32x4::lanes() != 0 {
            return Err(()); // Return error if matrix dimensions are not divisible by SIMD lanes
        }

        let a_chunks = a.chunks_exact(width);
        let b_chunks = b.chunks_exact(o_cols);

        for (i, a_chunk) in a_chunks.enumerate() {
            let o_start = i * o_cols;
            for (j, b_chunk) in b_chunks.clone().enumerate() {
                let mut v = f32x4::splat(0.0);
                for k in (0..width).step_by(f32x4::lanes()) {
                    let a_simd = f32x4::from_slice_unaligned(&a_chunk[k..]);
                    let b_simd = f32x4::from_slice_unaligned(&b_chunk[j * o_cols + k..]);
                    v += a_simd * b_simd;
                }
                v.write_to_slice_unaligned(&mut o[o_start + j * f32x4::lanes()..]);
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_mul() {
        let a_host = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_host = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut c_host = [0.0f32; 4];
        let _ = CPU::matmul(&mut c_host, &a_host, &b_host, 3, 2, 2);
        assert_eq!(c_host, [22.0, 28.0, 49.0, 64.0]);

    }

}
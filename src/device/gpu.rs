use cudarc::driver::{CudaDevice, DriverError, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;

use super::device::*;

const PTX_SRC: &str = "
extern \"C\" __global__ void matmul(float* A, float* B, float* C, int width, int C_rows, int C_cols) {
    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;

    float tmpSum = 0;

    if (ROW < C_rows && COL < C_cols) {
        // each thread computes one element of the block sub-matrix
        for (int i = 0; i < width; i++) {
            tmpSum += A[ROW * width + i] * B[i * C_cols + COL];
        }
    }
    C[ROW * C_cols + COL] = tmpSum;
}
";

///
/// Brief Introduction to CUDA Programming
/// blockIdx.x, blockIdx.y, blockIdx.z are built-in variables that returns the block ID
/// in the x-axis, y-axis, and z-axis of the block that is executing the given block of code.
///
/// threadIdx.x, threadIdx.y, threadIdx.z are built-in variables that return the
/// thread ID in the x-axis, y-axis, and z-axis of the thread that is being executed by this
/// stream processor in this particular block.
///
/// blockDim.x, blockDim.y, blockDim.z are built-in variables that return the “block
/// dimension” (i.e., the number of threads in a block in the x-axis, y-axis, and z-axis).
///
/// The full global thread ID in x dimension can be computed by:
///  x = blockIdx.x * blockDim.x + threadIdx.x;
///
/// Personally I found this blog post quite easy to follow or as a reference:
///     https://www.quantstart.com/articles/Matrix-Matrix-Multiplication-on-the-GPU-with-Nvidia-CUDA/
///

// const ROW_TILE_WIDTH: usize = 32;
// const COL_TILE_WIDTH: usize = 32;

pub struct GPU {}
impl Device for GPU {
    type Err = DriverError;
    fn matmul(o: &mut [f32], a: &[f32], b: &[f32], width: usize, o_rows: usize, o_cols: usize) -> Result<(), DriverError> {
        // let start = std::time::Instant::now();
        let ptx = compile_ptx(PTX_SRC).unwrap();

        let dev = CudaDevice::new(0)?;

        dev.load_ptx(ptx, "matmul", &["matmul"]).unwrap();
        let f = dev.get_func("matmul", "matmul").unwrap();
        let w_dev = dev.htod_sync_copy(&a)?;
        let x_dev = dev.htod_sync_copy(&b)?;
        let mut o_dev = dev.htod_sync_copy(&o)?;
        // println!("Copied in {:?}", start.elapsed());

        let cfg = LaunchConfig {
            block_dim: (o_cols as u32, o_rows as u32, 1),
            grid_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe { f.launch(cfg, (&w_dev, &x_dev, &mut o_dev, width, o_rows, o_cols)) }?;
        dev.dtoh_sync_copy_into(&o_dev,  o)?;
        // println!("Found {:?} in {:?}", o, start.elapsed());

        Ok(())

    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_mul2() {
        let a_host = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_host = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut c_host = [0.0f32; 4];
        let _ = GPU::matmul(&mut c_host, &a_host, &b_host, 3, 2, 2);

        assert_eq!(c_host, [22.0, 28.0, 49.0, 64.0]);
        print!("C1 IS {:?}", c_host);



    }

    // fn test_softmax() {}
    // fn test_rmsnorm() {}
}

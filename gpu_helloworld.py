import numpy as np
import pyopencl as cl
import time

# ==================== 共通パラメータ ====================
WIDTH = 1024
HEIGHT = 1024
MAX_ITER = 1000

print(f"画像サイズ: {WIDTH} x {HEIGHT} ({WIDTH*HEIGHT:,} ピクセル) / 最大反復: {MAX_ITER}")
print("=" * 50)

# ==================== GPU側 ====================
platforms = cl.get_platforms()
gpu_devices = []
for p in platforms:
    gpu_devices += p.get_devices(device_type=cl.device_type.GPU)

if not gpu_devices:
    raise RuntimeError("GPUデバイスがない！！！")

device = gpu_devices[0]
print(f"使用デバイス: {device.name} ({device.platform.name})")

ctx = cl.Context([device])
queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

output_gpu = np.empty(WIDTH * HEIGHT, dtype=np.int32)
mf = cl.mem_flags
out_buf = cl.Buffer(ctx, mf.WRITE_ONLY, output_gpu.nbytes)

kernel_code = """
__kernel void mandelbrot(__global int *output,
                          const int width,
                          const int height,
                          const int max_iter)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    float re = (x - width / 1.5f) * 4.0f / width;
    float im = (y - height / 2.0f) * 4.0f / height;

    float zre = 0.0f, zim = 0.0f;
    int iter = 0;

    while (zre * zre + zim * zim < 4.0f && iter < max_iter) {
        float new_re = zre * zre - zim * zim + re;
        float new_im = 2.0f * zre * zim + im;
        zre = new_re;
        zim = new_im;
        iter++;
    }

    output[y * width + x] = iter;
}
"""

program = cl.Program(ctx, kernel_code).build()

t_gpu_start = time.perf_counter()
event = program.mandelbrot(queue, (WIDTH, HEIGHT), None,
                             out_buf, np.int32(WIDTH), np.int32(HEIGHT), np.int32(MAX_ITER))
event.wait()
gpu_kernel_time_ms = (event.profile.end - event.profile.start) / 1_000_000
cl.enqueue_copy(queue, output_gpu, out_buf)
queue.finish()
t_gpu_end = time.perf_counter()
gpu_total_time_ms = (t_gpu_end - t_gpu_start) * 1000

total_pixels = WIDTH * HEIGHT
gpu_pixels_per_sec = total_pixels / (gpu_kernel_time_ms / 1000)

print(f"\n[GPU] カーネル実行時間(計算のみ): {gpu_kernel_time_ms:.3f} ms")
print(f"[GPU] 全体時間(転送含む): {gpu_total_time_ms:.3f} ms")
print(f"[GPU] 処理速度: {gpu_pixels_per_sec/1e6:.2f} Mpixels/sec")

# ==================== CPU側(必ず実行される) ====================
print("\nCPU計算中……(サイズによっては時間かかります)")

output_cpu = np.zeros((HEIGHT, WIDTH), dtype=np.int32)

t_cpu_start = time.perf_counter()
for y in range(HEIGHT):
    for x in range(WIDTH):
        re = (x - WIDTH / 1.5) * 4.0 / WIDTH
        im = (y - HEIGHT / 2.0) * 4.0 / HEIGHT

        zre, zim = 0.0, 0.0
        iter_count = 0

        while zre * zre + zim * zim < 4.0 and iter_count < MAX_ITER:
            new_re = zre * zre - zim * zim + re
            new_im = 2.0 * zre * zim + im
            zre, zim = new_re, new_im
            iter_count += 1

        output_cpu[y, x] = iter_count
t_cpu_end = time.perf_counter()
cpu_time_ms = (t_cpu_end - t_cpu_start) * 1000
cpu_pixels_per_sec = total_pixels / (cpu_time_ms / 1000)

print(f"\n[CPU] 計算時間: {cpu_time_ms:.3f} ms")
print(f"[CPU] 処理速度: {cpu_pixels_per_sec/1e6:.4f} Mpixels/sec")

# ==================== 最終比較 ====================
print("\n" + "=" * 50)
print(f"GPUはCPUの {cpu_time_ms / gpu_kernel_time_ms:.1f} 倍速い")
print(f"(GPU: {gpu_kernel_time_ms:.3f} ms  vs  CPU: {cpu_time_ms:.3f} ms)")
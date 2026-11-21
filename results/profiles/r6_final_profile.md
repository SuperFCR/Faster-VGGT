# profile

tree vggt, config configs/r6_bf16_depth_tritonup.json

5 iterations after warmup of 6; instrumented (not comparable with `time`)

window 4.07 s for 5 iterations; GPU busy 3.31 s (81%); kernel time 3.31 s (rank 0)

host time per iteration (s): 1.072, 0.631, 0.765, 0.943, 0.662

## kernel time by category (ms, rank 0)

| category | ms |
|---|---|
| gemm | 1518.2 |
| attention | 956.8 |
| elementwise/reduce | 330.6 |
| memcpy/cast | 225.9 |
| norm | 197.3 |
| optimizer | 34.1 |
| other | 9.5 |
| conv | 1.9 |

## GPU time inside model calls (forward only)

| call | GPU ms | calls | CPU ms |
|---|---|---|---|
| aggregator | 4389.93 | 20 | 1199.54 |
| camera_head | 99.19 | 20 | 252.14 |
| depth_head | 560.02 | 20 | 175.27 |
| dinov2_patch_embed | 958.15 | 20 | 357.74 |
| multitask_loss | 62.98 | 20 | 471.85 |

## top kernels

| ms | calls | share | kernel |
|---|---|---|---|
| 946.744 | 720 | 0.2859 | `void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<64, 128, 128, 4, false, false, cutlass::bfloat16_t, Flash_kernel_traits<64, 128` |
| 668.889 | 2016 | 0.202 | `ampere_bf16_s16816gemm_bf16_128x128_ldg8_relu_f2f_stages_32x5_tn` |
| 343.15 | 864 | 0.1036 | `ampere_bf16_s16816gemm_bf16_128x256_ldg8_relu_f2f_stages_64x3_tn` |
| 130.038 | 1420 | 0.0393 | `_ls_add_layernorm_bf16_kernel` |
| 98.303 | 40 | 0.0297 | `Memcpy HtoD (Pinned -> Device)` |
| 93.089 | 720 | 0.0281 | `void at::native::vectorized_elementwise_kernel<4, at::native::GeluCUDAKernelImpl(at::TensorIteratorBase&, at::native::GeluType)::{lambda()#2` |
| 63.919 | 4275 | 0.0193 | `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>, false>(int, at::native::CUDAFu` |
| 61.688 | 128 | 0.0186 | `void cutlass__5x_cudnn::Kernel<cutlass_tensorop_bf16_s16816fprop_optimized_bf16_128x128_32x3_nhwc_align8>(cutlass_tensorop_bf16_s16816fprop_` |
| 61.614 | 330 | 0.0186 | `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<c10::BFloat16> >(at::TensorIterat` |
| 61.512 | 100 | 0.0186 | `sm80_xmma_wgrad_implicit_gemm_indexed_bf16bf16_bf16f32_f32_nhwckrsc_nhwc_tilesize256x128x32_stage3_warpsize4x2x1_g1_tensor16x8x16_execute_ke` |
| 54.311 | 50 | 0.0164 | `_bilinear_nhwc_bwd_kernel` |
| 52.245 | 480 | 0.0158 | `_fused_qkv_qknorm_rope_kernel` |
| 42.496 | 46 | 0.0128 | `void cutlass__5x_cudnn::Kernel<cutlass_tensorop_bf16_s16816dgrad_optimized_bf16_128x128_32x4_nhwc_unity_stride_align8>(cutlass_tensorop_bf16` |
| 37.493 | 68 | 0.0113 | `sm80_xmma_dgrad_implicit_gemm_indexed_bf16bf16_bf16f32_f32_nhwckrsc_nhwc_tilesize128x128x32_stage4_warpsize2x2x1_g1_tensor16x8x16_execute_ke` |
| 33.763 | 70 | 0.0102 | `sm80_xmma_fprop_implicit_gemm_bf16bf16_bf16f32_f32_nhwckrsc_nhwc_tilesize128x128x32_stage4_warpsize2x2x1_g1_tensor16x8x16_execute_kernel__5x` |
| 29.085 | 50 | 0.0088 | `_bilinear_nhwc_fwd_kernel` |
| 25.926 | 480 | 0.0078 | `ampere_sgemm_128x64_nt` |
| 25.436 | 270 | 0.0077 | `void at::native::reduce_kernel<128, 4, at::native::ReduceOp<c10::BFloat16, at::native::func_wrapper_t<c10::BFloat16, at::native::sum_functor` |
| 21.248 | 10 | 0.0064 | `void at::native::(anonymous namespace)::multi_tensor_apply_kernel<at::native::(anonymous namespace)::FusedOptimizerTensorListMetadata<4>, at` |
| 20.873 | 560 | 0.0063 | `void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_64x64_32x6_nn_align4>(cutlass_80_tensorop_s1688gemm_64x64_32x6_nn_align4::Params)` |
| 20.638 | 10 | 0.0062 | `sm80_xmma_wgrad_implicit_gemm_indexed_bf16bf16_bf16f32_f32_nhwckrsc_nhwc_tilesize128x128x64_stage3_warpsize2x2x2_g1_tensor16x8x16_execute_ke` |
| 19.701 | 10 | 0.0059 | `sm80_xmma_wgrad_implicit_gemm_indexed_bf16bf16_bf16f32_f32_nhwckrsc_nhwc_tilesize32x128x32_stage4_warpsize1x4x1_g1_tensor16x8x16_execute_ker` |
| 19.528 | 10 | 0.0059 | `ampere_sgemm_32x128_tn` |
| 17.796 | 480 | 0.0054 | `void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x64_32x6_tn_align4>(cutlass_80_tensorop_s1688gemm_128x64_32x6_tn_align4::Params)` |
| 15.636 | 1025 | 0.0047 | `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&):` |
| 14.408 | 10 | 0.0044 | `sm80_xmma_fprop_implicit_gemm_indexed_bf16bf16_bf16f32_f32_nhwckrsc_nhwc_tilesize256x32x32_stage4_warpsize4x1x1_g1_tensor16x8x16_execute_ker` |
| 14.164 | 659 | 0.0043 | `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const:` |
| 13.717 | 66 | 0.0041 | `ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_32x5_nn` |
| 12.509 | 68 | 0.0038 | `sm80_xmma_dgrad_implicit_gemm_bf16bf16_bf16f32_f32_nhwckrsc_nhwc_tilesize128x128x32_stage4_warpsize2x2x1_g1_tensor16x8x16_execute_kernel__5x` |
| 12.356 | 150 | 0.0037 | `void at::native::vectorized_elementwise_kernel<4, at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::(anonym` |
| 11.876 | 170 | 0.0036 | `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char*, 3ul>, false>(int, at::native` |
| 10.399 | 36 | 0.0031 | `ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_32x5_tn` |
| 9.905 | 39 | 0.003 | `void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_64x64_16x6_nn_align1>(cutlass_80_tensorop_s1688gemm_64x64_16x6_nn_align1::Params)` |
| 9.573 | 652 | 0.0029 | `void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::ar` |
| 9.4 | 460 | 0.0028 | `void at::native::(anonymous namespace)::vectorized_layer_norm_kernel<float, float, false>(int, float, float const*, float const*, float cons` |
| 9.298 | 6 | 0.0028 | `sm80_xmma_dgrad_implicit_gemm_bf16bf16_bf16f32_f32_nhwckrsc_nhwc_tilesize256x64x32_stage3_warpsize4x1x1_g1_tensor16x8x16_execute_kernel__5x_` |
| 9.138 | 160 | 0.0028 | `void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x64_16x6_nn_align4>(cutlass_80_tensorop_s1688gemm_128x64_16x6_nn_align4::Params)` |
| 8.323 | 150 | 0.0025 | `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::launch_clamp_scalar(at::TensorIteratorBase&, c10::Scala` |
| 7.968 | 28 | 0.0024 | `ampere_bf16_s16816gemm_bf16_256x128_ldg8_f2f_stages_64x3_nt` |
| 7.833 | 240 | 0.0024 | `void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_64x64_32x6_tn_align4>(cutlass_80_tensorop_s1688gemm_64x64_32x6_tn_align4::Params)` |

## top CPU ops (self time)

| ms | calls | op |
|---|---|---|
| 178.53 | 3680 | aten::addmm |
| 102.03 | 13741 | aten::empty |
| 79.75 | 310 | aten::convolution_backward |
| 63.54 | 3960 | aten::mul |
| 58.09 | 1600 | aten::mm |
| 56.22 | 21114 | aten::view |
| 48.01 | 1930 | aten::sum |
| 46.52 | 300 | aten::cudnn_convolution |
| 45.16 | 3692 | aten::linear |
| 38.34 | 14360 | aten::transpose |
| 35.71 | 3605 | aten::add_ |
| 29.88 | 2549 | aten::copy_ |
| 27.78 | 22519 | aten::as_strided |
| 25.73 | 2411 | aten::empty_strided |
| 25.54 | 6900 | aten::t |
| 25.32 | 1460 | aten::add |
| 20.13 | 920 | aten::gelu |
| 20.1 | 1520 | aten::where |
| 19.16 | 880 | aten::scaled_dot_product_attention |
| 16.8 | 720 | aten::_flash_attention_forward |
| 13.36 | 6771 | aten::reshape |
| 13.2 | 1669 | aten::fill_ |
| 12.93 | 460 | aten::native_layer_norm |
| 11.86 | 501 | aten::cat |
| 11.18 | 1483 | aten::_to_copy |

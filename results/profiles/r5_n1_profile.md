# profile

tree vggt, config configs/r5_n1_all.json

5 iterations after warmup of 6; instrumented (not comparable with `time`)

window 5.48 s for 5 iterations; GPU busy 4.00 s (73%); kernel time 4.00 s (rank 0)

host time per iteration (s): 1.775, 0.813, 0.995, 1.231, 0.681

## kernel time by category (ms, rank 0)

| category | ms |
|---|---|
| gemm | 1924.2 |
| attention | 954.1 |
| elementwise/reduce | 447.4 |
| memcpy/cast | 215.6 |
| conv | 193.7 |
| norm | 180.3 |
| optimizer | 34.2 |
| other | 9.1 |

## GPU time inside model calls (forward only)

| call | GPU ms | calls | CPU ms |
|---|---|---|---|
| aggregator | 4990.55 | 20 | 1834.99 |
| camera_head | 80.27 | 20 | 261.08 |
| depth_head | 1041.2 | 20 | 145.36 |
| dinov2_patch_embed | 1029.91 | 20 | 479.79 |
| multitask_loss | 195.39 | 20 | 1516.4 |

## top kernels

| ms | calls | share | kernel |
|---|---|---|---|
| 944.092 | 720 | 0.2362 | `void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<64, 128, 128, 4, false, false, cutlass::bfloat16_t, Flash_kernel_traits<64, 128` |
| 667.751 | 2016 | 0.1671 | `ampere_bf16_s16816gemm_bf16_128x128_ldg8_relu_f2f_stages_32x5_tn` |
| 342.616 | 864 | 0.0857 | `ampere_bf16_s16816gemm_bf16_128x256_ldg8_relu_f2f_stages_64x3_tn` |
| 183.945 | 1370 | 0.046 | `void cudnn::engines_precompiled::convertTensor_kernel<float, float, float, (cudnnKernelDataType_t)2>(float, float const*, float, float*, uns` |
| 177.565 | 186 | 0.0444 | `void cutlass__5x_cudnn::Kernel<cutlass_tensorop_s1688fprop_optimized_tf32_128x128_16x4_nhwc_align4>(cutlass_tensorop_s1688fprop_optimized_tf` |
| 166.516 | 172 | 0.0417 | `sm80_xmma_dgrad_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nhwc_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_execute_kernel__5x_c` |
| 137.881 | 170 | 0.0345 | `_ZN17cutlass__5x_cudnn6KernelINS_4conv6kernel23ImplicitGemmConvolutionINS1_11threadblock22ImplicitGemmMultistageINS_4gemm9GemmShapeILi256ELi` |
| 126.875 | 1420 | 0.0317 | `_ls_add_layernorm_bf16_kernel` |
| 98.302 | 40 | 0.0246 | `Memcpy HtoD (Pinned -> Device)` |
| 93.057 | 720 | 0.0233 | `void at::native::vectorized_elementwise_kernel<4, at::native::GeluCUDAKernelImpl(at::TensorIteratorBase&, at::native::GeluType)::{lambda()#2` |
| 88.14 | 4445 | 0.0221 | `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>, false>(int, at::native::CUDAFu` |
| 85.46 | 12 | 0.0214 | `sm80_xmma_wgrad_implicit_gemm_indexed_tf32f32_tf32f32_f32_nhwckrsc_nhwc_tilesize64x128x32_stage5_warpsize2x2x1_g1_tensor16x8x8_execute_kerne` |
| 79.631 | 50 | 0.0199 | `void at::native::(anonymous namespace)::upsample_bilinear2d_nhwc_out_frame<float, float, int>(float, float, bool, int, int, int, int, int, f` |
| 76.328 | 835 | 0.0191 | `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<float> >(at::TensorIteratorBase&,` |
| 74.345 | 50 | 0.0186 | `void at::native::(anonymous namespace)::upsample_bilinear2d_backward_nhwc_out_frame<float, float, int>(int, int, int, int, float, float, boo` |
| 50.166 | 8 | 0.0126 | `sm80_xmma_wgrad_implicit_gemm_indexed_tf32f32_tf32f32_f32_nhwckrsc_nhwc_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_execute_kern` |
| 41.395 | 1350 | 0.0104 | `void at::native::reduce_kernel<128, 4, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, f` |
| 38.385 | 480 | 0.0096 | `_fused_qkv_qknorm_rope_kernel` |
| 38.176 | 10 | 0.0096 | `sm80_xmma_fprop_implicit_gemm_indexed_wo_smem_tf32f32_tf32f32_f32_nhwckrsc_nhwc_tilesize128x32x64_stage1_warpsize4x1x1_g1_tensor16x8x8_align` |
| 25.873 | 480 | 0.0065 | `ampere_sgemm_128x64_nt` |
| 24.533 | 150 | 0.0061 | `void at::native::vectorized_elementwise_kernel<4, at::native::BinaryFunctor<float, float, float, at::native::(anonymous namespace)::threshol` |
| 21.515 | 20 | 0.0054 | `void cutlass__5x_cudnn::Kernel<cutlass_tensorop_s1688dgrad_optimized_tf32_128x128_16x4_nhwc_unity_stride_align4>(cutlass_tensorop_s1688dgrad` |
| 21.263 | 10 | 0.0053 | `void at::native::(anonymous namespace)::multi_tensor_apply_kernel<at::native::(anonymous namespace)::FusedOptimizerTensorListMetadata<4>, at` |
| 21.142 | 560 | 0.0053 | `void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_64x64_32x6_nn_align4>(cutlass_80_tensorop_s1688gemm_64x64_32x6_nn_align4::Params)` |
| 20.715 | 42 | 0.0052 | `void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x128_16x5_nn_align4>(cutlass_80_tensorop_s1688gemm_128x128_16x5_nn_align4::Params)` |
| 19.053 | 24 | 0.0048 | `void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x128_16x5_tn_align4>(cutlass_80_tensorop_s1688gemm_128x128_16x5_tn_align4::Params)` |
| 18.813 | 505 | 0.0047 | `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::launch_clamp_scalar(at::TensorIteratorBase&, c10::Scala` |
| 17.815 | 480 | 0.0045 | `void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x64_32x6_tn_align4>(cutlass_80_tensorop_s1688gemm_128x64_32x6_tn_align4::Params)` |
| 15.497 | 1025 | 0.0039 | `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&):` |
| 14.93 | 40 | 0.0037 | `void cutlass__5x_cudnn::Kernel<cutlass_tensorop_s1688wgrad_optimized_tf32_256x128_16x3_nhwc_align4>(cutlass_tensorop_s1688wgrad_optimized_tf` |
| 12.059 | 12 | 0.003 | `void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x128_32x3_nt_align4>(cutlass_80_tensorop_s1688gemm_128x128_32x3_nt_align4::Params)` |
| 9.849 | 39 | 0.0025 | `void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_64x64_16x6_nn_align1>(cutlass_80_tensorop_s1688gemm_64x64_16x6_nn_align1::Params)` |
| 9.405 | 460 | 0.0024 | `void at::native::(anonymous namespace)::vectorized_layer_norm_kernel<float, float, false>(int, float, float const*, float const*, float cons` |
| 9.257 | 160 | 0.0023 | `void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x64_16x6_nn_align4>(cutlass_80_tensorop_s1688gemm_128x64_16x6_nn_align4::Params)` |
| 9.032 | 28 | 0.0023 | `void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x128_32x3_nn_align4>(cutlass_80_tensorop_s1688gemm_256x128_32x3_nn_align4::Params)` |
| 8.323 | 16 | 0.0021 | `void cutlass__5x_cudnn::Kernel<cutlass_tensorop_s1688dgrad_optimized_tf32_128x128_16x4_nhwc_align4>(cutlass_tensorop_s1688dgrad_optimized_tf` |
| 7.851 | 240 | 0.002 | `void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_64x64_32x6_tn_align4>(cutlass_80_tensorop_s1688gemm_64x64_32x6_tn_align4::Params)` |
| 7.793 | 18 | 0.002 | `void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x256_32x3_nt_align4>(cutlass_80_tensorop_s1688gemm_128x256_32x3_nt_align4::Params)` |
| 7.757 | 1082 | 0.0019 | `void at::native::vectorized_elementwise_kernel<4, at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<flo` |
| 7.35 | 160 | 0.0018 | `fmha_cutlassB_f32_aligned_128x64_k128_sm80(PyTorchMemEffAttention::AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 1` |

## top CPU ops (self time)

| ms | calls | op |
|---|---|---|
| 180.27 | 3670 | aten::addmm |
| 102.49 | 13651 | aten::empty |
| 83.01 | 320 | aten::convolution_backward |
| 65.15 | 3960 | aten::mul |
| 57.57 | 21064 | aten::view |
| 56.87 | 1580 | aten::mm |
| 51.67 | 310 | aten::cudnn_convolution |
| 47.49 | 1930 | aten::sum |
| 44.7 | 3680 | aten::linear |
| 38.33 | 3615 | aten::add_ |
| 37.28 | 14310 | aten::transpose |
| 28.46 | 22449 | aten::as_strided |
| 26.82 | 6850 | aten::t |
| 25.03 | 1460 | aten::add |
| 21.14 | 920 | aten::gelu |
| 20.82 | 1520 | aten::where |
| 19.4 | 880 | aten::scaled_dot_product_attention |
| 16.91 | 720 | aten::_flash_attention_forward |
| 16.61 | 1287 | aten::copy_ |
| 14.56 | 1719 | aten::fill_ |
| 14.05 | 1149 | aten::empty_strided |
| 13.57 | 460 | aten::native_layer_norm |
| 12.61 | 6751 | aten::reshape |
| 11.75 | 501 | aten::cat |
| 10.7 | 720 | aten::_scaled_dot_product_flash_attention |

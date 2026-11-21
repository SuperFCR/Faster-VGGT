# profile

tree vggt, config configs/r3_all.json

5 iterations after warmup of 6; instrumented (not comparable with `time`)

window 5.89 s for 5 iterations; GPU busy 4.19 s (71%); kernel time 4.19 s (rank 0)

host time per iteration (s): 1.861, 0.882, 1.023, 1.28, 0.843

## kernel time by category (ms, rank 0)

| category | ms |
|---|---|
| gemm | 1952.5 |
| attention | 950.8 |
| elementwise/reduce | 504.5 |
| memcpy/cast | 270.1 |
| conv | 184.8 |
| norm | 152.7 |
| optimizer | 72.5 |
| other | 6.0 |

## GPU time inside model calls (forward only)

| call | GPU ms | calls | CPU ms |
|---|---|---|---|
| aggregator | 5048.68 | 20 | 1902.24 |
| camera_head | 79.78 | 20 | 246.08 |
| depth_head | 1212.85 | 20 | 840.55 |
| dinov2_patch_embed | 975.44 | 20 | 408.05 |
| multitask_loss | 209.26 | 20 | 877.37 |

## top kernels

| ms | calls | share | kernel |
|---|---|---|---|
| 940.771 | 720 | 0.2247 | `void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<64, 128, 128, 4, false, false, cutlass::bfloat16_t, Flash_kernel_traits<64, 128` |
| 665.722 | 2016 | 0.159 | `ampere_bf16_s16816gemm_bf16_128x128_ldg8_relu_f2f_stages_32x5_tn` |
| 341.537 | 864 | 0.0816 | `ampere_bf16_s16816gemm_bf16_128x256_ldg8_relu_f2f_stages_64x3_tn` |
| 184.762 | 2746 | 0.0441 | `void cudnn::engines_precompiled::convertTensor_kernel<float, float, float, (cudnnKernelDataType_t)2>(float, float const*, float, float*, uns` |
| 167.512 | 288 | 0.04 | `sm80_xmma_dgrad_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nhwc_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_execute_kernel__5x_c` |
| 155.055 | 242 | 0.037 | `void cutlass__5x_cudnn::Kernel<cutlass_tensorop_s1688fprop_optimized_tf32_128x128_16x4_nhwc_align4>(cutlass_tensorop_s1688fprop_optimized_tf` |
| 147.902 | 320 | 0.0353 | `_ZN17cutlass__5x_cudnn6KernelINS_4conv6kernel23ImplicitGemmConvolutionINS1_11threadblock22ImplicitGemmMultistageINS_4gemm9GemmShapeILi256ELi` |
| 125.092 | 40 | 0.0299 | `sm80_xmma_wgrad_implicit_gemm_indexed_tf32f32_tf32f32_f32_nhwckrsc_nhwc_tilesize64x128x32_stage5_warpsize2x2x1_g1_tensor16x8x8_execute_kerne` |
| 98.301 | 40 | 0.0235 | `Memcpy HtoD (Pinned -> Device)` |
| 92.883 | 720 | 0.0222 | `void at::native::vectorized_elementwise_kernel<4, at::native::GeluCUDAKernelImpl(at::TensorIteratorBase&, at::native::GeluType)::{lambda()#2` |
| 90.529 | 5293 | 0.0216 | `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>, false>(int, at::native::CUDAFu` |
| 78.969 | 1415 | 0.0189 | `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<float> >(at::TensorIteratorBase&,` |
| 78.285 | 100 | 0.0187 | `void at::native::(anonymous namespace)::upsample_bilinear2d_nhwc_out_frame<float, float, int>(float, float, bool, int, int, int, int, int, f` |
| 73.494 | 100 | 0.0176 | `void at::native::(anonymous namespace)::upsample_bilinear2d_backward_nhwc_out_frame<float, float, int>(int, int, int, int, float, float, boo` |
| 65.46 | 720 | 0.0156 | `_ls_add_layernorm_bf16_kernel` |
| 50.6 | 720 | 0.0121 | `_ls_add_kernel` |
| 50.069 | 1620 | 0.012 | `void at::native::reduce_kernel<128, 4, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, f` |
| 38.108 | 20 | 0.0091 | `sm80_xmma_fprop_implicit_gemm_indexed_wo_smem_tf32f32_tf32f32_f32_nhwckrsc_nhwc_tilesize128x32x64_stage1_warpsize4x1x1_g1_tensor16x8x8_align` |
| 37.502 | 720 | 0.009 | `_layernorm_to_bf16_kernel` |
| 36.842 | 480 | 0.0088 | `_fused_qkv_qknorm_rope_kernel` |
| 33.647 | 480 | 0.008 | `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&):` |
| 26.794 | 2171 | 0.0064 | `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&):` |
| 25.841 | 480 | 0.0062 | `ampere_sgemm_128x64_nt` |
| 24.693 | 330 | 0.0059 | `void at::native::vectorized_elementwise_kernel<4, at::native::BinaryFunctor<float, float, float, at::native::(anonymous namespace)::threshol` |
| 24.012 | 78 | 0.0057 | `void cutlass__5x_cudnn::Kernel<cutlass_tensorop_s1688dgrad_optimized_tf32_128x128_16x4_nhwc_unity_stride_align4>(cutlass_tensorop_s1688dgrad` |
| 23.019 | 90 | 0.0055 | `void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x128_16x5_nn_align4>(cutlass_80_tensorop_s1688gemm_128x128_16x5_nn_align4::Params)` |
| 21.065 | 562 | 0.005 | `void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_64x64_32x6_nn_align4>(cutlass_80_tensorop_s1688gemm_64x64_32x6_nn_align4::Params)` |
| 20.658 | 70 | 0.0049 | `sm80_xmma_fprop_implicit_gemm_indexed_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize256x64x32_stage3_warpsize4x2x1_g1_tensor16x8x8_execute_kerne` |
| 18.307 | 675 | 0.0044 | `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::launch_clamp_scalar(at::TensorIteratorBase&, c10::Scala` |
| 17.9 | 482 | 0.0043 | `void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x64_32x6_tn_align4>(cutlass_80_tensorop_s1688gemm_128x64_32x6_tn_align4::Params)` |
| 16.955 | 84 | 0.0041 | `void cutlass__5x_cudnn::Kernel<cutlass_tensorop_s1688wgrad_optimized_tf32_256x128_16x3_nhwc_align4>(cutlass_tensorop_s1688wgrad_optimized_tf` |
| 15.562 | 2 | 0.0037 | `sm80_xmma_wgrad_implicit_gemm_indexed_tf32f32_tf32f32_f32_nhwckrsc_nhwc_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_execute_kern` |
| 12.147 | 20 | 0.0029 | `void at::native::(anonymous namespace)::multi_tensor_apply_kernel<at::native::(anonymous namespace)::TensorListMetadata<1>, at::native::(ano` |
| 11.379 | 10 | 0.0027 | `void at::native::(anonymous namespace)::multi_tensor_apply_kernel<at::native::(anonymous namespace)::TensorListScalarListMetadata<float, 3>,` |
| 11.196 | 40 | 0.0027 | `void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x128_32x3_tn_align4>(cutlass_80_tensorop_s1688gemm_256x128_32x3_tn_align4::Params)` |
| 10.921 | 24 | 0.0026 | `void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x128_16x5_tn_align4>(cutlass_80_tensorop_s1688gemm_128x128_16x5_tn_align4::Params)` |
| 10.713 | 66 | 0.0026 | `sm80_xmma_fprop_implicit_gemm_indexed_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_alignc4_exec` |
| 10.253 | 20 | 0.0024 | `void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x128_32x5_nt_align4>(cutlass_80_tensorop_s1688gemm_128x128_32x5_nt_align4::Params)` |
| 9.841 | 41 | 0.0024 | `void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_64x64_16x6_nn_align1>(cutlass_80_tensorop_s1688gemm_64x64_16x6_nn_align1::Params)` |
| 9.794 | 508 | 0.0023 | `void at::native::(anonymous namespace)::vectorized_layer_norm_kernel<float, float, false>(int, float, float const*, float const*, float cons` |

## top CPU ops (self time)

| ms | calls | op |
|---|---|---|
| 179.64 | 3670 | aten::addmm |
| 176.6 | 704 | aten::convolution_backward |
| 127.46 | 16596 | aten::empty |
| 96.9 | 670 | aten::cudnn_convolution |
| 73.18 | 4474 | aten::mul |
| 60.49 | 22174 | aten::view |
| 58.04 | 2266 | aten::sum |
| 57.99 | 1580 | aten::mm |
| 49.95 | 5482 | aten::add_ |
| 44.71 | 3353 | aten::copy_ |
| 43.03 | 3680 | aten::linear |
| 31.45 | 13380 | aten::transpose |
| 29.1 | 23979 | aten::as_strided |
| 28.88 | 1683 | aten::add |
| 25.3 | 6880 | aten::t |
| 22.88 | 8839 | aten::reshape |
| 20.52 | 920 | aten::gelu |
| 20.14 | 2132 | aten::empty_strided |
| 19.39 | 909 | aten::cat |
| 19.26 | 880 | aten::scaled_dot_product_attention |
| 18.72 | 1480 | aten::where |
| 16.82 | 1999 | aten::fill_ |
| 16.72 | 720 | aten::_flash_attention_forward |
| 15.05 | 508 | aten::native_layer_norm |
| 12.29 | 2280 | aten::clone |

# profile

tree vggt, config configs/r1_all.json

5 iterations after warmup of 6; instrumented (not comparable with `time`)

window 6.89 s for 5 iterations; GPU busy 5.93 s (86%); kernel time 5.93 s (rank 0)

host time per iteration (s): 1.556, 1.184, 1.458, 1.637, 1.056

## kernel time by category (ms, rank 0)

| category | ms |
|---|---|
| gemm | 1915.2 |
| elementwise/reduce | 1105.7 |
| memcpy/cast | 968.4 |
| attention | 915.8 |
| norm | 638.0 |
| conv | 185.5 |
| optimizer | 72.6 |
| other | 10.0 |

## GPU time inside model calls (forward only)

| call | GPU ms | calls | CPU ms |
|---|---|---|---|
| aggregator | 7511.49 | 20 | 2598.98 |
| camera_head | 78.07 | 20 | 471.38 |
| depth_head | 1199.74 | 20 | 886.97 |
| dinov2_patch_embed | 1244.24 | 20 | 369.23 |
| multitask_loss | 659.65 | 20 | 997.82 |

## top kernels

| ms | calls | share | kernel |
|---|---|---|---|
| 908.473 | 720 | 0.1533 | `void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<64, 128, 128, 4, false, false, cutlass::bfloat16_t, Flash_kernel_traits<64, 128` |
| 647.553 | 2016 | 0.1093 | `ampere_bf16_s16816gemm_bf16_128x128_ldg8_relu_f2f_stages_32x5_tn` |
| 637.979 | 2908 | 0.1077 | `void at::native::(anonymous namespace)::vectorized_layer_norm_kernel<float, float, false>(int, float, float const*, float const*, float cons` |
| 331.858 | 864 | 0.056 | `ampere_bf16_s16816gemm_bf16_128x256_ldg8_relu_f2f_stages_64x3_tn` |
| 319.535 | 8443 | 0.0539 | `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>, false>(int, at::native::CUDAFu` |
| 220.726 | 20 | 0.0373 | `void at::native::(anonymous namespace)::gatherKthValue<float, int, 1>(at::cuda::detail::TensorInfo<float const, int>, int, int, int, int, at` |
| 185.465 | 2746 | 0.0313 | `void cudnn::engines_precompiled::convertTensor_kernel<float, float, float, (cudnnKernelDataType_t)2>(float, float const*, float, float*, uns` |
| 167.318 | 8190 | 0.0282 | `void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::ar` |
| 167.108 | 288 | 0.0282 | `sm80_xmma_dgrad_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nhwc_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_execute_kernel__5x_c` |
| 159.029 | 4564 | 0.0268 | `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::bi` |
| 151.405 | 242 | 0.0256 | `void cutlass__5x_cudnn::Kernel<cutlass_tensorop_s1688fprop_optimized_tf32_128x128_16x4_nhwc_align4>(cutlass_tensorop_s1688fprop_optimized_tf` |
| 147.624 | 320 | 0.0249 | `_ZN17cutlass__5x_cudnn6KernelINS_4conv6kernel23ImplicitGemmConvolutionINS1_11threadblock22ImplicitGemmMultistageINS_4gemm9GemmShapeILi256ELi` |
| 137.89 | 1500 | 0.0233 | `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl<at::native::BinaryFunctor<float, float, float, at::native::binary_in` |
| 124.772 | 40 | 0.0211 | `sm80_xmma_wgrad_implicit_gemm_indexed_tf32f32_tf32f32_f32_nhwckrsc_nhwc_tilesize64x128x32_stage5_warpsize2x2x1_g1_tensor16x8x8_execute_kerne` |
| 110.272 | 2100 | 0.0186 | `void at::native::(anonymous namespace)::CatArrayBatchedCopy<at::native::(anonymous namespace)::OpaqueType<4u>, unsigned int, 4, 64, 64>(at::` |
| 104.619 | 3131 | 0.0177 | `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&):` |
| 98.291 | 40 | 0.0166 | `Memcpy HtoD (Pinned -> Device)` |
| 91.919 | 720 | 0.0155 | `void at::native::vectorized_elementwise_kernel<4, at::native::GeluCUDAKernelImpl(at::TensorIteratorBase&, at::native::GeluType)::{lambda()#2` |
| 80.314 | 970 | 0.0136 | `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambd` |
| 78.637 | 1415 | 0.0133 | `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<float> >(at::TensorIteratorBase&,` |
| 76.834 | 100 | 0.013 | `void at::native::(anonymous namespace)::upsample_bilinear2d_nhwc_out_frame<float, float, int>(float, float, bool, int, int, int, int, int, f` |
| 73.272 | 100 | 0.0124 | `void at::native::(anonymous namespace)::upsample_bilinear2d_backward_nhwc_out_frame<float, float, int>(int, int, int, int, float, float, boo` |
| 69.135 | 1000 | 0.0117 | `void at::native::(anonymous namespace)::CatArrayBatchedCopy_vectorized<at::native::(anonymous namespace)::OpaqueType<4u>, unsigned int, 4, 1` |
| 61.581 | 50 | 0.0104 | `void (anonymous namespace)::indexing_backward_kernel_stride_1<float>(long const*, long const*, float const*, float*, long, long, long, long,` |
| 49.979 | 1620 | 0.0084 | `void at::native::reduce_kernel<128, 4, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, f` |
| 48.285 | 1920 | 0.0081 | `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::neg_kernel_cuda(at::TensorIteratorBase&)::{lambda` |
| 37.998 | 20 | 0.0064 | `sm80_xmma_fprop_implicit_gemm_indexed_wo_smem_tf32f32_tf32f32_f32_nhwckrsc_nhwc_tilesize128x32x64_stage1_warpsize4x1x1_g1_tensor16x8x8_align` |
| 32.661 | 480 | 0.0055 | `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&):` |
| 25.904 | 480 | 0.0044 | `ampere_sgemm_128x64_nt` |
| 24.696 | 330 | 0.0042 | `void at::native::vectorized_elementwise_kernel<4, at::native::BinaryFunctor<float, float, float, at::native::(anonymous namespace)::threshol` |
| 23.85 | 78 | 0.004 | `void cutlass__5x_cudnn::Kernel<cutlass_tensorop_s1688dgrad_optimized_tf32_128x128_16x4_nhwc_unity_stride_align4>(cutlass_tensorop_s1688dgrad` |
| 22.955 | 90 | 0.0039 | `void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x128_16x5_nn_align4>(cutlass_80_tensorop_s1688gemm_128x128_16x5_nn_align4::Params)` |
| 21.111 | 562 | 0.0036 | `void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_64x64_32x6_nn_align4>(cutlass_80_tensorop_s1688gemm_64x64_32x6_nn_align4::Params)` |
| 20.311 | 70 | 0.0034 | `sm80_xmma_fprop_implicit_gemm_indexed_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize256x64x32_stage3_warpsize4x2x1_g1_tensor16x8x8_execute_kerne` |
| 18.117 | 655 | 0.0031 | `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::launch_clamp_scalar(at::TensorIteratorBase&, c10::Scala` |
| 17.69 | 482 | 0.003 | `void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x64_32x6_tn_align4>(cutlass_80_tensorop_s1688gemm_128x64_32x6_tn_align4::Params)` |
| 16.935 | 84 | 0.0029 | `void cutlass__5x_cudnn::Kernel<cutlass_tensorop_s1688wgrad_optimized_tf32_256x128_16x3_nhwc_align4>(cutlass_tensorop_s1688wgrad_optimized_tf` |
| 15.428 | 2 | 0.0026 | `sm80_xmma_wgrad_implicit_gemm_indexed_tf32f32_tf32f32_f32_nhwckrsc_nhwc_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_execute_kern` |
| 12.155 | 20 | 0.0021 | `void at::native::(anonymous namespace)::multi_tensor_apply_kernel<at::native::(anonymous namespace)::TensorListMetadata<1>, at::native::(ano` |
| 11.428 | 10 | 0.0019 | `void at::native::(anonymous namespace)::multi_tensor_apply_kernel<at::native::(anonymous namespace)::TensorListScalarListMetadata<float, 3>,` |

## top CPU ops (self time)

| ms | calls | op |
|---|---|---|
| 157.92 | 3670 | aten::addmm |
| 156.62 | 9414 | aten::mul |
| 145.76 | 704 | aten::convolution_backward |
| 136.42 | 20836 | aten::empty |
| 128.23 | 13553 | aten::copy_ |
| 99.24 | 670 | aten::cudnn_convolution |
| 87.74 | 11372 | aten::empty_strided |
| 74.9 | 5084 | aten::add |
| 70.8 | 2908 | aten::native_layer_norm |
| 69.08 | 6560 | aten::linear |
| 66.24 | 3829 | aten::cat |
| 55.04 | 1580 | aten::mm |
| 53.96 | 9587 | aten::_to_copy |
| 52.96 | 2196 | aten::sum |
| 48.94 | 20934 | aten::view |
| 46.65 | 5362 | aten::add_ |
| 42.8 | 13130 | aten::transpose |
| 35.75 | 32239 | aten::as_strided |
| 30.58 | 2035 | aten::neg |
| 30.36 | 7937 | aten::slice |
| 23.74 | 7110 | aten::t |
| 22.37 | 3869 | aten::layer_norm |
| 19.99 | 1360 | aten::scaled_dot_product_attention |
| 18.65 | 11940 | aten::to |
| 17.89 | 6129 | aten::reshape |

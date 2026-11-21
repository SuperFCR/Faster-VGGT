# profile

tree targets/reference, config None

5 iterations after warmup of 6; instrumented (not comparable with `time`)

window 8.90 s for 5 iterations; GPU busy 6.11 s (69%); kernel time 6.11 s (rank 0)

host time per iteration (s): 2.032, 1.612, 1.877, 2.053, 1.332

## kernel time by category (ms, rank 0)

| category | ms |
|---|---|
| gemm | 1895.0 |
| memcpy/cast | 1147.2 |
| elementwise/reduce | 1144.1 |
| attention | 911.6 |
| norm | 635.2 |
| conv | 183.8 |
| optimizer | 72.5 |
| other | 10.0 |

## GPU time inside model calls (forward only)

| call | GPU ms | calls | CPU ms |
|---|---|---|---|
| aggregator | 7987.77 | 20 | 4192.34 |
| camera_head | 208.68 | 20 | 211.65 |
| depth_head | 1186.27 | 20 | 383.29 |
| dinov2_patch_embed | 1237.08 | 20 | 335.04 |
| multitask_loss | 678.92 | 20 | 679.85 |

## top kernels

| ms | calls | share | kernel |
|---|---|---|---|
| 904.236 | 720 | 0.1479 | `void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<64, 128, 128, 4, false, false, cutlass::bfloat16_t, Flash_kernel_traits<64, 128` |
| 643.469 | 2016 | 0.1053 | `ampere_bf16_s16816gemm_bf16_128x128_ldg8_relu_f2f_stages_32x5_tn` |
| 635.246 | 2908 | 0.1039 | `void at::native::(anonymous namespace)::vectorized_layer_norm_kernel<float, float, false>(int, float, float const*, float const*, float cons` |
| 330.472 | 864 | 0.0541 | `ampere_bf16_s16816gemm_bf16_128x256_ldg8_relu_f2f_stages_64x3_tn` |
| 316.547 | 8443 | 0.0518 | `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>, false>(int, at::native::CUDAFu` |
| 278.925 | 70 | 0.0456 | `Memcpy HtoD (Pageable -> Device)` |
| 225.994 | 20 | 0.037 | `void at::native::(anonymous namespace)::gatherKthValue<float, int, 1>(at::cuda::detail::TensorInfo<float const, int>, int, int, int, int, at` |
| 183.805 | 2746 | 0.0301 | `void cudnn::engines_precompiled::convertTensor_kernel<float, float, float, (cudnnKernelDataType_t)2>(float, float const*, float, float*, uns` |
| 167.048 | 288 | 0.0273 | `sm80_xmma_dgrad_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nhwc_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_execute_kernel__5x_c` |
| 166.214 | 8190 | 0.0272 | `void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::ar` |
| 161.155 | 4564 | 0.0264 | `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::bi` |
| 150.878 | 242 | 0.0247 | `void cutlass__5x_cudnn::Kernel<cutlass_tensorop_s1688fprop_optimized_tf32_128x128_16x4_nhwc_align4>(cutlass_tensorop_s1688fprop_optimized_tf` |
| 147.653 | 320 | 0.0242 | `_ZN17cutlass__5x_cudnn6KernelINS_4conv6kernel23ImplicitGemmConvolutionINS1_11threadblock22ImplicitGemmMultistageINS_4gemm9GemmShapeILi256ELi` |
| 137.121 | 1500 | 0.0224 | `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl<at::native::BinaryFunctor<float, float, float, at::native::binary_in` |
| 124.677 | 40 | 0.0204 | `sm80_xmma_wgrad_implicit_gemm_indexed_tf32f32_tf32f32_f32_nhwckrsc_nhwc_tilesize64x128x32_stage5_warpsize2x2x1_g1_tensor16x8x8_execute_kerne` |
| 108.84 | 2100 | 0.0178 | `void at::native::(anonymous namespace)::CatArrayBatchedCopy<at::native::(anonymous namespace)::OpaqueType<4u>, unsigned int, 4, 64, 64>(at::` |
| 104.095 | 3066 | 0.017 | `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&):` |
| 91.773 | 720 | 0.015 | `void at::native::vectorized_elementwise_kernel<4, at::native::GeluCUDAKernelImpl(at::TensorIteratorBase&, at::native::GeluType)::{lambda()#2` |
| 80.069 | 970 | 0.0131 | `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambd` |
| 77.742 | 1410 | 0.0127 | `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<float> >(at::TensorIteratorBase&,` |
| 76.387 | 100 | 0.0125 | `void at::native::(anonymous namespace)::upsample_bilinear2d_nhwc_out_frame<float, float, int>(float, float, bool, int, int, int, int, int, f` |
| 73.183 | 100 | 0.012 | `void at::native::(anonymous namespace)::upsample_bilinear2d_backward_nhwc_out_frame<float, float, int>(int, int, int, int, float, float, boo` |
| 67.427 | 1000 | 0.011 | `void at::native::(anonymous namespace)::CatArrayBatchedCopy_vectorized<at::native::(anonymous namespace)::OpaqueType<4u>, unsigned int, 4, 1` |
| 61.305 | 50 | 0.01 | `void (anonymous namespace)::indexing_backward_kernel_stride_1<float>(long const*, long const*, float const*, float*, long, long, long, long,` |
| 49.828 | 1620 | 0.0082 | `void at::native::reduce_kernel<128, 4, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, f` |
| 46.496 | 1920 | 0.0076 | `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::neg_kernel_cuda(at::TensorIteratorBase&)::{lambda` |
| 37.699 | 20 | 0.0062 | `sm80_xmma_fprop_implicit_gemm_indexed_wo_smem_tf32f32_tf32f32_f32_nhwckrsc_nhwc_tilesize128x32x64_stage1_warpsize4x1x1_g1_tensor16x8x8_align` |
| 32.578 | 480 | 0.0053 | `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&):` |
| 28.832 | 3840 | 0.0047 | `void at::native::_scatter_gather_elementwise_kernel<128, 8, at::native::_cuda_scatter_gather_internal_kernel<false, at::native::OpaqueType<4` |
| 25.81 | 480 | 0.0042 | `ampere_sgemm_128x64_nt` |
| 24.684 | 330 | 0.004 | `void at::native::vectorized_elementwise_kernel<4, at::native::BinaryFunctor<float, float, float, at::native::(anonymous namespace)::threshol` |
| 23.807 | 78 | 0.0039 | `void cutlass__5x_cudnn::Kernel<cutlass_tensorop_s1688dgrad_optimized_tf32_128x128_16x4_nhwc_unity_stride_align4>(cutlass_tensorop_s1688dgrad` |
| 22.875 | 90 | 0.0037 | `void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x128_16x5_nn_align4>(cutlass_80_tensorop_s1688gemm_128x128_16x5_nn_align4::Params)` |
| 21.093 | 562 | 0.0035 | `void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_64x64_32x6_nn_align4>(cutlass_80_tensorop_s1688gemm_64x64_32x6_nn_align4::Params)` |
| 20.256 | 70 | 0.0033 | `sm80_xmma_fprop_implicit_gemm_indexed_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize256x64x32_stage3_warpsize4x2x1_g1_tensor16x8x8_execute_kerne` |
| 18.115 | 650 | 0.003 | `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::launch_clamp_scalar(at::TensorIteratorBase&, c10::Scala` |
| 17.637 | 482 | 0.0029 | `void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x64_32x6_tn_align4>(cutlass_80_tensorop_s1688gemm_128x64_32x6_tn_align4::Params)` |
| 16.962 | 84 | 0.0028 | `void cutlass__5x_cudnn::Kernel<cutlass_tensorop_s1688wgrad_optimized_tf32_256x128_16x3_nhwc_align4>(cutlass_tensorop_s1688wgrad_optimized_tf` |
| 15.321 | 2 | 0.0025 | `sm80_xmma_wgrad_implicit_gemm_indexed_tf32f32_tf32f32_f32_nhwckrsc_nhwc_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_execute_kern` |
| 12.528 | 960 | 0.002 | `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<long, at::native::func_wrapper_t<long, at::native::MaxNanFunctor<long> >, unsign` |

## top CPU ops (self time)

| ms | calls | op |
|---|---|---|
| 247.07 | 210 | aten::ne |
| 244.21 | 300 | aten::eq |
| 216.24 | 420 | aten::any |
| 165.55 | 13413 | aten::copy_ |
| 157.19 | 9094 | aten::mul |
| 155.38 | 25415 | aten::empty |
| 148.87 | 3670 | aten::addmm |
| 144.11 | 704 | aten::convolution_backward |
| 116.63 | 5043 | aten::add |
| 102.01 | 830 | aten::abs |
| 93.54 | 25 | aten::linalg_vector_norm |
| 81.36 | 11247 | aten::empty_strided |
| 79.45 | 670 | aten::cudnn_convolution |
| 67.09 | 2196 | aten::sum |
| 64.94 | 3784 | aten::cat |
| 64.31 | 2908 | aten::native_layer_norm |
| 58.02 | 6560 | aten::linear |
| 57.09 | 375 | aten::div |
| 53.88 | 1580 | aten::mm |
| 53.76 | 3840 | aten::index_select |
| 52.68 | 3840 | aten::gather |
| 49.15 | 28854 | aten::view |
| 48.72 | 50334 | aten::as_strided |
| 48.59 | 9652 | aten::_to_copy |
| 44.78 | 5362 | aten::add_ |

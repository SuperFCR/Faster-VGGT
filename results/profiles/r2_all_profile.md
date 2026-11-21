# profile

tree vggt, config configs/r2_all.json

5 iterations after warmup of 6; instrumented (not comparable with `time`)

window 6.07 s for 5 iterations; GPU busy 4.49 s (74%); kernel time 4.49 s (rank 0)

host time per iteration (s): 1.885, 0.9, 1.107, 1.289, 0.893

## kernel time by category (ms, rank 0)

| category | ms |
|---|---|
| gemm | 1947.6 |
| attention | 947.6 |
| elementwise/reduce | 794.1 |
| memcpy/cast | 266.0 |
| conv | 184.6 |
| norm | 152.6 |
| optimizer | 72.7 |
| other | 10.0 |

## GPU time inside model calls (forward only)

| call | GPU ms | calls | CPU ms |
|---|---|---|---|
| aggregator | 5038.52 | 20 | 1907.84 |
| camera_head | 79.7 | 20 | 243.66 |
| depth_head | 1204.23 | 20 | 823.0 |
| dinov2_patch_embed | 974.83 | 20 | 372.71 |
| multitask_loss | 669.4 | 20 | 1117.67 |

## top kernels

| ms | calls | share | kernel |
|---|---|---|---|
| 940.21 | 720 | 0.2094 | `void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<64, 128, 128, 4, false, false, cutlass::bfloat16_t, Flash_kernel_traits<64, 128` |
| 665.197 | 2016 | 0.1482 | `ampere_bf16_s16816gemm_bf16_128x128_ldg8_relu_f2f_stages_32x5_tn` |
| 341.231 | 864 | 0.076 | `ampere_bf16_s16816gemm_bf16_128x256_ldg8_relu_f2f_stages_64x3_tn` |
| 223.158 | 20 | 0.0497 | `void at::native::(anonymous namespace)::gatherKthValue<float, int, 1>(at::cuda::detail::TensorInfo<float const, int>, int, int, int, int, at` |
| 184.563 | 2746 | 0.0411 | `void cudnn::engines_precompiled::convertTensor_kernel<float, float, float, (cudnnKernelDataType_t)2>(float, float const*, float, float*, uns` |
| 166.817 | 288 | 0.0372 | `sm80_xmma_dgrad_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nhwc_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_execute_kernel__5x_c` |
| 155.69 | 242 | 0.0347 | `void cutlass__5x_cudnn::Kernel<cutlass_tensorop_s1688fprop_optimized_tf32_128x128_16x4_nhwc_align4>(cutlass_tensorop_s1688fprop_optimized_tf` |
| 147.257 | 320 | 0.0328 | `_ZN17cutlass__5x_cudnn6KernelINS_4conv6kernel23ImplicitGemmConvolutionINS1_11threadblock22ImplicitGemmMultistageINS_4gemm9GemmShapeILi256ELi` |
| 124.796 | 40 | 0.0278 | `sm80_xmma_wgrad_implicit_gemm_indexed_tf32f32_tf32f32_f32_nhwckrsc_nhwc_tilesize64x128x32_stage5_warpsize2x2x1_g1_tensor16x8x8_execute_kerne` |
| 98.293 | 40 | 0.0219 | `Memcpy HtoD (Pinned -> Device)` |
| 92.888 | 720 | 0.0207 | `void at::native::vectorized_elementwise_kernel<4, at::native::GeluCUDAKernelImpl(at::TensorIteratorBase&, at::native::GeluType)::{lambda()#2` |
| 89.689 | 5083 | 0.02 | `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>, false>(int, at::native::CUDAFu` |
| 79.043 | 1415 | 0.0176 | `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<float> >(at::TensorIteratorBase&,` |
| 78.66 | 100 | 0.0175 | `void at::native::(anonymous namespace)::upsample_bilinear2d_nhwc_out_frame<float, float, int>(float, float, bool, int, int, int, int, int, f` |
| 73.192 | 100 | 0.0163 | `void at::native::(anonymous namespace)::upsample_bilinear2d_backward_nhwc_out_frame<float, float, int>(int, int, int, int, float, float, boo` |
| 65.445 | 720 | 0.0146 | `_ls_add_layernorm_bf16_kernel` |
| 61.735 | 50 | 0.0138 | `void (anonymous namespace)::indexing_backward_kernel_stride_1<float>(long const*, long const*, float const*, float*, long, long, long, long,` |
| 50.611 | 720 | 0.0113 | `_ls_add_kernel` |
| 50.053 | 1620 | 0.0111 | `void at::native::reduce_kernel<128, 4, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, f` |
| 38.282 | 20 | 0.0085 | `sm80_xmma_fprop_implicit_gemm_indexed_wo_smem_tf32f32_tf32f32_f32_nhwckrsc_nhwc_tilesize128x32x64_stage1_warpsize4x1x1_g1_tensor16x8x8_align` |
| 37.5 | 720 | 0.0084 | `_layernorm_to_bf16_kernel` |
| 36.745 | 480 | 0.0082 | `_fused_qkv_qknorm_rope_kernel` |
| 33.662 | 480 | 0.0075 | `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&):` |
| 26.776 | 2171 | 0.006 | `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&):` |
| 25.807 | 480 | 0.0057 | `ampere_sgemm_128x64_nt` |
| 24.698 | 330 | 0.0055 | `void at::native::vectorized_elementwise_kernel<4, at::native::BinaryFunctor<float, float, float, at::native::(anonymous namespace)::threshol` |
| 23.932 | 78 | 0.0053 | `void cutlass__5x_cudnn::Kernel<cutlass_tensorop_s1688dgrad_optimized_tf32_128x128_16x4_nhwc_unity_stride_align4>(cutlass_tensorop_s1688dgrad` |
| 23.0 | 90 | 0.0051 | `void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x128_16x5_nn_align4>(cutlass_80_tensorop_s1688gemm_128x128_16x5_nn_align4::Params)` |
| 21.024 | 562 | 0.0047 | `void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_64x64_32x6_nn_align4>(cutlass_80_tensorop_s1688gemm_64x64_32x6_nn_align4::Params)` |
| 20.744 | 70 | 0.0046 | `sm80_xmma_fprop_implicit_gemm_indexed_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize256x64x32_stage3_warpsize4x2x1_g1_tensor16x8x8_execute_kerne` |
| 18.153 | 655 | 0.004 | `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::launch_clamp_scalar(at::TensorIteratorBase&, c10::Scala` |
| 17.882 | 482 | 0.004 | `void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x64_32x6_tn_align4>(cutlass_80_tensorop_s1688gemm_128x64_32x6_tn_align4::Params)` |
| 16.907 | 84 | 0.0038 | `void cutlass__5x_cudnn::Kernel<cutlass_tensorop_s1688wgrad_optimized_tf32_256x128_16x3_nhwc_align4>(cutlass_tensorop_s1688wgrad_optimized_tf` |
| 15.364 | 2 | 0.0034 | `sm80_xmma_wgrad_implicit_gemm_indexed_tf32f32_tf32f32_f32_nhwckrsc_nhwc_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_execute_kern` |
| 12.186 | 20 | 0.0027 | `void at::native::(anonymous namespace)::multi_tensor_apply_kernel<at::native::(anonymous namespace)::TensorListMetadata<1>, at::native::(ano` |
| 11.414 | 10 | 0.0025 | `void at::native::(anonymous namespace)::multi_tensor_apply_kernel<at::native::(anonymous namespace)::TensorListScalarListMetadata<float, 3>,` |
| 11.266 | 40 | 0.0025 | `void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x128_32x3_tn_align4>(cutlass_80_tensorop_s1688gemm_256x128_32x3_tn_align4::Params)` |
| 10.977 | 24 | 0.0024 | `void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x128_16x5_tn_align4>(cutlass_80_tensorop_s1688gemm_128x128_16x5_tn_align4::Params)` |
| 10.75 | 66 | 0.0024 | `sm80_xmma_fprop_implicit_gemm_indexed_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x16_stage4_warpsize2x2x1_g1_tensor16x8x8_alignc4_exec` |
| 10.21 | 20 | 0.0023 | `void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x128_32x5_nt_align4>(cutlass_80_tensorop_s1688gemm_128x128_32x5_nt_align4::Params)` |

## top CPU ops (self time)

| ms | calls | op |
|---|---|---|
| 171.67 | 3670 | aten::addmm |
| 151.04 | 704 | aten::convolution_backward |
| 121.4 | 15756 | aten::empty |
| 69.25 | 4134 | aten::mul |
| 58.59 | 22494 | aten::view |
| 56.31 | 1580 | aten::mm |
| 54.81 | 2196 | aten::sum |
| 48.47 | 5362 | aten::add_ |
| 45.44 | 3453 | aten::copy_ |
| 42.34 | 3680 | aten::linear |
| 32.64 | 13610 | aten::transpose |
| 29.83 | 25129 | aten::as_strided |
| 29.61 | 1683 | aten::add |
| 24.54 | 7110 | aten::t |
| 23.23 | 9169 | aten::reshape |
| 21.76 | 670 | aten::cudnn_convolution |
| 20.56 | 2232 | aten::empty_strided |
| 20.28 | 920 | aten::gelu |
| 19.87 | 909 | aten::cat |
| 18.32 | 880 | aten::scaled_dot_product_attention |
| 16.47 | 260 | aten::nonzero |
| 16.32 | 720 | aten::_flash_attention_forward |
| 14.98 | 508 | aten::native_layer_norm |
| 13.96 | 1609 | aten::fill_ |
| 12.33 | 2370 | aten::clone |

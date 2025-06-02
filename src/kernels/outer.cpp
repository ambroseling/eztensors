


#include <arm_neon.h>
#include <iostream>
#include <vector>


std::vector<float> kernel_outer_product_naive(
    std::shared_ptr<std::vector<float>> A, 
    std::shared_ptr<std::vector<float>> B ){
    // basically 2 vectors, A is of size n, B is of size of m, output is of size n x m
    int c = B->size(); int r = A->size();
    std::vector<float> C(r*c, 0);
    for(int i=0;i<B->size();i++){
        for(int j=0;j<A->size();j++){
            C[j*c+i] = (*A)[j] * (*B)[i];
        }
    }
    return C;
}

std::vector<float> kernel_outer_product_simd(
    std::shared_ptr<std::vector<float>> A, 
    std::shared_ptr<std::vector<float>> B ){
    // basically 2 vectors, A is of size n, B is of size of m, output is of size n x m
    int c = B->size(); int r = A->size();
    std::vector<float> C(r*c, 0);
    for(int i=0;i<B->size();i++){
        for(int j=0;j<A->size();j++){
            C[j*c+i] = (*A)[j] * (*B)[i];
        }
    }
    return C;
}

// std::vector<float> kernel_outer_product_simd(
//     std::shared_ptr<std::vector<float>> A, 
//     std::shared_ptr<std::vector<float>> B ){
//     // basically 2 vectors, A is of size n, B is of size of m, output is of size n x m
//     int c = B->size(); int r = A->size();
//     std::vector<float> C(r*c, 0);
//     float32x4_t a_pack, c_pack;

//     for(int i=0;i<B->size();i++){
//         for(int j=0;j<A->size();j+=4){
//             a_pack = vld1q_f32(A->data()+j);
//             if (A->size() - j < 4){
//                 float32x4_t zeroes = vdupq_n_f32(0.0f); 
//                 uint32x4_t mask = vdupq_n_u32(~0);
//                 for (int k = 0; k < A->size() - j; ++k) {
//                     vsetq_lane_u32(1, mask, k);
//                 }
//                 vsetq_lane_u32(0, mask, k);
//                 a_pack = vbslq_f32(mask, a_pack, zeroes);
//             }
//             c_pack = vfmaq_n_f32(c_pack, a_pack, (*B)[i]);
//             vst1q_f32(&C[j*c+i], c_pack);
//         }
//     }
//     return C;
// }
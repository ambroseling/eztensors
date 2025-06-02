
#include <arm_neon.h>
#include <iostream>
#include <vector>
#define MR 16
#define NR 6

void pack_buffer(  ) {

}


void matmul_naive(std::shared_ptr<std::vector<float>> A, std::shared_ptr<std::vector<float>> B, std::shared_ptr<std::vector<float>> C, int M, int N, int K){
    for (int i=0; i < M; i++){
        for (int j=0; j < N; j++) {
            for (int p=0; p < K; p++) {
            (*C)[i*N+j] += (*A)[i*K+p] * (*B)[p*K+j];
            }
        }  
    }
}

void kernel_16x6(float* A, float* B, float* C, int M, int N, int K, int n){

    float32x4_t c_accum[6][4] = {0};
    float32x4_t a0_pack; 
    float32x4_t a1_pack;
    float32x4_t a2_pack;
    float32x4_t a3_pack;
    float32x4_t b0_pack;

    // go through all the columns in A (there are K number of columns in A)
    for (int p =0; p < K ;p++) {

        // We load 1 column of A into the SIMD registers a0_pack to a3_pack
        a0_pack = vsetq_lane_f32(A[0*K+p] ,a0_pack, 0);
        a0_pack = vsetq_lane_f32(A[1*K+p] ,a0_pack, 1);
        a0_pack = vsetq_lane_f32(A[2*K+p] ,a0_pack, 2);
        a0_pack = vsetq_lane_f32(A[3*K+p] ,a0_pack, 3);

        a1_pack = vsetq_lane_f32(A[4*K+p] ,a1_pack, 0);
        a1_pack = vsetq_lane_f32(A[5*K+p] ,a1_pack, 1);
        a1_pack = vsetq_lane_f32(A[6*K+p] ,a1_pack, 2);
        a1_pack = vsetq_lane_f32(A[7*K+p] ,a1_pack, 3);

        a2_pack = vsetq_lane_f32(A[8*K+p] ,a2_pack, 0);
        a2_pack = vsetq_lane_f32(A[9*K+p] ,a2_pack, 1);
        a2_pack = vsetq_lane_f32(A[10*K+p] ,a2_pack, 2);
        a2_pack = vsetq_lane_f32(A[11*K+p] ,a2_pack, 3);

        a3_pack = vsetq_lane_f32(A[12*K+p] ,a3_pack, 0);
        a3_pack = vsetq_lane_f32(A[13*K+p] ,a3_pack, 1);
        a3_pack = vsetq_lane_f32(A[14*K+p] ,a3_pack, 2);
        a3_pack = vsetq_lane_f32(A[15*K+p] ,a3_pack, 3);

        // we go down a row of B
        // we broadcast value of B to the size of 4 lane register then do a MAC
        c_accum[0][0] =  vfmaq_n_f32(c_accum[0][0], a0_pack, B[p*N]);
        c_accum[0][1] =  vfmaq_n_f32(c_accum[0][1], a1_pack, B[p*N]);
        c_accum[0][2] =  vfmaq_n_f32(c_accum[0][2], a2_pack, B[p*N]);
        c_accum[0][3] =  vfmaq_n_f32(c_accum[0][3], a3_pack, B[p*N]);

        c_accum[1][0] =  vfmaq_n_f32(c_accum[1][0], a0_pack, B[p*N+1]);
        c_accum[1][1] =  vfmaq_n_f32(c_accum[1][1], a1_pack, B[p*N+1]);
        c_accum[1][2] =  vfmaq_n_f32(c_accum[1][2], a2_pack, B[p*N+1]);
        c_accum[1][3] =  vfmaq_n_f32(c_accum[1][3], a3_pack, B[p*N+1]);

        c_accum[2][0] =  vfmaq_n_f32(c_accum[2][0], a0_pack, B[p*N+2]);
        c_accum[2][1] =  vfmaq_n_f32(c_accum[2][1], a1_pack, B[p*N+2]);
        c_accum[2][2] =  vfmaq_n_f32(c_accum[2][2], a2_pack, B[p*N+2]);
        c_accum[2][3] =  vfmaq_n_f32(c_accum[2][3], a3_pack, B[p*N+2]);

        c_accum[3][0] =  vfmaq_n_f32(c_accum[3][0], a0_pack, B[p*N+3]);
        c_accum[3][1] =  vfmaq_n_f32(c_accum[3][1], a1_pack, B[p*N+3]);
        c_accum[3][2] =  vfmaq_n_f32(c_accum[3][2], a2_pack, B[p*N+3]);
        c_accum[3][3] =  vfmaq_n_f32(c_accum[3][3], a3_pack, B[p*N+3]);

        c_accum[4][0] =  vfmaq_n_f32(c_accum[4][0], a0_pack, B[p*N+4]);
        c_accum[4][1] =  vfmaq_n_f32(c_accum[4][1], a1_pack, B[p*N+4]);
        c_accum[4][2] =  vfmaq_n_f32(c_accum[4][2], a2_pack, B[p*N+4]);
        c_accum[4][3] =  vfmaq_n_f32(c_accum[4][3], a3_pack, B[p*N+4]);

        c_accum[5][0] =  vfmaq_n_f32(c_accum[5][0], a0_pack, B[p*N+5]);
        c_accum[5][1] =  vfmaq_n_f32(c_accum[5][1], a1_pack, B[p*N+5]);
        c_accum[5][2] =  vfmaq_n_f32(c_accum[5][2], a2_pack, B[p*N+5]);
        c_accum[5][3] =  vfmaq_n_f32(c_accum[5][3], a3_pack, B[p*N+5]);
    }

    // this goes column by column in C for storing
    for ( int j=0; j < n; j++) {

        vst1q_lane_f32(&C[0*N+j], c_accum[j][0], 0);
        vst1q_lane_f32(&C[1*N+j], c_accum[j][0], 1);
        vst1q_lane_f32(&C[2*N+j], c_accum[j][0], 2);
        vst1q_lane_f32(&C[3*N+j], c_accum[j][0], 3);

        vst1q_lane_f32(&C[4*N+j], c_accum[j][1], 0);
        vst1q_lane_f32(&C[5*N+j], c_accum[j][1], 1);
        vst1q_lane_f32(&C[6*N+j], c_accum[j][1], 2);
        vst1q_lane_f32(&C[7*N+j], c_accum[j][1], 3);

        vst1q_lane_f32(&C[8*N+j], c_accum[j][2], 0);
        vst1q_lane_f32(&C[9*N+j], c_accum[j][2], 1);
        vst1q_lane_f32(&C[10*N+j], c_accum[j][2], 2);
        vst1q_lane_f32(&C[11*N+j], c_accum[j][2], 3);

        vst1q_lane_f32(&C[12*N+j], c_accum[j][3], 0);
        vst1q_lane_f32(&C[13*N+j], c_accum[j][3], 1);
        vst1q_lane_f32(&C[14*N+j], c_accum[j][3], 2);
        vst1q_lane_f32(&C[15*N+j], c_accum[j][3], 3);
    }
}

void matmul_simd(std::shared_ptr<std::vector<float>> A, 
                 std::shared_ptr<std::vector<float>> B, 
                 std::shared_ptr<std::vector<float>> C, int M, int N, int K){
    static std::vector<float> block_A_buffer;
    static std::vector<float> block_B_buffer;                    
    for ( int i=0; i < M; i += MR) {
        const int m = std::min(MR, M-i);
        if ( m != MR ){

        }
        for ( int j=0; j < N; j += NR) {
            const int n = std::min(NR, N - j);
            if ( n != NR ){

            }
            kernel_16x6(A->data()+K*i,B->data()+j,C->data()+i*N+j,M,N,K,n);    
    }  
    } 
}



// __m256i masks[2]; -> each holds 8 32bit integers, each element refers to a data element in b
// if (m != 16) {
//   const uint32_t bit_mask = 65535;
//   masks[0] = _mm256_setr_epi32(bit_mask << (m + 15),
//                                bit_mask << (m + 14),
//                                bit_mask << (m + 13),
//                                bit_mask << (m + 12),
//                                bit_mask << (m + 11),
//                                bit_mask << (m + 10),
//                                bit_mask << (m + 9),
//                                bit_mask << (m + 8));
//   masks[1] = _mm256_setr_epi32(bit_mask << (m + 7),
//                                bit_mask << (m + 6),
//                                bit_mask << (m + 5),
//                                bit_mask << (m + 4),
//                                bit_mask << (m + 3),
//                                bit_mask << (m + 2),
//                                bit_mask << (m + 1),
//                                bit_mask << m);
//   for (int j = 0; j < n; j++) {
//     _mm256_maskstore_ps(&C_start[j * M], masks[0], C_accum[j][0]);
//     _mm256_maskstore_ps(&C_start[j * M + 8], masks[1], C_accum[j][1]);
//   }
// }
#ifndef GE_RECURSIVE_PARALLEL_KERNELS
#define GE_RECURSIVE_PARALLEL_KERNELS

#include <omp.h> // NOTE: You need to add this #include to the top of GE.cc file as well
using namespace std;

/* How to call?
#pragma omp parallel
{
    #pragma omp single
    {
        #pragma omp task
        ge_recursive_parallel_kernelA(X, size, args.top_x1, args.top_y1, args.top_x4,
                                      problem_size, threshold2, 2);
        #pragma omp taskwait
    }
}
*/
int get_pt(int x, int y, int st){
    return x + y*st;
}


/* How to call?
#pragma omp parallel
{
    #pragma omp single
    {
        #pragma omp task
        ge_recursive_parallel_kernelD(X, U, V, W, size, args.top_x1, args.top_y1, args.top_x4,
                                      problem_size, threshold2, 2);
        #pragma omp taskwait
    }
}
*/
void D_non_legion_task(double* input1, double* input2, double* input3, double* input4, int stride1, int stride2, int stride3, int stride4, int l1, int l2, int l3, int size, int cilk_threshold, int recurisve_fan_out) { // always pass the value 2
    if(size <= cilk_threshold) {
        for(int k = l1; k < l1+size; k++){
        for(int i = l2; i < l2+size; i++){
            for(int j = l3; j < l3+size; j++){
                if((k<i)&&(k<=j)){
                    int i_relative = i%size;
                    int j_relative = j%size;
                    int k_relative = k%size;
                    input1[get_pt(i_relative,j_relative,stride1)] -= (input2[get_pt(i_relative,k_relative,stride2)]/(input4[get_pt(k_relative,k_relative,stride4)]))*(input3[get_pt(k_relative,j_relative,stride3)]); 
                }
            }
        }
      }
    }
    else {
        int tile_size = size / recurisve_fan_out;
        for(int kk = 0; kk < recurisve_fan_out; ++kk) {
            for(int ii = 0; ii < recurisve_fan_out; ++ii) {
                for(int jj = 0; jj < recurisve_fan_out; ++jj) {
                    #pragma omp task
                    D_non_legion_task(input1, input2, input3, input4, stride1, stride2, stride3, stride4, l1+kk*tile_size, l2+ii*tile_size, l3+jj*tile_size, tile_size, cilk_threshold, recurisve_fan_out);
                    // ge_recursive_parallel_kernelD(X, U, V, W, problem_size, tile_size,
                    //                                i_lb + ii * tile_size,
                    //                                j_lb + jj * tile_size,
                    //                                k_lb + kk * tile_size,
                    //                                recurisve_fan_out, base_size);
                }
            }
            #pragma omp taskwait
        }
    }
}


/* How to call?
#pragma omp parallel
{
    #pragma omp single
    {
        #pragma omp task
        ge_recursive_parallel_kernelC(X, V, size, args.top_x1, args.top_y1, args.top_x4,
                                      problem_size, threshold2, 2);
        #pragma omp taskwait
    }
}
*/
void C_non_legion_task(double* input1, double* input2, int stride1, int stride2, int l1, int l2, int l3, int size, int cilk_threshold, int recurisve_fan_out) { // always pass the value 2
    if(size <= cilk_threshold) {
        for(int k = l1; k < l1+size; k++){
        for(int i = l2; i < l2+size; i++){
            for(int j = l3; j < l3+size; j++){
                if((k<i)&&(k<=j)){
                    int i_relative = i%size;
                    int j_relative = j%size;
                    int k_relative = k%size;
                    input1[get_pt(i_relative,j_relative,stride1)] -= (input1[get_pt(i_relative,k_relative,stride1)]/(input2[get_pt(k_relative,k_relative,stride2)]))*(input2[get_pt(k_relative,j_relative,stride2)]); 
                }
            }
        }
      }
    }
    else {
        int tile_size = size / recurisve_fan_out;
        for(int kk = 0; kk < recurisve_fan_out; ++kk) {
            for(int ii = 0; ii < recurisve_fan_out; ++ii) {
                #pragma omp task
                C_non_legion_task(input1, input2, stride1, stride2, l1+kk*tile_size, l2+ii*tile_size, l3+kk*tile_size, tile_size, cilk_threshold, recurisve_fan_out);
                // ge_recursive_parallel_kernelC(X, V, problem_size,
                //                               tile_size,
                //                               i_lb + ii * tile_size,
                //                               j_lb + kk * tile_size,
                //                               k_lb + kk * tile_size,
                //                               recurisve_fan_out, base_size);
            }
            #pragma omp taskwait
            for(int jj = kk+1; jj < recurisve_fan_out; ++jj) {
                for(int ii = 0; ii < recurisve_fan_out; ++ii) {
                    // #pragma omp task
                    D_non_legion_task(input1, input1, input2, input2, stride1, stride1, stride2, stride2, l1+kk*tile_size, l2+ii*tile_size, l3+jj*tile_size, tile_size, cilk_threshold, recurisve_fan_out);
                    // ge_recursive_parallel_kernelD(X, X, V, V, problem_size, tile_size,
                    //                               i_lb + ii * tile_size,
                    //                               j_lb + jj * tile_size,
                    //                               k_lb + kk * tile_size,
                    //                               recurisve_fan_out, base_size);
                }
            }
            #pragma omp taskwait
        }
    }
}

/* How to call?
#pragma omp parallel
{
    #pragma omp single
    {
        #pragma omp task
        ge_recursive_parallel_kernelB(X, U, size, args.top_x1, args.top_y1, args.top_x4,
                                      problem_size, threshold2, 2);
        #pragma omp taskwait
    }
}
*/
void B_non_legion_task(double* input1, double* input2, int stride1, int stride2, int l1, int l2, int l3, int size, int cilk_threshold, int recurisve_fan_out) { // always pass the value 2
    if(size <= cilk_threshold) {
        for(int k = l1; k < l1+size; k++){
        for(int i = l2; i < l2+size; i++){
            for(int j = l3; j < l3+size; j++){
                if((k<i)&&(k<=j)){
                    int i_relative = i%size;
                    int j_relative = j%size;
                    int k_relative = k%size;
                    input1[get_pt(i_relative,j_relative,stride1)] -= (input2[get_pt(i_relative,k_relative,stride2)]/(input2[get_pt(k_relative,k_relative,stride2)]))*(input1[get_pt(k_relative,j_relative,stride1)]); 
                }
            }
        }
      }
    }
    else {
        int tile_size = size / recurisve_fan_out;
        for(int kk = 0; kk < recurisve_fan_out; ++kk) {
            for(int jj = 0; jj < recurisve_fan_out; ++jj) {
                #pragma omp task
                B_non_legion_task(input1, input2, stride1, stride2, l1+kk*tile_size, l2+kk*tile_size, l3+jj*tile_size, tile_size, cilk_threshold, recurisve_fan_out);
                // ge_recursive_parallel_kernelB(X, U, problem_size,
                //                               tile_size,
                //                               i_lb + kk * tile_size,
                //                               j_lb + jj * tile_size,
                //                               k_lb + kk * tile_size,
                //                               recurisve_fan_out, base_size);
            }
            #pragma omp taskwait
            for(int ii = kk+1; ii < recurisve_fan_out; ++ii) {
                for(int jj = 0; jj < recurisve_fan_out; ++jj) {
                    #pragma omp task
                    D_non_legion_task(input1, input2, input1, input2, stride1, stride2, stride1, stride2, l1+kk*tile_size, l2+ii*tile_size, l3+jj*tile_size, tile_size, cilk_threshold, recurisve_fan_out);
                    // ge_recursive_parallel_kernelD(X, U, X, U, problem_size, tile_size,
                    //                               i_lb + ii * tile_size,
                    //                               j_lb + jj * tile_size,
                    //                               k_lb + kk * tile_size,
                    //                               recurisve_fan_out, base_size);
                }
            }
            #pragma omp taskwait
        }
    }
}


void A_non_legion_task(double* input, int stride, int l1, int l2, int l3, int size, int cilk_threshold, int recurisve_fan_out) { 
    if(size <= cilk_threshold) {
        for(int k = l1; k < l1+size; k++){
        for(int i = l2; i < l2+size; i++){
            for(int j = l3; j < l3+size; j++){
                if((k<i)&&(k<=j)){
                    int i_relative = i%size;
                    int j_relative = j%size;
                    int k_relative = k%size;
                    input[get_pt(i_relative,j_relative,stride)] -= (input[get_pt(i_relative,k_relative,stride)]/(input[get_pt(k_relative,k_relative,stride)]))*(input[get_pt(k_relative,j_relative,stride)]); 
                }
            }
        }
      }
    }
    else {
        int tile_size = size / recurisve_fan_out;
        for(int kk = 0; kk < recurisve_fan_out; ++kk) {
            A_non_legion_task(input, stride, l1+kk*tile_size, l2+kk*tile_size, l3+kk*tile_size, tile_size, cilk_threshold, recurisve_fan_out);
            // ge_recursive_parallel_kernelA(X, problem_size, tile_size,
            //                               i_lb + kk * tile_size,
            //                               j_lb + kk * tile_size,
            //                               k_lb + kk * tile_size,
            //                               recurisve_fan_out, base_size);
            // Calling functions B and C
            for(int ii = kk+1; ii < recurisve_fan_out; ++ii) {
                #pragma omp task
                B_non_legion_task(input, input, stride, stride, l1+kk*tile_size, l2+kk*tile_size, l3+ii*tile_size, tile_size, cilk_threshold, recurisve_fan_out);
                // ge_recursive_parallel_kernelB(X, X, problem_size, tile_size,
                //                               i_lb + kk * tile_size,
                //                               j_lb + ii * tile_size,
                //                               k_lb + kk * tile_size,
                //                               recurisve_fan_out, base_size);
                #pragma omp task
                C_non_legion_task(input, input, stride, stride, l1+kk*tile_size, l2+ii*tile_size, l3+kk*tile_size, tile_size, cilk_threshold, recurisve_fan_out);
                // ge_recursive_parallel_kernelC(X, X, problem_size, tile_size,
                //                               i_lb + ii * tile_size,
                //                               j_lb + kk * tile_size,
                //                               k_lb + kk * tile_size,
                //                               recurisve_fan_out, base_size);
            }
            #pragma omp taskwait
            // Calling functions D
            for(int ii = kk+1; ii < recurisve_fan_out; ++ii) {
                for(int jj = kk+1; jj < recurisve_fan_out; ++jj) {
                    #pragma omp task
                    D_non_legion_task(input, input, input, input, stride, stride, stride, stride, l1+kk*tile_size, l2+ii*tile_size, l3+jj*tile_size, tile_size, cilk_threshold, recurisve_fan_out);
                    // ge_recursive_parallel_kernelD(X, X, X, X, problem_size,
                    //                               tile_size, i_lb + ii * tile_size,
                    //                               j_lb + jj * tile_size,
                    //                               k_lb + kk * tile_size,
                    //                               recurisve_fan_out, base_size);
                }
            }
            #pragma omp taskwait
        }
    }
}

#endif
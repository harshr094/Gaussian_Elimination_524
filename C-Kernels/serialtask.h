#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath> 
#include <cstdio>
#include <vector>
#include <queue>
#include <utility>
#include <string>
using namespace std;

int get_pt(int x, int y, int st){
    return x + y*st;
}

void A_non_legion_task(double* input, int stride, int l1, int l2, int l3, int size){
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

void B_non_legion_task(double* input1, double* input2, int stride1, int stride2, int l1, int l2, int l3, int size){
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

void C_non_legion_task(double* input1, double* input2, int stride1, int stride2, int l1, int l2, int l3, int size){
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

void D_non_legion_task(double* input1, double* input2, double* input3, double* input4, int stride1, int stride2, int stride3, int stride4, int l1, int l2, int l3, int size){
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
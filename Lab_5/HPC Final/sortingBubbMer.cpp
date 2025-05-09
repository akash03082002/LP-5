#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <omp.h>

using namespace std;
using namespace chrono;


void bubbleSortSerial(vector<long int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
            }
        }
    }
}


void bubbleSortParallel(vector<long int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; i++) {
        #pragma omp parallel for
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
            }
        }
    }
}


void mergeSortSerial(vector<long int>& arr) {
    if (arr.size() <= 1) return;
    int mid = arr.size() / 2;
    vector<long int> left(arr.begin(), arr.begin() + mid);
    vector<long int> right(arr.begin() + mid, arr.end());
    mergeSortSerial(left);
    mergeSortSerial(right);
    merge(left.begin(), left.end(), right.begin(), right.end(), arr.begin());
}


void mergeSortParallel(vector<long int>& arr) {
    int n = arr.size();
    if (n <= 1) return;
    vector<long int> left(arr.begin(), arr.begin() + n / 2);
    vector<long int> right(arr.begin() + n / 2, arr.end());
    
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            mergeSortParallel(left);
        }
        #pragma omp section
        {
            mergeSortParallel(right);
        }
    }
    merge(left.begin(), left.end(), right.begin(), right.end(), arr.begin());
}

int main() {
    int n;
    cout << "Enter number of elements: ";
    cin >> n;
    
    vector<long int> arr1(n), arr2(n), arr3(n), arr4(n);
    for (int i = 0; i < n; i++) {
        arr1[i] = arr2[i] = arr3[i] = arr4[i] = rand() % n;
    }
    
  
    auto start = high_resolution_clock::now();
    bubbleSortSerial(arr1);
    auto end = high_resolution_clock::now();
    double bubble_seq_time = duration<double, milli>(end - start).count();
    
  
    start = high_resolution_clock::now();
    bubbleSortParallel(arr2);
    end = high_resolution_clock::now();
    double bubble_par_time = duration<double, milli>(end - start).count();
    
    
    start = high_resolution_clock::now();
    mergeSortSerial(arr3);
    end = high_resolution_clock::now();
    double merge_seq_time = duration<double, milli>(end - start).count();
    
   
    start = high_resolution_clock::now();
    mergeSortParallel(arr4);
    end = high_resolution_clock::now();
    double merge_par_time = duration<double, milli>(end - start).count();
    

    cout << "TIME TAKEN FOR SEQUENTIAL BUBBLE SORT: " << bubble_seq_time << " ms" << endl;
    cout << "TIME TAKEN FOR PARALLEL BUBBLE SORT: " << bubble_par_time << " ms" << endl;
    cout << "Speedup Factor (Bubble Sort): " << bubble_seq_time / bubble_par_time << endl << endl;
    
    cout << "TIME TAKEN FOR SEQUENTIAL MERGE SORT: " << merge_seq_time << " ms" << endl;
    cout << "TIME TAKEN FOR PARALLEL MERGE SORT: " << merge_par_time << " ms" << endl;
    cout << "Speedup Factor (Merge Sort): " << merge_seq_time / merge_par_time << endl;
    
    return 0;
}
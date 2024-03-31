#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <oneapi/tbb.h>
#include <chrono>

using namespace std;

const int N = 8;

int get_random_number(int min, int max) {
    return rand() % (max - min + 1) + min;
}

void solve_gauss(double** matrix, double x[N]) {
    for (int i = 0; i < N - 1; i++) {
        tbb::parallel_for(tbb::blocked_range<int>(i + 1, N), [&](const tbb::blocked_range<int>& range) {
            for (int j = range.begin(); j < range.end(); j++) {
                double coeff = matrix[j][i] / matrix[i][i];
                for (int k = i; k < N + 1; k++) {
                    matrix[j][k] = matrix[j][k] - coeff * matrix[i][k];
                }
            }
        });
    }

    for (int i = N - 1; i >= 0; i--) {
        x[i] = matrix[i][N];
        for (int j = i + 1; j < N; j++) {
            x[i] -= matrix[i][j] * x[j];
        }
        x[i] = x[i] / matrix[i][i];
    }
}

int main() {
    srand(time(0));

    double** matrix = new double* [N];
    for (int i = 0; i < N; ++i) {
        matrix[i] = new double[N + 1];
        for (int j = 0; j < N + 1; ++j) {
            matrix[i][j] = get_random_number(1, 20);
        }
    }

    cout << "Matrix:" << endl;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N + 1; ++j) {
            cout << matrix[i][j] << "\t";
        }
        cout << endl;
    }
    cout << endl;

    double x[N];

    auto begin = std::chrono::steady_clock::now();
    solve_gauss(matrix, x);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> solving_time = end - begin;
    
    cout << "X:\n";
    for (int i = 0; i < N; i++) {
        cout << "x" << i+1 << " = " << x[i] << endl;
    }

    cout << "Solution time: " << solving_time.count() << endl;

    for (int i = 0; i < N; ++i) {
        delete[] matrix[i];
    }
    delete[] matrix;

    return 0;
}
#include <chrono>
#include <cstring>
#include <thread>
#include <iomanip>
#include <iostream>

#include <mpi.h>

#include "solver.hpp"
#include "../matrix/matrix.hpp"
#include "../output/output.hpp"

using std::memcpy;

using std::cout;
using std::endl;
using std::fixed;
using std::flush;
using std::setprecision;
using std::setw;
using std::this_thread::sleep_for;
using std::chrono::microseconds;

void solveSeq(int rows, int cols, int iterations, double td, double h, int sleep, double ** matrix) {
    double c, l, r, t, b;
    
    double h_square = h * h;

    double * linePrevBuffer = new double[cols];
    double * lineCurrBuffer = new double[cols];

    for(int k = 0; k < iterations; k++) {

        memcpy(linePrevBuffer, matrix[0], cols * sizeof(double));
        for(int i = 1; i < rows - 1; i++) {

            memcpy(lineCurrBuffer, matrix[i], cols * sizeof(double));
            for(int j = 1; j < cols - 1; j++) {
                c = lineCurrBuffer[j];
                t = linePrevBuffer[j];
                b = matrix[i + 1][j];
                l = lineCurrBuffer[j - 1];
                r = lineCurrBuffer[j + 1];


                sleep_for(microseconds(sleep));
                matrix[i][j] = c * (1.0 - 4.0 * td / h_square) + (t + b + l + r) * (td / h_square);
            }

            memcpy(linePrevBuffer, lineCurrBuffer, cols * sizeof(double));
        }
    }
}

void solvePar(int rows, int cols, int iterations, double td, double h, int sleep, double ** matrix) {
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    double c, l, r, t, b;
    double h_square = h * h;
    int current_rank;

    double * linePrevBuffer = new double[cols];
    double * currentBuffer = new double[cols];
    double * lineAfterBuffer = new double[cols];


    if(rank == 0) {

        for(int k = 0; k < iterations; ++k) {
        
            current_rank = 1;

            for(int i = 1; i < rows - 1; ++i) {

                current_rank = current_rank == world_size ? 1 : current_rank;

                memcpy(linePrevBuffer, matrix[i - 1], cols * sizeof(double));
                memcpy(currentBuffer, matrix[i], cols * sizeof(double));
                memcpy(lineAfterBuffer, matrix[i + 1], cols * sizeof(double));

                MPI_Send(linePrevBuffer, cols, MPI_DOUBLE, current_rank, i, MPI_COMM_WORLD);
                MPI_Send(currentBuffer, cols, MPI_DOUBLE, current_rank, i, MPI_COMM_WORLD);
                MPI_Send(lineAfterBuffer, cols, MPI_DOUBLE, current_rank, i, MPI_COMM_WORLD);

                current_rank += 1;
            }

            current_rank = 1;

            for(int i = 1; i < rows - 1; ++i) {

                current_rank = current_rank == world_size ? 1 : current_rank;

                MPI_Recv(currentBuffer, cols, MPI_DOUBLE, current_rank, k, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                memcpy(matrix[i], currentBuffer, cols * sizeof(double));

                current_rank += 1;
            }
        }
    }
    else {

        for(int k = 0; k < iterations; ++k) {

            double * top_buffer = new double[cols];
            double * center_buffer = new double[cols];
            double * bottom_buffer = new double[cols];
            double * sending_buffer = new double[cols];

            for(int i = rank; i < rows - 1; i += world_size - 1) {

                MPI_Recv(top_buffer, cols, MPI_DOUBLE, 0, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(center_buffer, cols, MPI_DOUBLE, 0, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(bottom_buffer, cols, MPI_DOUBLE, 0, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                for(int j = 1; j < cols - 1; ++j) {

                    t = top_buffer[j];
                    l = center_buffer[j - 1];
                    c = center_buffer[j];
                    r = center_buffer[j + 1];
                    b = bottom_buffer[j];

                    sending_buffer[j] = c * (1.0 - 4.0 * td / h_square) + (t + b + l + r) * (td / h_square);
                }

                MPI_Send(sending_buffer, cols, MPI_DOUBLE, 0, k, MPI_COMM_WORLD);
            }
        }
        
        deallocateMatrix(rows, matrix); 
    }
    sleep_for(microseconds(500000));
}

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include <stddef.h>
#include <string.h>
#include <mpi.h>

#define SLEEP_TIME    5
#define MASTER_WORKER 0
#define LEFT_TAG      3
#define RIGHT_TAG     4
#define DONE_TAG      5
#define START_TAG     6

// #define DEBUG 1

#ifdef DEBUG
#  define DEBUG_PRINT(...) do{ fprintf( stderr, __VA_ARGS__ ); } while(0)
#else
#  define DEBUG_PRINT(...) do{ } while (0)
#endif

void chaleur_seq(int m, int n, double matrix[2][m][n], int np, double td, double h);
void chaleur_par(int m, int n, double matrix[2][m][n], int np, double td, double h);

void matrix_init(int m, int n, double matrix[2][m][n]);
void matrix_zero(int m, int n, double matrix[2][m][n]);
void matrix_print(int m, int n, double matrix[m][n]);

void types_init();
double get_current_time();

MPI_Datatype mpi_message_type;

struct Message {
  int x_offset;
  int y_offset;
  int left;
  int right;
};

int main(int argc, char **argv) {
  int err, rank, m, n, np, cpus;
  double start, end, td, h;

  m    = atoi(argv[1]);
  n    = atoi(argv[2]);
  np   = atoi(argv[3]);
  td   = atof(argv[4]);
  h    = atof(argv[5]);
  cpus = atoi(argv[6]);

  err = MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  types_init();

  if (rank == MASTER_WORKER) {
    printf("m=%d, n=%d, np=%d, td=%.5f, h=%.5f\n", m, n, np, td, h);
    double matrix[2][m][n];

    printf("Version parallèle\n");
    matrix_init(m, n, matrix);
    printf("init\n====\n");
    matrix_print(m, n, matrix[0]);
    start = get_current_time();
    chaleur_par(m, n, matrix, np, td, h);
    end = get_current_time();
    double dt_par = end - start;
    printf("final\n=====\n");
    matrix_print(m, n, matrix[0]);


    printf("Version séquentielle\n");
    matrix_init(m, n, matrix);
    printf("init\n====\n");
    matrix_print(m, n, matrix[0]);
    start = get_current_time();
    chaleur_seq(m, n, matrix, np, td, h);
    end = get_current_time();
    double dt_seq = end - start;
    printf("final\n====\n");
    matrix_print(m, n, matrix[0]);
    printf("Temps d'éxecution parrallèle: %f\n", dt_par);
    printf("Temps d'éxecution séquentiel: %f\n", dt_seq);
    printf("Accélération : %f\n", dt_seq/dt_par);
    printf("Efficacité: %f\n", (dt_seq/dt_par)/cpus);
    printf("|%d|%f|%f|\n", m, dt_seq, dt_par);
  } else {
    double matrix[2][m][n];
    matrix_zero(m, n, matrix);
    chaleur_par(m, n, matrix, np, td, h);
  }

  err = MPI_Finalize();
  return err;
}

void chaleur_seq(int m, int n, double matrix[2][m][n], int np, double td, double h) {
  int i, j, k;
  int current = 0;

  for(k = 1; k < np; k++) {
    for(i = 1; i < m - 1; i++) {
      for(j = 1; j < n - 1; j++) {
        usleep(SLEEP_TIME);

        matrix[current][i][j] = (1.0 - 4*td / h*h) * matrix[1-current][i][j] +
          (td/h*h) * (matrix[1-current][i - 1][j] +
                  matrix[1-current][i + 1][j] +
                  matrix[1-current][i][j - 1] +
                  matrix[1-current][i][j + 1]);
      }
    }

    current = k % 2;
  }
}

void chaleur_par(int m, int n, double matrix[2][m][n], int np, double td, double h) {
  int processors, rank, i, j, k, slice_start, slice_end;
  int workers, columns, remaining, x_offset;
  int current = 0;

  MPI_Status status;
  MPI_Comm_size(MPI_COMM_WORLD, &processors);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if(rank == MASTER_WORKER) {
    workers   = processors - 1;
    columns   = m / workers;
    remaining = m % workers;
    x_offset  = 0;

    DEBUG_PRINT("workers=%d columns=%d remaining=%d\n", workers, columns, remaining);
    for(i = 1; i <= workers; i++) {
      struct Message msg;
      msg.x_offset = x_offset;
      msg.y_offset = (i <= remaining) ? columns + 1 : columns;
      msg.left     = (i == 1 ? -1 : i - 1);
      msg.right    = (i == workers ? -1 : i + 1);

      MPI_Send(&msg, 1, mpi_message_type, i, START_TAG, MPI_COMM_WORLD);
      MPI_Send(&matrix[current][msg.x_offset][0], msg.y_offset * n, MPI_DOUBLE, i, START_TAG, MPI_COMM_WORLD);

      DEBUG_PRINT(
        "send => destination=%d, x_offset=%d, y_offset=%d, left=%d, right=%d\n",
        i, msg.x_offset, msg.y_offset, msg.left, msg.right
      );

      x_offset += msg.y_offset;
    }

    for(i = 1; i <= workers; i++) {
      struct Message msg;
      MPI_Recv(&msg, 1, mpi_message_type, i, DONE_TAG, MPI_COMM_WORLD, &status);
      MPI_Recv(&matrix[current][msg.x_offset][0], msg.y_offset * n, MPI_DOUBLE, i, DONE_TAG, MPI_COMM_WORLD, &status);
    }
  } else {
    current = 0;
    struct Message msg;
    MPI_Recv(&msg, 1, mpi_message_type, MASTER_WORKER, START_TAG, MPI_COMM_WORLD, &status);
    MPI_Recv(&matrix[current][msg.x_offset][0], msg.y_offset * n, MPI_DOUBLE, MASTER_WORKER, START_TAG, MPI_COMM_WORLD, &status);

    // On découpe la matrice en rangées en fonction du nombre de workers
    slice_start = msg.x_offset;
    slice_end   = msg.x_offset + msg.y_offset - 1;
    if(msg.x_offset == 0) slice_start = 1;
    if(msg.x_offset + msg.y_offset == m) slice_end -= 1;

    for(k = 1; k < np; k++) {
      // left neighbor
      if(msg.left != -1) {
        MPI_Send(
          &matrix[current][msg.x_offset][0],
          n, MPI_DOUBLE, msg.left, RIGHT_TAG, MPI_COMM_WORLD
        );
        MPI_Recv(
          &matrix[current][msg.x_offset - 1][0],
          n, MPI_DOUBLE, msg.left, LEFT_TAG, MPI_COMM_WORLD, &status
        );
      }

      // right neighbor
      if(msg.right != -1) {
        MPI_Send(
          &matrix[current][msg.x_offset + msg.y_offset - 1][0],
          n, MPI_DOUBLE, msg.right, LEFT_TAG, MPI_COMM_WORLD
        );
        MPI_Recv(
          &matrix[current][msg.x_offset + msg.y_offset][0],
          n, MPI_DOUBLE, msg.right, RIGHT_TAG, MPI_COMM_WORLD, &status
        );
      }

      // we are now able to set the value
      for(i = slice_start; i <= slice_end; i++) {
        for(j = 1; j < n - 1; j++) {
          usleep(SLEEP_TIME);
          matrix[1-current][i][j] = (1.0 - 4*td / h*h) * matrix[current][i][j] +
            (td/h*h) * (matrix[current][i - 1][j] +
                    matrix[current][i + 1][j] +
                    matrix[current][i][j - 1] +
                    matrix[current][i][j + 1]);
        }
      }

      // flip-flop
      current = 1 - current;
    }

    // send back the message to the master
    MPI_Send(&msg, 1, mpi_message_type, MASTER_WORKER, DONE_TAG, MPI_COMM_WORLD);
    MPI_Send(&matrix[current][msg.x_offset][0], msg.y_offset * n, MPI_DOUBLE, MASTER_WORKER, DONE_TAG, MPI_COMM_WORLD);
  }
}

void matrix_init(int m, int n, double matrix[2][m][n]) {
  int i, j;

  for(i = 0; i < m; i++) {
    for(j = 0; j < n; j++) {
      matrix[0][i][j] = (double)(i * (m - i - 1)) * (j * (n - j - 1));
      matrix[1][i][j] = matrix[0][i][j];
    }
  }
}

void matrix_zero(int m, int n, double matrix[2][m][n]) {
  int i, j;

  for(i = 0; i < m ; i++){
    for(j = 0; j < n; j++){
      matrix[0][i][j] = 0.0;
      matrix[1][i][j] = matrix[0][i][j];
    }
  }
}

void matrix_print(int m, int n, double matrix[m][n]) {
  int i, j;
  for(i = 0; i < m; i++) {
    for(j = 0; j < n; j++) {
      printf("|%15.2f", matrix[i][j]);
    }
    printf("|\n");
  }
}

double get_current_time() {
  struct timeval tp;
  gettimeofday (&tp, NULL);
  return (double) (tp.tv_sec) + (double) (tp.tv_usec) / 1e6;
}

void types_init() {
  int blocklengths[4] = {1, 1, 1, 1};
  MPI_Datatype types[4] = {MPI_INT, MPI_INT, MPI_INT, MPI_INT};
  MPI_Aint offsets[4];
  offsets[0] = offsetof(struct Message, x_offset);
  offsets[1] = offsetof(struct Message, y_offset);
  offsets[2] = offsetof(struct Message, left);
  offsets[3] = offsetof(struct Message, right);
  MPI_Type_create_struct(4, blocklengths, offsets, types, &mpi_message_type);
  MPI_Type_commit(&mpi_message_type);
}
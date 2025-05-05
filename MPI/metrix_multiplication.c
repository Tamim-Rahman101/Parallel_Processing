#include <stdio.h>    // Include standard input/output library for printf, scanf
#include <stdlib.h>   // Include standard library for rand()
#include <mpi.h>      // Include MPI (Message Passing Interface) library for parallel processing

// Function to print a matrix
void display(int rows, int cols, int matrix[rows][cols]) {
    for(int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%3d ", matrix[i][j]);  // Print each element formatted to width 3
        }
        printf("\n");  // New line after each row
    }
    printf("\n");      // Extra line after the matrix
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);  // Initialize the MPI environment

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get the rank (ID) of the current process
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Get the total number of processes

    // Define default dimensions
    int K = 100, M = 50, N = 50, P = 50;

    // Optional user input for matrix sizes (commented out)
    // if(rank == 0) {
    //     printf("Enter Number of Matrices: ");
    //     scanf("%d", &K);
    //     printf("Enter Number of Rows in Matrix A: ");
    //     scanf("%d", &M);
    //     printf("Enter Number of Columns in Matrix A: ");
    //     scanf("%d", &N);
    //     printf("Enter Number of Columns in Matrix B: ");
    //     scanf("%d", &P);
    // }

    // Broadcast matrix dimensions from root process (rank 0) to all processes
    MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&P, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Check that the number of matrices K is divisible by the number of processes
    if(K % size != 0) {
        if(rank == 0)  // Only root process should print the error
            printf("Number of matrices must be divisible by the number of processes.\n");
        MPI_Finalize(); // Clean up and exit MPI
        return 1;
    }

    // Declare matrices
    int A[K][M][N], B[K][N][P], R[K][M][P];  // A = input matrices, B = input matrices, R = result matrices

    // Initialize matrices A and B only in the root process
    if(rank == 0) {
        for(int k = 0; k < K; k++) {
            for(int i = 0; i < M; i++) {
                for(int j = 0; j < N; j++) {
                    A[k][i][j] = rand() % 100;  // Random numbers between 0 and 99
                }
            }
            for(int i = 0; i < N; i++) {
                for(int j = 0; j < P; j++) {
                    B[k][i][j] = rand() % 100;  // Random numbers between 0 and 99
                }
            }
        }
    }

    // Local buffers to hold the matrices that each process will work on
    int localA[K / size][M][N], localB[K / size][N][P], localR[K / size][M][P];

    // Scatter matrices A and B equally to all processes
    MPI_Scatter(A, (K / size) * M * N, MPI_INT, localA, (K / size) * M * N, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(B, (K / size) * N * P, MPI_INT, localB, (K / size) * N * P, MPI_INT, 0, MPI_COMM_WORLD);

    // Start measuring time before computation
    double startTime = MPI_Wtime();

    // Each process performs matrix multiplication on its own share
    for(int k = 0; k < (K / size); k++) {       // Loop through matrices assigned to this process
        for(int i = 0; i < M; i++) {             // Loop through rows of matrix A
            for(int j = 0; j < P; j++) {         // Loop through columns of matrix B
                localR[k][i][j] = 0;             // Initialize result cell
                for(int l = 0; l < N; l++) {     // Loop through shared dimension
                    localR[k][i][j] += (localA[k][i][l] * localB[k][l][j]) % 100;  // Multiply and accumulate
                }
                localR[k][i][j] %= 100;          // Keep the final value modulo 100
            }
        }
    }

    // End measuring time after computation
    double endTime = MPI_Wtime();

    // Gather all local results back to the root process
    MPI_Gather(localR, (K / size) * M * P, MPI_INT, R, (K / size) * M * P, MPI_INT, 0, MPI_COMM_WORLD);

    // Optional: Uncomment to print all result matrices (only root process)
    // if(rank == 0) {
    //     for(int k = 0; k < K; k++) {
    //         printf("Result Matrix R%d\n", k);
    //         display(M, P, R[k]);  // Print each matrix
    //     }
    // }

    MPI_Barrier(MPI_COMM_WORLD); // Synchronize all processes before printing times

    // Print time taken by this process
    printf("Process %d: Time taken = %f seconds\n", rank, endTime - startTime);

    MPI_Finalize(); // Shut down the MPI environment
    return 0;
}

/*
  C++ implementation of serial matrix multiply using different allocation strategies

  Size of the matrix is given as a commandline argument or user prompt

  The idea is that improvements and implementation changes can be dropped in to different namespaces
  without changing the calling code in main.

  This implementation uses more C++ features rather than C ones, including:
  - <random> for better quality random number generation
  - new/delete for memory allocation instead of rand()
  - 'using' instead of 'typedef'
  - namespaces to separate features, algorithms, and implementations
  - iostreams and file streams for output

  Author: Tim Quelch (t.quelch@qut.edu.au)
*/

#include <chrono>   // For timing
#include <fstream>  // File input/output
#include <iostream> // Console input/output
#include <random>   // Random number generator
#include <thread>   // C++ threads

#include <pthread.h> // POSIX threads (pthreads)

// Change this between int, double, short, float etc.
using Type = double;

// We're using a better RNG than C style rand(). For testing purposes rand() is probably fine, but
// using engines and distributions from <random> will give us better quality random numbers
Type rng() {
    static auto engine = std::mt19937{42}; // Seed rng with constant value (42)

    // Generate numbers between -10 and 10. This is arbitrary and can be changed
    // Will need to change to uniform_real_distribution if Type is floating point
    // static auto distribution = std::uniform_int_distribution<Type>{-10, 10};
    static auto distribution = std::uniform_real_distribution<Type>{-10, 10};

    return distribution(engine); // Generate the random number
}

// This namespace contains functions that work with 'flat' allocations of matrices. These matrices
// are of Type* and require only a single allocation of NxN values.
namespace flat {
    // Helper function to calculate the index in the flat array because we can't use 2D matrix
    // notation. 'constexpr' tells the compiler that this function can be evaluated at compile time
    constexpr unsigned index(unsigned i, unsigned j, unsigned N) { return i * N + j; }

    // Allocate a mx in memory
    Type* allocateMx(unsigned N) { return new Type[N * N]; }

    // Deallocate a mx from memory
    // N parameter is not used with the flat allocation. It is included just so the API is the same
    // as the other version
    void deallocateMx(Type* mx, [[maybe_unused]] unsigned N) { delete[] mx; }

    // Populate a mx with random numbers
    void populateMx(Type* mx, unsigned N) {
        for (unsigned i = 0; i < N; i++) {
            for (unsigned j = 0; j < N; j++) {
                mx[index(i, j, N)] = rng();
            }
        }
    }

    // Transpose a matrix in place. e.g. for all (i, j) combinations, mx[i, j] <- mx[j, i]
    void transposeMx(Type* mx, unsigned N) {
        for (unsigned i = 1; i < N; i++) {
            for (unsigned j = 0; j < i; j++) {
                std::swap(mx[index(i, j, N)], mx[index(j, i, N)]);
            }
        }
    }

    // Multiply the matrices a and b into result c. A * B = C
    void multiplyMx(Type const* a, Type const* b, Type* c, unsigned N) {
        for (unsigned i = 0; i < N; i++) {
            for (unsigned j = 0; j < N; j++) {
                c[index(i, j, N)] = 0;
                for (unsigned k = 0; k < N; k++) {
                    c[index(i, j, N)] += a[index(i, k, N)] * b[index(k, j, N)];
                }
            }
        }
    }

    namespace threaded {
        void multiplyMxChunk(Type const* a,
                             Type const* b,
                             Type* c,
                             unsigned N,
                             unsigned nThreads,
                             unsigned threadId) {
            unsigned chunkSize = (N + nThreads - 1) / nThreads; // Calculate the size of our chunks
            // This is N / nThreads with forced ceil rounding, rather than floor rounding

            // Calculate the start and end indexes
            unsigned startIndex = chunkSize * threadId;
            unsigned finishIndex = std::min(startIndex + chunkSize, N);

            // std::cout << "Starting thread " << threadId << " "
            //          << "(" << std::this_thread::get_id() << "). Index " << startIndex << " to "
            //          << finishIndex << "\n";

            // Compute the matrix multiplication for the specified chunk
            for (unsigned i = startIndex; i < finishIndex; i++) {
                for (unsigned j = 0; j < N; j++) {
                    c[index(i, j, N)] = 0;
                    for (unsigned k = 0; k < N; k++) {
                        c[index(i, j, N)] += a[index(i, k, N)] * b[index(k, j, N)];
                    }
                }
            }
        }

        void multiplyMx(Type const* a, Type const* b, Type* c, unsigned N, unsigned nThreads) {
            auto threads = new std::thread[N]; // Allocate array to hold our thread handles

            // Start each thread with a different threadId (i)
            for (unsigned i = 0; i < nThreads; i++) {
                threads[i] = std::thread{multiplyMxChunk, a, b, c, N, nThreads, i};
            }

            // Wait for all our threads to complete their work
            for (unsigned i = 0; i < nThreads; i++) {
                threads[i].join();
            }

            delete[] threads; // Deallocate our thread array
        }
    } // namespace threaded

    // Output the mx to a binary file
    void outputFile(Type const* mx, std::string filename, unsigned N) {
        std::ofstream file(filename, std::ios::binary);
        file.write((char*)mx, sizeof(Type) * N * N);
    }

    // Compare a mx to a previously written binary file
    void validateAgainstFile(Type const* mx, std::string filename, unsigned N) {
        // Open validation file
        std::ifstream file(filename, std::ios::binary);

        // Read in validatio nmatrix
        auto checkMx = allocateMx(N);
        file.read((char*)checkMx, sizeof(Type) * N * N);

        bool matching = true;
        for (unsigned i = 0; i < N; i++) {
            for (unsigned j = 0; j < N; j++) {
                if (checkMx[index(i, j, N)] != mx[index(i, j, N)]) {
                    matching = false;
                }
            }
        }

        if (!matching) {
            std::cout << "Matrix values do not match validation file!\n";
        }

        deallocateMx(checkMx, N);
    }
} // namespace flat

int main(int argc, char const* argv[]) {
    // Set N from the arguments or user input
    unsigned N = 0;
    if (argc > 1) {
        N = std::atoi(argv[1]);
    } else {
        std::cout << "Enter the dimension of the matrix: ";
        std::cin >> N;
    }

    // So we don't need to prefix everything from here on with flat::
    using namespace flat;

    // Pointers to flat allocation that is N*N elements
    Type* a = allocateMx(N);
    Type* b = allocateMx(N);
    Type* c = allocateMx(N);

    // Populate our input matrices
    populateMx(a, N);
    populateMx(b, N);

    std::cout << "Running serial computation...\n";
    {
        auto start = std::chrono::high_resolution_clock::now(); // Start a timer
        multiplyMx(a, b, c, N);                                 // Execute matrix multiply
        auto end = std::chrono::high_resolution_clock::now();   // Top the timer

        // Calculate the time as floating point in milliseconds
        auto time = std::chrono::duration<double, std::milli>(end - start).count();

        // Print time to output
        std::cout << "N = " << N << ":  " << time << " ms\n";
        outputFile(c, "validation.bin", N);          // Write output to file
        validateAgainstFile(c, "validation.bin", N); // Compare output to validation file
    }

    std::cout << "Running threaded (std::thread) computation...\n";
    for (unsigned i = 1; i <= 256; i *= 2) {
        populateMx(c, N); // Reset output mx to random values to ensure validation works correctly

        auto start = std::chrono::high_resolution_clock::now(); // Start a timer
        threaded::multiplyMx(a, b, c, N, i);                    // Execute matrix multiply
        auto end = std::chrono::high_resolution_clock::now();   // Top the timer

        // Calculate the time as floating point in milliseconds
        auto time = std::chrono::duration<double, std::milli>(end - start).count();

        // Print time to output
        std::cout << "N = " << N << ", nt = " << i << ":  " << time << " ms\n";
        validateAgainstFile(c, "validation.bin", N); // Compare output to validation file
    }

    // Deallocate memory
    // (This isn't strictly necesssary, because this is right before we exit the program when it
    // will be cleaned up by the OS anyway. However, deallocating memory is a good habit to get
    // in to avoid memory leaks)
    deallocateMx(a, N);
    deallocateMx(b, N);
    deallocateMx(c, N);
}

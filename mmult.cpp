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

#include <chrono>     // For timing
#include <fstream>    // File input/output
#include <functional> // for passing functions as parameters
#include <iostream>   // Console input/output
#include <random>     // Random number generator
#include <thread>     // C++ threads

#include <pthread.h> // POSIX threads (pthreads)

#include <omp.h> // OpenMP runtime

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

// Function to time and validate test
double timeAndValidate(std::function<void()> test,
                       std::function<void()> preTest,
                       std::function<void()> postTest,
                       std::function<void()> validate) {
    preTest();
    auto start = std::chrono::high_resolution_clock::now();
    test();
    auto end = std::chrono::high_resolution_clock::now();
    postTest();

    validate();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// This namespace contains functions that work with 'jagged' allocations of matrices. These
// matrices are of Type** and require N+1 memory allocations of size N
namespace jagged {
    // Allocate a mx in memory
    Type** allocateMx(unsigned N) {
        auto mx = new Type*[N];
        for (unsigned i = 0; i < N; i++) {
            mx[i] = new Type[N];
        }
        return mx;
    }

    // Deallocate a mx from memory
    void deallocateMx(Type** mx, unsigned N) {
        for (unsigned i = 0; i < N; i++) {
            delete[] mx[i];
        }
        delete[] mx;
    }

    // Populate a mx with random numbers
    void populateMx(Type** mx, unsigned N) {
        for (unsigned i = 0; i < N; i++) {
            for (unsigned j = 0; j < N; j++) {
                mx[i][j] = rng();
            }
        }
    }

    // Multiply the matrices a and b into result c. A * B = C
    void multiplyMx(Type** a, Type** b, Type** c, unsigned N) {
        for (unsigned i = 0; i < N; i++) {
            for (unsigned j = 0; j < N; j++) {
                c[i][j] = 0;
                for (unsigned k = 0; k < N; k++) {
                    c[i][j] += a[i][k] * b[k][j];
                }
            }
        }
    }

    // Output the mx to a binary file
    void outputFile(Type** mx, std::string filename, unsigned N) {
        std::ofstream file(filename, std::ios::binary);
        for (unsigned i = 0; i < N; i++) {
            file.write((char*)mx[i], sizeof(Type) * N);
        }
    }

    // Compare a mx to a previously written binary file
    void validateAgainstFile(Type** mx, std::string filename, unsigned N) {
        // Open validation file
        std::ifstream file(filename, std::ios::binary);

        // Allocate validation matrix and read in all rows
        auto checkMx = allocateMx(N);
        for (unsigned i = 0; i < N; i++) {
            file.read((char*)checkMx[i], sizeof(Type) * N);
        }

        // Check all values match
        bool matching = true;
        for (unsigned i = 0; i < N; i++) {
            for (unsigned j = 0; j < N; j++) {
                if (checkMx[i][j] != mx[i][j]) {
                    matching = false;
                }
            }
        }

        if (!matching) {
            std::cout << "Matrix values do not match validation file!\n";
        }

        // Deallocate validation matrix
        deallocateMx(checkMx, N);
    }
} // namespace jagged

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

    namespace tiled {
        void multiplyMx(Type const* a, Type const* b, Type* c, unsigned N, unsigned tileSize) {
            for (unsigned tr = 0; tr < N; tr += tileSize) {
                for (unsigned tc = 0; tc < N; tc += tileSize) {
                    for (unsigned i = tr; i < std::min(tr + tileSize, N); i++) {
                        for (unsigned j = tc; j < std::min(tc + tileSize, N); j++) {
                            c[index(i, j, N)] = 0;
                            for (unsigned k = 0; k < N; k++) {
                                c[index(i, j, N)] += a[index(i, k, N)] * b[index(k, j, N)];
                            }
                        }
                    }
                }
            }
        }
    } // namespace tiled

    namespace transposed {
        void multiplyMx(Type const* a, Type const* bt, Type* c, unsigned N) {
            for (unsigned i = 0; i < N; i++) {
                for (unsigned j = 0; j < N; j++) {
                    c[index(i, j, N)] = 0;
                    for (unsigned k = 0; k < N; k++) {
                        c[index(i, j, N)] += a[index(i, k, N)] * bt[index(j, k, N)];
                    }
                }
            }
        }
    } // namespace transposed

    namespace tiled_transposed {
        void multiplyMx(Type const* a, Type const* bt, Type* c, unsigned N, unsigned tileSize) {
            for (unsigned tr = 0; tr < N; tr += tileSize) {
                for (unsigned tc = 0; tc < N; tc += tileSize) {
                    for (unsigned i = tr; i < std::min(tr + tileSize, N); i++) {
                        for (unsigned j = tc; j < std::min(tc + tileSize, N); j++) {
                            c[index(i, j, N)] = 0;
                            for (unsigned k = 0; k < N; k++) {
                                c[index(i, j, N)] += a[index(i, k, N)] * bt[index(j, k, N)];
                            }
                        }
                    }
                }
            }
        }
    } // namespace tiled_transposed

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

    namespace threaded_pthreads {
        struct Args {
            Type const* a;
            Type const* b;
            Type* c;
            unsigned N;
            unsigned nThreads;
            unsigned threadId;
        };

        void* multiplyMxChunk(void* rawArgs) {
            Args args = *(Args*)rawArgs;
            unsigned chunkSize =
                (args.N + args.nThreads - 1) / args.nThreads; // Calculate the size of our chunks
            // This is N / nThreads with forced ceil rounding, rather than floor rounding

            // Calculate the start and end indexes
            unsigned startIndex = chunkSize * args.threadId;
            unsigned finishIndex = std::min(startIndex + chunkSize, args.N);

            // std::cout << "Starting thread " << threadId << " "
            //          << "(" << std::this_thread::get_id() << "). Index " << startIndex << " to "
            //          << finishIndex << "\n";

            // Compute the matrix multiplication for the specified chunk
            for (unsigned i = startIndex; i < finishIndex; i++) {
                for (unsigned j = 0; j < args.N; j++) {
                    args.c[index(i, j, args.N)] = 0;
                    for (unsigned k = 0; k < args.N; k++) {
                        args.c[index(i, j, args.N)] +=
                            args.a[index(i, k, args.N)] * args.b[index(k, j, args.N)];
                    }
                }
            }
            return nullptr;
        }

        void multiplyMx(Type const* a, Type const* b, Type* c, unsigned N, unsigned nThreads) {
            auto args = new Args[nThreads];

            for (unsigned i = 0; i < nThreads; i++) {
                args[i].a = a;
                args[i].b = b;
                args[i].c = c;
                args[i].N = N;
                args[i].nThreads = nThreads;
                args[i].threadId = i;
            }

            auto threads = new pthread_t[N]; // Allocate array to hold our thread handles

            // Start each thread with a different threadId (i)
            for (unsigned i = 0; i < nThreads; i++) {
                pthread_create(&threads[i], nullptr, multiplyMxChunk, &args[i]);
            }

            // Wait for all our threads to complete their work
            for (unsigned i = 0; i < nThreads; i++) {
                pthread_join(threads[i], nullptr);
            }

            delete[] threads; // Deallocate our thread array
        }
    } // namespace threaded_pthreads

    namespace threaded_openmp {
        void multiplyMx(Type const* a, Type const* b, Type* c, unsigned N, unsigned nThreads) {
            omp_set_num_threads(nThreads);
#pragma omp parallel for
            for (unsigned i = 0; i < N; i++) {
                for (unsigned j = 0; j < N; j++) {
                    c[index(i, j, N)] = 0;
                    for (unsigned k = 0; k < N; k++) {
                        c[index(i, j, N)] += a[index(i, k, N)] * b[index(k, j, N)];
                    }
                }
            }
        }
    } // namespace threaded_openmp

    namespace threaded_transposed {
        void multiplyMxChunk(Type const* a,
                             Type const* bt,
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
                        c[index(i, j, N)] += a[index(i, k, N)] * bt[index(j, k, N)];
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
    } // namespace threaded_transposed

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

    std::cout << "Running serial (jagged allocation) computation...\n";
    {
        // Array of arrays. First level points to each row
        Type** a = jagged::allocateMx(N);
        Type** b = jagged::allocateMx(N);
        Type** c = jagged::allocateMx(N);

        // Populate our input matrices
        jagged::populateMx(a, N);
        jagged::populateMx(b, N);

        auto time = timeAndValidate([=] { jagged::multiplyMx(a, b, c, N); }, [] {}, [] {}, [] {});

        // Print time to output
        std::cout << "N = " << N << ":  " << time << " ms\n";

        jagged::deallocateMx(a, N);
        jagged::deallocateMx(b, N);
        jagged::deallocateMx(c, N);
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
        auto time =
            timeAndValidate([=] { multiplyMx(a, b, c, N); },
                            [] {},
                            [=] { outputFile(c, "validation.bin", N); }, // Output validation file
                            [] {});

        std::cout << "N = " << N << ":  " << time << " ms\n";
    }

    std::cout << "Running transposed serial computation...\n";
    {
        auto time = timeAndValidate(
            [=] {
                transposeMx(b, N); // Transpose matrix before computation
                transposed::multiplyMx(a, b, c, N);
            },
            [=] { populateMx(c, N); },  // Reset output with random values
            [=] { transposeMx(b, N); }, // Reset b matrix afterwards
            [=] { validateAgainstFile(c, "validation.bin", N); }); // Validate output

        std::cout << "N = " << N << ":  " << time << " ms\n";
    }

    std::cout << "Running tiled serial computation...\n";
    for (unsigned i = 1; i <= 1024; i *= 2) {
        auto time = timeAndValidate(
            [=] { tiled::multiplyMx(a, b, c, N, i); },
            [=] { populateMx(c, N); }, // Reset output with random values
            [] {},
            [=] { validateAgainstFile(c, "validation.bin", N); }); // Validate output

        std::cout << "N = " << N << ", ts = " << i << ":  " << time << " ms\n";
    }

    std::cout << "Running tiled and transposed serial computation...\n";
    for (unsigned i = 1; i <= 1024; i *= 2) {
        auto time = timeAndValidate(
            [=] {
                transposeMx(b, N);
                tiled_transposed::multiplyMx(a, b, c, N, i);
            },
            [=] { populateMx(c, N); }, // Reset output with random values
            [=] { transposeMx(b, N); },
            [=] { validateAgainstFile(c, "validation.bin", N); }); // Validate output

        std::cout << "N = " << N << ", ts = " << i << ":  " << time << " ms\n";
    }

    std::cout << "Running threaded (std::thread) computation...\n";
    for (unsigned i = 1; i <= 256; i *= 2) {
        auto time = timeAndValidate(
            [=] { threaded::multiplyMx(a, b, c, N, i); },
            [=] { populateMx(c, N); }, // Reset output with random values
            [] {},
            [=] { validateAgainstFile(c, "validation.bin", N); }); // Validate output

        std::cout << "N = " << N << ", nt = " << i << ":  " << time << " ms\n";
    }

    std::cout << "Running threaded (pthreads) computation...\n";
    for (unsigned i = 1; i <= 256; i *= 2) {
        auto time = timeAndValidate(
            [=] { threaded_pthreads::multiplyMx(a, b, c, N, i); },
            [=] { populateMx(c, N); }, // Reset output with random values
            [] {},
            [=] { validateAgainstFile(c, "validation.bin", N); }); // Validate output

        std::cout << "N = " << N << ", nt = " << i << ":  " << time << " ms\n";
    }

    std::cout << "Running threaded (OpenMP) computation...\n";
    for (unsigned i = 1; i <= 256; i *= 2) {
        auto time = timeAndValidate(
            [=] { threaded_openmp::multiplyMx(a, b, c, N, i); },
            [=] { populateMx(c, N); }, // Reset output with random values
            [] {},
            [=] { validateAgainstFile(c, "validation.bin", N); }); // Validate output

        std::cout << "N = " << N << ", nt = " << i << ":  " << time << " ms\n";
    }

    std::cout << "Running threaded transposed computation...\n";
    for (unsigned i = 1; i <= 256; i *= 2) {
        auto time = timeAndValidate(
            [=] {
                transposeMx(b, N);
                threaded_transposed::multiplyMx(a, b, c, N, i);
            },
            [=] { populateMx(c, N); }, // Reset output with random values
            [=] { transposeMx(b, N); },
            [=] { validateAgainstFile(c, "validation.bin", N); }); // Validate output

        std::cout << "N = " << N << ", nt = " << i << ":  " << time << " ms\n";
    }

    // Deallocate memory
    // (This isn't strictly necesssary, because this is right before we exit the program when it
    // will be cleaned up by the OS anyway. However, deallocating memory is a good habit to get
    // in to avoid memory leaks)
    deallocateMx(a, N);
    deallocateMx(b, N);
    deallocateMx(c, N);
}

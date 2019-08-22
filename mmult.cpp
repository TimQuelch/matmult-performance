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

// Helper function to calculate the index in the flat array because we can't use 2D matrix
// notation. 'constexpr' tells the compiler that this function can be evaluated at compile time
constexpr unsigned index(unsigned i, unsigned j, unsigned N) { return i * N + j; }

// Allocate a mx in memory
Type* allocateMx(unsigned N) { return new Type[N * N]; }

// Deallocate a mx from memory
void deallocateMx(Type* mx) { delete[] mx; }

// Populate a mx with random numbers
void populateMx(Type* mx, unsigned N) {
    for (unsigned i = 0; i < N; i++) {
        for (unsigned j = 0; j < N; j++) {
            mx[index(i, j, N)] = rng();
        }
    }
}

// Multiply the matrices a and b into result c. A * B = C
void multiplyMx(Type* a, Type* b, Type* c, unsigned N) {
    for (unsigned i = 0; i < N; i++) {
        for (unsigned j = 0; j < N; j++) {
            c[index(i, j, N)] = 0;
            for (unsigned k = 0; k < N; k++) {
                c[index(i, j, N)] += a[index(i, k, N)] * b[index(k, j, N)];
            }
        }
    }
}

// Output the mx to a binary file
void outputFile(Type* mx, std::string filename, unsigned N) {
    std::ofstream file(filename, std::ios::binary);
    file.write((char*)mx, sizeof(Type) * N * N);
}

// Compare a mx to a previously written binary file
void validateAgainstFile(Type* mx, std::string filename, unsigned N) {
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

    deallocateMx(checkMx);
}

int main(int argc, char const* argv[]) {
    // Set N from the arguments or user input
    unsigned N = 0;
    if (argc > 1) {
        N = std::atoi(argv[1]);
    } else {
        std::cout << "Enter the dimension of the matrix: ";
        std::cin >> N;
    }

    // Pointers to flat allocation that is N*N elements
    Type* a = allocateMx(N);
    Type* b = allocateMx(N);
    Type* c = allocateMx(N);

    // Populate our input matrices
    populateMx(a, N);
    populateMx(b, N);

    std::cout << "Running serial computation...\n";
    auto start = std::chrono::high_resolution_clock::now(); // Start a timer
    multiplyMx(a, b, c, N);                                 // Execute matrix multiply
    auto end = std::chrono::high_resolution_clock::now();   // Top the timer

    // Calculate the time as floating point in milliseconds
    auto time = std::chrono::duration<double, std::milli>(end - start).count();

    // Print time to output
    std::cout << "N = " << N << ":  " << time << " ms\n";
    outputFile(c, "validation.bin", N);          // Write output to file
    validateAgainstFile(c, "validation.bin", N); // Compare output to validation file

    // Deallocate memory
    // (This isn't strictly necesssary, because this is right before we exit the program when it
    // will be cleaned up by the OS anyway. However, deallocating memory is a good habit to get
    // in to avoid memory leaks)
    deallocateMx(a);
    deallocateMx(b);
    deallocateMx(c);
}

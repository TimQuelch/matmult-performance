#include <CL/cl.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>

// Check opencl error code
// These codes are generated in two ways
//    Some functions return the values (clGetPlatformIDs, clEnqueueNDRangeKernel, etc.)
//    Some functions accept and address to a cl_int as a parameter (clCreateContext, etc.)
void check(cl_int returnCode) {
    if (returnCode != CL_SUCCESS) {
        printf("OpenCL error: %d\n", returnCode);
        exit(1);
    }
}

// Helper function to get a string version of a cl_device_type
const char* deviceType(cl_device_type type) {
    switch (type) {
    case CL_DEVICE_TYPE_CPU:
        return "CPU";
    case CL_DEVICE_TYPE_GPU:
        return "GPU";
    case CL_DEVICE_TYPE_ACCELERATOR:
        return "ACCELERATOR";
    default:
        return "OTHER";
    }
}

// Print out a message and the time since the last call
auto last = std::chrono::steady_clock::now();
void TimeSinceLast(std::string const& message) {
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration<float>{now - last}.count();
    printf("%f s: %s\n", elapsed, message.data());
    last = std::chrono::steady_clock::now();
}

// The OpenCL source code for the matrix multiply kernel
const char* matMultSource =
    R"STR(
kernel void matMult(global double* A, global double* B, global double* C) {
    const int i = get_global_id(0);
    const int j = get_global_id(1);
    const int n = get_global_size(0);
    double sum = 0;
    for (int k = 0; k < n; k++) {
        sum += A[i * n + k] * B[j * n + k];
    }
    C[i * n + j] = sum;
})STR";

int main() {
    TimeSinceLast("Starting!");

    // Get the platform IDs
    const int maxPlatforms = 10;
    cl_platform_id platforms[maxPlatforms];
    cl_uint nPlatforms;
    check(clGetPlatformIDs(maxPlatforms, platforms, &nPlatforms));

    // For every found platform, print the platform information
    for (int i = 0; i < nPlatforms; i++) {
        const int maxStringLength = 128;

        // Get the name of the platform
        char platformName[maxStringLength];
        check(clGetPlatformInfo(
            platforms[i], CL_PLATFORM_NAME, maxStringLength, platformName, nullptr));

        // Get the version of OpenCL supported by the platform
        char clVersion[maxStringLength];
        check(clGetPlatformInfo(
            platforms[i], CL_PLATFORM_VERSION, maxStringLength, clVersion, nullptr));

        // Print this info
        printf("Platform: %d: %p %s (%s)\n", i, platforms[i], platformName, clVersion);

        // Find all the devices on this platform (usually only 1)
        const int maxDevices = 10;
        cl_device_id devices[maxDevices];
        cl_uint nDevices;
        check(clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, maxDevices, devices, &nDevices));

        // For every found device, print the device information
        for (int j = 0; j < nDevices; j++) {
            // Get the device name
            char deviceName[maxStringLength];
            check(
                clGetDeviceInfo(devices[j], CL_DEVICE_NAME, maxStringLength, deviceName, nullptr));

            // Get the number of compute units
            cl_uint computeUnits;
            check(clGetDeviceInfo(devices[j],
                                  CL_DEVICE_MAX_COMPUTE_UNITS,
                                  sizeof(computeUnits),
                                  &computeUnits,
                                  nullptr));

            // Get the amount of global memory
            cl_ulong globalMemory;
            check(clGetDeviceInfo(devices[j],
                                  CL_DEVICE_GLOBAL_MEM_SIZE,
                                  sizeof(globalMemory),
                                  &globalMemory,
                                  nullptr));

            // Get the amount of local memory
            cl_ulong localMemory;
            check(clGetDeviceInfo(
                devices[j], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(localMemory), &localMemory, nullptr));

            // Get the device type
            cl_device_type type;
            check(clGetDeviceInfo(devices[j], CL_DEVICE_TYPE, sizeof(type), &type, nullptr));

            // Print device info
            printf("    Device: %d: %p %s\n", j, devices[j], deviceName);
            printf("            %d: Type: %s, Compute units: %d, global mem: %lu, local mem: %lu\n",
                   j,
                   deviceType(type),
                   computeUnits,
                   globalMemory,
                   localMemory);
        }
    }

    TimeSinceLast("Printing all devices");

    // Chose a platform and device, and requery it
    const int chosenPlatform = 0;
    const int chosenDevice = 0;
    cl_device_id device;
    {
        const int maxDevices = 10;
        cl_device_id devices[maxDevices];
        cl_uint nDevices;
        check(clGetDeviceIDs(
            platforms[chosenPlatform], CL_DEVICE_TYPE_ALL, maxDevices, devices, &nDevices));
        device = devices[chosenDevice];
    }

    cl_int error; // The error variable that will be passed as a parameter

    // Create the context. This is the environment where buffers and kernels are created
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &error);
    check(error);
    printf("context: %p\n", context);

    // Create a command queue
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &error);
    check(error);
    printf("queue:   %p\n", queue);

    TimeSinceLast("Creating context and queues");

    // Create a program from source
    cl_program program = clCreateProgramWithSource(context, 1, &matMultSource, nullptr, &error);
    check(error);
    printf("program: %p\n", program);

    // Compile the program for the specified device. Programs can have multiple kernels (functions)
    check(clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr));
    printf("Program built\n");

    // Create a runnable kernel from the program
    cl_kernel kernel = clCreateKernel(program, "matMult", &error);
    printf("kernel: %p\n", kernel);

    TimeSinceLast("Compiling kernel");

    // Initialise our matrices as usual
    constexpr auto N = 2048;

    auto A = new double[N * N];
    auto B = new double[N * N];
    auto C = new double[N * N];

    TimeSinceLast("Allocate host buffers");

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = rand() % 100;
            B[i * N + j] = rand() % 100;
        }
    }

    TimeSinceLast("Initialise host buffers");

    // Allocate memory on the device for our matrices
    cl_mem deviceA =
        clCreateBuffer(context, CL_MEM_READ_ONLY, N * N * sizeof(double), nullptr, &error);
    check(error);
    cl_mem deviceB =
        clCreateBuffer(context, CL_MEM_READ_ONLY, N * N * sizeof(double), nullptr, &error);
    check(error);
    cl_mem deviceC =
        clCreateBuffer(context, CL_MEM_WRITE_ONLY, N * N * sizeof(double), nullptr, &error);
    check(error);

    TimeSinceLast("Allocate device buffers");

    // Write our matrices to the device buffers
    check(clEnqueueWriteBuffer(
        queue, deviceA, CL_TRUE, 0, N * N * sizeof(double), A, 0, nullptr, nullptr));
    check(clEnqueueWriteBuffer(
        queue, deviceB, CL_TRUE, 0, N * N * sizeof(double), B, 0, nullptr, nullptr));

    TimeSinceLast("Write to device bufers");

    // Set the arguments of our kernel to be the pointers to our buffers
    check(clSetKernelArg(kernel, 0, sizeof(cl_mem), &deviceA));
    check(clSetKernelArg(kernel, 1, sizeof(cl_mem), &deviceB));
    check(clSetKernelArg(kernel, 2, sizeof(cl_mem), &deviceC));

    TimeSinceLast("Setting kernel arguments");

    // Set up the dimensions that we need to range our kernel over
    size_t globalDims[] = {N, N}; // Global dims are the whole range
    size_t localDims[] = {8, 8};  // Local dims are the work-groups that we split our range into

    // Queue up our kernel and wait for it to finish
    cl_event kernelEvent;
    check(clEnqueueNDRangeKernel(
        queue, kernel, 2, nullptr, globalDims, localDims, 0, nullptr, &kernelEvent));
    clWaitForEvents(1, &kernelEvent);

    TimeSinceLast("Running matrix multiply");

    // Read the result from the device back to our local matrix
    check(clEnqueueReadBuffer(
        queue, deviceC, CL_TRUE, 0, N * N * sizeof(double), C, 0, nullptr, nullptr));

    TimeSinceLast("Reading from device output buffer");

    TimeSinceLast("Finished!");
}

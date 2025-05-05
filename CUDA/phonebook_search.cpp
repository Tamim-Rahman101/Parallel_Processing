// %%writefile phonebook_search.cu      // uncomment it in google colab

/*
    How to run in google colab:
        !nvcc -arch=sm_75 phonebook_search.cu -o phonebook_search
        !time ./phonebook_search MUMU 1 > output.txt
                search = "MUMU", thread = 1, output in output.txt file
*/


#include <bits/stdc++.h>  // Standard C++ libraries
using namespace std;
#include <cuda.h>         // CUDA library for GPU programming

// Define a structure to hold contact information
struct Contact {
    char name[65];
    char phone_number[65];
};

// Function to read a quoted string (e.g., "John Doe") from the file
string readQuotedString(ifstream& file) {
    string result;
    char ch;
    bool insideQuotes = false;

    while (file.get(ch)) {
        if (ch == '\"') {
            if (insideQuotes) break;  // End reading when closing quote is found
            insideQuotes = true;       // Start reading when opening quote is found
        } else {
            if (insideQuotes) {
                result.push_back(ch);  // Add characters inside quotes
            }
        }
    }
    return result;
}

// CUDA device function to check if `pattern` is a substring of `text`
__device__ bool containsSubstring(char* text, char* pattern) {
    for (int i = 0; text[i] != '\0'; i++) {
        bool match = true;
        for (int j = 0; pattern[j] != '\0'; j++) {
            if (text[i + j] != pattern[j]) {
                match = false;
                break;
            }
        }
        if (match) return true;
    }
    return false;
}

// CUDA kernel function: Each thread checks one contact
__global__ void searchContacts(Contact* d_phoneBook, char* d_pattern, int offset) {
    int idx = threadIdx.x + offset;  // Calculate the global index for the contact
    if (containsSubstring(d_phoneBook[idx].name, d_pattern)) {
        printf("%s %s\n", d_phoneBook[idx].name, d_phoneBook[idx].phone_number);
    }
}

int main(int argc, char* argv[]) {
    // Check if required command-line arguments are passed
    if (argc < 3) {
        cout << "Usage: ./phonebook_search <search_name> <max_threads_per_batch>\n";
        return 1;
    }

    // Maximum number of threads to launch per kernel call
    int maxThreadsPerBatch = atoi(argv[2]);

    // Open the file containing the phonebook entries
    ifstream inputFile("one.txt");

    vector<Contact> phoneBook;  // Vector to store contacts

    int contactCount = 0;  // Counter to limit total contacts read

    // Read the phonebook entries from the file
    while (inputFile.peek() != EOF) {
        if (contactCount > 10000) break; // Safety limit on number of contacts
        contactCount++;

        string name = readQuotedString(inputFile);
        string phoneNumber = readQuotedString(inputFile);

        Contact contact;
        strcpy(contact.name, name.c_str());
        strcpy(contact.phone_number, phoneNumber.c_str());

        phoneBook.push_back(contact);
    }

    // Get the search pattern (name to search)
    string searchName = argv[1];
    char pattern[65];
    strcpy(pattern, searchName.c_str());

    // Allocate memory on device (GPU) for search pattern
    char* d_pattern;
    cudaMalloc(&d_pattern, 65);
    cudaMemcpy(d_pattern, pattern, 65, cudaMemcpyHostToDevice);

    // Allocate memory on device for the phonebook array
    int n = phoneBook.size();
    Contact* d_phoneBook;
    cudaMalloc(&d_phoneBook, n * sizeof(Contact));
    cudaMemcpy(d_phoneBook, phoneBook.data(), n * sizeof(Contact), cudaMemcpyHostToDevice);

    // Process the contacts in batches to respect thread limit
    int remainingContacts = n;
    int offset = 0;
    while (remainingContacts > 0) {
        int batchSize = min(maxThreadsPerBatch, remainingContacts);

        // Launch CUDA kernel: 1 block, batchSize threads
        searchContacts<<<1, batchSize>>>(d_phoneBook, d_pattern, offset);
        cudaDeviceSynchronize();  // Wait for the GPU to finish

        remainingContacts -= batchSize;  // Update remaining contacts
        offset += batchSize;             // Move offset for next batch
    }

    // Free device memory
    cudaFree(d_pattern);
    cudaFree(d_phoneBook);

    return 0;
}
// %%writefile phonebook_search.cpp    // Uncomment it for google coleb

/*
    How to run in local pc:
    mpic++ -o search phonebook_search.cpp
    mpirun -np 4 ./search phonebook1.txt Bob
    
    How to run in google coleb:
    !mpic++ -o search phonebook_search.cpp
    !mpirun --allow-run-as-root -np 1 ./search phonebook1.txt Bob
*/

#include <bits/stdc++.h>
#include <mpi.h>
using namespace std;

struct Contact {
    string name;
    string phone;
};


void read_phonebook(const vector<string> &files, vector<Contact> &contacts) {
    for (const string &file : files) {      // Loop through each file
        ifstream f(file);                   // Open file for reading
        string line;                        // variable to store readed line as string from the file
        while (getline(f, line)) {          // Read each line from the file (f) and store in line
            if (line.empty()) continue;     // Skip empty lines
            int comma = line.find(",");     // Find comma that seperate name and phone
            if (comma == string::npos) continue;    // Skip if comma not found
            contacts.push_back({                    // Structure of contact has two fields name, phone
                line.substr(1, comma - 2),          // therefore we extract name
                line.substr(comma + 2, line.size() - comma - 3)});  // and phone number from each line of data
        }
    }
}


// Convert a portion of contacts (from index 'start' to 'end') into a string
string vector_to_string(const vector<Contact> &contacts, int start, int end) {
    string result;
    for (int i = start; i < min((int)contacts.size(), end); i++) {
        result += contacts[i].name + "," + contacts[i].phone + "\n";    // Append contact in format: name,phone\n
    }
    return result;      // return string that combines all the contact info into one string
}


// Send a string from the current process to a specific receiver process
void send_string(const string &text, int receiver) {
    int len = text.size() + 1;      // Length of string including null terminator
    MPI_Send(&len, 1, MPI_INT, receiver, 1, MPI_COMM_WORLD);    // Send the length first
    MPI_Send(text.c_str(), len, MPI_CHAR, receiver, 1, MPI_COMM_WORLD);     // Then send the actual characters
}


// Check if contact name contains the search term
string check(const Contact &c, const string &search) {
    if (c.name.find(search) != string::npos) {
        return c.name + " " + c.phone + "\n";   // Return contact info as a string
    }
    return "";      // Otherwise return empty string
}


// Receive a string from a specific sender process
string receive_string(int sender) {
    int len;    // Variable to store incoming string length
    MPI_Recv(&len, 1, MPI_INT, sender, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);   // Receive the length first
    char *buf = new char[len];      // Dynamically allocate memory for the incoming characters
    MPI_Recv(buf, len, MPI_CHAR, sender, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // Receive the actual characters
    string res(buf);    // Convert char array to C++ string
    delete[] buf;       // Free the allocated memory
    return res;         // Return the received string
}


// Convert a formatted string back into a vector of Contact structure
vector<Contact> string_to_contacts(const string &text) {
    vector<Contact> contacts;   // Store parsed contacts
    istringstream iss(text);    // Create stream (iss) for reading lines
    string line;                // Variable to store lines
    while (getline(iss, line)) {    // Read each line from the stream (iss) and store in line
        if (line.empty()) continue;     // Skip empty lines
        int comma = line.find(",");     // Find position of comma
        if (comma == string::npos) continue;    // Skip if comma not found
        contacts.push_back({            // Structure of contact has two fields name, phone
            line.substr(0, comma),      // therefore we extract name
            line.substr(comma + 1)});   // and phone number from each line of data
    }
    return contacts;
}




int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);     // Initialize MPI with argc = num of process, argv = arg vector
    int rank, size;     // rank = process id
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);   // assign process id (rank) to each process (stored locally)
    MPI_Comm_size(MPI_COMM_WORLD, &size);   // assign communication world size (num of process) to each process (stored locally)

    if (argc < 3) {         // Check if enough arguments are provided
        if (rank == 0)      // Only master process prints usage instructions
            cerr << "Usage: mpirun -n <procs> " << argv[0] << " <file>... <search>\n";
        MPI_Finalize();     // Clean up MPI
        return 1;
    }

    string search_term = argv[argc - 1];    // Last argument is the search term
    double start, end;      // Variables to track timing

    if (rank == 0) {    // Master process
        vector<string> files(argv + 1, argv + argc - 1);    // might exist multiple file to work on
        vector<Contact> contacts;           // Vector to store all contacts (Contact has name, phone)
        read_phonebook(files, contacts);    // Read contacts from the each file 
        int total = contacts.size();        // Total number of contacts
        int chunk = (total + size - 1) / size;  // Calculate chunk size for each process on which they will work

        for (int i = 1; i < size; i++) {    // Distribute chunk to other processes
            string text = vector_to_string(contacts, i * chunk, (i + 1) * chunk);   // converts required contacts info into a string in format: name,phone\n,name,phone\n
            send_string(text, i);       // send the string containing contact info to other processes
        }

        // Master process starts searching
        start = MPI_Wtime();    // Start timer
        string result;
        for (int i = 0; i < min(chunk, total); i++) {
            string match = check(contacts[i], search_term);     // Search for term in name
            if (!match.empty())
                result += match;        // Append matching results
        }
        end = MPI_Wtime();

        for (int i = 1; i < size; i++) {        // Receive results from all worker processes
            string recv = receive_string(i);    // Receive result string
            if (!recv.empty()) 
                result += recv;         // Append to final result
        }
        

        ofstream out("output.txt");     // Save all results to a file
        out << result;
        out.close();
        printf("Process %d took %f seconds.\n", rank, end - start);     // Print processing time

    } 
    else {      // Other processes
        string recv_text = receive_string(0);   // Receive contact chunk from master
        vector<Contact> contacts = string_to_contacts(recv_text);   // Convert string back to contacts
        
        start = MPI_Wtime();
        string result;
        for (auto &c : contacts) {
            string match = check(c, search_term);   // Search for term in name
            if (!match.empty())
                result += match;    // Append matching results
        }
        end = MPI_Wtime();
        
        send_string(result, 0);     // Send results back to master
        printf("Process %d took %f seconds.\n", rank, end - start);
    }

    MPI_Finalize();     // Shut down MPI
    return 0;
}
#include <iostream>
#include <vector>
#include <queue>
#include <stack>
#include <omp.h>
#include <chrono>
#include <cstdlib>

using namespace std;
using namespace std::chrono;

class Graph {
    int V;
    vector<vector<int>> adj;

public:
    Graph(int V) {
        this->V = V;
        adj.resize(V);
    }

    void addEdge(int v, int w) {
        adj[v].push_back(w);
        adj[w].push_back(v);  
    }

    void serialBFS(int start);
    void parallelBFS(int start);
    void serialDFS(int start);
    void parallelDFS(int start);
    void measurePerformance(int start);
};

void Graph::serialBFS(int start) {
    vector<bool> visited(V, false);
    queue<int> q;
    visited[start] = true;
    q.push(start);

    while (!q.empty()) {
        int node = q.front();
        q.pop();
        for (int neighbor : adj[node]) {
            if (!visited[neighbor]) {
                visited[neighbor] = true;
                q.push(neighbor);
            }
        }
    }
}

void Graph::parallelBFS(int start) {
    vector<bool> visited(V, false);
    queue<int> q;
    visited[start] = true;
    q.push(start);

    while (!q.empty()) {
        int qSize = q.size();

        #pragma omp parallel for
        for (int i = 0; i < qSize; i++) {
            int node;
            #pragma omp critical
            {
                node = q.front();
                q.pop();
            }

            for (int neighbor : adj[node]) {
                #pragma omp critical
                {
                    if (!visited[neighbor]) {
                        visited[neighbor] = true;
                        q.push(neighbor);
                    }
                }
            }
        }
    }
}

void Graph::serialDFS(int start) {
    vector<bool> visited(V, false);
    stack<int> s;
    s.push(start);

    while (!s.empty()) {
        int node = s.top();
        s.pop();

        if (!visited[node]) {
            visited[node] = true;
        }

        for (int neighbor : adj[node]) {
            if (!visited[neighbor]) {
                s.push(neighbor);
            }
        }
    }
}

void Graph::parallelDFS(int start) {
    vector<bool> visited(V, false);
    stack<int> s;
    s.push(start);

    while (!s.empty()) {
        int node;
        #pragma omp critical
        {
            node = s.top();
            s.pop();
        }

        if (!visited[node]) {
            visited[node] = true;
        }

        #pragma omp parallel for
        for (int neighbor : adj[node]) {
            #pragma omp critical
            {
                if (!visited[neighbor]) {
                    s.push(neighbor);
                }
            }
        }
    }
}

void Graph::measurePerformance(int start) {
    int runs = 10;
    long long bfsSerialTime = 0, bfsParallelTime = 0, dfsSerialTime = 0, dfsParallelTime = 0;

    for (int i = 0; i < runs; i++) {
        auto startTime = high_resolution_clock::now();
        serialBFS(start);
        auto endTime = high_resolution_clock::now();
        bfsSerialTime += duration_cast<microseconds>(endTime - startTime).count();
    }

    for (int i = 0; i < runs; i++) {
        auto startTime = high_resolution_clock::now();
        parallelBFS(start);
        auto endTime = high_resolution_clock::now();
        bfsParallelTime += duration_cast<microseconds>(endTime - startTime).count();
    }

    for (int i = 0; i < runs; i++) {
        auto startTime = high_resolution_clock::now();
        serialDFS(start);
        auto endTime = high_resolution_clock::now();
        dfsSerialTime += duration_cast<microseconds>(endTime - startTime).count();
    }

    for (int i = 0; i < runs; i++) {
        auto startTime = high_resolution_clock::now();
        parallelDFS(start);
        auto endTime = high_resolution_clock::now();
        dfsParallelTime += duration_cast<microseconds>(endTime - startTime).count();
    }

    cout << "\nPerformance Measurement:\n";
    cout << "Serial BFS Time: " << bfsSerialTime / runs << " µs\n";
    cout << "Parallel BFS Time: " << bfsParallelTime / runs << " µs\n";
    cout << "Speedup (BFS): " << (double)bfsSerialTime / bfsParallelTime << "\n";

    cout << "Serial DFS Time: " << dfsSerialTime / runs << " µs\n";
    cout << "Parallel DFS Time: " << dfsParallelTime / runs << " µs\n";
    cout << "Speedup (DFS): " << (double)dfsSerialTime / dfsParallelTime << "\n";
}

int main() {
    int N = 100;  
    Graph g(N);
    
    cout<<"N"<<N<<endl;

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < 5; j++) {
            int neighbor = rand() % N;
            if (neighbor != i) {
                g.addEdge(i, neighbor);
            }
        }
    }

    g.measurePerformance(0);

    return 0;
}
////g++ -fopenmp filename.cpp -o filename
//./a.out
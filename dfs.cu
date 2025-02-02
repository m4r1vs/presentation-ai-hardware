#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>

#define MAX_NODES 25000

// Graph structure
typedef struct {
  int numNodes;
  int **adjMatrix;
} Graph;

// Stack for iterative DFS
typedef struct {
  int items[MAX_NODES];
  int top;
} Stack;

// Thread arguments structure
typedef struct {
  Graph *graph;
  int node;
  int *visited;
} ThreadArgs;

//
// Function prototypes
Graph *createGraph(int numNodes);
void addEdge(Graph *graph, int src, int dest);
void freeGraph(Graph *graph);
void *dfsRecursiveThread(void *args);
void dfsIterative(Graph *graph, int startNode);
void benchmarkDFS(Graph *graph);

// Stack functions
void push(Stack *stack, int value);
int pop(Stack *stack);
int isEmpty(Stack *stack);

// CUDA kernel for parallel DFS
__global__ void dfs_kernel(int *adjMatrix, int numNodes, int *stack, int *top, int *visited) {
    int current_top = atomicAdd(top, -1); // Pop operation
    if (current_top < 0) {
        return; // Stack is empty
    }

    int node = stack[current_top];

    // Attempt to mark the node as visited
    if (atomicCAS(&visited[node], 0, 1) == 0) {
        // Iterate through all possible neighbors
        for (int neighbor = 0; neighbor < numNodes; neighbor++) {
            if (adjMatrix[node * numNodes + neighbor]) {
                // Try to mark the neighbor as visited
                if (atomicCAS(&visited[neighbor], 0, 1) == 0) {
                    // Push the neighbor onto the stack
                    int push_pos = atomicAdd(top, 1) + 1;
                    if (push_pos < MAX_NODES) {
                        stack[push_pos] = neighbor;
                    } else {
                        // Stack overflow handling (not expected in this benchmark)
                        atomicAdd(top, -1);
                    }
                }
            }
        }
    }
}

int main() {
  int numNodes = MAX_NODES; // Adjust for benchmarking
  Graph *graph = createGraph(numNodes);

  // Creating a dense graph for stress testing
  for (int i = 0; i < numNodes - 1; i++) {
    addEdge(graph, i, i + 1);
  }
  addEdge(graph, numNodes - 1, 0); // Make it cyclic

  benchmarkDFS(graph);
  freeGraph(graph);
  return 0;
}

Graph *createGraph(int numNodes) {
  Graph *graph = (Graph *)malloc(sizeof(Graph));
  graph->numNodes = numNodes;
  graph->adjMatrix = (int **)malloc(numNodes * sizeof(int *));
  for (int i = 0; i < numNodes; i++) {
    graph->adjMatrix[i] = (int *)calloc(numNodes, sizeof(int));
  }
  return graph;
}

void addEdge(Graph *graph, int src, int dest) {
  graph->adjMatrix[src][dest] = 1;
  graph->adjMatrix[dest][src] = 1; // Undirected graph
}

void freeGraph(Graph *graph) {
  for (int i = 0; i < graph->numNodes; i++) {
    free(graph->adjMatrix[i]);
  }
  free(graph->adjMatrix);
  free(graph);
}

// Helper function to benchmark CUDA DFS
void benchmarkCudaDFS(Graph *graph) {
    int numNodes = graph->numNodes;
    int *h_adjMatrix = (int *)malloc(numNodes * numNodes * sizeof(int));
    for (int i = 0; i < numNodes; i++) {
        for (int j = 0; j < numNodes; j++) {
            h_adjMatrix[i * numNodes + j] = graph->adjMatrix[i][j];
        }
    }

    int *d_adjMatrix, *d_stack, *d_top, *d_visited;
    cudaMalloc(&d_adjMatrix, numNodes * numNodes * sizeof(int));
    cudaMemcpy(d_adjMatrix, h_adjMatrix, numNodes * numNodes * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_stack, MAX_NODES * sizeof(int));
    cudaMalloc(&d_top, sizeof(int));
    cudaMalloc(&d_visited, numNodes * sizeof(int));

    // Initialize stack with start node 0
    int initial_top = 0;
    cudaMemcpy(d_top, &initial_top, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_visited, 0, numNodes * sizeof(int));
    int initial_node = 0;
    cudaMemcpy(d_stack, &initial_node, sizeof(int), cudaMemcpyHostToDevice);

    // Launch configuration
    int threadsPerBlock = 256;
    int numBlocks = (numNodes + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int h_top;
    do {
        dfs_kernel<<<numBlocks, threadsPerBlock>>>(d_adjMatrix, numNodes, d_stack, d_top, d_visited);
        cudaMemcpy(&h_top, d_top, sizeof(int), cudaMemcpyDeviceToHost);
    } while (h_top >= 0);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Cleanup
    cudaFree(d_adjMatrix);
    cudaFree(d_stack);
    cudaFree(d_top);
    cudaFree(d_visited);
    free(h_adjMatrix);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// Existing functions from the original code remain mostly unchanged, 
// except for adding the CUDA benchmark call in the benchmarkDFS function.

void benchmarkDFS(Graph *graph) {
  clock_t start, end;
  double time_taken;

  start = clock();

  // CUDA DFS Benchmark
  benchmarkCudaDFS(graph);
  
  end = clock();
  time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
  printf("Multi-threaded Recursive DFS Time: %f seconds\n", time_taken);
}

// The rest of the original code (createGraph, addEdge, freeGraph, main, etc.)
// remains unchanged except for including the CUDA benchmark in benchmarkDFS.

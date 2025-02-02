#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MAX_NODES 25000
#define NUM_THREADS 12

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

void *dfsRecursiveThread(void *args) {
  ThreadArgs *tArgs = (ThreadArgs *)args;
  Graph *graph = tArgs->graph;
  int node = tArgs->node;
  int *visited = tArgs->visited;

  visited[node] = 1;
  for (int i = 0; i < graph->numNodes; i++) {
    if (graph->adjMatrix[node][i] == 1 && !visited[i]) {
      ThreadArgs newArgs = {graph, i, visited};
      dfsRecursiveThread(&newArgs);
    }
  }
  return NULL;
}

void dfsIterative(Graph *graph, int startNode) {
  int visited[MAX_NODES] = {0};
  Stack stack;
  stack.top = -1;
  push(&stack, startNode);

  while (!isEmpty(&stack)) {
    int node = pop(&stack);
    if (!visited[node]) {
      visited[node] = 1;
      for (int i = graph->numNodes - 1; i >= 0; i--) {
        if (graph->adjMatrix[node][i] == 1 && !visited[i]) {
          push(&stack, i);
        }
      }
    }
  }
}

void benchmarkDFS(Graph *graph) {
  clock_t start, end;
  double time_taken;

  // Multi-threaded Recursive DFS Benchmark
  int *visited = (int *)calloc(graph->numNodes, sizeof(int));
  pthread_t thread;
  ThreadArgs args = {graph, 0, visited};
  start = clock();
  pthread_create(&thread, NULL, dfsRecursiveThread, &args);
  pthread_join(thread, NULL);
  end = clock();
  time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
  printf("Multi-threaded Recursive DFS Time: %f seconds\n", time_taken);
  free(visited);

  // Iterative DFS Benchmark
  start = clock();
  dfsIterative(graph, 0);
  end = clock();
  time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
  printf("Iterative DFS Time: %f seconds\n", time_taken);
}

void push(Stack *stack, int value) { stack->items[++stack->top] = value; }

int pop(Stack *stack) { return stack->items[stack->top--]; }

int isEmpty(Stack *stack) { return stack->top == -1; }

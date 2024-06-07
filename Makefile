# Define the CUDA compiler
NVCC = nvcc

# Define compiler flags
NVCC_FLAGS = -arch=sm_60 -std=c++11

# Define the source files
SRC = main.cu kernel.cu

# Define the object files
OBJ = $(SRC:.cu=.o)

# Define the executable name
EXEC = cnn_model

# Build the executable
all: $(EXEC)

# Rule to link the object files into the executable
$(EXEC): $(OBJ)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

# Rule to compile the CUDA source files into object files
%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Rule to clean up the object files and executable
clean:
	rm -f $(OBJ) $(EXEC)

# Phony targets
.PHONY: all clean

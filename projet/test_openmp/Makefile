CXX = g++
CXXFLAGS = -O3 -fopenmp -std=c++17

TARGET = bench_openmp
SRC = main.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET)
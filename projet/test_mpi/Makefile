CXX = mpicxx
CXXFLAGS = -O3 -std=c++17 -Wall

TARGET = bench_mpi
SRC = mpi_bench.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET)
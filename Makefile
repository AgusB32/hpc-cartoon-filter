CXXFLAGS = -O3 -std=c++17 -Iinclude

all: bin/cartoon_seq bin/cartoon_omp bin/cartoon_mpi bin/cartoon_hybrid

bin/cartoon_seq: src/cartoon_seq.cpp include/cartoon_common.hpp
	@mkdir -p bin
	g++ $(CXXFLAGS) -fopenmp src/cartoon_seq.cpp -o bin/cartoon_seq -lm

bin/cartoon_omp: src/cartoon_omp.cpp include/cartoon_common.hpp
	@mkdir -p bin
	g++ $(CXXFLAGS) -fopenmp src/cartoon_omp.cpp -o bin/cartoon_omp -lm

bin/cartoon_mpi: src/cartoon_mpi.cpp include/cartoon_common.hpp
	@mkdir -p bin
	mpicxx $(CXXFLAGS) src/cartoon_mpi.cpp -o bin/cartoon_mpi -lm

bin/cartoon_hybrid: src/cartoon_hybrid.cpp include/cartoon_common.hpp
	@mkdir -p bin
	mpicxx $(CXXFLAGS) -fopenmp src/cartoon_hybrid.cpp -o bin/cartoon_hybrid -lm

clean:
	rm -rf bin/* output/*


CC=g++
LD=-lopencv_core -lopencv_imgcodecs

all: main

main: main.cpp
	$(CC) main.cpp -o main $(LD)

.PHONY: clean
clean:
	rm main

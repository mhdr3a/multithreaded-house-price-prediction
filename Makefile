all: main

main : main.o
	g++ main.o -o HousePricePrediction.out -lpthread

main.o : main.cpp
	g++ -std=c++11 -c main.cpp -lpthread
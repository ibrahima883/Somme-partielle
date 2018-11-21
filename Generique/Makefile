DEBUG = no
PARALLEL = yes
CXX = g++
ifeq ($(DEBUG)$(PARALLEL),yesyes)
	CXXFLAGS=-Wall -fopenmp -O2 -s -std=c++14 -g
	LDFLAGS= -lgomp
else ifeq ($(DEBUG)$(PARALLEL),yesno)
	CXXFLAGS=-ansi -pedantic -Wextra -std=c++14 -g
	LDFLAGS=
else ifeq ($(DEBUG)$(PARALLEL),noyes)
  CXXFLAGS=-Wall -fopenmp -O2 -s -std=c++14 
  LDFLAGS= -lgomp
else
	CXXFLAGS= -ansi -pedantic -Wextra -std=c++14 
	LDFLAGS= 
endif
PROGS=Main
OBJS=Main.o
LIBS = -lm 
all: $(PROGS)
ifeq ($(DEBUG),yes)
	@echo "Génération en mode debug"
else
	@echo "Génération en mode release"
endif

Main:Main.o $(OBJS)
	$(CXX) -o $@ $^ $(CXXFLGAS) $(LDFLAGS) $(LIBS) 
	
Main.o: Matrix.h Operations.h

.PHONY: clean mrproper

clean:
	@rm -rf *.o

mrproper: clean
	@rm -rf $(PROGS)

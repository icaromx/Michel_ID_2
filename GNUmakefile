CXXFLAGS += -I. $(shell root-config --cflags) -g
LDFLAGS += $(shell root-config --libs) -g

PROGRAMS = Michel_ID

all:		clean $(PROGRAMS)

$(PROGRAMS):
	@echo '<<compiling' $@'>>'
	@$(CXX) $@.cpp -o run_$@ $(CXXFLAGS) $(LDFLAGS)
	@rm -rf *.dSYM
clean:	
	rm -f $(PROGRAMS)

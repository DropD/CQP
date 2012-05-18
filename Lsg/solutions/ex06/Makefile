# set to the path of your ALPS installation
ALPS_ROOT = /opt/alps
include $(ALPS_ROOT)/share/alps/include.mk

PROG = heised

$(PROG):   $(PROG).o
	$(CXX) -o $(PROG) $(PROG).o $(LDFLAGS) $(LIBS)

clean:
	$(RM) $(RMFLAGS) $(PROG) *.o

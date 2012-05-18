include $(ALPS_ROOT)/share/alps/include.mk

HEADERS = $(wildcard *.h)
SOURCES = $(wildcard *.cpp)
OBJS = $(patsubst %.cpp, objects/%.o, $(SOURCES))
OBJS_DBG = $(patsubst %.o, %.o_dbg, $(OBJS))


# Debugging flags
CPPFLAGS_DBG = $(BOOST_CPPFLAGS) -I$(includedir)
CXXFLAGS_DBG = -Wall -pthread -ftemplate-depth-150 -O0 -fexceptions -g
LDFLAGS_DBG = -L$(libdir) $(LAPACK_LDFLAGS)

all:    pimc pimc.dbg

pimc:   $(OBJS)
	$(CXX) -o pimc $(OBJS) $(LDFLAGS) $(LIBS)

pimc.dbg:       $(OBJS_DBG)
	$(CXX) -o pimc.dbg $(OBJS_DBG) $(LDFLAGS_DBG) $(LIBS)

clean:
	$(RM) $(RMFLAGS) $(OBJS) $(OBJS_DBG) $(OBJS_PROF) pimc pimc.dbg

$(OBJS) $(OBJS_DBG):       $(HEADERS)

objects/%.o:    %.cpp
	mkdir -p objects
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

objects/%.o_dbg:        %.cpp
	mkdir -p objects
	$(CXX) $(CPPFLAGS_DBG) $(CXXFLAGS_DBG) -c $< -o $@

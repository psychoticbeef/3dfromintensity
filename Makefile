CC=clang
CXX=clang++
RM=rm -f
LIBS=opencv eigen3
CPPFLAGS=-std=c++11 -Wall -Wno-deprecated $(shell pkg-config --cflags ${LIBS}) -pthread -fPIC -fstack-protector-all -fno-omit-frame-pointer -D_FORTIFY_SOURCE=2 -Os
LDFLAGS=-pie -fstack-protector-all -fno-omit-frame-pointer -z relro -z now -s -flto
LDLIBS=-lboost_system -lboost_filesystem -lboost_program_options -lopencv_highgui -lopencv_imgproc -lopencv_core

SRCS=beamprofile.cc
OBJS=$(subst .cc,.o,$(SRCS))

all: beamprofile

beamprofile: $(OBJS)
	    $(CXX) $(LDFLAGS) -o beamprofile $(OBJS) $(LDLIBS)

beamprofile.o: beamprofile.cc

clean:
	    $(RM) $(OBJS)
			$(RM) out/*.png

dist-clean: clean
	    $(RM) beamprofile


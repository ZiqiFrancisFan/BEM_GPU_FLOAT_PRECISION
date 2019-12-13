# Location of the CUDA Toolkit
CUDA_PATH ?= /usr/local/cuda

# architecture
HOST_ARCH   := $(shell uname -m)
TARGET_ARCH ?= $(HOST_ARCH)

# Adjust this for ARMv7 with a 32-bit filesystem
ifeq ($(TARGET_ARCH), aarch64)
    ifeq ($(shell file /sbin/init | grep 32-bit), 1)
        TARGET_ARCH=armv7l
    endif
endif
 
ifneq (,$(filter $(TARGET_ARCH),x86_64 aarch64 ppc64le armv7l))
    ifneq ($(TARGET_ARCH),$(HOST_ARCH))
        ifneq (,$(filter $(TARGET_ARCH),x86_64 aarch64 ppc64le))
            TARGET_SIZE := 64
        else ifneq (,$(filter $(TARGET_ARCH),armv7l))
            TARGET_SIZE := 32
        endif
    else
        TARGET_SIZE := $(shell getconf LONG_BIT)
    endif
else
    $(error ERROR - unsupported value $(TARGET_ARCH) for TARGET_ARCH!)
endif
ifneq ($(TARGET_ARCH),$(HOST_ARCH))
    ifeq (,$(filter $(HOST_ARCH)-$(TARGET_ARCH),aarch64-armv7l x86_64-armv7l x86_64-aarch64 x86_64-ppc64le))
        $(error ERROR - cross compiling from $(HOST_ARCH) to $(TARGET_ARCH) is not supported!)
    endif
endif

# operating system
HOST_OS   := $(shell uname -s 2>/dev/null | tr "[:upper:]" "[:lower:]")
TARGET_OS ?= $(HOST_OS)

ifeq ($(TARGET_OS),QNX)
override TARGET_OS := qnx
endif

ifeq (,$(filter $(TARGET_OS),linux darwin qnx android))
    $(error ERROR - unsupported value $(TARGET_OS) for TARGET_OS!)
endif

# host compiler
ifeq ($(TARGET_OS),darwin)
    ifeq ($(shell expr `xcodebuild -version | grep -i xcode | awk '{print $$2}' | cut -d'.' -f1` \>= 5),1)
        HOST_COMPILER ?= clang++
    endif
else ifneq ($(TARGET_ARCH),$(HOST_ARCH))
    ifeq ($(HOST_ARCH)-$(TARGET_ARCH),x86_64-armv7l)
        ifeq ($(TARGET_OS),linux)
            HOST_COMPILER ?= arm-linux-gnueabihf-g++
        else ifeq ($(TARGET_OS),qnx)
            ifeq ($(QNX_HOST),)
                $(error ERROR - QNX_HOST must be passed to the QNX host toolchain)
            endif
            ifeq ($(QNX_TARGET),)
                $(error ERROR - QNX_TARGET must be passed to the QNX target toolchain)
            endif
            export QNX_HOST
            export QNX_TARGET
            HOST_COMPILER ?= $(QNX_HOST)/usr/bin/arm-unknown-nto-qnx6.6.0eabi-g++
        else ifeq ($(TARGET_OS),android)
            HOST_COMPILER ?= arm-linux-androideabi-g++
        endif
    else ifeq ($(TARGET_ARCH),aarch64)
        ifeq ($(TARGET_OS), linux)
            HOST_COMPILER ?= aarch64-linux-gnu-g++
        else ifeq ($(TARGET_OS), android)
            HOST_COMPILER ?= aarch64-linux-android-g++
        endif
    else ifeq ($(TARGET_ARCH),ppc64le)
        HOST_COMPILER ?= powerpc64le-linux-gnu-g++
    endif
endif
HOST_COMPILER ?= g++
NVCCONLY          := $(CUDA_PATH)/bin/nvcc -use_fast_math
NVCC 		  := $(CUDA_PATH)/bin/nvcc  -ccbin $(HOST_COMPILER) -use_fast_math
#
# internal flags
NVCCFLAGS   := -m${TARGET_SIZE}
CCFLAGS     := -std=c++11 -O3
LDFLAGS     :=

# build flags
ifeq ($(TARGET_OS),darwin)
    LDFLAGS += -rpath $(CUDA_PATH)/lib
    CCFLAGS += -arch $(HOST_ARCH)
else ifeq ($(HOST_ARCH)-$(TARGET_ARCH)-$(TARGET_OS),x86_64-armv7l-linux)
    LDFLAGS += --dynamic-linker=/lib/ld-linux-armhf.so.3
    CCFLAGS += -mfloat-abi=hard
else ifeq ($(TARGET_OS),android)
    LDFLAGS += -pie
    CCFLAGS += -fpie -fpic -fexceptions
endif

ifneq ($(TARGET_ARCH),$(HOST_ARCH))
    ifeq ($(TARGET_ARCH)-$(TARGET_OS),armv7l-linux)
        ifneq ($(TARGET_FS),)
            GCCVERSIONLTEQ46 := $(shell expr `$(HOST_COMPILER) -dumpversion` \<= 4.6)
            ifeq ($(GCCVERSIONLTEQ46),1)
                CCFLAGS += --sysroot=$(TARGET_FS)
            endif
            LDFLAGS += --sysroot=$(TARGET_FS)
            LDFLAGS += -rpath-link=$(TARGET_FS)/lib
            LDFLAGS += -rpath-link=$(TARGET_FS)/usr/lib
            LDFLAGS += -rpath-link=$(TARGET_FS)/usr/lib/arm-linux-gnueabihf
        endif
    endif
endif

# Debug build flags
# ifeq ($(dbg),1)
      NVCCFLAGS :=
      BUILD_TYPE := release
# else
#       BUILD_TYPE := release
# endif

ALL_CCFLAGS := -std=c++11 -O3
ALL_CCFLAGS += $(NVCCFLAGS)
ALL_CCFLAGS += $(EXTRA_NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(EXTRA_CCFLAGS))
ALL_CCFLAGS += -Xptxas -O3 -Xptxas -allow-expensive-optimizations=true -maxrregcount 255
# ALL_CCFLAGS += -lineinfo
#ALL_CCFLAGS += -maxrregcount 255
# -res-usage : this will print the resources usage.

SAMPLE_ENABLED := 1

ALL_LDFLAGS :=
ALL_LDFLAGS += $(ALL_CCFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))
ALL_LDFLAGS += $(addprefix -Xlinker ,$(EXTRA_LDFLAGS))

# Common includes and paths for CUDA
ifneq ($(TARGET_ARCH), ppc64le)
INCLUDES := -I$(CUDA_PATH)/include -I$(CUDA_PATH)/include/crt -I/usr/include/python2.7 -I.
else
INCLUDES := -I$(CUDA_PATH)/targets/ppc64le-linux/include 
endif
LIBRARIES :=

################################################################################
#30 35 50 
# Gencode arguments
SMS ?= 61

ifeq ($(SMS),)
$(info >>> WARNING - no SM architectures have been specified - waiving sample <<<)
SAMPLE_ENABLED := 0
endif

ifeq ($(GENCODE_FLAGS),)
# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

# Generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
HIGHEST_SM := $(lastword $(sort $(SMS)))
ifneq ($(HIGHEST_SM),)
GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
endif
endif

#INCLUDES += -IFreeImage/include
#-LFreeImage/lib/$(TARGET_OS)/$(TARGET_ARCH) -LFreeImage/lib/$(TARGET_OS) 
#-lfreeimage

LIBRARIES += -lcudart -lcublas -lcurand -lstdc++ -lpthread -lm
LIBRARIES += -lcudart -lcublas -lcurand -lcusolver -lcufft -lcusparse 
LIBRARIES += -lstdc++ -lm -lgsl -lgslcblas


################################################################################
vpath %.c ./src
vpath %.h ./src
vpath %.o ./bin
vpath main .bin

# Target rules
all: build

build: main

check.deps:
ifeq ($(SAMPLE_ENABLED),0)
	@echo "Sample will be waived due to the above missing dependencies"
else
	@echo "Sample is ready - all dependencies have been met"
endif


OBJ = main.o numerical.o mesh.o octree.o geometry.o
#     Bias.o  

main : $(OBJ)
	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

main.o : main.cpp numerical.h mesh.h dataStructs.h geometry.h
	$(EXEC) $(HOST_COMPILER) $(INCLUDES) $(CCFLAGS) $(EXTRA_CCFLAGS) -c main.cpp

numerical.o : numerical.cu numerical.h mesh.h dataStructs.h octree.h
	$(NVCCONLY) -dc $(INCLUDES) $(ALL_CCFLAGS) $(EXTRA_CCFLAGS) $(GENCODE_FLAGS) numerical.cu
	
mesh.o: mesh.cpp mesh.h numerical.h dataStructs.h
	$(NVCCONLY) -dc $(INCLUDES) $(ALL_CCFLAGS) $(EXTRA_CCFLAGS) $(GENCODE_FLAGS) mesh.cpp
	
octree.o: octree.cpp octree.h numerical.h dataStructs.h mesh.h
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(EXTRA_CCFLAGS) $(GENCODE_FLAGS) -c octree.cpp
	
geometry.o: geometry.cu geometry.h numerical.h dataStructs.h octree.h
	$(NVCCONLY) -dc $(INCLUDES) $(ALL_CCFLAGS) $(EXTRA_CCFLAGS) $(GENCODE_FLAGS) geometry.cu


# main: $(OBJ)
# 	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

# %.o: %.cpp
# 	$(EXEC) $(HOST_COMPILER) $(INCLUDES) $(CCFLAGS) $(EXTRA_CCFLAGS) -o $@ -c $<

# %.o: %.cu
# 	$(EXEC) $(NVCCONLY) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

run: build
	touch log.txt
	$(EXEC) ./main | tee log.txt 

clean:
	rm $(OBJ)
	rm main

clobber: clean

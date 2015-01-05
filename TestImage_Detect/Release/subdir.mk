################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../Main.cpp 

OBJS += \
./Main.o 

CPP_DEPS += \
./Main.d 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-6.5/bin/nvcc -O3 -ccbin /usr/bin/arm-linux-gnueabihf-g++-4.8 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_32,code=sm_32 --target-cpu-architecture ARM -m32 -odir "" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-6.5/bin/nvcc -O3 -ccbin /usr/bin/arm-linux-gnueabihf-g++-4.8 --compile --target-cpu-architecture ARM -m32  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '



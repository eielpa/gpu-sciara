############
# COMPILER #
############

# Usiamo nvcc come driver principale
CPPC = nvcc

# Flag essenziali:
# -O3: Massima ottimizzazione
# -gencode: Specifica l'architettura della GTX 980 (Compute 5.2)
# -ftz=true: Flush To Zero (velocizza i calcoli decimali piccoli)
NVFLAGS = -O3 -use_fast_math -gencode arch=compute_52,code=compute_52 -ftz=true

# Flag per il linker (se servono librerie specifiche, aggiungile qui)
LDFLAGS = 

###########
# DATASET #
###########
INPUT_CONFIG=./data/2006/2006_000000000000.cfg
OUTPUT_CONFIG=./data/2006/output_2007
OUTPUT=./data/2006/output_2006_000000016000_Temperature.asc
STEPS=1000
REDUCE_INTERVL=1000
THICKNESS_THRESHOLD=1.0

###############
# COMPILATION #
###############

EXEC_SERIAL = sciara_serial
EXEC_CUDA = sciara_cuda

# Trova tutti i sorgenti .cu e .cpp nella cartella corrente
SRC = $(wildcard *.cu *.cpp)

default: cuda

# Target CUDA (Compila tutto con nvcc)
cuda:
	$(CPPC) $(SRC) -o $(EXEC_CUDA) $(NVFLAGS)

# Target Serial (Opzionale, utile per debug se nvcc dà problemi)
# Nota: anche questo userà nvcc ma senza kernel attivi
serial:
	$(CPPC) $(SRC) -o $(EXEC_SERIAL) $(NVFLAGS)

#############
# EXECUTION #
#############

# Lancia l'eseguibile CUDA
run:
	./$(EXEC_CUDA) $(INPUT_CONFIG) $(OUTPUT_CONFIG) $(STEPS) $(REDUCE_INTERVL) $(THICKNESS_THRESHOLD) && md5sum $(OUTPUT)

############
# CLEAN UP #
############

clean:
	rm -f $(EXEC_SERIAL) $(EXEC_CUDA) *.o *output*
	
wipe:
	rm -f *.o *output*

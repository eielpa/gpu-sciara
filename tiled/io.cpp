//---------------------------------------------------------------------------
#include <iostream>
#include <new>
#include <string.h> 
#include <cuda_runtime.h>
#include "io.h"
#include "cal2DBuffer.h"
#include "cal2DBufferIO.h"
//---------------------------------------------------------------------------
#define FILE_ERROR	0
#define FILE_OK		1
//---------------------------------------------------------------------------
bool storing = false; 
TGISInfo gis_info_Sz;
TGISInfo gis_info_generic;
TGISInfo gis_info_nodata0;

/****************************************************************************************
 * PRIVATE FUNCTIONS
 ****************************************************************************************/

void saveMatrixr(double * M, char configuration_path[1024],Sciara * sciara){
}
void saveMatrixi(int * M, char configuration_path[1024],Sciara * sciara){
}

int SaveConfigurationEmission(Sciara* sciara, char const *path, char const *name)
{
  char s[1024];
  if (ConfigurationFileSavingPath((char*)path, sciara->simulation->step, (char*)name, ".txt", s) == false)
    return FILE_ERROR;
  else
  {
    FILE *s_file;
    if ( ( s_file = fopen(s,"w") ) == NULL) return FILE_ERROR;
    saveEmissionRates(s_file, sciara->simulation->emission_time, sciara->simulation->emission_rate);
    fclose(s_file);
    return FILE_OK;
  }
}

/****************************************************************************************
 * PUBLIC FUNCTIONS
 ****************************************************************************************/

int loadParameters(char const * path, Sciara* sciara) {
  printf("DEBUG: [loadParameters] Loading %s\n", path); fflush(stdout);
  char str[256];
  FILE *f;
  fpos_t position;

  if ((f = fopen(path, "r")) == NULL) {
    printf("ERROR: [loadParameters] Cannot open file: %s\n", path);
    return FILE_ERROR;
  }

  fgetpos(f, &position);
  fscanf(f, "%s", str); 
  if (strcmp(str, "maximum_steps_(0_for_loop)") != 0) {
      printf("ERROR: Header mismatch in cfg file.\n");
      return FILE_ERROR; 
  }
  
  fsetpos(f, &position);

  fscanf(f, "%s", str); fscanf(f, "%s", str); sciara->simulation->maximum_steps = atoi(str);
  fscanf(f, "%s", str); fscanf(f, "%s", str); sciara->simulation->stopping_threshold = atof(str);
  fscanf(f, "%s", str); fscanf(f, "%s", str); sciara->simulation->refreshing_step = atoi(str);
  fscanf(f, "%s", str); fscanf(f, "%s", str); sciara->simulation->thickness_visual_threshold = atof(str);
  fscanf(f, "%s", str); fscanf(f, "%s", str); sciara->parameters->Pclock = atof(str);
  fscanf(f, "%s", str); fscanf(f, "%s", str); sciara->parameters->PTsol = atof(str);
  fscanf(f, "%s", str); fscanf(f, "%s", str); sciara->parameters->PTvent = atof(str);
  fscanf(f, "%s", str); fscanf(f, "%s", str); sciara->parameters->Pr_Tsol = atof(str);
  fscanf(f, "%s", str); fscanf(f, "%s", str); sciara->parameters->Pr_Tvent = atof(str);
  fscanf(f, "%s", str); fscanf(f, "%s", str); sciara->parameters->Phc_Tsol = atof(str);
  fscanf(f, "%s", str); fscanf(f, "%s", str); sciara->parameters->Phc_Tvent = atof(str);
  fscanf(f, "%s", str); fscanf(f, "%s", str); sciara->parameters->Pcool = atof(str);
  fscanf(f, "%s", str); fscanf(f, "%s", str); sciara->parameters->Prho = atof(str);
  fscanf(f, "%s", str); fscanf(f, "%s", str); sciara->parameters->Pepsilon = atof(str);
  fscanf(f, "%s", str); fscanf(f, "%s", str); sciara->parameters->Psigma = atof(str);
  fscanf(f, "%s", str); fscanf(f, "%s", str); sciara->parameters->Pcv = atof(str);
  fscanf(f, "%s", str); fscanf(f, "%s", str);
  if (strcmp(str, "PROP") == 0) sciara->parameters->algorithm = PROP_ALG;
  else if (strcmp(str, "MIN") == 0) sciara->parameters->algorithm = MIN_ALG;

  fclose(f);
  return FILE_OK;
}

int saveParameters(char* path, Sciara* sciara) {
    return FILE_OK; 
}

void printParameters(Sciara* sciara) {
}

//---------------------------------------------------------------------------
int loadMorphology(char* path, Sciara* sciara) 
{
  printf("DEBUG: [loadMorphology] Reading %s\n", path); fflush(stdout);
  FILE *input_file;

  if ((input_file = fopen(path, "r")) == NULL) return FILE_ERROR;

  if (readGISInfo(gis_info_Sz, input_file) != GIS_FILE_OK) {
      fclose(input_file); return FILE_ERROR;
  }
  
  initGISInfoNODATA0(gis_info_Sz, gis_info_nodata0);

  sciara->domain->cols = gis_info_Sz.ncols;
  sciara->domain->rows = gis_info_Sz.nrows;
  sciara->parameters->Pc   = gis_info_Sz.cell_size;
  sciara->parameters->Pac  = sciara->parameters->Pc * sciara->parameters->Pc;

  printf("DEBUG: [loadMorphology] Allocating GPU memory...\n"); fflush(stdout);
  allocateSubstates(sciara);
  
  // Allocazione sicura su Host
  printf("DEBUG: [loadMorphology] Reading matrix to Host buffer...\n"); fflush(stdout);
  double *tempSz = new double[sciara->domain->rows * sciara->domain->cols];
  calfLoadMatrix2Dr(tempSz, sciara->domain->rows, sciara->domain->cols, input_file);
  
  // Copia su GPU
  printf("DEBUG: [loadMorphology] Copying to GPU...\n"); fflush(stdout);
  size_t size = sciara->domain->rows * sciara->domain->cols * sizeof(double);
  cudaMemcpy(sciara->substates->Sz, tempSz, size, cudaMemcpyHostToDevice);
  cudaMemcpy(sciara->substates->Sz_next, tempSz, size, cudaMemcpyHostToDevice);
  
  delete[] tempSz;
  fclose(input_file);
  
  printf("DEBUG: [loadMorphology] Done.\n"); fflush(stdout);
  return FILE_OK;
}

//---------------------------------------------------------------------------
int loadVents(char* path, Sciara* sciara) 
{
  printf("DEBUG: [loadVents] Reading %s\n", path); fflush(stdout);
  FILE *input_file;
  if ((input_file = fopen(path,"r")) == NULL) return FILE_ERROR;

  readGISInfo(gis_info_generic, input_file);

  // Mv è usato solo su Host per inizializzare la struct vent
  sciara->substates->Mv = new int[sciara->domain->rows * sciara->domain->cols];

  calfLoadMatrix2Di(sciara->substates->Mv, sciara->domain->rows, sciara->domain->cols, input_file);
  fclose(input_file);

  initVents(sciara->substates->Mv, sciara->domain->cols, sciara->domain->rows, sciara->simulation->vent);

  delete[] sciara->substates->Mv;
  sciara->substates->Mv = NULL;
  
  printf("DEBUG: [loadVents] Done.\n"); fflush(stdout);
  return FILE_OK;
}

//---------------------------------------------------------------------------
int loadEmissionRate(char *path, Sciara* sciara) 
{
  printf("DEBUG: [loadEmissionRate] Reading %s\n", path); fflush(stdout);
  FILE *input_file;
  if ((input_file = fopen(path, "r")) == NULL) {
      printf("ERROR: Cannot open EmissionRate file: %s\n", path);
      return FILE_ERROR;
  }

  int status = loadEmissionRates(input_file, sciara->simulation->emission_time, sciara->simulation->emission_rate, sciara->simulation->vent);
  fclose(input_file);

  int error = defineVents(sciara->simulation->emission_rate, sciara->simulation->vent);
  
  if (error || status != EMISSION_RATE_FILE_OK) {
      printf("ERROR: Failed in loadEmissionRate logic.\n");
      return FILE_ERROR;
  }

  printf("DEBUG: [loadEmissionRate] Done.\n"); fflush(stdout);
  return 1;
}

//---------------------------------------------------------------------------
// Helper for loading optional maps
//---------------------------------------------------------------------------
int loadAlreadyAllocatedMap(char *path, double* S, double* nS, int lx, int ly) {
  printf("DEBUG: [loadMap] Reading %s\n", path); fflush(stdout);
  FILE *input_file;
  
  if ((input_file = fopen(path, "r")) == NULL) {
    printf("WARNING: File not found (skipping): %s\n", path);
    return FILE_ERROR;
  }

  readGISInfo(gis_info_generic, input_file);

  double *tempBuffer = new double[lx * ly];
  calfLoadMatrix2Dr(tempBuffer, ly, lx, input_file);
  fclose(input_file);

  size_t size = lx * ly * sizeof(double);
  cudaMemcpy(S, tempBuffer, size, cudaMemcpyHostToDevice);
  if (nS != NULL) cudaMemcpy(nS, tempBuffer, size, cudaMemcpyHostToDevice);

  delete[] tempBuffer;
  return FILE_OK;
}

//---------------------------------------------------------------------------------------------------
int loadConfiguration(char const *path, Sciara* sciara)
{
  printf("DEBUG: [loadConfiguration] Start.\n"); fflush(stdout);
  char configuration_path[1024];

  if (!loadParameters(path, sciara)) 
  {
    char alt_path[1024];
    strcpy(alt_path, path);
    strcat(alt_path, "_000000000000.cfg");
    if (!loadParameters(alt_path, sciara)) return FILE_ERROR;
  }

  ConfigurationFilePath((char*)path, "Morphology", ".asc", configuration_path);
  if (!loadMorphology(configuration_path, sciara)) return FILE_ERROR;

  ConfigurationFilePath((char*)path, "Vents", ".asc", configuration_path);
  if (!loadVents(configuration_path, sciara)) return FILE_ERROR;

  ConfigurationFilePath((char*)path, "EmissionRate", ".txt", configuration_path);
  if (!loadEmissionRate(configuration_path, sciara)) return FILE_ERROR;

  ConfigurationFilePath((char*)path, "Thickness", ".asc", configuration_path);
  loadAlreadyAllocatedMap(configuration_path, sciara->substates->Sh, sciara->substates->Sh_next, sciara->domain->cols, sciara->domain->rows);

  ConfigurationFilePath((char*)path, "Temperature", ".asc", configuration_path);
  loadAlreadyAllocatedMap(configuration_path, sciara->substates->ST, sciara->substates->ST_next, sciara->domain->cols, sciara->domain->rows);

  ConfigurationFilePath((char*)path, "SolidifiedLavaThickness", ".asc", configuration_path);
  loadAlreadyAllocatedMap(configuration_path, sciara->substates->Mhs, NULL, sciara->domain->cols, sciara->domain->rows);

  sciara->simulation->step = GetStepFromConfigurationFile((char*)path);
  
  printf("DEBUG: [loadConfiguration] Done.\n"); fflush(stdout);
  return FILE_OK;
}

int saveConfiguration(char const *path, Sciara* sciara)
{
  return FILE_OK;
}
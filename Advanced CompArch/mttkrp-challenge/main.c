#include <stdio.h>
/*
    This file is part of ParTI!.

    ParTI! is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as
    published by the Free Software Foundation, either version 3 of
    the License, or (at your option) any later version.

    ParTI! is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with ParTI!.
    If not, see <http://www.gnu.org/licenses/>.
*/

#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
//#include <pasta.h>
//#include "../src/sptensor/sptensor.h"
#include "structs.h"
#include <string.h>
#include "error.h"
#include "helper_funcs.h"
#include "types.h"
#include "sptensors.h"
#include "matricies.h"

static void print_usage(char ** argv) {
	printf("Usage: %s [options] \n\n", argv[0]);
	printf("Options: -i INPUT, --input=INPUT (.tns file)\n");
	printf("         -o OUTPUT, --output=OUTPUT (output file name)\n");
	printf("         -m MODE, --mode=MODE (specify a mode, e.g., 0 (default) or 1 or 2 for third-order tensors.)\n");
	printf("         -d DEV_ID, --dev-id=DEV_ID (-2:sequential,default; -1:OpenMP parallel)\n");
	printf("         -r RANK (the number of matrix columns, 16:default)\n");
	printf("         -v VALIDATION, --validate=VALIDFILE (a previous output file to compare against). This also removes randomisation from matrix creation\n");
	printf("         --help\n");
	printf("\n");
}
/* Function declaration */
int compareFile(FILE * fPtr1, FILE * fPtr2);

/**
 * Benchmark Matriced Tensor Times Khatri-Rao Product (MTTKRP), tensor in COO format, matrices are dense.
 */
int main(int argc, char ** argv)
{
	FILE *fo = NULL;
	char fname[1000];
	char fvname[1000];
	char foname[1000];
	sptSparseTensor X;
	sptMatrix ** U;

	bool random = true;
	sptIndex mode = 0;
	sptIndex R = 16;
	int dev_id = -2;
	int niters = 5;
	int nthreads = 1;
	printf("niters: %d\n", niters);

	if(argc <= 3) { // #Required arguments
		print_usage(argv);
		exit(1);
	}

	static struct option long_options[] = {
			{"input", required_argument, 0, 'i'},
			{"mode", required_argument, 0, 'm'},
			{"output", optional_argument, 0, 'o'},
			{"dev-id", optional_argument, 0, 'd'},
			{"rank", optional_argument, 0, 'r'},
			{"nthreads", optional_argument, 0, 't'},
			{"help", no_argument, 0, 0},
			{"validate", optional_argument, 0, 'v'},
			{0, 0, 0, 0}
	};
	int c;
	for(;;) {
		int option_index = 0;
		c = getopt_long(argc, argv, "i:m:o:d:r:v:", long_options, &option_index);
		if(c == -1) {
			break;
		}
		switch(c) {
			case 'i':
				strcpy(fname, optarg);
				printf("input file: %s\n", fname); fflush(stdout);
				break;
			case 'o':
				fo = fopen(optarg, "w");
				strcpy(foname, optarg);
				sptAssert(fo != NULL);
				printf("output file: %s\n", optarg); fflush(stdout);
				break;
			case 'm':
				sscanf(optarg, "%"PASTA_SCN_INDEX, &mode);
				break;
			case 'd':
				sscanf(optarg, "%d", &dev_id);
				if(dev_id < -2 || dev_id >= 0) {
					fprintf(stderr, "Error: set dev_id to -2/-1.\n");
					exit(1);
				}
				break;
			case 'r':
				sscanf(optarg, "%u"PASTA_SCN_INDEX, &R);
				break;
			case 'v':
				random = false;
				strcpy(fvname, optarg);
				printf("validation input file: %s\n", fvname); fflush(stdout);
				break;
			case '?':   /* invalid option */
			case 'h':
			default:
				print_usage(argv);
				exit(1);
		}
	}

	printf("mode: %"PASTA_PRI_INDEX "\n", mode);
	printf("dev_id: %d\n", dev_id);

	/* Load a sparse tensor from file as it is */
	sptAssert(sptLoadSparseTensor(&X, 1, fname) == 0);
	sptSparseTensorStatus(&X, stdout);

	sptIndex nmodes = X.nmodes;
	U = (sptMatrix **)malloc((nmodes+1) * sizeof(sptMatrix*));
	for(sptIndex m=0; m<nmodes+1; ++m) {
		U[m] = (sptMatrix *)malloc(sizeof(sptMatrix));
	}
	sptIndex max_ndims = 0;
	for(sptIndex m=0; m<nmodes; ++m) {
		sptAssert(sptNewMatrix(U[m], X.ndims[m], R) == 0);
		// sptAssert(sptConstantMatrix(U[m], 1) == 0);
		sptAssert(sptRandomizeMatrix(U[m], random) == 0);
		if(X.ndims[m] > max_ndims)
			max_ndims = X.ndims[m];
	}
	sptAssert(sptNewMatrix(U[nmodes], max_ndims, R) == 0);
	sptAssert(sptConstantMatrix(U[nmodes], 0) == 0);
	sptIndex stride = U[0]->stride;

	sptIndex * mats_order = (sptIndex*)malloc(nmodes * sizeof(sptIndex));
	mats_order[0] = mode;
	for(sptIndex i=1; i<nmodes; ++i)
		mats_order[i] = (mode+i) % nmodes;

	/* For warm-up caches, timing not included */
	if(dev_id == -2) {
		nthreads = 1;
		sptAssert(sptMTTKRP(&X, U, mats_order, mode) == 0);
	} else if(dev_id == -1) {
#ifdef PASTA_USE_OPENMP
		#pragma omp parallel
        {
            nthreads = omp_get_num_threads();
        }
        printf("\nnthreads: %d\n", nthreads);
        sptAssert(sptOmpMTTKRP(&X, U, mats_order, mode, nthreads) == 0);
#endif
	}


	sptTimer timer;
	sptNewTimer(&timer, 0);
	sptStartTimer(timer);

	for(int it=0; it<niters; ++it) {
		sptAssert(sptConstantMatrix(U[nmodes], 0) == 0);
		if(dev_id == -2) {
			sptAssert(sptMTTKRP(&X, U, mats_order, mode) == 0);
		} else if(dev_id == -1) {
#ifdef PASTA_USE_OPENMP
			sptAssert(sptOmpMTTKRP(&X, U, mats_order, mode, nthreads) == 0);
#endif
		}
	}

	sptStopTimer(timer);

	double aver_time = sptPrintAverageElapsedTime(timer, niters, "Average CooMTTKRP");
	double gflops = (double)nmodes * R * X.nnz / aver_time / 1e9;
	uint64_t bytes = ( nmodes * sizeof(sptIndex) + sizeof(sptValue) ) * X.nnz;
	for (sptIndex m=0; m<nmodes; ++m) {
		bytes += X.ndims[m] * R * sizeof(sptValue);
	}
	double gbw = (double)bytes / aver_time / 1e9;
	printf("Performance: %.10lf GFlop/s, Bandwidth: %.2lf GB/s\n\n", gflops, gbw);

	if(fo != NULL) {
		sptAssert(sptDumpMatrix(U[nmodes], fo) == 0);
		fclose(fo);
	}

	sptFreeTimer(timer);
	for(sptIndex m=0; m<nmodes; ++m) {
		sptFreeMatrix(U[m]);
	}
	sptFreeSparseTensor(&X);
	free(mats_order);
	sptFreeMatrix(U[nmodes]);
	free(U);

	if (!random){
		FILE* fPtr1 = fopen(fvname, "r");
		FILE* fPtr2 = fopen(foname, "r");

		if (fPtr1 == NULL || fPtr2 == NULL) {
			printf("\nUnable to open file.\n");
			printf("Please check whether file exists and you have read privilege.\n");
		} else {

			int diff = compareFile(fPtr1, fPtr2);
			if (diff == 0) {
				printf("Validation Successful \n %s matchs %s\n", foname, fvname);
			} else {
				printf("\nFiles are not equal.\n Validation FAILED \n");
			}
		}
		if(fPtr1 != NULL) fclose(fPtr1);
		if(fPtr2 != NULL) fclose(fPtr2);

	}
	return 0;
}

int is_really_different(int a, int b);

int compareFile(FILE * fPtr1, FILE * fPtr2)
{
	int ch1, ch2;
	do {
		ch1 = fgetc(fPtr1);
		ch2 = fgetc(fPtr2);
		if (ch1 != ch2) {
			if (is_really_different(ch1, ch2)) {
				printf("%c, %c\n", ch1, ch2);
				return -1;
			}
		}
	} while (ch1 != EOF && ch2 != EOF);

	/* If both files have reached end */
	if (ch1 == EOF && ch2 == EOF)
		return 0;
	else
		return -1;
}

int is_really_different(int a, int b) {
	return !((a == '0' && b == '9') || (b == '0' && a == '9')) && (a - b > 1 || b - a > 1);
}

#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include "GVD.H"
#include "structure.h"
#include "miner.h"
#include "timelog.h"


FILE*		g_pfRes;

long long**	g_ppllGreedyPreR;
long long**	g_ppllGreedyCurR;
UINT4**		g_ppuPreMatNumArr;
UINT4**		g_ppuCurMatNumArr;

#if PARAM
char	INPUT_FILE1[INPUT_LEN];
char	INPUT_FILE2[INPUT_LEN];
UINT4	WORD_SIZE;
int		ALLOW_SIZE;
UINT4	SPACE_SIZE;
UINT4	MIN_SEED_LEN;
int		SCORE_MAT;
int		SCORE_MIS;
int		SCORE_THR;
int		GREEDY_X;
int		GREEDY_MIN_L;
int		GREEDY_MAX_U;
int		WD_SIZE;
float	T_THR;
int	ALIGN_MODE;
#endif

extern "C" int* GetResFile(int SeedNum, int mode, bool print)
{
	FILE* g_pfRes2;
	FILE* g_pfRes3;
	PRE pstRE2;
	pstRE2	= (PRE) calloc (SeedNum, sizeof(RE));

	int* data = new int[SeedNum];

	g_pfRes2	= fopen(RES_FILE, "r");

	if (print)
	{
		g_pfRes3	= fopen(CSV_FILE, "w");
	}

#pragma omp critical (file_lock)
	{
		fread(pstRE2, sizeof(RE), SeedNum, g_pfRes2);
	}
	for (int i = 0; i<SeedNum; i++)
	{
		if (mode == 0)
		{
			data[i] = pstRE2[i].uX1;
		}
		if (mode == 1)
		{
			data[i] = pstRE2[i].uX2;
		}
		if (mode == 2)
		{
			data[i] = pstRE2[i].uY1;
		}
		if (mode == 3)
		{
			data[i] = pstRE2[i].uY2;
		}
		if (print)
		{
			fprintf(g_pfRes3, "%d,%d,%d,%d,%f,%d,%d,%d\n",pstRE2[i].uX1,pstRE2[i].uX2,pstRE2[i].uY1,pstRE2[i].uY2,pstRE2[i].fIdentity,pstRE2[i].uSeedX,pstRE2[i].uSeedY,pstRE2[i].uSeedLen);
		}
	}
	fclose(g_pfRes2);
	if (print)
	{
		fclose(g_pfRes3);
	}

	return data;
}

extern "C" int* GetSeedFile(int SeedNum, int mode, bool print)
{
	FILE* g_pfRes2;
	FILE* g_pfRes3;
	PSEED pstSEED2;
	pstSEED2	= (PSEED) calloc (SeedNum, sizeof(SEED));

	int* data = new int[SeedNum];

	g_pfRes2	= fopen(SEED_FILE, "r");

	if (print)
	{
		g_pfRes3	= fopen(CSV_FILE, "w");
	}

#pragma omp critical (file_lock)
	{
		fread(pstSEED2, sizeof(SEED), SeedNum, g_pfRes2);
	}
	for (int i = 0; i<SeedNum; i++)
	{
		if (mode == 0)
		{
			data[i] = pstSEED2[i].uX1;
		}
		if (mode == 1)
		{
			data[i] = pstSEED2[i].uX2;
		}
		if (mode == 2)
		{
			data[i] = pstSEED2[i].uY1;
		}
		if (mode == 3)
		{
			data[i] = pstSEED2[i].uY2;
		}
		if (print)
		{
			fprintf(g_pfRes3, "%d,%d,%d,%d\n",pstSEED2[i].uX1,pstSEED2[i].uX2,pstSEED2[i].uY1,pstSEED2[i].uY2);
		}
	}
	fclose(g_pfRes2);
	if (print)
	{
		fclose(g_pfRes3);
	}

	return data;
}

int main()
{
	char*	pbQry		= NULL;		// input file 1 data processing
	char*	pbRC		= NULL;		// reverse complement of input file 1 data
	char*	pbData		= NULL;		// input file 2 data processing

	UINT4	uQryLen	= 0;

	UINT4	uDataLen	= 0;

	int	errno;
	int	SeedNum;

	TimeLogPush("REMiner Ver.2", 0, 1);


#if PARAM
	/* get user parameter */
	if ( GetParam() )
	{
		system("pause");
		return -1;
	}
#endif


	/* make qry file & get qry length */
	TimeLogPush("Make Qry & Database File", 2, 1);
	uQryLen = DataFileWrite1();			// In input file 1, process CR and save to qry file (return: size of qry file)
	uDataLen = DataFileWrite2();			// In input file 2, process CR and save to data file (return: size of data file)
	printf("%d,%d\n",uQryLen,uDataLen);
	if (uQryLen == 0)
	{
		system("pause");
		return -2;
	}
	if (uDataLen == 0)
	{
		system("pause");
		return -2;
	}
	TimeLogPush("Make Qry & Database File", 2, 0);


	/* get file data */
	pbQry		= (char*) calloc (uQryLen, sizeof(char));
	pbRC		= (char*) calloc (uDataLen, sizeof(char));
	pbData		= (char*) calloc (uDataLen, sizeof(char));

	TimeLogPush("Get Data", 2, 1);
	if ( GetData(pbQry, uQryLen, 0) )	// Read qry file, and save to pbQry
	{
		system("pause");
		return -3;
	}
	if ( GetData(pbData, uDataLen, 1) )	// Read data file, and save to pbData
	{
		system("pause");
		return -3;
	}
	TimeLogPush("Get Data", 2, 0);


	/* result file open */
	if (ALIGN_MODE == 0)
	{
		g_pfRes = fopen(RES_FILE, "wb");
	}
	if (ALIGN_MODE == 1)
	{
		g_pfRes = fopen(SEED_FILE, "wb");
	}


	/* initialize greedy memory */
	InitGreedyMem();


	/* Stage 1. Preprocessing */
	TimeLogPush("Preprocessing", 1, 1);
	Preprocessing(pbQry, pbData, pbRC, uQryLen, uDataLen);		// Read & process pbQry into pbQry, pbRC (filtering file also added)
	TimeLogPush("Preprocessing", 1, 0);


	/* Stage 2 & 3. Seeding & Extension */
	TimeLogPush("Seeding & Extension", 1, 1);
	if (ALIGN_MODE == 0)
	{
		SeedNum = SeedingAndExt(pbQry, pbData, pbRC, uQryLen, uDataLen);		// From pbQry and pbRC, do seeding and extension for writing into file
	}
	if (ALIGN_MODE == 1)
	{
		SeedNum = Seeding(pbQry, pbData, pbRC, uQryLen, uDataLen);		// Only seeding processed, then seeds will be written into file
	}
	TimeLogPush("Seeding & Extension", 1, 0);


	/* cleanup */
	fclose(g_pfRes);
	CleanGreedyMem();

	TimeLogPush("REMiner Ver.2", 0, 0);
	
	TimeLogPop(uQryLen, uDataLen);

	/*
	PRE pstRE2;
	pstRE2 = GetResFile(SeedNum);
	printf("%d\n",pstRE2[0].uX1);
	printf("%d\n",pstRE2[1].uX1);
	*/


	//system("read -p 'Press Enter to continue...' var");
	return SeedNum;
}

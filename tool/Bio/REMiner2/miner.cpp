#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <fstream>
#include <iostream>
#include <omp.h>
#include "GVD.H"
#include "structure.h"
#include "miner.h"
#include "preprocess.h"
#include "seed.h"
#include "timelog.h"



#if PARAM
int GetParam()
{
	FILE*	pfParam				= NULL;
	char	abLine[BLOCK_LEN]	= {};
	char	abTemp[INPUT_LEN]	= {};
	char	abParam[INPUT_LEN]	= {};
	
	int	errno;


	/* file open */
	pfParam = fopen(PARAM_FILE, "rt");


	/* read param & save data */
	fgets(abLine, BLOCK_LEN, pfParam);		// QUERY_FILE (0)
	strcpy(abParam, strchr(abLine, '\t') + 1);					// Cut forward
	strncpy(abTemp, abParam, strlen(abParam) - strlen(strrchr(abParam, '\t')));	// Cut backward
	strcpy(INPUT_FILE1, abTemp);
	StrTrimLeft(INPUT_FILE1);
	StrTrimRight(INPUT_FILE1);

	memset(abLine, 0, BLOCK_LEN);
	memset(abTemp, 0, INPUT_LEN);
	fgets(abLine, BLOCK_LEN, pfParam);		// DATABASE_FILE (1)
	strcpy(abParam, strchr(abLine, '\t') + 1);					// Cut forward
	strncpy(abTemp, abParam, strlen(abParam) - strlen(strrchr(abParam, '\t')));	// Cut backward
	strcpy(INPUT_FILE2, abTemp);
	StrTrimLeft(INPUT_FILE2);
	StrTrimRight(INPUT_FILE2);

	memset(abLine, 0, BLOCK_LEN);
	fgets(abLine, BLOCK_LEN, pfParam);		// WORD_SIZE (2)
	strcpy(abParam, strchr(abLine, '\t') + 1);					// Cut forward
	strncpy(abParam, abParam, strrchr(abParam, '\t') - abParam);	// Cut backward
	WORD_SIZE	= (UINT4) atoi (abParam);

	memset(abLine, 0, BLOCK_LEN);
	fgets(abLine, BLOCK_LEN, pfParam);		// ALLOW_SIZE (3)
	strcpy(abParam, strchr(abLine, '\t') + 1);					// Cut forward
	strncpy(abParam, abParam, strrchr(abParam, '\t') - abParam);	// Cut backward
	ALLOW_SIZE	= atoi (abParam);

	memset(abLine, 0, BLOCK_LEN);
	fgets(abLine, BLOCK_LEN, pfParam);		// SPACE_SIZE (4)
	strcpy(abParam, strchr(abLine, '\t') + 1);					// Cut forward
	strncpy(abParam, abParam, strrchr(abParam, '\t') - abParam);	// Cut backward
	SPACE_SIZE	= (UINT4) atoi (abParam);

	memset(abLine, 0, BLOCK_LEN);
	fgets(abLine, BLOCK_LEN, pfParam);		// MIN_SEED_LEN (5)
	strcpy(abParam, strchr(abLine, '\t') + 1);					// Cut forward
	strncpy(abParam, abParam, strrchr(abParam, '\t') - abParam);	// Cut backward
	MIN_SEED_LEN	= (UINT4) atoi (abParam);

	memset(abLine, 0, BLOCK_LEN);
	fgets(abLine, BLOCK_LEN, pfParam);		// SCORE_MAT (6)
	strcpy(abParam, strchr(abLine, '\t') + 1);					// Cut forward
	strncpy(abParam, abParam, strrchr(abParam, '\t') - abParam);	// Cut backward
	SCORE_MAT	= atoi (abParam);

	memset(abLine, 0, BLOCK_LEN);
	fgets(abLine, BLOCK_LEN, pfParam);		// SCORE_MIS (7)
	strcpy(abParam, strchr(abLine, '\t') + 1);					// Cut forward
	strncpy(abParam, abParam, strrchr(abParam, '\t') - abParam);	// Cut backward
	SCORE_MIS	= atoi (abParam);

	memset(abLine, 0, BLOCK_LEN);
	fgets(abLine, BLOCK_LEN, pfParam);		// SCORE_THR (8)
	strcpy(abParam, strchr(abLine, '\t') + 1);					// Cut forward
	strncpy(abParam, abParam, strrchr(abParam, '\t') - abParam);	// Cut backward
	SCORE_THR	= atoi (abParam);

	memset(abLine, 0, BLOCK_LEN);
	fgets(abLine, BLOCK_LEN, pfParam);		// GREEDY_X (9)
	strcpy(abParam, strchr(abLine, '\t') + 1);					// Cut forward
	strncpy(abParam, abParam, strrchr(abParam, '\t') - abParam);	// Cut backward
	GREEDY_X	= atoi (abParam);

	memset(abLine, 0, BLOCK_LEN);
	fgets(abLine, BLOCK_LEN, pfParam);		// GREEDY_MIN_L (10)
	strcpy(abParam, strchr(abLine, '\t') + 1);					// Cut forward
	strncpy(abParam, abParam, strrchr(abParam, '\t') - abParam);	// Cut backward
	GREEDY_MIN_L	= atoi (abParam);

	memset(abLine, 0, BLOCK_LEN);
	fgets(abLine, BLOCK_LEN, pfParam);		// GREEDY_MAX_U (11)
	strcpy(abParam, strchr(abLine, '\t') + 1);					// Cut forward
	strncpy(abParam, abParam, strrchr(abParam, '\t') - abParam);	// Cut backward
	GREEDY_MAX_U	= atoi (abParam);

	memset(abLine, 0, BLOCK_LEN);
	fgets(abLine, BLOCK_LEN, pfParam);		// WD_SIZE (12)
	strcpy(abParam, strchr(abLine, '\t') + 1);					// Cut forward
	strncpy(abParam, abParam, strrchr(abParam, '\t') - abParam);	// Cut backward
	WD_SIZE	= atoi (abParam);

	memset(abLine, 0, BLOCK_LEN);
	fgets(abLine, BLOCK_LEN, pfParam);		// T_THR (13)
	strcpy(abParam, strchr(abLine, '\t') + 1);					// Cut forward
	strncpy(abParam, abParam, strrchr(abParam, '\t') - abParam);	// Cut backward
	T_THR	= (float) atof (abParam);

	memset(abLine, 0, BLOCK_LEN);
	fgets(abLine, BLOCK_LEN, pfParam);		// ALIGN_MODE (13)
	strcpy(abParam, strchr(abLine, '\t') + 1);					// Cut forward
	strncpy(abParam, abParam, strrchr(abParam, '\t') - abParam);	// Cut backward
	ALIGN_MODE	= atoi (abParam);


	
	fclose(pfParam);

	return 0;
}
#endif

UINT4 DataFileWrite1()
{
	FILE*	pfInput1				= NULL;
	FILE*	pfQry				= NULL;
	char	abInput[BLOCK_LEN]	= {};
	UINT4	uQryLen				= 0;
	
	int	errno;


	/* file open */
	pfInput1 = fopen(INPUT_FILE1, "r");

	pfQry = fopen(QRY_FILE, "w");

        fgets(abInput, BLOCK_LEN, pfInput1);

	/* read input & write data */
	while ( fgets(abInput, BLOCK_LEN, pfInput1) )
	{
		if ( strchr(abInput, '>') == NULL )				// No remark
		{
			if ( abInput[ strlen(abInput) - 1 ] == '\n' )	// Process CR (Carriage Return)
			{
				abInput[ strlen(abInput) - 1 ] = 0;
			}

			fwrite(abInput, sizeof(char), strlen(abInput), pfQry);

			uQryLen += (UINT4) strlen(abInput);
		}

		memset(abInput, 0, BLOCK_LEN);
	}


	fclose(pfInput1);
	fclose(pfQry);


	return uQryLen;
}

UINT4 DataFileWrite2()
{
	FILE*	pfInput2				= NULL;
	FILE*	pfData				= NULL;
	char	abInput[BLOCK_LEN]	= {};
	UINT4	uDataLen				= 0;
	
	int	errno;


	/* file open */
	pfInput2 = fopen(INPUT_FILE2, "r");

	pfData = fopen(DATA_FILE, "w");

        fgets(abInput, BLOCK_LEN, pfInput2);

	/* read input & write data */
	while ( fgets(abInput, BLOCK_LEN, pfInput2) )
	{
		if ( strchr(abInput, '>') == NULL )				// No remark
		{
			if ( abInput[ strlen(abInput) - 1 ] == '\n' )	// Process CR (Carriage Return)
			{
				abInput[ strlen(abInput) - 1 ] = 0;
			}

			fwrite(abInput, sizeof(char), strlen(abInput), pfData);

			uDataLen += (UINT4) strlen(abInput);
		}

		memset(abInput, 0, BLOCK_LEN);
	}


	fclose(pfInput2);
	fclose(pfData);


	return uDataLen;
}

int GetData(char* pb, UINT4 uLen, UINT4 mode)
{
	FILE*	pf	= NULL;
	
	int	errno;


	/* file open */
	if (mode == 0)
	{
		pf = fopen(QRY_FILE, "r");
	}
	else
	{
		pf = fopen(DATA_FILE, "r");
	}

	/* get data */
	fread(pb, sizeof(char), uLen, pf);


	fclose(pf);

	return 0;
}

int Preprocessing(char* pbQry, char* pbData, char* pbRC, UINT4 uQryLen, UINT4 uDataLen)
{
	FILE*	pfQry		= NULL;
	FILE*	pfQryM		= NULL;
	char*	pbQryM		= NULL;

	FILE*	pfData		= NULL;
	FILE*	pfDataM		= NULL;
	char*	pbDataM		= NULL;
	FILE*	pfRC		= NULL;

	int	errno;


	/* 1. Case Matching & save file */
	TimeLogPush("Case Matching", 2, 1);
	CaseMatch(pbQry, uQryLen);					// Replace 'a' to 'A' in pbQry
	CaseMatch(pbData, uDataLen);					// Replace 'a' to 'A' in pbData
	TimeLogPush("Case Matching", 2, 0);

	pfQry = fopen(QRY_FILE, "w");
	fwrite(pbQry, sizeof(char), uQryLen, pfQry);
	fclose(pfQry);

	pfData = fopen(DATA_FILE, "w");
	fwrite(pbData, sizeof(char), uDataLen, pfData);
	fclose(pfData);


	/* 2. Low Complexity Filtering & save file */
	pbQryM	= (char*) calloc (uQryLen, sizeof(char));
	pbDataM	= (char*) calloc (uDataLen, sizeof(char));

	TimeLogPush("Low Complexity Filtering", 2, 1);
	Filtering(pbQryM, pbQry, uQryLen);		// Insert low complexity filtering of pbQry to pbQryM
	Filtering(pbDataM, pbData, uDataLen);		// Insert low complexity filtering of pbData to pbDataM
	TimeLogPush("Low Complexity Filtering", 2, 0);

	pfQryM = fopen(QRY_M_FILE, "w");
	fwrite(pbQryM, sizeof(char), uQryLen, pfQryM);
	fclose(pfQryM);

	free(pbQryM);

	pfDataM = fopen(DATA_M_FILE, "w");
	fwrite(pbDataM, sizeof(char), uDataLen, pfDataM);
	fclose(pfDataM);

	free(pbDataM);



	/* 3. Reverse Complement Conversion & save file */
	TimeLogPush("Reverse Complement", 2, 1);
	RC_Conversion(pbRC, pbData, uDataLen);			// Insert reverse complement of pbQry to pbRC
	TimeLogPush("Reverse Complement", 2, 0);

	pfRC = fopen(RC_FILE, "w");
	fwrite(pbRC, sizeof(char), uDataLen, pfRC);
	fclose(pfRC);

	return 0;
}

int SeedingAndExt(char* pbQry, char* pbData, char* pbRC, UINT4 uQryLen, UINT4 uDataLen)
{
	UINT4		uTmpWordNum		= 0;

	PWORD_ELE	pstWordEle1		= NULL;
	PTMP_WORD	pstTmpWord1		= NULL;

	FILE*		pfQryM			= NULL;
	char*		pbQryM			= NULL;

	PWORD_ELE	pstWordEle2		= NULL;
	PTMP_WORD	pstTmpWord2		= NULL;

	FILE*		pfDataM			= NULL;
	char*		pbDataM			= NULL;

	int		errno;
	int		SeedNum			= 0;


	uTmpWordNum	= (UINT4) pow ( (float)4, (float)WORD_SIZE );


	/* Initialize word ele & tmp word */
	pstWordEle1	= (PWORD_ELE) malloc ( (size_t) uQryLen * sizeof(WORD_ELE) );
	memset(pstWordEle1, INFINITY4U, (size_t) uQryLen * sizeof(WORD_ELE));

	pstTmpWord1	= (PTMP_WORD) malloc ( (size_t) uTmpWordNum * sizeof(TMP_WORD) );
	memset(pstTmpWord1, INFINITY4U, (size_t) uTmpWordNum * sizeof(TMP_WORD));

	pstWordEle2	= (PWORD_ELE) malloc ( (size_t) uDataLen * sizeof(WORD_ELE) );
	memset(pstWordEle2, INFINITY4U, (size_t) uDataLen * sizeof(WORD_ELE));

	pstTmpWord2	= (PTMP_WORD) malloc ( (size_t) uTmpWordNum * sizeof(TMP_WORD) );
	memset(pstTmpWord2, INFINITY4U, (size_t) uTmpWordNum * sizeof(TMP_WORD));


	/* 1. Word Matching */
	pbQryM	= (char*) calloc (uQryLen, sizeof(char));
	pbDataM	= (char*) calloc (uDataLen, sizeof(char));

	pfQryM	= fopen(QRY_M_FILE, "r");
	fread(pbQryM, sizeof(char), uQryLen, pfQryM);
	fclose(pfQryM);

	pfDataM	= fopen(DATA_M_FILE, "r");
	fread(pbDataM, sizeof(char), uDataLen, pfDataM);
	fclose(pfDataM);

	TimeLogPush("Word Matching", 2, 1);
	WordMatch(pbQryM, uQryLen, pstWordEle1, pstTmpWord1);		// From pbQryM, word matching results are inserted in pstIndex, pstTmpIndex
	WordMatch(pbDataM, uDataLen, pstWordEle2, pstTmpWord2);		// From pbQryM, word matching results are inserted in pstIndex, pstTmpIndex
	TimeLogPush("Word Matching", 2, 0);

	free(pbQryM);
	free(pbDataM);


	/* 2-1. Seeding & Extension (Query) */
	TimeLogPush("Seeding & Extension (Query)", 2, 1);
	if ( ALLOW_SIZE )
	{
		SeedNum += NHitMethodQryAllow(pstWordEle1, pstTmpWord1, pstWordEle2, pstTmpWord2, pbQry, pbData, uQryLen, uDataLen);		// From pstIndex and pstTmpIndex, Seeding & Extension results are recored in file
	}
	else
	{
		SeedNum += NHitMethodQry(pstWordEle1, pstTmpWord1, pstWordEle2, pstTmpWord2, pbQry, pbData, uQryLen, uDataLen);						// From pstIndex, Seeding & Extension results are recored in file
	}
	TimeLogPush("Seeding & Extension (Query)", 2, 0);


	/* 2-2. Seeding & Extension (RC) */
	TimeLogPush("Seeding & Extension (RC)", 2, 1);
	if ( ALLOW_SIZE )
	{
		SeedNum += NHitMethodRCAllow(pstWordEle1, pstTmpWord1, pstWordEle2, pstTmpWord2, pbQry, pbRC, uQryLen, uDataLen);	// From pstIndex and pstTmpIndex, Seeding & Extension results are recored in file
	}
	else
	{
		SeedNum += NHitMethodRC(pstWordEle1, pstTmpWord1, pstWordEle2, pstTmpWord2, pbQry, pbRC, uQryLen, uDataLen);			// From pstIndex and pstTmpIndex, Seeding & Extension results are recored in file
	}
	TimeLogPush("Seeding & Extension (RC)", 2, 0);


	/* free word ele & tmp word */
	free ( pstWordEle1 );
	free ( pstTmpWord1 );
	free ( pstWordEle2 );
	free ( pstTmpWord2 );
	pstWordEle1	= NULL;
	pstTmpWord1	= NULL;
	pstWordEle2	= NULL;
	pstTmpWord2	= NULL;


	return SeedNum;
}

int Seeding(char* pbQry, char* pbData, char* pbRC, UINT4 uQryLen, UINT4 uDataLen)
{
	UINT4		uTmpWordNum		= 0;

	PWORD_ELE	pstWordEle1		= NULL;
	PTMP_WORD	pstTmpWord1		= NULL;

	FILE*		pfQryM			= NULL;
	char*		pbQryM			= NULL;

	PWORD_ELE	pstWordEle2		= NULL;
	PTMP_WORD	pstTmpWord2		= NULL;

	FILE*		pfDataM			= NULL;
	char*		pbDataM			= NULL;

	int		errno;
	int		SeedNum			= 0;


	uTmpWordNum	= (UINT4) pow ( (float)4, (float)WORD_SIZE );


	/* Initialize word ele & tmp word */
	pstWordEle1	= (PWORD_ELE) malloc ( (size_t) uQryLen * sizeof(WORD_ELE) );
	memset(pstWordEle1, INFINITY4U, (size_t) uQryLen * sizeof(WORD_ELE));

	pstTmpWord1	= (PTMP_WORD) malloc ( (size_t) uTmpWordNum * sizeof(TMP_WORD) );
	memset(pstTmpWord1, INFINITY4U, (size_t) uTmpWordNum * sizeof(TMP_WORD));

	pstWordEle2	= (PWORD_ELE) malloc ( (size_t) uDataLen * sizeof(WORD_ELE) );
	memset(pstWordEle2, INFINITY4U, (size_t) uDataLen * sizeof(WORD_ELE));

	pstTmpWord2	= (PTMP_WORD) malloc ( (size_t) uTmpWordNum * sizeof(TMP_WORD) );
	memset(pstTmpWord2, INFINITY4U, (size_t) uTmpWordNum * sizeof(TMP_WORD));


	/* 1. Word Matching */
	pbQryM	= (char*) calloc (uQryLen, sizeof(char));
	pbDataM	= (char*) calloc (uDataLen, sizeof(char));

	pfQryM	= fopen(QRY_M_FILE, "r");
	fread(pbQryM, sizeof(char), uQryLen, pfQryM);
	fclose(pfQryM);

	pfDataM	= fopen(DATA_M_FILE, "r");
	fread(pbDataM, sizeof(char), uDataLen, pfDataM);
	fclose(pfDataM);

	TimeLogPush("Word Matching", 2, 1);
	WordMatch(pbQryM, uQryLen, pstWordEle1, pstTmpWord1);		// From pbQryM, word matching results are inserted in pstIndex, pstTmpIndex
	WordMatch(pbDataM, uDataLen, pstWordEle2, pstTmpWord2);		// From pbQryM, word matching results are inserted in pstIndex, pstTmpIndex
	TimeLogPush("Word Matching", 2, 0);

	free(pbQryM);
	free(pbDataM);


	/* 2-1. Seeding (Query) */
	TimeLogPush("Seeding & Extension (Query)", 2, 1);
	if ( ALLOW_SIZE )
	{
		SeedNum += NHitMethodQryAllow2(pstWordEle1, pstTmpWord1, pstWordEle2, pstTmpWord2, pbQry, pbData, uQryLen, uDataLen);		// From pstIndex and pstTmpIndex, Seeding & Extension results are recored in file
	}
	else
	{
		SeedNum += NHitMethodQry2(pstWordEle1, pstTmpWord1, pstWordEle2, pstTmpWord2, pbQry, pbData, uQryLen, uDataLen);						// From pstIndex, Seeding & Extension results are recored in file
	}
	TimeLogPush("Seeding & Extension (Query)", 2, 0);


	/* 2-2. Seeding (RC) */
	TimeLogPush("Seeding & Extension (RC)", 2, 1);
	if ( ALLOW_SIZE )
	{
		SeedNum += NHitMethodRCAllow2(pstWordEle1, pstTmpWord1, pstWordEle2, pstTmpWord2, pbQry, pbRC, uQryLen, uDataLen);	// From pstIndex and pstTmpIndex, Seeding & Extension results are recored in file
	}
	else
	{
		SeedNum += NHitMethodRC2(pstWordEle1, pstTmpWord1, pstWordEle2, pstTmpWord2, pbQry, pbRC, uQryLen, uDataLen);			// From pstIndex and pstTmpIndex, Seeding & Extension results are recored in file
	}
	TimeLogPush("Seeding & Extension (RC)", 2, 0);


	/* free word ele & tmp word */
	free ( pstWordEle1 );
	free ( pstTmpWord1 );
	free ( pstWordEle2 );
	free ( pstTmpWord2 );
	pstWordEle1	= NULL;
	pstTmpWord1	= NULL;
	pstWordEle2	= NULL;
	pstTmpWord2	= NULL;


	return SeedNum;
}

int StrTrimLeft(char* pbSeq)
{
	UINT4	i		= 0;
	UINT4	uSeqLen	= 0;

	uSeqLen	= (UINT4) strlen(pbSeq);

	while ( isspace(pbSeq[i]) )
	{
		++i;
	}

	if ( i )
	{
		strcpy(pbSeq, pbSeq + i);
		memset(pbSeq + uSeqLen - i, 0, i);
	}

	return 0;
}

int StrTrimRight(char* pbSeq)
{
	UINT4	i		= 0;
	UINT4	uSeqLen	= 0;

	uSeqLen	= (UINT4) strlen(pbSeq);

	while ( isspace(pbSeq[uSeqLen - 1 - i]) )
	{
		++i;
	}

	if ( i )
	{
		memset(pbSeq + uSeqLen - i, 0, i);
	}

	return 0;
}

int InitGreedyMem()
{
	UINT4	i	= 0;

	UINT4	uProcNum	= 0;
	UINT4	uSizeR		= GREEDY_MAX_U - GREEDY_MIN_L + 5;


//	omp_set_num_threads(1);

#pragma omp parallel
	{
#pragma omp master
		{
			uProcNum	= omp_get_num_threads();
		}
	}

	g_ppllGreedyCurR	= (long long**) calloc (uProcNum, sizeof(long long*));
	g_ppllGreedyPreR	= (long long**) calloc (uProcNum, sizeof(long long*));
	g_ppuCurMatNumArr	= (UINT4**) calloc (uProcNum, sizeof(UINT4*));
	g_ppuPreMatNumArr	= (UINT4**) calloc (uProcNum, sizeof(UINT4*));

	for ( i = 0 ; i < uProcNum ; ++i )
	{
		g_ppllGreedyCurR[i]		= (long long*) calloc (uSizeR, sizeof(long long));
		g_ppllGreedyPreR[i]		= (long long*) calloc (uSizeR, sizeof(long long));
		g_ppuCurMatNumArr[i]	= (UINT4*) calloc (uSizeR, sizeof(UINT4));
		g_ppuPreMatNumArr[i]	= (UINT4*) calloc (uSizeR, sizeof(UINT4));
	}


	return 0;
}

int CleanGreedyMem()
{
	UINT4	i	= 0;

	UINT4	uProcNum	= 0;


//	omp_set_num_threads(1);

#pragma omp parallel
	{
#pragma omp master
		{
			uProcNum	= omp_get_num_threads();
		}
	}

	for ( i = 0 ; i < uProcNum ; ++i )
	{
		free (g_ppllGreedyCurR[i]);
		free (g_ppllGreedyPreR[i]);
		free (g_ppuCurMatNumArr[i]);
		free (g_ppuPreMatNumArr[i]);
	}

	free (g_ppllGreedyCurR);
	free (g_ppllGreedyPreR);
	free (g_ppuCurMatNumArr);
	free (g_ppuPreMatNumArr);


	return 0;
}
